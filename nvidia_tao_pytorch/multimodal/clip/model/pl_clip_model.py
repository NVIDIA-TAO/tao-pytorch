# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP Model PyTorch Lightning Module."""

import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

_MAX_LOGIT_SCALE = math.log(100)

from open_clip.loss import ClipLoss, SigLipLoss  # noqa: E402

from nvidia_tao_pytorch.core.tlt_logging import logging  # noqa: E402
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import (  # noqa: E402
    TAOLightningModule,
)
from nvidia_tao_pytorch.core.loggers import (  # noqa: E402
    api_logging as status_logging,
)
from nvidia_tao_pytorch.multimodal.clip.model.clip import build_model  # noqa: E402
from nvidia_tao_pytorch.multimodal.clip.loss.masked_siglip_loss import (  # noqa: E402
    MetadataMaskedSigLipLoss,
)
from nvidia_tao_pytorch.multimodal.clip.utils.utils import (  # noqa: E402
    build_optimizer,
    compute_lr,
)
from nvidia_tao_pytorch.multimodal.clip.model.evaluation.retrieval import (  # noqa: E402
    RetrievalEvaluator,
    log_retrieval_metrics,
)
from nvidia_tao_pytorch.multimodal.clip.model.evaluation.pas import (  # noqa: E402
    PAS_METADATA_QUERY_TYPES,
    build_pas_embedding_maps_from_rows,
    evaluate_pas_metadata_embeddings,
    load_pas_pairs,
)


def _batch_hard_image_text_triplet_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Compute symmetric batch-hard triplet loss for matched image-text pairs."""
    batch_size = image_features.shape[0]
    if batch_size < 2:
        return image_features.new_zeros(())

    image_norm = F.normalize(image_features, dim=-1)
    text_norm = F.normalize(text_features, dim=-1)
    similarity = image_norm @ text_norm.mT
    diagonal_mask = torch.eye(
        batch_size, device=similarity.device, dtype=torch.bool
    )

    def _row_loss(scores):
        negatives = scores.masked_fill(diagonal_mask, float("-inf"))
        hardest_negative = negatives.max(dim=1).values
        positives = scores.diag()
        return F.relu(margin - positives + hardest_negative).mean()

    return 0.5 * (_row_loss(similarity) + _row_loss(similarity.mT))


def _get_attribute_metadata_from_batch(
    batch,
    context="siglip_loss_mask_mode='attribute_match_ignore'",
    require_accessories=False,
):
    """Get attribute metadata from a training batch."""
    if batch is None or len(batch) < 3 or not isinstance(batch[2], dict):
        raise ValueError(
            f"{context} requires batch metadata with image_attr_values and "
            "text_attr_values."
        )
    metadata = batch[2]
    required_keys = ["image_attr_values", "text_attr_values"]
    if require_accessories:
        required_keys.extend(["image_accessory_ids", "text_accessory_ids"])
    missing_keys = [key for key in required_keys if key not in metadata]
    if missing_keys:
        raise ValueError(
            f"{context} requires batch metadata keys {required_keys}; "
            f"missing {missing_keys}."
        )
    return metadata


def _config_value(config, key, default=None):
    """Read a value from a mapping or dataclass-like config."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _pas_pairs_file(dataset_config) -> Path:
    """Resolve the split-aligned PAS pairs file for training validation."""
    explicit = _config_value(dataset_config, "attribute_pairs_file")
    if explicit:
        path = Path(explicit)
    else:
        image_list_file = _config_value(dataset_config, "image_list_file")
        if not image_list_file:
            raise ValueError(
                "Metadata-aware validation requires attribute_pairs_file or "
                "image_list_file."
            )
        image_list_path = Path(image_list_file)
        suffix = "_list.txt"
        if not image_list_path.name.endswith(suffix):
            raise ValueError(
                "Metadata-aware validation requires attribute_pairs_file or "
                f"an image_list_file ending with {suffix!r}."
            )
        path = image_list_path.with_name(
            image_list_path.name[:-len(suffix)] + "_pairs.json"
        )
    if not path.is_file():
        raise ValueError(
            "Metadata-aware validation requires a valid PAS pairs file, "
            f"got {path}."
        )
    return path


def _distributed_ready() -> bool:
    """Return whether torch distributed collectives are available."""
    return dist.is_available() and dist.is_initialized()


def _collective_device(device: torch.device) -> torch.device:
    """Use CUDA tensors for NCCL and CPU tensors for other backends."""
    if _distributed_ready() and str(dist.get_backend()).lower() == "nccl":
        return device
    return torch.device("cpu")


def _all_gather_variable_rows(
    values: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    """Gather variable first-dimension tensors and return them on rank zero."""
    values = values.detach()
    if not _distributed_ready():
        return values.cpu()

    collective_device = _collective_device(device)
    local_size = torch.tensor(
        [len(values)],
        dtype=torch.long,
        device=collective_device,
    )
    sizes = [
        torch.zeros_like(local_size)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(sizes, local_size)
    sizes = [int(size.item()) for size in sizes]
    max_size = max(sizes)

    padded_shape = (max_size, *values.shape[1:])
    padded = torch.zeros(
        padded_shape,
        dtype=values.dtype,
        device=collective_device,
    )
    if len(values):
        padded[:len(values)] = values.to(collective_device)
    gathered = [torch.empty_like(padded) for _ in sizes]
    dist.all_gather(gathered, padded)

    if dist.get_rank() != 0:
        return None
    return torch.cat(
        [
            shard[:size].cpu()
            for shard, size in zip(gathered, sizes)
        ],
        dim=0,
    )


def _broadcast_pas_metrics(
    weighted_rows,
    device: torch.device,
) -> dict:
    """Broadcast query-weighted PAS mAP/query counts from rank zero."""
    collective_device = _collective_device(device)
    payload = torch.full(
        (len(PAS_METADATA_QUERY_TYPES), 2),
        float("nan"),
        dtype=torch.float64,
        device=collective_device,
    )
    if not _distributed_ready() or dist.get_rank() == 0:
        by_query_type = {
            row["QueryType"]: row for row in (weighted_rows or [])
        }
        for index, query_type in enumerate(PAS_METADATA_QUERY_TYPES):
            row = by_query_type.get(query_type)
            if row is not None:
                payload[index, 0] = float(row["mAP"])
                payload[index, 1] = float(row["num_queries"])
    if _distributed_ready():
        dist.broadcast(payload, src=0)
    payload = payload.cpu()
    return {
        query_type: {
            "mAP": float(payload[index, 0]),
            "num_queries": int(payload[index, 1]),
        }
        for index, query_type in enumerate(PAS_METADATA_QUERY_TYPES)
        if not torch.isnan(payload[index, 0])
    }


class CLIPPlModel(TAOLightningModule):
    """PTL module for CLIP Model with retrieval-based validation."""

    def __init__(self, experiment_spec, export=False):
        """Initialize CLIP model for training."""
        super().__init__(experiment_spec)
        self.experiment_spec = experiment_spec
        self.checkpoint_filename = 'clip'

        clip_model = build_model(
            experiment_config=self.experiment_spec, export=export
        )
        self.model = clip_model.model
        self.tokenizer = clip_model.tokenizer
        self.preprocess_train, self.preprocess_val = (
            clip_model.preprocess_train,
            clip_model.preprocess_val,
        )

        if getattr(self.experiment_spec.train, "grad_checkpointing", False):
            self.model.set_grad_checkpointing()
            logging.info("Gradient checkpointing enabled")

        self.loss_type = self.experiment_spec.train.loss_type
        self.siglip_loss_dist_impl = getattr(
            self.experiment_spec.train, "siglip_loss_dist_impl", "gather"
        )
        self.siglip_loss_mask_mode = getattr(
            self.experiment_spec.train, "siglip_loss_mask_mode", "none"
        )
        self.triplet_loss_weight = getattr(
            self.experiment_spec.train, "triplet_loss_weight", 0.0
        )
        self.triplet_margin = getattr(
            self.experiment_spec.train, "triplet_margin", 0.2
        )

        # Check if retrieval validation is configured
        val_cfg = getattr(self.experiment_spec.dataset, 'val', None)
        self.retrieval_enabled = (
            val_cfg is not None and
            getattr(val_cfg, 'datasets', None) and
            len(val_cfg.datasets) > 0
        )
        self.metadata_match_eval = bool(
            val_cfg is not None and
            getattr(val_cfg, 'metadata_match_eval', False)
        )
        self.metadata_match_mode = (
            getattr(val_cfg, 'metadata_match_mode', "scalar_attributes")
            if val_cfg is not None
            else "scalar_attributes"
        )
        self.pas_validation_datasets = (
            list(val_cfg.datasets)
            if self.metadata_match_eval
            else []
        )
        self._pas_validation_pairs = None

    def setup(self, stage=None):
        """Set up training after Trainer is initialized."""
        if stage == 'fit':
            self.max_steps = self.trainer.estimated_stepping_batches
            self._build_criterion()

    def _build_criterion(self):
        """Build the loss function."""
        if self.loss_type == 'siglip':
            siglip_mask_mode = getattr(
                self, "siglip_loss_mask_mode", "none"
            )
            if siglip_mask_mode in (
                "attribute_match_ignore",
                "attribute_plus_accessory_match_ignore",
            ):
                train_data_cfg = self.experiment_spec.dataset.train
                if not getattr(
                    train_data_cfg, "include_attribute_metadata", False
                ):
                    raise ValueError(
                        f"siglip_loss_mask_mode={siglip_mask_mode!r} requires "
                        "dataset.train.include_attribute_metadata=True."
                    )
                train_data_type = getattr(train_data_cfg, "type", None)
                if train_data_type != "custom":
                    raise ValueError(
                        f"siglip_loss_mask_mode={siglip_mask_mode!r} requires "
                        "dataset.train.type='custom', got "
                        f"{train_data_type!r}."
                    )
                if self.siglip_loss_dist_impl not in ("local", "gather"):
                    raise NotImplementedError(
                        "Metadata-masked SigLIP loss currently supports "
                        "siglip_loss_dist_impl='local' or 'gather'; got "
                        f"{self.siglip_loss_dist_impl!r}."
                    )
                siglip_loss_world_size = self.trainer.world_size
                siglip_loss_rank = self.global_rank
                if self.siglip_loss_dist_impl == "local":
                    siglip_loss_world_size = 1
                    siglip_loss_rank = 0
                if self.global_rank == 0:
                    logging.info(
                        "Using metadata-masked SigLIP loss with mode=%s, "
                        "dist_impl=%s, trainer_world_size=%s, "
                        "loss_world_size=%s",
                        siglip_mask_mode,
                        self.siglip_loss_dist_impl,
                        self.trainer.world_size,
                        siglip_loss_world_size,
                    )
                self.loss = MetadataMaskedSigLipLoss(
                    dist_impl=self.siglip_loss_dist_impl,
                    world_size=siglip_loss_world_size,
                    rank=siglip_loss_rank,
                    accessory_aware=(
                        siglip_mask_mode
                        == "attribute_plus_accessory_match_ignore"
                    ),
                )
                self.criterion = self.loss
                return
            if siglip_mask_mode != "none":
                raise ValueError(
                    "Unsupported siglip_loss_mask_mode "
                    f"{siglip_mask_mode!r}."
                )

            siglip_loss_world_size = self.trainer.world_size
            siglip_loss_rank = self.global_rank
            siglip_dist_impl = self.siglip_loss_dist_impl
            open_clip_dist_impl = siglip_dist_impl
            if siglip_dist_impl == "local":
                siglip_loss_world_size = 1
                siglip_loss_rank = 0
                open_clip_dist_impl = "gather"

            if self.global_rank == 0:
                logging.info(
                    "Using SigLIP loss with dist_impl=%s, trainer_world_size=%s, "
                    "loss_world_size=%s",
                    siglip_dist_impl,
                    self.trainer.world_size,
                    siglip_loss_world_size,
                )
            self.loss = SigLipLoss(
                rank=siglip_loss_rank,
                world_size=siglip_loss_world_size,
                dist_impl=open_clip_dist_impl,
            )
            self.loss.dist_impl = siglip_dist_impl
        elif self.loss_type == 'clip':
            self.loss = ClipLoss(
                rank=self.global_rank,
                world_size=self.trainer.world_size,
            )
        else:
            raise NotImplementedError(
                f"loss function {self.loss_type} is not implemented"
            )
        self.criterion = self.loss

    def configure_optimizers(self):
        """Configure optimizer with per-tower parameter groups."""
        self.optimizer = build_optimizer(
            self.model, self.experiment_spec.train
        )
        self._build_tower_schedule_config()
        return self.optimizer

    def _build_tower_schedule_config(self):
        """Pre-compute per-tower schedule configs for training_step."""
        cfg = self.experiment_spec.train.optim

        self._tower_schedules = {
            'vision': {
                'lr': cfg.vision_lr,
                'warmup': cfg.warmup_steps,
                'scheduler': cfg.scheduler,
            },
            'text': {
                'lr': cfg.text_lr,
                'warmup': cfg.warmup_steps,
                'scheduler': cfg.scheduler,
            },
            'logit': {
                'lr': cfg.text_lr,
                'warmup': cfg.warmup_steps,
                'scheduler': cfg.scheduler,
            },
        }

    def on_train_start(self):
        """Training epoch start."""
        self.trainer.datamodule.resume_step = self.trainer.global_step

    def _forward_pass(self, batch):
        """Run forward pass."""
        image, text = batch[0], batch[1]
        return self.model(image=image, text=text)

    def _backward(self, outputs, batch=None):
        """Compute loss from model outputs."""
        if len(outputs) == 3:
            image_features, text_features, logit_scale = outputs
            logit_bias = None
        else:
            image_features, text_features, logit_scale, logit_bias = outputs

        if isinstance(self.loss, MetadataMaskedSigLipLoss):
            metadata = _get_attribute_metadata_from_batch(
                batch,
                context=(
                    "siglip_loss_mask_mode="
                    f"{getattr(self, 'siglip_loss_mask_mode', 'attribute_match_ignore')!r}"
                ),
                require_accessories=self.loss.accessory_aware,
            )
            clip_loss = self.loss(
                image_features,
                text_features,
                logit_scale,
                logit_bias,
                image_attr_values=metadata["image_attr_values"],
                text_attr_values=metadata["text_attr_values"],
                image_accessory_ids=metadata.get("image_accessory_ids"),
                text_accessory_ids=metadata.get("text_accessory_ids"),
            )
        elif len(outputs) == 3:
            clip_loss = self.loss(image_features, text_features, logit_scale)
        else:
            clip_loss = self.loss(
                image_features, text_features, logit_scale, logit_bias
            )
        return clip_loss, logit_scale, image_features, text_features

    def training_step(self, batch):
        """Training step."""
        image = batch[0]
        batch_size = (
            image['pixel_values'].shape[0]
            if isinstance(image, dict)
            else image.shape[0]
        )
        outputs = self._forward_pass(batch)
        loss, logit_scale, image_features, text_features = self._backward(
            outputs, batch
        )

        # Update per-tower learning rates
        for param_group in self.optimizer.param_groups:
            tower = param_group.get('_tower', 'text')
            sched = self._tower_schedules.get(
                tower, self._tower_schedules['text']
            )
            param_group['lr'] = compute_lr(
                self.global_step,
                sched['lr'],
                sched['warmup'],
                self.max_steps,
                sched['scheduler'],
            )

        vision_lr = self._tower_schedules['vision']['lr']
        text_lr = self._tower_schedules['text']['lr']
        current_vision_lr = compute_lr(
            self.global_step,
            vision_lr,
            self._tower_schedules['vision']['warmup'],
            self.max_steps,
            self._tower_schedules['vision']['scheduler'],
        )
        current_text_lr = compute_lr(
            self.global_step,
            text_lr,
            self._tower_schedules['text']['warmup'],
            self.max_steps,
            self._tower_schedules['text']['scheduler'],
        )
        self.log(
            "train/vision_lr", current_vision_lr,
            on_step=True, on_epoch=False, prog_bar=False
        )
        self.log(
            "train/text_lr", current_text_lr,
            on_step=True, on_epoch=False, prog_bar=False
        )
        self.log(
            "train/lr", current_text_lr,
            on_step=True, on_epoch=False, prog_bar=True
        )
        contrastive_loss = (
            loss['contrastive_loss'] if isinstance(loss, dict) else loss
        )
        if self.triplet_loss_weight > 0:
            triplet_loss = _batch_hard_image_text_triplet_loss(
                image_features, text_features, self.triplet_margin
            )
            loss_value = (
                contrastive_loss + self.triplet_loss_weight * triplet_loss
            )
            self.log(
                "train/triplet_loss", triplet_loss,
                on_step=True, on_epoch=True, prog_bar=False,
                sync_dist=True, batch_size=batch_size,
            )
            self.log(
                "train/contrastive_loss", contrastive_loss,
                on_step=True, on_epoch=True, prog_bar=False,
                sync_dist=True, batch_size=batch_size,
            )
        else:
            loss_value = contrastive_loss
        self.log(
            "train_loss", loss_value,
            on_step=True, on_epoch=True, prog_bar=True,
            sync_dist=True, batch_size=batch_size,
        )
        self.log(
            "train/logit_scale", logit_scale.item(),
            on_step=True, on_epoch=False, prog_bar=False
        )

        with torch.no_grad():
            self.model.logit_scale.clamp_(0, _MAX_LOGIT_SCALE)
        return loss_value

    def on_train_epoch_end(self):
        """Log training metrics to status.json."""
        average_train_loss = (
            self.trainer.logged_metrics["train_loss_epoch"].item()
        )

        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_validation_epoch_start(self) -> None:
        """Set up retrieval evaluator for validation."""
        if self.retrieval_enabled:
            self.retrieval_evaluator = RetrievalEvaluator(
                k_values=(1, 5, 10),
                device=self.device
            )
            self.image_embeddings = []
            self.text_embeddings = []
            self.pas_row_indices = []
            logging.info("Retrieval evaluator initialized for validation.")
            if self.metadata_match_eval:
                logging.info(
                    "PAS metadata validation enabled with mode=%s; "
                    "rank-local embeddings will be deduplicated and evaluated "
                    "on rank zero.",
                    self.metadata_match_mode,
                )
        else:
            self.retrieval_evaluator = None
            self.image_embeddings = []
            self.text_embeddings = []
            self.pas_row_indices = []
            logging.warning(
                "No validation configured. Add datasets to val.datasets "
                "to enable retrieval evaluation."
            )

    def validation_step(self, batch, batch_idx):
        """Run validation: collect image/text embeddings for retrieval."""
        if self.retrieval_evaluator is None:
            return

        image = batch[0]
        text = batch[1]

        # Get image features
        output = self.model(image=image)
        image_features = (
            output["image_features"]
            if isinstance(output, dict)
            else output[0]
        )

        # Handle different text formats from dataloader
        if isinstance(text, list) and len(text) > 0:
            if isinstance(text[0], dict):
                text = {
                    k: torch.stack([t[k] for t in text])
                    for k in text[0].keys()
                }
            elif isinstance(text[0], torch.Tensor):
                text = torch.stack(text)
            elif isinstance(text[0], str):
                text = self.tokenizer(text)
                if isinstance(text, list):
                    text = text[0]

        # Get text features
        text_output = self.model(text=text)
        text_features = (
            text_output["text_features"]
            if isinstance(text_output, dict)
            else text_output[1]
        )

        self.image_embeddings.append(image_features.cpu())
        self.text_embeddings.append(text_features.cpu())
        if self.metadata_match_eval:
            accessory_aware = (
                self.metadata_match_mode == "scalar_plus_accessories"
            )
            metadata = _get_attribute_metadata_from_batch(
                batch,
                context="dataset.val.metadata_match_eval=True",
                require_accessories=accessory_aware,
            )
            if "pas_row_index" not in metadata:
                raise ValueError(
                    "dataset.val.metadata_match_eval=True requires validation "
                    "metadata key 'pas_row_index'."
                )
            self.pas_row_indices.append(metadata["pas_row_index"].cpu())

    def _evaluate_accumulated_retrieval(self, image_emb, text_emb):
        """Evaluate accumulated embeddings with paired ground truth."""
        return self.retrieval_evaluator.evaluate_bidirectional(
            image_emb, text_emb
        )

    def _load_pas_validation_pairs(self):
        """Load the validation PAS export once on rank zero."""
        if self._pas_validation_pairs is None:
            if not self.pas_validation_datasets:
                raise ValueError(
                    "Metadata-aware validation requires at least one PAS "
                    "validation dataset."
                )
            pairs = []
            for dataset in self.pas_validation_datasets:
                pairs_file = _pas_pairs_file(dataset)
                dataset_pairs = load_pas_pairs(
                    dataset,
                    pairs_file,
                    ground_truth_mode=self.metadata_match_mode,
                )
                if not dataset_pairs:
                    raise ValueError(
                        "No PAS validation pairs were loaded from "
                        f"{pairs_file}."
                    )
                pairs.extend(dataset_pairs)
            self._pas_validation_pairs = pairs
            logging.info(
                "Loaded %s PAS validation pairs from %s dataset configs.",
                f"{len(pairs):,}",
                len(self.pas_validation_datasets),
            )
        return self._pas_validation_pairs

    def _evaluate_accumulated_pas(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> dict:
        """Gather, deduplicate, and evaluate exact PAS metrics on rank zero."""
        if len(image_emb) != len(text_emb) or len(image_emb) != len(row_indices):
            raise ValueError(
                "PAS validation embeddings and row indices must have the same "
                f"length, got {len(image_emb)}, {len(text_emb)}, and "
                f"{len(row_indices)}."
            )

        gathered_images = _all_gather_variable_rows(image_emb, self.device)
        gathered_text = _all_gather_variable_rows(text_emb, self.device)
        gathered_indices = _all_gather_variable_rows(
            row_indices.reshape(-1, 1),
            self.device,
        )

        weighted_rows = None
        evaluation_error = None
        if not _distributed_ready() or dist.get_rank() == 0:
            try:
                pairs = self._load_pas_validation_pairs()
                image_embeddings, text_embeddings = (
                    build_pas_embedding_maps_from_rows(
                        pairs,
                        gathered_indices.flatten().numpy(),
                        gathered_images,
                        gathered_text,
                    )
                )
                evaluation = evaluate_pas_metadata_embeddings(
                    pairs,
                    image_embeddings,
                    text_embeddings,
                    ground_truth_mode=self.metadata_match_mode,
                )
                weighted_rows = evaluation[
                    "metadata_weighted_aggregate"
                ]
                present_query_types = {
                    row["QueryType"] for row in weighted_rows
                }
                missing_query_types = (
                    set(PAS_METADATA_QUERY_TYPES) - present_query_types
                )
                if missing_query_types:
                    raise ValueError(
                        "PAS validation did not produce all required query "
                        f"types; missing {sorted(missing_query_types)}."
                    )
            except Exception as error:  # Keep peer ranks out of a deadlock.
                evaluation_error = error

        if _distributed_ready():
            status = torch.tensor(
                [1 if evaluation_error is not None else 0],
                dtype=torch.uint8,
                device=_collective_device(self.device),
            )
            dist.broadcast(status, src=0)
            if status.item():
                if evaluation_error is not None:
                    raise evaluation_error
                raise RuntimeError(
                    "PAS validation failed on distributed rank zero."
                )
        elif evaluation_error is not None:
            raise evaluation_error
        return _broadcast_pas_metrics(weighted_rows, self.device)

    def _log_pas_metrics(self, metrics: dict, prefix: str) -> None:
        """Log query-weighted PAS mAP for easy, medium, and hard queries."""
        for query_type in PAS_METADATA_QUERY_TYPES:
            query_metrics = metrics.get(query_type)
            if query_metrics is None:
                continue
            name = f"{prefix}/pas/{query_type}_mAP"
            map_score = query_metrics["mAP"]
            self.log(name, map_score, sync_dist=True)
            self.status_logging_dict[name] = str(map_score)
            logging.info(
                "%s: %.6f (%s deduplicated queries)",
                name,
                map_score,
                f"{query_metrics['num_queries']:,}",
            )

    def _evaluate_and_log_paired_retrieval(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        prefix: str,
    ) -> None:
        """Evaluate and log the existing paired retrieval metrics."""
        retrieval_metrics = self._evaluate_accumulated_retrieval(
            image_emb.numpy(),
            text_emb.numpy(),
        )
        log_retrieval_metrics(retrieval_metrics, prefix=prefix)
        for direction in ['image_to_text', 'text_to_image']:
            metrics = retrieval_metrics[direction]
            dir_prefix = 'i2t' if direction == 'image_to_text' else 't2i'
            values = {
                "mAP": metrics.map_score,
                "R@1": metrics.recall_at_k[1],
                "R@5": metrics.recall_at_k[5],
                "MedR": metrics.median_rank,
                "MeanR": metrics.mean_rank,
                "AUC": metrics.auc,
            }
            for name, value in values.items():
                self.log(
                    f"{prefix}/{dir_prefix}_{name}",
                    value,
                    sync_dist=True,
                )
            for name in ("mAP", "R@1", "MedR", "AUC"):
                self.status_logging_dict[
                    f"{prefix}/{dir_prefix}_{name}"
                ] = str(values[name])

    def on_validation_epoch_end(self):
        """Compute and log retrieval metrics."""
        self.status_logging_dict = {}

        if self.retrieval_evaluator is not None and self.image_embeddings:
            image_emb = torch.cat(self.image_embeddings, dim=0)
            text_emb = torch.cat(self.text_embeddings, dim=0)
            if self.metadata_match_eval:
                if not self.trainer.sanity_checking:
                    if not self.pas_row_indices:
                        raise ValueError(
                            "PAS validation requires aligned row indices."
                        )
                    pas_metrics = self._evaluate_accumulated_pas(
                        image_emb,
                        text_emb,
                        torch.cat(self.pas_row_indices, dim=0),
                    )
                    self._log_pas_metrics(pas_metrics, "val")
            else:
                self._evaluate_and_log_paired_retrieval(
                    image_emb,
                    text_emb,
                    "val",
                )

        if not self.trainer.sanity_checking and self.status_logging_dict:
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

    # Test methods (reuse validation logic)
    def on_test_epoch_start(self) -> None:
        """Test epoch start - reuse validation setup."""
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        """Test step - reuse validation step."""
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """Test epoch end - compute and log retrieval metrics."""
        self.status_logging_dict = {}

        if self.retrieval_evaluator is not None and self.image_embeddings:
            image_emb = torch.cat(self.image_embeddings, dim=0)
            text_emb = torch.cat(self.text_embeddings, dim=0)
            if self.metadata_match_eval:
                if not self.pas_row_indices:
                    raise ValueError(
                        "PAS validation requires aligned row indices."
                    )
                pas_metrics = self._evaluate_accumulated_pas(
                    image_emb,
                    text_emb,
                    torch.cat(self.pas_row_indices, dim=0),
                )
                self._log_pas_metrics(pas_metrics, "test")
            else:
                self._evaluate_and_log_paired_retrieval(
                    image_emb,
                    text_emb,
                    "test",
                )

        if self.status_logging_dict:
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Test metrics generated.",
                status_level=status_logging.Status.RUNNING
            )
