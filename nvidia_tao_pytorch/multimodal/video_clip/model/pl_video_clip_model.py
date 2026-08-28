# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VideoCLIP Model PyTorch Lightning Module."""

import json
import math
import os

import numpy as np
import torch

_MAX_LOGIT_SCALE = math.log(100)

from open_clip.loss import ClipLoss, SigLipLoss  # noqa: E402

from nvidia_tao_pytorch.core.tlt_logging import logging  # noqa: E402
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import (  # noqa: E402
    TAOLightningModule,
)
from nvidia_tao_pytorch.core.loggers import (  # noqa: E402
    api_logging as status_logging,
)
from nvidia_tao_pytorch.multimodal.video_clip.model.video_clip import build_model  # noqa: E402
from nvidia_tao_pytorch.multimodal.clip.model.lora import inject_lora  # noqa: E402
from nvidia_tao_pytorch.multimodal.clip.model.preservation_loss import (  # noqa: E402
    build_preservation_loss,
    normalize_teacher_state,
)
from nvidia_tao_pytorch.multimodal.video_clip.utils.utils import (  # noqa: E402
    build_optimizer,
    compute_lr,
    validate_peft_state_dict,
)
from nvidia_tao_pytorch.multimodal.video_clip.utils.embedding_io import (  # noqa: E402
    write_embeddings_h5,
)
from nvidia_tao_pytorch.multimodal.video_clip.model.losses import (  # noqa: E402
    InternVideo2VTCLoss,
)
from nvidia_tao_pytorch.multimodal.video_clip.model.evaluation.retrieval import (  # noqa: E402
    RetrievalEvaluator,
    log_retrieval_metrics,
)
from nvidia_tao_pytorch.multimodal.video_clip.model.evaluation.classification import (  # noqa: E402
    evaluate_classification,
    log_classification_metrics,
)
from nvidia_tao_pytorch.multimodal.video_clip.dataloader.video_text_loader import (  # noqa: E402
    load_eval_queries,
)


class VideoCLIPPlModel(TAOLightningModule):
    """PTL module for CLIP Model with retrieval-based validation."""

    def __init__(self, experiment_spec, export=False):
        """Initialize CLIP model for training."""
        super().__init__(experiment_spec)
        self.experiment_spec = experiment_spec
        self.checkpoint_filename = 'video_clip'

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
        # The exploding caption modes rely on the InternVideo2 VTC loss's
        # idx-based multi-positive targets; clip/siglip use eye() targets and
        # would treat a chunk's other captions as false negatives. Fail fast.
        _vt = getattr(
            getattr(self.experiment_spec.dataset, "train", None),
            "video_text", None,
        )
        if getattr(_vt, "caption_mode", "first") in ("all", "one_per_field") \
                and self.loss_type != "internvideo2_vtc":
            raise ValueError(
                "dataset.train.video_text.caption_mode='all'/'one_per_field' "
                "requires train.loss_type='internvideo2_vtc' (clip/siglip use "
                "eye() targets and would treat same-clip captions as false "
                "negatives)."
            )

        # PEFT (LoRA) + preservation-loss setup. Reuses the architecture-agnostic
        # clip implementation, which only needs the adapter's get_encoder_blocks
        # and the 4-tuple forward (both provided by InternVideo2CLIP). Build the
        # frozen teacher BEFORE injecting LoRA so it keeps the original pretrained
        # weights; inject_lora then freezes the backbone and adds the trainable
        # LoRA params (the per-tower optimizer drops the frozen params).
        peft_cfg = getattr(self.experiment_spec, 'peft', None)
        reg_cfg = getattr(self.experiment_spec, 'regularization', None)
        self.peft_enabled = peft_cfg is not None and peft_cfg.enabled

        self.preservation_loss = None
        if reg_cfg is not None and reg_cfg.enabled:
            self.preservation_loss = build_preservation_loss(self.model, reg_cfg)

        self.lora_stats = None
        if self.peft_enabled:
            self.lora_stats = inject_lora(self.model, peft_cfg)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logging.info(
                "LoRA trainable parameter summary: %s trainable / %s total (%.4f%%)",
                f"{trainable:,}", f"{total:,}", 100.0 * trainable / max(total, 1),
            )

        # Check if retrieval validation is configured (video_text only).
        val_cfg = getattr(self.experiment_spec.dataset, 'val', None)
        video_text_cfg = getattr(val_cfg, 'video_text', None) if val_cfg else None
        self.retrieval_enabled = (
            video_text_cfg is not None and
            bool(getattr(video_text_cfg, 'metadata', None))
        )
        # Optional explicit-relevance eval-query file (domain_test.json shape):
        # when set, the val dataset is the shared gallery and these text queries
        # are scored against it per slice using their relevant_clip_ids.
        self.eval_relevance_file = (
            getattr(video_text_cfg, 'relevance_file', None) if video_text_cfg else None
        )

    def on_load_checkpoint(self, checkpoint):
        """Validate PEFT structure using Lightning's loaded checkpoint."""
        super().on_load_checkpoint(checkpoint)
        state_dict = checkpoint.get("state_dict", checkpoint)
        normalize_teacher_state(
            state_dict, getattr(self, "preservation_loss", None)
        )
        validate_peft_state_dict(
            state_dict,
            self.experiment_spec,
            checkpoint_label="Lightning checkpoint",
        )

    def setup(self, stage=None):
        """Set up training after Trainer is initialized."""
        if stage == 'fit':
            self.max_steps = self.trainer.estimated_stepping_batches
            self._build_criterion()

    def _build_criterion(self):
        """Build the loss function."""
        if self.loss_type == 'siglip':
            self.loss = SigLipLoss(
                rank=self.global_rank,
                world_size=self.trainer.world_size,
            )
        elif self.loss_type == 'clip':
            self.loss = ClipLoss(
                rank=self.global_rank,
                world_size=self.trainer.world_size,
            )
        elif self.loss_type == 'internvideo2_vtc':
            self.loss = InternVideo2VTCLoss(
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
        train_loader = self.trainer.train_dataloader
        train_dataset = getattr(train_loader, "dataset", None)
        try:
            self._train_num_samples = (
                len(train_dataset) if train_dataset is not None else None
            )
        except TypeError:
            self._train_num_samples = None

    def _forward_pass(self, batch):
        """Run forward pass."""
        image, text = batch[0], batch[1]
        outputs = self.model(image=image, text=text)
        if self.loss_type == 'internvideo2_vtc':
            idx = batch[2] if len(batch) > 2 else None
            return outputs, idx
        return outputs

    def _backward(self, outputs, batch=None):
        """Compute loss from model outputs.

        When preservation losses are active (``regularization.enabled``), returns
        a dict ``{total, contrastive, embedding_mse, cosine, similarity}`` plus the
        logit_scale; otherwise returns the bare contrastive loss and logit_scale.
        ``batch`` is required for the teacher forward in the preservation path.
        """
        idx = None
        if self.loss_type == 'internvideo2_vtc':
            outputs, idx = outputs

        if len(outputs) == 3:
            image_features, text_features, logit_scale = outputs
            logit_bias = None
        else:
            image_features, text_features, logit_scale, logit_bias = outputs

        if self.loss_type == 'internvideo2_vtc':
            clip_loss = self.loss(
                image_features, text_features, logit_scale, idx=idx
            )
        elif logit_bias is None:
            clip_loss = self.loss(image_features, text_features, logit_scale)
        else:
            clip_loss = self.loss(
                image_features, text_features, logit_scale, logit_bias
            )

        contrastive_value = (
            clip_loss['contrastive_loss']
            if isinstance(clip_loss, dict)
            else clip_loss
        )

        preservation_loss = getattr(self, "preservation_loss", None)
        if preservation_loss is not None and batch is not None:
            image, text = batch[0], batch[1]
            pres_losses = preservation_loss(
                image_features, text_features, image, text
            )
            total_loss = contrastive_value + pres_losses['preservation_total']
            return {
                'total': total_loss,
                'contrastive': contrastive_value,
                'embedding_mse': pres_losses['embedding_mse'],
                'cosine': pres_losses['cosine'],
                'similarity': pres_losses['similarity'],
            }, logit_scale

        return clip_loss, logit_scale

    def training_step(self, batch):
        """Training step."""
        image = batch[0]
        batch_size = (
            image['pixel_values'].shape[0]
            if isinstance(image, dict)
            else image.shape[0]
        )
        outputs = self._forward_pass(batch)
        loss, logit_scale = self._backward(outputs, batch=batch)

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
        if getattr(self, "_train_num_samples", None) is not None:
            self.log(
                "train_samples",
                float(self._train_num_samples),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=False,
                batch_size=batch_size,
            )
        if isinstance(loss, dict) and 'total' in loss:
            # Preservation losses active: total = contrastive + weighted preservation.
            loss_value = loss['total']
            for key in ('contrastive', 'embedding_mse', 'cosine', 'similarity'):
                self.log(
                    f"train/{key}_loss", loss[key],
                    on_step=True, on_epoch=False, prog_bar=False,
                    sync_dist=True, batch_size=batch_size,
                )
        elif isinstance(loss, dict):
            loss_value = loss['contrastive_loss']
        else:
            loss_value = loss
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
        """Set up retrieval/classification evaluator for validation."""
        ds = self.experiment_spec.dataset
        eval_cfg = getattr(ds, 'metrics', None)
        mode_cfg = getattr(eval_cfg, 'mode', 'retrieval') or 'retrieval'
        # task_type on the val dataset also drives eval mode, so setting
        # task_type=classification puts both training-validation and the
        # standalone evaluate into classification mode (no separate flag needed).
        val_cfg = getattr(ds, 'val', None)
        val_vt = getattr(val_cfg, 'video_text', None) if val_cfg is not None else None
        val_task = getattr(val_vt, 'task_type', 'retrieval') or 'retrieval'
        self.eval_mode = (
            'classification'
            if 'classification' in (mode_cfg, val_task)
            else 'retrieval'
        )
        self.eval_exclude = list(
            getattr(eval_cfg, 'exclude_categories', None) or ["Normal", "Abnormal"]
        )
        self.image_embeddings = []
        self.text_embeddings = []
        self.sample_idxs = []
        self.row_positions = []
        if self.retrieval_enabled:
            self.retrieval_evaluator = RetrievalEvaluator(
                k_values=(1, 5, 10),
                device=self.device
            )
            logging.info(
                "Evaluator initialized for validation (mode=%s).", self.eval_mode
            )
        else:
            self.retrieval_evaluator = None
            logging.warning(
                "No validation configured. Set val.video_text.metadata "
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
        if len(batch) > 2 and batch[2] is not None:
            self.sample_idxs.append(torch.as_tensor(batch[2]).detach().cpu())
        if len(batch) > 3 and batch[3] is not None:
            self.row_positions.append(torch.as_tensor(batch[3]).detach().cpu())

    def _gather_across_ranks(self, arr):
        """All-gather a numpy array across DDP ranks (no-op when single-rank)."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return arr
        world = torch.distributed.get_world_size()
        if world <= 1:
            return arr
        gathered = [None for _ in range(world)]
        torch.distributed.all_gather_object(gathered, arr)
        parts = [g for g in gathered if g is not None and len(g)]
        return np.concatenate(parts, axis=0) if parts else arr

    def _val_dataset_entries(self):
        """Return the entries list backing the val/test dataloader, if available."""
        for attr in ("test_dataloaders", "val_dataloaders"):
            loaders = getattr(self.trainer, attr, None)
            if loaders is None:
                continue
            loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
            entries = getattr(getattr(loader, "dataset", None), "entries", None)
            if entries:
                return entries
        return None

    def _encode_texts(self, texts):
        """Encode a list of text strings through the text tower -> numpy (N, D)."""
        tokens = self.tokenizer(texts)
        if isinstance(tokens, list):
            tokens = tokens[0]
        tokens = tokens.to(self.device)
        with torch.no_grad():
            text_output = self.model(text=tokens)
        feats = (
            text_output["text_features"]
            if isinstance(text_output, dict)
            else text_output[1]
        )
        return feats.detach().cpu().numpy()

    def _compute_and_log_eval(self, prefix):
        """Compute + log retrieval or classification metrics for prefix (val/test)."""
        self.status_logging_dict = {}
        if self.retrieval_evaluator is None or not self.image_embeddings:
            return
        image_emb = self._gather_across_ranks(
            torch.cat(self.image_embeddings, dim=0).numpy()
        )
        text_emb = self._gather_across_ranks(
            torch.cat(self.text_embeddings, dim=0).numpy()
        )
        idxs = None
        if self.sample_idxs:
            idxs = self._gather_across_ranks(
                torch.cat(self.sample_idxs, dim=0).numpy().astype(int)
            )

        eval_mode = getattr(self, "eval_mode", "retrieval")
        if self.eval_relevance_file and eval_mode != "classification":
            self._eval_retrieval_relevance(prefix, image_emb, idxs)
        elif eval_mode == "classification":
            self._eval_classification(prefix, image_emb, idxs)
        else:
            self._eval_retrieval(prefix, image_emb, text_emb, idxs)

    def _eval_retrieval_relevance(self, prefix, image_emb, idxs):
        """Per-slice retrieval with the val set as a shared gallery.

        The val dataset is the gallery; the text queries and their
        ``relevant_clip_ids`` come from ``self.eval_relevance_file`` (requires
        ``idx_mode=sample_id`` so each gallery clip maps 1:1 to a chunk_id).
        Scores each ``slice`` (and overall) with the shipped RetrievalEvaluator,
        reporting mAP / recall@k / hit@k. ``near_universal`` queries are dropped
        from the headline aggregation. nDCG (computed by the evaluator) rides
        along in the JSON report but is not headlined.
        """
        entries = self._val_dataset_entries()
        if idxs is None or entries is None:
            logging.warning(
                "Explicit-relevance eval needs per-sample idx (idx_mode=sample_id) "
                "and dataset entries; falling back to idx-grouped retrieval."
            )
            text_emb = self._gather_across_ranks(
                torch.cat(self.text_embeddings, dim=0).numpy()
            )
            self._eval_retrieval(prefix, image_emb, text_emb, idxs)
            return

        # Map integer idx -> chunk_id (sample_id) and dedup the gallery to one
        # embedding per clip (the distributed sampler pads with repeats).
        idx_to_cid = {}
        for e in entries:
            if "idx" in e:
                idx_to_cid.setdefault(int(e["idx"]), str(e.get("sample_id")))
        if len({str(e.get("sample_id")) for e in entries}) > len(idx_to_cid):
            logging.warning(
                "Multiple clips share an idx; set dataset.val.video_text."
                "idx_mode=sample_id for explicit-relevance eval (gallery may be "
                "under-counted otherwise)."
            )
        gallery_cids, gallery_rows, seen = [], [], set()
        for pos, i in enumerate(idxs.tolist()):
            if i in seen or i not in idx_to_cid:
                continue
            seen.add(i)
            gallery_cids.append(idx_to_cid[i])
            gallery_rows.append(image_emb[pos])
        gallery_emb = np.stack(gallery_rows, axis=0).astype(np.float32)
        cid_to_row = {c: r for r, c in enumerate(gallery_cids)}

        queries = load_eval_queries(self.eval_relevance_file)
        q_emb = self._encode_texts([q["query"] for q in queries]).astype(np.float32)

        # Group query rows by slice; ground truth = relevant gallery indices.
        slices, n_excluded, n_missing_home = {}, 0, 0
        for i, q in enumerate(queries):
            if q["near_universal"]:
                n_excluded += 1
                continue
            if q["chunk_id"] not in cid_to_row:
                n_missing_home += 1
            rel = [cid_to_row[c] for c in q["relevant_clip_ids"] if c in cid_to_row]
            if not rel:
                continue
            slices.setdefault(q["slice"], {"rows": [], "gt": []})
            slices[q["slice"]]["rows"].append(i)
            slices[q["slice"]]["gt"].append(rel)

        report = {
            "gallery_size": len(gallery_cids),
            "n_queries_total": len(queries),
            "n_near_universal_excluded": n_excluded,
            "n_missing_home_clip": n_missing_home,
            "slices": {},
        }
        all_rows, all_gt = [], []
        for sl, d in sorted(slices.items()):
            md = self.retrieval_evaluator.evaluate(
                q_emb[d["rows"]], gallery_emb, d["gt"]
            ).to_dict()
            report["slices"][sl] = md
            self.log(f"{prefix}/{sl}_mAP", md["mAP"], sync_dist=True)
            for k in (1, 5, 10):
                if f"recall@{k}" in md:
                    self.log(f"{prefix}/{sl}_R@{k}", md[f"recall@{k}"], sync_dist=True)
                if f"hit@{k}" in md:
                    self.log(f"{prefix}/{sl}_Hit@{k}", md[f"hit@{k}"], sync_dist=True)
            self.status_logging_dict[f"{prefix}/{sl}_mAP"] = str(md["mAP"])
            logging.info(
                "[%s] slice=%s n=%d mAP=%.4f R@1=%.4f R@5=%.4f Hit@1=%.4f",
                prefix, sl, md["num_queries"], md["mAP"],
                md.get("recall@1", 0.0), md.get("recall@5", 0.0),
                md.get("hit@1", 0.0),
            )
            all_rows.extend(d["rows"])
            all_gt.extend(d["gt"])

        if all_rows:
            overall = self.retrieval_evaluator.evaluate(
                q_emb[all_rows], gallery_emb, all_gt
            ).to_dict()
            report["overall"] = overall
            self.log(f"{prefix}/overall_mAP", overall["mAP"], sync_dist=True)
            self.status_logging_dict[f"{prefix}/overall_mAP"] = str(overall["mAP"])
            logging.info(
                "[%s] OVERALL n=%d mAP=%.4f R@1=%.4f R@5=%.4f gallery=%d",
                prefix, overall["num_queries"], overall["mAP"],
                overall.get("recall@1", 0.0), overall.get("recall@5", 0.0),
                report["gallery_size"],
            )

        results_dir = getattr(self.experiment_spec, "results_dir", None)
        if results_dir and (self.trainer is None or self.trainer.is_global_zero):
            os.makedirs(results_dir, exist_ok=True)
            out = os.path.join(results_dir, f"{prefix}_retrieval_by_slice.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logging.info("Wrote per-slice retrieval report -> %s", out)

    def _eval_retrieval(self, prefix, image_emb, text_emb, idxs):
        """N-to-N retrieval metrics; relevance = shared idx (1:1 when idxs is None)."""
        # The distributed eval sampler pads each rank's shard with repeated
        # samples; drop those repeats (by unique dataset row position) so
        # multi-GPU metrics match a single-GPU run. idx alone is not unique
        # under idx_mode=category/video_id, hence the separate position key.
        positions = None
        if getattr(self, "row_positions", None):
            positions = self._gather_across_ranks(
                torch.cat(self.row_positions, dim=0).numpy().astype(int)
            )
        if positions is not None and len(positions) == len(image_emb):
            seen, keep = set(), []
            for pos, rid in enumerate(positions.tolist()):
                if rid not in seen:
                    seen.add(rid)
                    keep.append((rid, pos))
            keep = [p for _, p in sorted(keep)]
            image_emb, text_emb = image_emb[keep], text_emb[keep]
            if idxs is not None:
                idxs = idxs[keep]
        i2t_gt = t2i_gt = None
        if idxs is not None and len(idxs) == len(image_emb):
            groups = {}
            for pos, key in enumerate(idxs.tolist()):
                groups.setdefault(key, []).append(pos)
            gt = [groups[key] for key in idxs.tolist()]
            i2t_gt = t2i_gt = gt
        retrieval_metrics = self.retrieval_evaluator.evaluate_bidirectional(
            image_emb, text_emb, i2t_gt, t2i_gt
        )
        log_retrieval_metrics(retrieval_metrics, prefix=prefix)
        for direction in ['image_to_text', 'text_to_image']:
            m = retrieval_metrics[direction]
            dp = 'i2t' if direction == 'image_to_text' else 't2i'
            self.log(f"{prefix}/{dp}_mAP", m.map_score, sync_dist=True)
            self.log(f"{prefix}/{dp}_R@1", m.recall_at_k[1], sync_dist=True)
            self.log(f"{prefix}/{dp}_R@5", m.recall_at_k[5], sync_dist=True)
            if m.hit_at_k:
                self.log(f"{prefix}/{dp}_Hit@1", m.hit_at_k[1], sync_dist=True)
            self.log(f"{prefix}/{dp}_MedR", m.median_rank, sync_dist=True)
            self.log(f"{prefix}/{dp}_MeanR", m.mean_rank, sync_dist=True)
            self.log(f"{prefix}/{dp}_AUC", m.auc, sync_dist=True)
            self.status_logging_dict[f"{prefix}/{dp}_mAP"] = str(m.map_score)
            self.status_logging_dict[f"{prefix}/{dp}_R@1"] = str(m.recall_at_k[1])
            self.status_logging_dict[f"{prefix}/{dp}_MedR"] = str(m.median_rank)
            self.status_logging_dict[f"{prefix}/{dp}_AUC"] = str(m.auc)

    def _eval_classification(self, prefix, image_emb, idxs):
        """Category-level classification metrics (cosmos-embed1 parity)."""
        entries = self._val_dataset_entries()
        if idxs is None or entries is None:
            logging.warning(
                "Classification eval needs per-sample idx (set idx_mode=category) "
                "and dataset entries; falling back to retrieval metrics."
            )
            text_emb = self._gather_across_ranks(
                torch.cat(self.text_embeddings, dim=0).numpy()
            )
            self._eval_retrieval(prefix, image_emb, text_emb, idxs)
            return
        idx_to_name = {}
        for e in entries:
            if "idx" in e:
                idx_to_name.setdefault(
                    int(e["idx"]), str(e.get("category", e["idx"]))
                )
        cat_ids = sorted(idx_to_name.keys())
        names = [idx_to_name[c] for c in cat_ids]
        id_to_pos = {c: p for p, c in enumerate(cat_ids)}
        idx_list = [int(i) for i in idxs.tolist()]
        keep = np.array([i in id_to_pos for i in idx_list])
        if not keep.all():
            logging.warning(
                "%d sample(s) had idx with no category name; skipping them.",
                int((~keep).sum()),
            )
        sample_cat_idx = np.array(
            [id_to_pos[i] for i in idx_list if i in id_to_pos]
        )
        video_emb = image_emb[keep]
        category_embs = self._encode_texts(names)
        metrics = evaluate_classification(
            video_emb, category_embs, names, sample_cat_idx,
            top_k=(1, 5, 10),
            exclude=getattr(self, "eval_exclude", ["Normal", "Abnormal"]),
        )
        log_classification_metrics(metrics, prefix=prefix)
        self.log(f"{prefix}/cls_mAP", metrics.map_score, sync_dist=True)
        self.log(f"{prefix}/cls_MRR", metrics.mrr, sync_dist=True)
        self.log(f"{prefix}/cls_macroF1", metrics.macro_f1, sync_dist=True)
        for k, v in metrics.top_k_hit.items():
            self.log(f"{prefix}/cls_top{k}", v, sync_dist=True)
        self.status_logging_dict[f"{prefix}/cls_mAP"] = str(metrics.map_score)
        self.status_logging_dict[f"{prefix}/cls_MRR"] = str(metrics.mrr)
        self.status_logging_dict[f"{prefix}/cls_macroF1"] = str(metrics.macro_f1)

    def on_validation_epoch_end(self):
        """Compute and log validation metrics (retrieval or classification)."""
        self._compute_and_log_eval(prefix="val")
        if not self.trainer.sanity_checking and self.status_logging_dict:
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

    # Test methods (reuse validation logic, or extract inference embeddings)
    def on_test_epoch_start(self) -> None:
        """Test epoch start - inference extraction setup, or validation setup."""
        if getattr(self, "_inference_embed_cfg", None) is not None:
            # Embedding extraction does not need the retrieval evaluator or a
            # configured dataset.val; collect straight from the corpus loader.
            self.image_embeddings = []
            self.text_embeddings = []
            self.sample_idxs = []
            return
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        """Test step - collect inference embeddings, or validation step."""
        if getattr(self, "_inference_embed_cfg", None) is not None:
            return self._collect_inference_batch(batch)
        return self.validation_step(batch, batch_idx)

    def _collect_inference_batch(self, batch):
        """Collect visual (video) features + per-sample idx for one batch."""
        output = self.model(image=batch[0])
        feats = (
            output["image_features"]
            if isinstance(output, dict)
            else output[0]
        )
        self.image_embeddings.append(feats.detach().cpu())
        if len(batch) > 2 and batch[2] is not None:
            self.sample_idxs.append(torch.as_tensor(batch[2]).detach().cpu())

    def on_test_epoch_end(self):
        """Test epoch end - save inference embeddings, or compute eval metrics."""
        if getattr(self, "_inference_embed_cfg", None) is not None:
            self._save_inference_video_embeddings()
            return
        self._compute_and_log_eval(prefix="test")
        if self.status_logging_dict:
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Test metrics generated.",
                status_level=status_logging.Status.RUNNING
            )

    def _save_inference_video_embeddings(self):
        """Gather video embeddings across DDP ranks and write them once.

        Reuses the validation/test embedding collection (``self.image_embeddings``
        / ``self.sample_idxs``). The distributed eval sampler pads with duplicate
        samples so the per-rank shards divide evenly; we de-dup by the integer
        sample idx (1:1 with ``sample_id`` under ``idx_mode=sample_id``) and
        restore the dataset's original order. The all-gather is a collective and
        must run on every rank; only global rank 0 writes the file.
        """
        cfg = self._inference_embed_cfg
        if not self.image_embeddings:
            logging.warning("No video embeddings collected during inference.")
            return
        image_emb = self._gather_across_ranks(
            torch.cat(self.image_embeddings, dim=0).numpy()
        )
        idxs = None
        if self.sample_idxs:
            idxs = self._gather_across_ranks(
                torch.cat(self.sample_idxs, dim=0).numpy().astype(int)
            )
        # Only rank 0 writes (gather above already ran on all ranks).
        if self.trainer is not None and not self.trainer.is_global_zero:
            return

        entries = self._val_dataset_entries() or []
        idx_to_sid, idx_to_order = {}, {}
        for order, entry in enumerate(entries):
            if "idx" in entry:
                i = int(entry["idx"])
                idx_to_sid.setdefault(
                    i, str(entry.get("sample_id") or
                           entry.get("video_id") or
                           entry.get("video_path") or i)
                )
                idx_to_order.setdefault(i, order)

        if idxs is None or len(idxs) != len(image_emb):
            logging.warning(
                "No per-sample idx available; saving in encounter order without "
                "de-duplication."
            )
            ids = [str(i) for i in range(len(image_emb))]
            emb = image_emb
        else:
            seen, rows = set(), []
            for pos, i in enumerate(int(x) for x in idxs.tolist()):
                if i in seen:
                    continue
                seen.add(i)
                rows.append((idx_to_order.get(i, 1 << 30), pos, i))
            rows.sort(key=lambda r: r[0])
            keep_pos = [pos for _, pos, _ in rows]
            emb = image_emb[keep_pos]
            ids = [idx_to_sid.get(i, str(i)) for _, _, i in rows]

        write_embeddings_h5(
            cfg["path"], ids, emb, "video", provenance=cfg.get("provenance"),
        )
        logging.info(
            "Saved %d video embeddings to %s", len(ids), cfg["path"]
        )
        status_logging.get_status_logger().write(
            message=f"Video embeddings saved to {cfg['path']}",
            status_level=status_logging.Status.RUNNING,
        )
