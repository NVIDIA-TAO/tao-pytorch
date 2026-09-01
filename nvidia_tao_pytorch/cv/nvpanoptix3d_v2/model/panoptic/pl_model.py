# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3Dv2 PyTorch Lightning Module."""

import json
import logging
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.callbacks.model_checkpoint import TAOExceptionCheckpoint
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.panoptic.panoptic_model import NVPanoptix3Dv2Panoptic
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.vggt.models.vggt import VGGT
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.panoptic.feature_fusion import ConcatenatedFeatureFusion
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.panoptic.panoptic_decoder import NVPanoptix3Dv2PanopticDecoder
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.panoptic.loftup import LoftUpUpscaler
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.metric_scale_head import MetricScaleHead
# The ``nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils`` package is delivered by a
# follow-up patch in this feature series, so it may not be present on disk yet.
# Import it defensively -- mirroring the deferred-import convention already used
# by ``build_pl_model.pl_module_class`` -- so this module stays importable (and
# statically checkable) in the meantime. Each symbol falls back to a placeholder
# that raises a descriptive ImportError if it is ever actually used, rather than
# failing later with an opaque ``NoneType`` error far from the real cause. Once
# the utils package lands the ``try`` branch simply succeeds and behavior is
# unchanged.
try:
    from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.panoptic.losses import NVPanoptix3Dv2PanopticLoss
    from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.panoptic.engine_eval import panoptic_inference
    from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.panoptic.mAP import (
        evaluate_instance_map_segvggt,
        prepare_instance_map_sample,
    )
except ImportError as _utils_import_error:  # pragma: no cover - only before utils lands
    def _missing_panoptic_util(symbol, cause=_utils_import_error):
        """Build a placeholder for *symbol* that raises a descriptive ImportError."""
        def _raise(*_args, **_kwargs):
            """Raise an ImportError naming the symbol and the missing package."""
            raise ImportError(
                f"'{symbol}' requires the "
                f"'nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.panoptic' package, "
                f"which is not available in this installation. "
                f"Original import error: {cause}"
            )
        return _raise

    NVPanoptix3Dv2PanopticLoss = _missing_panoptic_util("NVPanoptix3Dv2PanopticLoss")
    panoptic_inference = _missing_panoptic_util("panoptic_inference")
    evaluate_instance_map_segvggt = _missing_panoptic_util("evaluate_instance_map_segvggt")
    prepare_instance_map_sample = _missing_panoptic_util("prepare_instance_map_sample")

logger = logging.getLogger(__name__)


_DECAY_WEIGHT_MODULES = (
    torch.nn.Linear,
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.MultiheadAttention,
)


def adamw_parameter_groups(
    module: torch.nn.Module,
    lr: float,
    weight_decay: float,
    component_name: str,
):
    """Split one component into strict decay and no-decay AdamW groups.

    Decay is applied only to matrix/conv weights. Everything else—including
    normalization parameters, biases, scalar temperatures, nn.Embedding
    tables, and free positional embeddings—has zero weight decay.
    """
    decay_parameter_ids = set()
    for submodule in module.modules():
        if not isinstance(submodule, _DECAY_WEIGHT_MODULES):
            continue
        for parameter_name, parameter in submodule.named_parameters(
            recurse=False,
        ):
            if parameter_name.endswith("weight") and parameter.ndim >= 2:
                decay_parameter_ids.add(id(parameter))

    decay_parameters = []
    no_decay_parameters = []
    for _, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in decay_parameter_ids:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    groups = []
    if decay_parameters:
        groups.append({
            "params": decay_parameters,
            "lr": lr,
            "lr_scale": 1.0,
            "weight_decay": weight_decay,
            "group_name": f"{component_name}_decay",
        })
    if no_decay_parameters:
        groups.append({
            "params": no_decay_parameters,
            "lr": lr,
            "lr_scale": 1.0,
            "weight_decay": 0.0,
            "group_name": f"{component_name}_no_decay",
        })
    return groups


def get_cfg_val(cfg, key, default=None):
    """Helper to get value from OmegaConf or dict."""
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def build_vggt(cfg) -> VGGT:
    """Load pretrained VGGT model from config."""
    model_cfg = cfg.model if hasattr(cfg, "model") else cfg["model"]
    backbone_cfg = get_cfg_val(model_cfg, "backbone", {})
    vggt_ckpt = get_cfg_val(backbone_cfg, "pretrained_backbone_path", None)

    img_size = model_cfg.img_size if hasattr(model_cfg, "img_size") else model_cfg["img_size"]
    patch_size = model_cfg.patch_size if hasattr(model_cfg, "patch_size") else model_cfg["patch_size"]
    embed_dim = model_cfg.embed_dim if hasattr(model_cfg, "embed_dim") else model_cfg["embed_dim"]

    model_kwargs = {
        "img_size": img_size,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
    }

    if vggt_ckpt and os.path.exists(vggt_ckpt):
        # The point-tracking head is not vendored (neither variant tracks
        # points), so drop its weights before the strict load and the remaining
        # keys still verify exactly.
        vggt = VGGT(**model_kwargs)
        state_dict = torch.load(vggt_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and "model" in state_dict \
                and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        state_dict = {
            key: value for key, value in state_dict.items()
            if not key.startswith("track_head.")
        }
        strict = bool(get_cfg_val(backbone_cfg, "strict_load", True))
        incompatible = vggt.load_state_dict(state_dict, strict=strict)
        if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            logger.warning(
                "VGGT checkpoint loaded non-strictly: %d missing keys, %d unexpected keys",
                len(incompatible.missing_keys), len(incompatible.unexpected_keys),
            )
    elif bool(get_cfg_val(backbone_cfg, "load_from_hf", False)):
        vggt = VGGT.from_pretrained("facebook/VGGT-1B", **model_kwargs)
    else:
        raise FileNotFoundError(
            f"VGGT checkpoint not found at {vggt_ckpt!r}; set "
            "model.backbone.pretrained_backbone_path or enable "
            "model.backbone.load_from_hf."
        )
    return vggt


def build_model_from_config(cfg) -> NVPanoptix3Dv2Panoptic:
    """Build the complete NVPanoptix3Dv2 model from experiment config."""
    model_cfg = cfg.model if hasattr(cfg, "model") else cfg["model"]
    variant_cfg = get_cfg_val(model_cfg, "panoptic", {})

    vggt = build_vggt(cfg)

    fusion_cfg = get_cfg_val(variant_cfg, "feature_fusion", {})
    feature_fusion = ConcatenatedFeatureFusion(
        dino_dim=get_cfg_val(fusion_cfg, "dino_dim", 1024),
        vggt_dim=get_cfg_val(fusion_cfg, "vggt_dim", 2048),
        hidden_dim=get_cfg_val(fusion_cfg, "hidden_dim", 768),
        num_heads=get_cfg_val(fusion_cfg, "num_heads", 12),
        num_layers=get_cfg_val(fusion_cfg, "num_layers", 3),
        ff_dim_mult=get_cfg_val(fusion_cfg, "ff_dim_mult", 4),
    )

    up_cfg = get_cfg_val(variant_cfg, "upscaler")
    upscaler = LoftUpUpscaler(
        input_dim=get_cfg_val(up_cfg, "input_dim", 768),
        dim=get_cfg_val(up_cfg, "dim", 384),
        output_stride=get_cfg_val(up_cfg, "output_stride", 2),
        patch_size=get_cfg_val(up_cfg, "patch_size", 14),
        color_feats=get_cfg_val(up_cfg, "color_feats", True),
        n_freqs=get_cfg_val(up_cfg, "n_freqs", 20),
        num_heads=get_cfg_val(up_cfg, "num_heads", 4),
        num_layers=get_cfg_val(up_cfg, "num_layers", 2),
    )

    pd_cfg = get_cfg_val(variant_cfg, "panoptic_decoder")

    panoptic_decoder = NVPanoptix3Dv2PanopticDecoder(
        feature_fusion=feature_fusion,
        upscaler=upscaler,
        hidden_dim=get_cfg_val(pd_cfg, "hidden_dim", 768),
        mask_dim=get_cfg_val(pd_cfg, "mask_dim", 384),
        ff_dim=get_cfg_val(pd_cfg, "ff_dim", 2048),
        num_queries=get_cfg_val(pd_cfg, "num_queries", 200),
        num_heads=get_cfg_val(pd_cfg, "num_heads", 8),
        dec_layers=get_cfg_val(pd_cfg, "dec_layers", 6),
        fixed_vocab=get_cfg_val(pd_cfg, "fixed_vocab", True),
        label_mode=get_cfg_val(pd_cfg, "label_mode", "sigmoid"),
        deep_supervision=get_cfg_val(pd_cfg, "deep_supervision", True),
        patch_size=get_cfg_val(model_cfg, "patch_size", 14),
        enable_objectness=bool(get_cfg_val(pd_cfg, "enable_objectness", True)),
    )

    backbone_cfg = get_cfg_val(model_cfg, "backbone", {})
    depth_head_cfg = get_cfg_val(backbone_cfg, "metric_depth_head", None)
    metric_depth_head = None
    if depth_head_cfg is not None and get_cfg_val(depth_head_cfg, "enable", True):
        hidden_dims = get_cfg_val(depth_head_cfg, "hidden_dims", None)
        if hidden_dims is None:
            # Back-compat: a single ``hidden_dim`` int collapses to ``[h, h//4]``.
            h = get_cfg_val(depth_head_cfg, "hidden_dim", 256)
            hidden_dims = [h, max(h // 4, 16)]
        metric_depth_head = MetricScaleHead(
            scene_token_dim=get_cfg_val(depth_head_cfg, "feat_dim", 2048),
            hidden_dims=tuple(hidden_dims),
            metric_context_views=int(
                get_cfg_val(depth_head_cfg, "metric_context_views", 5)
            ),
            # Panoptic metric depth is deliberately scale-only. Reasoning uses
            # the shared config's scale-plus-shift default.
            predict_shift=False,
        )
        logger.info(
            "MetricScaleHead enabled (%d params, context_views=%d, predict_shift=%s)",
            sum(p.numel() for p in metric_depth_head.parameters()),
            metric_depth_head.metric_context_views,
            metric_depth_head.predict_shift,
        )

    model = NVPanoptix3Dv2Panoptic(
        vggt_backbone=vggt,
        panoptic_decoder=panoptic_decoder,
        metric_depth_head=metric_depth_head,
    )

    return model


def build_criterion_from_config(cfg) -> NVPanoptix3Dv2PanopticLoss:
    """Build the loss module from experiment config."""
    train_cfg = cfg.train if hasattr(cfg, "train") else cfg["train"]
    model_cfg = cfg.model if hasattr(cfg, "model") else cfg["model"]
    pd_cfg = get_cfg_val(get_cfg_val(model_cfg, "panoptic", {}), "panoptic_decoder")

    loss_cfg = get_cfg_val(train_cfg, "panoptic", {})
    metric_cfg = get_cfg_val(train_cfg, "metric_depth", {})

    return NVPanoptix3Dv2PanopticLoss(
        dec_layers=get_cfg_val(pd_cfg, "dec_layers", 6),
        deep_supervision=get_cfg_val(pd_cfg, "deep_supervision", True),
        class_weight=get_cfg_val(loss_cfg, "class_weight", 2.0),
        rank_weight=get_cfg_val(loss_cfg, "rank_weight", 0.5),
        objectness_weight=get_cfg_val(loss_cfg, "objectness_weight", 1.0),
        objectness_no_object_weight=get_cfg_val(
            loss_cfg, "objectness_no_object_weight", 0.1,
        ),
        objectness_ignore_overlap_threshold=get_cfg_val(
            loss_cfg, "objectness_ignore_overlap_threshold", 0.5,
        ),
        mask_weight=get_cfg_val(loss_cfg, "mask_weight", 20.0),
        dice_weight=get_cfg_val(loss_cfg, "dice_weight", 1.0),
        num_points=get_cfg_val(loss_cfg, "num_loss_points", 12288),
        label_mode=get_cfg_val(pd_cfg, "label_mode", "sigmoid"),
        metric_depth_weight=get_cfg_val(metric_cfg, "weight", 5.0),
        metric_silog_weight=get_cfg_val(metric_cfg, "silog_weight", 1.0),
        metric_absrel_weight=get_cfg_val(metric_cfg, "absrel_weight", 0.1),
        metric_silog_lambda=get_cfg_val(metric_cfg, "silog_lambda", 0.85),
        metric_min_depth=get_cfg_val(metric_cfg, "min_depth", 0.1),
        metric_max_depth=get_cfg_val(metric_cfg, "max_depth", 20.0),
        # Decoupled Hungarian matching costs. ``None`` falls back to the
        # corresponding ``*_weight`` for backward compat.
    )


class NVPanoptix3Dv2PanopticPlModule(TAOLightningModule):
    """PyTorch Lightning module for NVPanoptix3Dv2 training and evaluation.

    Inherits from TAOLightningModule to get:
      - ModelCheckpoint (periodic saves every N steps + *_latest.pth symlink)
      - TAOExceptionCheckpoint (saves on SLURM signal / exception for resume)
      - TAOStatusLogger
    """

    def __init__(self, experiment_config):
        """Initialize the module, its criterion, and the pretrained weights."""
        super().__init__(experiment_config)
        self.cfg = experiment_config
        self.checkpoint_filename = 'nvpanoptix3d_v2_panoptic_model'
        # Best-checkpoint selection ranks on instance-segmentation mAP.
        self.monitor_metric = "val/mAP"
        self.monitor_mode = "max"
        self.train_cfg = experiment_config.train if hasattr(experiment_config, "train") else experiment_config["train"]

        self.model = build_model_from_config(experiment_config)
        self.criterion = build_criterion_from_config(experiment_config)

        self.model.freeze_vggt_weights()

        resume_ckpt = get_cfg_val(self.train_cfg, "resume_training_checkpoint_path", None)
        has_resume = resume_ckpt and os.path.isfile(str(resume_ckpt))
        if not has_resume:
            results_dir = getattr(experiment_config, "results_dir", None)
            if results_dir:
                latest_link = os.path.join(str(results_dir), self.checkpoint_filename + "_latest.pth")
                if os.path.exists(latest_link):
                    has_resume = True

        pretrained = get_cfg_val(self.train_cfg, "pretrained_model_path", None)
        if pretrained and os.path.isfile(str(pretrained)) and not has_resume:
            self.load_pretrained_weights(str(pretrained))
        elif pretrained and has_resume:
            logger.info("Skipping pretrained_model_path (resume checkpoint found; it takes priority)")
        elif pretrained:
            raise FileNotFoundError(
                f"Panoptic pretrained_model_path does not exist: {pretrained}"
            )

        self.classes = None
        self.eval_classes = None
        self._eval_categories = None
        self.eval_category_ids = None
        self._vocab_set = False

        eval_cfg = (
            experiment_config.evaluate
            if hasattr(experiment_config, "evaluate")
            else experiment_config.get("evaluate", {})
        )
        self.cls_threshold = get_cfg_val(eval_cfg, "cls_threshold", 0.1)
        self.mask_threshold = get_cfg_val(eval_cfg, "mask_threshold", 0.25)
        self.overlap_threshold = get_cfg_val(eval_cfg, "overlap_threshold", 0.5)
        inference_cfg = (
            experiment_config.inference
            if hasattr(experiment_config, "inference")
            else experiment_config.get("inference", {})
        )
        self.inference_cls_threshold = get_cfg_val(
            inference_cfg, "cls_threshold", 0.1,
        )
        self.inference_mask_threshold = get_cfg_val(
            inference_cfg, "mask_threshold", 0.25,
        )
        self.inference_overlap_threshold = get_cfg_val(
            inference_cfg, "overlap_threshold", 0.5,
        )

        # Compact per-sample intersection tables used for epoch-level AP.
        self.all_map_samples = []

        self._log_interval = get_cfg_val(self.train_cfg, "log_interval", 50)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"NVPanoptix3Dv2: {total_params / 1e6:.1f}M total params, {trainable_params / 1e6:.1f}M trainable")

    def load_pretrained_weights(self, ckpt_path: str):
        """Warm-start compatible model weights without optimizer state.

        Parameters absent from the current model or having incompatible shapes
        are skipped. Optimizer state, epoch counters, and schedules are not
        restored.
        """
        logger.info("Loading pretrained weights from: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "state_dict" in ckpt:
            raw_sd = ckpt["state_dict"]
            prefix = "model."
            src_sd = {k[len(prefix):]: v for k, v in raw_sd.items() if k.startswith(prefix)}
            if not src_sd:
                src_sd = raw_sd
            logger.info("  PL checkpoint (epoch=%s, step=%s)",
                        ckpt.get("epoch", "?"), ckpt.get("global_step", "?"))
        elif "model" in ckpt:
            src_sd = ckpt["model"]
        else:
            src_sd = ckpt

        dst_sd = self.model.state_dict()

        compatible_sd = {}
        loaded, skipped_shape, skipped_missing = [], [], []
        for key, src_val in src_sd.items():
            if key not in dst_sd:
                skipped_missing.append(key)
                continue
            if src_val.shape != dst_sd[key].shape:
                skipped_shape.append((key, tuple(src_val.shape), tuple(dst_sd[key].shape)))
                continue
            compatible_sd[key] = src_val
            loaded.append(key)

        # Missing parameters retain their current initialization.
        load_result = self.model.load_state_dict(compatible_sd, strict=False)
        new_only = list(load_result.missing_keys)

        logger.info("  Loaded %d/%d params", len(loaded), len(dst_sd))
        if skipped_shape:
            logger.info("  Skipped %d params (shape mismatch):", len(skipped_shape))
            for k, s_old, s_new in skipped_shape:
                logger.info("    %s: %s -> %s", k, s_old, s_new)
        if skipped_missing:
            n_vggt = sum(1 for k in skipped_missing if k.startswith("vggt_backbone."))
            n_other = len(skipped_missing) - n_vggt
            if n_other > 0:
                logger.info("  %d checkpoint keys not in model (dropped):", n_other)
                for k in skipped_missing:
                    if not k.startswith("vggt_backbone."):
                        logger.info("    %s", k)
        if new_only:
            n_vggt = sum(1 for k in new_only if k.startswith("vggt_backbone."))
            n_other = len(new_only) - n_vggt
            if n_other > 0:
                logger.info("  %d new model params (randomly initialised):", n_other)
                for k in new_only:
                    if not k.startswith("vggt_backbone."):
                        logger.info("    %s", k)

    def configure_callbacks(self) -> Sequence[Callback] | pl.Callback:
        """Configure standard periodic, best, and exception checkpoints."""
        results_dir = self.experiment_spec["results_dir"]
        checkpoint_interval = self.experiment_spec["train"]["checkpoint_interval"]
        checkpoint_interval_unit = self.experiment_spec["train"].get(
            "checkpoint_interval_unit", "epoch",
        )

        status_logger_callback = TAOStatusLogger(results_dir, append=True)

        ModelCheckpoint.FILE_EXTENSION = ".pth"
        ModelCheckpoint.CHECKPOINT_EQUALS_CHAR = "_"

        if not self.checkpoint_filename:
            raise NotImplementedError("checkpoint_filename not set in __init__() of model")
        ModelCheckpoint.CHECKPOINT_NAME_LAST = f"{self.checkpoint_filename}_latest"

        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=(
                checkpoint_interval if checkpoint_interval_unit == "epoch" else None
            ),
            every_n_train_steps=(
                checkpoint_interval if checkpoint_interval_unit == "step" else None
            ),
            dirpath=results_dir,
            save_on_train_epoch_end=True,
            monitor=None,
            save_top_k=-1,
            save_last='link',
            filename='model_{epoch:03d}_{step:06d}',
            enable_version_counter=False,
        )

        TAOExceptionCheckpoint.FILE_EXTENSION = ModelCheckpoint.FILE_EXTENSION
        TAOExceptionCheckpoint.CHECKPOINT_NAME_LAST = ModelCheckpoint.CHECKPOINT_NAME_LAST
        exception_checkpoint_callback = TAOExceptionCheckpoint(dirpath=results_dir)

        callbacks = [status_logger_callback, checkpoint_callback, exception_checkpoint_callback]

        # Shared TAO top-k stream, ranked on val/mAP by this model's default or
        # by an explicit train.checkpointer.monitor override.
        callbacks = self._configure_best_checkpoint(callbacks, results_dir)

        return callbacks

    def set_classes(
        self,
        classes: List[str],
        categories: Optional[List[Dict]] = None,
    ):
        """Set the class vocabulary for open-vocabulary prediction.

        Args:
            classes: bare class names. Indexed by ``pan_cls_id`` and used
                as the canonical prediction vocabulary.
            categories: optional list of category dicts with ``id``, ``name``,
                and ``isthing`` fields. Instance mAP evaluates only categories
                marked as things.
        """
        self.classes = list(classes)
        self.eval_classes = list(classes)
        self._eval_categories = categories
        if self._eval_categories is None:
            self.eval_category_ids = list(range(len(self.eval_classes)))
        else:
            if len(self._eval_categories) != len(self.eval_classes):
                raise ValueError(
                    "eval_categories must align one-to-one with eval_classes: "
                    f"{len(self._eval_categories)} != {len(self.eval_classes)}"
                )
            self.eval_category_ids = [
                category_id
                for category_id, category in enumerate(self._eval_categories)
                if bool(category.get("isthing", 1))
            ]
        if not self.eval_category_ids:
            raise ValueError("instance mAP requires at least one evaluated thing category")

        self.model.set_vocab(self.classes, device=self.device)
        self._vocab_set = True

    def on_train_epoch_start(self):
        """Update the train sampler at the start of each epoch."""
        # Propagate the epoch to the train sampler so the data is reshuffled
        # every epoch. ``train.py`` sets ``use_distributed_sampler=False``
        # (the data modules build their own DistributedSampler /
        # HomogeneousBatchSampler), so PyTorch Lightning does NOT call
        # ``sampler.set_epoch`` for us. Without this the sampler keeps
        # re-using its epoch-0 seed and every epoch sees the identical batch
        # ordering / dataset-choice sequence.
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "set_train_epoch"):
            datamodule.set_train_epoch(self.current_epoch)

    def zero_loss(self):
        """Return a zero loss connected to model params so DDP gradient sync works."""
        return sum(p.reshape(-1)[0] * 0.0 for p in self.model.parameters() if p.requires_grad)

    def training_step(self, batch, batch_idx):
        """Training step."""
        if not self._vocab_set:
            logger.warning("Vocabulary not set. Call set_classes() before training.")
            return self.zero_loss()

        imgs = torch.stack([b["img"] for b in batch], dim=1)
        true_shape = torch.stack([b["true_shape"] for b in batch], dim=1)

        precision = str(get_cfg_val(self.train_cfg, "precision", "fp32")).lower()
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Collators doing per-batch text augmentation attach a ``vocab`` to the
        # first view dict, replacing the model's text input and aligning the
        # criterion's class indices with the rewritten ``pan_cls_id`` pixels.
        vocab = batch[0].get("vocab", self.classes) if isinstance(batch[0], dict) else self.classes

        with torch.amp.autocast("cuda", dtype=dtype):
            panout, geo_out = self.model(imgs, true_shape, vocab)

        gt_depth = None
        if "depthmap" in batch[0]:
            gt_depth = torch.stack([b["depthmap"] for b in batch], dim=1)  # [B, S, H, W]

        with torch.amp.autocast("cuda", dtype=torch.float32):
            loss, loss_details = self.criterion(
                batch, panout, vocab,
                geometry_output=geo_out,
                gt_depth=gt_depth,
            )

        if not torch.isfinite(loss):
            logger.warning(
                "Non-finite loss at step %d (rank %d) — using zero loss for DDP safety.",
                batch_idx, self.global_rank,
            )
            loss = self.zero_loss()
            loss_details = {k: torch.zeros(1, device=imgs.device) for k in loss_details}

        total_steps = self.trainer.num_training_batches
        step = batch_idx + 1
        should_log = step % self._log_interval == 0 or step == total_steps
        if should_log:
            for k, v in loss_details.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self.log(f"train/{k}", v, prog_bar=False, sync_dist=True)

        return loss

    def extract_scene_id(self, batch, bi: int):
        """Extract scene_id from a batch sample's label."""
        label = batch[0].get("label")
        if label is None:
            return None
        if isinstance(label, (list, tuple)):
            label = label[bi]
        parts = label.rsplit("_", 1)
        return parts[0] if len(parts) >= 2 else label

    def forward_panoptic(self, batch, *, for_inference=False):
        """Run the model and panoptic post-processing on one evaluation batch.

        Returns:
            tuple of (pred_results, geometry_output, batch_size, num_views).
        """
        imgs = torch.stack([b["img"] for b in batch], dim=1)
        true_shape = torch.stack([b["true_shape"] for b in batch], dim=1)

        # Evaluation uses the same native ScanNet++ taxonomy as training.
        vocab = self.eval_classes

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            panout, geo_out = self.model(imgs, true_shape, vocab)

        if for_inference:
            cls_threshold = self.inference_cls_threshold
            mask_threshold = self.inference_mask_threshold
            overlap_threshold = self.inference_overlap_threshold
        else:
            cls_threshold = self.cls_threshold
            mask_threshold = self.mask_threshold
            overlap_threshold = self.overlap_threshold

        H, W = true_shape[0, 0].tolist()
        pred_results = panoptic_inference(
            mask_cls=panout["pred_logits"],
            mask_pred=panout["pred_masks"],
            target_hw=(int(H), int(W)),
            label_mode=self.model.panoptic_decoder.label_mode,
            cls_threshold=cls_threshold,
            mask_threshold=mask_threshold,
            overlap_threshold=overlap_threshold,
            objectness_logits=panout.get("pred_objectness"),
        )
        return pred_results, geo_out, imgs.shape[0], imgs.shape[1]

    def accumulate_map(self, batch, pred_results, batch_size, num_views):
        """Convert one batch into compact per-sample AP matching tables."""
        num_classes = len(self.eval_classes)
        for bi in range(batch_size):
            pred_pan = pred_results[bi]["pan"].cpu().numpy()
            pred_segments = pred_results[bi]["segments_info"]
            gt_instance_ids = np.stack([
                batch[v]["pan_inst_id"][bi].cpu().numpy()
                for v in range(num_views)
            ])
            gt_class_ids = np.stack([
                batch[v]["pan_cls_id"][bi].cpu().numpy()
                for v in range(num_views)
            ])
            self.all_map_samples.append(prepare_instance_map_sample(
                pred_pan,
                pred_segments,
                gt_instance_ids,
                gt_class_ids,
                num_categories=num_classes,
                evaluated_category_ids=self.eval_category_ids,
            ))

    def aggregate_map(self, prefix: str):
        """Aggregate all ranks into dataset-level mAP, mAP50, and mAP25.

        Per-sample intersection tables are gathered from every DDP rank before
        AP is computed, so confidence ranking and false-negative accounting
        cover the complete evaluation set.

        Args:
            prefix: metric namespace, e.g. ``val`` or ``test``.

        Returns:
            The three-metric dictionary, or ``None`` if no samples were seen.
        """
        if self.eval_category_ids is None:
            return None

        if dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, self.all_map_samples)
            all_samples = []
            for rank_samples in gathered:
                all_samples.extend(rank_samples)
        else:
            all_samples = list(self.all_map_samples)

        if not all_samples:
            self.all_map_samples.clear()
            return None

        metrics = evaluate_instance_map_segvggt(
            all_samples, self.eval_category_ids,
        )
        self.log(f"{prefix}/mAP", metrics["mAP"], prog_bar=True)
        self.log(f"{prefix}/mAP50", metrics["mAP50"])
        self.log(f"{prefix}/mAP25", metrics["mAP25"])

        self.all_map_samples.clear()
        return metrics

    def validation_step(self, batch, batch_idx):
        """Validation step with instance mAP tracking."""
        del batch_idx
        if not self._vocab_set:
            return None

        pred_results, _, batch_size, num_views = self.forward_panoptic(batch)
        self.accumulate_map(batch, pred_results, batch_size, num_views)
        return None

    def on_validation_epoch_end(self):
        """Aggregate validation mAP, mAP50, and mAP25."""
        self.aggregate_map("val")

    def test_step(self, batch, batch_idx):
        """Evaluation step. Identical scoring to validation."""
        del batch_idx
        if not self._vocab_set:
            return None

        pred_results, _, batch_size, num_views = self.forward_panoptic(batch)
        self.accumulate_map(batch, pred_results, batch_size, num_views)
        return None

    def on_test_epoch_end(self):
        """Aggregate the evaluation metrics and persist them to results_dir."""
        metrics = self.aggregate_map("test")
        if metrics is None or self.global_rank != 0:
            return

        results_dir = self.experiment_spec["results_dir"]
        if not results_dir:
            return
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "instance_map.json")
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump({k: float(v) for k, v in metrics.items()}, handle, indent=2)
        logger.info("Wrote evaluation metrics to %s", results_path)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Inference step. Dumps one panoptic map per sample to results_dir.

        Each sample is written as a compressed ``.npz`` holding the multi-view
        panoptic ID map, the per-segment ``id``/``category_id``/``score``
        table, and the class names those category IDs index into. Metric depth
        and metric world points are included when the metric-scale head is
        enabled.
        """
        del dataloader_idx
        if not self._vocab_set:
            return None

        pred_results, geo_out, batch_size, _ = self.forward_panoptic(
            batch, for_inference=True,
        )

        output_dir = self.inference_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        for bi in range(batch_size):
            payload = {
                "pan": pred_results[bi]["pan"].cpu().numpy(),
                "segments_info": np.array(
                    json.dumps(pred_results[bi]["segments_info"]), dtype=object,
                ),
                "classes": np.array(self.eval_classes, dtype=object),
            }
            for key in ("metric_depth", "metric_points"):
                if key in geo_out:
                    payload[key] = geo_out[key][bi].float().cpu().numpy()

            label = self.extract_scene_id(batch, bi) or "sample"
            index = batch_idx * batch_size + bi
            out_path = os.path.join(
                output_dir, f"{label}_{self.global_rank:02d}_{index:06d}.npz",
            )
            np.savez_compressed(out_path, **payload)

        return None

    def inference_output_dir(self):
        """Resolve where predict_step writes, honouring inference.output_dir."""
        inference_cfg = get_cfg_val(self.cfg, "inference", None)
        output_dir = get_cfg_val(inference_cfg, "output_dir", None)
        if output_dir:
            return str(output_dir)
        return os.path.join(str(self.experiment_spec["results_dir"]), "predictions")

    def configure_optimizers(self):
        """Configure optimizer with per-component learning rates and a
        linear-warmup -> cosine-decay LR schedule.

        The schedule is stepped **per optimizer step** and honours the
        ``train`` config's ``warmup_epochs``, ``num_epochs`` and ``min_lr``.
        It is implemented as a multiplicative :class:`LambdaLR`, so every
        parameter group is annealed by the same factor.

        Previously this method returned only the optimizer, so
        ``warmup_epochs`` / ``min_lr`` were silently ignored and the LR
        stayed constant at ``lr`` for the entire run.
        """
        lr = get_cfg_val(self.train_cfg, "lr", 1e-4)
        min_lr = get_cfg_val(self.train_cfg, "min_lr", 1e-6)
        weight_decay = get_cfg_val(self.train_cfg, "weight_decay", 0.05)

        param_groups = adamw_parameter_groups(
            self.model.panoptic_decoder,
            lr=lr,
            weight_decay=weight_decay,
            component_name="panoptic_decoder",
        )

        if self.model.metric_depth_head is not None:
            head_groups = adamw_parameter_groups(
                self.model.metric_depth_head,
                lr=lr,
                weight_decay=weight_decay,
                component_name="metric_depth_head",
            )
            param_groups.extend(head_groups)

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )

        # LR schedule: linear warmup -> cosine decay (per optim step)
        # ``estimated_stepping_batches`` is the total optimizer-step budget
        # over the whole run; it already accounts for max_epochs,
        # grad accumulation, devices and nodes. Guard with a fallback so a
        # missing/odd trainer state degrades to the old constant-LR behaviour
        # instead of crashing.
        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            total_steps = 0
        if total_steps <= 1:
            return optimizer

        num_epochs = int(get_cfg_val(self.train_cfg, "num_epochs", 10) or 0)
        warmup_epochs = float(get_cfg_val(self.train_cfg, "warmup_epochs", 2.0) or 0)

        # Convert warmup epochs to a fraction of the DDP-aware optimizer-step
        # budget. ``estimated_stepping_batches`` already accounts for devices,
        # accumulation, the epoch limit, and the distributed sampler.
        if num_epochs > 0 and warmup_epochs > 0:
            warmup_frac = min(max(warmup_epochs / num_epochs, 0.0), 0.5)
            warmup_steps = int(total_steps * warmup_frac)
        else:
            warmup_steps = 0
        warmup_steps = max(0, min(warmup_steps, total_steps - 1))

        min_ratio = (float(min_lr) / float(lr)) if lr > 0 else 0.0
        min_ratio = max(0.0, min(min_ratio, 1.0))

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        logger.info(
            "LR schedule: linear warmup %d steps -> cosine decay over %d total "
            "steps (lr %.2e -> %.2e, %d param groups)",
            warmup_steps, total_steps, lr, min_lr, len(param_groups),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
