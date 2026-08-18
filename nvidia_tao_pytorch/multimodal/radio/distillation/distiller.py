# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distiller module for RADIO model"""
import os
import logging
import inspect
import re
import copy
from typing import Sequence
import numpy as np

from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import distributed as dist

import torch.optim as optim
from torch.optim import lr_scheduler
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from transformers.optimization import get_cosine_schedule_with_warmup

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.callbacks.ema import EMA, EMAModelCheckpoint
from nvidia_tao_pytorch.core.utilities import get_latest_checkpoint
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank

from nvidia_tao_pytorch.core.distillation.distiller import Distiller

from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from nvidia_tao_pytorch.multimodal.radio.distillation.loss import DistillationLoss
from nvidia_tao_pytorch.multimodal.radio.distillation.validate import (
    build_knn_index,
    knn_eval_batch,
)
from nvidia_tao_pytorch.multimodal.radio.distillation.vitdet import (
    VitDetArgs,
    apply_vitdet_to_vit,
)
from nvidia_tao_pytorch.multimodal.radio.distillation.featsharp_adaptor import (
    wrap_teacher_with_featsharp,
)
from nvidia_tao_pytorch.multimodal.radio.model.model_builder import build_model
logger = logging.getLogger(__name__)


def _resolve_distillation_runtime_modes(distill_config):
    """Resolve DDP and BatchNorm modes for the configured topology."""
    if getattr(distill_config, "partitioned_ranks", False):
        # Partitioned training keeps running statistics local, disables rank-0
        # buffer broadcast, then averages student BatchNorm buffers at epoch end.
        return False, False, "reduce"
    return (
        getattr(distill_config, "sync_bn_mode", "global") == "global",
        bool(getattr(distill_config, "broadcast_buffers", True)),
        str(getattr(distill_config, "dist_bn_mode", "off")),
    )


def _validate_mosaic_config(teacher_config):
    """Validate fixed-canvas mosaic batching before model construction."""
    inner = int(teacher_config.get("mosaic_inner_size", 0) or 0)
    outer = int(teacher_config.get("mosaic_outer_size", 0) or 0)
    downsample = int(
        teacher_config.get("mosaic_downsample", 0) or 0
    )
    if inner == outer == downsample == 0:
        return
    if inner <= 0 or outer <= 0 or downsample <= 0:
        raise ValueError(
            "Mosaic batching requires positive inner, outer, and downsample sizes"
        )
    if teacher_config.get("match_student_resolution", True):
        raise ValueError(
            "Mosaic teachers must set match_student_resolution=false"
        )
    if teacher_config.get("mode") not in ("spatial", "combo"):
        raise ValueError(
            "Mosaic batching is only valid for spatial or combo teacher arms"
        )
    if float(teacher_config.get("summary_loss_weight", 1.0)) != 0.0:
        raise ValueError(
            "Mosaic teachers must set summary_loss_weight=0"
        )
    if inner % downsample or outer % downsample:
        raise ValueError(
            "Mosaic inner and outer sizes must be divisible by "
            "mosaic_downsample"
        )
    inner_grid = inner // downsample
    outer_grid = outer // downsample
    if inner_grid < outer_grid:
        raise ValueError(
            f"Mosaic canvas {inner} cannot contain outer view {outer} "
            f"at feature stride {downsample}"
        )


@torch.no_grad()
def _distribute_batch_norm_buffers(module: nn.Module, mode: str) -> int:
    """Distribute BatchNorm running buffers at the epoch boundary."""
    if mode == "off" or not dist.is_initialized():
        return 0
    if mode not in ("broadcast", "reduce"):
        raise ValueError(f"Unsupported dist_bn_mode: {mode}")

    world_size = dist.get_world_size()
    distributed_buffers = 0
    for child in module.modules():
        if not isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            continue
        for buffer in (child.running_mean, child.running_var):
            if buffer is None:
                continue
            if mode == "reduce":
                dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
                buffer.div_(world_size)
            else:
                dist.broadcast(buffer, src=0)
            distributed_buffers += 1
    return distributed_buffers


class MultiTeacherDistiller(Distiller):
    """Multi-Teacher Distiller for RADIO."""

    def __init__(self, experiment_spec, export=False):
        """Initializes the distiller from given experiment_spec."""
        # Init local params
        self.experiment_spec = experiment_spec
        self.checkpoint_filename = "classifier_model"
        self.dataset_config = self.experiment_spec.dataset
        self.model_config = self.experiment_spec.model
        self.train_config = self.experiment_spec.train
        self.eval_config = self.experiment_spec.evaluate
        self.infer_config = self.experiment_spec.inference
        self.distill_config = self.experiment_spec.distill

        self.status_logging_dict = {}
        self.lr = self.train_config.optim.lr
        self.optimizer = self.train_config.optim
        self.lr_policy = self.optimizer.policy
        self.lr_policy_params = self.optimizer.policy_params
        self.max_epochs = self.train_config.num_epochs
        self.monitor_name = self.train_config.optim.monitor_name

        self.num_classes = 0

        # Parse teacher configurations (support single or multiple teachers)
        self.teacher_configs = self._parse_teacher_configs()
        self._validate_partitioned_config()
        self._head_warmup_active = None
        self._student_norm_freeze_logged = False
        self._student_bn_distribution_logged = False
        self._partitioned_state_sync_step = None
        self._baseline_anchor_model = None
        self._baseline_anchor_hooks = []
        self._baseline_anchor_student_features = {}
        self._baseline_anchor_anchor_features = {}

        self._pretrained_head_sd = None
        self._native_head_sd = None
        pretrained_path = getattr(self.model_config.backbone, 'pretrained_backbone_path', None)
        if pretrained_path and getattr(self.distill_config, "warmstart_projection_heads", True):
            self._detect_upstream_head_info(pretrained_path)

        # Global defaults for backward compatibility
        self.distill_weight = self.distill_config.loss_lambda
        self.distill_loss = self.distill_config.loss_type

        # #  training log
        self.epoch_acc = 0
        self.max_num_epochs = self.train_config.num_epochs
        self.batch = None
        self.vis_dir = self.experiment_spec.results_dir
        self.optimizer_G = None

        self.vis_after_n_batches = self.eval_config.vis_after_n_batches
        self.vis_after_n_batches_infer = self.infer_config.vis_after_n_batches
        # init the model
        super().__init__(experiment_spec, export)

        self.batch_size = self.dataset_config.batch_size

        if self._pretrained_head_sd is not None:
            self._warmstart_projection_heads(self._pretrained_head_sd)
            self._pretrained_head_sd = None
        if self._native_head_sd is not None:
            self._warmstart_native_heads()
            self._native_head_sd = None

        self._configure_projection_head_spectral_norm()
        self._configure_backbone_spectral_norm()
        self._warmstart_training_weights()
        self._freeze_distillation_statistics_if_configured()
        self._prefreeze_projection_heads_if_configured()
        self._setup_baseline_anchor_model()
        self._enforce_frozen_student_norms()

        student_mean, student_std = self._resolve_student_normalization()
        self.register_buffer(
            "_student_mean", student_mean
        )
        self.register_buffer(
            "_student_std", student_std
        )

        # Best-checkpoint default: monitor kNN top-1 probe accuracy (logged as "knn_top1"
        # in on_validation_epoch_end). Set after super().__init__ so it is not overwritten.
        # The base fallback ("val_loss", min) is never logged by this module, so without
        # this the enable_topk best-checkpoint would monitor a metric that never appears.
        # Note: knn_top1 is only logged when kNN eval runs; otherwise pass an explicit
        # train.checkpointer.monitor.
        self.monitor_metric = "knn_top1"
        self.monitor_mode = "max"

    def configure_callbacks(self) -> Sequence[Callback] | pl.Callback:
        """Configures logging and checkpoint-saving callbacks"""
        # This is called when trainer.fit() is called
        self.checkpoint_filename = "classifier_model"
        callbacks = []
        results_dir = self.experiment_spec["results_dir"]
        checkpoint_interval = self.experiment_spec["train"]["checkpoint_interval"]
        checkpoint_interval_unit = self.experiment_spec["train"].get("checkpoint_interval_unit", "epoch")
        checkpoint_keep_last_n = self.experiment_spec["train"].get("checkpoint_keep_last_n", -1)

        status_logger_callback = TAOStatusLogger(
            results_dir,
            append=True,
        )

        resume_ckpt = self.experiment_spec["train"][
            "resume_training_checkpoint_path"
        ] or get_latest_checkpoint(results_dir)

        resumed_epoch = 0
        if resume_ckpt:
            resumed_epoch = re.search("epoch_(\\d+)", resume_ckpt)
            if resumed_epoch is not None:
                resumed_epoch = int(resumed_epoch.group(1))
            else:
                resumed_epoch = 0

        status_logger_callback.epoch_counter = resumed_epoch + 1
        callbacks.append(status_logger_callback)

        if self.experiment_spec["train"]["enable_ema"]:
            # Apply Exponential Moving Average Callback
            ema_callback = EMA(**self.experiment_spec["train"]["ema"])
            ckpt_func = EMAModelCheckpoint
            callbacks.append(ema_callback)
        else:
            ckpt_func = ModelCheckpoint

        ModelCheckpoint.FILE_EXTENSION = ".pth"
        ModelCheckpoint.CHECKPOINT_EQUALS_CHAR = "_"

        if not self.checkpoint_filename:
            raise NotImplementedError(
                "checkpoint_filename not set in __init__() of model"
            )
        ModelCheckpoint.CHECKPOINT_NAME_LAST = f"{self.checkpoint_filename}_latest"

        # Lightning rejects save_top_k>0 with monitor=None. When keeping a bounded
        # number of checkpoints (checkpoint_keep_last_n > 0):
        #   - if checkpoint_monitor is set, keep the best N by that metric;
        #   - otherwise monitor 'epoch' (mode 'max') to keep the N most-recent.
        # -1 (legacy) keeps everything.
        checkpoint_monitor = (self.experiment_spec["train"].get("checkpoint_monitor", "") or "").strip()
        checkpoint_monitor_mode = self.experiment_spec["train"].get("checkpoint_monitor_mode", "min") or "min"
        if checkpoint_keep_last_n > 0:
            if checkpoint_monitor:
                ckpt_monitor = checkpoint_monitor
                ckpt_mode = checkpoint_monitor_mode
            else:
                ckpt_monitor = "epoch"
                ckpt_mode = "max"
        else:
            ckpt_monitor = None
            ckpt_mode = "min"
        checkpoint_callback = ckpt_func(
            every_n_epochs=checkpoint_interval if checkpoint_interval_unit == "epoch" else None,
            every_n_train_steps=checkpoint_interval if checkpoint_interval_unit == "step" else None,
            dirpath=results_dir,
            save_on_train_epoch_end=True,
            monitor=ckpt_monitor,
            mode=ckpt_mode,
            save_top_k=checkpoint_keep_last_n,
            save_last="link",
            filename="model_{epoch:03d}",
            enable_version_counter=False,
        )
        callbacks.append(checkpoint_callback)

        # Permanent milestone checkpoints: independently of the rolling keep_last_n
        # window, keep a checkpoint every N epochs under <results_dir>/milestones with
        # save_top_k=-1 (never pruned). Lets long runs be probed/eval'd at fixed epochs
        # without racing the rolling-window deletion. Separate dirpath so the rolling
        # callback's pruning never touches these.
        checkpoint_keep_milestone_every = int(
            self.experiment_spec["train"].get("checkpoint_keep_milestone_every", 0) or 0
        )
        if checkpoint_keep_milestone_every > 0 and checkpoint_interval_unit == "epoch":
            milestone_callback = ckpt_func(
                every_n_epochs=checkpoint_keep_milestone_every,
                dirpath=os.path.join(results_dir, "milestones"),
                save_on_train_epoch_end=True,
                monitor=None,
                mode="min",
                save_top_k=-1,
                save_last=False,
                filename="model_{epoch:03d}",
                enable_version_counter=False,
            )
            callbacks.append(milestone_callback)

        # Best-checkpoint saving (additive by default, or replacing the periodic
        # callback when train.checkpointer.replace_periodic is enabled).
        # This module overrides configure_callbacks without super(), so call the shared helper.
        callbacks = self._configure_best_checkpoint(callbacks, results_dir)

        return callbacks

    def _parse_teacher_configs(self):
        """Parse teacher configurations from config.

        Returns:
            List of dicts, each containing:
                - 'model_config': ModelConfig for the teacher
                - 'loss_type': Loss type for this teacher (str)
                - 'loss_lambda': Weight for this teacher (float)
                - 'pretrained_path': Path to pretrained model (str)
                - 'mode': Distillation mode (str)
        """
        teacher_cfg = list(self.distill_config.teacher)

        # Check if teacher is a list (multiple teachers)
        if isinstance(teacher_cfg, (list, tuple)):
            teacher_list = teacher_cfg
        else:
            teacher_list = [teacher_cfg]
            assert 0, "Teacher is not a list"
        parsed_configs = []
        for idx, teacher in enumerate(teacher_list):
            config = {}

            # Check if this is a TeacherConfig (with model, loss_type, loss_lambda fields)
            # or a plain ModelConfig
            if hasattr(teacher, 'model'):
                # This is a TeacherConfig
                config['model_config'] = teacher.model
                config['loss_type'] = teacher.loss_type if teacher.loss_type is not None else self.distill_config.loss_type
                config['loss_lambda'] = teacher.loss_lambda if teacher.loss_lambda is not None else self.distill_config.loss_lambda
                teacher_pretrained = getattr(teacher, 'pretrained_teacher_model_path', None) or None
                model_pretrained = getattr(teacher.model.backbone, 'pretrained_backbone_path', None) or None
                config['pretrained_path'] = teacher_pretrained or model_pretrained
                config['mode'] = getattr(teacher, 'mode', self.distill_config.mode or 'auto')
            else:
                # This is a plain ModelConfig - use global settings
                config['model_config'] = teacher
                config['loss_type'] = self.distill_config.loss_type
                config['loss_lambda'] = self.distill_config.loss_lambda
                global_pretrained = getattr(self.distill_config, 'pretrained_teacher_model_path', None) or None
                model_pretrained = getattr(teacher.backbone, 'pretrained_backbone_path', None) or None
                config['pretrained_path'] = global_pretrained or model_pretrained
                config['mode'] = self.distill_config.mode or 'auto'

            # Multi-view teacher input and resolution settings.
            config['input_size'] = getattr(teacher, 'input_size', None)
            config['match_student_resolution'] = getattr(teacher, 'match_student_resolution', True)
            config['student_resolution'] = getattr(teacher, 'student_resolution', None)
            config['stochastic_resolutions'] = getattr(teacher, 'stochastic_resolutions', None)
            # Per-teacher image normalization (e.g. [0.5, 0.5, 0.5] for SAM3/SigLIP2; ImageNet for DINOv3)
            _nm = getattr(teacher, 'norm_mean', None)
            _ns = getattr(teacher, 'norm_std', None)
            config['norm_mean'] = list(_nm) if _nm and len(_nm) == 3 else None
            config['norm_std'] = list(_ns) if _ns and len(_ns) == 3 else None
            config['summary_loss_weight'] = getattr(teacher, 'summary_loss_weight', 1.0)
            config['fd_loss_weight'] = getattr(teacher, 'fd_loss_weight', 1.0)
            config['summary_loss_type'] = getattr(teacher, 'summary_loss_type', 'CE')
            config['spatial_loss_type'] = getattr(teacher, 'spatial_loss_type', 'mse')
            config['spatial_focal_weight'] = getattr(teacher, 'spatial_focal_weight', 0.0)
            config['spatial_focal_gamma'] = getattr(teacher, 'spatial_focal_gamma', 1.0)
            config['spatial_focal_max_weight'] = getattr(teacher, 'spatial_focal_max_weight', 4.0)
            config['intermediate_loss_weight'] = getattr(teacher, 'intermediate_loss_weight', 0.0)
            config['intermediate_loss_weights'] = list(getattr(teacher, 'intermediate_loss_weights', []) or [])
            config['intermediate_feature_dims'] = list(getattr(teacher, 'intermediate_feature_dims', []) or [])
            config['intermediate_focal_weight'] = getattr(teacher, 'intermediate_focal_weight', 0.0)
            config['intermediate_mlp_version'] = getattr(teacher, 'intermediate_mlp_version', 'residual')
            config['intermediate_num_inner'] = getattr(teacher, 'intermediate_num_inner', None)
            config['summary_token_idx'] = getattr(teacher, 'summary_token_idx', None)
            config['spatial_mlp_version'] = getattr(teacher, 'spatial_mlp_version', 'v2')
            config['spatial_num_inner'] = getattr(teacher, 'spatial_num_inner', None)
            config['summary_mlp_version'] = getattr(teacher, 'summary_mlp_version', None)
            config['summary_num_inner'] = getattr(teacher, 'summary_num_inner', None)
            config['spatial_norm_type'] = getattr(teacher, 'spatial_norm_type', 'phi')
            config['spatial_whiten_update_period'] = getattr(teacher, 'spatial_whiten_update_period', 100)
            config['spatial_whiten_freeze_after_steps'] = getattr(teacher, 'spatial_whiten_freeze_after_steps', 0)
            config['spatial_whiten_shrinkage'] = getattr(teacher, 'spatial_whiten_shrinkage', 0.0)
            config['spatial_whiten_eigen_floor'] = getattr(teacher, 'spatial_whiten_eigen_floor', 1.0e-6)
            config['spatial_whiten_max_gain'] = getattr(teacher, 'spatial_whiten_max_gain', 0.0)
            config['spatial_projector_residual_scale'] = getattr(teacher, 'spatial_projector_residual_scale', 0.25)
            config['spatial_projector_output_norm'] = getattr(teacher, 'spatial_projector_output_norm', False)
            config['upstream_name'] = None
            # FeatSharp adaptor
            config['adaptor'] = getattr(teacher, 'adaptor', None)
            config['upsampler_checkpoint'] = getattr(teacher, 'upsampler_checkpoint', None)
            config['do_upsample'] = getattr(teacher, 'do_upsample', True)
            config['featsharp_lib_path'] = getattr(teacher, 'featsharp_lib_path', None)
            config['shared_teacher_key'] = str(
                getattr(teacher, 'shared_teacher_key', '') or ''
            )
            config['rank_partition'] = int(
                getattr(teacher, 'rank_partition', -1)
            )
            config['local_batch_size'] = int(
                getattr(teacher, 'local_batch_size', 0)
            )
            config['mosaic_inner_size'] = int(
                getattr(teacher, 'mosaic_inner_size', 0) or 0
            )
            config['mosaic_outer_size'] = int(
                getattr(teacher, 'mosaic_outer_size', 0) or 0
            )
            config['mosaic_downsample'] = int(
                getattr(teacher, 'mosaic_downsample', 0) or 0
            )

            parsed_configs.append(config)
            logger.info(f"Teacher {idx}: loss_type={config['loss_type']}, "
                        f"loss_lambda={config['loss_lambda']}, mode={config['mode']}")

        return parsed_configs

    def _validate_partitioned_config(self):
        """Validate partitioned-distillation invariants before loading weights."""
        if not getattr(
            self.distill_config, "partitioned_ranks", False
        ):
            return

        num_partitions = int(
            getattr(self.distill_config, "num_rank_partitions", 4) or 4
        )
        partition_batches = {}
        teacher_keys = set()
        for idx, config in enumerate(self.teacher_configs):
            key = config.get("shared_teacher_key", "")
            partition = int(config.get("rank_partition", -1))
            local_batch = int(config.get("local_batch_size", 0))
            if not key:
                raise ValueError(
                    "partitioned_ranks requires shared_teacher_key on "
                    f"teacher entry {idx}"
                )
            if partition < 0 or partition >= num_partitions:
                raise ValueError(
                    "rank_partition must be in "
                    f"[0, {num_partitions - 1}], got {partition} on teacher {idx}"
                )
            if local_batch <= 0:
                raise ValueError(
                    "local_batch_size must be positive on "
                    f"teacher entry {idx}"
                )
            previous = partition_batches.setdefault(partition, local_batch)
            if previous != local_batch:
                raise ValueError(
                    "Teacher arms in one partition must use the "
                    f"same local batch; partition={partition}, "
                    f"values={previous},{local_batch}"
                )
            teacher_keys.add(key)
            _validate_mosaic_config(config)

        expected_partitions = list(range(num_partitions))
        if sorted(partition_batches) != expected_partitions:
            raise ValueError(
                "Partitioned training requires at least one teacher in every "
                f"partition; expected={expected_partitions}, "
                f"got={sorted(partition_batches)}"
            )
        if not teacher_keys:
            raise ValueError(
                "partitioned_ranks requires at least one teacher"
            )

        configured_world = (
            int(self.train_config.get("num_gpus", 1)) *
            int(self.train_config.get("num_nodes", 1))
        )
        if (
            configured_world < num_partitions or
            configured_world % num_partitions
        ):
            raise ValueError(
                "partitioned_ranks requires configured world size "
                "divisible by num_rank_partitions; "
                f"world={configured_world}, partitions={num_partitions}"
            )

        train_dataset = self.dataset_config["train_dataset"]
        if not train_dataset.get("tar_data_sources", []):
            raise ValueError(
                "partitioned_ranks requires "
                "train_dataset.tar_data_sources so rank-local views, "
                "batches, and deterministic shard partitioning are preserved"
            )

        checkpoint_unit = self.train_config.get(
            "checkpoint_interval_unit", "epoch"
        )
        if checkpoint_unit != "epoch":
            raise ValueError(
                "partitioned_ranks requires epoch checkpoints; "
                "mid-epoch step checkpoints precede canonical BN/PHI state "
                "synchronization"
            )

    @staticmethod
    def _normalize_input(
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize an image tensor from [0,1] using the given mean/std buffers."""
        return (x - mean) / std

    def _normalize_student_input(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize student input from [0,1] using the student model conditioner."""
        return self._normalize_input(img, self._student_mean, self._student_std)

    @staticmethod
    def _as_norm_buffer(value: torch.Tensor, name: str) -> torch.Tensor:
        """Convert a model-provided normalization value to a RADIO buffer."""
        norm = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if norm.numel() != 3:
            value_shape = getattr(value, "shape", None)
            raise ValueError(f"Student normalization {name} must have 3 values, got shape {value_shape}")
        return norm.view(1, 3, 1, 1)

    def _resolve_student_normalization(self):
        """Resolve model-owned student normalization, falling back to OpenAI CLIP."""
        mean = torch.tensor(OPENAI_CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(OPENAI_CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)

        model = getattr(self, "model", None)
        conditioner = getattr(model, "input_conditioner", None)
        if (
            conditioner is not None and
            hasattr(conditioner, "norm_mean") and
            hasattr(conditioner, "norm_std")
        ):
            mean = self._as_norm_buffer(conditioner.norm_mean, "input_conditioner.norm_mean")
            std = self._as_norm_buffer(conditioner.norm_std, "input_conditioner.norm_std")
            if get_global_rank() == 0:
                logger.info("Using student input_conditioner normalization.")
            return mean, std

        if model is not None and hasattr(model, "student_norm_mean") and hasattr(model, "student_norm_std"):
            mean = self._as_norm_buffer(model.student_norm_mean, "student_norm_mean")
            std = self._as_norm_buffer(model.student_norm_std, "student_norm_std")
            if get_global_rank() == 0:
                logger.info("Using student model normalization buffers.")
            return mean, std

        if get_global_rank() == 0:
            logger.info("Using default OpenAI CLIP student normalization.")
        return mean, std

    @staticmethod
    def _is_norm_module(module: nn.Module) -> bool:
        """Return True for normalization modules whose state should stay fixed."""
        norm_types = [
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
            nn.GroupNorm,
        ]
        rms_norm = getattr(nn, "RMSNorm", None)
        if rms_norm is not None:
            norm_types.append(rms_norm)
        return isinstance(module, tuple(norm_types))

    def _enforce_frozen_student_norms(self):
        """Freeze student normalization affine parameters and running stats."""
        if not getattr(self.distill_config, "freeze_student_norms", False):
            return

        frozen_modules = 0
        frozen_params = 0
        for module in self.model.modules():
            if not self._is_norm_module(module):
                continue
            frozen_modules += 1
            module.eval()
            for param in module.parameters(recurse=False):
                if param.requires_grad:
                    frozen_params += param.numel()
                param.requires_grad = False

        if get_global_rank() == 0 and not self._student_norm_freeze_logged:
            logger.info(
                "Frozen student normalization layers: modules=%s, params=%s",
                frozen_modules,
                frozen_params,
            )
            self._student_norm_freeze_logged = True

    def _baseline_anchor_enabled(self) -> bool:
        """Whether any frozen-baseline anchor loss is configured."""
        weights = (
            "baseline_anchor_spatial_loss_weight",
            "baseline_anchor_summary_loss_weight",
            "baseline_anchor_intermediate_loss_weight",
        )
        return any(float(getattr(self.distill_config, name, 0.0) or 0.0) > 0.0 for name in weights)

    def _setup_baseline_anchor_model(self):
        """Build the frozen starting-student anchor used for preservation losses."""
        if not self._baseline_anchor_enabled():
            return

        anchor_path = getattr(self.distill_config, "baseline_anchor_model_path", None) or ""
        if anchor_path:
            anchor_spec = copy.deepcopy(self.experiment_spec)
            anchor_spec.model.backbone.pretrained_backbone_path = anchor_path
            anchor_model = build_model(experiment_config=anchor_spec, export=False)
            anchor_source = anchor_path
        else:
            anchor_model = copy.deepcopy(self.model)
            anchor_source = "initial student copy"

        anchor_model.eval()
        for param in anchor_model.parameters():
            param.requires_grad = False
        self._baseline_anchor_model = anchor_model
        self._register_baseline_anchor_hooks()

        if get_global_rank() == 0:
            logger.info("Using frozen baseline student anchor from %s.", anchor_source)

    @staticmethod
    def _extract_first_tensor(value):
        """Extract a tensor from nested module outputs for anchor losses."""
        if torch.is_tensor(value):
            return value
        if isinstance(value, dict):
            for item in value.values():
                tensor = MultiTeacherDistiller._extract_first_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, (list, tuple)):
            for item in value:
                tensor = MultiTeacherDistiller._extract_first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    @staticmethod
    def _make_anchor_feature_hook(cache: dict, name: str):
        """Create a forward hook that caches the first tensor-like output."""
        def _hook(_module, _inputs, output):
            tensor = MultiTeacherDistiller._extract_first_tensor(output)
            if tensor is not None:
                cache[name] = tensor
        return _hook

    def _register_baseline_anchor_hooks(self):
        """Register current/anchor hooks for configured intermediate layers."""
        layers = list(getattr(self.distill_config, "baseline_anchor_intermediate_layers", []) or [])
        weight = float(getattr(self.distill_config, "baseline_anchor_intermediate_loss_weight", 0.0) or 0.0)
        if not layers or weight <= 0.0:
            return

        student_modules = dict(self.model.named_modules())
        anchor_modules = dict(self._baseline_anchor_model.named_modules())
        missing = [
            name for name in layers
            if name not in student_modules or name not in anchor_modules
        ]
        if missing:
            raise ValueError(
                "baseline_anchor_intermediate_layers contains unknown modules: "
                f"{missing}. Available student modules include: {list(student_modules)[:20]}..."
            )

        for name in layers:
            self._baseline_anchor_hooks.append(
                student_modules[name].register_forward_hook(
                    self._make_anchor_feature_hook(self._baseline_anchor_student_features, name)
                )
            )
            self._baseline_anchor_hooks.append(
                anchor_modules[name].register_forward_hook(
                    self._make_anchor_feature_hook(self._baseline_anchor_anchor_features, name)
                )
            )

    def _reset_baseline_anchor_cache(self):
        """Clear cached intermediate features before a paired student/anchor forward."""
        self._baseline_anchor_student_features.clear()
        self._baseline_anchor_anchor_features.clear()

    @staticmethod
    def _align_anchor_tensors(student: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Align feature tensors for baseline-anchor comparison."""
        student = student.float()
        target = target.detach().float()
        if student.shape == target.shape:
            return student, target

        if student.ndim == 4 and target.ndim == 4 and student.shape[:2] == target.shape[:2]:
            student = F.interpolate(
                student,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if student.shape != target.shape:
            raise ValueError(
                "Baseline anchor tensors have incompatible shapes: "
                f"student={list(student.shape)}, target={list(target.shape)}"
            )
        return student, target

    def _baseline_anchor_pair_loss(self, student: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the configured preservation loss between student and anchor tensors."""
        student, target = self._align_anchor_tensors(student, target)
        loss_type = (getattr(self.distill_config, "baseline_anchor_loss_type", "normalized_mse") or "normalized_mse").lower()
        if loss_type == "mse":
            return F.mse_loss(student, target)

        if student.ndim < 2:
            return F.mse_loss(student, target)

        feature_dim = 1 if student.ndim == 4 else -1
        student_norm = F.normalize(student, dim=feature_dim, eps=1e-8)
        target_norm = F.normalize(target, dim=feature_dim, eps=1e-8)
        if loss_type == "cosine":
            return 1.0 - F.cosine_similarity(student_norm, target_norm, dim=feature_dim, eps=1e-8).mean()
        if loss_type == "normalized_mse":
            return F.mse_loss(student_norm, target_norm)
        raise ValueError(
            "baseline_anchor_loss_type must be one of normalized_mse, mse, cosine; "
            f"got {loss_type!r}"
        )

    def _compute_baseline_anchor_loss(
        self,
        student_summary: torch.Tensor,
        student_spatial: torch.Tensor,
        anchor_summary: torch.Tensor,
        anchor_spatial: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute frozen-baseline preservation loss and component logs."""
        zero = student_summary.float().sum() * 0.0
        if self._baseline_anchor_model is None:
            return zero, {}

        logs = {}
        total = zero

        spatial_weight = float(getattr(self.distill_config, "baseline_anchor_spatial_loss_weight", 0.0) or 0.0)
        if spatial_weight > 0.0:
            student_tensor = self._extract_first_tensor(student_spatial)
            anchor_tensor = self._extract_first_tensor(anchor_spatial)
            if student_tensor is None or anchor_tensor is None:
                raise ValueError("Could not extract spatial tensors for baseline anchor loss.")
            spatial_loss = self._baseline_anchor_pair_loss(student_tensor, anchor_tensor)
            logs["baseline_anchor_spatial_loss"] = spatial_loss.detach()
            total = total + spatial_weight * spatial_loss

        summary_weight = float(getattr(self.distill_config, "baseline_anchor_summary_loss_weight", 0.0) or 0.0)
        if summary_weight > 0.0:
            summary_loss = self._baseline_anchor_pair_loss(student_summary, anchor_summary)
            logs["baseline_anchor_summary_loss"] = summary_loss.detach()
            total = total + summary_weight * summary_loss

        intermediate_weight = float(getattr(self.distill_config, "baseline_anchor_intermediate_loss_weight", 0.0) or 0.0)
        layers = list(getattr(self.distill_config, "baseline_anchor_intermediate_layers", []) or [])
        if intermediate_weight > 0.0 and layers:
            layer_losses = []
            for name in layers:
                student_tensor = self._baseline_anchor_student_features.get(name)
                anchor_tensor = self._baseline_anchor_anchor_features.get(name)
                if student_tensor is None or anchor_tensor is None:
                    raise ValueError(f"Missing baseline anchor intermediate feature for layer {name!r}.")
                layer_losses.append(self._baseline_anchor_pair_loss(student_tensor, anchor_tensor))
            intermediate_loss = torch.stack(layer_losses).mean()
            logs["baseline_anchor_intermediate_loss"] = intermediate_loss.detach()
            total = total + intermediate_weight * intermediate_loss

        logs["baseline_anchor_loss"] = total.detach()
        return total, logs

    def _apply_teacher_normalization(
        self, teacher_input: torch.Tensor, teacher_config: dict, device: torch.device
    ) -> torch.Tensor:
        """Normalize teacher input from [0,1] to per-teacher normalization.

        All dataloaders now output [0,1] images, so normalization is always
        ``(x - mean_t) / std_t``.
        """
        norm_mean = teacher_config.get("norm_mean")
        norm_std = teacher_config.get("norm_std")
        if not norm_mean or not norm_std:
            logger.warning(
                "Teacher has no norm_mean/norm_std configured -- "
                "passing raw [0,1] input which is likely incorrect. "
                "Add norm_mean/norm_std to the teacher config."
            )
            return teacher_input

        mean_t = torch.tensor(norm_mean, dtype=teacher_input.dtype, device=device).view(1, 3, 1, 1)
        std_t = torch.tensor(norm_std, dtype=teacher_input.dtype, device=device).view(1, 3, 1, 1)
        return self._normalize_input(teacher_input, mean_t, std_t)

    def _setup_bindings(self):
        """Setup bindings to be captured during training for distillation."""
        pass

    def _build_model(self, export=False):
        """Internal function to build the model."""
        # Build multiple teacher models
        self.teachers = nn.ModuleList()

        for idx, teacher_config in enumerate(self.teacher_configs):
            # Build the teacher config
            teacher_cfg = copy.deepcopy(self.experiment_spec)
            teacher_cfg.model = teacher_config['model_config']

            # Set pretrained path if available
            if teacher_config['pretrained_path']:
                teacher_cfg.model.backbone.pretrained_backbone_path = teacher_config['pretrained_path']

            # Build the teacher model
            teacher_model = build_model(experiment_config=teacher_cfg, export=export)
            teacher_model.eval()

            # Freeze teacher
            for _, param in teacher_model.named_parameters():
                param.requires_grad = False

            for module in teacher_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                if isinstance(module, nn.LayerNorm):
                    module.eval()
                if isinstance(module, nn.Dropout):
                    module.eval()

            # Apply FeatSharp adaptor if configured
            adaptor = teacher_config.get('adaptor')
            if adaptor == 'featsharp':
                ckpt = teacher_config.get('upsampler_checkpoint')
                if not ckpt:
                    raise ValueError(
                        f"Teacher {idx} has adaptor='featsharp' but no "
                        f"'upsampler_checkpoint' path specified."
                    )
                t_input_size = teacher_config.get('input_size') or 224
                teacher_model = wrap_teacher_with_featsharp(
                    teacher_model,
                    checkpoint_path=ckpt,
                    input_size=t_input_size,
                    do_upsample=teacher_config.get('do_upsample', True),
                    featsharp_lib_path=teacher_config.get('featsharp_lib_path'),
                )
            elif adaptor is not None:
                raise ValueError(f"Unknown adaptor type '{adaptor}' for teacher {idx}")

            # Cast frozen teachers to bf16 to halve their resident weight memory. Needed to fit
            # heavy high-res teacher sets (e.g. DINOv3-7B ~28GB fp32 -> ~14GB bf16) in 80GB under
            # per-rank partitioning; teachers are inference-only targets so bf16 linear/matmul
            # weights are fine. Applied after the FeatSharp wrap so the whole teacher
            # (backbone + upsampler) shares one dtype and avoids featurizer<->upsampler mismatches.
            #
            # CRITICAL: normalization layers are kept in fp32. A blanket .to(bfloat16) also casts
            # every LayerNorm/RMSNorm weight to bf16, and bf16 normalization badly degrades deep
            # transformer features (accumulated over a deep teacher's many blocks) -- this produced
            # collapsed distillation targets and a downstream detection-collapse.
            # Restoring norms to fp32 preserves the sensitive
            # normalization statistics while keeping the bulk weight-memory savings; autocast
            # handles the resulting mixed-dtype forward.
            if getattr(self.distill_config, "teacher_bf16", False):
                # teacher_bf16 keeps only bf16 teacher *weights* (norms stay fp32) and
                # relies on autocast to reconcile the mixed-dtype forward. Pure fp32
                # training ('fp32' -> '32-true') runs with no autocast, so fp32 student
                # inputs would hit bf16 teacher linears and raise a dtype mismatch.
                # Fail fast with an actionable message instead of a cryptic runtime error.
                self._validate_teacher_bf16_precision(
                    self.experiment_spec["train"].get("precision", "fp32")
                )
                teacher_model = teacher_model.to(torch.bfloat16)
                for _mod in teacher_model.modules():
                    _is_norm = isinstance(
                        _mod, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)
                    ) or "norm" in type(_mod).__name__.lower() or "rms" in type(_mod).__name__.lower()
                    if _is_norm:
                        _mod.float()

            self.teachers.append(teacher_model)
            logger.info(f"Built teacher model {idx}: {teacher_cfg.model.backbone.type}")

        # Build the student model
        self.model = build_model(experiment_config=self.experiment_spec, export=export)
        self.model.train()

        # Apply ViTDet windowed-attention augmentation to the student.
        vitdet_cfg = getattr(self.distill_config, 'vitdet', None)
        if vitdet_cfg is not None:
            vitdet_args = VitDetArgs(
                prob=getattr(vitdet_cfg, 'prob', 0.0),
                window_sizes=list(getattr(vitdet_cfg, 'window_sizes', [])),
                num_global=getattr(vitdet_cfg, 'num_global', None),
                num_windowed=getattr(vitdet_cfg, 'num_windowed', None),
            )
        else:
            vitdet_args = VitDetArgs()
        self._vitdet_hook = apply_vitdet_to_vit(self.model, vitdet_args)

        # For backward compatibility, keep single teacher reference if only one teacher
        if len(self.teachers) == 1:
            self.teacher = self.teachers[0]

    def _build_criterion(self):
        """Build distillation loss modules, one per teacher."""
        self.distillation_loss_fns = nn.ModuleList()

        for idx, (teacher_model, teacher_config) in enumerate(zip(self.teachers, self.teacher_configs)):
            loss_fn = DistillationLoss(
                loss_type=teacher_config['loss_type'],
                student_model=self.model,
                teacher_model=teacher_model,
                distillation_mode=teacher_config['mode'],
                num_classes=self.num_classes,
                temperature=getattr(self.distill_config, 'temperature', 1.0),
                use_mlp=getattr(self.distill_config, 'use_mlp', True),
                mlp_hidden_size=getattr(self.distill_config, 'mlp_hidden_size', 1024),
                mlp_num_inner=getattr(self.distill_config, 'mlp_num_inner', 0),
                spatial_mlp_version=teacher_config.get('spatial_mlp_version', 'v2'),
                spatial_num_inner=teacher_config.get('spatial_num_inner', None),
                summary_mlp_version=teacher_config.get('summary_mlp_version', None),
                summary_num_inner=teacher_config.get('summary_num_inner', None),
                summary_loss_weight=teacher_config.get('summary_loss_weight', 1.0),
                fd_loss_weight=teacher_config.get('fd_loss_weight', 1.0),
                summary_loss_type=teacher_config.get('summary_loss_type', 'CE'),
                spatial_loss_type=teacher_config.get('spatial_loss_type', 'mse'),
                spatial_focal_weight=teacher_config.get('spatial_focal_weight', 0.0),
                spatial_focal_gamma=teacher_config.get('spatial_focal_gamma', 1.0),
                spatial_focal_max_weight=teacher_config.get('spatial_focal_max_weight', 4.0),
                intermediate_loss_weight=teacher_config.get('intermediate_loss_weight', 0.0),
                intermediate_loss_weights=teacher_config.get('intermediate_loss_weights', []),
                intermediate_feature_dims=teacher_config.get('intermediate_feature_dims', []),
                intermediate_focal_weight=teacher_config.get('intermediate_focal_weight', 0.0),
                intermediate_mlp_version=teacher_config.get('intermediate_mlp_version', 'residual'),
                intermediate_num_inner=teacher_config.get('intermediate_num_inner', None),
                spatial_norm_type=teacher_config.get('spatial_norm_type', 'phi'),
                spatial_whiten_update_period=teacher_config.get('spatial_whiten_update_period', 100),
                spatial_whiten_freeze_after_steps=teacher_config.get('spatial_whiten_freeze_after_steps', 0),
                spatial_whiten_shrinkage=teacher_config.get('spatial_whiten_shrinkage', 0.0),
                spatial_whiten_eigen_floor=teacher_config.get('spatial_whiten_eigen_floor', 1.0e-6),
                spatial_whiten_max_gain=teacher_config.get('spatial_whiten_max_gain', 0.0),
                spatial_projector_residual_scale=teacher_config.get('spatial_projector_residual_scale', 0.25),
                spatial_projector_output_norm=teacher_config.get('spatial_projector_output_norm', False),
                summary_token_idx=teacher_config.get('summary_token_idx'),
                partitioned_ranks=getattr(
                    self.distill_config, 'partitioned_ranks', False
                ),
                mosaic_inner_size=teacher_config.get('mosaic_inner_size', 0),
                mosaic_outer_size=teacher_config.get('mosaic_outer_size', 0),
                mosaic_downsample=teacher_config.get('mosaic_downsample', 0),
            )
            self.distillation_loss_fns.append(loss_fn)
            logger.info(f"Created distillation loss for teacher {idx}: "
                        f"type={teacher_config['loss_type']}, mode={teacher_config['mode']}")

        if getattr(self.distill_config, "partitioned_ranks", False):
            self._share_partitioned_teacher_state()

        # For backward compatibility, keep single loss reference if only one teacher
        if len(self.distillation_loss_fns) == 1:
            self.distillation_loss_fn = self.distillation_loss_fns[0]

    def _share_partitioned_teacher_state(self):
        """Alias low/high projection and normalization state by teacher identity."""
        canonical = {}
        shared_attrs = (
            "projection_layer",
            "projection_layer_summary",
            "intermediate_projection_layers",
            "phi_norm",
            "summary_criterion",
        )
        for idx, (loss_fn, teacher_config) in enumerate(
            zip(self.distillation_loss_fns, self.teacher_configs)
        ):
            key = teacher_config.get("shared_teacher_key", "")
            if not key:
                raise ValueError(
                    "partitioned_ranks requires shared_teacher_key on "
                    f"teacher entry {idx}"
                )
            if key not in canonical:
                canonical[key] = loss_fn
                continue
            source = canonical[key]
            for attr in shared_attrs:
                source_module = getattr(source, attr, None)
                target_module = getattr(loss_fn, attr, None)
                if (source_module is None) != (target_module is None):
                    raise ValueError(
                        f"Low/high teacher {key!r} disagree on {attr} presence"
                    )
                if source_module is not None:
                    setattr(loss_fn, attr, source_module)
        if get_global_rank() == 0:
            logger.info(
                "Partitioned distillation: %d teacher identities share "
                "projection heads and PHI state: %s",
                len(canonical),
                sorted(canonical),
            )

    def _detect_upstream_head_info(self, ckpt_path):
        """Inspect a RADIO checkpoint to pick and warm-start per-teacher heads.

        Args:
            ckpt_path (str): Path to the upstream RADIO checkpoint whose
                adapter heads should seed the local per-teacher loss heads.

        Returns:
            None: The method updates ``teacher_configs`` and caches matching
                checkpoint tensors on ``self._pretrained_head_sd``.
        """
        up = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        up_sd = up['state_dict'] if isinstance(up, dict) and 'state_dict' in up else up
        up_args = up.get('args') if isinstance(up, dict) else None
        token_slot_by_name = self._extract_upstream_token_slots(up_args)
        head_keys = [k for k in up_sd if k.startswith('_heads.') or k.startswith('_feature_projections.')]
        if not head_keys:
            # tao-native stage-1 checkpoints store trained heads under
            # distillation_loss_fns.{N}.projection_layer[_summary].* instead of the
            # RADIO _heads./_feature_projections. layout; fall back to that.
            self._detect_native_head_info(up, up_sd)
            return

        upstream_names = sorted({k.split('.')[1] for k in head_keys if k.startswith('_heads.')})
        if get_global_rank() == 0:
            logger.info(f"[head-detect] upstream adapters in ckpt: {upstream_names}")

        self._pretrained_head_sd = {k: up_sd[k] for k in head_keys}
        del up, up_sd

        def _norm(s):
            return ''.join(c for c in s.lower() if c.isalnum())

        def _common_prefix(a, b):
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            return n

        for idx, teacher_config in enumerate(self.teacher_configs):
            t_type = str(teacher_config['model_config'].backbone.type)
            t_norm = _norm(t_type)
            best_name, best_score = None, 0
            for name in upstream_names:
                score = _common_prefix(t_norm, _norm(name))
                if score > best_score:
                    best_name, best_score = name, score
            if best_name is None or best_score < 3:
                if get_global_rank() == 0:
                    logger.warning(
                        f"[head-detect] teacher {idx} (type={t_type}): no upstream adapter name "
                        f"matched from {upstream_names}; skipping warm-start for this teacher"
                    )
                continue

            teacher_config['upstream_name'] = best_name
            summary_token_idx = self._lookup_upstream_token_slot(best_name, token_slot_by_name)
            if summary_token_idx is None:
                summary_token_idx = idx
                if get_global_rank() == 0:
                    logger.warning(
                        f"[head-detect] teacher {idx} (type={t_type}) matched upstream '{best_name}' "
                        f"but checkpoint args did not expose token_slot; falling back to slot {summary_token_idx}"
                    )
            teacher_config['summary_token_idx'] = summary_token_idx

            fp_keys = [k for k in self._pretrained_head_sd if k.startswith(f'_feature_projections.{best_name}.')]
            if any('blocks.' in k and ('.attn.' in k or '.norm1.' in k) for k in fp_keys):
                teacher_config['spatial_mlp_version'] = 'attn'
                teacher_config['spatial_num_inner'] = 0
            else:
                teacher_config['spatial_mlp_version'] = 'v2'
                num_inner = 0
                while any(k == f'_feature_projections.{best_name}.blocks.{num_inner}.0.weight' for k in fp_keys):
                    num_inner += 1
                if num_inner > 0:
                    teacher_config['spatial_num_inner'] = num_inner

            if get_global_rank() == 0:
                logger.info(
                    f"[head-detect] teacher {idx} (type={t_type}) -> upstream '{best_name}', "
                    f"spatial_mlp_version={teacher_config['spatial_mlp_version']}, "
                    f"spatial_num_inner={teacher_config['spatial_num_inner']}, "
                    f"summary_token_idx={teacher_config['summary_token_idx']}"
                )

    @staticmethod
    def _cfg_get(cfg, name, default=None):
        """Read a value from a dict-like or attribute-like config object.

        Args:
            cfg (Any): Checkpoint config represented as a dictionary or an
                object with attributes.
            name (str): Field name to read from the config.
            default (Any): Value returned when ``name`` is not present.

        Returns:
            Any: The requested value, or ``default`` when unavailable.
        """
        if isinstance(cfg, dict):
            return cfg.get(name, default)
        return getattr(cfg, name, default)

    @classmethod
    def _extract_upstream_token_slots(cls, args):
        """Extract summary-token slots from upstream RADIO checkpoint args.

        Args:
            args (Any): Checkpoint ``args`` metadata containing teacher
                entries and optional ``cls_token_per_teacher`` settings.

        Returns:
            dict: Mapping from upstream teacher name to summary-token slot.
        """
        if args is None:
            return {}

        teachers = cls._cfg_get(args, 'teachers', None)
        if teachers is None:
            return {}

        cls_token_per_teacher = cls._cfg_get(args, 'cls_token_per_teacher', True)
        token_slot_by_name = {}
        for tidx, teacher_cfg in enumerate(teachers):
            name = cls._cfg_get(teacher_cfg, 'name', None)
            if name is None:
                continue
            token_slot = 0
            if cls_token_per_teacher:
                token_slot = cls._cfg_get(teacher_cfg, 'token_slot', tidx)
            token_slot_by_name[str(name)] = int(token_slot)
        return token_slot_by_name

    @staticmethod
    def _lookup_upstream_token_slot(upstream_name, token_slot_by_name):
        """Lookup a token slot by exact or normalized upstream adapter name.

        Args:
            upstream_name (str): Adapter name detected in the upstream
                checkpoint state dict.
            token_slot_by_name (dict): Mapping from upstream teacher name to
                summary-token slot.

        Returns:
            Optional[int]: Matching summary-token slot, or ``None`` when the
                checkpoint metadata does not expose one.
        """
        if upstream_name in token_slot_by_name:
            return token_slot_by_name[upstream_name]

        def _norm(s):
            return ''.join(c for c in str(s).lower() if c.isalnum())

        target = _norm(upstream_name)
        for name, token_slot in token_slot_by_name.items():
            if _norm(name) == target:
                return token_slot
        return None

    def _extract_spec_teacher_types(self, spec):
        """Read per-teacher backbone types (in order) from a stored experiment spec.

        Args:
            spec (Any): The ``radio_experiment_spec`` object saved in a tao
                checkpoint (dict-like or attribute-like / OmegaConf).

        Returns:
            list: Backbone type strings, one per teacher, in teacher order.
        """
        if spec is None:
            return []
        distill = self._cfg_get(spec, 'distill', None)
        teachers = self._cfg_get(distill, 'teacher', None) if distill is not None else None
        if not teachers:
            return []
        types = []
        for teacher in teachers:
            model = self._cfg_get(teacher, 'model', None)
            backbone = self._cfg_get(model, 'backbone', None) if model is not None else None
            btype = self._cfg_get(backbone, 'type', None) if backbone is not None else None
            types.append(str(btype) if btype is not None else '')
        return types

    def _detect_native_head_info(self, up, up_sd):
        """Warm-start heads from a tao-native stage-1 checkpoint.

        Upstream RADIO checkpoints expose heads under ``_heads.`` /
        ``_feature_projections.``; a tao stage-1 checkpoint instead stores the
        trained per-teacher heads under
        ``distillation_loss_fns.{N}.projection_layer[_summary].*``. This maps each
        local (e.g. stage-2) teacher to the stage-1 teacher of the SAME backbone
        type (so a 3-teacher stage-1 seeds both the low- and high-arm copies of a
        6-teacher stage-2) and caches the tensors for ``_warmstart_native_heads``.

        Args:
            up (Any): The full loaded checkpoint object.
            up_sd (dict): The checkpoint ``state_dict``.

        Returns:
            None: Sets ``self._native_head_sd`` and per-teacher ``native_src_idx``.
        """
        native_keys = [k for k in up_sd if k.startswith('distillation_loss_fns.')]
        if not native_keys:
            return
        src_spec = up.get('radio_experiment_spec') if isinstance(up, dict) else None
        src_types = self._extract_spec_teacher_types(src_spec)
        if not src_types:
            if get_global_rank() == 0:
                logger.warning(
                    "[head-detect] native checkpoint has distillation_loss_fns heads but no "
                    "radio_experiment_spec teacher types to map by; skipping head warm-start"
                )
            return

        def _norm(s):
            return ''.join(c for c in str(s).lower() if c.isalnum())

        type_to_src_idx = {}
        for src_idx, src_type in enumerate(src_types):
            type_to_src_idx.setdefault(_norm(src_type), src_idx)

        self._native_head_sd = {k: up_sd[k] for k in native_keys}
        matched = 0
        for idx, teacher_config in enumerate(self.teacher_configs):
            t_type = str(teacher_config['model_config'].backbone.type)
            src_idx = type_to_src_idx.get(_norm(t_type))
            if src_idx is None:
                if get_global_rank() == 0:
                    logger.warning(
                        f"[head-detect] native: local teacher {idx} (type={t_type}) has no "
                        f"stage-1 teacher of that type in {src_types}; skipping warm-start for it"
                    )
                continue
            teacher_config['native_src_idx'] = src_idx
            matched += 1
        if get_global_rank() == 0:
            logger.info(
                f"[head-detect] native tao stage-1 heads: src teachers {src_types}; "
                f"mapped {matched}/{len(self.teacher_configs)} local teachers by backbone type"
            )

    def _warmstart_native_heads(self):
        """Copy trained stage-1 heads (distillation_loss_fns.{src}.*) into local heads.

        Normalizes spectral-reparam keys: the checkpoint stores wrapped weights as
        ``fc.parametrizations.weight.original`` (+ a ``.0.scale``), but at warm-start
        time (before ``_configure_projection_head_spectral_norm``) the live modules
        hold raw ``fc.weight``. We map ``.parametrizations.weight.original`` -> raw
        ``.weight`` and drop the learnable spectral ``.scale`` (it re-inits when the
        parametrization is applied afterward).

        Returns:
            None: Matching tensors are loaded into each local head in place.
        """
        if not self._native_head_sd:
            return

        def _normalize(sub_key):
            return sub_key.replace('.parametrizations.weight.original', '.weight')

        for idx, (loss_fn, teacher_config) in enumerate(
            zip(self.distillation_loss_fns, self.teacher_configs)
        ):
            src_idx = teacher_config.get('native_src_idx')
            if src_idx is None:
                continue
            for sub_attr in ('projection_layer', 'projection_layer_summary'):
                dst_module = getattr(loss_fn, sub_attr, None)
                if dst_module is None:
                    continue
                src_prefix = f'distillation_loss_fns.{src_idx}.{sub_attr}.'
                src_sd = {}
                for k, v in self._native_head_sd.items():
                    if not k.startswith(src_prefix):
                        continue
                    sub_k = k[len(src_prefix):]
                    if '.parametrizations.weight.' in sub_k and not sub_k.endswith('.original'):
                        continue  # drop learnable spectral scale (re-inits on parametrize)
                    src_sd[_normalize(sub_k)] = v
                if not src_sd:
                    continue
                dst_sd = dst_module.state_dict()
                loaded, skipped_shape, missing = 0, [], []
                for k, v in src_sd.items():
                    if k not in dst_sd:
                        missing.append(k)
                    elif dst_sd[k].shape != v.shape:
                        skipped_shape.append(f"{k} (src {list(v.shape)} vs local {list(dst_sd[k].shape)})")
                    else:
                        dst_sd[k] = v
                        loaded += 1
                # Guard against a SILENT no-op: a type-matched teacher whose source
                # tensors load NOTHING (or a partially-mapped summary head) means the
                # stage-2 head geometry drifted from stage-1 (mlp_version / num_inner /
                # hidden / dims). Fail loudly rather than warm-starting nothing.
                if loaded == 0:
                    raise ValueError(
                        f"[warmstart-native] teacher {idx} <- stage-1 teacher {src_idx} "
                        f"{sub_attr}: 0/{len(src_sd)} tensors loaded (all unmapped or "
                        f"shape-mismatched) -- stage-2 head geometry likely differs from "
                        f"stage-1. unmapped={missing[:3]}, shape-skipped={skipped_shape[:3]}"
                    )
                if sub_attr == 'projection_layer_summary' and (skipped_shape or missing):
                    raise ValueError(
                        f"[warmstart-native] teacher {idx} <- stage-1 teacher {src_idx} "
                        f"summary head did not fully map: loaded {loaded}/{len(src_sd)}, "
                        f"shape-skipped={skipped_shape[:3]}, unmapped={missing[:3]}"
                    )
                dst_module.load_state_dict(dst_sd, strict=False)
                if get_global_rank() == 0:
                    message = (
                        f"[warmstart-native] teacher {idx} <- stage-1 teacher {src_idx} "
                        f"{sub_attr}: loaded {loaded}/{len(src_sd)}"
                    )
                    if skipped_shape:
                        message += f", shape-skipped: {skipped_shape[:3]}"
                    if missing:
                        message += f", unmapped: {missing[:3]}"
                    logger.info(message)

    def _warmstart_projection_heads(self, head_sd):
        """Copy upstream projection-head weights into per-teacher loss heads.

        Args:
            head_sd (dict): Cached checkpoint tensors for ``_heads`` and
                ``_feature_projections`` from the upstream RADIO checkpoint.

        Returns:
            None: Matching tensors are loaded into each distillation loss head
                in place.
        """
        for idx, (loss_fn, teacher_config) in enumerate(zip(self.distillation_loss_fns, self.teacher_configs)):
            name = teacher_config.get('upstream_name')
            if not name:
                continue

            for src_prefix, dst_attr in (
                (f'_heads.{name}.', 'projection_layer_summary'),
                (f'_feature_projections.{name}.', 'projection_layer'),
            ):
                dst_module = getattr(loss_fn, dst_attr, None)
                if dst_module is None:
                    continue
                src_sd = {k[len(src_prefix):]: v for k, v in head_sd.items() if k.startswith(src_prefix)}
                if not src_sd:
                    continue

                dst_sd = dst_module.state_dict()
                loaded, skipped_shape, missing = 0, [], []
                for k, v in src_sd.items():
                    if k not in dst_sd:
                        missing.append(k)
                    elif dst_sd[k].shape != v.shape:
                        skipped_shape.append(f"{k} (upstream {list(v.shape)} vs local {list(dst_sd[k].shape)})")
                    else:
                        dst_sd[k] = v
                        loaded += 1
                if dst_attr == 'projection_layer_summary' and (skipped_shape or missing):
                    raise ValueError(
                        f"[warmstart] teacher {idx} ({name}) summary head did not fully map: "
                        f"loaded {loaded}/{len(src_sd)}, "
                        f"shape-skipped={skipped_shape[:3]}, unmapped={missing[:3]}. "
                        "This usually means the RADIO summary token dimension does not match "
                        "the upstream per-teacher head."
                    )
                dst_module.load_state_dict(dst_sd, strict=False)
                if get_global_rank() == 0:
                    message = f"[warmstart] teacher {idx} ({name}) {dst_attr}: loaded {loaded}/{len(src_sd)}"
                    if skipped_shape:
                        message += f", shape-skipped: {skipped_shape[:3]}{'...' if len(skipped_shape) > 3 else ''}"
                    if missing:
                        message += f", unmapped src keys: {missing[:3]}{'...' if len(missing) > 3 else ''}"
                    logger.info(
                        message
                    )

    def _iter_projection_head_named_parameters(self):
        for prefix, module in self._iter_projection_head_modules():
            for name, param in module.named_parameters():
                yield f"{prefix}.{name}", param

    def _freeze_distillation_statistics_if_configured(self):
        """Keep checkpoint-calibrated teacher statistics fixed during continuation."""
        if not getattr(self.distill_config, "freeze_distillation_statistics", False):
            return

        frozen = []
        for loss_idx, loss_fn in enumerate(self.distillation_loss_fns):
            for name in ("phi_norm", "summary_criterion"):
                state = getattr(loss_fn, name, None)
                if state is None or not hasattr(state, "freeze_updates"):
                    continue
                state.freeze_updates = True
                frozen.append(f"distillation_loss_fns.{loss_idx}.{name}")
        if not frozen:
            raise ValueError(
                "freeze_distillation_statistics=true but no compatible distillation state was found"
            )
        if get_global_rank() == 0:
            logger.info("Frozen checkpoint-calibrated distillation statistics: %s", frozen)

    def _set_projection_head_trainable(self, trainable: bool):
        for _, module in self._iter_projection_head_modules():
            module.train(trainable)
            for param in module.parameters():
                param.requires_grad = trainable

    def _prefreeze_projection_heads_if_configured(self):
        """Freeze heads before DDP and optimizer construction when warmup is disabled."""
        should_freeze = (
            getattr(self.distill_config, "train_projection_heads", False) and
            getattr(self.distill_config, "freeze_projection_heads_after_warmup", False) and
            int(getattr(self.distill_config, "head_warmup_epochs", 0) or 0) == 0
        )
        if not should_freeze:
            return
        self._set_projection_head_trainable(False)
        if get_global_rank() == 0:
            logger.info("Projection heads frozen before DDP and optimizer construction.")

    def _warmstart_training_weights(self):
        """Restore trainable feature state without restoring trainer state."""
        checkpoint_path = getattr(self.train_config, "warmstart_training_checkpoint_path", None) or ""
        if not checkpoint_path:
            return
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Weights-only warm-start checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint has no state dict: {checkpoint_path}")

        allowed_prefixes = (
            "model.",
            "distillation_loss_fns.",
            "distillation_loss_fn.",
        )
        warmstart_state = {
            key: value
            for key, value in state_dict.items()
            if (
                key.startswith(allowed_prefixes) and
                ".teacher_model." not in key
            )
        }
        model_keys = [key for key in warmstart_state if key.startswith("model.")]
        projector_keys = [key for key in warmstart_state if "projection_layer" in key]
        if not model_keys or not projector_keys:
            raise ValueError(
                "Weights-only warm start requires both student and projection-head weights; "
                f"found model={len(model_keys)}, projector={len(projector_keys)} in {checkpoint_path}"
            )

        incompatible = self.load_state_dict(warmstart_state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise ValueError(
                f"Weights-only warm start has unexpected keys in {checkpoint_path}: {unexpected[:10]}"
            )
        if get_global_rank() == 0:
            logger.info(
                "Weights-only warm start loaded from %s: student=%s, projector=%s; "
                "missing current-state keys=%s.",
                checkpoint_path,
                len(model_keys),
                len(projector_keys),
                len(incompatible.missing_keys),
            )

    def _iter_projection_head_modules(self):
        for loss_idx, loss_fn in enumerate(self.distillation_loss_fns):
            for attr in ("projection_layer", "projection_layer_summary", "intermediate_projection_layers"):
                module = getattr(loss_fn, attr, None)
                if module is None:
                    continue
                yield f"distillation_loss_fns.{loss_idx}.{attr}", module

    def _iter_projection_head_parameters(self):
        for _, param in self._iter_projection_head_named_parameters():
            yield param

    def _graph_connected_zero_loss(self, reference_tensor):
        """Return scalar zero with a grad path to currently trainable parameters."""
        zero_loss = reference_tensor.float().sum() * 0.0
        if zero_loss.requires_grad:
            return zero_loss

        head_zero_loss = None
        for param in self._iter_projection_head_parameters():
            if not param.requires_grad:
                continue
            param_zero_loss = param.float().sum() * 0.0
            head_zero_loss = (
                param_zero_loss if head_zero_loss is None else head_zero_loss + param_zero_loss
            )

        if head_zero_loss is not None:
            return head_zero_loss

        return zero_loss

    def _configure_backbone_spectral_norm(self):
        """Apply learnable-scale spectral reparam to the student backbone.

        This parametrizes every eligible Linear in the student backbone
        (stage-4 attention qkv/proj and MLP Linears). It runs after the
        stage-1 checkpoint has been loaded into ``self.model`` (in build_model) so
        ``init_norm_to_current=True`` initializes each scale to preserve the
        loaded weights' effective magnitude. Projection heads live in the loss
        fns (not ``self.model``), so this walker only touches the student.
        """
        if not getattr(self.distill_config, "spectral_reparam_backbone", False):
            return
        from nvidia_tao_pytorch.multimodal.radio.distillation.spectral_reparam import (
            enable_spectral_reparam,
        )
        n_power_iterations = int(
            getattr(self.distill_config, "spectral_projection_heads_power_iterations", 1)
        )
        eps = float(getattr(self.distill_config, "spectral_projection_heads_eps", 1.0e-6))
        alpha = float(getattr(self.distill_config, "spectral_projection_heads_alpha", 0.05))
        num = enable_spectral_reparam(
            self.model,
            n_power_iterations=n_power_iterations,
            eps=eps,
            alpha=alpha,
            init_norm_to_current=True,
        )
        if get_global_rank() == 0:
            logger.info("Backbone spectral reparam enabled: %d weight tensors parametrized.", num)

    def _configure_projection_head_spectral_norm(self):
        """Apply spectral normalization to projection-head Linear layers only."""
        if not getattr(self.distill_config, "spectral_projection_heads", False):
            return

        learnable_scale = bool(
            getattr(self.distill_config, "spectral_projection_heads_learnable_scale", False)
        )
        n_power_iterations = int(
            getattr(self.distill_config, "spectral_projection_heads_power_iterations", 1)
        )
        eps = float(getattr(self.distill_config, "spectral_projection_heads_eps", 1.0e-12))

        if learnable_scale:
            # Learnable gain: weight * (softplus(scale) + alpha) / sigma.
            from nvidia_tao_pytorch.multimodal.radio.distillation.spectral_reparam import (
                apply_learnable_spectral_norm,
            )
            alpha = float(getattr(self.distill_config, "spectral_projection_heads_alpha", 0.05))

            def _spectral_norm(module, name, n_power_iterations, eps):
                apply_learnable_spectral_norm(
                    module, name=name, n_power_iterations=n_power_iterations,
                    eps=eps, alpha=alpha, init_norm_to_current=True,
                )
        else:
            try:
                from torch.nn.utils.parametrizations import spectral_norm
            except ImportError:
                from torch.nn.utils import spectral_norm

            def _spectral_norm(module, name, n_power_iterations, eps):
                spectral_norm(module, name=name, n_power_iterations=n_power_iterations, eps=eps)

        wrapped = []
        skipped_zero = []

        for prefix, projection_module in self._iter_projection_head_modules():
            for name, module in projection_module.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                if hasattr(module, "weight_orig"):
                    continue
                if hasattr(module, "parametrizations") and hasattr(module.parametrizations, "weight"):
                    continue
                if torch.count_nonzero(module.weight.detach()).item() == 0:
                    skipped_zero.append(f"{prefix}.{name}" if name else prefix)
                    continue
                _spectral_norm(module, "weight", n_power_iterations, eps)
                wrapped.append(f"{prefix}.{name}" if name else prefix)

        if get_global_rank() == 0:
            logger.info(
                "Applied %sspectral normalization to %s projection-head Linear layers: %s",
                "learnable-scale " if learnable_scale else "",
                len(wrapped),
                wrapped,
            )
            if skipped_zero:
                logger.info(
                    "Skipped spectral normalization for zero-initialized projection-head Linear layers: %s",
                    skipped_zero,
                )

    def _needs_student_intermediate_features(self):
        """Return True when any active teacher loss supervises intermediate maps."""
        return any(
            float(getattr(loss_fn, "intermediate_loss_weight", 0.0) or 0.0) > 0.0
            for loss_fn in self.distillation_loss_fns
        )

    def _forward_student_features(self, student_input):
        """Forward the student with optional intermediate feature maps."""
        kwargs = {"return_features": True}
        if self._needs_student_intermediate_features():
            student_sig = inspect.signature(self.model.forward)
            if "return_intermediate_features" not in student_sig.parameters:
                raise RuntimeError(
                    "Intermediate distillation requires the student forward to accept "
                    "return_intermediate_features."
                )
            kwargs["return_intermediate_features"] = True
        return self.model(student_input, **kwargs)

    @staticmethod
    def _get_parameter_groups_from_named_parameters(
        named_parameters,
        weight_decay,
        skip_names=(),
        lr=None,
        allow_empty=False,
        exclude_param_ids=None,
    ):
        decay = []
        no_decay = []
        seen_params = set()
        exclude_param_ids = set(exclude_param_ids or ())

        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in exclude_param_ids:
                continue
            if param_id in seen_params:
                continue
            seen_params.add(param_id)

            if any(s in name for s in skip_names):
                no_decay.append(param)
            else:
                decay.append(param)

        parameters = []
        if no_decay:
            group = {"params": no_decay, "weight_decay": 0.0}
            if lr is not None:
                group["lr"] = lr
            parameters.append(group)
        if decay:
            group = {"params": decay, "weight_decay": weight_decay}
            if lr is not None:
                group["lr"] = lr
            parameters.append(group)
        if not parameters and not allow_empty:
            raise ValueError("No trainable parameters found for optimizer.")

        return parameters

    @staticmethod
    def _validate_unique_parameter_groups(parameters):
        seen_params = {}
        for group_idx, group in enumerate(parameters):
            for param in group["params"]:
                param_id = id(param)
                if param_id in seen_params:
                    raise ValueError(
                        "Parameter appears in more than one optimizer parameter group: "
                        f"groups {seen_params[param_id]} and {group_idx}"
                    )
                seen_params[param_id] = group_idx

    @staticmethod
    def _get_parameter_groups(model, weight_decay, skip_names=()):
        return MultiTeacherDistiller._get_parameter_groups_from_named_parameters(
            model.named_parameters(), weight_decay, skip_names
        )

    @staticmethod
    def _validate_teacher_bf16_precision(train_precision):
        """Reject teacher_bf16 combined with pure-fp32 training.

        teacher_bf16 relies on autocast to reconcile fp32 inputs with bf16 teacher
        weights; ``train.precision='fp32'`` ('32-true') runs without autocast, so the
        forward would raise a dtype mismatch. Fail fast with an actionable message.
        """
        if str(train_precision).lower() == "fp32":
            raise ValueError(
                "teacher_bf16=true requires an autocast-enabled training precision "
                "(train.precision 'bf16' or 'fp16'), but train.precision is 'fp32'. "
                "Set train.precision to 'bf16' (recommended) or disable teacher_bf16."
            )

    def configure_optimizers(self):
        """Configure optimizers for training"""
        train_projection_heads = getattr(self.distill_config, "train_projection_heads", False)
        projector_lr = getattr(self.distill_config, "projector_lr", None)
        if train_projection_heads and projector_lr is not None:
            model_named_parameters = list(self.model.named_parameters())
            model_param_ids = {id(param) for _, param in model_named_parameters}
            parameters = self._get_parameter_groups_from_named_parameters(
                model_named_parameters,
                self.optimizer.weight_decay,
                self.optimizer.skip_names,
                lr=self.lr,
            )
            projector_parameters = self._get_parameter_groups_from_named_parameters(
                list(self._iter_projection_head_named_parameters()),
                self.optimizer.weight_decay,
                self.optimizer.skip_names,
                lr=float(projector_lr),
                allow_empty=True,
                exclude_param_ids=model_param_ids,
            )
            parameters.extend(projector_parameters)
            if get_global_rank() == 0:
                logger.info(
                    "Using separate projection-head LR: backbone=%s, projector=%s",
                    self.lr,
                    projector_lr,
                )
        else:
            named_parameters = list(self.model.named_parameters())
            if train_projection_heads:
                named_parameters.extend(
                    self._iter_projection_head_named_parameters()
                )
            parameters = self._get_parameter_groups_from_named_parameters(
                named_parameters, self.optimizer.weight_decay, self.optimizer.skip_names
            )
        self._validate_unique_parameter_groups(parameters)
        # define optimizers
        if self.optimizer.optim == "sgd":
            self.optimizer_G = optim.SGD(
                parameters,
                lr=self.lr,
                momentum=self.optimizer.momentum,  # 0.9
                weight_decay=self.optimizer.weight_decay,
            )  # 5e-4
        elif self.optimizer.optim == "adam":
            self.optimizer_G = optim.Adam(
                parameters,
                lr=self.lr,
                weight_decay=self.optimizer.weight_decay,
            )  # 0
        elif self.optimizer.optim == "adamw":
            self.optimizer_G = optim.AdamW(
                parameters,
                lr=self.lr,
                betas=self.optimizer.betas,
                weight_decay=self.optimizer.weight_decay,
            )
        elif self.optimizer.optim == "lamb":
            # The standard LAMB option uses timm's pure-PyTorch implementation;
            # only fusedlamb below requires Apex.
            from timm.optim.lamb import Lamb
            self.optimizer_G = Lamb(
                parameters,
                lr=self.lr,
                betas=tuple(self.optimizer.betas),
                weight_decay=self.optimizer.weight_decay,
            )
        elif self.optimizer.optim == "fusedlamb":
            try:
                from apex.optimizers import FusedLAMB
            except ImportError as exc:
                raise ImportError(
                    "Optimizer 'fusedlamb' requires apex.optimizers.FusedLAMB in the training container. "
                    "Use optim='lamb' for the apex-free timm.optim.lamb.Lamb equivalent."
                ) from exc
            self.optimizer_G = FusedLAMB(
                parameters,
                lr=self.lr,
                betas=tuple(self.optimizer.betas),
                weight_decay=self.optimizer.weight_decay,
            )
        else:
            raise NotImplementedError(
                "Optimizer {} is not implemented".format(self.optimizer.optim)
            )

        # ``sched_on_updates`` makes per-step vs per-epoch stepping independent
        # of the selected policy.
        lr_policy = self.lr_policy.lower()
        sched_on_updates = bool(getattr(self.optimizer, "sched_on_updates", False))
        # Optimizer steps per epoch. ``estimated_stepping_batches`` is Lightning's
        # total number of optimizer.step() calls over the whole run and already
        # accounts for accumulate_grad_batches, so dividing only by max_epochs
        # yields per-epoch update steps (dividing again by accumulate_grad_batches
        # would make step/multistep milestones early by that same factor).
        epoch_steps = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        if lr_policy == "linear":
            if sched_on_updates:
                total_steps = self.trainer.estimated_stepping_batches

                def lambda_rule(step):
                    return 1 - step / float(total_steps + 1)
            else:
                def lambda_rule(epoch):
                    # gradually decay learning rate from epoch 0 to max_epochs
                    lr_l = 1 - (epoch) / float(self.max_epochs + 1)
                    return lr_l

            interval = "step" if sched_on_updates else "epoch"
            scheduler = lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule)
        elif lr_policy == "step":
            if self.lr_policy_params is not None:
                step_size = self.lr_policy_params.step_size
                gamma = self.lr_policy_params.gamma
            else:   # default values
                step_size = self.max_epochs // 4
                gamma = 0.1
            interval = "step" if sched_on_updates else "epoch"
            if sched_on_updates:
                step_size = step_size * epoch_steps
            scheduler = lr_scheduler.StepLR(
                self.optimizer_G, step_size=step_size, gamma=gamma
            )
        elif lr_policy == "multistep":
            if self.lr_policy_params is not None:
                milestones = self.lr_policy_params.milestones
                gamma = self.lr_policy_params.gamma
            else:
                milestones = [self.max_epochs // 2]
                gamma = 0.1
            interval = "step" if sched_on_updates else "epoch"
            if sched_on_updates:
                milestones = [m * epoch_steps for m in milestones]
            scheduler = lr_scheduler.MultiStepLR(self.optimizer_G, milestones, gamma=gamma)
        elif lr_policy == "cosine":
            # cosine already always steps per-update; sched_on_updates has no effect here.
            interval = "step"
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer_G,
                num_training_steps=self.trainer.estimated_stepping_batches,
                num_warmup_steps=epoch_steps * self.optimizer.warmup_epochs,
            )
        else:
            raise NotImplementedError('learning rate policy [{}] is not implemented'.format(self.lr_policy))

        self.lr_scheduler = scheduler

        optim_dict = {}
        optim_dict["optimizer"] = self.optimizer_G
        optim_dict["lr_scheduler"] = {
            "scheduler": self.lr_scheduler,
            "interval": interval,
            "frequency": 1
        }
        optim_dict["monitor"] = self.monitor_name
        return optim_dict

    def _get_stochastic_resolution(self):
        """Return (res_list, prob_list) for per-batch resize, or (None, None) to skip.
        When augmentation.stochastic_resolutions is set (global), or any teacher has
        stochastic_resolutions (per-teacher / multiview), the dataloader does per-sample
        resolution sampling, so we skip per-batch resize here. Otherwise use multi_scales if set.
        """
        aug = self.experiment_spec.dataset.augmentation
        stoch = getattr(aug, "stochastic_resolutions", None)
        if stoch is not None and hasattr(stoch, 'items') and len(stoch) > 0:
            return None, None
        # Per-teacher stochastic_resolutions: multiview pipeline already does per-sample resolution
        if any(c.get("stochastic_resolutions") for c in self.teacher_configs):
            return None, None
        multi_scales = list(getattr(aug, "multi_scales", []) or [])
        if not multi_scales:
            return None, None
        res = [list(i.keys())[0] for i in multi_scales]
        prob = [list(i.values())[0] for i in multi_scales]
        total = sum(prob)
        if total <= 0:
            return None, None
        prob = [p / total for p in prob]
        return res, prob

    @staticmethod
    def _dump_batch(dump_dir, batch_idx, batch):
        """Save a batch to disk for runtime diagnostics."""
        os.makedirs(dump_dir, exist_ok=True)
        data = {
            "student_img": batch["img"].detach().cpu(),
            "class": batch.get("class", torch.tensor([])).detach().cpu(),
        }
        if "valid_mask" in batch:
            data["student_mask"] = batch["valid_mask"].detach().cpu()
        tvs = batch.get("teacher_views", [])
        data["num_teachers"] = len(tvs)
        for i, tv in enumerate(tvs):
            data[f"teacher_{i}_img"] = tv["img"].detach().cpu()
            data[f"teacher_{i}_mask"] = tv["valid_mask"].detach().cpu()
            data[f"teacher_{i}_stm"] = tv["spatial_transform"].detach().cpu()
        path = os.path.join(dump_dir, f"batch_{batch_idx:03d}.pt")
        torch.save(data, path)
        if batch_idx == 0:
            logger = logging.getLogger("batch_dump")
            logger.info("Batch dump: saving to %s", dump_dir)

    @staticmethod
    def _dump_forward(dump_dir, batch_idx, forward_data):
        """Save forward-pass tensors and losses for runtime diagnostics."""
        fwd_dir = os.path.join(dump_dir, "forward")
        os.makedirs(fwd_dir, exist_ok=True)
        path = os.path.join(fwd_dir, f"batch_{batch_idx:03d}.pt")
        torch.save(forward_data, path)
        if batch_idx == 0:
            logger = logging.getLogger("forward_dump")
            logger.info("Forward dump: saving to %s", fwd_dir)

    def _set_head_warmup_active(self, active: bool):
        """Freeze/unfreeze the student backbone for projection-head warmup."""
        if self._head_warmup_active == active:
            return

        warmup_epochs = int(getattr(self.distill_config, "head_warmup_epochs", 0) or 0)
        train_heads = getattr(self.distill_config, "train_projection_heads", False)
        freeze_after = (
            train_heads and
            getattr(self.distill_config, "freeze_projection_heads_after_warmup", False) and
            not active and
            self.current_epoch >= warmup_epochs
        )

        for param in self.model.parameters():
            param.requires_grad = not active

        self._set_projection_head_trainable(not freeze_after)

        self._enforce_frozen_student_norms()

        self._head_warmup_active = active
        if get_global_rank() == 0:
            state = "enabled" if active else "disabled"
            logger.info("Distillation projection-head warmup %s.", state)
            if freeze_after:
                logger.info("Distillation projection heads frozen after warmup.")

    def on_train_epoch_start(self):
        """Apply projection-head warmup when configured."""
        datamodule = self._get_trainer_datamodule()
        if datamodule is not None and hasattr(datamodule, "set_epoch"):
            datamodule.set_epoch(self.current_epoch)

        warmup_epochs = int(getattr(self.distill_config, "head_warmup_epochs", 0) or 0)
        active = self.current_epoch < warmup_epochs
        if active and not getattr(self.distill_config, "train_projection_heads", False):
            if get_global_rank() == 0:
                logger.warning(
                    "head_warmup_epochs is set but train_projection_heads is false; "
                    "skipping projection-head warmup."
                )
            active = False
        self._set_head_warmup_active(active)
        self._enforce_frozen_student_norms()

    def _rank_partition(self):
        """Map this rank to its contiguous partition, resolution, and teachers."""
        world = int(self.trainer.world_size)
        num_partitions = int(
            getattr(self.distill_config, "num_rank_partitions", 4) or 4
        )
        if (
            num_partitions < 2 or
            world < num_partitions or
            world % num_partitions != 0
        ):
            raise ValueError(
                "partitioned_ranks requires world_size divisible by "
                f"num_rank_partitions; got world={world}, "
                f"partitions={num_partitions}"
            )
        ranks_per_partition = world // num_partitions
        partition = min(
            int(self.global_rank) // ranks_per_partition,
            num_partitions - 1,
        )
        active = [
            idx for idx, config in enumerate(self.teacher_configs)
            if int(config.get("rank_partition", -1)) == partition
        ]
        if not active:
            raise ValueError(
                f"No teacher configured for rank partition {partition}"
            )
        img_size = int(self.dataset_config.img_size)
        resolution = max(
            int(self.teacher_configs[idx].get("student_resolution") or img_size)
            for idx in active
        )
        return resolution, active, ranks_per_partition

    def _init_partition_groups(self):
        """Create contiguous rank groups and per-teacher collective groups."""
        if not dist.is_initialized():
            return
        self._init_contiguous_partition_groups()

    def _init_contiguous_partition_groups(self):
        """Build contiguous rank groups and per-teacher union groups."""
        world = dist.get_world_size()
        num_partitions = int(
            getattr(self.distill_config, "num_rank_partitions", 4) or 4
        )
        if num_partitions < 2 or world < num_partitions or world % num_partitions != 0:
            raise ValueError(
                "partitioned_ranks requires world_size divisible by "
                f"num_rank_partitions; got world={world}, "
                f"partitions={num_partitions}"
            )
        ranks_per_partition = world // num_partitions
        partition_ranks = {
            partition: list(
                range(
                    partition * ranks_per_partition,
                    (partition + 1) * ranks_per_partition,
                )
            )
            for partition in range(num_partitions)
        }
        self._rank_partition_groups = {
            partition: dist.new_group(ranks)
            for partition, ranks in partition_ranks.items()
        }

        keys = []
        for config in self.teacher_configs:
            key = config.get("shared_teacher_key", "")
            partition = int(config.get("rank_partition", -1))
            if not key or partition not in partition_ranks:
                raise ValueError(
                    "Every partitioned teacher needs shared_teacher_key and "
                    f"rank_partition in [0,{num_partitions - 1}]; "
                    f"got key={key!r}, partition={partition}"
                )
            if key not in keys:
                keys.append(key)

        teacher_groups = {}
        teacher_ranks = {}
        for key in keys:
            partitions = sorted({
                int(config["rank_partition"])
                for config in self.teacher_configs
                if config["shared_teacher_key"] == key
            })
            ranks = [
                rank
                for partition in partitions
                for rank in partition_ranks[partition]
            ]
            teacher_ranks[key] = ranks
            teacher_groups[key] = dist.new_group(ranks)
        self._teacher_groups = teacher_groups
        self._teacher_ranks = teacher_ranks

        for loss_fn, config in zip(
            self.distillation_loss_fns, self.teacher_configs
        ):
            group = teacher_groups[config["shared_teacher_key"]]
            loss_fn.dist_group = group
            for attr in ("phi_norm", "summary_criterion"):
                state = getattr(loss_fn, attr, None)
                if state is not None and hasattr(state, "dist_group"):
                    state.dist_group = group

        self._head_grad_hooks = getattr(self, "_head_grad_hooks", [])
        hooked = set()
        for loss_fn, config in zip(
            self.distillation_loss_fns, self.teacher_configs
        ):
            key = config["shared_teacher_key"]
            scale = float(world) / float(len(teacher_ranks[key]))
            for attr in (
                "projection_layer",
                "projection_layer_summary",
                "intermediate_projection_layers",
            ):
                head = getattr(loss_fn, attr, None)
                if head is None:
                    continue
                for parameter in head.parameters():
                    if not parameter.requires_grad or id(parameter) in hooked:
                        continue
                    hooked.add(id(parameter))
                    self._head_grad_hooks.append(
                        parameter.register_hook(lambda grad, value=scale: grad * value)
                    )
        if get_global_rank() == 0:
            logger.info(
                "Partitioned distillation: %d contiguous partitions, "
                "teacher groups=%s, head hooks=%d",
                num_partitions,
                {key: ranks for key, ranks in teacher_ranks.items()},
                len(self._head_grad_hooks),
            )

    def _global_batch_size(self):
        """Return the configured global batch across rank partitions."""
        world = int(self.trainer.world_size)
        num_partitions = int(
            getattr(self.distill_config, "num_rank_partitions", 4) or 4
        )
        ranks_per_partition = world // num_partitions
        batch_by_partition = {}
        for config in self.teacher_configs:
            partition = int(config.get("rank_partition", -1))
            batch = int(config.get("local_batch_size", 0))
            previous = batch_by_partition.setdefault(partition, batch)
            if previous != batch or batch <= 0:
                raise ValueError(
                    "Partition entries must share one positive local "
                    f"batch size; partition={partition}, values={previous},{batch}"
                )
        expected_partitions = list(range(num_partitions))
        if sorted(batch_by_partition) != expected_partitions:
            raise ValueError(
                "Partitioned training requires every configured partition; "
                f"expected={expected_partitions}, got={sorted(batch_by_partition)}"
            )
        return ranks_per_partition * sum(batch_by_partition.values())

    def _arm_loss_scale(self, teacher_config, actual_batch_size):
        """Return teacher rebalance times local/global sample weighting."""
        world = int(self.trainer.world_size)
        key = teacher_config["shared_teacher_key"]
        teacher_partitions = {
            int(config["rank_partition"])
            for config in self.teacher_configs
            if config["shared_teacher_key"] == key
        }
        num_partitions = int(
            getattr(self.distill_config, "num_rank_partitions", 4) or 4
        )
        teacher_world = len(teacher_partitions) * (world // num_partitions)
        teacher_rebalance = (
            world / teacher_world
            if getattr(self.distill_config, "rebalance_teacher_loss", True)
            else 1.0
        )
        sample_weight = (
            int(actual_batch_size) *
            world /
            self._global_batch_size()
        )
        return teacher_rebalance * sample_weight

    def training_step(self, batch, batch_idx):
        """Training step"""
        _dump_dir = os.environ.get("RADIO_DUMP_DIR")
        _dump_max = int(os.environ.get("RADIO_DUMP_BATCHES") or "5")
        _dumping = _dump_dir and batch_idx < _dump_max
        if _dumping:
            self._dump_batch(_dump_dir, batch_idx, batch)
            _fwd = {}

        partitioned = bool(
            getattr(self.distill_config, "partitioned_ranks", False)
        )
        if partitioned:
            if not getattr(self, "_partition_groups_ready", False):
                self._init_partition_groups()
                self._partition_groups_ready = True
            partition_res, active_teachers, ranks_at_res = self._rank_partition()
            teacher_rebalance = (
                self.trainer.world_size / ranks_at_res
                if getattr(self.distill_config, "rebalance_teacher_loss", True) else 1.0
            )
            # This rank runs one student forward at its partition resolution.
            # Skip stochastic multi-scale resize so BatchNorm is not updated by
            # an unsupervised off-resolution forward.
            if int(batch["img"].shape[-1]) != partition_res:
                batch["img"] = F.interpolate(
                    batch["img"], size=(partition_res, partition_res),
                    mode="bilinear", align_corners=False,
                )
        else:
            active_teachers = None
            teacher_rebalance = 1.0
            res_list, prob_list = self._get_stochastic_resolution()
            if res_list is not None and prob_list is not None:
                sz = int(np.random.choice(a=res_list, p=prob_list))
                if isinstance(sz, int):
                    batch["img"] = F.interpolate(batch["img"], size=[sz, sz])
                elif isinstance(sz, (list, tuple)):
                    batch["img"] = F.interpolate(batch["img"], size=sz)
                else:
                    raise TypeError(f"{sz} is {type(sz)}. Need to pass int / list / tuple for multi_scale")

        student_input = self._normalize_student_input(batch["img"])
        if _dumping:
            _fwd["student_input_normalized"] = student_input.detach().cpu()

        self._enforce_frozen_student_norms()
        if self._baseline_anchor_model is not None:
            self._baseline_anchor_model.eval()
            self._reset_baseline_anchor_cache()

        student_summary, student_spatial = self._forward_student_features(student_input)

        if partitioned:
            # The rank already ran one forward at its partition resolution.
            def _student_feats_for_resolution(target_res):
                """Partitioned: this rank has one forward at its resolution; target_res is ignored."""
                return student_input, student_summary, student_spatial
        else:
            # Non-partitioned routing runs the student once per distinct per-teacher
            # student_resolution and route each teacher to the forward at its resolution. The default
            # forward above (at img_size) is cached; extra-resolution forwards run BN in eval so the
            # deployed running stats are not double-updated.
            default_student_res = int(batch["img"].shape[-1])
            student_feats_cache = {default_student_res: (student_input, student_summary, student_spatial)}

            def _student_feats_for_resolution(target_res):
                """Return (student_input, summary, spatial) for a square target_res, caching per resolution."""
                res = int(target_res) if target_res else default_student_res
                if res not in student_feats_cache:
                    resized = F.interpolate(
                        batch["img"], size=(res, res), mode="bilinear", align_corners=False
                    )
                    res_input = self._normalize_student_input(resized)
                    bn_was_training = [
                        m for m in self.model.modules()
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.training
                    ]
                    for m in bn_was_training:
                        m.eval()
                    try:
                        res_summary, res_spatial = self._forward_student_features(res_input)
                    finally:
                        for m in bn_was_training:
                            m.train()
                    student_feats_cache[res] = (res_input, res_summary, res_spatial)
                return student_feats_cache[res]

        zero_loss = self._graph_connected_zero_loss(student_summary)
        baseline_anchor_loss = zero_loss
        baseline_anchor_logs = {}
        if self._baseline_anchor_model is not None:
            with torch.no_grad():
                anchor_summary, anchor_spatial = self._baseline_anchor_model(student_input, return_features=True)
            baseline_anchor_loss, baseline_anchor_logs = self._compute_baseline_anchor_loss(
                student_summary,
                student_spatial,
                anchor_summary,
                anchor_spatial,
            )
        loss = zero_loss

        # Compute distillation loss from all teachers
        total_distillation_loss = zero_loss
        total_teacher_weight = 0.0
        distill_scale = 1.0
        use_multiview = "teacher_views" in batch and len(batch["teacher_views"]) > 0
        view_indices = list(
            batch.get(
                "teacher_view_indices",
                range(len(batch.get("teacher_views", []))),
            )
        )
        teacher_views = {
            int(teacher_idx): view
            for teacher_idx, view in zip(
                view_indices, batch.get("teacher_views", [])
            )
        }

        for idx, (loss_fn, teacher_config) in enumerate(zip(self.distillation_loss_fns, self.teacher_configs)):
            # Under partitioning this rank only computes its partition's teachers; the others are
            # run by other ranks and mixed in via DDP gradient-averaging.
            if partitioned and idx not in active_teachers:
                continue
            # Route this teacher to the student forward at its student_resolution.
            t_student_input, t_student_summary, t_student_spatial = _student_feats_for_resolution(
                teacher_config.get("student_resolution")
            )
            if use_multiview and idx in teacher_views:
                tv = teacher_views[idx]
                teacher_input = tv["img"].to(batch["img"].device)
                teacher_input = self._apply_teacher_normalization(
                    teacher_input, teacher_config, batch["img"].device
                )
                if "valid_mask" in batch:
                    student_valid_mask = batch["valid_mask"].to(batch["img"].device)
                    if student_valid_mask.dim() == 4 and student_valid_mask.shape[1] == 1:
                        student_valid_mask = student_valid_mask.squeeze(1)
                else:
                    student_valid_mask = torch.ones(
                        batch["img"].shape[0], batch["img"].shape[2], batch["img"].shape[3],
                        dtype=torch.float32, device=batch["img"].device
                    )
                teacher_valid_mask = tv["valid_mask"].to(batch["img"].device)
                if teacher_valid_mask.dim() == 4 and teacher_valid_mask.shape[1] == 1:
                    teacher_valid_mask = teacher_valid_mask.squeeze(1)
                spatial_transform = tv["spatial_transform"].to(batch["img"].device)
                teacher_distill_loss = loss_fn(
                    t_student_input,
                    teacher_batch_input=teacher_input,
                    student_valid_mask=student_valid_mask,
                    teacher_valid_mask=teacher_valid_mask,
                    spatial_transform=spatial_transform,
                    student_summary=t_student_summary,
                    student_spatial=t_student_spatial,
                )
            else:
                teacher_input = self._apply_teacher_normalization(
                    batch["img"], teacher_config, batch["img"].device
                )
                teacher_distill_loss = loss_fn(
                    t_student_input,
                    teacher_batch_input=teacher_input,
                    student_summary=t_student_summary,
                    student_spatial=t_student_spatial,
                )

            if _dumping:
                _fwd[f"teacher_{idx}_input_normalized"] = teacher_input.detach().cpu()
                _fwd[f"teacher_{idx}_loss"] = teacher_distill_loss.detach().cpu()
                for name, value in getattr(loss_fn, "last_alignment_metrics", {}).items():
                    _fwd[f"teacher_{idx}_{name}"] = value.detach().cpu()

            for name, value in getattr(loss_fn, "last_alignment_metrics", {}).items():
                self.log(
                    f"align_teacher_{idx}_{name}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    # Partitioned teacher arms run on rank subsets. A global
                    # Lightning reduction here would require inactive ranks to
                    # enter a collective they never reach.
                    sync_dist=not partitioned,
                    batch_size=int(batch["img"].shape[0]),
                    rank_zero_only=True,
                )

            teacher_weight = teacher_config['loss_lambda']
            # Rebalancing cancels DDP averaging over ranks where a teacher is
            # inactive, preserving the full sum-over-teachers gradient.
            if partitioned:
                arm_scale = self._arm_loss_scale(
                    teacher_config, int(batch["img"].shape[0])
                )
            else:
                arm_scale = teacher_rebalance
            weighted_loss = (
                teacher_weight *
                teacher_distill_loss *
                distill_scale *
                arm_scale
            )
            debug_steps = int(os.environ.get("RADIO_DISTILL_DEBUG_STEPS") or "0")
            if debug_steps > 0 and batch_idx < debug_steps and get_global_rank() == 0:
                debug_metrics = []
                for name, value in getattr(loss_fn, "last_alignment_metrics", {}).items():
                    if not torch.is_tensor(value):
                        continue
                    value_detached = value.detach().float()
                    debug_metrics.append(
                        f"{name}={value_detached.item():.8g}"
                        if value_detached.numel() == 1
                        else f"{name}_shape={tuple(value_detached.shape)}"
                    )
                logger.info(
                    "distill_debug batch=%s teacher=%s loss=%s weighted=%s finite=%s %s",
                    batch_idx,
                    idx,
                    teacher_distill_loss.detach().float().item()
                    if teacher_distill_loss.detach().numel() == 1 else teacher_distill_loss.detach().shape,
                    weighted_loss.detach().float().item()
                    if weighted_loss.detach().numel() == 1 else weighted_loss.detach().shape,
                    bool(torch.isfinite(weighted_loss).all().item()),
                    " ".join(debug_metrics),
                )

            if torch.isfinite(weighted_loss).all():
                total_distillation_loss = total_distillation_loss + weighted_loss
                total_teacher_weight += teacher_weight
            elif get_global_rank() == 0 and batch_idx < 5:
                logger.warning(
                    "Skipping non-finite distillation loss at batch=%s teacher=%s loss=%s weighted=%s",
                    batch_idx,
                    idx,
                    teacher_distill_loss.detach().float().item()
                    if teacher_distill_loss.detach().numel() == 1 else teacher_distill_loss.detach().shape,
                    weighted_loss.detach().float().item()
                    if weighted_loss.detach().numel() == 1 else weighted_loss.detach().shape,
                )

            # Log per-teacher loss (unscaled, for readability)
            self.log(
                f"distill_loss_teacher_{idx}",
                teacher_distill_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=not partitioned,
                batch_size=int(batch["img"].shape[0]),
                rank_zero_only=True
            )

        # Normalize supervised loss weight
        if total_teacher_weight > 0:
            supervised_weight = 1.0 - total_teacher_weight
        else:
            supervised_weight = 1.0

        supervised_loss = supervised_weight * loss
        distill_loss = total_distillation_loss
        # Unscaled distillation loss (raw value from loss_fn, ~1–3) for logging/prog_bar. Divide out
        # teacher_rebalance so the logged value is not inflated under partitioning (== 1.0 otherwise).
        distill_loss_raw = (
            distill_loss.detach()
            if partitioned
            else (
                distill_loss / distill_scale / teacher_rebalance
                if total_teacher_weight > 0 else distill_loss
            )
        )

        if torch.isnan(supervised_loss):
            supervised_loss = zero_loss

        if torch.isnan(distill_loss):
            distill_loss = zero_loss

        if torch.isnan(baseline_anchor_loss):
            baseline_anchor_loss = zero_loss

        total_loss = supervised_loss + distill_loss + baseline_anchor_loss
        for name, value in baseline_anchor_logs.items():
            self.log(
                name,
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=(name == "baseline_anchor_loss"),
                sync_dist=True,
                batch_size=self.batch_size,
                rank_zero_only=True,
            )
        self.log(
            "distillation_loss_raw",
            distill_loss_raw,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True,
        )
        self.log(
            "lr",
            self.lr_schedulers().get_last_lr()[-1],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            "supervised_loss",
            supervised_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True
        )
        self.log(
            "distillation_loss",
            distill_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True
        )
        self.log(
            "total_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True
        )
        if _dumping:
            _fwd["total_loss"] = total_loss.detach().cpu()
            _fwd["supervised_loss"] = supervised_loss.detach().cpu()
            _fwd["distillation_loss"] = distill_loss.detach().cpu()
            _fwd["baseline_anchor_loss"] = baseline_anchor_loss.detach().cpu()
            self._dump_forward(_dump_dir, batch_idx, _fwd)

        return {"loss": total_loss}

    @torch.no_grad()
    def _synchronize_partitioned_distillation_state(self):
        """Put canonical teacher-state buffers on every checkpointing rank."""
        if (
            not getattr(self.distill_config, "partitioned_ranks", False) or
            not dist.is_initialized()
        ):
            return 0

        if not getattr(self, "_partition_groups_ready", False):
            self._init_partition_groups()
            self._partition_groups_ready = True

        synchronized = 0
        seen = set()
        for loss_fn, teacher_config in zip(
            self.distillation_loss_fns, self.teacher_configs
        ):
            source_rank = self._teacher_ranks[
                teacher_config["shared_teacher_key"]
            ][0]
            for attr in ("phi_norm", "summary_criterion"):
                state = getattr(loss_fn, attr, None)
                if state is None or id(state) in seen:
                    continue
                seen.add(id(state))

                # Stateful losses first reduce within their teacher subgroup,
                # then broadcast the complete state over the global group.
                if hasattr(state, "synchronize"):
                    state.synchronize()
                else:
                    # Keep supported stateful summary criteria checkpoint-safe.
                    for buffer in state.buffers():
                        dist.broadcast(buffer, src=source_rank)
                synchronized += 1
        return synchronized

    def _synchronize_partitioned_epoch_state(self):
        """Synchronize partitioned BN and loss state before validation/save."""
        if not getattr(self.distill_config, "partitioned_ranks", False):
            return

        global_step = int(self.global_step)
        if self._partitioned_state_sync_step == global_step:
            return

        _, _, dist_bn_mode = _resolve_distillation_runtime_modes(
            self.distill_config
        )
        distributed_buffers = _distribute_batch_norm_buffers(
            self.model, dist_bn_mode
        )
        distributed_states = (
            self._synchronize_partitioned_distillation_state()
        )
        self._partitioned_state_sync_step = global_step

        if get_global_rank() == 0 and not self._student_bn_distribution_logged:
            logger.info(
                "Partitioned state synchronized before validation/"
                "checkpoint: BatchNorm buffers=%d (mode=%s), teacher states=%d.",
                distributed_buffers,
                dist_bn_mode,
                distributed_states,
            )
            self._student_bn_distribution_logged = True

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Synchronize partitioned state after the final optimizer batch."""
        del outputs, batch
        if not getattr(self.distill_config, "partitioned_ranks", False):
            return

        is_last_batch = bool(getattr(self.trainer, "is_last_batch", False))
        num_batches = getattr(self.trainer, "num_training_batches", None)
        if isinstance(num_batches, int):
            is_last_batch = is_last_batch or batch_idx + 1 >= num_batches
        if is_last_batch:
            self._synchronize_partitioned_epoch_state()

    def on_train_epoch_end(self):
        """Log Training metrics to status.json"""
        partitioned = getattr(
            self.distill_config, "partitioned_ranks", False
        )
        if partitioned:
            # Idempotent safeguard for loops that do not expose ``is_last_batch``.
            self._synchronize_partitioned_epoch_state()
        else:
            _, _, dist_bn_mode = _resolve_distillation_runtime_modes(
                self.distill_config
            )
            distributed_buffers = _distribute_batch_norm_buffers(
                self.model, dist_bn_mode
            )
            if (
                distributed_buffers and
                get_global_rank() == 0 and
                not self._student_bn_distribution_logged
            ):
                logger.info(
                    "Distributed %s student BatchNorm running buffers at train "
                    "epoch end with mode=%s.",
                    distributed_buffers,
                    dist_bn_mode,
                )
                self._student_bn_distribution_logged = True

        average_train_loss = self.trainer.logged_metrics["total_loss_epoch"].item()
        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING,
        )

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Partitioned validation evaluates the synchronized student/kNN probe
        without re-entering rank-local teacher collectives. Non-partitioned
        validation retains per-teacher loss and CKA.
        """
        student_input = self._normalize_student_input(batch["img"])
        student_summary, student_spatial = self._forward_student_features(student_input)
        out = student_summary
        loss = torch.tensor(0.0).to(out.device)

        # Compute distillation loss from all teachers
        total_distillation_loss = torch.tensor(0.0, device=out.device)
        use_multiview = "teacher_views" in batch and len(batch["teacher_views"]) > 0

        partitioned = getattr(
            self.distill_config, "partitioned_ranks", False
        )
        teacher_pairs = (
            ()
            if partitioned
            else enumerate(zip(self.distillation_loss_fns, self.teacher_configs))
        )
        for idx, (loss_fn, teacher_config) in teacher_pairs:
            if use_multiview and idx < len(batch["teacher_views"]):
                tv = batch["teacher_views"][idx]
                teacher_input = tv["img"].to(out.device)
                teacher_input = self._apply_teacher_normalization(
                    teacher_input, teacher_config, out.device
                )
                student_valid_mask = torch.ones(
                    batch["img"].shape[0], batch["img"].shape[2], batch["img"].shape[3],
                    dtype=torch.float32, device=out.device
                )
                teacher_valid_mask = tv["valid_mask"].to(out.device)
                if teacher_valid_mask.dim() == 3:
                    teacher_valid_mask = teacher_valid_mask[:, 0]
                spatial_transform = tv["spatial_transform"].to(out.device)
                teacher_distill_loss = loss_fn(
                    student_input,
                    teacher_batch_input=teacher_input,
                    student_valid_mask=student_valid_mask,
                    teacher_valid_mask=teacher_valid_mask,
                    spatial_transform=spatial_transform,
                    student_summary=student_summary,
                    student_spatial=student_spatial,
                )
                self._accumulate_cka_stats(
                    idx, loss_fn, student_input, teacher_input,
                    student_valid_mask, teacher_valid_mask,
                    spatial_transform, student_spatial,
                )
            else:
                # Val loader has no teacher views. Skip teachers requiring a fixed
                # non-student resolution (e.g. SAM3) to avoid RoPE shape mismatches.
                t_size = teacher_config.get('input_size')
                if t_size and not teacher_config.get('match_student_resolution', True):
                    continue
                teacher_input = self._apply_teacher_normalization(
                    batch["img"], teacher_config, out.device
                )
                teacher_distill_loss = loss_fn(
                    student_input,
                    teacher_batch_input=teacher_input,
                    student_summary=student_summary,
                    student_spatial=student_spatial,
                )
                self._accumulate_cka_stats(
                    idx, loss_fn, student_input, teacher_input,
                    None, None, None, student_spatial,
                )

            if not torch.isnan(teacher_distill_loss):
                total_distillation_loss += teacher_distill_loss

            # Log per-teacher validation loss
            self.log(
                f"val_distill_loss_teacher_{idx}",
                teacher_distill_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.batch_size,
                rank_zero_only=True
            )

        # Log total distillation loss
        self.log(
            "val_distillation_loss",
            total_distillation_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
            rank_zero_only=True
        )
        loss += total_distillation_loss

        if self._knn_index is not None:
            knn_acc, knn_batch_size = knn_eval_batch(
                model=self.model,
                normalize_fn=self._normalize_student_input,
                batch=batch,
                train_embeddings=self._knn_index[0],
                train_labels=self._knn_index[1],
                device=self.device,
                K=20,
                num_classes=self.dataset_config.get("knn_num_classes", 1000),
                distributed=torch.distributed.is_initialized(),
            )
            self._knn_correct += (knn_acc / 100.0 * knn_batch_size).long()
            self._knn_total += knn_batch_size

        return loss

    def _accumulate_cka_stats(self, idx, loss_fn, student_input, teacher_input,
                              student_valid_mask, teacher_valid_mask,
                              spatial_transform, student_spatial):
        """Accumulate per-teacher linear-CKA sufficient statistics for the epoch.

        Computed on raw backbone features so the metric is independent of the
        distillation projector/normalization and comparable across runs.
        """
        if getattr(loss_fn, "distillation_mode", None) not in ("spatial", "combo"):
            return
        if not hasattr(loss_fn, "compute_spatial_cka_stats"):
            return
        try:
            stats = loss_fn.compute_spatial_cka_stats(
                student_input,
                teacher_batch_input=teacher_input,
                student_valid_mask=student_valid_mask,
                teacher_valid_mask=teacher_valid_mask,
                spatial_transform=spatial_transform,
                student_spatial=student_spatial,
            )
        except Exception as e:  # noqa: BLE001 - metric must never break validation
            if get_global_rank() == 0 and not getattr(self, "_cka_warned", False):
                logger.warning(f"val_spatial_cka computation failed, skipping: {e}")
                self._cka_warned = True
            return
        if stats is None:
            return
        acc = self._cka_stats.get(idx)
        if acc is None:
            self._cka_stats[idx] = {k: v.clone() for k, v in stats.items()}
        else:
            for k, v in stats.items():
                acc[k] += v

    @staticmethod
    def _linear_cka_from_stats(acc, eps: float = 1e-8) -> float:
        """Compute global linear CKA from accumulated covariance statistics.

        Centering uses the full-set means recovered from the running sums, so
        this matches CKA computed over the entire validation set in one pass
        (not a biased per-batch average).
        """
        n = acc["count"].clamp_min(1.0)
        sum_x = acc["sum_x"]
        sum_y = acc["sum_y"]
        xc_yc = acc["sum_xy"] - torch.outer(sum_x, sum_y) / n
        xc_xc = acc["sum_xx"] - torch.outer(sum_x, sum_x) / n
        yc_yc = acc["sum_yy"] - torch.outer(sum_y, sum_y) / n
        hsic_xy = xc_yc.square().sum()
        hsic_xx = xc_xc.square().sum()
        hsic_yy = yc_yc.square().sum()
        cka = hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt()).clamp_min(eps)
        return float(cka.item())

    def on_validation_start(self):
        """Skip the parent batch-size check for WDS loaders."""
        # Idempotent safeguard: in the normal fit loop this already ran after
        # the final training batch, before validation/checkpoint callbacks.
        self._synchronize_partitioned_epoch_state()
        if self.trainer.datamodule.val_dataset_type != "WebDataset":
            super().on_validation_start()

    def on_validation_epoch_start(self):
        """Build KNN index from the train split before val batches arrive."""
        dm = self.trainer.datamodule

        self._knn_index = None
        self._cka_stats = {}

        if dm.val_train_split_loader is not None and not self.trainer.sanity_checking:
            self._knn_correct = torch.tensor(0, dtype=torch.int64, device=self.device)
            self._knn_total = torch.tensor(0, dtype=torch.int64, device=self.device)
            self._knn_index = build_knn_index(
                model=self.model,
                normalize_fn=self._normalize_student_input,
                train_loader=dm.val_train_split_loader,
                device=self.device,
                distributed=torch.distributed.is_initialized(),
                max_train_batches=self.dataset_config.get(
                    "knn_max_train_batches", None,
                ),
            )

    def on_validation_epoch_end(self):
        """Aggregate validation metrics."""
        self.status_logging_dict = {}

        cka_stats = getattr(self, "_cka_stats", None)
        if cka_stats:
            cka_values = []
            for idx in sorted(cka_stats.keys()):
                acc = cka_stats[idx]
                if torch.distributed.is_initialized():
                    for v in acc.values():
                        torch.distributed.all_reduce(
                            v, op=torch.distributed.ReduceOp.SUM
                        )
                cka = self._linear_cka_from_stats(acc)
                self.log(
                    f"val_spatial_cka_teacher_{idx}", cka,
                    on_step=False, on_epoch=True, prog_bar=False, sync_dist=False,
                )
                self.status_logging_dict[f"val_spatial_cka_teacher_{idx}"] = cka
                cka_values.append(cka)
            if cka_values:
                mean_cka = sum(cka_values) / len(cka_values)
                self.log(
                    "val_spatial_cka", mean_cka,
                    on_step=False, on_epoch=True, prog_bar=True, sync_dist=False,
                )
                self.status_logging_dict["val_spatial_cka"] = mean_cka
            self._cka_stats = {}

        if self._knn_index is not None and self._knn_total > 0:
            knn_top1 = 100.0 * self._knn_correct.float() / self._knn_total.float()
            self.log(
                "knn_top1", knn_top1.item(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.status_logging_dict["knn_top1"] = knn_top1.item()
            self._knn_index = None
            torch.cuda.empty_cache()

        if not self.trainer.sanity_checking and self.status_logging_dict:
            status_logging.get_status_logger().kpi = self.status_logging_dict
            status_logging.get_status_logger().write(
                message="Eval metrics generated.",
                status_level=status_logging.Status.RUNNING,
            )

        pl.utilities.memory.garbage_collection_cuda()

    def _get_trainer_datamodule(self):
        """Return the attached Lightning datamodule when available."""
        try:
            trainer = self.trainer
        except RuntimeError:
            return None
        return getattr(trainer, "datamodule", None)

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint without the frozen teachers or the duplicated student.

        Each per-teacher ``DistillationLoss`` keeps references to its frozen
        ``teacher_model`` (rebuilt from config on load) and to ``student_model``
        (the same object as ``self.model``, already saved under ``model.*``). Those
        show up as ``distillation_loss_fns.N.{teacher,student}_model.*`` and are
        ~90% of the file (frozen teachers) plus a 6x student duplicate -- strip them
        so checkpoints stay small (avoids the multi-teacher high-res checkpoint-write
        hang). PHI states and projection heads under distillation_loss_fns are kept.
        """
        keys_to_pop = [
            key for key in checkpoint["state_dict"].keys()
            if (
                key.startswith("teacher") or
                key.startswith("teachers") or
                key.startswith("_baseline_anchor_model") or
                ".teacher_model." in key or
                ".student_model." in key
            )
        ]
        for key in keys_to_pop:
            checkpoint["state_dict"].pop(key)
        checkpoint["radio_experiment_spec"] = OmegaConf.to_container(
            self.experiment_spec,
            resolve=True,
        )

        datamodule = self._get_trainer_datamodule()
        loader_state = getattr(datamodule, "loader_state", None) if datamodule is not None else None
        if loader_state is not None:
            checkpoint["radio_loader_state"] = loader_state.state_dict()

        checkpoint["tao_model"] = "classification"

    def on_load_checkpoint(self, checkpoint):
        """Restore checkpointed RADIO dataloader state when present."""
        loader_checkpoint = checkpoint.get("radio_loader_state")
        if loader_checkpoint is None:
            return

        datamodule = self._get_trainer_datamodule()
        loader_state = getattr(datamodule, "loader_state", None) if datamodule is not None else None
        if loader_state is None:
            if get_global_rank() == 0:
                logger.warning("RADIO loader state was found in the checkpoint, but no loader_state is attached.")
            return

        restored = loader_state.restore(loader_checkpoint)
        if restored and get_global_rank() == 0:
            logger.info("Restored RADIO WebDataset loader state from checkpoint.")
