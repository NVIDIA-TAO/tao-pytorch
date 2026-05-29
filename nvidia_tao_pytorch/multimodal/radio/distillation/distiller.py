# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distiller module for RADIO model"""
import os
import logging
import re
import copy
from typing import Sequence
import numpy as np

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn

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

        self._pretrained_head_sd = None
        pretrained_path = getattr(self.model_config.backbone, 'pretrained_backbone_path', None)
        if pretrained_path:
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

        self.register_buffer(
            "_student_mean", torch.tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_student_std", torch.tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)
        )

    def configure_callbacks(self) -> Sequence[Callback] | pl.Callback:
        """Configures logging and checkpoint-saving callbacks"""
        # This is called when trainer.fit() is called
        self.checkpoint_filename = "classifier_model"
        callbacks = []
        results_dir = self.experiment_spec["results_dir"]
        checkpoint_interval = self.experiment_spec["train"]["checkpoint_interval"]

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

        checkpoint_callback = ckpt_func(
            every_n_epochs=checkpoint_interval,
            dirpath=results_dir,
            save_on_train_epoch_end=True,
            monitor=None,
            save_top_k=-1,
            save_last="link",
            filename="model_{epoch:03d}",
            enable_version_counter=False,
        )
        callbacks.append(checkpoint_callback)
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
                config['pretrained_path'] = getattr(teacher, 'pretrained_teacher_model_path', None)
                config['mode'] = getattr(teacher, 'mode', self.distill_config.mode or 'auto')
            else:
                # This is a plain ModelConfig - use global settings
                config['model_config'] = teacher
                config['loss_type'] = self.distill_config.loss_type
                config['loss_lambda'] = self.distill_config.loss_lambda
                config['pretrained_path'] = getattr(self.distill_config, 'pretrained_teacher_model_path', None)
                config['mode'] = self.distill_config.mode or 'auto'

            # Multi-view: per-teacher input_size, match_student_resolution, stochastic_resolutions (EVFM-style)
            config['input_size'] = getattr(teacher, 'input_size', None)
            config['match_student_resolution'] = getattr(teacher, 'match_student_resolution', True)
            config['stochastic_resolutions'] = getattr(teacher, 'stochastic_resolutions', None)
            # Per-teacher image normalization (e.g. [0.5, 0.5, 0.5] for SAM3/SigLIP2; ImageNet for DINOv3)
            _nm = getattr(teacher, 'norm_mean', None)
            _ns = getattr(teacher, 'norm_std', None)
            config['norm_mean'] = list(_nm) if _nm and len(_nm) == 3 else None
            config['norm_std'] = list(_ns) if _ns and len(_ns) == 3 else None
            config['summary_loss_weight'] = getattr(teacher, 'summary_loss_weight', 1.0)
            config['fd_loss_weight'] = getattr(teacher, 'fd_loss_weight', 1.0)
            config['summary_loss_type'] = getattr(teacher, 'summary_loss_type', 'CE')
            config['summary_token_idx'] = getattr(teacher, 'summary_token_idx', None)
            config['spatial_mlp_version'] = getattr(teacher, 'spatial_mlp_version', 'v2')
            config['spatial_num_inner'] = getattr(teacher, 'spatial_num_inner', None)
            config['upstream_name'] = None
            # FeatSharp adaptor
            config['adaptor'] = getattr(teacher, 'adaptor', None)
            config['upsampler_checkpoint'] = getattr(teacher, 'upsampler_checkpoint', None)
            config['do_upsample'] = getattr(teacher, 'do_upsample', True)
            config['featsharp_lib_path'] = getattr(teacher, 'featsharp_lib_path', None)

            parsed_configs.append(config)
            logger.info(f"Teacher {idx}: loss_type={config['loss_type']}, "
                        f"loss_lambda={config['loss_lambda']}, mode={config['mode']}")

        return parsed_configs

    @staticmethod
    def _normalize_input(
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize an image tensor from [0,1] using the given mean/std buffers."""
        return (x - mean) / std

    def _normalize_student_input(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize student input from [0,1] using OPENAI_CLIP statistics."""
        return self._normalize_input(img, self._student_mean, self._student_std)

    def _apply_teacher_normalization(
        self, teacher_input: torch.Tensor, teacher_config: dict, device: torch.device
    ) -> torch.Tensor:
        """Normalize teacher input from [0,1] to per-teacher normalization.

        All dataloaders now output [0,1] images, so normalization is always
        ``(x - mean_t) / std_t``, matching EVFM's ``InputConditioner``.
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
            if teacher_config['pretrained_path'] is not None:
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
                summary_loss_weight=teacher_config.get('summary_loss_weight', 1.0),
                fd_loss_weight=teacher_config.get('fd_loss_weight', 1.0),
                summary_loss_type=teacher_config.get('summary_loss_type', 'CE'),
                summary_token_idx=teacher_config.get('summary_token_idx'),
            )
            self.distillation_loss_fns.append(loss_fn)
            logger.info(f"Created distillation loss for teacher {idx}: "
                        f"type={teacher_config['loss_type']}, mode={teacher_config['mode']}")

        # For backward compatibility, keep single loss reference if only one teacher
        if len(self.distillation_loss_fns) == 1:
            self.distillation_loss_fn = self.distillation_loss_fns[0]

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

    @staticmethod
    def _get_parameter_groups(model, weight_decay, skip_names=()):
        decay = []
        no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if any(s in name for s in skip_names):
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay},
        ]

    def configure_optimizers(self):
        """Configure optimizers for training"""
        parameters = self._get_parameter_groups(
            self.model, self.optimizer.weight_decay, self.optimizer.skip_names
        )
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
        else:
            raise NotImplementedError(
                "Optimizer {} is not implemented".format(self.optimizer.optim)
            )

        # Create main scheduler based on policy
        lr_policy = self.lr_policy.lower()
        if lr_policy == "linear":

            def lambda_rule(epoch):
                # gradually decay learning rate from epoch 0 to max_epochs
                lr_l = 1 - (epoch) / float(self.max_epochs + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule)
        elif lr_policy == "step":
            interval = "epoch"
            if self.lr_policy_params is not None:
                step_size = self.lr_policy_params.step_size
                gamma = self.lr_policy_params.gamma
            else:   # default values
                step_size = self.max_epochs // 4
                gamma = 0.1
            # args.lr_decay_iters
            scheduler = lr_scheduler.StepLR(
                self.optimizer_G, step_size=step_size, gamma=gamma
            )
        elif lr_policy == "multistep":
            interval = "epoch"
            if self.lr_policy_params is not None:
                milestones = self.lr_policy_params.milestones
                gamma = self.lr_policy_params.gamma
            else:
                milestones = [self.max_epochs // 2]
                gamma = 0.1
            scheduler = lr_scheduler.MultiStepLR(self.optimizer_G, milestones, gamma=gamma)
        elif lr_policy == "cosine":
            interval = "step"
            epoch_steps = self.trainer.estimated_stepping_batches // (self.trainer.max_epochs * self.trainer.accumulate_grad_batches)
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
        """Save a batch to disk for parity testing."""
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
            logger = logging.getLogger("parity_batch_dump")
            logger.info("Parity batch dump: saving to %s", dump_dir)

    @staticmethod
    def _dump_forward(dump_dir, batch_idx, forward_data):
        """Save forward-pass tensors (normalized inputs, losses) for parity testing."""
        fwd_dir = os.path.join(dump_dir, "forward")
        os.makedirs(fwd_dir, exist_ok=True)
        path = os.path.join(fwd_dir, f"batch_{batch_idx:03d}.pt")
        torch.save(forward_data, path)
        if batch_idx == 0:
            logger = logging.getLogger("parity_forward_dump")
            logger.info("Parity forward dump: saving to %s", fwd_dir)

    def training_step(self, batch, batch_idx):
        """Training step"""
        _dump_dir = os.environ.get("PARITY_DUMP_DIR")
        _dump_max = int(os.environ.get("PARITY_DUMP_BATCHES", "5"))
        _dumping = _dump_dir and batch_idx < _dump_max
        if _dumping:
            self._dump_batch(_dump_dir, batch_idx, batch)
            _fwd = {}

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

        student_summary, student_spatial = self.model(student_input, return_features=True)
        loss = torch.tensor(0.0).to(student_summary.device)

        # Compute distillation loss from all teachers
        total_distillation_loss = torch.tensor(0.0, device=loss.device if torch.is_tensor(loss) else 'cpu')
        total_teacher_weight = 0.0
        distill_scale = 1.0
        use_multiview = "teacher_views" in batch and len(batch["teacher_views"]) > 0

        for idx, (loss_fn, teacher_config) in enumerate(zip(self.distillation_loss_fns, self.teacher_configs)):
            if use_multiview and idx < len(batch["teacher_views"]):
                tv = batch["teacher_views"][idx]
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
                    student_input,
                    teacher_batch_input=teacher_input,
                    student_valid_mask=student_valid_mask,
                    teacher_valid_mask=teacher_valid_mask,
                    spatial_transform=spatial_transform,
                    student_summary=student_summary,
                    student_spatial=student_spatial,
                )
            else:
                teacher_input = self._apply_teacher_normalization(
                    batch["img"], teacher_config, batch["img"].device
                )
                teacher_distill_loss = loss_fn(
                    student_input,
                    teacher_batch_input=teacher_input,
                    student_summary=student_summary,
                    student_spatial=student_spatial,
                )

            if _dumping:
                _fwd[f"teacher_{idx}_input_normalized"] = teacher_input.detach().cpu()
                _fwd[f"teacher_{idx}_loss"] = teacher_distill_loss.detach().cpu()

            teacher_weight = teacher_config['loss_lambda']
            weighted_loss = teacher_weight * teacher_distill_loss * distill_scale

            if not torch.isnan(weighted_loss):
                total_distillation_loss += weighted_loss
                total_teacher_weight += teacher_weight

            # Log per-teacher loss (unscaled, for readability)
            self.log(
                f"distill_loss_teacher_{idx}",
                teacher_distill_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.batch_size,
                rank_zero_only=True
            )

        # Normalize supervised loss weight
        if total_teacher_weight > 0:
            supervised_weight = 1.0 - total_teacher_weight
        else:
            supervised_weight = 1.0

        supervised_loss = supervised_weight * loss
        distill_loss = total_distillation_loss
        # Unscaled distillation loss (raw value from loss_fn, ~1–3) for logging/prog_bar
        distill_loss_raw = distill_loss / distill_scale if total_teacher_weight > 0 else distill_loss

        if torch.isnan(supervised_loss):
            supervised_loss = torch.tensor(0.0)

        if torch.isnan(distill_loss):
            distill_loss = torch.tensor(0.0)

        total_loss = supervised_loss + distill_loss
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
            self._dump_forward(_dump_dir, batch_idx, _fwd)

        return {"loss": total_loss}

    def on_train_epoch_end(self):
        """Log Training metrics to status.json"""
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

        Always runs standard validation (loss, accuracy, distillation).
        If a KNN index was built in ``on_validation_epoch_start``,
        additionally evaluates per-batch KNN accuracy.
        """
        student_input = self._normalize_student_input(batch["img"])
        student_summary, student_spatial = self.model(student_input, return_features=True)
        out = student_summary
        loss = torch.tensor(0.0).to(out.device)

        # Compute distillation loss from all teachers
        total_distillation_loss = torch.tensor(0.0, device=out.device)
        use_multiview = "teacher_views" in batch and len(batch["teacher_views"]) > 0

        for idx, (loss_fn, teacher_config) in enumerate(zip(self.distillation_loss_fns, self.teacher_configs)):
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

            if not torch.isnan(teacher_distill_loss):
                total_distillation_loss += teacher_distill_loss

            # Log per-teacher validation loss
            self.log(
                f"val_distill_loss_teacher_{idx}",
                teacher_distill_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.batch_size,
                rank_zero_only=True
            )

        # Log total distillation loss
        self.log(
            "distillation_loss",
            total_distillation_loss,
            on_step=True,
            on_epoch=False,
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

    def on_validation_start(self):
        """Skip the parent batch-size check for WDS loaders."""
        if self.trainer.datamodule.val_dataset_type != "WebDataset":
            super().on_validation_start()

    def on_validation_epoch_start(self):
        """Build KNN index from the train split before val batches arrive."""
        dm = self.trainer.datamodule

        self._knn_index = None

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

    def on_save_checkpoint(self, checkpoint):
        """Save the checkpoint but ignore the teacher weights."""
        keys_to_pop = [
            key for key in checkpoint["state_dict"].keys()
            if key.startswith("teacher") or key.startswith("teachers")
        ]
        for key in keys_to_pop:
            checkpoint["state_dict"].pop(key)
        checkpoint["tao_model"] = "classification"
