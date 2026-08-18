# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVDINOv2 Model Module"""
import copy
import os
import re
from typing import Any, Dict, Sequence

import pandas as pd
import torch
import torch._dynamo.config
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
try:
    from torch.distributed.fsdp._runtime_utils import _reshard
except ImportError:  # removed in newer PyTorch; the manual teacher reshard becomes a no-op
    _reshard = None
from torch.distributed.fsdp.wrap import wrap
import pytorch_lightning as pl
from pytorch_lightning.strategies.fsdp import FSDPStrategy
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from xformers.ops.fmha import BlockDiagonalMask
import nvidia_tao_pytorch.config.nvdinov2.default_config as model_params

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.callbacks.loggers import TAOStatusLogger
from nvidia_tao_pytorch.core.callbacks.model_checkpoint import TAOExceptionCheckpoint
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.lightning.tao_lightning_module import TAOLightningModule
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.ssl.nvdinov2.model.head import DinoHead
from nvidia_tao_pytorch.ssl.nvdinov2.model.loss import DinoV2Loss, KoLeoLoss
from nvidia_tao_pytorch.ssl.nvdinov2.model.vit import DinoV2VisionTransformer, SwiGLUFused
from nvidia_tao_pytorch.ssl.nvdinov2.model.warmup_cosine import LambdaWarmUpCosineScheduler

torch._dynamo.config.suppress_errors = True


class CustomModelCheckpoint(ModelCheckpoint):
    """Custom callback for saving NVDINOv2 checkpoint"""

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Saves the model checkpoint, including custom handling for student and teacher states.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning trainer instance, providing access to model and training information.
            filepath (str): The file path where the checkpoint will be saved.
        """
        # Call the original save_checkpoint method to save the checkpoint as usual
        trainer.save_checkpoint(filepath, self.save_weights_only)
        # Custom checkpoint saving with model conversion
        state_dict = trainer.lightning_module.state_dict()

        if trainer.lightning_module.model_config.distill.enable:
            student_state_dict = {}
            student_ema_state_dict = {}
            for k, v in list(state_dict.items()):
                k_save = k
                if "student.backbone." in k:
                    k_save = k.replace("student.backbone.", "")
                elif "student_ema.backbone." in k:
                    k_save = k.replace("student_ema.backbone.", "")
                else:
                    continue

                if re.match(r"dino_head\.", k_save):
                    continue
                if re.match(r"mask_token", k_save):
                    continue

                if "student.backbone." in k:
                    student_state_dict[k_save] = v
                elif "student_ema.backbone." in k:
                    student_ema_state_dict[k_save] = v

            torch.save(student_state_dict, os.path.join(trainer.default_root_dir, f'student_epoch_{trainer.current_epoch:03d}_step_{trainer.global_step:05d}' + self.FILE_EXTENSION))
            torch.save(student_ema_state_dict, os.path.join(trainer.default_root_dir, f'student_ema_epoch_{trainer.current_epoch:03d}_step_{trainer.global_step:05d}' + self.FILE_EXTENSION))

        else:
            student_state_dict = {}
            teacher_state_dict = {}
            for k, v in list(state_dict.items()):
                k_save = k
                if "student.backbone." in k:
                    k_save = k.replace("student.backbone.", "")
                elif "teacher.backbone." in k:
                    k_save = k.replace("teacher.backbone.", "")
                else:
                    continue

                if re.match(r"dino_head\.", k_save):
                    continue
                if re.match(r"mask_token", k_save):
                    continue

                if "student.backbone." in k:
                    student_state_dict[k_save] = v
                elif "teacher.backbone." in k:
                    teacher_state_dict[k_save] = v

            torch.save(student_state_dict, os.path.join(trainer.default_root_dir, f'student_epoch_{trainer.current_epoch:03d}_step_{trainer.global_step:05d}' + self.FILE_EXTENSION))
            torch.save(teacher_state_dict, os.path.join(trainer.default_root_dir, f'teacher_epoch_{trainer.current_epoch:03d}_step_{trainer.global_step:05d}' + self.FILE_EXTENSION))

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # Notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(self)


class DinoV2PlModel(TAOLightningModule):
    """Pytorch Lightning module for NVDINOv2"""

    # Backbone hyper-parameter table (embed_dim/depth/num_heads/...) keyed by backbone type.
    param_map = model_params.map_params

    @staticmethod
    def _validate_backbone_types(backbone_config, param_map):
        """Validate the teacher/student backbone names against the supported architectures.

        ``param_map`` is the subclass's arch table (its keys are the supported architectures),
        so this stays correct for every SSL family (nvdinov2, dinov3, ...) without a second
        source of truth. Raises a clear ValueError listing the allowed values instead of the
        bare KeyError the arch lookups would otherwise raise. See bug 6460904.
        """
        supported = sorted(param_map["depth"].keys())
        for role in ("teacher_type", "student_type"):
            name = backbone_config[role]
            if name not in param_map["depth"]:
                raise ValueError(
                    f"Unsupported model.backbone.{role}: '{name}'. Supported architectures are: "
                    f"{', '.join(supported)}."
                )

    def __init__(self, experiment_spec):
        """Initializes the DinoV2PlModel with the specified experiment configuration.

        Args:
            experiment_spec: The configuration for the experiment, including model architecture, dataset settings, and training parameters.
        """
        super().__init__(experiment_spec)

        # Basic configs
        self.dataset_config = experiment_spec.dataset
        self.train_config = experiment_spec.train
        self.model_config = experiment_spec.model

        # Fail up front on an unsupported backbone name. The packaged JSON-schema enum
        # (STR_FIELD valid_options) is documentation only -- backbone type is an unconstrained
        # str at runtime, so a typo/unsupported arch otherwise dies later with a bare KeyError
        # deep in model init. See bug 6460904.
        self._validate_backbone_types(self.model_config.backbone, self.param_map)

        # Dataset configs
        self.batch_size = self.dataset_config["batch_size"]
        self.n_global_crops = self.dataset_config.transform["n_global_crops"]
        self.n_local_crops = self.dataset_config.transform["n_local_crops"]

        # Train configs
        self.layerwise_decay = self.train_config["layerwise_decay"]
        self.clip_grad_norm = self.train_config["clip_grad_norm"]
        self.num_prototypes = self.train_config["num_prototypes"]
        self.num_gpus = max(self.train_config["num_gpus"], len(self.train_config["gpu_ids"]))
        if self._custom_attention_supported():
            self.use_custom_attention = self.train_config["use_custom_attention"]
            logging.info("Using flash attention if set by user")
        else:
            self.use_custom_attention = False
            props = torch.cuda.get_device_properties(0)
            if (props.major, props.minor) >= (10, 0):
                logging.info(
                    "Disabling xformers custom attention on Blackwell (SM%d%d): FA3 is not yet "
                    "supported; using the SDPA fallback attention.", props.major, props.minor
                )
            else:
                logging.info(
                    "Disabling xformers custom attention on Hopper (SM%d%d): memory_efficient_attention "
                    "fails to launch with this xformers build (bug 6459926); using the SDPA fallback "
                    "attention.", props.major, props.minor
                )
        # Teacher Backbone
        self.teacher_backbone_type = self.model_config.backbone['teacher_type']
        self.teacher_depth = self.param_map['depth'][self.teacher_backbone_type]
        self.teacher_num_heads = self.param_map['num_heads'][self.teacher_backbone_type]
        self.teacher_init_values = self.param_map['init_values'][self.teacher_backbone_type]
        self.teacher_drop_path_schedule = self.param_map['drop_path_schedule'][self.teacher_backbone_type]
        self.teacher_num_classes = self.param_map['num_classes'][self.teacher_backbone_type]
        self.teacher_embed_dim = self.param_map['embed_dim'][self.teacher_backbone_type]  # self.teacher_embed_dim should be equal to self.student_embed_dim
        # Student Backbone
        self.student_backbone_type = self.model_config.backbone['student_type']
        self.student_depth = self.param_map['depth'][self.student_backbone_type]
        self.student_num_heads = self.param_map['num_heads'][self.student_backbone_type]
        self.student_init_values = self.param_map['init_values'][self.student_backbone_type]
        self.student_drop_path_schedule = self.param_map['drop_path_schedule'][self.student_backbone_type]
        self.student_num_classes = self.param_map['num_classes'][self.student_backbone_type]
        self.student_embed_dim = self.param_map['embed_dim'][self.student_backbone_type]

        self.patch_size = self.model_config.backbone['patch_size']
        self.img_size = self.model_config.backbone['img_size']
        self.register_tokens = self.model_config.backbone['num_register_tokens']
        self.drop_path_rate = self.model_config.backbone['drop_path_rate']

        # Heads
        self.head_layers = self.model_config.head['num_layers']
        self.hidden_dim = self.model_config.head['hidden_dim']
        self.bottleneck_dim = self.model_config.head['bottleneck_dim']

        # Losses
        self.dino_cls_token_loss_weight = 1.0
        self.dino_cls_token_loss = DinoV2Loss(
            num_prototypes=self.num_prototypes,
            centering_method="softmax"
        )

        self.koleo_loss_weight = 0.1
        self.koleo_loss = KoLeoLoss()

        self.ibot_separate_head = True
        self.ibot_patch_tokens_loss_weight = 1.0
        self.ibot_patch_tokens_loss = DinoV2Loss(
            num_prototypes=self.num_prototypes,
            centering_method="softmax"
        )

        # Build model
        self._build_model()

        # Optimizer
        if self.train_config.optim["optim"] == "adamw":
            self.optimizer_builder = optim.AdamW

        # Schedulers
        self.schedulers = {
            'learning_rate': LambdaWarmUpCosineScheduler(
                val_base=self.train_config.schedulers.learning_rate["val_base"],
                val_final=self.train_config.schedulers.learning_rate["val_final"],
                val_start=self.train_config.schedulers.learning_rate["val_start"],
                warm_up_steps=self.train_config.schedulers.learning_rate["warm_up_steps"],
                max_decay_steps=self.train_config.schedulers.learning_rate["max_decay_steps"],
            ),
            'last_layer_learning_rate': LambdaWarmUpCosineScheduler(
                val_base=self.train_config.schedulers.last_layer_learning_rate["val_base"],
                val_final=self.train_config.schedulers.last_layer_learning_rate["val_final"],
                val_start=self.train_config.schedulers.last_layer_learning_rate["val_start"],
                warm_up_steps=self.train_config.schedulers.last_layer_learning_rate["warm_up_steps"],
                max_decay_steps=self.train_config.schedulers.last_layer_learning_rate["max_decay_steps"],
                freeze_steps=self.train_config.schedulers.last_layer_learning_rate["freeze_steps"],
            ),
            'weight_decay': LambdaWarmUpCosineScheduler(
                val_base=self.train_config.schedulers.weight_decay["val_base"],
                val_final=self.train_config.schedulers.weight_decay["val_final"],
                val_start=self.train_config.schedulers.weight_decay["val_start"],
                warm_up_steps=self.train_config.schedulers.weight_decay["warm_up_steps"],
                max_decay_steps=self.train_config.schedulers.weight_decay["max_decay_steps"],
            ),
            'momentum': LambdaWarmUpCosineScheduler(
                val_base=self.train_config.schedulers.momentum["val_base"],
                val_final=self.train_config.schedulers.momentum["val_final"],
                val_start=self.train_config.schedulers.momentum["val_start"],
                warm_up_steps=self.train_config.schedulers.momentum["warm_up_steps"],
                max_decay_steps=self.train_config.schedulers.momentum["max_decay_steps"],
            ),
            'teacher_temperature': LambdaWarmUpCosineScheduler(
                val_base=self.train_config.schedulers.teacher_temperature["val_base"],
                val_final=self.train_config.schedulers.teacher_temperature["val_final"],
                val_start=self.train_config.schedulers.teacher_temperature["val_start"],
                warm_up_steps=self.train_config.schedulers.teacher_temperature["warm_up_steps"],
                max_decay_steps=self.train_config.schedulers.teacher_temperature["max_decay_steps"],
            ),
        }

        # Disable automatic optimization
        self.automatic_optimization = False
        self.need_to_synchronize_streams = True

        # Sync teacher weight
        if self.model_config.distill.enable:
            pass  # teacher will be loaded with a pretrained checkpoint when distillation
        else:
            self.teacher.load_state_dict(self.student.state_dict(), strict=False)

        # Disable teacher gradients
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Disable student_ema gradients
        if self.model_config.distill.enable:
            for param in self.student_ema.parameters():
                param.requires_grad = False
            # Disable gradients for mask_token
            self.student.backbone.mask_token.requires_grad = False

        self.checkpoint_filename = 'nvdinov2_model'
        self.dm = []

    def _custom_attention_supported(self):
        """Whether the xformers custom-attention path is usable on the current GPU.

        The xformers ``memory_efficient_attention`` kernel fails to launch on Hopper (SM90A,
        compute capability ``(9, x)``) with ``cudaErrorLaunchFailure`` (bug 6459926), and FA3
        is not yet supported on Blackwell (``>= (10, 0)``). So the custom path is enabled only
        on pre-Hopper GPUs (Ampere and older, ``< (9, 0)``); Hopper and newer fall back to the
        shared SDPA path (``MemoryEfficientAttention._fallback_attention``), which is
        numerically stable and launches on those archs. This is deliberately an
        xformers-build-scoped arch hardcode - the true gate is whether this xformers build
        launches ``memory_efficient_attention`` on the arch - and is the pragmatic P0 unblock.
        """
        if not torch.cuda.is_available():
            # No device to probe (e.g. CPU-only construction); honor the configured flag.
            return True
        props = torch.cuda.get_device_properties(0)
        return (props.major, props.minor) < (9, 0)

    def _build_model(self):
        """Build Teacher and Student"""
        self.student = torch.nn.ModuleDict(
            {
                'backbone': DinoV2VisionTransformer(
                    img_size=self.img_size,
                    patch_size=self.patch_size,
                    embed_dim=self.student_embed_dim,
                    depth=self.student_depth,
                    num_heads=self.student_num_heads,
                    init_values=self.student_init_values,
                    drop_path_schedule=self.student_drop_path_schedule,
                    num_classes=self.student_num_classes,
                    drop_path_rate=self.drop_path_rate,
                    mlp_layer=SwiGLUFused,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.SiLU,
                    register_tokens=self.register_tokens,
                    use_custom_attention=self.use_custom_attention
                ),
                'dino_head': DinoHead(
                    in_dim=self.student_embed_dim,
                    out_dim=self.num_prototypes,
                    num_layers=self.head_layers,
                    hidden_dim=self.hidden_dim,
                    bottleneck_dim=self.bottleneck_dim
                ),
                'ibot_head': DinoHead(
                    in_dim=self.student_embed_dim,
                    out_dim=self.num_prototypes,
                    num_layers=self.head_layers,
                    hidden_dim=self.hidden_dim,
                    bottleneck_dim=self.bottleneck_dim
                )
            }
        )
        self.teacher = torch.nn.ModuleDict(
            {
                'backbone': DinoV2VisionTransformer(
                    img_size=self.img_size,
                    patch_size=self.patch_size,
                    embed_dim=self.teacher_embed_dim,
                    depth=self.teacher_depth,
                    num_heads=self.teacher_num_heads,
                    init_values=self.teacher_init_values,
                    drop_path_schedule=self.teacher_drop_path_schedule,
                    num_classes=self.teacher_num_classes,
                    mlp_layer=SwiGLUFused,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.SiLU,
                    register_tokens=self.register_tokens,
                    use_custom_attention=self.use_custom_attention
                ),
                'dino_head': DinoHead(
                    in_dim=self.teacher_embed_dim,
                    out_dim=self.num_prototypes,
                    num_layers=self.head_layers,
                    hidden_dim=self.hidden_dim,
                    bottleneck_dim=self.bottleneck_dim
                ),
                'ibot_head': DinoHead(
                    in_dim=self.teacher_embed_dim,
                    out_dim=self.num_prototypes,
                    num_layers=self.head_layers,
                    hidden_dim=self.hidden_dim,
                    bottleneck_dim=self.bottleneck_dim
                )
            }
        )
        if self.model_config.distill.enable:
            # Create a student ema for distillation
            self.student_ema = copy.deepcopy(self.student)

            # Strictly load teacher (backbone + head) forzen weights from full pl checkpoint
            assert self.model_config.distill.pretrained_non_distill_pl_model_path is not None, (
                "In distillation mode, you need to provide the pretrained_non_distill_pl_model_path to initialize a frozen teacher."
            )
            pretrained_backbone_head_state_dict = torch.load(self.model_config.distill.pretrained_non_distill_pl_model_path, map_location="cpu")['state_dict']
            teacher_state_dict = {}
            for k, v in list(pretrained_backbone_head_state_dict.items()):
                k_save = k
                if "teacher." in k:
                    k_save = k.replace("teacher.", "")
                    teacher_state_dict[k_save] = v

            self.teacher.load_state_dict(teacher_state_dict)
        else:
            assert self.student_backbone_type == self.teacher_backbone_type, (
                f"In non-distillation mode, student_type and teacher_type should be the same. "
                f"Currently, the teacher_type is {self.teacher_backbone_type}, and the student_type is {self.student_backbone_type}."
            )

    def restore_pretrained_weights(self):
        """Load pretrained weight"""
        cur_student_backbone_weights = self.student.backbone.state_dict()
        pretrained_state_dict = torch.load(self.pretrained_weights, map_location="cpu")

        weights_not_loaded = []
        unexpected_keys = list(pretrained_state_dict.keys())
        for key, weight in cur_student_backbone_weights.items():
            if key in pretrained_state_dict and (pretrained_state_dict[key].shape == weight.shape):
                cur_student_backbone_weights[key] = pretrained_state_dict[key]
                unexpected_keys.remove(key)
            else:
                weights_not_loaded.append(key)
        if weights_not_loaded:
            for item in weights_not_loaded:
                if get_global_rank() == 0:
                    logging.info(f"Weights not loaded for {item}")
        if unexpected_keys:
            if get_global_rank() == 0:
                logging.info(f"Unexpected keys: {unexpected_keys}")

        self.student.backbone.load_state_dict(cur_student_backbone_weights)
        if not self.model_config.distill.enable:
            self.teacher.load_state_dict(self.student.state_dict(), strict=False)

    def configure_model(self):
        """Warp models"""
        self.teacher = nn.ModuleDict({k: wrap(v) for k, v in self.teacher.items()})
        self.student = nn.ModuleDict({k: wrap(v) for k, v in self.student.items()})
        if self.model_config.distill.enable:
            self.student_ema = nn.ModuleDict({k: wrap(v) for k, v in self.student_ema.items()})

    def update_param_groups(self, optimizer, schedules: Dict[str, float]):
        """Update parameters with schedulers

        Args:
            optimizer (torch.optim.Optimizer): The optimizer instance used for updating model parameters.
            schedules (Dict[str, float]): A dictionary containing schedules.
        """
        for param_group in optimizer.param_groups:
            param_group["weight_decay"] = (
                schedules["weight_decay"] * param_group["wd_multiplier"]
            )

            if param_group["is_last_layer"]:
                param_group["lr"] = (
                    schedules["last_layer_learning_rate"] * param_group["lr_multiplier"]
                )
            else:
                param_group["lr"] = (
                    schedules["learning_rate"] * param_group["lr_multiplier"]
                )

    @torch.no_grad()
    def teacher_forward(
        self, *, global_crops, global_masks_indices, teacher_temperature
    ):
        """Forward pass for the teacher model without gradient computation.

        Args:
            global_crops (torch.Tensor): The input images or patches passed to the teacher model.
            global_masks_indices (torch.Tensor): Indices for the masked positions of the global patch tokens.
            teacher_temperature (float): The temperature parameter used for centering the tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
                - teacher_dino_centered (torch.Tensor): Centered class tokens after passing through the DINO head.
                - teacher_ibot_centered (torch.Tensor): Centered patch tokens after passing through the iBOT head.
                - teacher_backbone_global_output (Dict[str, torch.Tensor]): The output of the teacher model's backbone.
        """
        teacher_backbone_global_output = self.teacher.backbone(global_crops)

        # The following code switch the pair from (0, 1) to (1, 0)
        # For example, if the batch size is 32, then the 0 and 16 are from same image,
        # 1 and 17 are from same image, etc.

        teacher_global_cls_token = teacher_backbone_global_output["x_norm_clstoken"]
        teacher_global_cls_token = teacher_global_cls_token.chunk(self.n_global_crops)
        teacher_global_cls_token = torch.cat(
            (teacher_global_cls_token[1], teacher_global_cls_token[0])
        )
        n_cls_tokens = teacher_global_cls_token.shape[0]

        # Get the global patch tokens at masked positions
        teacher_global_patch_tokens = teacher_backbone_global_output[
            "x_norm_patchtokens"
        ].flatten(0, 1)[global_masks_indices]

        # If ibot_separate_head is False, we use the same head for both DINO and IBOT
        if self.ibot_separate_head is False:
            global_tokens_after_head = self.teacher.dino_head(
                torch.cat(
                    (teacher_global_cls_token, teacher_global_patch_tokens), dim=0
                )
            )
            teacher_cls_tokens_after_head = global_tokens_after_head[:n_cls_tokens]
            teacher_patch_tokens_after_head = global_tokens_after_head[n_cls_tokens:]
        else:
            teacher_cls_tokens_after_head = self.teacher.dino_head(
                teacher_global_cls_token
            )
            teacher_patch_tokens_after_head = self.teacher.ibot_head(
                teacher_global_patch_tokens
            )

        # Now we need to determine which centering method to use
        teacher_dino_centered = self.dino_cls_token_loss.centering(
            teacher_cls_tokens_after_head, teacher_temperature
        ).view(self.n_global_crops, -1, teacher_cls_tokens_after_head.shape[-1])
        teacher_ibot_centered = self.ibot_patch_tokens_loss.centering(
            teacher_patch_tokens_after_head, teacher_temperature
        )
        # teacher_dino_centered -> (2, 16, 65536), teacher_ibot_centered -> (926, 65536)

        return (
            teacher_dino_centered,
            teacher_ibot_centered,
            teacher_backbone_global_output,
        )

    def _extra_losses(self, **ctx):
        """Hook for subclasses to inject additional loss terms into ``student_forward``.

        The base (DINOv2) implementation returns an empty list, so the total loss is
        identical to the original. Subclasses (e.g. DINOv3 Gram anchoring) override this
        to return a list of extra loss tensors that get summed with the DINO/iBOT/KoLeo
        terms. The keyword context exposes the student backbone outputs (which carry the
        patch tokens and masks) and the raw crops/masks so an extra term can run its own
        teacher and build whatever it needs.

        Args:
            **ctx: Context forwarded from ``student_forward`` (global/local crops,
                global masks/indices/weights, and the student global/local backbone
                outputs).

        Returns:
            list: Extra loss tensors to add. Empty for DINOv2.
        """
        return []

    def student_forward(
        self,
        *,
        global_crops,
        global_masks,
        global_masks_indices,
        global_masks_weight,
        local_crops,
        teacher_dino_centered,
        teacher_ibot_centered,
    ):
        """Forward pass for the student model.

        Args:
            global_crops (torch.Tensor): Input images for global crops.
            global_masks (torch.Tensor): Masked positions for global crops.
            global_masks_indices (torch.Tensor): Indices for masked global patch tokens.
            global_masks_weight (torch.Tensor): Weights for masked global crops.
            local_crops (torch.Tensor): Input images for local crops.
            teacher_dino_centered (torch.Tensor): Centered class tokens from the teacher's DINO head.
            teacher_ibot_centered (torch.Tensor): Centered patch tokens from the teacher's iBOT head.


        Returns:
            torch.Tensor: Total computed loss for the student model.
        """
        # Init vars
        n_local_crops_loss_terms = self.n_local_crops * self.n_global_crops
        n_global_crops_loss_terms = self.n_global_crops * (self.n_global_crops - 1)

        # Now we need to process student
        if self.model_config.distill.enable and self.model_config.distill.disable_masking:
            (
                student_backbone_global_output,
                student_backbone_local_output,
            ) = self.student.backbone(
                [global_crops, local_crops], masks=[None, None]
            )
        else:
            (
                student_backbone_global_output,
                student_backbone_local_output,
            ) = self.student.backbone(
                [global_crops, local_crops], masks=[global_masks, None]
            )

        # Student local crops cls tokens, global crops cls tokens, and global crops patch tokens
        inputs_for_student_head_list = [
            student_backbone_local_output["x_norm_clstoken"][None],
            student_backbone_global_output["x_norm_clstoken"][None],
        ]

        student_global_patch_tokens = student_backbone_global_output[
            "x_norm_patchtokens"
        ].flatten(0, 1)[global_masks_indices]

        if self.ibot_separate_head is False:
            inputs_for_student_head_list.append(student_global_patch_tokens[None])
        else:
            student_global_patch_tokens_after_head = self.student.ibot_head(
                student_global_patch_tokens
            )

        # Student forward
        attn_bias, cat_inputs = BlockDiagonalMask.from_tensor_list(
            inputs_for_student_head_list
        )
        outputs_list = attn_bias.split(self.student.dino_head(cat_inputs))

        # Extract
        student_local_cls_tokens_after_head = outputs_list[0].squeeze(0)
        student_global_cls_tokens_after_head = outputs_list[1].squeeze(0)

        if self.ibot_separate_head is False:
            student_global_patch_tokens_after_head = outputs_list[2].squeeze(0)

        # Begin calculating losses
        losses = []

        # Calculate local crops loss
        dino_local_loss = self.dino_cls_token_loss(
            student_local_cls_tokens_after_head.chunk(self.n_local_crops),
            teacher_dino_centered,
        ) / (n_local_crops_loss_terms + n_global_crops_loss_terms)

        losses.append(dino_local_loss * self.dino_cls_token_loss_weight)
        self.log(
            "losses/dino_local_loss",
            dino_local_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )

        # Calculate global crops loss
        dino_global_loss = (
            self.dino_cls_token_loss(
                [student_global_cls_tokens_after_head],
                [teacher_dino_centered.flatten(0, 1)],
            ) /
            (n_local_crops_loss_terms + n_global_crops_loss_terms) *
            self.n_global_crops
        )

        losses.append(dino_global_loss * self.dino_cls_token_loss_weight)
        self.log(
            "losses/dino_global_loss",
            dino_global_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )

        # Calculate koleo loss
        koleo_loss = self.koleo_loss_weight * sum(
            self.koleo_loss(i)
            for i in student_backbone_global_output["x_norm_clstoken"].chunk(
                self.n_global_crops
            )
        )

        losses.append(koleo_loss)

        self.log(
            "losses/koleo_loss",
            koleo_loss / self.n_global_crops,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )

        # Calculate ibot loss
        ibot_loss = self.ibot_patch_tokens_loss.forward_masked(
            student_global_patch_tokens_after_head,
            teacher_ibot_centered,
            student_masks_flat=global_masks,
            n_masked_patches=torch.sum(global_masks).item(),
            masks_weight=global_masks_weight,
        )  # * (1.0 / self.n_global_crops) * self.n_global_crops (basically 1)

        losses.append(ibot_loss * self.ibot_patch_tokens_loss_weight)

        self.log(
            "losses/ibot_loss",
            ibot_loss / self.n_global_crops,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )

        # Subclass-injected extra loss terms (e.g. DINOv3 Gram anchoring).
        # Base implementation returns [], so DINOv2 numerics are unchanged.
        losses += self._extra_losses(
            global_crops=global_crops,
            local_crops=local_crops,
            global_masks=global_masks,
            global_masks_indices=global_masks_indices,
            global_masks_weight=global_masks_weight,
            student_backbone_global_output=student_backbone_global_output,
            student_backbone_local_output=student_backbone_local_output,
        )

        # Calculate final loss
        loss = sum(losses)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size,)

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        """Performs a training step for the model.

        Args:
            batch (Any): Input data for training.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Computed loss for the current training step.
        """
        # Teacher need to be in eval mode
        self.teacher.eval()
        if self.model_config.distill.enable:
            self.student_ema.eval()
            # self.student.backbone.mask_token.eval()
        optimizer = self.optimizers()
        schedules = {k: v(self.global_step) for k, v in self.schedulers.items()}

        # Log schedules
        for k, v in schedules.items():
            self.log(f"schedules/{k}", v, on_step=True, on_epoch=False, prog_bar=False, batch_size=self.batch_size,)

        # Get batch
        global_crops = batch["global_crops"]  # (N, C, H, W)
        local_crops = batch["local_crops"]  # (N, C, H, W)
        # assert 1==2, (global_crops.shape, local_crops.shape)
        global_masks = batch[
            "global_masks"
        ]  # (N, H // patch_size * W // patch_size) e.g. (32, 196)
        global_masks_indices = batch[
            "global_masks_indices"
        ]  # (N * H // patch_size * W // patch_size) e.g. (948,)
        global_masks_weight = batch[
            "global_masks_weight"
        ]  # (N * H // patch_size * W // patch_size) e.g. (948,)
        # Teacher forward
        teacher_dino_centered, teacher_ibot_centered, _ = self.teacher_forward(
            global_crops=global_crops,
            global_masks_indices=global_masks_indices,
            teacher_temperature=schedules["teacher_temperature"],
        )

        # Free the teacher's full params gathered during the teacher forward to trim peak
        # memory. FULL_SHARD already reshards automatically in the FSDP post-forward hook, so
        # this is only a best-effort optimization. The PyTorch <=2.1 manual-reshard internals
        # (``m._handles`` / ``_runtime_utils._reshard``) were removed in newer PyTorch, so guard
        # on their presence and otherwise rely on the automatic resharding.
        for m in FullyShardedDataParallel.fsdp_modules(self.teacher):
            if isinstance(m, FullyShardedDataParallel) is False or \
                    m.sharding_strategy == ShardingStrategy.NO_SHARD:
                continue

            handles = getattr(m, "_handles", None)
            if _reshard is not None and handles:
                _reshard(m, handles, [True] * len(handles))

        loss = self.student_forward(
            global_crops=global_crops,
            global_masks=global_masks,
            global_masks_indices=global_masks_indices,
            global_masks_weight=global_masks_weight,
            local_crops=local_crops,
            teacher_dino_centered=teacher_dino_centered,
            teacher_ibot_centered=teacher_ibot_centered,
        )
        self.manual_backward(loss)

        # Gradient clipping, we need to do this manually, reference:
        # https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
        for k, v in self.student.items():
            if isinstance(v, FullyShardedDataParallel):
                v.clip_grad_norm_(self.clip_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(v.parameters(), self.clip_grad_norm)

        self.update_param_groups(optimizer, schedules)
        optimizer.step()
        optimizer.zero_grad()

        # EMA to update teacher
        self.update_teacher(schedules["momentum"])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return loss

    @staticmethod
    def _assert_train_loss_finite(model):
        """Raise if the epoch train loss is non-finite.

        ``train_loss`` is the sole in-loop AutoML monitoring KPI (minimize), so a NaN/Inf loss
        must abort the run rather than be silently reported as PASS -- otherwise HPO and
        loss-based milestone selection are poisoned. Kept as a static helper (taking the model
        explicitly) so it is unit-testable without a full Lightning trainer. A common cause is
        fp16 (16-mixed) attention overflow on the Blackwell fallback path; use bf16-mixed or a
        build with the fp32 fallback-attention fix. See bug 6460915.
        """
        train_loss = model.trainer.logged_metrics.get("train_loss_epoch")
        if train_loss is not None and not torch.isfinite(train_loss).all():
            raise ValueError(
                f"Training loss is non-finite (train_loss_epoch={train_loss}); aborting. A "
                "NaN/Inf loss must not be reported as a successful run."
            )

    def on_train_epoch_end(self):
        """Log Training metrics to status.json (and fail loudly on a non-finite loss)."""
        # Check finiteness first so a NaN/Inf run raises before we emit a RUNNING/nan status
        # line -- the last status the run writes should be the FAILURE, not RUNNING. See 6460915.
        self._assert_train_loss_finite(self)

        average_train_loss = self.trainer.logged_metrics["train_loss_epoch"].item()

        self.status_logging_dict = {}
        self.status_logging_dict["train_loss"] = average_train_loss

        status_logging.get_status_logger().kpi = self.status_logging_dict
        status_logging.get_status_logger().write(
            message="Train metrics generated.",
            status_level=status_logging.Status.RUNNING
        )

    def on_predict_epoch_start(self):
        """Predict epoch start"""
        self.feat = []
        self.input_path = []

    def predict_step(self, batch, batch_idx):
        """Predict step. Inference """
        images = batch["images"]
        input_path = batch["input_path"]
        teacher_backbone_global_output = self.teacher.backbone(images)
        teacher_global_cls_token = teacher_backbone_global_output["x_norm_clstoken"]
        if batch_idx == 0:
            self.feat = teacher_global_cls_token
        else:
            self.feat = torch.cat((self.feat, teacher_global_cls_token), 0)
        self.input_path.extend(input_path)

    @staticmethod
    def _collate_predictions(gathered_feat, gathered_paths, distributed):
        """Assemble deduped ``(input_path, features)`` rows for the inference CSV.

        Args:
            gathered_feat: In the distributed case the ``all_gather`` output shaped
                ``[world, N_local, D]``; in the single-device case the local features ``[N, D]``.
            gathered_paths: In the distributed case a list-of-lists (one path list per rank,
                from ``all_gather_object``); in the single-device case the local path list.
            distributed (bool): Whether ``gathered_*`` carry the extra per-rank leading axis.

        Returns:
            list[dict]: One ``{"input_path", "features"}`` row per unique image.

        The multi-GPU path previously (a) never flattened the ``[world, N_local, D]`` feature
        tensor -- so iterating it yielded one row *per rank* instead of per image -- and (b)
        called ``all_gather`` on a ``list[str]``, which is a silent no-op (only tensors are
        gathered), leaving paths rank-local. The mismatched column lengths then raised on rank 0
        inside this DDP hook, which manifested as a hang (one GPU pinned) with no CSV written.
        See bug 6469109.
        """
        if distributed:
            gathered_feat = gathered_feat.reshape(-1, gathered_feat.shape[-1])
            gathered_paths = [path for rank_paths in gathered_paths for path in rank_paths]
        if gathered_feat.dim() == 1:
            gathered_feat = gathered_feat.unsqueeze(dim=0)

        features = [str(row.cpu().numpy().tolist()) for row in gathered_feat]

        # The predict DistributedSampler pads an uneven split by repeating whole samples; drop
        # those duplicates so there is exactly one row per input image. Precondition: input paths
        # are unique -- the dataset enumerates each file once (rglob), so a repeated path can only
        # come from the sampler's padding, making the path a safe dedup key.
        seen = set()
        rows = []
        for path, feature in zip(gathered_paths, features):
            if path in seen:
                continue
            seen.add(path)
            rows.append({"input_path": path, "features": feature})
        return rows

    @staticmethod
    def _all_gather_predictions(feat, input_path, world_size):
        """All-gather per-rank features (tensor) and paths (list[str]) across the default PG.

        Kept ``torch.distributed``-based (rather than Lightning's ``self.all_gather``) so the exact
        collective sequence that fixes the multi-GPU hang is unit-testable with a gloo process
        group. Every rank must call this -- the collectives are the sync point, and a rank-0-only
        path here is what deadlocked inference.

        Returns:
            tuple: ``(gathered_feat [world, N_local, D], paths_per_rank list[list[str]])``.

        See bug 6469109.
        """
        gathered_feat = [torch.empty_like(feat) for _ in range(world_size)]
        dist.all_gather(gathered_feat, feat)
        gathered_feat = torch.stack(gathered_feat, dim=0)
        paths_per_rank = [None] * world_size
        dist.all_gather_object(paths_per_rank, input_path)
        return gathered_feat, paths_per_rank

    def on_predict_epoch_end(self):
        """Predict epoch end: gather per-rank features/paths and write inference.csv on rank 0."""
        # Every rank must invoke the collectives symmetrically (a rank-0-only path here is what
        # deadlocked multi-GPU inference). all_gather handles the equal-shape feature tensors;
        # string paths are not tensors, so use the object collective for them.
        if self.trainer.world_size > 1:
            gathered_feat, gathered_paths = self._all_gather_predictions(
                self.feat, self.input_path, self.trainer.world_size
            )
            distributed = True
        else:
            gathered_feat = self.feat
            gathered_paths = self.input_path
            distributed = False

        if self.trainer.is_global_zero:
            rows = self._collate_predictions(gathered_feat, gathered_paths, distributed)
            df = pd.DataFrame(rows, columns=["input_path", "features"])
            df.to_csv(
                os.path.join(self.experiment_spec.results_dir, "inference.csv"),
                header=True,
                index=False
            )
            status_logging.get_status_logger().write(
                message="Inference completed.",
                status_level=status_logging.Status.RUNNING
            )

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """Updates the teacher model using Exponential Moving Averages (EMA).

        Args:
            momentum (float): Momentum factor for the EMA update.
        """
        if self.model_config.distill.enable:
            teacher = self.student_ema
            student = self.student
        else:
            teacher = self.teacher
            student = self.student
        # wait for all to sync up before moving on
        if not isinstance(self.trainer.strategy, SingleDeviceStrategy):
            torch.cuda.synchronize()
            dist.barrier()

        teacher_params_list = []
        student_params_list = []

        if isinstance(self.trainer.strategy, FSDPStrategy):
            # Under FSDP (FULL_SHARD) each wrapped module's ``.parameters()`` yields the *local*
            # sharded flat-params. The student and teacher ModuleDicts are wrapped identically,
            # so their shards line up index-for-index and the element-wise EMA below is
            # equivalent to a full-tensor update with no all-gather. (The previous path read
            # ``fsdp_modules(...).params``, a PyTorch <=2.1 internal removed in newer PyTorch.)
            for key in student.keys():
                for student_param, teacher_param in zip(
                    student[key].parameters(), teacher[key].parameters()
                ):
                    teacher_params_list.append(teacher_param.data)
                    student_params_list.append(student_param.data)
        else:
            for teacher_param, student_param in zip(
                teacher.parameters(), student.parameters()
            ):
                teacher_params_list.append(teacher_param.data)
                student_params_list.append(student_param.data)

        # Update teacher
        torch._foreach_mul_(teacher_params_list, momentum)
        torch._foreach_add_(
            teacher_params_list, student_params_list, alpha=1.0 - momentum
        )

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        # Build optimizer params
        params_groups = []
        seen_params = {}

        n_blocks = self.student.backbone.n_blocks

        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue

            name = (
                name.replace("module.", "")
                .replace("_fsdp_wrapped__flat_param", "")
                .replace("_fsdp_wrapped_", "")
            )

            if id(param) in seen_params:
                assert (
                    seen_params[id(param)] == name
                ), f"Param {name} is duplicated under different names"
                continue

            seen_params[id(param)] = name

            # Begin calculate vit params
            layer_id = n_blocks + 1

            if "backbone." in name:
                if ".pos_embed" in name or ".patch_embed" in name or ".mask_token" in name or ".cls_token" in name:
                    layer_id = 0
                elif ".blocks." in name:
                    after_block = name.split(".blocks.")[-1].split(".")
                    try:
                        layer_id = int(after_block[1]) + 1
                    except ValueError:
                        layer_id = int(after_block[0]) + 1

            lr_multiplier = self.layerwise_decay ** (n_blocks + 1 - layer_id)

            if "patch_embed" in name:
                lr_multiplier *= (
                    0.2  # lower the learning rate of the patch embedding layer
                )

            params_groups.append(
                dict(
                    params=param,
                    is_last_layer="last_layer" in name,
                    wd_multiplier=0.0
                    if "bias" in name or "norm" in name or "gamma" in name
                    else 1.0,
                    lr_multiplier=lr_multiplier,
                )
            )

        # we need to fuse the param groups
        fused_params_groups = {}

        for params_group in params_groups:
            key = (
                params_group["is_last_layer"],
                params_group["wd_multiplier"],
                params_group["lr_multiplier"],
            )

            if key not in fused_params_groups:
                fused_params_groups[key] = {
                    "params": [],
                    "is_last_layer": params_group["is_last_layer"],
                    "wd_multiplier": params_group["wd_multiplier"],
                    "lr_multiplier": params_group["lr_multiplier"],
                }

            fused_params_groups[key]["params"].append(params_group["params"])

        return self.optimizer_builder(params=list(fused_params_groups.values()))

    def on_after_backward(self) -> None:
        """Synchronize streams if needed"""
        if self.need_to_synchronize_streams:
            torch.cuda.synchronize()

            # check if fsdp
            if getattr(self.student.dino_head, "_streams", None) is not None:
                self.log("Synchronizing streams")
                if get_global_rank() == 0:
                    logging.info("Synchronizing streams")
                self.student.dino_head._streams = (
                    self.teacher.dino_head._streams
                ) = self.student.backbone._streams = self.teacher.backbone._streams

            self.need_to_synchronize_streams = False

    def configure_callbacks(self) -> Sequence[Callback] | pl.Callback:
        """Configures logging and checkpoint-saving callbacks.
        This is called when trainer.fit() is called

        Returns:
            Sequence[Callback] | pl.Callback: List of configured callbacks.
        """
        results_dir = self.experiment_spec["results_dir"]
        checkpoint_interval = self.experiment_spec["train"]["checkpoint_interval"]

        status_logger_callback = TAOStatusLogger(
            results_dir,
            append=True,
        )

        CustomModelCheckpoint.FILE_EXTENSION = ".pth"
        CustomModelCheckpoint.CHECKPOINT_EQUALS_CHAR = "_"

        if not self.checkpoint_filename:
            raise NotImplementedError("checkpoint_filename not set in __init__() of model")
        CustomModelCheckpoint.CHECKPOINT_NAME_LAST = f"{self.checkpoint_filename}_latest"

        checkpoint_callback = CustomModelCheckpoint(every_n_epochs=checkpoint_interval,
                                                    dirpath=results_dir,
                                                    save_on_train_epoch_end=True,
                                                    monitor=None,
                                                    save_top_k=-1,
                                                    save_last='link',
                                                    filename='model_{epoch:03d}_{step:05d}',
                                                    enable_version_counter=False
                                                    )

        # For now, we use our custom one since Lightning's callback for this is minimal
        TAOExceptionCheckpoint.FILE_EXTENSION = CustomModelCheckpoint.FILE_EXTENSION
        TAOExceptionCheckpoint.CHECKPOINT_NAME_LAST = CustomModelCheckpoint.CHECKPOINT_NAME_LAST
        exception_checkpoint_callback = TAOExceptionCheckpoint(dirpath=results_dir)

        callbacks = [status_logger_callback, checkpoint_callback, exception_checkpoint_callback]
        # Best-checkpoint saving (additive by default, or replacing the periodic
        # callback when train.checkpointer.replace_periodic is enabled).
        # NVDINOv2 is SSL pretraining with no validation loop / no logged val metric, so the
        # best callback is inert unless a validation metric is added (or an explicit
        # train.checkpointer.monitor is supplied). Wired for consistency with other trainers.
        callbacks = self._configure_best_checkpoint(callbacks, results_dir)
        return callbacks
