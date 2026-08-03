# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 Model Module.

``DinoV3PlModel`` inherits the entire ``nvdinov2`` Lightning flow (training step, teacher
EMA, optimizer/scheduler config, callbacks, checkpoint saving) and overrides only
``_build_model`` to construct the DINOv3 RoPE ViT (no absolute pos-embed, per-type FFN)
using the patch-16 v3 param map.

The checkpoint remapper (Meta/timm DINOv3 -> this ViT) and the Gram-anchoring loss
(``_extra_losses``) land in later steps; this file is the build + inheritance scaffold so
the family imports, ``dinov3 --help`` resolves, and the ViT-B builds.
"""

import copy
import os

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from timm.layers import Mlp

import nvidia_tao_pytorch.config.dinov3.default_config as v3_params
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.ssl.nvdinov2.model.head import DinoHead
from nvidia_tao_pytorch.ssl.nvdinov2.model.loss import DinoV2Loss
from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import DinoV2PlModel
from nvidia_tao_pytorch.ssl.dinov3.model.vit import DinoV3VisionTransformer, SwiGLUFusedFull
from nvidia_tao_pytorch.ssl.dinov3.model.loss import GramLoss
from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import (
    TAO_ONLY_KEYS,
    fuse_timm_swiglu_fc1,
    timm_to_tao,
)

# Resolve the param-map FFN name to a layer class without importing torch in the config.
# DINOv3 ViT-H+/7B use SwiGLUFusedFull (full inner width matching the public DINOv3 SwiGLU
# checkpoint).
_MLP_LAYERS = {
    "mlp": Mlp,
    "swiglu": SwiGLUFusedFull,
}


class DinoV3PlModel(DinoV2PlModel):
    """PyTorch Lightning module for DINOv3 (inherits nvdinov2)."""

    _PRETRAINED_DIR_FILENAMES = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.pth",
    )
    _PRETRAINED_FILE_EXTENSIONS = (".safetensors", ".pth", ".bin", ".tlt")

    # Use the DINOv3 (patch-16) param map for the dim attributes the inherited
    # DinoV2PlModel.__init__ reads (depth/num_heads/embed_dim/...).
    param_map = v3_params.map_params

    def __init__(self, experiment_spec):
        """Initialize the DINOv3 Lightning module.

        Args:
            experiment_spec: The DINOv3 experiment configuration.
        """
        # ``valid_options`` generates the packaged schema enum but Hydra does not enforce
        # that metadata at runtime. Validate before the parent initializes CUDA/model state.
        v3_params.validate_img_size(experiment_spec.model.backbone)
        super().__init__(experiment_spec)
        self.checkpoint_filename = 'dinov3_model'

        # DINOv3 centers teacher outputs with Sinkhorn-Knopp (SwAV), not the softmax centering
        # nvdinov2 hardcodes. Rebuild the DINO/iBOT losses with the configured method (default
        # sinkhorn). This is a dinov3-side override only; nvdinov2 is untouched. The base
        # DinoV2Loss sinkhorn is numerically safe here because DinoHead L2-normalizes features
        # and uses unit-norm prototypes, so logits are cosine sims in [-1, 1] (no exp overflow).
        # See ssl/dinov3/README.md "Centering" for the watch-for note / softmax fallback.
        centering_method = getattr(self.model_config, "centering_method", "sinkhorn")
        if centering_method != "softmax":
            self.dino_cls_token_loss = DinoV2Loss(
                num_prototypes=self.num_prototypes,
                centering_method=centering_method,
            )
            self.ibot_patch_tokens_loss = DinoV2Loss(
                num_prototypes=self.num_prototypes,
                centering_method=centering_method,
            )

        # Gram anchoring (DINOv3). The frozen Gram teacher is constructed in _build_model
        # when enabled; sync it from the (now teacher==student) weights here so it is
        # consistent even when no pretrained checkpoint is supplied. When a pretrained
        # checkpoint is loaded, restore_pretrained_weights re-syncs it from the loaded
        # teacher (the intended provenance for continual pre-training).
        if getattr(self, 'gram_teacher', None) is not None:
            self._sync_gram_teacher()
        # Tracks the last global step the Gram teacher was EMA-refreshed (Phase 1 high-res).
        self._gram_last_refresh_step = -1

    def _resolve_arch(self, backbone_type):
        """Look up DINOv3 ViT hyper-parameters for a backbone type from the v3 param map.

        Args:
            backbone_type (str): One of the v3 ``SUPPORTED_BACKBONES`` (e.g. ``vit_b``).

        Returns:
            dict: ``embed_dim``, ``depth``, ``num_heads``, ``init_values``,
            ``drop_path_schedule``, ``num_classes`` and the resolved ``mlp_layer`` class.
        """
        mp = v3_params.map_params
        return {
            "embed_dim": mp['embed_dim'][backbone_type],
            "depth": mp['depth'][backbone_type],
            "num_heads": mp['num_heads'][backbone_type],
            "init_values": mp['init_values'][backbone_type],
            "drop_path_schedule": mp['drop_path_schedule'][backbone_type],
            "num_classes": mp['num_classes'][backbone_type],
            "mlp_layer": _MLP_LAYERS[mp['mlp_layer'][backbone_type]],
            "mlp_ratio": mp['mlp_ratio'][backbone_type],
        }

    def _make_backbone(self, arch):
        """Construct a DINOv3 ViT backbone from a resolved arch dict.

        Args:
            arch (dict): Output of :meth:`_resolve_arch`.

        Returns:
            DinoV3VisionTransformer: The constructed backbone.
        """
        return DinoV3VisionTransformer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=arch["embed_dim"],
            depth=arch["depth"],
            num_heads=arch["num_heads"],
            init_values=arch["init_values"],
            drop_path_schedule=arch["drop_path_schedule"],
            num_classes=arch["num_classes"],
            drop_path_rate=self.drop_path_rate,
            mlp_layer=arch["mlp_layer"],
            mlp_ratio=arch["mlp_ratio"],
            norm_layer=nn.LayerNorm,
            # DINOv3 ViT-B/L use a standard GELU MLP with no QKV bias (timm
            # vit_base_patch16_dinov3 reference); these are the v3 ViT defaults.
            act_layer=nn.GELU,
            qkv_bias=False,
            register_tokens=self.register_tokens,
            use_custom_attention=self.use_custom_attention,
            rope_theta=self.model_config.backbone['rope_theta'],
        )

    def _make_head(self, embed_dim):
        """Construct a DINO/iBOT head for the given embedding dimension.

        Args:
            embed_dim (int): Backbone embedding dimension.

        Returns:
            DinoHead: The constructed head.
        """
        return DinoHead(
            in_dim=embed_dim,
            out_dim=self.num_prototypes,
            num_layers=self.head_layers,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
        )

    def _build_model(self):
        """Build the DINOv3 student and teacher (RoPE ViT) using the v3 param map.

        Overrides :meth:`DinoV2PlModel._build_model`: same student/teacher (+optional
        ``student_ema`` for distillation) structure, but built from ``DinoV3VisionTransformer``
        and the patch-16 v3 ``map_params`` (with per-type FFN selection). Dimension
        attributes are re-derived here from the v3 param map so they are self-consistent
        regardless of the (nvdinov2) param map the parent ``__init__`` read.
        """
        # Re-derive dims from the v3 (patch-16) param map.
        student_arch = self._resolve_arch(self.student_backbone_type)
        teacher_arch = self._resolve_arch(self.teacher_backbone_type)
        self.student_embed_dim = student_arch["embed_dim"]
        self.teacher_embed_dim = teacher_arch["embed_dim"]

        self.student = torch.nn.ModuleDict(
            {
                'backbone': self._make_backbone(student_arch),
                'dino_head': self._make_head(self.student_embed_dim),
                'ibot_head': self._make_head(self.student_embed_dim),
            }
        )
        self.teacher = torch.nn.ModuleDict(
            {
                'backbone': self._make_backbone(teacher_arch),
                'dino_head': self._make_head(self.teacher_embed_dim),
                'ibot_head': self._make_head(self.teacher_embed_dim),
            }
        )

        if self.model_config.distill.enable:
            # Create a student ema for distillation
            self.student_ema = copy.deepcopy(self.student)

            assert self.model_config.distill.pretrained_non_distill_pl_model_path is not None, (
                "In distillation mode, you need to provide the pretrained_non_distill_pl_model_path "
                "to initialize a frozen teacher."
            )
            pretrained_backbone_head_state_dict = torch.load(
                self.model_config.distill.pretrained_non_distill_pl_model_path, map_location="cpu"
            )['state_dict']
            teacher_state_dict = {}
            for k, v in list(pretrained_backbone_head_state_dict.items()):
                if "teacher." in k:
                    teacher_state_dict[k.replace("teacher.", "")] = v

            self.teacher.load_state_dict(teacher_state_dict)
        else:
            assert self.student_backbone_type == self.teacher_backbone_type, (
                f"In non-distillation mode, student_type and teacher_type should be the same. "
                f"Currently, the teacher_type is {self.teacher_backbone_type}, and the "
                f"student_type is {self.student_backbone_type}."
            )

        # Gram anchoring: build a separate frozen Gram teacher (a copy of the teacher
        # backbone) when enabled. It is intentionally NOT placed inside self.teacher /
        # self.student (so the parent's FSDP wrapping and CustomModelCheckpoint, which act on
        # those ModuleDicts, leave it alone) and never receives gradients or EMA updates.
        # Its weights are (re)synced from the teacher by _sync_gram_teacher.
        if self.model_config.gram.enable:
            self.gram_teacher = self._make_backbone(teacher_arch)
            for param in self.gram_teacher.parameters():
                param.requires_grad = False
            self.gram_teacher.eval()
            self.gram_loss = GramLoss()

    def _sync_gram_teacher(self):
        """Copy the current teacher backbone weights into the frozen Gram teacher.

        Called once at construction (teacher == student) and again after a pretrained
        checkpoint is loaded (``restore_pretrained_weights``), so the Gram teacher always
        anchors to the same DINOv3 weights the run starts from.
        """
        teacher_backbone = self.teacher.backbone
        if isinstance(teacher_backbone, FSDP):
            # Under FSDP the teacher backbone is sharded across ranks; gather a full, unsharded
            # state dict on every rank (rank0_only=False) so it can be copied into the unsharded,
            # replicated Gram teacher. This runs inside the training step at the same global step
            # on every rank, so the all-gather is symmetric and will not deadlock. (Risk R3.)
            with FSDP.state_dict_type(
                teacher_backbone,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
            ):
                state_dict = teacher_backbone.state_dict()
        else:
            state_dict = teacher_backbone.state_dict()
        self.gram_teacher.load_state_dict(state_dict)
        for param in self.gram_teacher.parameters():
            param.requires_grad = False
        self.gram_teacher.eval()

    @staticmethod
    def _gram_refresh_due(step, interval, last_refreshed):
        """Whether the Gram teacher should be EMA-refreshed at this step.

        Args:
            step (int): Current global step.
            interval (int): Refresh period in steps (0 disables refresh).
            last_refreshed (int): Step at which the last refresh happened.

        Returns:
            bool: True if a refresh is due now.
        """
        if not interval or interval <= 0:
            return False
        return step > 0 and step % interval == 0 and step != last_refreshed

    @staticmethod
    def _pool_tokens(tokens, src_grid, dst_grid):
        """Average-pool a patch-token grid to a smaller grid (Gram teacher -> student grid).

        Args:
            tokens (torch.Tensor): Patch tokens ``[B, src_h*src_w, C]`` (row-major).
            src_grid (tuple): Source ``(h, w)`` patch grid.
            dst_grid (tuple): Target ``(h, w)`` patch grid.

        Returns:
            torch.Tensor: Pooled tokens ``[B, dst_h*dst_w, C]``. Returned unchanged if grids match.
        """
        if tuple(src_grid) == tuple(dst_grid):
            return tokens
        b, _, c = tokens.shape
        src_h, src_w = src_grid
        grid = tokens.reshape(b, src_h, src_w, c).permute(0, 3, 1, 2)  # [B, C, src_h, src_w]
        grid = nn.functional.adaptive_avg_pool2d(grid, dst_grid)        # [B, C, dst_h, dst_w]
        return grid.permute(0, 2, 3, 1).reshape(b, dst_grid[0] * dst_grid[1], c)

    def _maybe_refresh_gram_teacher(self):
        """Refresh the Gram teacher from the EMA teacher on the configured cadence (Phase 1).

        Only acts when ``gram.teacher_source='ema'`` and ``gram.refresh_interval>0``. This is the
        early-EMA-snapshot scheme DINOv3 uses for the high-res phase. Single-device path: under
        FSDP the EMA teacher is sharded, so refreshing needs a ``summon_full_params`` gather —
        tracked as a Phase-1 follow-up (see the high-res plan, risk R3).
        """
        gram_cfg = self.model_config.gram
        if gram_cfg.teacher_source != "ema":
            return
        if not self._gram_refresh_due(self.global_step, getattr(gram_cfg, "refresh_interval", 0),
                                      self._gram_last_refresh_step):
            return
        self._sync_gram_teacher()
        self._gram_last_refresh_step = self.global_step
        if get_global_rank() == 0:
            logging.info(f"Gram teacher refreshed from EMA teacher at step {self.global_step}.")

    def on_save_checkpoint(self, checkpoint):
        """Persist the Gram-teacher refresh bookkeeping so resume keeps a consistent cadence.

        ``_gram_last_refresh_step`` drives the EMA-refresh schedule; without it a resumed run
        re-initializes to -1 and would refresh on the wrong steps relative to the original run.
        """
        super().on_save_checkpoint(checkpoint)
        checkpoint["gram_last_refresh_step"] = self._gram_last_refresh_step

    def on_load_checkpoint(self, checkpoint):
        """Restore the Gram-teacher refresh bookkeeping on resume (mirror of on_save_checkpoint)."""
        super().on_load_checkpoint(checkpoint)
        if "gram_last_refresh_step" in checkpoint:
            self._gram_last_refresh_step = checkpoint["gram_last_refresh_step"]

    @classmethod
    def _invalid_pretrained_checkpoint(cls, path, reason):
        """Build the actionable error shared by DINOv3 checkpoint validation failures."""
        filenames = ", ".join(cls._PRETRAINED_DIR_FILENAMES)
        extensions = ", ".join(cls._PRETRAINED_FILE_EXTENSIONS)
        return ValueError(
            f"Invalid DINOv3 pretrained_model_path '{path}': {reason}. "
            f"Expected a directory containing {filenames}, or a direct {extensions} "
            "DINOv3 checkpoint matching the configured backbone. "
            "DINOv2/NVDINOv2 checkpoints are not supported."
        )

    @classmethod
    def _load_pretrained_state_dict(cls, path):
        """Load a DINOv3 checkpoint into a flat ``{key: tensor}`` state dict.

        Accepts either a directory holding timm-format weights (``model.safetensors`` or
        ``pytorch_model.bin``) or a direct ``.safetensors`` / ``.pth`` / ``.bin`` file, and
        unwraps a ``state_dict`` / ``model`` container if present.

        Args:
            path (str): File or directory path to the pretrained weights.

        Returns:
            dict: Flat parameter-name -> tensor mapping.

        Raises:
            ValueError: If the path or checkpoint payload is invalid.
        """
        original_path = os.fspath(path) if path else str(path)
        if not path or not os.path.exists(path):
            raise cls._invalid_pretrained_checkpoint(original_path, "path does not exist")

        if os.path.isdir(path):
            resolved = next(
                (
                    os.path.join(path, candidate)
                    for candidate in cls._PRETRAINED_DIR_FILENAMES
                    if os.path.isfile(os.path.join(path, candidate))
                ),
                None,
            )
            if resolved is None:
                raise cls._invalid_pretrained_checkpoint(
                    original_path,
                    "directory does not contain a supported checkpoint file",
                )
            path = resolved
        elif not os.path.isfile(path):
            raise cls._invalid_pretrained_checkpoint(
                original_path,
                "path is not a regular file",
            )

        extension = os.path.splitext(path)[1].lower()
        if extension not in cls._PRETRAINED_FILE_EXTENSIONS:
            raise cls._invalid_pretrained_checkpoint(
                original_path,
                f"unsupported checkpoint extension '{extension or '<none>'}'",
            )

        try:
            if extension == ".safetensors":
                from safetensors.torch import load_file
                state_dict = load_file(path)
            else:
                state_dict = torch.load(path, map_location="cpu")
        except Exception:
            raise cls._invalid_pretrained_checkpoint(
                original_path,
                "checkpoint file could not be loaded",
            ) from None

        for container_key in ("state_dict", "model"):
            if isinstance(state_dict, dict) and container_key in state_dict:
                state_dict = state_dict[container_key]
                break

        if (
            not isinstance(state_dict, dict) or
            not state_dict or
            not all(isinstance(key, str) and torch.is_tensor(value)
                    for key, value in state_dict.items())
        ):
            raise cls._invalid_pretrained_checkpoint(
                original_path,
                "checkpoint does not contain a non-empty tensor state dictionary",
            )
        return state_dict

    @staticmethod
    def _remap_dinov3_state_dict(timm_state_dict, reference_state_dict):
        """Translate timm DINOv3 ViT keys to ``DinoV3VisionTransformer`` keys.

        Renames the LayerScale gammas (``blocks.N.gamma_1/2`` -> ``blocks.N.ls1/ls2.gamma``)
        and the register parameter (``reg_token`` -> ``register_tokens``); all other keys
        (``cls_token``, ``patch_embed.proj.*``, ``blocks.N.{norm1,norm2,attn.qkv,attn.proj,
        mlp.fc1,mlp.fc2}.*``, ``norm.*``) map by identity. timm carries no ``pos_embed`` (RoPE
        replaces it) and no QKV bias. Only keys present in the reference state dict with a
        matching shape are kept.

        Args:
            timm_state_dict (dict): Source timm/Meta DINOv3 state dict.
            reference_state_dict (dict): ``self.student.backbone.state_dict()`` (target keys).

        Returns:
            Tuple[dict, list]: ``(remapped, unmapped)`` where ``remapped`` is loadable into
            the backbone and ``unmapped`` lists source keys that had no shape-matching target.
        """
        # Fuse split SwiGLU projections (fc1_g/fc1_x) into the fused fc1 for ViT-H+/7B before
        # the per-key shape match; plain-MLP (ViT-B/L) checkpoints are unaffected.
        timm_state_dict = fuse_timm_swiglu_fc1(timm_state_dict)

        remapped = {}
        unmapped = []
        for key, weight in timm_state_dict.items():
            new_key = timm_to_tao(key)

            if new_key in reference_state_dict and reference_state_dict[new_key].shape == weight.shape:
                remapped[new_key] = weight
            else:
                unmapped.append(key)
        return remapped, unmapped

    @classmethod
    def _validate_and_remap_pretrained_state_dict(
        cls,
        state_dict,
        reference_state_dict,
        path,
    ):
        """Validate DINOv3 identity and backbone compatibility before loading any tensors."""
        if any(key == "pos_embed" or key.endswith(".pos_embed") for key in state_dict):
            raise cls._invalid_pretrained_checkpoint(
                path,
                "checkpoint contains an absolute positional embedding and appears to be DINOv2",
            )

        try:
            remapped, unmapped = cls._remap_dinov3_state_dict(
                state_dict,
                reference_state_dict,
            )
        except Exception:
            raise cls._invalid_pretrained_checkpoint(
                path,
                "checkpoint tensors could not be remapped to the configured DINOv3 backbone",
            ) from None
        required_keys = set(reference_state_dict) - set(TAO_ONLY_KEYS)
        missing_keys = sorted(required_keys - set(remapped))
        if missing_keys:
            preview = ", ".join(missing_keys[:5])
            if len(missing_keys) > 5:
                preview += ", ..."
            raise cls._invalid_pretrained_checkpoint(
                path,
                "checkpoint does not match the configured DINOv3 backbone; "
                f"missing or shape-incompatible tensors: {preview}",
            )
        return remapped, unmapped

    def restore_pretrained_weights(self, preloaded_state_dict=None):
        """Load timm/Meta DINOv3 weights via the v3 key remapper, sync teacher + Gram teacher.

        Replaces the inherited loader (which ``torch.load``s a single ``.pth`` of already-
        matching keys): DINOv3 ships timm-format weights whose keys and token layout differ,
        so this resolves the checkpoint, remaps the keys, loads them into the student backbone
        (non-strict, reporting residual missing/unexpected), mirrors the student into the
        teacher (non-distill), and re-anchors the frozen Gram teacher to the loaded weights.

        Args:
            preloaded_state_dict (dict, optional): An already-loaded checkpoint state dict.
                When omitted, load ``self.pretrained_weights`` as before. This avoids reading
                the checkpoint twice when export first inspects its shape.
        """
        reference_state_dict = self.student.backbone.state_dict()
        timm_state_dict = (
            preloaded_state_dict
            if preloaded_state_dict is not None
            else self._load_pretrained_state_dict(self.pretrained_weights)
        )
        remapped, unmapped = self._validate_and_remap_pretrained_state_dict(
            timm_state_dict,
            reference_state_dict,
            self.pretrained_weights,
        )

        missing_keys, unexpected_keys = self.student.backbone.load_state_dict(remapped, strict=False)

        if get_global_rank() == 0:
            # Denominator is the post-fusion source-key count (remapped + unmapped); for SwiGLU
            # backbones the split fc1_g/fc1_x are fused into one fc1 before this count.
            logging.info(
                f"DINOv3 remap: loaded {len(remapped)}/{len(remapped) + len(unmapped)} checkpoint tensors "
                f"into the ViT backbone."
            )
            # ``mask_token`` is an iBOT parameter absent from the (inference) DINOv3 checkpoint;
            # it is initialized by the constructor, so flag the rest as the meaningful residual.
            residual_missing = [k for k in missing_keys if k not in TAO_ONLY_KEYS]
            if residual_missing:
                logging.info(f"DINOv3 remap missing keys (kept as initialized): {residual_missing}")
            if unexpected_keys:
                logging.info(f"DINOv3 remap unexpected keys: {unexpected_keys}")
            if unmapped:
                logging.info(f"DINOv3 checkpoint keys with no matching backbone param: {unmapped}")

        if not self.model_config.distill.enable:
            self.teacher.load_state_dict(self.student.state_dict(), strict=False)

        if getattr(self, 'gram_teacher', None) is not None:
            self._sync_gram_teacher()

    def _extra_losses(self, **ctx):
        """Inject the DINOv3 Gram-anchoring loss into the inherited student_forward.

        Returns an empty list (matching the DINOv2 base) unless Gram anchoring is enabled,
        the frozen Gram teacher has been built, and the global step has reached
        ``gram.start_step``. Otherwise it runs the frozen Gram teacher on the global crops and
        returns ``[w_gram * GramLoss(student_patches, teacher_patches)]``.

        Args:
            **ctx: Context forwarded from ``DinoV2PlModel.student_forward`` (global/local
                crops, masks, and the student global/local backbone outputs).

        Returns:
            list: ``[weighted_gram_loss]`` when active, else ``[]``.
        """
        gram_cfg = self.model_config.gram
        if not gram_cfg.enable or getattr(self, 'gram_teacher', None) is None:
            return []
        if self.global_step < gram_cfg.start_step:
            return []

        student_backbone_global_output = ctx.get("student_backbone_global_output")
        global_crops = ctx.get("global_crops")
        if student_backbone_global_output is None or global_crops is None:
            return []

        # Phase 1: EMA-refresh the Gram teacher on cadence (no-op when teacher_source != 'ema').
        self._maybe_refresh_gram_teacher()

        student_patch_tokens = student_backbone_global_output["x_norm_patchtokens"]

        # Student patch grid (global crops are square; derive from the input + patch size).
        ps = self.patch_size
        student_grid = (global_crops.shape[-2] // ps, global_crops.shape[-1] // ps)

        # Frozen Gram teacher: optionally at a higher resolution (gram.teacher_scale), no grad.
        # Force eval so stochastic depth / dropout stay off even though the parent is in train mode.
        scale = getattr(gram_cfg, "teacher_scale", 1.0) or 1.0
        self.gram_teacher.eval()
        # The frozen Gram teacher lives outside the FSDP-wrapped student/teacher ModuleDicts, so
        # FSDP never places it on the sharding device — co-locate it with the input here (no-op
        # once moved / on single-GPU, where the module already follows the LightningModule). It
        # then runs under the ambient autocast just like the student/teacher backbones (GramLoss
        # upcasts to fp32 internally for the Gram product).
        if next(self.gram_teacher.parameters()).device != global_crops.device:
            self.gram_teacher.to(global_crops.device)
        with torch.no_grad():
            if scale != 1.0:
                teacher_input = nn.functional.interpolate(
                    global_crops, scale_factor=scale, mode="bilinear", align_corners=False
                )
            else:
                teacher_input = global_crops
            teacher_patch_tokens = self.gram_teacher(teacher_input)["x_norm_patchtokens"]
            # Pool the (higher-res) teacher grid back to the student grid before the Gram loss.
            teacher_grid = (teacher_input.shape[-2] // ps, teacher_input.shape[-1] // ps)
            teacher_patch_tokens = self._pool_tokens(teacher_patch_tokens, teacher_grid, student_grid)

        gram_loss = self.gram_loss(student_patch_tokens, teacher_patch_tokens)
        self.log(
            "losses/gram_loss",
            gram_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )
        return [gram_cfg.w_gram * gram_loss]
