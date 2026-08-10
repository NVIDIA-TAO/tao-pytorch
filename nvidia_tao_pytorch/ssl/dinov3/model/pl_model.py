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
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from timm.layers import Mlp

import nvidia_tao_pytorch.config.dinov3.default_config as v3_params
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.ssl.nvdinov2.model.head import DinoHead
from nvidia_tao_pytorch.ssl.nvdinov2.model.loss import DinoV2Loss
from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import CustomModelCheckpoint, DinoV2PlModel
from nvidia_tao_pytorch.ssl.dinov3.model.vit import DinoV3VisionTransformer, SwiGLUFusedFull
from nvidia_tao_pytorch.ssl.dinov3.model.loss import GramLoss, ClsPreservationLoss
from nvidia_tao_pytorch.ssl.dinov3.model.lora import (
    inject_lora,
    is_lora_key,
    lora_parameter_report,
    strip_lora_keys,
)
from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import (
    TAO_ONLY_KEYS,
    fuse_timm_swiglu_fc1,
    merge_lora_state_dict,
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

        # Gram anchoring (DINOv3). The frozen anchor teacher is constructed in _build_model
        # when Gram anchoring or CLS preservation is enabled; sync it from the (now
        # teacher==student) weights here so it is consistent even when no pretrained
        # checkpoint is supplied. When a pretrained checkpoint is loaded,
        # restore_pretrained_weights re-syncs it from the loaded teacher (the intended
        # provenance for continual pre-training).
        if getattr(self, 'gram_teacher', None) is not None:
            self._sync_gram_teacher()
        # Tracks the last global step the Gram teacher was EMA-refreshed (Phase 1 high-res).
        self._gram_last_refresh_step = -1

        # LoRA is injected later (train.py, after restore_pretrained_weights and before
        # trainer.fit). The guardrails were already checked at the top of _build_model.
        self._lora_injected = False

    @property
    def preservation_config(self):
        """CLS-preservation sub-config, tolerating specs built before it existed.

        Returns:
            PreservationConfig | None: The block, or ``None`` if absent from the spec.
        """
        return getattr(self.model_config, "preservation", None)

    def _preservation_enabled(self):
        """Whether CLS-token preservation is switched on.

        Returns:
            bool: True when the ``preservation`` block exists and is enabled.
        """
        preservation = self.preservation_config
        return bool(preservation is not None and preservation.enable)

    def _validate_lora_config(self):
        """Assert the LoRA guardrails from the design doc (v1 scope).

        LoRA is incompatible with three things in v1:

        * ``distill.enable`` -- distillation adds a third ``student_ema`` ViT and loads a
          teacher from a non-distill checkpoint; the injection lifecycle is unverified there.
        * ``gram.teacher_source='ema'`` -- refreshing the anchor from the drifting EMA teacher
          defeats the point of preservation (and would re-anchor to LoRA-adapted weights).
        * FSDP -- flat-param wrapping of mostly-frozen modules needs ``use_orig_params=True``;
          deferred. LoRA's optimizer state is tiny, so DDP is sufficient for ViT-B/L.

        Raises:
            AssertionError: If ``lora.enable`` is combined with any of the above.
        """
        lora_cfg = getattr(self.model_config, "lora", None)
        if lora_cfg is None or not lora_cfg.enable:
            return

        assert not self.model_config.distill.enable, (
            "model.lora.enable is not supported together with model.distill.enable in v1. "
            "Distillation builds an extra student_ema backbone whose LoRA injection lifecycle "
            "is not covered; run LoRA continual pre-training without distillation."
        )
        assert self.model_config.gram.teacher_source != "ema", (
            "model.lora.enable requires model.gram.teacher_source='pretrained'. An EMA-refreshed "
            "anchor tracks the drifting teacher, which defeats preservation -- the frozen "
            "pretrained weights are the only exogenous anchor in the system."
        )

        strategy = os.environ.get(
            "DINOV3_STRATEGY", getattr(self.train_config, "distributed_strategy", "auto")
        ).lower()
        assert strategy != "fsdp", (
            "model.lora.enable is not supported with train.distributed_strategy='fsdp' in v1 "
            "(FSDP flat-param wrapping of mostly-frozen modules needs use_orig_params=True). "
            "Use 'auto' or 'ddp' -- LoRA's optimizer state is tiny, so DDP fits comfortably."
        )

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """EMA-update the teacher, skipping frozen base weights when LoRA is active.

        The inherited implementation zips *all* student and teacher parameters. Under LoRA the
        base weights are frozen and identical in both, so ``m*W + (1-m)*W`` is a no-op --
        in exact arithmetic. In floating point it is not: the round-trip through
        ``_foreach_mul_`` / ``_foreach_add_`` leaves a small residual every step, which
        accumulates. Measured on the Stage-2 smoke, the teacher's base weights had drifted by
        5.67e-05 (relative, worst tensor) after only 500 steps while the student's stayed
        bit-exact -- so the "frozen" anchor was quietly moving, and an exported teacher backbone
        would not have been exactly ``frozen base + merged delta``.

        Restricting the EMA to parameters that are trainable in the student (the LoRA adapters,
        the DINO/iBOT heads and ``mask_token``) makes the no-op exact, and is cheaper.

        Falls back to the inherited behaviour whenever LoRA is not injected, so full-fine-tune
        runs are numerically unchanged.

        Args:
            momentum (float): EMA momentum for the teacher update.
        """
        lora_cfg = getattr(self.model_config, "lora", None)
        if not (lora_cfg is not None and lora_cfg.enable and self._lora_injected):
            super().update_teacher(momentum)
            return

        # FSDP is asserted against in LoRA mode (_validate_lora_config), so the flat-param
        # path the parent handles cannot arise here.
        if not isinstance(self.trainer.strategy, SingleDeviceStrategy):
            torch.cuda.synchronize()
            dist.barrier()

        teacher_params = []
        student_params = []
        for (name, student_param), (teacher_name, teacher_param) in zip(
            self.student.named_parameters(), self.teacher.named_parameters()
        ):
            assert name == teacher_name, (
                f"student/teacher parameter order diverged ({name} vs {teacher_name}); "
                "LoRA must be injected into both backbones identically."
            )
            if not student_param.requires_grad:
                continue  # frozen base weight: already identical in both, leave it untouched
            teacher_params.append(teacher_param.data)
            student_params.append(student_param.data)

        torch._foreach_mul_(teacher_params, momentum)
        torch._foreach_add_(teacher_params, student_params, alpha=1.0 - momentum)

    def inject_lora_adapters(self):
        """Inject LoRA into the student and EMA-teacher backbones. No-op unless enabled.

        Called from the train script **after** ``restore_pretrained_weights`` (so the loader
        sees stock keys) and **before** ``trainer.fit`` (so Lightning resume checkpoints and
        any distributed wrapping see the final module tree).

        Three invariants are established here:

        * **EMA zip alignment** -- the student and teacher are injected with identical
          arguments, so ``update_teacher``'s element-wise ``zip`` of ``parameters()`` stays
          aligned (gate G1.3).
        * **Teacher starts equal to the student** -- ``lora_A`` is randomly initialized, so the
          two backbones would otherwise disagree at step 0 and the teacher's adapter would not
          be an EMA of the student's. The student's backbone state is mirrored into the
          teacher after injection to guarantee equality (gate G2.4).
        * **Teacher stays frozen** -- newly created adapter parameters default to
          ``requires_grad=True``; the teacher's are switched off again (it is EMA-updated,
          never gradient-updated).

        The frozen anchor (Gram) teacher deliberately receives no LoRA: it anchors to the
        pretrained weights, which under LoRA are exactly the frozen base weights.

        Returns:
            dict | None: The trainable-parameter report, or ``None`` when LoRA is disabled.
        """
        lora_cfg = getattr(self.model_config, "lora", None)
        if lora_cfg is None or not lora_cfg.enable:
            return None
        if self._lora_injected:
            return None

        self._validate_lora_config()

        target_modules = list(lora_cfg.target_modules)
        kwargs = dict(
            rank=lora_cfg.rank,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
            target_modules=target_modules,
            num_last_blocks=lora_cfg.num_last_blocks,
            freeze_base=True,
        )

        injected = inject_lora(self.student.backbone, **kwargs)
        teacher_injected = inject_lora(self.teacher.backbone, **kwargs)

        # An empty target set would otherwise freeze the whole backbone, mark LoRA as injected,
        # and quietly train only the heads and mask_token -- a run that looks like LoRA and is
        # not. Comparing the two lists also catches any future student/teacher topology
        # divergence at injection time rather than as a silently misaligned EMA (gate G1.3).
        assert injected, (
            f"LoRA injected no modules (target_modules={target_modules}, "
            f"num_last_blocks={lora_cfg.num_last_blocks}). model.lora.enable is set, so this "
            "would train only the heads and mask_token while reporting a LoRA run."
        )
        assert injected == teacher_injected, (
            "student and teacher received different LoRA injections "
            f"(student={injected}, teacher={teacher_injected}); the EMA zip requires identical "
            "parameter structure in both backbones."
        )

        # Mirror the student's backbone (base + freshly initialized adapters) into the teacher
        # so both start identical; otherwise the teacher's random lora_A would make its EMA
        # trajectory meaningless.
        self.teacher.backbone.load_state_dict(self.student.backbone.state_dict())

        # The teacher is EMA-updated, never gradient-updated.
        for param in self.teacher.parameters():
            param.requires_grad = False

        self._lora_injected = True

        if get_global_rank() == 0:
            logging.info(
                f"LoRA injected into {len(injected)} modules per backbone "
                f"(rank={lora_cfg.rank}, alpha={lora_cfg.alpha}, dropout={lora_cfg.dropout}, "
                f"targets={target_modules}, "
                f"num_last_blocks={lora_cfg.num_last_blocks or 'all'})."
            )
        return lora_parameter_report(self.student, name="student (LoRA)")

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
        # Validate the LoRA guardrails before anything is constructed, so an unsupported
        # combination reports *its own* reason rather than tripping a downstream assert
        # (e.g. distillation's missing-checkpoint check) first.
        self._validate_lora_config()

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

        # Preservation: build a separate frozen *anchor* teacher (a copy of the teacher
        # backbone) when Gram anchoring OR CLS preservation is enabled -- both terms read the
        # same single anchor forward, so one module serves both. It is intentionally NOT
        # placed inside self.teacher / self.student (so the parent's FSDP wrapping and
        # CustomModelCheckpoint, which act on those ModuleDicts, leave it alone) and never
        # receives gradients, EMA updates, or LoRA adapters. Its weights are (re)synced from
        # the teacher by _sync_gram_teacher.
        if self.model_config.gram.enable or self._preservation_enabled():
            self.gram_teacher = self._make_backbone(teacher_arch)
            for param in self.gram_teacher.parameters():
                param.requires_grad = False
            self.gram_teacher.eval()
            self.gram_loss = GramLoss()
            self.cls_preservation_loss = ClsPreservationLoss()

    def _sync_gram_teacher(self):
        """Copy the current teacher backbone weights into the frozen anchor teacher.

        Called once at construction (teacher == student) and again after a pretrained
        checkpoint is loaded (``restore_pretrained_weights``), so the anchor teacher always
        anchors to the same DINOv3 weights the run starts from.

        Under LoRA the teacher backbone carries ``lora_A``/``lora_B`` keys that the anchor
        teacher (deliberately un-injected) does not have. They are filtered out before the
        load; because the LoRA design is key-preserving, what remains is exactly the anchor
        teacher's key set, and those base tensors are the frozen pretrained originals. So
        this stays a strict load and the anchor provably cannot drift (gate G1.6).
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

        # Drop LoRA adapter keys (absent from the un-injected anchor teacher); the remaining
        # base keys must cover it exactly.
        state_dict = strip_lora_keys(state_dict)
        anchor_keys = set(self.gram_teacher.state_dict())
        missing = anchor_keys - set(state_dict)
        assert not missing, (
            f"Anchor (Gram) teacher sync is missing base keys after LoRA filtering: {sorted(missing)}"
        )

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
        # Source carries adapters but the destination backbone cannot hold them (inference,
        # export, or the start of a fresh run before injection): fold them into the base so the
        # weights loaded are the *adapted* model. Without this the adapter keys are simply
        # unmapped and the frozen base is silently used instead. When the destination is
        # already LoRA-injected the adapters load normally and this does nothing.
        if any(is_lora_key(k) for k in timm_state_dict) and \
                not any(is_lora_key(k) for k in reference_state_dict):
            timm_state_dict = merge_lora_state_dict(timm_state_dict)

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

    def _log_loss(self, name, value):
        """Log a scalar loss/diagnostic under the run's standard step-logging settings.

        Args:
            name (str): Scalar name (e.g. ``losses/cls_mse``).
            value (torch.Tensor): Scalar value.
        """
        self.log(
            name,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )

    def _extra_losses(self, **ctx):
        """Inject the DINOv3 preservation losses into the inherited student_forward.

        Two terms, both measured against the **same single forward** of the frozen anchor
        teacher (its output dict carries ``x_norm_clstoken`` alongside ``x_norm_patchtokens``,
        so CLS preservation costs no extra teacher pass):

        * **Gram anchoring** -- MSE between student and anchor patch-cosine Gram matrices,
          preserving the dense/patch geometry. Gated on ``gram.enable`` and ``gram.start_step``.
        * **CLS preservation** -- MSE + cosine distance between student and anchor CLS tokens,
          preserving the global embedding geometry that k-NN/linear-probe/retrieval consume.
          Gated on ``preservation.enable``.

        **Input contract (important when reading these values).** The student tensors come from
        the *masked* global crops the SSL step already computed, while the anchor runs on the
        *unmasked* crops -- matching how DINOv3 applies Gram anchoring, and avoiding a second
        student forward. Both terms are therefore a sum of two things: preservation of the
        pretrained geometry (the intent) **and** sensitivity to iBOT masking. They are
        consequently **not zero at step 0** even when the weights are identical, and they must
        not be read as pure drift metrics.

        Measured on ViT-B at init, where student and anchor weights are identical by
        construction (LoRA is an identity at ``lora_B = 0``):

        =====================  =========  =========  =========
        condition              gram       cls_mse    cls_cos
        =====================  =========  =========  =========
        unmasked, eval          3.7e-09    4.3e-08    1.2e-07
        30% masked, train       2.4e-02    5.8e-01    9.3e-01
        =====================  =========  =========  =========

        So the identity property is exact, but only observable with masking and stochastic
        depth disabled -- which is how a sync/remap regression should be checked. During
        training the value is dominated by masking and drop_path noise. For a true drift
        measurement, run the anchor comparison periodically in eval mode on unmasked crops.

        Both terms are logged unconditionally when active (even at weight 0). Returns an empty
        list -- matching the DINOv2 base, hence unchanged numerics -- when neither is active.

        Args:
            **ctx: Context forwarded from ``DinoV2PlModel.student_forward`` (global/local
                crops, masks, and the student global/local backbone outputs).

        Returns:
            list: Weighted extra loss tensors to add to the total, possibly empty.
        """
        gram_cfg = self.model_config.gram
        preservation_cfg = self.preservation_config

        gram_active = gram_cfg.enable and self.global_step >= gram_cfg.start_step
        preservation_active = self._preservation_enabled()
        if not (gram_active or preservation_active):
            return []
        if getattr(self, 'gram_teacher', None) is None:
            return []

        student_backbone_global_output = ctx.get("student_backbone_global_output")
        global_crops = ctx.get("global_crops")
        if student_backbone_global_output is None or global_crops is None:
            return []

        # Phase 1: EMA-refresh the Gram teacher on cadence (no-op when teacher_source != 'ema').
        self._maybe_refresh_gram_teacher()

        # Student patch grid (global crops are square; derive from the input + patch size).
        ps = self.patch_size
        student_grid = (global_crops.shape[-2] // ps, global_crops.shape[-1] // ps)

        # Frozen anchor teacher: optionally at a higher resolution (gram.teacher_scale), no grad.
        # Force eval so stochastic depth / dropout stay off even though the parent is in train mode.
        scale = getattr(gram_cfg, "teacher_scale", 1.0) or 1.0
        self.gram_teacher.eval()
        # The frozen anchor teacher lives outside the FSDP-wrapped student/teacher ModuleDicts, so
        # FSDP never places it on the sharding device — co-locate it with the input here (no-op
        # once moved / on single-GPU, where the module already follows the LightningModule). It
        # then runs under the ambient autocast just like the student/teacher backbones (the
        # preservation losses upcast to fp32 internally).
        if next(self.gram_teacher.parameters()).device != global_crops.device:
            self.gram_teacher.to(global_crops.device)
        with torch.no_grad():
            if scale != 1.0:
                teacher_input = nn.functional.interpolate(
                    global_crops, scale_factor=scale, mode="bilinear", align_corners=False
                )
            else:
                teacher_input = global_crops
            anchor_output = self.gram_teacher(teacher_input)
            teacher_patch_tokens = anchor_output["x_norm_patchtokens"]
            # Pool the (higher-res) teacher grid back to the student grid before the Gram loss.
            teacher_grid = (teacher_input.shape[-2] // ps, teacher_input.shape[-1] // ps)
            teacher_patch_tokens = self._pool_tokens(teacher_patch_tokens, teacher_grid, student_grid)

        losses = []

        if gram_active:
            gram_loss = self.gram_loss(
                student_backbone_global_output["x_norm_patchtokens"], teacher_patch_tokens
            )
            self._log_loss("losses/gram_loss", gram_loss)
            losses.append(gram_cfg.w_gram * gram_loss)

        if preservation_active:
            # The CLS token is unaffected by the patch-grid pooling above, so it is read
            # straight off the same anchor forward.
            cls_mse, cls_cosine = self.cls_preservation_loss(
                student_backbone_global_output["x_norm_clstoken"],
                anchor_output["x_norm_clstoken"],
            )
            self._log_loss("losses/cls_mse", cls_mse)
            self._log_loss("losses/cls_cos", cls_cosine)
            # Mean cosine between the (masked) student CLS and the (unmasked) anchor CLS.
            # Deliberately *not* named as a drift metric: per the contract above it also
            # carries the masking and drop_path response, so it is a trend/stability signal
            # rather than a measure of how far the weights have moved.
            self._log_loss("diagnostics/cls_anchor_cosine_masked", 1.0 - cls_cosine)

            if preservation_cfg.cls_mse_weight:
                losses.append(preservation_cfg.cls_mse_weight * cls_mse)
            if preservation_cfg.cls_cosine_weight:
                losses.append(preservation_cfg.cls_cosine_weight * cls_cosine)

        return losses

    def configure_callbacks(self):
        """Configure callbacks, honouring ``train.checkpoint_interval_unit``.

        The inherited NVDINOv2 implementation wires the periodic checkpoint with
        ``every_n_epochs=checkpoint_interval`` only -- it never reads
        ``checkpoint_interval_unit``, unlike the core base class, which supports both. That is
        harmless for NVDINOv2's own recipes, but it makes step-unit checkpointing silently
        no-op: a spec asking for "every 1000 steps" becomes "every 1000 *epochs*", i.e. never.

        On cw-pdx that is not a cosmetic difference. GPU jobs are capped at 4 hours and one
        Foxconn epoch is ~4,668 optimizer steps at ~7.7 s/step (~10 h), so an epoch boundary is
        never reached inside a single allocation. With epoch-only checkpointing a requeued arm
        would restart from step 0 every 4 hours and never finish -- and the periodic callback
        is also what writes the full Lightning checkpoint that resume needs, so without it
        there is nothing to resume *from*. Step-unit checkpointing is what makes the
        requeue+resume loop viable at all.

        The periodic callback is rebuilt through its public constructor rather than by poking
        Lightning's private ``_every_n_*`` attributes, so this does not depend on Lightning
        internals.

        Returns:
            Sequence[Callback]: the inherited callbacks, with the periodic checkpoint
            re-wired to step cadence when the spec asks for it.
        """
        callbacks = super().configure_callbacks()

        unit = self.experiment_spec["train"].get("checkpoint_interval_unit", "epoch")
        if unit != "step":
            return callbacks

        interval = self.experiment_spec["train"]["checkpoint_interval"]
        results_dir = self.experiment_spec["results_dir"]

        rebuilt = []
        for callback in callbacks:
            # The periodic checkpoint is the unmonitored, keep-everything one. The best-metric
            # callback (when enabled) is monitored, and TAOExceptionCheckpoint is a different
            # class, so neither is touched.
            if (isinstance(callback, CustomModelCheckpoint) and
                    callback.monitor is None and
                    callback.save_top_k == -1):
                rebuilt.append(CustomModelCheckpoint(
                    every_n_train_steps=interval,
                    every_n_epochs=None,
                    dirpath=results_dir,
                    save_on_train_epoch_end=False,
                    monitor=None,
                    save_top_k=-1,
                    save_last="link",
                    filename="model_{epoch:03d}_{step:05d}",
                    enable_version_counter=False,
                ))
                logging.info(
                    "DINOv3: checkpointing every %d steps (checkpoint_interval_unit='step').",
                    interval,
                )
            else:
                rebuilt.append(callback)
        return rebuilt
