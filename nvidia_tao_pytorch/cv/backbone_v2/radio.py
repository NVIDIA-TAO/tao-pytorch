# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RADIO backbone."""

import math
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models import checkpoint_seq
from timm.models.vision_transformer import Mlp, VisionTransformer

from nvidia_tao_pytorch.cv.backbone_v2 import BACKBONE_REGISTRY
from nvidia_tao_pytorch.cv.backbone_v2.backbone_base import BackboneBase


# Note: The `img_size` is unused since RADIO models replace the `patch_embed` with `ViTPatchGenerator` in `_enable_cpe`
# which supports arbitrary input sizes (please check `RADIOWrapper._validate_input()` for the details).
radio_model_cfg = {
    # CRADIOV1.
    "vit_huge_patch16_224_mlpnorm": {
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
    },
    # CRADIOV2.
    "vit_base_patch16_224": {"img_size": 224, "patch_size": 16, "embed_dim": 768, "depth": 12, "num_heads": 12},
    "vit_large_patch16_224": {"img_size": 224, "patch_size": 16, "embed_dim": 1024, "depth": 24, "num_heads": 16},
    "vit_huge_patch16_224": {"img_size": 224, "patch_size": 16, "embed_dim": 1280, "depth": 32, "num_heads": 16},
    # CRADIOV3.
    "vit_base_patch16_reg4_dinov2": {
        "img_size": 518 * 16 // 14,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "init_values": 1e-5,
        "reg_tokens": 4,
        "no_embed_class": True,
    },
    "vit_large_patch16_reg4_dinov2": {
        "img_size": 518 * 16 // 14,
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "init_values": 1e-5,
        "reg_tokens": 4,
        "no_embed_class": True,
    },
    "vit_huge_patch16_reg4_dinov2": {
        "img_size": 518 * 16 // 14,
        "patch_size": 16,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "init_values": 1e-5,
        "reg_tokens": 4,
        "no_embed_class": True,
    },
    # CRADIOV4 SO400M.
    "vit_so400m_patch16_224": {
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 1152,
        "depth": 27,
        "num_heads": 16,
        "mlp_ratio": 4304 / 1152,
    },
}


def remove_state_dict_prefix(state_dict: Dict[str, Any], prefix: str):
    """Remove a specified prefix from the keys in a state dict.

    Args:
        state_dict (dict): The pretrained model weights.
        prefix (str): The prefix to be removed from the keys.

    Returns:
        Dict[str, Any]: A new state dictionary with the prefix removed from the keys.
    """
    mod_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k.replace(prefix, "", 1)
            mod_state_dict[new_k] = v
        else:
            mod_state_dict[k] = v
    return mod_state_dict


def replace_state_dict_key(state_dict: Dict[str, Any], old_key: str, new_key: str):
    """Replace an old key with a new key in a state dict.

    Args:
        state_dict (dict): The pretrained model weights.
        old_key (str): The old key to be replaced.
        new_key (str): The new key.

    Returns:
        Dict[str, Any]: A new state dictionary with the old key replaced with the new key.
    """
    mod_state_dict = {}
    for k, v in state_dict.items():
        if old_key in k:
            new_k = k.replace(old_key, new_key, 1)
            mod_state_dict[new_k] = v
        else:
            mod_state_dict[k] = v
    return mod_state_dict


class Im2Patches(nn.Module):
    """Image patches module."""

    def __init__(self, patch_size: int):
        """Image patches module."""
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function to transform input image to patches."""
        if self.patch_size == 1:
            patches = x.flatten(2)
            patches = patches.permute(0, 2, 1)
            return patches

        py = x.shape[-2] // self.patch_size
        px = x.shape[-1] // self.patch_size
        patches = rearrange(
            x, "b c (py yy) (px xx) -> b (py px) (c yy xx)", py=py, yy=self.patch_size, px=px, xx=self.patch_size
        )
        return patches


class ViTPatchLinear(nn.Linear):
    """Patch embedding module."""

    def __init__(self, patch_size: int, embed_dim: int, **kwargs):
        """Patch embedding module.

        Args:
            patch_size (int): patch size.
            embed_dim (int): patch embedding dimension.
        """
        super().__init__(3 * (patch_size**2), embed_dim, bias=False, **kwargs)
        self.patch_size = patch_size

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.bias is not None:
            self.bias.data.copy_(state_dict[f"{prefix}bias"])
        chk_weight = state_dict[f"{prefix}weight"]
        if chk_weight.shape != self.weight.shape:
            src_patch_size = int(math.sqrt(chk_weight.shape[1] // 3))
            assert (src_patch_size**2) * 3 == chk_weight.shape[1], "Unable to interpolate non-square patch size"
            chk_weight = rearrange(chk_weight, "b (c h w) -> b c h w", c=3, h=src_patch_size, w=src_patch_size)
            chk_weight = F.interpolate(
                chk_weight,
                size=(self.patch_size, self.patch_size),
                mode="bicubic",
                align_corners=True,
                antialias=False,
            )
            chk_weight = rearrange(chk_weight, "b c h w -> b (c h w)")
        self.weight.data.copy_(chk_weight)


class ClsToken(nn.Module):
    """Class token class for RADIO ViT backbone."""

    def __init__(self, ndim: int, num_tokens: int = 1, enabled: bool = True, register_multiple: int = 0):
        """Class token class for RADIO ViT backbone.

        Args:
            ndim (int): class token dimension.
            num_tokens (int): number of class tokens. Default: `1`.
            enabled (bool): whether to enable class token. Default: `True`.
            register_multiple (int): number of extra tokens. Default: `0`.
        """
        super().__init__()
        self.ndim = ndim
        self.enabled = enabled
        self.num_registers = 0
        self.num_tokens = num_tokens
        if enabled:
            if register_multiple > 0:
                self.num_registers = register_multiple - (num_tokens % register_multiple)

            scale = ndim**-0.5
            self.token = nn.Parameter(torch.randn(num_tokens + self.num_registers, ndim) * scale)
        else:
            self.token = None
        self.num_patches = self.num_tokens + self.num_registers

    def no_weight_decay(self):
        """No weight decay."""
        return ["token"]

    def disable(self):
        """Disable ClsToken."""
        self.token = None
        self.enabled = False

    def forward(self, x: torch.Tensor):
        """Forward function and return the input with class tokens concatenated"""
        if self.token is None:
            return x
        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([token, x], dim=1)


class ViTPatchGenerator(nn.Module):
    """ViT patch generator class for CPE (Cropped Position Embedding) purpose"""

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        input_dims: Union[int, Tuple[int, int]],
        abs_pos: bool = True,
        normalize_patches: bool = False,
        cls_token: bool = False,
        max_input_dims: Optional[Union[int, Tuple[int, int]]] = None,
        pos_dropout: float = 0.0,
        return_pos_enc: bool = False,
        num_cls_tokens: int = 1,
        register_multiple: int = 0,
        device=None,
        dtype=None,
    ):
        """ViT patch generator class for CPE (Cropped Position Embedding) purpose

        Args:
            patch_size (int): patch size
            embed_dim (int): patch embedding dimension
            input_dims (Union[int, Tuple[int, int]]): input dimension
            abs_pos (bool): absolute position embedding. Default: `True`.
            normalize_patches (bool): whether to normalize patch. Default: `False`.
            cls_token (bool): whether to enable class token. Default: `False`.
            max_input_dims (Optional[Union[int, Tuple[int, int]]]): maximum input dimension. Default: `None`.
            pos_dropout (float): position embedding dropout rate. Default: `0.0`.
            return_pos_enc (bool): whether to return position embedding. Default: `False`.
            num_cls_tokens (int): number of class tokens. Default: `1`.
            register_multiple (int): number of extra tokens. Default: `0`.
            device (str): device. Default: `None`.
            dtype (str): data type. Default: `None`.
        """
        super().__init__()

        if isinstance(input_dims, int):
            input_dims = (input_dims, input_dims)
        if max_input_dims is None:
            max_input_dims = input_dims
        if isinstance(max_input_dims, int):
            max_input_dims = (max_input_dims, max_input_dims)
        max_input_dims = tuple(int(math.ceil(d / patch_size) * patch_size) for d in max_input_dims)

        self.cpe_mode = max_input_dims != input_dims
        self.pos_dropout = pos_dropout
        self.return_pos_enc = return_pos_enc
        self.patch_size = patch_size
        self.abs_pos = abs_pos
        self.embed_dim = embed_dim
        self.num_rows = max_input_dims[0] // patch_size
        self.num_cols = max_input_dims[1] // patch_size
        self.input_dims = tuple(d // patch_size for d in input_dims)
        self.num_patches = self.num_rows * self.num_cols
        self.max_input_dims = max_input_dims

        self.im_to_patches = Im2Patches(patch_size)
        self.embedder = ViTPatchLinear(patch_size, embed_dim, device=device, dtype=dtype)
        if abs_pos:
            scale = embed_dim**-0.5
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_patches, embed_dim, device=device, dtype=dtype) * scale
            )
        self.cls_token = ClsToken(
            embed_dim,
            num_tokens=num_cls_tokens,
            enabled=cls_token,
            register_multiple=register_multiple,
        )
        self.patch_normalizer = nn.LayerNorm(embed_dim) if normalize_patches else nn.Identity()

    def forward(self, x: torch.Tensor):
        """Forward function to return the patch embeddings with position embedding applied."""
        patches = self.embed_patches(x)
        patches, pos_enc = self.apply_pos_enc(patches, input_size=x.shape[2:])
        patches = self.cls_token(patches)
        patches = self.patch_normalizer(patches)
        if self.return_pos_enc:
            return patches, pos_enc
        return patches

    @property
    def num_cls_tokens(self):
        """Number of class tokens"""
        return self.cls_token.num_tokens

    @property
    def num_registers(self):
        """Number of register tokens"""
        return self.cls_token.num_registers

    @property
    def num_skip(self):
        """Total number of extra tokens (class tokens + register tokens)"""
        return self.num_cls_tokens + self.num_registers

    def no_weight_decay(self):
        """No weight decay."""
        return ["pos_embed"]

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.abs_pos:
            self._load_embed(state_dict[f"{prefix}pos_embed"], self.pos_embed)

    def _load_embed(self, src_embed: torch.Tensor, targ_embed: nn.Parameter):
        if src_embed.shape != targ_embed.shape:
            src_size = int(math.sqrt(src_embed.shape[1]))
            assert src_size**2 == src_embed.shape[1], "Unable to interpolate non-square embedding"
            src_embed = rearrange(src_embed, "b (h w) c -> b c h w", h=src_size, w=src_size)
            src_embed = F.interpolate(
                src_embed, size=(self.num_rows, self.num_cols), mode="bicubic", align_corners=True, antialias=False
            )
            src_embed = rearrange(src_embed, "b c h w -> b (h w) c")
        targ_embed.data.copy_(src_embed)

    def _load_projection(self, src_proj_weight: torch.Tensor, targ_proj_weight: torch.Tensor):
        if src_proj_weight.shape != targ_proj_weight.shape:
            src_patch_size = int(math.sqrt(src_proj_weight.shape[1] // 3))
            assert (src_patch_size**2) * 3 == src_proj_weight.shape[1], "Unable to interpolate non-square patch size"
            src_proj_weight = rearrange(
                src_proj_weight, "b (c h w) -> b c h w", c=3, h=src_patch_size, w=src_patch_size
            )
            src_proj_weight = F.interpolate(
                src_proj_weight,
                size=(self.patch_size, self.patch_size),
                mode="bicubic",
                align_corners=True,
                antialias=False,
            )
            src_proj_weight = rearrange(src_proj_weight, "b c h w -> b (c h w)")
        targ_proj_weight.data.copy_(src_proj_weight)

    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Embed image patches"""
        patches = self.im_to_patches(x)
        return self.embedder(patches)

    def apply_pos_enc(
        self,
        patches: torch.Tensor,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """Apply position embedding"""
        if not self.abs_pos:
            return patches

        pos_enc = self.get_pos_enc(patches.shape[0], patch_idxs, input_size)
        if self.training and self.pos_dropout > 0:
            keeps = torch.rand(patches.shape[0], 1, 1, dtype=pos_enc.dtype, device=pos_enc.device) > self.pos_dropout
            pos_enc_drop = torch.where(keeps, pos_enc, 0)
        else:
            pos_enc_drop = pos_enc
        return patches + pos_enc_drop, pos_enc

    def get_pos_enc(
        self, batch_size: int, patch_idxs: Optional[torch.Tensor] = None, input_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Get position embedding"""
        if input_size is None:
            input_dims = self.input_dims
        else:
            input_dims = tuple(d // self.patch_size for d in input_size)
        pos_embed = self._get_pos_embeddings(batch_size, input_dims)
        if patch_idxs is None:
            return pos_embed
        exp_patch_idxs = patch_idxs.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1])
        return torch.gather(pos_embed.expand(patch_idxs.shape[0], -1, -1), dim=1, index=exp_patch_idxs)

    def _get_pos_embeddings(self, batch_size: int, input_dims: Tuple[int, int]):
        if (self.num_rows, self.num_cols) == input_dims:
            return self.pos_embed
        pos_embed = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1).permute(0, 3, 1, 2)

        def window_select(pos_embed):
            if input_dims[0] < pos_embed.shape[-2]:
                pos_embed = pos_embed[..., : input_dims[0], :]
            if input_dims[1] < pos_embed.shape[-1]:
                pos_embed = pos_embed[..., :, : input_dims[1]]
            return pos_embed

        if self.cpe_mode:
            if self.training:
                min_scale = math.sqrt(0.1)
                scale = torch.rand(batch_size, 1, 1, device=pos_embed.device) * (1 - min_scale) + min_scale
                aspect_min = math.log(3 / 4)
                aspect_max = -aspect_min
                aspect = torch.exp(
                    torch.rand(batch_size, 1, 1, device=pos_embed.device) * (aspect_max - aspect_min) + aspect_min
                )
                scale_x = scale * aspect
                scale_y = scale * (1 / aspect)
                scale_xy = torch.stack([scale_x, scale_y], dim=-1).clamp_(0, 1)
                pos_xy = torch.rand(batch_size, 1, 1, 2, device=pos_embed.device) * (1 - scale_xy)
                lin_x = torch.linspace(0, 1, steps=input_dims[1], device=pos_embed.device)[None, None].expand(
                    batch_size, input_dims[0], -1
                )
                lin_y = torch.linspace(0, 1, steps=input_dims[0], device=pos_embed.device)[None, :, None].expand(
                    batch_size, -1, input_dims[1]
                )
                lin_xy = torch.stack([lin_x, lin_y], dim=-1)
                grid_xy = lin_xy * scale_xy + pos_xy
                grid_xy.mul_(2).sub_(1)  # Convert to [-1, 1] range.
                pos_embed = F.grid_sample(
                    pos_embed.float().expand(batch_size, -1, -1, -1),
                    grid=grid_xy,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).to(pos_embed.dtype)
            else:
                max_dim = max(input_dims)
                pos_embed = F.interpolate(
                    pos_embed.float(), size=(max_dim, max_dim), align_corners=True, mode="bilinear"
                ).to(pos_embed.dtype)
                pos_embed = window_select(pos_embed)
        else:
            pos_embed = window_select(pos_embed)
        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(pos_embed.float(), size=input_dims, align_corners=True, mode="bilinear").to(
                pos_embed.dtype
            )
        return pos_embed.flatten(2).permute(0, 2, 1)


class RADIOBase(nn.Module):
    """RADIO base class."""

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 0,
        resolution: Tuple[int, int] = (224, 224),
        backbone: str = "vit_base_patch16_224",
        summary_idxs: Optional[List[int]] = None,
        window_size: Optional[int] = None,
        num_teacher: int = 4,
        cpe_max_size: int = 2048,
        register_multiple: int = 8,
    ):
        """Initialize the RADIO base model.

        Args:
            in_chans (int): Number of input image channels. Default: `3`.
            num_classes (int): Number of classes for classification head. Default: `0`.
            resolution (tuple): Input resolution. Default: `(224, 224)`.
            backbone (str): Name of the ViT backbone. Default: `"vit_base_patch16_224"`.
            summary_idxs (list): Indices of the summary tokens. Default: `None`.
            window_size (int): Window size for windowed attention. Default: `None`.
            num_teacher (int): Number of teachers. Default: `4`.
            cpe_max_size (int): Maximum size of the cropped positional embedding. Default: `2048`.
            register_multiple (int): Number of extra tokens. Default: `8`.
        """
        super().__init__()
        self.in_chans = int(in_chans)
        self.num_classes = int(num_classes)
        self.resolution = resolution
        self.backbone = str(backbone)
        self.summary_idxs = summary_idxs
        self._window_size = window_size
        self.num_teacher = int(num_teacher)
        self.cpe_max_size = int(cpe_max_size)
        self.register_multiple = int(register_multiple)

        # Instantiate the RADIO backbone using TIMM.
        model_cfg = radio_model_cfg.get(self.backbone, None)
        if model_cfg is None:
            raise ValueError(
                f"Unsupported backbone: {self.backbone}. Supported backbones are: {list(radio_model_cfg.keys())}"
            )
        vit_backbone = VisionTransformer(
            in_chans=self.in_chans,
            num_classes=self.num_classes,
            drop_rate=0.0,
            drop_path_rate=0.0,
            global_pool="token",
            weight_init="skip",
            **model_cfg,
        )
        # CRADIOV1 adds `nn.LayerNorm` to `Mlp` layers.
        if self.backbone == "vit_huge_patch16_224_mlpnorm":
            for m in vit_backbone.modules():
                if isinstance(m, Mlp) and not isinstance(m.norm, nn.LayerNorm):
                    m.norm = nn.LayerNorm(m.fc1.out_features)
        # CRADIO models replace `vit_backbone.norm` and `vit_backbone.head` with `nn.Identity()`.
        if hasattr(vit_backbone, "norm"):
            vit_backbone.norm = nn.Identity()
        vit_backbone.head = nn.Identity()

        # Enable cropped position embedding.
        # resolution is nominal for init; ViTPatchGenerator.forward uses x.shape[2:] for pos enc and supports variable input size
        vit_backbone = self._enable_cpe(
            vit_backbone,
            resolution=self.resolution,
            max_img_size=self.cpe_max_size,
            num_cls_tokens=self.num_teacher,
            register_multiple=self.register_multiple,
        )
        self.model = vit_backbone
        self.num_features = vit_backbone.embed_dim * len(self.summary_idxs)

    @property
    def num_summary_tokens(self) -> int:
        """Number of all extra tokens (class tokens + register tokens)"""
        return self.model.patch_generator.num_skip

    @property
    def patch_size(self) -> int:
        """Patch size"""
        return self.model.patch_generator.patch_size

    @property
    def window_size(self) -> int:
        """Window size for windowed attetion"""
        return self._window_size

    def _enable_cpe(
        self,
        model: VisionTransformer,
        resolution: Tuple[int, int] = (224, 224),
        max_img_size: Union[int, Tuple[int, int]] = 2048,
        num_cls_tokens: int = 4,
        pos_dropout: float = 0.1,
        register_multiple: int = 8,
    ):
        """Enable cropped position embedding (CPE) for the ViT model.

        Args:
            model (VisionTransformer): ViT model.
            resolution (tuple): Input resolution. Default: `(224, 224)`.
            max_img_size (tuple): Maximum image size. Default: `2048`.
            num_cls_tokens (int): Number of class tokens. Default: `4`.
            pos_dropout (float): Dropout rate of the position embedding. Default: `0.1`.
            register_multiple (int): Number of extra tokens. Default: `8`.
        """
        if not isinstance(model, VisionTransformer):
            raise ValueError(f"CPE only supports for VisionTransformer model. Received: {type(model)}")

        patch_size = model.patch_embed.patch_size[0]
        embed_dim = model.embed_dim
        normalize_patches = not isinstance(model.patch_embed.norm, nn.Identity)
        cls_token = model.cls_token is not None
        max_img_size = int(round(max_img_size / patch_size) * patch_size)

        # input_dims: nominal resolution for init; forward uses actual x.shape[2:] in apply_pos_enc (variable resolution)
        patch_generator = ViTPatchGenerator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            input_dims=resolution,
            normalize_patches=normalize_patches,
            cls_token=cls_token,
            max_input_dims=max_img_size,
            pos_dropout=pos_dropout,
            num_cls_tokens=num_cls_tokens,
            register_multiple=register_multiple,
        )

        model.patch_generator = patch_generator
        model.patch_embed = None
        model.cls_token = None
        model.pos_embed = None
        model.pos_drop = None
        model.patch_size = patch_size
        if hasattr(model, "reg_token"):
            model.reg_token = None
        model.num_cls_tokens = num_cls_tokens
        model.num_registers = patch_generator.num_registers

        # Replace the forward function to use the new patch generator.
        def _forward_cpe(self: VisionTransformer, x: torch.Tensor):
            """Forward function for CPE."""
            x = self.patch_generator(x)
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x)
            else:
                x = self.blocks(x)
            return self.norm(x)

        model.forward_features = types.MethodType(_forward_cpe, model)
        return model

    def forward(self, x: torch.Tensor):
        """Forward."""
        y = self.model.forward_features(x)
        patch_gen = self.model.patch_generator
        all_summary = y[:, : patch_gen.num_cls_tokens]
        if self.summary_idxs is not None:
            bb_summary = all_summary[:, self.summary_idxs]
        else:
            bb_summary = all_summary
        bb_summary = bb_summary.flatten(1)
        all_feat = y[:, patch_gen.num_skip:]
        return bb_summary, all_feat


class RADIOWrapper(nn.Module):
    """RADIO model wrapper."""

    def __init__(self, model: RADIOBase, resolution: tuple = None):
        """RADIO model wrapper.

        Args:
            model (VisionTransformer): RADIO model.
            resolution (tuple, optional): input resolution. Default: `None`.
        """
        super().__init__()
        self.radio = model
        if resolution is not None:
            self._validate_input(resolution)

    @property
    def min_resolution_step(self) -> int:
        """Minimum acceptable patch size."""
        res = self.radio.patch_size
        if self.radio.window_size is not None:
            res *= self.radio.window_size
        return res

    def _get_nearest_supported_resolution(self, height: int, width: int):
        height = int(round(height / self.min_resolution_step) * self.min_resolution_step)
        width = int(round(width / self.min_resolution_step) * self.min_resolution_step)
        height = max(height, self.min_resolution_step)
        width = max(width, self.min_resolution_step)
        return height, width

    def _validate_input(self, resolution):
        res_step = self.min_resolution_step
        if res_step is not None and (resolution[0] % res_step != 0 or resolution[1] % res_step != 0):
            raise ValueError(
                "The input resolution must be a multiple of `self.min_resolution_step`. "
                f"Input: {resolution}, Nearest: {self._get_nearest_supported_resolution(resolution[0], resolution[1])}"
            )

    def forward(self, x):
        """Forward."""
        return self.radio(x)


class RADIO(BackboneBase):
    """RADIO model.

    RADIO, a new vision foundation model, excels across visual domains, serving as a superior replacement for vision
    backbones. Integrating CLIP variants, DINOv2, and SAM through distillation, it preserves unique features like text
    grounding and segmentation correspondence.

    References:
    - [AM-RADIO: Agglomerative Vision Foundation Model -- Reduce All Domains Into One](
      https://arxiv.org/abs/2312.06709)
    - [RADIOv2.5: Improved Baselines for Agglomerative Vision Foundation Models](
      https://arxiv.org/abs/2412.07679)
    - [https://github.com/NVlabs/RADIO](https://github.com/NVlabs/RADIO)
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 0,
        resolution: Tuple[int, int] = (224, 224),
        backbone: str = "vit_base_patch16_224",
        summary_idxs: Optional[List[int]] = None,
        window_size: Optional[int] = None,
        num_teacher: int = 4,
        cpe_max_size: int = 2048,
        register_multiple: int = 8,
        activation_checkpoint=False,
        freeze_at=None,
        freeze_norm=False,
        head_init_scale=1.0,
        **kwargs,
    ):
        """Initialize the RADIO model.

        Args:
            in_chans (int): Number of input image channels. Default: `3`.
            num_classes (int): Number of classes for classification head. Default: `0`.
            resolution (tuple): Input resolution. Default: `(224, 224)`.
            backbone (str): Name of the ViT backbone. Default: `"vit_base_patch16_224"`.
            summary_idxs (list): Indices of the summary tokens. Default: `None`.
            window_size (int): Window size for windowed attention. Default: `None`.
            num_teacher (int): Number of teachers. Default: `4`.
            cpe_max_size (int): Maximum size of the cropped positional embedding. Default: `2048`.
            register_multiple (int): Number of extra tokens. Default: `8`.
            activation_checkpoint (bool): Whether to use activation checkpointing. Default: `False`.
            freeze_at (list): List of keys corresponding to the stages or layers to freeze. If `None`, no specific
                layers are frozen. If `"all"`, the entire model is frozen and set to eval mode. Default: `None`.
            freeze_norm (bool): If `True`, all normalization layers in the backbone will be frozen. Default: `False`.
            head_init_scale (float): Initialization scale for the head. Default: `1.0`.
        """
        super().__init__(
            in_chans=in_chans,
            num_classes=num_classes,
            activation_checkpoint=activation_checkpoint,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
        )
        self.resolution = resolution
        self.backbone = str(backbone)
        self.summary_idxs = summary_idxs
        self._window_size = window_size
        self.num_teacher = int(num_teacher)
        self.cpe_max_size = int(cpe_max_size)
        self.register_multiple = int(register_multiple)

        # Determine this is CRADIOV1 or CRADIOV2.
        if self.backbone == "vit_huge_patch16_224_mlpnorm":
            self.radio_version = "CRADIOV1"
        else:
            self.radio_version = "CRADIOV2"

        # Instantiate the RADIO base model.
        backbone = RADIOBase(
            in_chans=self.in_chans,
            num_classes=self.num_classes,
            resolution=self.resolution,
            backbone=self.backbone,
            summary_idxs=self.summary_idxs,
            window_size=self._window_size,
            num_teacher=self.num_teacher,
            cpe_max_size=self.cpe_max_size,
            register_multiple=self.register_multiple,
        )
        self.num_features = backbone.num_features
        self.patch_size = backbone.patch_size
        # Add an extra wrapper to the backbone.
        # TODO(@hongyuc): This is actually a redundant wrapper for the RADIO models. We can remove it in the future.
        self.radio = RADIOWrapper(backbone, resolution=self.resolution)
        if num_classes > 0:
            self.head = nn.Linear(self.num_features, num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Set the gradient checkpointing for the model."""
        self.radio.radio.model.set_grad_checkpointing(enable)

    def load_state_dict(self, state_dict, **kwargs):
        """Copy parameters and buffers from state_dict into this module and its descendants.

        Args:
            state_dict (dict): a dict containing parameters and persistent buffers.
            **kwargs: Additional arguments passed to `nn.Module.load_state_dict`.
        """
        state_dict = remove_state_dict_prefix(
            remove_state_dict_prefix(remove_state_dict_prefix(state_dict, "radio_model.model."), "base_model."),
            "radio.radio.model.",
        )
        if self.radio_version == "CRADIOV1":
            return self.radio.radio.model.load_state_dict(state_dict, **kwargs)
        elif self.radio_version == "CRADIOV2":
            return self.radio.radio.model.load_state_dict(
                # Typo in the V3 safetensors checkpoint from HF.
                replace_state_dict_key(state_dict, old_key="grandma", new_key="gamma"),
                **kwargs,
            )
        else:
            raise NotImplementedError(
                f"Unsupported RADIO version: {self.radio_version}. Supported versions are: CRADIOV1, CRADIOV2"
            )

    def get_stage_dict(self):
        """Get the stage dictionary."""
        stage_dict = {0: self.radio.radio.model.patch_generator}
        for i, block in enumerate(self.radio.radio.model.blocks, start=1):
            stage_dict[i] = block
        return stage_dict

    @torch.jit.ignore
    def get_classifier(self):
        """Get the classification head module.

        Returns:
            nn.Module: The classification head (Linear layer or Identity).
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        """Reset the classification head with a new number of classes.

        Args:
            num_classes (int): New number of classes for classification.
            global_pool (str, optional): Global pooling type (unused in current implementation).
                Defaults to "".
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_pre_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the backbone, excluding the head.

        Args:
            x (Tensor): Input tensor.

        Returns:
            summary (Tensor): Summary tensor.
            features (Tensor): Features tensor.
        """
        return self.radio(x)

    def forward_feature_pyramid(self, x):
        """Forward pass through the backbone to extract intermediate feature maps."""
        _, spatial_features = self.radio(x)
        B, _, C = spatial_features.shape
        assert C == self.num_features // len(self.summary_idxs), \
            f"Number of features mismatch: {C} != {self.num_features // len(self.summary_idxs)}"
        # [B, L, C] -> [B, C, H, W]
        H, W = self.resolution[0] // self.patch_size, self.resolution[1] // self.patch_size
        spatial_features = spatial_features.permute(0, 2, 1).view(B, C, H, W)
        return spatial_features

    def forward(self, x: torch.Tensor, return_features: bool = False, return_logits: bool = False) -> torch.Tensor:
        """Forward.

        Args:
            x (Tensor): Input tensor.
            return_features (bool): Whether to return the spatial features.
            return_logits (bool): Whether to return the summary logits.
        Returns:
            summary (Tensor): Summary tensor.
        """
        summary, spatial_features = self.radio(x)
        if return_features:
            B, _, C = spatial_features.shape
            assert C == self.num_features // len(self.summary_idxs), \
                f"Number of features mismatch: {C} != {self.num_features // len(self.summary_idxs)}"
            # [B, L, C] -> [B, C, H, W]; use actual input size so variable resolution works (e.g. multiview/stochastic)
            H = x.shape[2] // self.patch_size
            W = x.shape[3] // self.patch_size
            spatial_features = spatial_features.permute(0, 2, 1).view(B, C, H, W)
            return summary, spatial_features
        else:
            if return_logits:
                return summary
            else:
                return self.head(summary)


@BACKBONE_REGISTRY.register()
def c_radio_p1_vit_huge_patch16_mlpnorm(**kwargs):
    """CRADIO P1 ViT Huge Patch16 MLPNorm."""
    return RADIO(
        backbone="vit_huge_patch16_224_mlpnorm",
        summary_idxs=[0, 1, 2],
        window_size=None,
        num_teacher=3,
        cpe_max_size=2048,
        register_multiple=16,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def c_radio_p2_vit_huge_patch16_mlpnorm(**kwargs):
    """CRADIO P2 ViT Huge Patch16 MLPNorm."""
    return RADIO(
        backbone="vit_huge_patch16_224_mlpnorm",
        summary_idxs=[0, 1, 2, 3],
        window_size=None,
        num_teacher=4,
        cpe_max_size=2048,
        register_multiple=16,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def c_radio_p3_vit_huge_patch16_mlpnorm(**kwargs):
    """CRADIO P3 ViT Huge Patch16 MLPNorm."""
    return RADIO(
        backbone="vit_huge_patch16_224_mlpnorm",
        summary_idxs=[0, 1, 2],
        window_size=None,
        num_teacher=4,
        cpe_max_size=2048,
        register_multiple=16,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def c_radio_v2_vit_base_patch16(**kwargs):
    """CRADIOV2 ViT Base Patch16 MLPNorm."""
    return RADIO(
        backbone="vit_base_patch16_224",
        summary_idxs=[0, 1, 2],
        window_size=None,
        num_teacher=4,
        cpe_max_size=2048,
        register_multiple=8,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def c_radio_v2_vit_large_patch16(**kwargs):
    """CRADIOV2 ViT Large Patch16 MLPNorm."""
    return RADIO(
        backbone="vit_large_patch16_224",
        summary_idxs=[0, 1, 2],
        window_size=None,
        num_teacher=4,
        cpe_max_size=2048,
        register_multiple=8,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def c_radio_v2_vit_huge_patch16(**kwargs):
    """CRADIOV2 ViT Huge Patch16 MLPNorm."""
    return RADIO(
        backbone="vit_huge_patch16_224",
        summary_idxs=[0, 1, 2],
        window_size=None,
        num_teacher=4,
        cpe_max_size=2048,
        register_multiple=8,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def c_radio_v3_vit_base_patch16_reg4_dinov2(**kwargs):
    """CRADIOV3 ViT Base Patch16 Reg4."""
    return RADIO(
        backbone="vit_base_patch16_reg4_dinov2",
        summary_idxs=[0, 1, 2],
        window_size=None,
        num_teacher=4,
        cpe_max_size=2048,
        register_multiple=8,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def c_radio_v3_vit_large_patch16_reg4_dinov2(**kwargs):
    """CRADIOV3 ViT Large Patch16 Reg4."""
    return RADIO(
        backbone="vit_large_patch16_reg4_dinov2",
        summary_idxs=[0, 1, 2],
        window_size=None,
        num_teacher=4,
        cpe_max_size=2048,
        register_multiple=8,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def c_radio_v3_vit_huge_patch16_reg4_dinov2(**kwargs):
    """CRADIOV3 ViT Huge Patch16 Reg4."""
    return RADIO(
        backbone="vit_huge_patch16_reg4_dinov2",
        summary_idxs=[0, 1, 2],
        window_size=None,
        num_teacher=4,
        cpe_max_size=2048,
        register_multiple=8,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def c_radio_v4_vit_huge_patch16(**kwargs):
    """CRADIOV4 ViT Huge Patch16.

    The released v4 checkpoint has one CLS token per teacher, but only SigLIP2
    and DINOv3 are exposed as backbone summary tokens; SAM3 is spatial/adaptor
    only.
    """
    return RADIO(
        backbone="vit_huge_patch16_224",
        summary_idxs=[0, 1],
        window_size=None,
        num_teacher=3,
        cpe_max_size=2048,
        register_multiple=10,
        **kwargs,
    )


@BACKBONE_REGISTRY.register()
def c_radio_v4_vit_so400m_patch16(**kwargs):
    """CRADIOV4 SigLIP-SO400M Patch16.

    The released v4 checkpoint has one CLS token per teacher, but only SigLIP2
    and DINOv3 are exposed as backbone summary tokens; SAM3 is spatial/adaptor
    only.
    """
    return RADIO(
        backbone="vit_so400m_patch16_224",
        summary_idxs=[0, 1],
        window_size=None,
        num_teacher=3,
        cpe_max_size=2048,
        register_multiple=10,
        **kwargs,
    )
