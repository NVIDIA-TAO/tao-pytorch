# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOV2 Module"""

from functools import partial
import math
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from timm.models.vision_transformer import Attention

from .dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    """
    This function traverses a PyTorch module tree and applies the given function
    to each module. It can traverse in either depth-first or breadth-first order
    and can optionally include the root module.

    Args:
        fn (Callable): Function to apply to each module. Should accept keyword
            arguments 'module' and 'name'.
        module (nn.Module): Root module to start traversal from.
        name (str, optional): Name prefix for the current module. Defaults to "".
        depth_first (bool, optional): If True, apply function after visiting children
            (depth-first). If False, apply function before visiting children
            (breadth-first). Defaults to True.
        include_root (bool, optional): Whether to apply function to the root module.
            Defaults to False.

    Returns:
        nn.Module: The input module (for chaining).

    Note:
        - The function is applied to all child modules recursively
        - Module names are constructed by joining parent and child names with "."
        - Useful for weight initialization, parameter counting, or other module-wide operations
    """
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    """
    This module groups multiple transformer blocks together for efficient
    processing, particularly useful for gradient checkpointing and distributed
    training scenarios.

    Attributes:
        Inherits from nn.ModuleList, containing a sequence of transformer blocks.
    """

    def forward(self, x):
        """Forward pass through all blocks in the chunk.

        Args:
            x (torch.Tensor): Input tensor to process through the block chunk.

        Returns:
            torch.Tensor: Output tensor after processing through all blocks.
        """
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    """
    This module implements the DINOv2 Vision Transformer architecture, which is
    a self-supervised learning model based on Vision Transformers. It includes
    features like stochastic depth, layer scaling, and efficient attention mechanisms.

    The model processes images by:
    1. Dividing them into patches
    2. Embedding patches into tokens
    3. Processing through transformer blocks
    4. Outputting features for downstream tasks

    Attributes:
        patch_embed (PatchEmbed): Patch embedding layer.
        pos_drop (nn.Dropout): Positional embedding dropout.
        blocks (nn.ModuleList): List of transformer blocks.
        norm (nn.LayerNorm): Final layer normalization.
        head (nn.Linear): Classification head (if applicable).
        num_features (int): Number of output features.
        num_register_tokens (int): Number of register tokens.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """Initialize the DINOv2 Vision Transformer.

        Args:
            img_size (int, tuple, optional): Input image size. Can be a single integer
                (square image) or a tuple (height, width). Defaults to 224.
            patch_size (int, tuple, optional): Size of each patch. Can be a single
                integer (square patches) or a tuple (height, width). Defaults to 16.
            in_chans (int, optional): Number of input image channels. Defaults to 3.
            embed_dim (int, optional): Embedding dimension for tokens. Defaults to 768.
            depth (int, optional): Number of transformer blocks. Defaults to 12.
            num_heads (int, optional): Number of attention heads. Defaults to 12.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding
                dimension. Defaults to 4.0.
            qkv_bias (bool, optional): Whether to use bias in QKV projection.
                Defaults to True.
            ffn_bias (bool, optional): Whether to use bias in feed-forward network.
                Defaults to True.
            proj_bias (bool, optional): Whether to use bias in attention projection.
                Defaults to True.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.0.
            drop_path_uniform (bool, optional): Whether to apply uniform drop rate
                across blocks. Defaults to False.
            init_values (float, optional): Layer scale initialization values.
                None or 0 means no layer scaling. Defaults to None.
            embed_layer (nn.Module, optional): Patch embedding layer class.
                Defaults to PatchEmbed.
            act_layer (nn.Module, optional): Activation function for MLP.
                Defaults to nn.GELU.
            block_fn (nn.Module, optional): Transformer block class. Defaults to Block.
            ffn_layer (str, optional): Type of feed-forward layer. Options:
                "mlp", "swiglu", "swiglufused", or "identity". Defaults to "mlp".
            block_chunks (int, optional): Number of blocks to group together for
                efficient processing. Defaults to 1.
            num_register_tokens (int, optional): Number of additional register tokens.
                Defaults to 0.
            interpolate_antialias (bool, optional): Whether to apply anti-aliasing
                when interpolating positional embeddings. Defaults to False.
            interpolate_offset (float, optional): Offset for positional embedding
                interpolation. Defaults to 0.1.

        Note:
            - The model supports various Vision Transformer configurations
            - Supports efficient attention mechanisms through block_fn parameter
            - Includes stochastic depth for regularization
            - Supports layer scaling for training stability
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i: i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        """init_weights for DINOVisionTransformer."""
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        """Interpolate positional encoding.

        Args:
            x (torch.Tensor): Input tensor for positional encoding.
            w (int): Width of the input tensor.
            h (int): Height of the input tensor.

        Returns:
            torch.Tensor: Interpolated positional encoding.
        """
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        # DINOv2 with register modify the interpolate_offset from 0.1 to 0.0
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        # w0, h0 = w0 + 0.1, h0 + 0.1

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            # (int(w0), int(h0)), # to solve the upsampling shape issue
            mode="bicubic",
            antialias=self.interpolate_antialias
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        """preapre tokens with masks.

        Args:
            x (torch.Tensor): Input tensor.
            masks (torch.Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Prepared tokens with masks.
        """
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        """forward to process list of inputs

        Args:
            x_list (list): List of input tensors.
            masks_list (list): List of mask tensors.

        Returns:
            list: List of output dictionaries.
        """
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1: self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        """forward to process single input

        Args:
            x (torch.Tensor): Input tensor.
            masks (torch.Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            dict: Dictionary containing processed features.
        """
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1: self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        """Get intermediate layers when input is not chunked.

        Args:
            x (torch.Tensor): Input tensor.
            n (int, optional): Number of layers to take. Defaults to 1.

        Returns:
            list: List of intermediate layers.
        """
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        """Get intermediate layers when input is chunked.

        Args:
            x (torch.Tensor): Input tensor.
            n (int, optional): Number of layers to take. Defaults to 1.

        Returns:
            list: List of intermediate layers.
        """
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """Get intermediate layers for DINOVisionTransformer.

        Args:
            x (torch.Tensor): Input tensor.
            n (int, optional): Number of layers to take. Defaults to 1.
            reshape (bool, optional): Whether to reshape the output. Defaults to False.
            return_class_token (bool, optional): Whether to return the class token. Defaults to False.
            norm (bool, optional): Whether to normalize the output. Defaults to True.

        Returns:
            tuple: Tuple of intermediate layers.
        """
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        """Forward for DINOVisionTransformer.

        Args:
            *args: Variable length argument list.
            is_training (bool, optional): Whether to return the training output. Defaults to False.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Output tensor.
        """
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """Initialize weights using the original timm ViT initialization scheme.

    Args:
        module (nn.Module): Module to initialize weights for.
        name (str, optional): Name of the module (for debugging). Defaults to "".

    Note:
        - Uses trunc_normal_ with std=0.02 for linear layer weights
        - Initializes biases to zero if they exist
        - This is the standard initialization scheme for Vision Transformers
    """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, export=False, **kwargs):
    """Build a ViT-Small model with DINOv2 configuration.

    Args:
        patch_size (int, optional): Size of image patches. Defaults to 16.
        num_register_tokens (int, optional): Number of additional register tokens.
            Defaults to 0.
        export (bool, optional): Whether to export the model. Defaults to False.
        **kwargs: Additional arguments passed to DinoVisionTransformer.

    Returns:
        DinoVisionTransformer: Configured ViT-Small model with:
            - 384 embedding dimensions
            - 12 transformer blocks
            - 6 attention heads
            - 4x MLP ratio
            - Efficient attention mechanisms

    Note:
        - Uses MemEffAttention for memory-efficient attention computation
        - Suitable for medium-resolution images and moderate computational budgets
    """
    if export:
        block_fn = partial(Block, attn_class=Attention)
    else:
        block_fn = partial(Block, attn_class=MemEffAttention)

    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=block_fn,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, export=False, **kwargs):
    """Build a ViT-Base model with DINOv2 configuration.

    Args:
        patch_size (int, optional): Size of image patches. Defaults to 16.
        num_register_tokens (int, optional): Number of additional register tokens.
            Defaults to 0.
        export (bool, optional): Whether to export the model. Defaults to False.
        **kwargs: Additional arguments passed to DinoVisionTransformer.

    Returns:
        DinoVisionTransformer: Configured ViT-Base model with:
            - 768 embedding dimensions
            - 12 transformer blocks
            - 12 attention heads
            - 4x MLP ratio
            - Efficient attention mechanisms

    Note:
        - Uses MemEffAttention for memory-efficient attention computation
        - Standard model size for most applications
        - Good balance between performance and computational cost
    """
    if export:
        block_fn = partial(Block, attn_class=Attention)
    else:
        block_fn = partial(Block, attn_class=MemEffAttention)

    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=block_fn,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, export=False, **kwargs):
    """Build a ViT-Large model with DINOv2 configuration.

    Args:
        patch_size (int, optional): Size of image patches. Defaults to 16.
        num_register_tokens (int, optional): Number of additional register tokens.
            Defaults to 0.
        export (bool, optional): Whether to export the model. Defaults to False.
        **kwargs: Additional arguments passed to DinoVisionTransformer.

    Returns:
        DinoVisionTransformer: Configured ViT-Large model with:
            - 1024 embedding dimensions
            - 24 transformer blocks
            - 16 attention heads
            - 4x MLP ratio
            - Efficient attention mechanisms

    Note:
        - Uses MemEffAttention for memory-efficient attention computation
        - Higher capacity model for demanding applications
        - Requires more computational resources and memory
    """
    if export:
        block_fn = partial(Block, attn_class=Attention)
    else:
        block_fn = partial(Block, attn_class=MemEffAttention)

    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=block_fn,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, export=False, **kwargs):
    """Build a ViT-Giant2 model with DINOv2 configuration.

    Args:
        patch_size (int, optional): Size of image patches. Defaults to 16.
        num_register_tokens (int, optional): Number of additional register tokens.
            Defaults to 0
        export (bool, optional): Whether to export the model. Defaults to False.
        **kwargs: Additional arguments passed to DinoVisionTransformer.

    Returns:
        DinoVisionTransformer: Configured ViT-Giant2 model with:
            - 1536 embedding dimensions
            - 40 transformer blocks
            - 24 attention heads (64 dims per head)
            - 4x MLP ratio
            - Efficient attention mechanisms

    Note:
        - Uses MemEffAttention for memory-efficient attention computation
        - Highest capacity model for maximum performance
        - Requires significant computational resources and memory
        - Uses SwiGLU fused feed-forward network for efficiency
    """
    if export:
        block_fn = partial(Block, attn_class=Attention)
    else:
        block_fn = partial(Block, attn_class=MemEffAttention)

    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=block_fn,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def DINOV2(model_name, export=False):
    """Build a DINOv2 model based on the specified model name.

    Args:
        model_name (str): Name of the model to build. Options:
            - "vits": ViT-Small (384 dims, 12 blocks, 6 heads)
            - "vitb": ViT-Base (768 dims, 12 blocks, 12 heads)
            - "vitl": ViT-Large (1024 dims, 24 blocks, 16 heads)
            - "vitg": ViT-Giant2 (1536 dims, 40 blocks, 24 heads)
        export (bool, optional): Whether to export the model. Defaults to False.

    Returns:
        DinoVisionTransformer: Configured DINOv2 model with:
            - 518x518 input image size
            - 14x14 patch size
            - Layer scaling (init_values=1.0)
            - Appropriate feed-forward layer type
            - No block chunking for FSDP
            - No register tokens
            - Standard interpolation settings

    Raises:
        KeyError: If the specified model_name is not in the model zoo.

    Note:
        - All models use the same DINOv2 training configuration
        - ViT-Giant2 uses SwiGLU fused feed-forward network
        - Other models use standard MLP feed-forward network
        - Models are optimized for self-supervised learning tasks
    """
    model_zoo = {
        "vits": vit_small,
        "vitb": vit_base,
        "vitl": vit_large,
        "vitg": vit_giant2
    }

    return model_zoo[model_name](
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp" if model_name != "vitg" else "swiglufused",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        export=export
    )
