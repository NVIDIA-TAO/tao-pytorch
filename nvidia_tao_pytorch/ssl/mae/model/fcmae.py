# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FCMAE models."""

import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from nvidia_tao_pytorch.cv.backbone_v2.convnext_v2 import Block
from nvidia_tao_pytorch.cv.backbone_v2.nn.norm import LayerNorm2d


class MAEConvNeXtV2(nn.Module):
    """ Sparse ConvNeXtV2."""

    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.):
        """Init.
        Args:
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
            dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
            drop_path_rate (float): Stochastic depth rate. Default: 0.
            head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        """
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def upsample_mask(self, mask, scale):
        """Upsample mask."""
        assert len(mask.shape) == 2, "Dimension of mask should be 2."
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)

    def forward(self, x, mask):
        """Forward."""
        num_stages = len(self.stages)
        mask = self.upsample_mask(mask, 2 ** (num_stages - 1))
        mask = mask.unsqueeze(1).type_as(x)

        # patch embedding
        x = self.downsample_layers[0](x)
        x *= (1. - mask)

        # encoding
        for i in range(4):
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages[i](x)
        return x


class FCMAE(nn.Module):
    """ Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone."""

    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 decoder_depth=1,
                 decoder_embed_dim=512,
                 patch_size=32,
                 mask_ratio=0.6,
                 export=False,
                 norm_pix_loss=False):
        """Init.
        Args:
            img_size: Input image size.
            in_chans: Number of image input channels.
            dims: Feature dimension at each stage.
            depths: Number of blocks at each stage.
            decoder_embed_dim: Decoder embedding dimension.
            decoder_depth: Decoder depth.
            norm_pix_loss: Whether to normalize pix_loss
            mask_ratio: Masking ratio.
        """
        super().__init__()

        # configs
        self.img_size = img_size
        self.depths = depths
        self.dims = dims
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss
        self.export = export

        # encoder
        self.encoder = MAEConvNeXtV2(
            in_chans=in_chans, depths=depths, dims=dims)
        # decoder
        self.proj = nn.Conv2d(
            in_channels=dims[-1],
            out_channels=decoder_embed_dim,
            kernel_size=1)
        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Block(
            dim=decoder_embed_dim,
            drop_path=0.) for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        # pred
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size ** 2 * in_chans,
            kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):
            torch.nn.init.normal_(self.mask_token, std=.02)

    def patchify(self, imgs):
        """Patchify.
        Args:
            imgs: (N, 3, H, W)
        Returns:
            x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0, \
            "image height and weight must be equal and they can be divided by patch_size."

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """Unpatchify.
        Args:
            x: (N, L, patch_size**2 *3)
        Returns:
            imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1], "length of the input must be a perfect square."

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def gen_random_mask(self, x, mask_ratio):
        """Generate random mask."""
        N = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 2
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def upsample_mask(self, mask, scale):
        """Upsample mask."""
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p). \
            repeat_interleave(scale, axis=1). \
            repeat_interleave(scale, axis=2)

    def forward_encoder(self, imgs, mask_ratio):
        """Forward encoder."""
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask

    def forward_decoder(self, x, mask):
        """Forward decoder."""
        x = self.proj(x)
        # append mask token
        _, _, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred

    def forward_loss(self, imgs, pred, mask):
        """Foward loss.
        Args:
            imgs: [N, 3, H, W]
            pred: [N, L, p*p*3]
            mask: [N, L], 0 is keep, 1 is remove
        """
        # print(pred.shape) # torch.Size([64, 3072, 7, 7])
        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, labels=None):
        """Forward."""
        x, mask = self.forward_encoder(imgs, self.mask_ratio)
        if self.export:
            return x
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_convnextv2_atto(**kwargs):
    """convnextv2 atto"""
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], decoder_embed_dim=512, decoder_depth=1, **kwargs)
    return model


def mae_convnextv2_femto(**kwargs):
    """convnextv2 femto"""
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], decoder_embed_dim=512, decoder_depth=1, **kwargs)
    return model


def mae_convnextv2_pico(**kwargs):
    """convnextv2 pico"""
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], decoder_embed_dim=512, decoder_depth=1, **kwargs)
    return model


def mae_convnextv2_nano(**kwargs):
    """convnextv2 nano"""
    model = FCMAE(
        depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], decoder_embed_dim=512, decoder_depth=1, **kwargs)
    return model


def mae_convnextv2_tiny(**kwargs):
    """convnextv2 tiny"""
    model = FCMAE(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], decoder_embed_dim=512, decoder_depth=1, **kwargs)
    return model


def mae_convnextv2_base(**kwargs):
    """convnextv2 base"""
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], decoder_embed_dim=512, decoder_depth=1, **kwargs)
    return model


def mae_convnextv2_large(**kwargs):
    """convnextv2 large"""
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], decoder_embed_dim=512, decoder_depth=1, **kwargs)
    return model


def mae_convnextv2_huge(**kwargs):
    """convnextv2 huge"""
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], decoder_embed_dim=512, decoder_depth=1, **kwargs)
    return model


fcmae_group = [
    mae_convnextv2_atto, mae_convnextv2_femto, mae_convnextv2_pico, mae_convnextv2_nano,
    mae_convnextv2_tiny, mae_convnextv2_base, mae_convnextv2_large, mae_convnextv2_huge
]
