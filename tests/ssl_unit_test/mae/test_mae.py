# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from nvidia_tao_pytorch.ssl.mae.model.mae import MaskedAutoencoderViT

def test_model_initialization():
    model = MaskedAutoencoderViT(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16)
    assert isinstance(model, MaskedAutoencoderViT)

@pytest.mark.parametrize("patch_size, embed_dim, depth, num_heads", [
    (16, 768, 12, 12),
    (14, 1024, 24, 16),
])
def test_forward_encoder(patch_size, embed_dim, depth, num_heads):
    batch_size = 1
    img_size = 224
    channels = 3
    mask_ratio = 0.75
    inputs = torch.randn(batch_size, channels, img_size, img_size)
    model = MaskedAutoencoderViT(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads)

    latent, mask, ids_restore = model.forward_encoder(inputs, mask_ratio=mask_ratio)

    assert latent.shape == (batch_size, int(((img_size // patch_size) ** 2) * (1 - mask_ratio)) + 1, embed_dim)
    assert mask.shape == (batch_size, (img_size // patch_size) ** 2)
    assert ids_restore.shape == (batch_size, (img_size // patch_size) ** 2)

@pytest.mark.parametrize("patch_size, embed_dim, depth, num_heads", [
    (16, 768, 12, 12),
    (14, 1024, 24, 16),
])
def test_forward_decoder(patch_size, embed_dim, depth, num_heads):
    batch_size = 1
    img_size = 224

    latent = torch.randn(batch_size, (img_size // patch_size) ** 2 + 1, embed_dim)
    ids_restore = torch.arange((img_size // patch_size) ** 2).unsqueeze(0)

    model = MaskedAutoencoderViT(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads)

    pred = model.forward_decoder(latent, ids_restore)
    assert pred.shape == (batch_size, (img_size // patch_size) ** 2, 3 * patch_size ** 2)

@pytest.mark.parametrize("patch_size, embed_dim, depth, num_heads", [
    (16, 768, 12, 12),
    (14, 1024, 24, 16),
])
def test_forward_loss(patch_size, embed_dim, depth, num_heads):
    batch_size = 1
    img_size = 224
    channels = 3

    inputs = torch.randn(batch_size, channels, img_size, img_size)
    model = MaskedAutoencoderViT(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads)

    latent, mask, ids_restore = model.forward_encoder(inputs, mask_ratio=0.75)
    pred = model.forward_decoder(latent, ids_restore)

    loss = model.forward_loss(inputs, pred, mask)
    assert isinstance(loss, torch.Tensor)

@pytest.mark.parametrize("patch_size, embed_dim, depth, num_heads", [
    (16, 768, 12, 12),
    (14, 1024, 24, 16),
])
def test_forward(patch_size, embed_dim, depth, num_heads):
    batch_size = 1
    img_size = 224
    channels = 3

    inputs = torch.randn(batch_size, channels, img_size, img_size)
    model = MaskedAutoencoderViT(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads)

    loss, pred, mask = model.forward(inputs, mask_ratio=0.75)
    assert isinstance(loss, torch.Tensor)
    assert pred.shape == (batch_size, (img_size // patch_size) ** 2, 3 * patch_size ** 2)
    assert mask.shape == (batch_size, (img_size // patch_size) ** 2)
