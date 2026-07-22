# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from nvidia_tao_pytorch.cv.backbone_v2.convnext_v2 import (
    ConvNeXtV2, Block,
    convnextv2_atto, convnextv2_femto, convnextv2_pico, convnextv2_nano,
    convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge
)

# Test ConvNeXtV2 model
def test_convnextv2_model():
    # Test with default parameters
    model = ConvNeXtV2()
    assert isinstance(model, nn.Module)

    # Test with custom parameters
    depths = [3, 3, 9, 3]
    dims = [96, 192, 384, 768]
    drop_path_rate = 0.1
    head_init_scale = 0.5

    model = ConvNeXtV2(depths=depths, dims=dims, drop_path_rate=drop_path_rate, head_init_scale=head_init_scale)
    assert isinstance(model, nn.Module)

# Test Block module
def test_block_module():
    # Test with default parameters
    block = Block(dim=96)
    assert isinstance(block, nn.Module)

    # Test with custom parameters
    dim = 192
    drop_path = 0.1

    block = Block(dim=dim, drop_path=drop_path)
    assert isinstance(block, nn.Module)

# Test ConvNeXtV2 variants
def test_convnextv2_variants():
    variants = [
        convnextv2_atto,
        convnextv2_femto,
        convnextv2_pico,
        convnextv2_nano,
        convnextv2_tiny,
        convnextv2_base,
        convnextv2_large,
        convnextv2_huge
    ]

    for variant in variants:
        model = variant()
        assert isinstance(model, ConvNeXtV2)

# Test ConvNeXtV2 forward pass
def test_convnextv2_forward_pass():
    # Create a dummy input tensor
    x = torch.randn(1, 3, 224, 224)

    # Create a model instance
    model = ConvNeXtV2()

    # Perform forward pass
    output = model(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1000)  # Assuming 1000 classes

# Test ConvNeXtV2 feature extraction
def test_convnextv2_feature_extraction():
    # Create a dummy input tensor
    x = torch.randn(1, 3, 224, 224)

    # Create a model instance
    model = ConvNeXtV2()

    # Perform forward pass to extract features
    features = model.forward_pre_logits(x)

    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, 768)  # Assuming 768 feature dimensions

# Test ConvNeXtV2 with different input sizes
def test_convnextv2_input_sizes():
    # Create a model instance
    model = ConvNeXtV2()

    # Perform forward pass with different input sizes
    for size in [224, 256, 512]:
        x = torch.randn(1, 3, size, size)
        output = model(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 1000)  # Assuming 1000 classes

# Test ConvNeXtV2 with different batch sizes
def test_convnextv2_batch_sizes():
    # Create a model instance
    model = ConvNeXtV2()

    # Perform forward pass with different batch sizes
    for batch_size in [1, 4, 8]:
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, 1000)  # Assuming 1000 classes

# Test ConvNeXtV2 with different device types
def test_convnextv2_device_types():
    if torch.cuda.is_available():
        # Create a model instance on GPU
        model_gpu = ConvNeXtV2().to('cuda')
        x_gpu = torch.randn(1, 3, 224, 224).cuda()

        # Perform forward pass on GPU
        output_gpu = model_gpu(x_gpu)

        assert isinstance(output_gpu, torch.Tensor)
        assert output_gpu.shape == (1, 1000)  # Assuming 1000 classes

    # Create a model instance on CPU
    model_cpu = ConvNeXtV2()
    x_cpu = torch.randn(1, 3, 224, 224)

    # Perform forward pass on CPU
    output_cpu = model_cpu(x_cpu)

    assert isinstance(output_cpu, torch.Tensor)
    assert output_cpu.shape == (1, 1000)  # Assuming 1000 classes