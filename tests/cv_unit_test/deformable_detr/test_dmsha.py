# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test to verify that MSDeformAttnFunction and multi_scale_deformable_attn_pytorch are identical"""

import pytest
import torch
import os
import inspect
import sys

from nvidia_tao_pytorch.cv.deformable_detr.model.ops.modules import MSDeformAttnFunction, load_ops, multi_scale_deformable_attn_pytorch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA support')
def test_forward_equal_with_pytorch_double():
    # Load operator
    ops_dir = os.path.dirname(inspect.getfile(MSDeformAttnFunction))
    lib_name = f"MultiScaleDeformableAttention.cpython-{sys.version_info.major}{sys.version_info.minor}-{os.uname().machine}-linux-gnu.so"
    load_ops(ops_dir, lib_name)

    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long)
    level_start_index = torch.cat((shapes.new_zeros(
        (1, )), shapes.prod(1).cumsum(0)[:-1]))
    S = sum((H * W).item() for H, W in shapes)

    torch.manual_seed(3)
    value = torch.rand(N, S, M, D) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(
        -1, keepdim=True).sum(
            -2, keepdim=True)
    im2col_step = 2
    output_pytorch = multi_scale_deformable_attn_pytorch(
        value, shapes, sampling_locations, attention_weights).detach().cpu()
    output_pytorch = output_pytorch.view(N, M, -1)

    output_device = MSDeformAttnFunction.apply(
        value.cuda(), shapes.cuda(), level_start_index.cuda(),
        sampling_locations.cuda(), attention_weights.cuda(),
        im2col_step).detach().cpu()
    output_device = output_device.view(N, M, -1)

    assert torch.allclose(output_device, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_device - output_pytorch).abs().max()
    max_rel_err = ((output_device - output_pytorch).abs() /
                   output_pytorch.abs()).max()
    assert max_abs_err < 1e-9
    assert max_rel_err < 1e-6
