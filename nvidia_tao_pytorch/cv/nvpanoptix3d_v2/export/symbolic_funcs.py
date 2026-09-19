# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ONNX symbolic functions for the NVPanoptix3Dv2 model.

Five aten ops in the panoptic graph have no usable lowering at opset 17:

* ``aten::meshgrid`` -- LoftUp's Fourier featurizer, the feature-fusion RoPE
  grid, and VGGT's UV-grid helper.
* ``aten::cartesian_prod`` -- the aggregator's 2D rotary embedding.
* ``aten::_upsample_bicubic2d_aa`` -- DINOv2's positional-embedding
  interpolation.
* ``aten::expm1`` -- VGGT's point/depth output activation.
* ``aten::triu_indices`` -- metric-scale baseline pairs across the fixed views.

The first three are re-exported from NVPanoptix3D, whose copies emit the same
subgraphs. The last two lowerings are local because the older exporter did not
need them. Registration stays local, so NVPanoptix3D's two extra symbolics
(``nvidia_msda`` and ``layer_norm_onnx``) cannot redirect an op this model never
emits.
"""

import torch
from torch.onnx import symbolic_helper

from nvidia_tao_pytorch.cv.nvpanoptix3d.export.symbolic_funcs import (
    cartesian_prod_onnx,
    meshgrid_onnx,
    upsample_bicubic2d_aa,
)

__all__ = [
    "cartesian_prod_onnx",
    "expm1_onnx",
    "meshgrid_onnx",
    "triu_indices_onnx",
    "upsample_bicubic2d_aa",
    "register_symbolic_functions",
]


def expm1_onnx(g, input_tensor):
    """Lower ``aten::expm1(x)`` to the equivalent ``Exp(x) - 1`` graph.

    ONNX opset 17 has no Expm1 operator. ``CastLike`` keeps the scalar one in
    the input dtype, which avoids invalid mixed-dtype subtraction for fp16 and
    bf16 exports.

    Args:
        g: ONNX graph object used for constructing nodes.
        input_tensor: Input tensor to exponentiate.

    Returns:
        ONNX ``Sub(Exp(input_tensor), CastLike(1, input_tensor))`` node.
    """
    one = g.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float32))
    one = g.op("CastLike", one, input_tensor)
    return g.op("Sub", g.op("Exp", input_tensor), one)


def triu_indices_onnx(
    g,
    row,
    col,
    offset=0,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    """Lower fixed-size ``aten::triu_indices`` to an ONNX constant.

    The export contract freezes the number of views, so ``row``, ``col`` and
    ``offset`` are compile-time integers. The model uses the default int64
    output for advanced indexing; the remaining PyTorch factory arguments do
    not affect the exported constant.

    Args:
        g: ONNX graph object used for constructing nodes.
        row: Number of matrix rows (compile-time integer).
        col: Number of matrix columns (compile-time integer).
        offset: Diagonal offset (compile-time integer).
        dtype: PyTorch factory dtype; the model leaves it at int64.
        layout: Unused PyTorch factory layout argument.
        device: Unused PyTorch factory device argument.
        pin_memory: Unused PyTorch factory pin-memory argument.

    Returns:
        Constant int64 tensor with shape ``[2, N]``.
    """
    del dtype, layout, device, pin_memory
    row_value = symbolic_helper._parse_arg(row, "i")
    col_value = symbolic_helper._parse_arg(col, "i")
    offset_value = symbolic_helper._parse_arg(offset, "i")
    indices = torch.triu_indices(
        int(row_value),
        int(col_value),
        offset=int(offset_value),
        dtype=torch.long,
        device="cpu",
    )
    return g.op("Constant", value_t=indices)


def register_symbolic_functions(opset_version: int) -> None:
    """Register every NVPanoptix3Dv2 symbolic for ``opset_version``.

    Args:
        opset_version: Opset the ONNX graph is being exported at.
    """
    from torch.onnx import register_custom_op_symbolic

    register_custom_op_symbolic("aten::meshgrid", meshgrid_onnx, opset_version)
    register_custom_op_symbolic("aten::cartesian_prod", cartesian_prod_onnx, opset_version)
    register_custom_op_symbolic("aten::_upsample_bicubic2d_aa", upsample_bicubic2d_aa, opset_version)
    register_custom_op_symbolic("aten::expm1", expm1_onnx, opset_version)
    register_custom_op_symbolic("aten::triu_indices", triu_indices_onnx, opset_version)
