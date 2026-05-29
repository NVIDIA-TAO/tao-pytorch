# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ONNX Symbolic Functions for NVPanoptix3D Model."""

import torch
from torch.onnx import symbolic_helper


def upsample_bicubic2d_aa(g, input_tensor, output_size, align_corners, scales_h=None, scales_w=None):
    """ONNX symbolic for aten::_upsample_bicubic2d_aa.

    Anti-aliased bicubic upsampling is not natively supported in ONNX opset 17.
    We lower it to the standard ONNX Resize op with cubic mode, which is
    functionally equivalent for upsampling (anti-aliasing only matters when
    downsampling, so this is a safe approximation for export).

    output_size may be either a compile-time constant list or a dynamic
    prim::ListConstruct node (e.g. when the target size is derived from input
    tensor dimensions at runtime). Both cases are handled below.

    Args:
        g: ONNX graph object used for constructing nodes
        input_tensor: Input tensor node (N, C, H, W)
        output_size: Target spatial size as a static int list or dynamic prim::ListConstruct node
        align_corners: Boolean flag controlling coordinate transformation mode
        scales_h: Optional height scale factor (unused, kept for signature compatibility)
        scales_w: Optional width scale factor (unused, kept for signature compatibility)

    Returns:
        ONNX Resize node producing the bicubic-upsampled tensor (N, C, H_out, W_out)
    """
    align_corners_val = symbolic_helper._parse_arg(align_corners, "b")
    coordinate_transformation_mode = "align_corners" if align_corners_val else "half_pixel"

    # Batch and channel dims are taken directly from the input shape.
    input_shape = g.op("Shape", input_tensor)
    batch_channel = g.op(
        "Slice",
        input_shape,
        g.op("Constant", value_t=torch.LongTensor([0])),
        g.op("Constant", value_t=torch.LongTensor([2])),
    )

    # output_size can be static (constant int list) or dynamic (prim::ListConstruct
    # whose elements are scalar tensors computed at trace time).
    try:
        output_size_list = symbolic_helper._parse_arg(output_size, "is")
        hw = g.op("Constant", value_t=torch.LongTensor(output_size_list))
    except Exception:
        # Dynamic path: unpack the prim::ListConstruct node and reshape each
        # scalar tensor to a 1-element 1-D tensor, then concat.
        sizes = symbolic_helper._unpack_list(output_size)
        one = g.op("Constant", value_t=torch.LongTensor([1]))
        h_1d = g.op("Reshape", g.op("Cast", sizes[0], to_i=7), one)
        w_1d = g.op("Reshape", g.op("Cast", sizes[1], to_i=7), one)
        hw = g.op("Concat", h_1d, w_1d, axis_i=0)

    target_size = g.op("Concat", batch_channel, hw, axis_i=0)

    empty_roi = g.op("Constant", value_t=torch.FloatTensor([]))
    empty_scales = g.op("Constant", value_t=torch.FloatTensor([]))

    return g.op(
        "Resize",
        input_tensor,
        empty_roi,
        empty_scales,
        target_size,
        coordinate_transformation_mode_s=coordinate_transformation_mode,
        cubic_coeff_a_f=-0.75,
        mode_s="cubic",
        nearest_mode_s="floor",
    )


def nvidia_msda(
    g, value, value_spatial_shapes, value_level_start_index,
    sampling_locations, attention_weights
):
    """ONNX symbolic function for MultiscaleDeformableAttnPlugin_TRT.

    Registers the custom TensorRT plugin node for multi-scale deformable
    attention and attaches best-effort shape metadata to the output.

    Args:
        g: ONNX graph object used for constructing nodes
        value: Value tensor node of shape (N, Len_in, n_heads, c)
        value_spatial_shapes: Spatial shapes of each feature-map level
        value_level_start_index: Start index of each level in the flattened value
        sampling_locations: Sampling offsets of shape (N, Len_q, n_heads, n_levels, n_points, 2)
        attention_weights: Attention weights of shape (N, Len_q, n_heads, n_levels, n_points)

    Returns:
        ONNX node for the TRT plugin with shape (N, Len_q, n_heads * c)
    """
    out = g.op(
        "nvidia::MultiscaleDeformableAttnPlugin_TRT",
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
    )

    # Expected output is [N, Len_q, n_heads * c]
    # value: [N, Len_in, n_heads, c]
    bs = symbolic_helper._get_tensor_dim_size(value, 0)
    len_q = symbolic_helper._get_tensor_dim_size(sampling_locations, 1)
    n_heads = symbolic_helper._get_tensor_dim_size(value, 2)
    c = symbolic_helper._get_tensor_dim_size(value, 3)

    last = (n_heads * c) if isinstance(n_heads, int) and isinstance(c, int) else None

    # Preserve dtype; attach best-effort shape to silence shape inference warnings
    try:
        out.setType(value.type().with_sizes([bs, len_q, last]))
    except Exception:
        try:
            out.setType(value.type())
        except Exception:
            pass

    return out


def meshgrid_onnx(g, tensor_list, indexing=None):
    """ONNX symbolic function for torch.meshgrid.

    Lowers torch.meshgrid to ONNX Unsqueeze/Expand ops. Supports both "ij"
    (default) and "xy" indexing modes for exactly two input tensors.

    Args:
        g: ONNX graph object used for constructing nodes
        tensor_list: prim::ListConstruct node containing exactly two 1-D tensors
        indexing: Optional indexing mode string ("ij" or "xy"). Defaults to "ij".

    Returns:
        prim::ListConstruct of two grid tensors. For "ij" mode the shape is
        (len(t0), len(t1)); for "xy" mode the shape is (len(t1), len(t0)).

    Raises:
        NotImplementedError: If more or fewer than 2 tensors are provided.
    """
    tensors = symbolic_helper._unpack_list(tensor_list)
    if len(tensors) != 2:
        raise NotImplementedError("ONNX export for meshgrid only supports 2 tensors")

    t0, t1 = tensors[0], tensors[1]
    t0 = symbolic_helper._reshape_helper(g, t0, g.op("Constant", value_t=torch.LongTensor([-1])))
    t1 = symbolic_helper._reshape_helper(g, t1, g.op("Constant", value_t=torch.LongTensor([-1])))

    shape0 = g.op("Shape", t0)   # [len(x)]
    shape1 = g.op("Shape", t1)   # [len(y)]

    axes_1 = g.op("Constant", value_t=torch.LongTensor([1]))
    axes_0 = g.op("Constant", value_t=torch.LongTensor([0]))

    indexing_mode = None
    if indexing is not None and symbolic_helper._is_value(indexing):
        indexing_mode = symbolic_helper._parse_arg(indexing, "s")
    elif indexing is not None:
        indexing_mode = indexing

    if indexing_mode == "xy":
        # outputs are [len(y), len(x)]
        grid_shape = g.op("Concat", shape1, shape0, axis_i=0)

        t0_2d = g.op("Unsqueeze", t0, axes_0)      # [1, len(x)]
        t0_grid = g.op("Expand", t0_2d, grid_shape)  # [len(y), len(x)]

        t1_2d = g.op("Unsqueeze", t1, axes_1)       # [len(y), 1]
        t1_grid = g.op("Expand", t1_2d, grid_shape)  # [len(y), len(x)]

        return g.op("prim::ListConstruct", t0_grid, t1_grid)
    else:
        # default "ij": outputs are [len(x), len(y)]
        grid_shape = g.op("Concat", shape0, shape1, axis_i=0)

        t0_2d = g.op("Unsqueeze", t0, axes_1)       # [len(x), 1]
        t0_grid = g.op("Expand", t0_2d, grid_shape)  # [len(x), len(y)]

        t1_2d = g.op("Unsqueeze", t1, axes_0)       # [1, len(y)]
        t1_grid = g.op("Expand", t1_2d, grid_shape)  # [len(x), len(y)]

        return g.op("prim::ListConstruct", t0_grid, t1_grid)


def layer_norm_onnx(g, input_tensor, normalized_shape, weight, bias, eps, cudnn_enable):
    """Custom ONNX symbolic function for LayerNorm with optional affine.

    Maps torch.nn.LayerNorm to the ONNX LayerNormalization op. When the module
    has ``elementwise_affine=False`` (weight/bias are None), constant ones/zeros
    are synthesised so the ONNX node always receives all three inputs.

    Args:
        g: ONNX graph object used for constructing nodes
        input_tensor: Input tensor node of arbitrary rank
        normalized_shape: Shape over which to normalise (last N dimensions)
        weight: Affine scale parameter, or None if elementwise_affine is False
        bias: Affine bias parameter, or None if elementwise_affine is False
        eps: Epsilon value for numerical stability (constant or graph node)
        cudnn_enable: cuDNN flag (unused in ONNX, kept for signature compatibility)

    Returns:
        ONNX LayerNormalization node with the same shape as the input
    """
    # Extract epsilon value from constant node if needed
    if symbolic_helper._is_value(eps):
        # eps is an ONNX node, extract the constant value
        eps_value = symbolic_helper._parse_arg(eps, "f")
    else:
        eps_value = float(eps)

    # Get normalized_shape value
    if symbolic_helper._is_value(normalized_shape):
        # It's a graph node, try to extract the constant value
        norm_shape = symbolic_helper._parse_arg(normalized_shape, "is")
        if isinstance(norm_shape, int):
            norm_shape = [norm_shape]
    else:
        norm_shape = normalized_shape if isinstance(normalized_shape, list) else [normalized_shape]

    # Get the axis to normalize over (last len(norm_shape) dimensions)
    axes = -1  # LayerNorm typically normalizes over the last dimension

    # If weight or bias are None (elementwise_affine=False), create constant tensors
    if symbolic_helper._is_none(weight):
        # Get the last dimension size from input for creating weight
        input_shape = g.op("Shape", input_tensor)
        last_dim = g.op(
            "Slice", input_shape,
            g.op("Constant", value_t=torch.LongTensor([-1])),
            g.op("Constant", value_t=torch.LongTensor([2147483647]))
        )
        # Create ones with shape matching last dimension
        weight = g.op(
            "ConstantOfShape", last_dim,
            value_t=torch.tensor([1.0], dtype=torch.float32)
        )

    if symbolic_helper._is_none(bias):
        # Get the last dimension size from input for creating bias
        input_shape = g.op("Shape", input_tensor)
        last_dim = g.op(
            "Slice", input_shape,
            g.op("Constant", value_t=torch.LongTensor([-1])),
            g.op("Constant", value_t=torch.LongTensor([2147483647]))
        )
        # Create zeros with shape matching last dimension
        bias = g.op(
            "ConstantOfShape", last_dim,
            value_t=torch.tensor([0.0], dtype=torch.float32)
        )

    # Create LayerNormalization node with epsilon as a float attribute
    return g.op("LayerNormalization", input_tensor, weight, bias, epsilon_f=eps_value, axis_i=axes)


def cartesian_prod_onnx(g, tensor_list):
    """ONNX symbolic for torch.cartesian_prod.

    PyTorch ONNX export does not support ``aten::cartesian_prod`` natively, so
    we lower it to Reshape/Expand/Concat.

    Important:
    - ONNX ``Expand`` does NOT support -1 in the target shape.
    - ONNX ``Flatten`` always outputs rank-2, so don't use it to create 1D tensors.

    Args:
        g: ONNX graph object used for constructing nodes
        tensor_list: prim::ListConstruct node containing one or more 1-D tensors

    Returns:
        ONNX node of shape (M, K) where M is the product of all input lengths
        and K is the number of input tensors

    Raises:
        ValueError: If tensor_list is empty.
    """
    tensors = symbolic_helper._unpack_list(tensor_list)
    if len(tensors) == 0:
        raise ValueError("cartesian_prod requires at least one tensor")

    idx0 = g.op("Constant", value_t=torch.LongTensor([0]))
    idx1 = g.op("Constant", value_t=torch.LongTensor([1]))
    one = g.op("Constant", value_t=torch.LongTensor([1]))
    minus1 = g.op("Constant", value_t=torch.LongTensor([-1]))

    # Make all inputs true 1D.
    flats = [symbolic_helper._reshape_helper(g, t, minus1) for t in tensors]

    # Start with first tensor as column vector [M, 1]
    first = flats[0]
    m_shape = g.op("Shape", first)  # [M]
    result = g.op("Reshape", first, g.op("Concat", m_shape, one, axis_i=0))

    for next_tensor in flats[1:]:
        next_shape = g.op("Shape", next_tensor)  # [N]
        n = g.op("Gather", next_shape, idx0, axis_i=0)

        result_shape = g.op("Shape", result)
        m = g.op("Gather", result_shape, idx0, axis_i=0)
        k = g.op("Gather", result_shape, idx1, axis_i=0)

        # [M, K] -> [M, 1, K] -> [M, N, K]
        result_3d = g.op("Reshape", result, g.op("Concat", m, one, k, axis_i=0))
        result_tiled = g.op("Expand", result_3d, g.op("Concat", m, n, k, axis_i=0))

        # [N] -> [1, N, 1] -> [M, N, 1]
        next_3d = g.op("Reshape", next_tensor, g.op("Concat", one, n, one, axis_i=0))
        next_tiled = g.op("Expand", next_3d, g.op("Concat", m, n, one, axis_i=0))

        combined = g.op("Concat", result_tiled, next_tiled, axis_i=2)

        mn = g.op("Mul", m, n)
        k1 = g.op("Add", k, one)
        result = g.op("Reshape", combined, g.op("Concat", mn, k1, axis_i=0))
    return result
