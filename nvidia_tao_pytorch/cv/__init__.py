# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root module for CV models/tasks."""

import re
import torch
from nvidia_tao_pytorch.cv.version import __version__  # noqa: F401


numbering = re.search(r"^(\d+).(\d+).(\d+)([^\+]*)(\+\S*)?$", torch.__version__)
major_version, minor_version = [int(numbering.group(n)) for n in range(1, 3)]

if major_version >= 1 and minor_version >= 14:
    from third_party.onnx.utils import _export
    # Monkey Patch ONNX Export to disable onnxscript
    torch.onnx.utils._export = _export
    # Monkey Patch SDPA location
    torch.nn.functional.scaled_dot_product_attention = torch._C._nn._scaled_dot_product_attention  # noqa: pylint: disable=I1101


if major_version >= 2:
    # From https://github.com/pytorch/pytorch/blob/2efe4d809fdc94501fc38bf429e9a8d4205b51b6/torch/utils/tensorboard/_pytorch_graph.py#L384
    def _node_get(node: torch._C.Node, key: str):  # noqa: pylint: disable=I1101
        """Gets attributes of a node which is polymorphic over return type."""
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    torch._C.Node.__getitem__ = _node_get  # noqa: pylint: disable=I1101
