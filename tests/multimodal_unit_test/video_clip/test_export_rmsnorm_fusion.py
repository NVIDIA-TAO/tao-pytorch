# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the opset-23 RMSNormalization fusion.

``_fuse_rms_normalization`` rewrites torch's decomposed RMSNorm subgraph into a
single opset-23 ``RMSNormalization`` node. The deploy side depends on that node
existing: ModelOpt AutoCast keeps RMSNorm in fp32 via ``--op_types_to_exclude
RMSNormalization``, and without the fused op the fp16 TensorRT engine has
nothing to exclude.

The rewrite is built on ``onnxscript.rewriter._ir_utils`` and ``_fusion_utils``,
which are private and carry no API-stability guarantee, so an onnxscript bump
can make it silently match nothing. These tests fail loudly when that happens
instead of leaving it to surface as an fp16 accuracy collapse at deploy time.
"""

import onnx
import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.multimodal.video_clip.scripts.export import (
    _fuse_rms_normalization,
)


class _DecomposedRMSNorm(nn.Module):
    """RMSNorm written out longhand, the way the exporter lowers it.

    Mirrors the ``Pow -> ReduceMean -> Add -> Sqrt -> Reciprocal -> Mul ->
    Mul(scale)`` chain that ``_RmsNormFlexAxes`` matches.
    """

    def __init__(self, dim, eps=1e-6):
        """Initialize with a learnable per-channel scale."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        """Normalize over the last axis and apply the scale."""
        mean_square = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(mean_square + self.eps) * self.weight


def _export_decomposed(tmp_path, n_layers=3, dim=8):
    """Export a stack of decomposed RMSNorms to an opset-23 ONNX file."""
    model = nn.Sequential(*[_DecomposedRMSNorm(dim) for _ in range(n_layers)])
    model.eval()
    onnx_path = str(tmp_path / "rmsnorm.onnx")
    torch.onnx.export(
        model,
        (torch.randn(1, 4, dim),),
        onnx_path,
        input_names=["x"],
        output_names=["y"],
        opset_version=23,
        dynamo=True,
    )
    return onnx_path


def _op_types(onnx_path):
    """Return the op-type counts of an ONNX graph."""
    graph = onnx.load(onnx_path, load_external_data=False).graph
    counts = {}
    for node in graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


@pytest.mark.multimodal_unit
def test_fuse_rms_normalization_rewrites_every_subgraph(tmp_path):
    """Each decomposed RMSNorm becomes one opset-23 RMSNormalization node."""
    pytest.importorskip("onnxscript")
    n_layers = 3
    onnx_path = _export_decomposed(tmp_path, n_layers=n_layers)

    before = _op_types(onnx_path)
    assert before.get("RMSNormalization", 0) == 0, (
        "torch exported a fused RMSNormalization on its own; the fusion "
        "helper may no longer be needed"
    )

    fused = _fuse_rms_normalization(onnx_path)

    assert fused == n_layers
    after = _op_types(onnx_path)
    assert after.get("RMSNormalization", 0) == n_layers
    # The decomposition is gone, not merely shadowed by the fused node.
    assert after.get("Pow", 0) == 0
    assert after.get("ReduceMean", 0) == 0


@pytest.mark.multimodal_unit
def test_fused_node_stashes_in_fp32(tmp_path):
    """The fused node must accumulate in fp32.

    ``stash_type`` fp32 is the whole point of the fusion: it is what marks the
    normalization as the op AutoCast should keep out of fp16.
    """
    pytest.importorskip("onnxscript")
    onnx_path = _export_decomposed(tmp_path, n_layers=1)
    assert _fuse_rms_normalization(onnx_path) == 1

    graph = onnx.load(onnx_path, load_external_data=False).graph
    node = next(n for n in graph.node if n.op_type == "RMSNormalization")
    attrs = {a.name: a for a in node.attribute}

    assert attrs["stash_type"].i == onnx.TensorProto.FLOAT
    assert attrs["axis"].i == -1
