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

"""Tests for the ONNX-export attention swap (CPU, no model/weights).

``_replace_mha_for_export`` swaps PyTorch's fused ``nn.MultiheadAttention``
(which has no ONNX symbolic) for ``ExportFriendlyMHA``, a decomposition built
from standard ops. That swap is fixed code, independent of any checkpoint, so
its numerical correctness is guarded here once rather than re-validated on every
export. The per-export parity check in export.py covers the checkpoint-specific
failure modes (tracing, external-data round-trip, dtype, onnxruntime semantics).
"""

import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.multimodal.video_clip.scripts.export import (
    ExportFriendlyMHA,
    _replace_mha_for_export,
)


class _SelfAttn(nn.Module):
    """Wrap an nn.MultiheadAttention used as self-attention (q == k == v)."""

    def __init__(self, embed_dim=32, num_heads=4, batch_first=True):
        """Build a single self-attention module."""
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=batch_first
        )

    def forward(self, x):
        """Return only the attention output (weights unused)."""
        return self.attn(x, x, x, need_weights=False)[0]


@pytest.mark.parametrize("batch_first", [True, False])
def test_replace_mha_matches_original_self_attention(batch_first):
    """ExportFriendlyMHA reproduces nn.MultiheadAttention self-attention."""
    torch.manual_seed(0)
    model = _SelfAttn(embed_dim=32, num_heads=4, batch_first=batch_first).eval()
    x = (
        torch.randn(2, 7, 32) if batch_first else torch.randn(7, 2, 32)
    )

    with torch.no_grad():
        ref = model(x)

    _replace_mha_for_export(model)
    assert isinstance(model.attn, ExportFriendlyMHA)

    with torch.no_grad():
        got = model(x)

    max_abs = (ref - got).abs().max().item()
    assert torch.allclose(ref, got, rtol=1e-4, atol=1e-5), (
        f"swap diverged from nn.MultiheadAttention: max_abs_diff={max_abs:.3e}"
    )


def test_replace_mha_swaps_all_attention_modules():
    """Every nn.MultiheadAttention in a tree is replaced; others untouched."""
    torch.manual_seed(0)

    class _TwoAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.MultiheadAttention(16, 2, batch_first=True)
            self.b = nn.MultiheadAttention(16, 4, batch_first=True)
            self.proj = nn.Linear(16, 16)

        def forward(self, x):
            x = self.a(x, x, x, need_weights=False)[0]
            x = self.b(x, x, x, need_weights=False)[0]
            return self.proj(x)

    model = _TwoAttn().eval()
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        ref = model(x)

    _replace_mha_for_export(model)
    assert isinstance(model.a, ExportFriendlyMHA)
    assert isinstance(model.b, ExportFriendlyMHA)
    assert isinstance(model.proj, nn.Linear)  # non-MHA modules untouched

    with torch.no_grad():
        got = model(x)
    max_abs = (ref - got).abs().max().item()
    assert torch.allclose(ref, got, rtol=1e-4, atol=1e-5), (
        f"swap diverged with multiple attention modules: "
        f"max_abs_diff={max_abs:.3e}"
    )
