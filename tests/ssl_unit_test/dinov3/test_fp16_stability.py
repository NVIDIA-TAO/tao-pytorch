# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for fp16 (16-mixed) numerical stability on the Blackwell fallback path.

On Blackwell-series GPUs FA3 is unsupported, so ``use_custom_attention`` is forced off and
attention runs through the naive ``RoPEMemoryEfficientAttention`` fallback. Because DINOv3
uses ``qk_norm=False`` the raw QK scores are unbounded; with the shipped default
``train.precision: 16-mixed`` a large-magnitude (pretrained-scale) residual stream drives the
fp16 score matmul past fp16's 65504 ceiling to ``inf``, and ``softmax(inf) = NaN`` poisons the
loss. bf16 (val_final ~3.4e38 range) and fp32 do not overflow. See bug 6460915.

The second test guards the reporting side: a non-finite training loss must not be reported as
a successful run (train_loss is the sole in-loop AutoML KPI).
"""
from types import SimpleNamespace

import pytest
import torch

from nvidia_tao_pytorch.ssl.dinov3.model.layers.attention import RoPEMemoryEfficientAttention
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")

# Residual-stream scale large enough that the unnormalized fp16 QK scores exceed fp16's 65504
# ceiling (stands in for the pretrained-weight activation magnitudes of the real EuroSAT smoke;
# random-init activations are too small to surface the overflow).
_OVERFLOW_SCALE = 256.0


@requires_cuda
@pytest.mark.ssl_unit
def test_dinov3_attention_fallback_fp16_finite_under_overflow():
    """The Blackwell fallback attention must stay finite under 16-mixed autocast.

    Reproduces bug 6460915: with ``qk_norm=False`` and large-magnitude activations the fp16
    score matmul saturates to ``inf`` and ``softmax(inf)=NaN``. The fallback must compute the
    scores/softmax in fp32 (as xformers' memory_efficient_attention does internally), so the
    output stays finite. bf16 is asserted as a control -- it never overflowed.
    """
    torch.manual_seed(0)
    attn = RoPEMemoryEfficientAttention(
        dim=768, num_heads=12, qkv_bias=False, qk_norm=False, attn_drop=0.0, proj_drop=0.0,
    ).cuda().eval()

    x = torch.randn(2, 197, 768).cuda() * _OVERFLOW_SCALE

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_bf16 = attn(x, use_custom_attention=False)
    assert torch.isfinite(out_bf16.float()).all(), "bf16 control unexpectedly non-finite"

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out_fp16 = attn(x, use_custom_attention=False)
    assert torch.isfinite(out_fp16.float()).all(), (
        "fp16 (16-mixed) fallback attention produced non-finite output -- the unnormalized QK "
        "score matmul overflowed fp16 (65504) and softmax(inf)=NaN. Compute scores/softmax in fp32."
    )


@pytest.mark.ssl_unit
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_dinov3_nonfinite_train_loss_raises(bad_value):
    """A non-finite epoch train loss must raise (so @monitor_status records FAILURE).

    Regression for the silent-PASS half of bug 6460915: the inherited hook writes train_loss to
    status.json as the sole in-loop AutoML KPI but never validates it, so a NaN loss was reported
    as PASS -- hiding the failure from HPO and loss-based milestone selection.
    """
    fake = SimpleNamespace(
        trainer=SimpleNamespace(logged_metrics={"train_loss_epoch": torch.tensor(bad_value)})
    )
    with pytest.raises(ValueError, match="non-finite"):
        DinoV3PlModel._assert_train_loss_finite(fake)


@pytest.mark.ssl_unit
def test_dinov3_finite_train_loss_passes():
    """A finite epoch train loss must not raise."""
    fake = SimpleNamespace(
        trainer=SimpleNamespace(logged_metrics={"train_loss_epoch": torch.tensor(10.5)})
    )
    DinoV3PlModel._assert_train_loss_finite(fake)  # should not raise
