# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 feature-parity smoke test vs the timm reference (GPU + weights).

Loads the public timm DINOv3 weights into our ``DinoV3VisionTransformer`` via the checkpoint
remapper and confirms the CLS + patch features are cosine-close to timm's own forward. This
validates the remapper key/shape coverage AND the RoPE convention end-to-end (rope_theta, coord
normalization, rotation layout).

Parametrized over architecture x resolution (**ViT-B** @ 256/768, **ViT-L** @ 256); each case
skips if its weights are not staged. Requires a CUDA GPU (xformers memory-efficient attention).
Point ``DINOV3_VITB_WEIGHTS`` / ``DINOV3_VITL_WEIGHTS`` at the dir/file if not in a default
location.
"""
import os

import pytest
import torch

from nvidia_tao_pytorch.ssl.dinov3.model.vit import DinoV3VisionTransformer
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel

# arch -> timm model name, dims, register count, and where to find the weights.
_ARCHS = {
    "vit_b": dict(
        timm="vit_base_patch16_dinov3", embed_dim=768, depth=12, num_heads=12, registers=4,
        weights=[os.environ.get("DINOV3_VITB_WEIGHTS", ""),
                 "/data/weights/dinov3/vitb16", os.path.expanduser("~/weights/dinov3/vitb16")],
    ),
    "vit_l": dict(
        timm="vit_large_patch16_dinov3", embed_dim=1024, depth=24, num_heads=16, registers=4,
        weights=[os.environ.get("DINOV3_VITL_WEIGHTS", ""),
                 "/data/weights/dinov3/vitl16", os.path.expanduser("~/weights/dinov3/vitl16")],
    ),
}

# (arch, img_size) cases: ViT-B at 256 and the Phase 1 high-res 768; ViT-L at 256.
# (ViT-L @ 768 is intentionally omitted -- very heavy and not a validated gate.)
_CASES = [("vit_b", 256), ("vit_b", 768), ("vit_l", 256)]

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


def _find_weights(candidates):
    """Return the first existing weights path among the candidates, else None."""
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None


def _cosine(a, b):
    """Mean per-row cosine similarity between two ``[..., C]`` tensors (fp32)."""
    a = a.reshape(-1, a.shape[-1]).float()
    b = b.reshape(-1, b.shape[-1]).float()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


@requires_cuda
@pytest.mark.ssl_unit
@pytest.mark.parametrize("arch,img_size", _CASES)
def test_dinov3_feature_parity_vs_timm(arch, img_size):
    """Our remapped ViT matches timm's CLS/patch features (cosine > 0.99), per arch and resolution.

    The 768 case (ViT-B) is the Phase 1 gate: it validates that our normalized-coord RoPE
    extrapolates to a 48x48 grid identically to timm (no pos-embed interpolation).
    """
    timm = pytest.importorskip("timm")
    spec = _ARCHS[arch]
    weights = _find_weights(spec["weights"])
    if weights is None:
        pytest.skip(f"DINOv3 {arch} weights not found")

    ckpt = os.path.join(weights, "model.safetensors") if os.path.isdir(weights) else weights
    # Keep fp32 master weights and compare under bf16 autocast (mirrors training). Pure .half()
    # eval overflows the deep ViT-L residual stream (fp16 max ~65504) and yields NaN; bf16 autocast
    # keeps the residual stream in fp32 range while still exercising the fp16 xformers attention.
    ref = timm.create_model(
        spec["timm"], pretrained=False, checkpoint_path=ckpt, img_size=img_size,
    ).cuda().eval()

    model = DinoV3VisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=spec["embed_dim"], depth=spec["depth"],
        num_heads=spec["num_heads"], init_values=1e-5, drop_path_schedule="linear",
        num_classes=0, drop_path_rate=0.0, register_tokens=spec["registers"],
        use_custom_attention=True,
    )
    # Load via the same remapper the pl_model uses (depth-agnostic; img_size doesn't change keys).
    timm_sd = DinoV3PlModel._load_pretrained_state_dict(weights)
    remapped, unmapped = DinoV3PlModel._remap_dinov3_state_dict(timm_sd, model.state_dict())
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    # Remapper must cover the whole checkpoint and leave only mask_token uninitialized.
    assert unmapped == [], f"[{arch}] checkpoint keys not mapped: {unmapped}"
    assert unexpected == [], f"[{arch}] unexpected keys after remap: {unexpected}"
    assert set(missing) <= {"mask_token"}, f"[{arch}] unexpected missing keys: {missing}"

    model = model.cuda().eval()

    torch.manual_seed(0)
    x = torch.randn(2, 3, img_size, img_size).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        feats = ref.forward_features(x)
        np = ref.num_prefix_tokens
        cls_ref, patch_ref = feats[:, 0], feats[:, np:]
        out = model(x)
        cls_ours, patch_ours = out["x_norm_clstoken"], out["x_norm_patchtokens"]

    assert torch.isfinite(cls_ours).all(), f"[{arch}@{img_size}] our CLS features are non-finite"
    assert torch.isfinite(cls_ref).all(), f"[{arch}@{img_size}] timm CLS features are non-finite"

    cls_cos = _cosine(cls_ours, cls_ref)
    patch_cos = _cosine(patch_ours, patch_ref)
    assert cls_cos > 0.99, f"[{arch}@{img_size}] CLS feature cosine too low: {cls_cos}"
    assert patch_cos > 0.99, f"[{arch}@{img_size}] patch feature cosine too low: {patch_cos}"
