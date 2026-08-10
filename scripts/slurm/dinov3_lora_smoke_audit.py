# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit a DINOv3 LoRA smoke checkpoint against the G2 gates, on the cluster.

Reproduces the devbox Stage-2 gates against a checkpoint produced on H100 + fp16 + xformers,
which is the combination the devbox never exercised (a4u8g-0146 is A100 and the fp32 numerical
checks there ran with custom attention disabled).

    python dinov3_lora_smoke_audit.py --checkpoint <model_*.pth> \
        --pretrained <hf snapshot dir> [--source teacher]

Gates checked:

  G2.3  frozen-base integrity -- every non-LoRA backbone tensor still bit-identical to the
        pretrained snapshot it was loaded from. This is the gate that catches an optimizer
        picking up frozen parameters, or weight decay leaking outside the parameter groups.
  G2.7  trainable-set shape -- lora_A/lora_B pairs present, count and rank as configured.
  G2.6  artifact integrity -- lora_* keys actually survive into the checkpoint (a checkpoint
        without them converts to the frozen base and silently discards the run).

Exits non-zero if any gate fails, so the smoke script can gate on it.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:  # pragma: no cover - safetensors ships in the training image
    load_safetensors = None


def parse_args():
    """Build the CLI."""
    parser = argparse.ArgumentParser(description="Audit a LoRA smoke checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained", required=True,
                        help="Pretrained snapshot dir (model.safetensors) used to start the run.")
    parser.add_argument("--source", default="teacher", choices=["teacher", "student"])
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--expect-rank", type=int, default=8)
    return parser.parse_args()


def tensor_hash(tensor):
    """Stable content hash of a tensor."""
    return hashlib.sha256(
        tensor.detach().to(torch.float32).cpu().numpy().tobytes()
    ).hexdigest()[:16]


def load_checkpoint_backbone(path, source):
    """Extract a backbone-level state dict from any of the checkpoint shapes we emit."""
    from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import extract_backbone_state_dict
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return extract_backbone_state_dict(raw, source=source)


def load_pretrained(path):
    """Load the pretrained snapshot as a flat tensor dict (timm-format safetensors)."""
    p = Path(path)
    candidates = [p] if p.is_file() else sorted(p.glob("*.safetensors")) + sorted(p.glob("*.bin"))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors/.bin under {path}")
    target = candidates[0]
    if target.suffix == ".safetensors":
        if load_safetensors is None:
            raise RuntimeError("safetensors not importable")
        return load_safetensors(str(target)), str(target)
    return torch.load(str(target), map_location="cpu", weights_only=False), str(target)


def main():
    """Run the gates and report."""
    args = parse_args()
    report = {"checkpoint": args.checkpoint, "source": args.source, "gates": {}}
    failures = []

    state = load_checkpoint_backbone(args.checkpoint, args.source)
    lora_keys = sorted(k for k in state if "lora_" in k)
    base_keys = sorted(k for k in state if "lora_" not in k)

    # --- G2.6 / G2.7: adapters present and well-formed -----------------------------------------
    a_keys = [k for k in lora_keys if k.endswith("lora_A")]
    b_keys = [k for k in lora_keys if k.endswith("lora_B")]
    ranks = sorted({tuple(state[k].shape) for k in a_keys})
    report["gates"]["G2.6_lora_keys_present"] = {
        "lora_key_count": len(lora_keys),
        "lora_A": len(a_keys), "lora_B": len(b_keys),
        "base_key_count": len(base_keys),
        "lora_A_shapes": [list(s) for s in ranks],
        "pass": bool(lora_keys) and len(a_keys) == len(b_keys) and len(a_keys) > 0,
    }
    if not report["gates"]["G2.6_lora_keys_present"]["pass"]:
        failures.append("G2.6: lora_A/lora_B keys missing or unpaired in the checkpoint")

    observed_rank = ranks[0][0] if ranks else None
    report["gates"]["G2.7_rank"] = {
        "observed": observed_rank, "expected": args.expect_rank,
        "pass": observed_rank == args.expect_rank,
    }
    if observed_rank != args.expect_rank:
        failures.append(f"G2.7: LoRA rank {observed_rank} != expected {args.expect_rank}")

    # B must be nonzero after training, or nothing was learned (it is initialized to zero).
    b_norms = {k: float(state[k].float().norm()) for k in b_keys}
    nonzero_b = sum(1 for v in b_norms.values() if v > 0)
    report["gates"]["G2.7_adapters_moved"] = {
        "lora_B_nonzero": nonzero_b, "lora_B_total": len(b_keys),
        "max_B_norm": max(b_norms.values()) if b_norms else 0.0,
        "pass": nonzero_b > 0,
    }
    if nonzero_b == 0:
        failures.append("G2.7: every lora_B is still exactly zero -- no gradient reached the adapters")

    # --- finiteness ----------------------------------------------------------------------------
    non_finite = [k for k, v in state.items()
                  if torch.is_floating_point(v) and not torch.isfinite(v).all()]
    report["gates"]["finite_weights"] = {"non_finite_tensors": non_finite[:10],
                                         "count": len(non_finite), "pass": not non_finite}
    if non_finite:
        failures.append(f"{len(non_finite)} tensors contain NaN/Inf (fp16 overflow?)")

    # --- G2.3: frozen base unchanged -----------------------------------------------------------
    pre, pre_path = load_pretrained(args.pretrained)
    report["pretrained_file"] = pre_path

    # The pretrained file is timm-format; the checkpoint is TAO-format. timm_to_tao translates
    # ONE key at a time, so map the dict through it. HF-format snapshots take the dedicated
    # dict-level converter instead (a fused-qkv tensor transform, not a rename).
    from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import (
        convert_hf_to_tao, is_hf_dinov3_state_dict, timm_to_tao,
    )
    if is_hf_dinov3_state_dict(pre):
        pre_tao = convert_hf_to_tao(pre)
        report["pretrained_format"] = "huggingface"
    else:
        pre_tao = {timm_to_tao(k): v for k, v in pre.items()}
        report["pretrained_format"] = "timm"

    compared, drifted = 0, []
    for key in base_keys:
        ref = pre_tao.get(key)
        if ref is None or ref.shape != state[key].shape:
            continue
        compared += 1
        if not torch.equal(ref.to(torch.float32), state[key].to(torch.float32)):
            delta = (ref.to(torch.float32) - state[key].to(torch.float32)).abs().max().item()
            scale = ref.to(torch.float32).abs().max().item() or 1.0
            drifted.append({"key": key, "max_abs_delta": delta, "relative": delta / scale})

    drifted.sort(key=lambda d: -d["relative"])
    report["gates"]["G2.3_frozen_base"] = {
        "tensors_compared": compared,
        "tensors_drifted": len(drifted),
        "worst": drifted[:5],
        # Exact equality is the gate. The devbox found a real 5.67e-05 EMA drift here, which is
        # precisely the class of bug this catches, so the threshold is zero rather than "small".
        "pass": len(drifted) == 0,
    }
    if compared == 0:
        failures.append("G2.3: no base tensors could be compared -- key remap mismatch, gate is vacuous")
    elif drifted:
        failures.append(f"G2.3: {len(drifted)} frozen base tensors changed "
                        f"(worst relative {drifted[0]['relative']:.2e} on {drifted[0]['key']})")

    # --- block coverage ------------------------------------------------------------------------
    blocks = sorted({int(m.group(1)) for k in a_keys
                     if (m := re.search(r"blocks\.(\d+)\.", k))})
    targets = sorted({m.group(1) for k in a_keys
                      if (m := re.search(r"\.(qkv|proj|fc1|fc2)\.lora_A$", k))})
    report["coverage"] = {"blocks_adapted": blocks, "num_blocks": len(blocks),
                          "target_modules": targets}

    report["pass"] = not failures
    report["failures"] = failures

    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

    if failures:
        print("\nAUDIT FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        sys.exit(1)
    print("\nAll audited gates passed.")


if __name__ == "__main__":
    main()
