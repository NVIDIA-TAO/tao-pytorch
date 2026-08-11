# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ImageNet k-NN retention evaluation for the DINOv3 LoRA Stage-4 arms.

Answers gate G4.2: how much of the *general* representation survives domain adaptation.
The drift probe measures how far a backbone moved from the pretrained anchor; this measures
whether that movement cost anything a downstream consumer would notice.

    # baseline first -- arm numbers are meaningless without KNN_IN_0
    python dinov3_lora_retention_eval.py --spec <spec> --baseline --output <dir>/baseline.json
    # then each arm
    python dinov3_lora_retention_eval.py --spec <spec> --checkpoint <model_*.pth> \
        --arm B --output <dir>/knn_B.json

There is no ``dinov3 evaluate`` subtask (only train/convert/export/inference), so this drives
``core/evaluation`` directly -- the option STATUS.md's open decision 2 anticipated. A real
subtask would be the better home if this outlives Stage 4.

Protocol notes that decide whether the numbers mean anything:

* **k=20** and ImageNet normalization, matching the established TAO k-NN protocol. DINOv2/v3
  expect ImageNet-normalized input, *not* [0,1]; feeding [0,1] would measure a preprocessing
  mismatch and call it drift.
* The **teacher** backbone is evaluated, because that is what ``convert``/``export`` ship.
* ``--max-train-samples`` bounds the database for runtime. It changes the absolute accuracy,
  so baseline and every arm must use the same value -- the script records it in the output
  and the comparison is only valid across runs that agree.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.evaluation.base import EvalContext
from nvidia_tao_pytorch.core.evaluation.knn import KNNEvaluator
from nvidia_tao_pytorch.core.evaluation.model_adapter import DinoV2Adapter
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel
from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import extract_backbone_state_dict

PROJ = "/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip"
IMAGENET = f"{PROJ}/datasets/imagenet2012"


def parse_args():
    """Build the CLI."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Arm checkpoint. Omit with --baseline to evaluate stock DINOv3.")
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate the pretrained backbone unchanged (KNN_IN_0).")
    parser.add_argument("--arm", default="baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-root", default=f"{IMAGENET}/train")
    parser.add_argument("--val-root", default=f"{IMAGENET}/val")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--crop", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=100000,
                        help="Cap the k-NN database. Must match across baseline and arms.")
    parser.add_argument("--cache-dir", default=None,
                        help="Reuse extracted embeddings across runs where the backbone is "
                             "identical. Off by default: every arm has a different backbone, "
                             "so a shared cache would silently serve one arm's features "
                             "for another.")
    parser.add_argument("--source", default="teacher", choices=["teacher", "student"])
    return parser.parse_args()


def load_spec(spec_path):
    """Compose a spec the way Hydra would (its `defaults:` list is a directive, not data)."""
    spec = OmegaConf.load(spec_path)
    parents = [d for d in spec.pop("defaults", []) if isinstance(d, str) and d != "_self_"]
    merged = OmegaConf.structured(ExperimentConfig())
    for parent in parents:
        merged = OmegaConf.merge(merged, load_spec(Path(spec_path).parent / f"{parent}.yaml"))
    return OmegaConf.merge(merged, spec)


def build_model(args, config, device):
    """Build the PL model, restore pretrained weights, optionally load an arm checkpoint."""
    # fp32 for the embedding extraction; the xformers custom-attention path hard-casts q/k/v
    # to .half() and is incompatible with it. (On H100 it is auto-disabled anyway per bug
    # 6459926, but this keeps the script correct on other hardware.)
    config.train.use_custom_attention = False

    model = DinoV3PlModel(config).to(device)
    model.pretrained_weights = config.train.pretrained_model_path
    model.restore_pretrained_weights()

    lora_enabled = bool(config.model.lora.enable)
    if lora_enabled:
        model.inject_lora_adapters()

    loaded = {"checkpoint": None, "missing": 0, "unexpected": 0, "lora_keys": 0}
    if not args.baseline:
        if not args.checkpoint:
            raise ValueError("Pass --checkpoint, or --baseline to evaluate stock DINOv3.")
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = extract_backbone_state_dict(raw, source=args.source)
        backbone = getattr(model, args.source).backbone
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        loaded = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "missing": len(missing), "unexpected": len(unexpected),
            "lora_keys": len([k for k in state if "lora_" in k]),
        }

    model.eval()
    return model, loaded, lora_enabled


def main():
    """Run ImageNet k-NN on one backbone and write a JSON result."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_spec(args.spec)
    model, loaded, lora_enabled = build_model(args, config, device)

    # DinoV2Adapter reads pl_model.teacher.backbone and returns (CLS, patch tokens) -- the
    # exact contract DINOv3 satisfies, since DinoV3PlModel derives from DinoV2PlModel and the
    # backbone emits x_norm_clstoken / x_norm_patchtokens.
    adapter = DinoV2Adapter(model, patch_size=int(config.model.backbone.patch_size),
                            feature_dim=768).to(device).eval()

    # imagenet_normalize=True is load-bearing: DINOv2/v3 expect ImageNet-normalized input,
    # not [0,1]. Feeding [0,1] would measure a preprocessing mismatch and report it as
    # retention loss.
    eval_cfg = SimpleNamespace(knn=SimpleNamespace(
        train_root=args.train_root,
        val_root=args.val_root,
        dataset_type="image_folder",
        k=args.k,
        crop=args.crop,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=1000,
        imagenet_normalize=True,
        max_train_samples=args.max_train_samples,
        amp=True,
        use_faiss=True,
    ))

    ctx = EvalContext(
        model=adapter,
        network="dinov3",
        device=device,
        distributed=torch.distributed.is_available() and torch.distributed.is_initialized(),
        build_loader=None,          # falls back to the core classification loader
        cfg=eval_cfg,
        results_dir=str(Path(args.output).parent),
        cache_dir=args.cache_dir,
    )

    metrics = KNNEvaluator().run(ctx) or {}

    result = {
        "arm": args.arm,
        "baseline": bool(args.baseline),
        "source": args.source,
        "lora_enabled": lora_enabled,
        "checkpoint_load": loaded,
        "protocol": {
            "metric": "ImageNet-1k k-NN top-1",
            "k": args.k,
            "crop": args.crop,
            "normalization": "imagenet",
            "max_train_samples": args.max_train_samples,
            "train_root": args.train_root,
            "val_root": args.val_root,
            "note": ("max_train_samples changes absolute accuracy, so this number is only "
                     "comparable against runs with the same value."),
        },
        "metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in metrics.items()},
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result["metrics"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
