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

import torch
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
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
    parser.add_argument("--train-per-class", type=int, default=100,
                        help="Database images per class. Must match across baseline and arms.")
    parser.add_argument("--val-per-class", type=int, default=10,
                        help="Query images per class.")
    parser.add_argument("--cache-dir", default=None,
                        help="Reuse extracted embeddings across runs where the backbone is "
                             "identical. Off by default: every arm has a different backbone, "
                             "so a shared cache would silently serve one arm's features "
                             "for another.")
    parser.add_argument("--source", default="teacher", choices=["teacher", "student"])
    parser.add_argument("--no-faiss", action="store_true",
                        help="Force the brute-force top-K path. The library's own docstring "
                             "warns faiss-gpu can be unstable in this container, and the two "
                             "paths are numerically identical, so this is the control.")
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


@torch.no_grad()
def embed_paths(adapter, paths, crop, device, batch_size):
    """L2-normalized CLS embeddings for a list of image paths (ImageNet normalization)."""
    import numpy as np
    from PIL import Image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    for start in range(0, len(paths), batch_size):
        tensors = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGB").resize((crop, crop), Image.BICUBIC),
                                 dtype="uint8").copy()
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)
        batch = ((torch.stack(tensors) - mean) / std).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            summary, _ = adapter(batch)
        out.append(torch.nn.functional.normalize(summary.float(), p=2, dim=1).cpu())
    return torch.cat(out)


def main():
    """Run ImageNet k-NN on one backbone and write a JSON result."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_spec(args.spec)
    model, loaded, lora_enabled = build_model(args, config, device)

    from nvidia_tao_pytorch.core.evaluation.model_adapter import DinoV2Adapter as _Adapter
    adapter = _Adapter(model, patch_size=int(config.model.backbone.patch_size),
                       feature_dim=768).to(device).eval()

    # Stratified enumeration, not the library's random subset.
    #
    # Two reasons. First, correctness: driving core/evaluation's KNNEvaluator over
    # build_classification_loader returned chance (0.06-0.20%) on a backbone that scores 82%
    # on a controlled subset, and the fault was isolated to that loader path -- every other
    # component (features, transform, labels, class ordering, k-NN math, dtype) was proven
    # correct in isolation. See docs/results/slurm_session_log.md. Second, protocol: a random
    # 100k draw over 1000 classes gives uneven per-class coverage, whereas a fixed number of
    # images per class makes the database balanced by construction and the number reproducible.
    classes = sorted(p.name for p in Path(args.train_root).iterdir() if p.is_dir())
    print(f"classes: {len(classes)}  chance: {100.0/len(classes):.3f}%")

    def enumerate_split(root, per_class):
        """Deterministic, class-balanced (path, label) list sharing one index mapping."""
        paths, labels = [], []
        for idx, name in enumerate(classes):
            files = sorted(Path(root, name).glob("*"))[:per_class]
            paths.extend(files)
            labels.extend([idx] * len(files))
        return paths, torch.tensor(labels)

    db_paths, db_lbl = enumerate_split(args.train_root, args.train_per_class)
    q_paths, q_lbl = enumerate_split(args.val_root, args.val_per_class)
    print(f"database: {len(db_paths)} images ({args.train_per_class}/class)  "
          f"queries: {len(q_paths)} ({args.val_per_class}/class)")

    db_emb = embed_paths(adapter, db_paths, args.crop, device, args.batch_size)
    q_emb = embed_paths(adapter, q_paths, args.crop, device, args.batch_size)

    # The library's k-NN math is proven correct (it reproduces 82% on known-good embeddings),
    # so reuse it -- keeping the documented k=20 / T=0.07 vote rather than reimplementing it.
    from nvidia_tao_pytorch.core.evaluation.knn import knn_top1_offline
    top1 = knn_top1_offline(db_emb, db_lbl, q_emb, q_lbl, num_classes=len(classes),
                            K=args.k, use_faiss=False)
    metrics = {"knn_top1": float(top1), "chance": round(100.0 / len(classes), 4)}
    print(f"\n  k-NN top-1: {top1:.2f}%   (chance {100.0/len(classes):.3f}%)")

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
            "train_per_class": args.train_per_class,
            "val_per_class": args.val_per_class,
            "sampling": "class-stratified (deterministic, first-N by sorted name)",
            "train_root": args.train_root,
            "val_root": args.val_root,
            "note": ("Per-class counts change absolute accuracy, so this number is only "
                     "comparable against runs with the same values."),
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
