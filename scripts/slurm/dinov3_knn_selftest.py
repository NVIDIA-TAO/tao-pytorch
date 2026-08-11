# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained k-NN sanity test: are the DINOv3 features discriminative at all?

Written after a first diagnostic gave a misleading answer. That one sampled 256 random
ImageNet images across 1000 classes and measured the 1-NN same-class rate — but with ~1
sample per class, a same-class neighbour usually does not exist in the sample, so a *perfect*
model would also score ~0%. The test could not fail informatively.

This builds a controlled subset instead: a fixed number of classes, with enough images per
class that same-class neighbours are guaranteed to exist, and computes cosine k-NN directly
rather than through the evaluation library. That separates two questions the full run
conflated:

  * features bad     -> low accuracy here too
  * harness/library  -> high accuracy here, low in the full k-NN run

Chance level is 1/num_classes, printed alongside so the number is interpretable.
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel

PROJ = "/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip"
IMAGENET = f"{PROJ}/datasets/imagenet2012"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_spec(spec_path):
    """Compose a spec the way Hydra would."""
    spec = OmegaConf.load(spec_path)
    parents = [d for d in spec.pop("defaults", []) if isinstance(d, str) and d != "_self_"]
    merged = OmegaConf.structured(ExperimentConfig())
    for parent in parents:
        merged = OmegaConf.merge(merged, load_spec(Path(spec_path).parent / f"{parent}.yaml"))
    return OmegaConf.merge(merged, spec)


def load_images(paths, crop):
    """Load and ImageNet-normalize a list of image paths."""
    import numpy as np
    tensors = []
    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((crop, crop), Image.BICUBIC)
            tensors.append(torch.from_numpy(np.asarray(img, dtype="uint8").copy())
                           .permute(2, 0, 1).float() / 255.0)
    batch = torch.stack(tensors)
    return (batch - MEAN) / STD


@torch.no_grad()
def embed(backbone, images, device, batch_size=32):
    """CLS embeddings for a stack of images."""
    out = []
    for start in range(0, images.shape[0], batch_size):
        chunk = images[start:start + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            res = backbone(chunk)
        cls = res["x_norm_clstoken"]
        out.append((cls[:, 0] if cls.dim() == 3 else cls).float().cpu())
    return torch.cat(out)


def main():
    """Run the controlled k-NN sanity test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-classes", type=int, default=50)
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--val-per-class", type=int, default=5)
    parser.add_argument("--crop", type=int, default=224)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda")
    config = load_spec(args.spec)
    config.train.use_custom_attention = False

    model = DinoV3PlModel(config).to(device)
    model.pretrained_weights = config.train.pretrained_model_path
    model.restore_pretrained_weights()
    if config.model.lora.enable:
        model.inject_lora_adapters()
    if args.checkpoint:
        from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import extract_backbone_state_dict
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.teacher.backbone.load_state_dict(
            extract_backbone_state_dict(raw, source="teacher"), strict=False)
    model.eval()
    backbone = model.teacher.backbone

    classes = sorted(p.name for p in Path(f"{IMAGENET}/train").iterdir() if p.is_dir())
    classes = classes[:args.num_classes]
    print(f"classes: {len(classes)}  chance level: {100.0/len(classes):.2f}%")

    db_emb, db_lbl, q_emb, q_lbl = [], [], [], []
    for idx, cls_name in enumerate(classes):
        tr = sorted(Path(f"{IMAGENET}/train/{cls_name}").glob("*"))[:args.train_per_class]
        va = sorted(Path(f"{IMAGENET}/val/{cls_name}").glob("*"))[:args.val_per_class]
        if not tr or not va:
            continue
        db_emb.append(embed(backbone, load_images(tr, args.crop), device))
        db_lbl += [idx] * len(tr)
        q_emb.append(embed(backbone, load_images(va, args.crop), device))
        q_lbl += [idx] * len(va)

    db = torch.nn.functional.normalize(torch.cat(db_emb), dim=1)
    qu = torch.nn.functional.normalize(torch.cat(q_emb), dim=1)
    db_lbl = torch.tensor(db_lbl)
    q_lbl = torch.tensor(q_lbl)
    print(f"database: {db.shape[0]} images, queries: {qu.shape[0]}")

    sim = qu @ db.T
    top1 = db_lbl[sim.argmax(dim=1)]
    acc1 = (top1 == q_lbl).float().mean().item() * 100

    k = min(args.k, db.shape[0])
    topk = sim.topk(k, dim=1).indices
    votes = db_lbl[topk]
    majority = torch.mode(votes, dim=1).values
    acck = (majority == q_lbl).float().mean().item() * 100

    print(f"\n  1-NN  top-1 accuracy: {acc1:.2f}%")
    print(f"  {k}-NN top-1 accuracy: {acck:.2f}%")
    print(f"  chance:               {100.0/len(classes):.2f}%")
    print("\n  A working DINOv3 ViT-B should be far above chance here. If it is, the features "
          "are fine and the fault is in the evaluation harness, not the backbone.")


if __name__ == "__main__":
    main()
