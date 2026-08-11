# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnose a k-NN result that came back at chance level.

The baseline ImageNet k-NN returned 0.20% top-1 where stock DINOv3 ViT-B should score ~75%.
That is a harness bug, not a measurement, and the failure could be in any of four places.
This checks them in order, cheaply, printing evidence rather than a verdict:

  1. embeddings degenerate?  -- collapsed/NaN features give chance accuracy
  2. labels aligned?         -- train and val must agree on the class ordering
  3. neighbours sane?        -- an image's nearest neighbours should share its class
  4. teacher populated?      -- the adapter reads teacher.backbone, which is mirrored from
                                the student inside restore_pretrained_weights

Run on one GPU with a small sample; seconds, not minutes.
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.evaluation.datasets.classification import (
    IMAGENET_MEAN, IMAGENET_STD, build_classification_loader,
)
from nvidia_tao_pytorch.core.evaluation.model_adapter import DinoV2Adapter
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel

PROJ = "/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip"
IMAGENET = f"{PROJ}/datasets/imagenet2012"


def load_spec(spec_path):
    """Compose a spec the way Hydra would."""
    spec = OmegaConf.load(spec_path)
    parents = [d for d in spec.pop("defaults", []) if isinstance(d, str) and d != "_self_"]
    merged = OmegaConf.structured(ExperimentConfig())
    for parent in parents:
        merged = OmegaConf.merge(merged, load_spec(Path(spec_path).parent / f"{parent}.yaml"))
    return OmegaConf.merge(merged, spec)


def main():
    """Print evidence for each candidate failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--crop", type=int, default=224)
    parser.add_argument("--n", type=int, default=512, help="images per split")
    args = parser.parse_args()

    device = torch.device("cuda")
    config = load_spec(args.spec)
    config.train.use_custom_attention = False

    model = DinoV3PlModel(config).to(device)
    model.pretrained_weights = config.train.pretrained_model_path
    model.restore_pretrained_weights()
    if config.model.lora.enable:
        model.inject_lora_adapters()
    model.eval()

    print("\n=== 4. teacher vs student weights (teacher is what the adapter reads) ===")
    sd_s = model.student.backbone.state_dict()
    sd_t = model.teacher.backbone.state_dict()
    same, diff = 0, []
    for k, v in sd_s.items():
        if k in sd_t and v.shape == sd_t[k].shape and torch.is_floating_point(v):
            if torch.equal(v.float(), sd_t[k].float()):
                same += 1
            else:
                diff.append(k)
    print(f"  identical tensors: {same}, differing: {len(diff)}")
    if diff:
        print(f"  first differing: {diff[:5]}")
    w = sd_t.get("blocks.0.attn.qkv.weight")
    if w is not None:
        print(f"  teacher blocks.0.attn.qkv.weight: mean={w.float().mean():.6f} "
              f"std={w.float().std():.6f}  (a random init would be ~N(0, small))")

    adapter = DinoV2Adapter(model, patch_size=int(config.model.backbone.patch_size),
                            feature_dim=768).to(device).eval()

    def embed(root, split):
        """Extract embeddings + labels for a small sample of a split."""
        info = build_classification_loader(
            "image_folder", root, batch_size=64, num_workers=8, crop=args.crop,
            mean=IMAGENET_MEAN, std=IMAGENET_STD, distributed=False,
            max_samples=args.n, num_classes=1000,
        )
        embs, lbls = [], []
        with torch.no_grad():
            for batch in info.loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"]
                with torch.autocast("cuda", dtype=torch.float16):
                    summary, _ = adapter(images)
                embs.append(summary.float().cpu())
                lbls.append(labels.cpu() if torch.is_tensor(labels) else torch.tensor(labels))
                if sum(e.shape[0] for e in embs) >= args.n:
                    break
        return torch.cat(embs)[:args.n], torch.cat(lbls)[:args.n], info.num_classes

    print("\n=== 0. input + token-index sanity ===")
    # Two candidates left after teacher/labels/resolution were cleared: the input pipeline is
    # feeding garbage, or the summary is being read from the wrong token. Distinguish them by
    # (a) inspecting the actual pixels and (b) comparing CLS against mean-pooled patch tokens.
    info0 = build_classification_loader(
        "image_folder", f"{IMAGENET}/val", batch_size=16, num_workers=4, crop=args.crop,
        mean=IMAGENET_MEAN, std=IMAGENET_STD, distributed=False, max_samples=16,
        num_classes=1000,
    )
    batch0 = next(iter(info0.loader))
    imgs0 = batch0["image"].to(device)
    print(f"  input tensor: shape={tuple(imgs0.shape)} dtype={imgs0.dtype}")
    print(f"    per-image mean spread: {imgs0.mean(dim=(1,2,3)).min():.3f} .. "
          f"{imgs0.mean(dim=(1,2,3)).max():.3f}")
    print(f"    per-image std  spread: {imgs0.std(dim=(1,2,3)).min():.3f} .. "
          f"{imgs0.std(dim=(1,2,3)).max():.3f}")
    print("    (images should differ from each other; a tight spread means identical inputs)")

    with torch.no_grad():
        out = model.teacher.backbone(imgs0)
    cls = out["x_norm_clstoken"]
    pat = out["x_norm_patchtokens"]
    print(f"  x_norm_clstoken: {tuple(cls.shape)}   x_norm_patchtokens: {tuple(pat.shape)}")
    for name, feat in (("CLS", cls if cls.dim() == 2 else cls[:, 0]),
                       ("mean-patch", pat.mean(dim=1))):
        n = torch.nn.functional.normalize(feat.float(), dim=1)
        off = (n @ n.T).fill_diagonal_(0)
        print(f"    {name:11s} pairwise cosine mean={off.mean():.4f} max={off.max():.4f}")
    print("    (if mean-patch separates but CLS does not, the summary token index is wrong)")

    print("\n=== 1. embedding sanity (train split) ===")
    tr_e, tr_l, ncls = embed(f"{IMAGENET}/train", "train")
    print(f"  shape={tuple(tr_e.shape)} classes={ncls}")
    print(f"  finite={torch.isfinite(tr_e).all().item()}  "
          f"mean={tr_e.mean():.4f} std={tr_e.std():.4f}")
    norms = tr_e.norm(dim=1)
    print(f"  L2 norms: min={norms.min():.3f} max={norms.max():.3f} mean={norms.mean():.3f}")
    normed = torch.nn.functional.normalize(tr_e, dim=1)
    off = (normed @ normed.T).fill_diagonal_(0)
    print(f"  pairwise cosine: mean={off.mean():.4f} max={off.max():.4f}  "
          f"(near 1.0 everywhere = collapsed embeddings)")

    print("\n=== 2. label alignment ===")
    va_e, va_l, ncls_v = embed(f"{IMAGENET}/val", "val")
    print(f"  train classes={ncls}  val classes={ncls_v}  (must match)")
    print(f"  train label range=[{tr_l.min()}, {tr_l.max()}]  "
          f"val label range=[{va_l.min()}, {va_l.max()}]")
    print(f"  distinct train labels={len(set(tr_l.tolist()))}  "
          f"distinct val labels={len(set(va_l.tolist()))}")

    print("\n=== 3. nearest-neighbour sanity (val query -> val database, same split) ===")
    # Same-split retrieval isolates the model from any train/val label mismatch: an image's
    # nearest neighbour within its own split should usually share its class.
    vn = torch.nn.functional.normalize(va_e, dim=1)
    sim = (vn @ vn.T).fill_diagonal_(-2)
    nn_idx = sim.argmax(dim=1)
    agree = (va_l[nn_idx] == va_l).float().mean().item()
    print(f"  1-NN same-class rate within val: {agree*100:.1f}%")
    print("  (a working DINOv3 should be well above chance here; ~0.1% means the features "
          "carry no class information at all)")

    print("\n=== 3b. cross-split 1-NN (val query -> train database) ===")
    tn = torch.nn.functional.normalize(tr_e, dim=1)
    sim_x = vn @ tn.T
    nn_x = sim_x.argmax(dim=1)
    agree_x = (tr_l[nn_x] == va_l).float().mean().item()
    print(f"  1-NN same-class rate across splits: {agree_x*100:.1f}%")
    print("  (if 3 is high but 3b is at chance, the two splits disagree on class ordering)")


if __name__ == "__main__":
    main()
