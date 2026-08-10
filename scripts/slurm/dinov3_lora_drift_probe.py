# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-training representational-drift probe for the DINOv3 LoRA Stage-4 arms.

Writes ``probe.json`` into the arm's results directory. Run by
``scripts/slurm/dinov3_lora_vitb.sbatch`` after ``train_done``, so every completed arm
carries its own diagnostics without a separate job.

Why this exists
---------------
The devbox gate G4.5 ("monitor ``losses/cls_cos`` every 500 steps; it should be smooth")
cannot work as written. Measured on a4u8g-0146 (STATUS finding 2), at init -- where LoRA is
*provably* an identity because B=0 -- the preservation scalars read:

    drop_path  mode   masking   gram      cls_mse   cls_cos
    0.0        eval   none      3.7e-09   4.3e-08   1.2e-07
    0.4        train  30%       2.4e-02   5.8e-01   9.3e-01

The in-training scalars are dominated by two things that have nothing to do with weight
drift: the student sees *masked* crops while the anchor sees unmasked ones, and stochastic
depth resamples a different sub-network every forward. A number that reads 0.93 when the
model is bit-for-bit the anchor cannot measure how far the model has moved from the anchor.

So drift is measured here instead, under conditions where identity actually implies zero:

  * ``model.eval()``       -- no stochastic depth, no dropout
  * ``masks=None``         -- student and anchor see identical pixels
  * ``torch.no_grad()``    -- inference only
  * fixed image subset     -- the same images for every arm, so arms are comparable
  * fp32 by default        -- see ``--fp32`` below

The anchor is the *pretrained* DINOv3 backbone, which under LoRA is exactly the frozen base
weights. So the reported numbers answer: how far has this arm's backbone moved away from
stock DINOv3, in the two geometries downstream consumers depend on?

    cls_cosine     global/CLS geometry  (k-NN, linear probe, retrieval consumers)
    gram_mse       patch geometry       (segmentation, depth, correspondence consumers)

Both are reported as raw drift, *not* as a loss, and the retention gates in the plan
(G4.2/G4.3) compare them across arms: the expected ordering is D <= C <= B < A.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel
from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import extract_backbone_state_dict

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def parse_args():
    """Build the CLI."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--spec", required=True, help="Path to the arm's training spec YAML.")
    parser.add_argument("--checkpoint", required=True,
                        help="Trained checkpoint (Lightning .ckpt/.pth or stripped *_.pth).")
    parser.add_argument("--images-dir", required=True,
                        help="Directory the probe image subset is drawn from.")
    parser.add_argument("--output", required=True, help="Path to write probe.json.")
    parser.add_argument("--source", default="teacher", choices=["teacher", "student"],
                        help="Which backbone to probe. 'teacher' is what convert/export ship.")
    parser.add_argument("--num-images", type=int, default=64,
                        help="Probe subset size. Deterministic: the first N by sorted name.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--arm", default=os.environ.get("ARM", "unknown"))
    parser.add_argument("--pretrained", default=None,
                        help=("Pretrained snapshot that defines the anchor. Defaults to the "
                              "spec's train.pretrained_model_path."))
    parser.add_argument("--fp32", action="store_true", default=True,
                        help=("Run the probe in fp32 (default). Drift is measured at 1e-7 "
                              "scale at identity, which fp16 cannot resolve."))
    return parser.parse_args()


def load_probe_images(images_dir, num_images, image_size):
    """Load a deterministic image subset as a normalized float tensor.

    The subset is the first ``num_images`` files by sorted relative path, so every arm is
    probed on exactly the same pixels and the drift numbers are directly comparable.

    Args:
        images_dir (str): Directory to draw images from.
        num_images (int): Number of images to load.
        image_size (int): Square resize target, matching the spec's global crop size.

    Returns:
        torch.Tensor: ``[N, 3, image_size, image_size]``, ImageNet-normalized.
    """
    root = Path(images_dir)
    paths = sorted(
        (p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p.relative_to(root)),
    )[:num_images]
    if not paths:
        raise ValueError(f"No images found under {images_dir}")

    # DINOv2/DINOv3 expect ImageNet normalization, NOT [0,1]: the probe has to match the
    # train/inference pipeline or the "drift" measured is really a preprocessing mismatch.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    tensors = []
    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
            array = np.asarray(img, dtype=np.uint8)
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).float() / 255.0)
    batch = torch.stack(tensors)
    return (batch - mean) / std, [str(p.relative_to(root)) for p in paths]


def load_spec(spec_path):
    """Compose a training spec the way Hydra would, without running Hydra.

    The spec's ``defaults:`` list is a Hydra *directive*, not config data, so merging the YAML
    straight onto the structured schema raises ``ConfigKeyError: Key 'defaults' not in
    'ExperimentConfig'``. Strip it and merge the parents it names in order, so the probe sees
    exactly the config the training run used (crop size, LoRA rank, gram/preservation flags).

    Args:
        spec_path (str): Path to the arm's spec YAML.

    Returns:
        DictConfig: the composed configuration.
    """
    spec = OmegaConf.load(spec_path)
    parents = [d for d in spec.pop("defaults", []) if isinstance(d, str) and d != "_self_"]
    merged = OmegaConf.structured(ExperimentConfig())
    spec_dir = Path(spec_path).parent
    for parent in parents:
        merged = OmegaConf.merge(merged, load_spec(spec_dir / f"{parent}.yaml"))
    return OmegaConf.merge(merged, spec)


def gram_matrix(patch_tokens):
    """Cosine-similarity Gram matrix of patch tokens, per image.

    Args:
        patch_tokens (torch.Tensor): ``[B, T, D]``.

    Returns:
        torch.Tensor: ``[B, T, T]``.
    """
    normed = torch.nn.functional.normalize(patch_tokens.float(), dim=-1)
    return normed @ normed.transpose(1, 2)


def main():
    """Measure eval-mode unmasked drift against the frozen pretrained anchor."""
    args = parse_args()

    config = load_spec(args.spec)

    # fp32: the identity case sits at ~1e-07 and fp16 has ~1e-03 resolution there, so a
    # fp16 probe cannot tell "no drift" from "some drift". The xformers custom-attention path
    # hard-casts q/k/v to .half() and is therefore incompatible with fp32 -- disable it for
    # the probe only (a pre-existing constraint, unrelated to LoRA; STATUS non-numerical notes).
    if args.fp32:
        config.train.use_custom_attention = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DinoV3PlModel(config).to(device)

    # Order matters. restore_pretrained_weights() populates the backbones from the pretrained
    # snapshot and syncs gram_teacher from them -- that sync is what makes gram_teacher the
    # *pretrained* anchor. Injecting adapters after the load matches the training lifecycle
    # (design doc "Injection point and lifecycle") and leaves base keys untouched.
    #
    # pretrained_weights is an attribute train.py assigns before calling restore; it is not
    # set in __init__, so the probe has to assign it the same way or restore raises
    # AttributeError.
    pretrained = args.pretrained or config.train.pretrained_model_path
    if not pretrained:
        raise ValueError(
            "No pretrained snapshot: pass --pretrained or set train.pretrained_model_path. "
            "The anchor IS the pretrained backbone, so drift cannot be measured without it."
        )
    model.pretrained_weights = pretrained
    model.restore_pretrained_weights()
    if config.model.lora.enable:
        model.inject_lora_adapters()

    # Load the trained weights into the probed backbone ONLY. gram_teacher is deliberately not
    # re-synced afterwards: it must keep holding the pretrained weights, or we would be
    # comparing the arm against itself and would measure zero drift no matter what happened.
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trained_state = extract_backbone_state_dict(raw, source=args.source)
    backbone = getattr(model, args.source).backbone
    missing, unexpected = backbone.load_state_dict(trained_state, strict=False)
    lora_keys = [k for k in trained_state if "lora_" in k]

    model.eval()
    anchor = model.gram_teacher
    anchor.eval()

    image_size = int(config.dataset.transform.global_crops_size)
    images, image_names = load_probe_images(args.images_dir, args.num_images, image_size)

    cls_cosines, cls_mses, gram_mses = [], [], []
    with torch.no_grad():
        for start in range(0, images.shape[0], args.batch_size):
            batch = images[start:start + args.batch_size].to(device)
            if args.fp32:
                batch = batch.float()

            # masks=None: student and anchor see identical pixels. Under eval() there is no
            # stochastic depth either, so at identity these outputs must agree to ~1e-07.
            trained_out = backbone([batch], masks=None)[0]
            anchor_out = anchor(batch)

            trained_cls = trained_out["x_norm_clstoken"].float()
            anchor_cls = anchor_out["x_norm_clstoken"].float()
            cls_cosines.append(
                torch.nn.functional.cosine_similarity(trained_cls, anchor_cls, dim=-1).cpu()
            )
            cls_mses.append(
                ((trained_cls - anchor_cls) ** 2).mean(dim=-1).cpu()
            )
            gram_mses.append(
                ((gram_matrix(trained_out["x_norm_patchtokens"]) -
                  gram_matrix(anchor_out["x_norm_patchtokens"])) ** 2)
                .mean(dim=(1, 2)).cpu()
            )

    cls_cosine = torch.cat(cls_cosines)
    cls_mse = torch.cat(cls_mses)
    gram_mse = torch.cat(gram_mses)

    result = {
        "arm": args.arm,
        "checkpoint": os.path.abspath(args.checkpoint),
        "spec": os.path.abspath(args.spec),
        "pretrained": pretrained,
        "source": args.source,
        "protocol": {
            "mode": "eval",
            "masking": "none",
            "precision": "fp32" if args.fp32 else "amp",
            "num_images": int(images.shape[0]),
            "image_size": image_size,
            "anchor": "pretrained DINOv3 backbone (frozen gram_teacher)",
            "note": ("Replaces gate G4.5, which is not measurable from the in-training "
                     "preservation scalars -- those read ~0.93 cls_cos at provable identity."),
        },
        "lora": {
            "enabled": bool(config.model.lora.enable),
            "lora_keys_in_checkpoint": len(lora_keys),
            "rank": int(config.model.lora.rank),
            "alpha": float(config.model.lora.alpha),
        },
        "checkpoint_load": {
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            # Non-empty here is a real red flag: it means the trained backbone and the
            # freshly-built one disagree structurally (e.g. adapters not injected).
            "missing_sample": sorted(missing)[:10],
            "unexpected_sample": sorted(unexpected)[:10],
        },
        "drift": {
            # 1 - cos. 0.0 = no global drift. This is the number G4.2 ranks across arms.
            "cls_cosine_distance_mean": float((1.0 - cls_cosine).mean()),
            "cls_cosine_distance_max": float((1.0 - cls_cosine).max()),
            "cls_cosine_mean": float(cls_cosine.mean()),
            "cls_mse_mean": float(cls_mse.mean()),
            # Patch-geometry drift. This is the number G4.3 ranks across arms.
            "gram_mse_mean": float(gram_mse.mean()),
            "gram_mse_max": float(gram_mse.max()),
        },
        "images": image_names[:10],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result["drift"], indent=2))
    print(f"Wrote {args.output}")
    if lora_keys:
        print(f"lora_* keys present in checkpoint: {len(lora_keys)}")
    if missing or unexpected:
        print(f"WARNING: {len(missing)} missing / {len(unexpected)} unexpected keys on load")


if __name__ == "__main__":
    main()
