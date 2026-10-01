# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train the frozen Loose-to-Tight MLP from frame-grouped ``ltt_data/v2`` caches."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Optional, Sequence, Tuple
import warnings

import torch

from nvidia_tao_pytorch.cv.sparse4d.model.loose_to_tight_mlp import (
    LooseToTightMLP,
    WAREHOUSE_V4_CLASSES,
)
from nvidia_tao_pytorch.cv.sparse4d.tools import ltt_data


def resolve_data_paths(patterns: Sequence[str]) -> list:
    """Expand input globs while preserving order and removing duplicates."""
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    return list(dict.fromkeys(paths))


def load_dataset(
    paths: Sequence[str], min_visibility: float = 0.0
) -> Tuple[dict, Optional[list], list]:
    """Load cache shards into model-ready CPU tensors."""
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be in [0, 1]")
    packed, class_id, group_id, metadata = ltt_data.load_cache(paths)
    if min_visibility > 0.0:
        keep = packed[:, ltt_data.I_VIS] >= min_visibility
        packed, class_id, group_id = packed[keep], class_id[keep], group_id[keep]
    if not len(class_id):
        raise ValueError("Loose-to-Tight cache is empty after filtering")

    class_names = None
    for meta in metadata:
        shard_names = meta.get("class_names")
        if shard_names is not None:
            shard_names = list(shard_names)
            if class_names is not None and shard_names != class_names:
                raise ValueError("Loose-to-Tight cache shards disagree on class_names")
            class_names = shard_names
    tensor = torch.as_tensor
    return (
        {
            "class_id": tensor(class_id, dtype=torch.long),
            "group_id": tensor(group_id, dtype=torch.long),
            "extent_wlh": tensor(packed[:, ltt_data.SL_EXTENT], dtype=torch.float32),
            "theta": tensor(packed[:, ltt_data.I_THETA], dtype=torch.float32),
            "phi": tensor(packed[:, ltt_data.I_PHI], dtype=torch.float32),
            "psi": tensor(packed[:, ltt_data.I_PSI], dtype=torch.float32),
            "dist": tensor(packed[:, ltt_data.I_DIST], dtype=torch.float32),
            "loose": tensor(packed[:, ltt_data.SL_LOOSE], dtype=torch.float32),
            "tight": tensor(packed[:, ltt_data.SL_TIGHT], dtype=torch.float32),
            "img_wh": tensor(packed[:, ltt_data.SL_IMGWH], dtype=torch.float32),
        },
        class_names,
        metadata,
    )


def resolve_class_names(
    cache_class_names: Optional[Sequence[str]],
    class_config: Optional[str] = None,
) -> list:
    """Resolve training taxonomy without reinterpreting cached class IDs."""
    configured_names = None
    if class_config:
        configured_names, _ = ltt_data.load_class_taxonomy(class_config)

    if cache_class_names is not None:
        cache_names = list(cache_class_names)
        if configured_names is not None and configured_names != cache_names:
            raise ValueError(
                "--class-config taxonomy must exactly match cache class_names "
                f"and order: config={configured_names!r}, cache={cache_names!r}"
            )
        return cache_names

    if configured_names is not None:
        return configured_names
    return list(WAREHOUSE_V4_CLASSES)


def giou_xyxy(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compute paired generalized IoU for xyxy boxes."""
    ax0, ay0, ax1, ay1 = first.unbind(dim=-1)
    bx0, by0, bx1, by1 = second.unbind(dim=-1)
    area_a = (ax1 - ax0).clamp_min(0.0) * (ay1 - ay0).clamp_min(0.0)
    area_b = (bx1 - bx0).clamp_min(0.0) * (by1 - by0).clamp_min(0.0)
    intersection = (torch.minimum(ax1, bx1) - torch.maximum(ax0, bx0)).clamp_min(
        0.0
    ) * (torch.minimum(ay1, by1) - torch.maximum(ay0, by0)).clamp_min(0.0)
    union = (area_a + area_b - intersection).clamp_min(1e-7)
    iou = intersection / union
    enclosing = (
        (torch.maximum(ax1, bx1) - torch.minimum(ax0, bx0)).clamp_min(0.0) *
        (torch.maximum(ay1, by1) - torch.minimum(ay0, by0)).clamp_min(0.0)
    ).clamp_min(1e-7)
    return iou - (enclosing - union) / enclosing


def per_class_giou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_id: torch.Tensor,
    class_names: Sequence[str],
) -> dict:
    """Summarize paired GIoU globally and by class."""
    values = giou_xyxy(prediction, target)
    metrics = {"_mean": float(values.mean().item())}
    for index, name in enumerate(class_names):
        keep = class_id == index
        if bool(keep.any()):
            metrics[name] = float(values[keep].mean().item())
    return metrics


def split_group_indices(
    group_id: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Partition whole source-frame groups into deterministic train/validation sets."""
    groups = torch.as_tensor(group_id, dtype=torch.long).flatten().cpu()
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    unique_groups = torch.unique(groups, sorted=True)
    if unique_groups.numel() < 2:
        raise ValueError("At least two source-frame groups are required")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(unique_groups.numel(), generator=generator)
    validation_count = min(
        unique_groups.numel() - 1,
        max(1, int(round(unique_groups.numel() * val_fraction))),
    )
    validation_groups = unique_groups[permutation[:validation_count]]
    validation_mask = torch.isin(groups, validation_groups)
    validation_indices = torch.nonzero(validation_mask, as_tuple=False).flatten()
    training_indices = torch.nonzero(~validation_mask, as_tuple=False).flatten()
    return training_indices, validation_indices


def train_mlp(
    data: dict,
    class_names: Sequence[str],
    *,
    hidden_dim: int = 64,
    epochs: int = 60,
    batch_size: int = 8192,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    loss_name: str = "mse",
    aux_weight: float = 0.1,
    lr_schedule: str = "none",
    val_fraction: float = 0.1,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Tuple[LooseToTightMLP, dict]:
    """Fit a model and return the frozen best checkpoint plus metrics."""
    if loss_name not in {"mse", "giou"}:
        raise ValueError("loss_name must be 'mse' or 'giou'")
    if lr_schedule not in {"none", "cosine"}:
        raise ValueError("lr_schedule must be 'none' or 'cosine'")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    sample_count = int(data["class_id"].numel())
    if sample_count < 2:
        raise ValueError("At least two samples are required to train and validate")
    if int(data["class_id"].min()) < 0 or int(data["class_id"].max()) >= len(
        class_names
    ):
        raise ValueError("class_id contains a value outside class_names")

    if data["group_id"].numel() != sample_count:
        raise ValueError("group_id must contain one value per training sample")
    torch.manual_seed(seed)
    device = torch.device(device)
    model = LooseToTightMLP(
        num_classes=len(class_names),
        hidden_dim=hidden_dim,
        class_names=list(class_names),
    ).to(device)
    features = model.featurize(
        data["class_id"],
        data["extent_wlh"],
        data["theta"],
        data["phi"],
        data["psi"],
        data["loose"],
        data["img_wh"],
        data["dist"],
    ).to(device)
    loose = data["loose"].to(device)
    tight = data["tight"].to(device)
    classes = data["class_id"].to(device)
    scale_target = model.shrinkage_target(loose, tight)

    training_indices, validation_indices = split_group_indices(
        data["group_id"], val_fraction, seed
    )
    training_indices = training_indices.to(device)
    validation_indices = validation_indices.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if lr_schedule == "cosine"
        else None
    )
    mse = torch.nn.MSELoss()

    def batch_loss(indices: torch.Tensor) -> torch.Tensor:
        predicted_scale = model(features[indices])
        scale_loss = mse(predicted_scale, scale_target[indices])
        if loss_name == "mse":
            return scale_loss
        predicted_box = model.apply_to_loose(loose[indices], predicted_scale)
        return (
            1.0 -
            giou_xyxy(predicted_box, tight[indices]).mean() +
            aux_weight * scale_loss
        )

    best_mean = float("-inf")
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        order = training_indices[
            torch.randperm(training_indices.numel(), device=device)
        ]
        total_loss = 0.0
        for start in range(0, order.numel(), batch_size):
            indices = order[start: start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(indices)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * indices.numel()
        if scheduler is not None:
            scheduler.step()
        model.eval()
        with torch.no_grad():
            predicted_box = model.apply_to_loose(
                loose[validation_indices], model(features[validation_indices])
            )
            metrics = per_class_giou(
                predicted_box,
                tight[validation_indices],
                classes[validation_indices],
                class_names,
            )
        mean_loss = total_loss / training_indices.numel()
        history.append(
            {"epoch": epoch, "loss": mean_loss, "val_giou": metrics["_mean"]}
        )
        if metrics["_mean"] > best_mean:
            best_mean = metrics["_mean"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training produced no model checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        final_box = model.apply_to_loose(
            loose[validation_indices], model(features[validation_indices])
        )
        final_metrics = per_class_giou(
            final_box,
            tight[validation_indices],
            classes[validation_indices],
            class_names,
        )
        baseline_metrics = per_class_giou(
            loose[validation_indices],
            tight[validation_indices],
            classes[validation_indices],
            class_names,
        )
    return model.freeze(), {
        "best_val_giou": best_mean,
        "val_giou": final_metrics,
        "baseline_giou": baseline_metrics,
        "num_train": int(training_indices.numel()),
        "num_val": int(validation_indices.numel()),
        "history": history,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--class-config")
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", choices=["mse", "giou"], default="mse")
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--forklift-gate", type=float, default=0.9)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    paths = resolve_data_paths(args.data)
    data, class_names, cache_metadata = load_dataset(paths, args.min_visibility)
    class_names = resolve_class_names(class_names, args.class_config)
    model, result = train_mlp(
        data,
        class_names,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        aux_weight=args.aux_weight,
        lr_schedule=args.lr_schedule,
        val_fraction=args.val_frac,
        seed=args.seed,
        device=args.device,
    )
    forklift_giou = result["val_giou"].get("forklift")
    if forklift_giou is not None and forklift_giou < args.forklift_gate:
        warnings.warn(
            f"forklift GIoU {forklift_giou:.3f} is below gate {args.forklift_gate}",
            stacklevel=1,
        )
    model.save(
        args.out,
        extra_meta={
            **result,
            "class_names": list(class_names),
            "data_paths": paths,
            "data_meta": cache_metadata,
            "train_cfg": {
                "loss": args.loss,
                "hidden_dim": args.hidden_dim,
                "epochs": args.epochs,
                "lr": args.lr,
                "lr_schedule": args.lr_schedule,
                "aux_weight": args.aux_weight,
            },
        },
    )
    print(
        f"[saved] {Path(args.out)}: best validation mean GIoU "
        f"{result['best_val_giou']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
