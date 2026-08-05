# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from vfm-eval/c-radiov4 eval_seg.py (Apache-2.0). The BNHead is the
# RADIO repo mmseg linear head (ported from Meta's DINOv2 repo, Apache-2.0).

"""Segmentation linear-probe evaluator (BNHead) — offline.

Protocol (preserve exactly — paper-validated, C-RADIOv4-H ADE20K 55.20 / VOC 87.24):
  - Backbone: frozen; spatial features reshaped to ``[B, D, H', W']``.
  - Head: ``BNHead`` = ``BatchNorm2d(D) → Conv1x1(D, num_classes)`` (SyncBN under DDP).
  - Optimizer: AdamW lr=1e-3, betas=(0.9,0.999), weight_decay=0.
  - Schedule: linear warmup 0→1500 (start 1e-6) then poly(power=1) decay to 80k.
  - Loss: CrossEntropy(ignore_index=255). Val: whole-image, bilinear upsample to
    original size, mIoU over classes present. ADE20K also reports small-object mIoU.

Train can run on cached dense features (backbone bypassed) or extract on the fly;
val supports whole-image or overlapping-tile (Gaussian-stitched) extraction.
"""

import logging
import math
import os

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank, get_world_size, is_dist_avail_and_initialized
from nvidia_tao_pytorch.core.evaluation.base import EvalContext, Evaluator, register_evaluator
from nvidia_tao_pytorch.core.evaluation.datasets.segmentation import build_seg_loaders
from nvidia_tao_pytorch.core.evaluation.model_adapter import features_to_map
from nvidia_tao_pytorch.core.evaluation.transforms import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)

SMALL_OBJECT_AREA_THRESH = 32 * 32   # COCO "small" instance threshold


def _is_main() -> bool:
    return get_global_rank() == 0


def _barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Head + schedule + optimizer
# ---------------------------------------------------------------------------
class BNHead(nn.Module):
    """``BatchNorm2d(D) → Conv1x1(D, num_classes)`` linear segmentation head."""

    def __init__(self, in_channels: int, num_classes: int):
        """Init BN + 1x1 conv."""
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, D, H', W']`` → logits ``[B, num_classes, H', W']``."""
        return self.conv_seg(self.bn(x))


def _get_lr(step, base_lr, warmup_steps=1500, total_steps=80_000) -> float:
    """Linear warmup to ``base_lr`` then poly(power=1) decay to 0."""
    if step < warmup_steps:
        return base_lr * max(1e-6, step / warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * max(0.0, 1.0 - progress)


def _set_lr(optimizer, lr: float):
    """Set LR on all param groups."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def _build_optimizer(head, lr=1e-3, weight_decay=0.0):
    """AdamW over trainable head params (betas=(0.9,0.999))."""
    params = [p for p in head.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))


# ---------------------------------------------------------------------------
# Spatial feature extraction (whole-image + tiled)
# ---------------------------------------------------------------------------
def _extract_spatial(model, images, patch_size, amp, device):
    """Forward the frozen backbone → spatial features ``[B, D, H', W']``."""
    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            _, features = model(images)
    return features_to_map(features, images, patch_size)


def _make_gaussian_weight(tile_feat_size: int, device) -> torch.Tensor:
    """2D Gaussian tile weight (center→1, edges→0) for overlap stitching."""
    coords = torch.linspace(-1, 1, tile_feat_size, device=device)
    g1d = torch.exp(-2.0 * coords ** 2)
    return g1d[:, None] * g1d[None, :]


@torch.no_grad()
def _tiled_extract_spatial(model, image, patch_size, amp, device,
                           tile_size=512, stride=256, tile_batch_size=8):
    """Stitch overlapping-tile features for a full-res image → ``[1, D, H', W']`` (CPU)."""
    _, _, H, W = image.shape
    tile_feat = tile_size // patch_size
    feat_H = math.ceil(H / patch_size)
    feat_W = math.ceil(W / patch_size)

    ys = list(range(0, H - tile_size + 1, stride))
    xs = list(range(0, W - tile_size + 1, stride))
    if not ys or ys[-1] + tile_size < H:
        ys.append(max(H - tile_size, 0))
    if not xs or xs[-1] + tile_size < W:
        xs.append(max(W - tile_size, 0))

    positions = [(y, x, y // patch_size, x // patch_size) for y in ys for x in xs]
    feat_sum = wgt_sum = gauss = None
    tile_batch_size = max(1, tile_batch_size)

    for start in range(0, len(positions), tile_batch_size):
        chunk = positions[start:start + tile_batch_size]
        tiles = []
        for y, x, _, _ in chunk:
            tile = image[:, :, y:y + tile_size, x:x + tile_size]
            th, tw = tile.shape[2], tile.shape[3]
            if th < tile_size or tw < tile_size:
                padded = torch.zeros(1, 3, tile_size, tile_size, dtype=tile.dtype, device=image.device)
                padded[:, :, :th, :tw] = tile
                tile = padded
            tiles.append(tile)

        tile_batch = torch.cat(tiles, dim=0).to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            _, features = model(tile_batch)
        features = features_to_map(features, tile_batch, patch_size)   # (N, D, tile_feat, tile_feat)
        D = features.shape[1]

        if feat_sum is None:
            gauss = _make_gaussian_weight(tile_feat, device=features.device)
            feat_sum = torch.zeros(1, D, feat_H, feat_W, device=features.device)
            wgt_sum = torch.zeros(1, 1, feat_H, feat_W, device=features.device)

        for i, (_, _, fy, fx) in enumerate(chunk):
            ey = min(fy + tile_feat, feat_H)
            ex = min(fx + tile_feat, feat_W)
            sy, sx = ey - fy, ex - fx
            feat_sum[:, :, fy:ey, fx:ex] += features[i:i + 1, :, :sy, :sx] * gauss[:sy, :sx]
            wgt_sum[:, :, fy:ey, fx:ex] += gauss[:sy, :sx]

    return (feat_sum / wgt_sum.clamp(min=1e-6)).cpu()


# ---------------------------------------------------------------------------
# IoU accumulation
# ---------------------------------------------------------------------------
class IoUAccumulator:
    """Per-class intersection/union accumulator → macro mIoU over present classes."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        """Init zeroed intersection/union vectors."""
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.intersection = torch.zeros(num_classes, dtype=torch.float64)
        self.union = torch.zeros(num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Accumulate from ``(H, W)`` prediction/target maps."""
        preds = preds.cpu().long()
        targets = targets.cpu().long()
        valid = targets != self.ignore_index
        for c in range(self.num_classes):
            pred_c = (preds == c) & valid
            target_c = (targets == c) & valid
            self.intersection[c] += (pred_c & target_c).sum().item()
            self.union[c] += (pred_c | target_c).sum().item()

    def miou(self) -> float:
        """Macro mIoU (%) over classes with non-zero union."""
        valid = self.union > 0
        if not valid.any():
            return float("nan")
        iou = self.intersection[valid] / (self.union[valid] + 1e-10)
        return iou.mean().item() * 100.0

    def reduce(self, device):
        """All-reduce intersection/union across ranks (no-op single-GPU)."""
        if is_dist_avail_and_initialized():
            inter = self.intersection.to(device)
            union = self.union.to(device)
            dist.all_reduce(inter, op=dist.ReduceOp.SUM)
            dist.all_reduce(union, op=dist.ReduceOp.SUM)
            self.intersection = inter.cpu()
            self.union = union.cpu()

    def per_class_iou(self) -> torch.Tensor:
        """Per-class IoU (%), NaN where union is 0."""
        return torch.where(
            self.union > 0,
            self.intersection / (self.union + 1e-10) * 100.0,
            torch.full_like(self.intersection, float("nan")),
        )


def _build_small_object_pixel_mask(targets, area_thresh=SMALL_OBJECT_AREA_THRESH, ignore_index=255):
    """True where a pixel lies in a GT connected component with area < ``area_thresh``."""
    mask = np.zeros(targets.shape, dtype=bool)
    valid = targets != ignore_index
    if not valid.any():
        return mask
    for class_id in np.unique(targets[valid]):
        cls_mask = (targets == int(class_id)).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue
        _, labels, stats, _ = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)
        for comp_idx in range(1, int(labels.max()) + 1):
            if int(stats[comp_idx, cv2.CC_STAT_AREA]) < area_thresh:
                mask |= labels == comp_idx
    return mask


class SmallObjectIoUAccumulator(IoUAccumulator):
    """IoU restricted to small-GT-component pixels (ADE20K diagnostic)."""

    def __init__(self, num_classes, ignore_index=255, area_thresh=SMALL_OBJECT_AREA_THRESH):
        """Init with a small-component area threshold."""
        super().__init__(num_classes, ignore_index)
        self.area_thresh = area_thresh

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Accumulate only over pixels in small GT components."""
        preds = preds.cpu().long()
        targets = targets.cpu().long()
        small_mask = torch.from_numpy(_build_small_object_pixel_mask(
            targets.numpy(), area_thresh=self.area_thresh, ignore_index=self.ignore_index))
        if not small_mask.any():
            return
        for c in range(self.num_classes):
            pred_c = (preds == c) & small_mask
            target_c = (targets == c) & small_mask
            self.intersection[c] += (pred_c & target_c).sum().item()
            self.union[c] += (pred_c | target_c).sum().item()


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def _save_checkpoint(path, step, best_miou, epoch, head, optimizer):
    """Persist head + optimizer + step/best_miou/epoch for resume."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"step": step, "best_miou": best_miou, "epoch": epoch,
                "head": head.state_dict(), "optimizer": optimizer.state_dict()}, path)


def _load_checkpoint(path, head, optimizer):
    """Restore head + optimizer; return (step, best_miou, epoch)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    head.load_state_dict(ckpt["head"])
    optimizer.load_state_dict(ckpt["optimizer"])
    # ``epoch`` was added later; default to 0 for checkpoints saved before then.
    return ckpt["step"], ckpt["best_miou"], ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def _validate(model, head, val_loader, device, patch_size, num_classes, ignore_index, amp,
              tiling=False, tile_size=512, tile_stride=256, tile_batch_size=8,
              compute_small_object=False):
    """Whole-image (or tiled) validation → mIoU (and small-object mIoU if requested)."""
    accum = IoUAccumulator(num_classes, ignore_index)
    small_accum = SmallObjectIoUAccumulator(num_classes, ignore_index) if compute_small_object else None
    head_module = head.module if isinstance(head, DDP) else head
    head_module.eval()

    for images, masks, orig_sizes in val_loader:
        images = images.to(device, non_blocking=True)
        orig_h, orig_w = orig_sizes[0, 0].item(), orig_sizes[0, 1].item()
        if model is None:
            feat = images
        elif tiling:
            feat = _tiled_extract_spatial(model, images, patch_size, amp, device,
                                          tile_size=tile_size, stride=tile_stride,
                                          tile_batch_size=tile_batch_size).to(device)
        else:
            feat = _extract_spatial(model, images, patch_size, amp, device)
        with torch.no_grad():
            logits = head_module(feat)
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1).squeeze(0)
        target = masks.squeeze(0)
        accum.update(preds, target)
        if small_accum is not None:
            small_accum.update(preds, target)

    accum.reduce(device)
    if small_accum is not None:
        small_accum.reduce(device)
        return accum.miou(), small_accum.miou()
    return accum.miou()


def _train(model, head, train_loader, device, patch_size, ignore_index, total_iters, amp,
           val_every=2000, val_fn=None, checkpoint_path=None, train_sampler=None,
           base_lr=1e-3, warmup_steps=1500, weight_decay=0.0):
    """Train the head for ``total_iters`` (resumable); periodic val; return best mIoU."""
    head_module = head.module if isinstance(head, DDP) else head
    optimizer = _build_optimizer(head_module, lr=base_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    step, best_miou, epoch = 0, 0.0, 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        _barrier()
        step, best_miou, epoch = _load_checkpoint(checkpoint_path, head_module, optimizer)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    head.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    data_iter = iter(train_loader)

    while step < total_iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        images, masks = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        _set_lr(optimizer, _get_lr(step, base_lr, warmup_steps, total_iters))

        feat = images if model is None else _extract_spatial(model, images, patch_size, amp, device)
        logits = head(feat)
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = criterion(logits, masks.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % 200 == 0 and _is_main():
            logger.info("[SEG] iter %d/%d  loss %.4f", step, total_iters, loss.item())

        if val_fn is not None and step % val_every == 0:
            head.eval()
            miou = val_fn(model, head, device, patch_size, amp)
            head.train()
            if _is_main():
                best_miou = max(best_miou, miou)
                logger.info("[SEG] iter %d  val mIoU %.2f%% (best %.2f%%)", step, miou, best_miou)
                if checkpoint_path:
                    _save_checkpoint(checkpoint_path, step, best_miou, epoch, head_module, optimizer)
            _barrier()

    if checkpoint_path and _is_main():
        _save_checkpoint(checkpoint_path, step, best_miou, epoch, head_module, optimizer)
    _barrier()
    return best_miou


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
@register_evaluator
class SegmentationEvaluator(Evaluator):
    """Linear-probe (BNHead) semantic-segmentation mIoU on dense features."""

    name = "segmentation"
    requires_fit = True
    feature_level = "dense"
    supports_online = False

    def run(self, ctx: EvalContext):
        """Train the BNHead probe and return ``{seg_miou[, seg_small_object_miou]}``."""
        cfg = getattr(ctx.cfg, self.name)
        dataset = getattr(cfg, "dataset", "ade20k")
        amp = getattr(cfg, "amp", True)
        patch_size = int(getattr(ctx.model, "patch_size", 16))
        feature_dim = int(getattr(ctx.model, "feature_dim", 0))
        if not feature_dim:
            raise ValueError("SegmentationEvaluator needs ctx.model.feature_dim (adapter dim).")
        pad_divisor = getattr(cfg, "pad_divisor", patch_size)
        total_iters = getattr(cfg, "total_iters", 80_000)
        tiling = getattr(cfg, "tiling", False)
        compute_small_object = dataset == "ade20k" and getattr(cfg, "compute_small_object_iou", True)

        mean = std = None
        if getattr(cfg, "imagenet_normalize", False):
            mean, std = IMAGENET_MEAN, IMAGENET_STD

        feature_cache_dir = getattr(cfg, "feature_cache_dir", None)
        train_loader, val_loader, num_classes, ignore_index, train_sampler = build_seg_loaders(
            dataset, getattr(cfg, "root", None),
            batch_size=getattr(cfg, "batch_size", 2),
            num_workers=getattr(cfg, "num_workers", 4),
            distributed=ctx.distributed,
            tiling=tiling,
            feature_cache_dir=feature_cache_dir,
            crop_size=getattr(cfg, "crop_size", 512),
            pad_divisor=pad_divisor,
            mean=mean, std=std,
            max_val_samples=getattr(cfg, "max_val_samples", 0),
        )
        # Cached features bypass the backbone entirely.
        model = None if feature_cache_dir else ctx.model

        head = BNHead(in_channels=feature_dim, num_classes=num_classes).to(ctx.device)
        if ctx.distributed and get_world_size() > 1:
            head = nn.SyncBatchNorm.convert_sync_batchnorm(head)
            head = DDP(head, device_ids=[ctx.device.index])

        def val_fn(m, h, dev, ps, a):
            out = _validate(m, h, val_loader, dev, ps, num_classes, ignore_index, a,
                            tiling=tiling, tile_size=getattr(cfg, "tile_size", 512),
                            tile_stride=getattr(cfg, "tile_stride", 256),
                            tile_batch_size=getattr(cfg, "tile_batch_size", 8),
                            compute_small_object=False)
            return out[0] if isinstance(out, tuple) else out

        logger.info("[SEG] %s | %d classes | %d iters | crop=%d | tiling=%s",
                    dataset, num_classes, total_iters, getattr(cfg, "crop_size", 512), tiling)
        best_miou = _train(
            model, head, train_loader, ctx.device, patch_size, ignore_index, total_iters, amp,
            val_every=getattr(cfg, "val_every", 2000), val_fn=val_fn,
            checkpoint_path=(os.path.join(ctx.results_dir, f"seg_{dataset}_head.pt")
                             if ctx.results_dir else None),
            train_sampler=train_sampler,
            base_lr=getattr(cfg, "base_lr", 1e-3),
            warmup_steps=getattr(cfg, "warmup_steps", 1500),
            weight_decay=getattr(cfg, "weight_decay", 0.0),
        )

        val_out = _validate(model, head, val_loader, ctx.device, patch_size, num_classes,
                            ignore_index, amp, tiling=tiling,
                            tile_size=getattr(cfg, "tile_size", 512),
                            tile_stride=getattr(cfg, "tile_stride", 256),
                            tile_batch_size=getattr(cfg, "tile_batch_size", 8),
                            compute_small_object=compute_small_object)
        results = {}
        if isinstance(val_out, tuple):
            results["seg_miou"] = val_out[0]
            results["seg_small_object_miou"] = val_out[1]
        else:
            results["seg_miou"] = val_out
        results["seg_best_miou"] = best_miou
        if _is_main():
            logger.info("[SEG] %s final mIoU %.2f%% (best %.2f%%)", dataset, results["seg_miou"], best_miou)
        return results
