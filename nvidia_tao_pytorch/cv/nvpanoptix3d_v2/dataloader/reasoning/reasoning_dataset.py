# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoning-segmentation dataset and collator, read from a JSONL manifest."""

from __future__ import annotations

import json
import os.path as osp
import random
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.utils import (
    center_crop_and_resize,
    normalize_resolution_arg,
    rgb2id,
)

SEG_TOKEN = "[SEG]"
_MAX_GETITEM_RETRIES = 5


def resolve_manifest_path(path: Optional[str], manifest_dir: str) -> Optional[str]:
    """Resolve relative assets next to the manifest and preserve absolute paths."""
    if path is None:
        return None
    value = str(path)
    if not osp.isabs(value):
        value = osp.join(manifest_dir, value)
    return osp.normpath(value)


class ReasoningSegDataset(Dataset):
    """Reads a JSONL manifest of (multi-view scene, implicit query) samples.

    Args:
        manifest: path to the JSONL manifest (see module docstring).
        resolution: target ``(H, W)`` (single tuple). All views resized to it.
        num_views: expected views per sample; samples with fewer are padded by
            repeating the last view, more are truncated (keeps batch stacking
            uniform). Set ``None`` to require exact match.
        require_seg_token: if True, drop manifest rows whose ``answer`` lacks
            ``[SEG]`` (and have a positive ``target_inst_id``); kept otherwise.
        depth_scale: stored depth units per meter.
    """

    def __init__(
        self,
        manifest: str,
        resolution: Tuple[int, int] = (518, 518),
        num_views: Optional[int] = 5,
        require_seg_token: bool = False,
        depth_scale: float = 1000.0,
    ):
        super().__init__()
        self.manifest_path = manifest
        self.resolution = normalize_resolution_arg(resolution)[0]
        self.num_views = num_views
        self.require_seg_token = require_seg_token
        self.depth_scale = float(depth_scale)
        if self.depth_scale <= 0:
            raise ValueError(f"depth_scale must be positive, got {depth_scale!r}")
        self.records: List[Dict[str, Any]] = self.load_manifest(manifest)

    def load_manifest(self, path: str) -> List[Dict[str, Any]]:
        """Load, validate, and resolve asset paths in a JSONL manifest."""
        recs: List[Dict[str, Any]] = []
        manifest_dir = osp.dirname(osp.abspath(path))
        with open(path) as f:
            for ln, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "images" not in rec or "instruction" not in rec or "answer" not in rec:
                    raise ValueError(
                        f"{path}:{ln+1} manifest record missing required keys "
                        f"(images/instruction/answer)."
                    )
                rec = dict(rec)
                if self.require_seg_token and int(rec.get("target_inst_id", 0)) > 0 \
                        and SEG_TOKEN not in rec["answer"]:
                    continue
                record_scale = float(rec.get("depth_scale", self.depth_scale))
                if record_scale <= 0:
                    raise ValueError(
                        "record depth_scale must be positive, "
                        f"got {record_scale!r} at {path}:{ln + 1}"
                    )
                images = [
                    resolve_manifest_path(asset, manifest_dir)
                    for asset in rec["images"]
                ]
                pans = rec.get("panoptic")
                if pans is not None:
                    pans = [
                        resolve_manifest_path(asset, manifest_dir)
                        for asset in pans
                    ]
                depths = rec.get("depth")
                if depths is None:
                    depths = [None] * len(images)
                else:
                    depths = list(depths)
                depths = [
                    resolve_manifest_path(asset, manifest_dir)
                    for asset in depths
                ]
                rec["images"] = images
                rec["panoptic"] = pans
                rec["depth"] = depths
                rec["_depth_scale"] = record_scale
                recs.append(rec)
        if not recs:
            raise RuntimeError(f"No usable records in manifest {path}")
        return recs

    def __len__(self) -> int:
        """Return the number of manifest records."""
        return len(self.records)

    def fit_num_views(self, paths: List[str]) -> List[int]:
        """Return index list of length ``num_views`` (pad/truncate)."""
        n = len(paths)
        if self.num_views is None:
            return list(range(n))
        if n >= self.num_views:
            return list(range(self.num_views))
        return list(range(n)) + [n - 1] * (self.num_views - n)  # repeat last

    def load_view(
        self,
        img_path: str,
        pan_path: Optional[str],
        depth_path: Optional[str],
        cls_sep: int,
        depth_scale: float,
    ) -> Dict[str, Any]:
        """Load and align one reasoning sample view and its annotations."""
        tgt_h, tgt_w = self.resolution
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        if pan_path:
            pan_rgb = np.array(Image.open(pan_path))
            pan_id = rgb2id(pan_rgb).astype(np.int32)
        else:
            pan_id = np.zeros((H, W), dtype=np.int32)

        if depth_path:
            depth = np.array(Image.open(depth_path), dtype=np.float32)
            depth /= depth_scale
            depth[~np.isfinite(depth)] = 0.0
        else:
            depth = np.zeros((H, W), dtype=np.float32)

        # Centered default intrinsics so the PP-crop degenerates to a center crop.
        K = np.array([[max(H, W), 0, W / 2.0],
                      [0, max(H, W), H / 2.0],
                      [0, 0, 1.0]], dtype=np.float32)

        img, K, depth, pan_id = center_crop_and_resize(
            img, K, depth, pan_id, (tgt_h, tgt_w)
        )

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        inst_id = (pan_id // cls_sep).astype(np.int64) if pan_id is not None \
            else np.zeros((tgt_h, tgt_w), dtype=np.int64)
        return {
            "img": img_t,
            "true_shape": torch.tensor([tgt_h, tgt_w], dtype=torch.int32),
            "pan_inst_id": torch.from_numpy(inst_id),
            "depthmap": torch.from_numpy(depth),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one multi-view reasoning sample."""
        for attempt in range(_MAX_GETITEM_RETRIES):
            rec = self.records[idx % len(self.records)]
            try:
                cls_sep = int(rec.get("cls_sep", 256))
                depth_scale = float(rec.get("_depth_scale", self.depth_scale))
                view_idx = self.fit_num_views(rec["images"])
                pans = rec.get("panoptic") or [None] * len(rec["images"])
                depths = rec.get("depth") or [None] * len(rec["images"])
                views = [
                    self.load_view(
                        rec["images"][i],
                        pans[i] if i < len(pans) else None,
                        depths[i] if i < len(depths) else None,
                        cls_sep,
                        depth_scale,
                    )
                    for i in view_idx
                ]
                target = int(rec.get("target_inst_id") or 0)
                raw_svi = rec.get("seg_view_indices")
                if raw_svi is None:
                    seg_view_indices = [0]
                else:
                    kept = len(view_idx)  # views actually loaded (after pad/truncate)
                    seg_view_indices = [int(v) for v in raw_svi if 0 <= int(v) < kept]
                    if not seg_view_indices:
                        seg_view_indices = [0]
                return {
                    "views": views,
                    "instruction": str(rec["instruction"]),
                    "answer": str(rec["answer"]),
                    "target_inst_id": target,
                    "seg_view_indices": seg_view_indices,
                    "scene": str(rec.get("scene", "")),
                }
            except Exception:  # pragma: no cover
                if attempt == _MAX_GETITEM_RETRIES - 1:
                    traceback.print_exc(file=sys.stderr)
                    raise
                idx = random.randint(0, len(self) - 1)

        # Unreachable: the final attempt either returns or raises above. Kept
        # so every path out of the method is explicit.
        raise RuntimeError(f"__getitem__ exhausted all attempts for idx={idx}")


class ReasoningCollator:
    """Collate samples into a per-view list with QA attached to view 0.

    Returns ``List[S]`` of dicts; view ``s`` has stacked tensors
    ``img`` ``[B,3,H,W]``, ``true_shape`` ``[B,2]``, ``pan_inst_id`` ``[B,H,W]``,
    and ``depthmap`` ``[B,H,W]``.
    The first view dict additionally carries the batch QA:
      * ``instruction``    : ``List[str]`` length B
      * ``answer``         : ``List[str]`` length B
      * ``target_inst_id`` : ``LongTensor[B]``
    """

    def __call__(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collate samples into the per-view dict list the model consumes."""
        if not samples:
            raise ValueError("empty batch")
        S = len(samples[0]["views"])
        if any(len(s["views"]) != S for s in samples):
            raise ValueError(
                "All samples in a batch must have the same number of views; "
                "set dataset.num_views to enforce this."
            )
        out: List[Dict[str, Any]] = []
        for s in range(S):
            view = {
                "img": torch.stack([smp["views"][s]["img"] for smp in samples], dim=0),
                "true_shape": torch.stack(
                    [smp["views"][s]["true_shape"] for smp in samples], dim=0
                ),
                "pan_inst_id": torch.stack(
                    [smp["views"][s]["pan_inst_id"] for smp in samples], dim=0
                ),
                "depthmap": torch.stack(
                    [smp["views"][s]["depthmap"] for smp in samples], dim=0
                ),
            }
            out.append(view)
        out[0]["instruction"] = [smp["instruction"] for smp in samples]
        out[0]["answer"] = [smp["answer"] for smp in samples]
        out[0]["target_inst_id"] = torch.tensor(
            [smp["target_inst_id"] for smp in samples], dtype=torch.long
        )
        out[0]["scene"] = [smp["scene"] for smp in samples]
        out[0]["seg_view_indices"] = [smp["seg_view_indices"] for smp in samples]
        return out
