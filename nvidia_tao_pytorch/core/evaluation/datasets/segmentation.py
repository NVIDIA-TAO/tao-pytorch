# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Datasets vendored from vfm-eval/c-radiov4 common/datasets.py (Apache-2.0).

"""Segmentation datasets for the linear-probe evaluator (ADE20K / VOC / Cityscapes).

Each dataset returns, with mmseg-parity transforms (``../transforms.py``):
    train: ``(img[3,H,W], mask[H,W])``  — resized/cropped/augmented
    val:   ``(img[3,H',W'], mask[H_orig,W_orig], (H_orig, W_orig))`` — image is the
           resized+padded model input; the mask is kept at ORIGINAL resolution
           (mmseg evaluates by upsampling logits back to the original size).

``CachedSegDataset`` loads pre-extracted tiled features (see ``caching.py``) so the
head trains without the backbone in the loop. ``mean``/``std`` are threaded into the
transforms for backbones whose adapter does not normalize (NV-DINOv2); leave ``None``
for backbones that normalize internally (RADIO).
"""

import glob
import os
import random as _random
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank, get_world_size
from nvidia_tao_pytorch.core.evaluation.transforms import (
    get_train_transforms,
    get_val_img_transform,
    get_val_img_transform_tiling,
)

# Per-dataset metadata (num_classes, ignore_index, paper mIoU for C-RADIOv4-H).
SEG_DATASETS = {
    "ade20k": {"num_classes": 150, "ignore_index": 255, "paper_miou": 55.20},
    "voc": {"num_classes": 21, "ignore_index": 255, "paper_miou": 87.24},
    "cityscapes": {"num_classes": 19, "ignore_index": 255, "paper_miou": None},
}

# Cityscapes raw labelId (0-33) → trainId (0-18); everything else → 255.
_CITYSCAPES_ID_TO_TRAINID = np.full(256, 255, dtype=np.uint8)
for _raw, _train in [
    (7, 0), (8, 1), (11, 2), (12, 3), (13, 4), (17, 5), (19, 6), (20, 7),
    (21, 8), (22, 9), (23, 10), (24, 11), (25, 12), (26, 13), (27, 14),
    (28, 15), (31, 16), (32, 17), (33, 18),
]:
    _CITYSCAPES_ID_TO_TRAINID[_raw] = _train


def _reduce_zero_label(mask_arr: np.ndarray) -> np.ndarray:
    """ADE20K: map label 0 → 255 (ignore), shift 1..N → 0..N-1."""
    out = mask_arr.astype(np.int32) - 1
    out[out == -1] = 255
    return out.astype(np.uint8)


def _open_rgb_with_retry(path: str, attempts: int = 5, delay: float = 1.0) -> Image.Image:
    """Open an RGB image, retrying transient CIFS/NFS read errors."""
    last_err = None
    for attempt in range(attempts):
        try:
            with Image.open(path) as image:
                return image.convert("RGB").copy()
        except (FileNotFoundError, OSError) as err:
            last_err = err
            if attempt + 1 == attempts:
                break
            time.sleep(delay * (attempt + 1))
    raise last_err


def _open_mask_array_with_retry(path: str, attempts: int = 5, delay: float = 1.0) -> np.ndarray:
    """Open a mask as a numpy array, retrying transient read errors."""
    last_err = None
    for attempt in range(attempts):
        try:
            with Image.open(path) as mask:
                return np.array(mask)
        except (FileNotFoundError, OSError) as err:
            last_err = err
            if attempt + 1 == attempts:
                break
            time.sleep(delay * (attempt + 1))
    raise last_err


class ADE20KDataset(Dataset):
    """ADE20K segmentation (``reduce_zero_label`` per mmseg LoadAnnotations)."""

    NUM_CLASSES = 150
    IGNORE_INDEX = 255

    def __init__(self, root, split="train", crop_size=(512, 512), tiling=False,
                 min_short_side=0, pad_divisor=16, train_scale=(2048, 512),
                 val_scale=(2048, 512), mean=None, std=None):
        """Index ADE20K {images,annotations}/{training,validation} + build transforms."""
        assert split in ("train", "val")
        self.is_train = split == "train"
        sub = "training" if self.is_train else "validation"
        img_dir = os.path.join(root, "images", sub)
        ann_dir = os.path.join(root, "annotations", sub)
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        assert img_paths, f"No images found under {img_dir}"
        if min_short_side > 0:
            img_paths = [ip for ip in img_paths if min(Image.open(ip).size) >= min_short_side]
        self.samples = []
        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            ap = os.path.join(ann_dir, stem + ".png")
            if os.path.exists(ap):
                self.samples.append((ip, ap))
        self.train_tf = get_train_transforms(
            crop_size=crop_size, pad_divisor=pad_divisor, scale=train_scale, mean=mean, std=std)
        self.val_tf = (get_val_img_transform_tiling(mean=mean, std=std) if tiling
                       else get_val_img_transform(pad_divisor=pad_divisor, scale=val_scale,
                                                  mean=mean, std=std))

    def __len__(self):
        """Number of (image, mask) pairs."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return train (img,mask) or val (img,mask,orig_size)."""
        img_path, ann_path = self.samples[idx]
        image = _open_rgb_with_retry(img_path)
        mask_arr = _reduce_zero_label(_open_mask_array_with_retry(ann_path))
        if self.is_train:
            return self.train_tf(image, Image.fromarray(mask_arr))
        orig_h, orig_w = mask_arr.shape
        img_t, _ = self.val_tf(image, None)
        return img_t, torch.from_numpy(mask_arr).long(), (orig_h, orig_w)


class VOCDataset(Dataset):
    """PASCAL VOC 2012 segmentation (0=bg, 1-20=objects, 255=ignore)."""

    NUM_CLASSES = 21
    IGNORE_INDEX = 255

    def __init__(self, root, split="train", crop_size=(512, 512), pad_divisor=16,
                 train_scale=(2048, 512), val_scale=(2048, 512), mean=None, std=None,
                 **_unused):
        """Index VOC via ImageSets/Segmentation/{split}.txt + build transforms."""
        assert split in ("train", "val")
        self.is_train = split == "train"
        split_file = os.path.join(root, "ImageSets", "Segmentation", f"{split}.txt")
        with open(split_file) as f:
            names = [line.strip() for line in f if line.strip()]
        img_dir = os.path.join(root, "JPEGImages")
        ann_dir = os.path.join(root, "SegmentationClass")
        self.samples = []
        for name in names:
            ip = os.path.join(img_dir, name + ".jpg")
            ap = os.path.join(ann_dir, name + ".png")
            if os.path.exists(ip) and os.path.exists(ap):
                self.samples.append((ip, ap))
        self.train_tf = get_train_transforms(
            crop_size=crop_size, pad_divisor=pad_divisor, scale=train_scale, mean=mean, std=std)
        self.val_tf = get_val_img_transform(pad_divisor=pad_divisor, scale=val_scale,
                                            mean=mean, std=std)

    def __len__(self):
        """Number of (image, mask) pairs."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return train (img,mask) or val (img,mask,orig_size)."""
        img_path, ann_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask_arr = np.array(Image.open(ann_path)).astype(np.uint8)
        if self.is_train:
            return self.train_tf(image, Image.fromarray(mask_arr))
        orig_h, orig_w = mask_arr.shape
        img_t, _ = self.val_tf(image, None)
        return img_t, torch.from_numpy(mask_arr).long(), (orig_h, orig_w)


class CityscapesDataset(Dataset):
    """Cityscapes segmentation (19 train classes; labelIds remapped to trainIds)."""

    NUM_CLASSES = 19
    IGNORE_INDEX = 255

    def __init__(self, root, split="train", crop_size=(512, 512), tiling=False,
                 pad_divisor=16, train_scale=(2048, 512), val_scale=(2048, 512),
                 mean=None, std=None, **_unused):
        """Index leftImg8bit/gtFine + build transforms."""
        assert split in ("train", "val")
        self.is_train = split == "train"
        img_dir = os.path.join(root, "leftImg8bit", split)
        ann_dir = os.path.join(root, "gtFine", split)
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*", "*_leftImg8bit.png")))
        assert img_paths, f"No images found under {img_dir}"
        self.samples = []
        for ip in img_paths:
            city = os.path.basename(os.path.dirname(ip))
            stem = os.path.basename(ip).replace("_leftImg8bit.png", "")
            ap = os.path.join(ann_dir, city, stem + "_gtFine_labelIds.png")
            if os.path.exists(ap):
                self.samples.append((ip, ap))
        self.train_tf = get_train_transforms(
            crop_size=crop_size, pad_divisor=pad_divisor, scale=train_scale, mean=mean, std=std)
        self.val_tf = (get_val_img_transform_tiling(mean=mean, std=std) if tiling
                       else get_val_img_transform(pad_divisor=pad_divisor, scale=val_scale,
                                                  mean=mean, std=std))

    def __len__(self):
        """Number of (image, mask) pairs."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return train (img,mask) or val (img,mask,orig_size)."""
        img_path, ann_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask_arr = _CITYSCAPES_ID_TO_TRAINID[np.array(Image.open(ann_path), dtype=np.uint8)]
        if self.is_train:
            return self.train_tf(image, Image.fromarray(mask_arr))
        orig_h, orig_w = mask_arr.shape
        img_t, _ = self.val_tf(image, None)
        return img_t, torch.from_numpy(mask_arr).long(), (orig_h, orig_w)


class CachedSegDataset(Dataset):
    """Pre-extracted tiled features from disk (see ``caching.save_dense_cache``).

    Bypasses the backbone — the head trains directly on cached features. Train
    randomly crops a ``crop_feat × crop_feat`` patch (and the matching mask region);
    val returns the full feature map + original-size mask (batch_size=1).
    """

    def __init__(self, cache_dir, split="train", crop_feat=32, patch_size=16):
        """Index ``{cache_dir}/{split}/*.pt`` shards."""
        assert split in ("train", "val")
        self.is_train = split == "train"
        self.split_dir = os.path.join(cache_dir, split)
        self.files = sorted(glob.glob(os.path.join(self.split_dir, "*.pt")))
        self.crop_feat = crop_feat
        self.patch_size = patch_size
        assert self.files, f"No cached feature files found under {self.split_dir}"

    def __len__(self):
        """Number of cached shards."""
        return len(self.files)

    def __getitem__(self, idx):
        """Return train (feat_crop, mask_crop) or val (feat, mask, orig_size)."""
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        feat = data["features"].float().clone()
        mask = data["mask"].clone()
        orig = data["orig_size"]
        if not self.is_train:
            return feat, mask, orig

        _, H_feat, W_feat = feat.shape
        cf, ps = self.crop_feat, self.patch_size
        target_H, target_W = max(H_feat, cf), max(W_feat, cf)
        if H_feat < target_H or W_feat < target_W:
            feat = torch.nn.functional.pad(feat, (0, target_W - W_feat, 0, target_H - H_feat))
            H_feat, W_feat = target_H, target_W
        tgt_mh, tgt_mw = H_feat * ps, W_feat * ps
        if mask.shape[0] < tgt_mh or mask.shape[1] < tgt_mw:
            mask_pad = torch.full((tgt_mh, tgt_mw), 255, dtype=mask.dtype)
            mask_pad[:mask.shape[0], :mask.shape[1]] = mask
            mask = mask_pad
        fy = _random.randint(0, H_feat - cf)
        fx = _random.randint(0, W_feat - cf)
        feat_crop = feat[:, fy:fy + cf, fx:fx + cf].clone()
        mask_crop = mask[fy * ps:(fy + cf) * ps, fx * ps:(fx + cf) * ps].clone()
        return feat_crop, mask_crop


class DummySegDataset(Dataset):
    """In-memory random seg dataset for fast dev/tests (skips I/O + augmentation)."""

    def __init__(self, num_classes, num_samples=50, crop_size=(512, 512), split="train"):
        """Init shape + sample count."""
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.crop_h, self.crop_w = crop_size
        self.is_train = split == "train"

    def __len__(self):
        """Number of synthetic samples."""
        return self.num_samples

    def __getitem__(self, idx):
        """Return random (img, mask[, orig_size]) with ~5% ignore pixels."""
        img_t = torch.randn(3, self.crop_h, self.crop_w, dtype=torch.float32)
        mask_t = torch.randint(0, self.num_classes, (self.crop_h, self.crop_w))
        mask_t[torch.rand(self.crop_h, self.crop_w) < 0.05] = 255
        if self.is_train:
            return img_t, mask_t
        return img_t, mask_t, (self.crop_h, self.crop_w)


_SEG_DATASET_CLASSES = {
    "ade20k": ADE20KDataset, "voc": VOCDataset, "cityscapes": CityscapesDataset,
}


def val_collate(batch):
    """Collate for val batch_size=1: stack image, keep mask at original size."""
    images = torch.stack([b[0] for b in batch])
    orig_sizes = torch.tensor([b[2] for b in batch])     # (1, 2)
    return images, batch[0][1].unsqueeze(0), orig_sizes


def build_seg_loaders(
    dataset_name, dataset_root, *, batch_size=2, num_workers=4, distributed=False,
    tiling=False, feature_cache_dir=None, crop_size=512, pad_divisor=16,
    train_scale=(2048, 512), val_scale=(2048, 512), mean=None, std=None,
    max_val_samples=0, dummy=False,
):
    """Build ``(train_loader, val_loader, num_classes, ignore_index, train_sampler)``.

    Val uses batch_size=1 (variable-size whole-image eval). ``feature_cache_dir``
    bypasses the backbone via :class:`CachedSegDataset`; ``dummy`` uses random data.
    """
    cfg = SEG_DATASETS[dataset_name]
    num_classes, ignore_index = cfg["num_classes"], cfg["ignore_index"]

    if feature_cache_dir:
        train_ds = CachedSegDataset(feature_cache_dir, split="train", patch_size=pad_divisor)
        val_ds = CachedSegDataset(feature_cache_dir, split="val", patch_size=pad_divisor)
    elif dummy:
        train_ds = DummySegDataset(num_classes, num_samples=50, split="train")
        val_ds = DummySegDataset(num_classes, num_samples=20, split="val")
    else:
        assert dataset_root, f"segmentation root required for '{dataset_name}' unless dummy"
        cls = _SEG_DATASET_CLASSES[dataset_name]
        ct = (crop_size, crop_size)
        train_ds = cls(root=dataset_root, split="train", crop_size=ct, pad_divisor=pad_divisor,
                       train_scale=train_scale, val_scale=val_scale, mean=mean, std=std)
        val_ds = cls(root=dataset_root, split="val", crop_size=ct, tiling=tiling,
                     pad_divisor=pad_divisor, train_scale=train_scale, val_scale=val_scale,
                     mean=mean, std=std)

    if max_val_samples > 0:
        val_ds = Subset(val_ds, range(min(max_val_samples, len(val_ds))))

    pin = not feature_cache_dir   # pin_memory breaks on the non-resizable cached storage
    train_sampler = None
    if distributed and get_world_size() > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=get_world_size(),
                                           rank=get_global_rank(), shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=get_world_size(),
                                         rank=get_global_rank(), shuffle=False, drop_last=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=pin, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1, sampler=val_sampler,
                                num_workers=num_workers, pin_memory=pin, collate_fn=val_collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=num_workers, pin_memory=pin, collate_fn=val_collate)
    return train_loader, val_loader, num_classes, ignore_index, train_sampler
