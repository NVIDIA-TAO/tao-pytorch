# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Labeled classification datasets for KNN / retrieval / linear-probe eval.

Two backends behind one uniform batch contract::

    {"image": FloatTensor[B, 3, H, W], "label": LongTensor[B], "path": List[str]}

- ``image_folder``: class-per-subdir folders (torchvision ``ImageFolder`` layout)
  via ``MetricLearnImageFolder`` (reused from ``cv/ml_recog``). Default for
  ImageNet (``train/`` + ``val/``) and generic retrieval folders.
- ``webdataset``: ``.tar`` shards streamed sequentially — far better small-file
  throughput over CIFS for ImageNet-1k KNN.

Distributed-aware: under ``torch.distributed`` each rank gets a
``DistributedSampler`` shard. :func:`build_classification_loader` returns the
*unpadded* dataset length so the KNN all-gather can strip sampler padding
(``DistributedSampler`` pads to a multiple of world size).

Transform parity: by default this matches the c-radiov4 KNN protocol —
``Resize(resize) -> CenterCrop(crop) -> ToTensor`` with BICUBIC interpolation.
Normalization is **optional**: pass ``mean``/``std`` to normalize in the dataset
(NV-DINOv2 path, whose adapter does not normalize), or leave them ``None`` for
backbones that normalize internally (RADIO via its ``input_conditioner``).
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import ImageFile
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torchvision.transforms as T

from nvidia_tao_pytorch.core.distributed.comm import get_global_rank, get_world_size
from nvidia_tao_pytorch.cv.ml_recog.dataloader.datasets.image_datasets import (
    MetricLearnImageFolder,
)

# Tolerate truncated JPEGs (ImageNet train has a few) instead of crashing a
# multi-hour extraction — same policy as c-radiov4 eval_cls.py and the mae /
# classification_pyt / radio dataloaders.
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_INTERP = {
    "bicubic": T.InterpolationMode.BICUBIC,
    "bilinear": T.InterpolationMode.BILINEAR,
    "nearest": T.InterpolationMode.NEAREST,
}


@dataclass
class ClassificationLoader:
    """A built classification loader plus the metadata KNN/retrieval need.

    Attributes:
        loader: The ``DataLoader`` yielding ``{image, label, path}`` batches.
        num_classes: Number of label classes.
        total_size: Unpadded dataset length (strip ``DistributedSampler`` padding
            after the cross-rank all-gather).
    """

    loader: DataLoader
    num_classes: int
    total_size: int


def build_eval_transform(
    crop: int = 224,
    resize: Optional[int] = None,
    interpolation: str = "bicubic",
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
) -> T.Compose:
    """Standard eval transform: ``Resize -> CenterCrop -> ToTensor [-> Normalize]``.

    Args:
        crop: center-crop (and model input) size.
        resize: shorter-side resize target; defaults to ``crop`` (c-radiov4 KNN
            parity). For the classic 0.875-ratio protocol pass ``int(crop*256/224)``.
        interpolation: ``"bicubic"`` (default), ``"bilinear"`` or ``"nearest"``.
        mean, std: normalization stats. When **both** are ``None`` no
            normalization is applied (the backbone normalizes internally).
    """
    ops = [
        T.Resize(resize or crop, interpolation=_INTERP[interpolation]),
        T.CenterCrop(crop),
        T.ToTensor(),
    ]
    if mean is not None and std is not None:
        ops.append(T.Normalize(mean=mean, std=std))
    return T.Compose(ops)


def _collate(batch) -> dict:
    """Collate ``(image, label, path)`` tuples into the uniform batch contract."""
    images = torch.stack([b[0] for b in batch], 0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    paths = [b[2] for b in batch]
    return {"image": images, "label": labels, "path": paths}


class _LabeledFolder(MetricLearnImageFolder):
    """``ImageFolder`` variant that also returns the sample path."""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """Return (transformed_image, class_index, path)."""
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


def _maybe_distributed_sampler(dataset, distributed: bool):
    """Build a non-shuffling ``DistributedSampler`` when distributed, else ``None``."""
    if not distributed or get_world_size() <= 1:
        return None
    return DistributedSampler(
        dataset, num_replicas=get_world_size(), rank=get_global_rank(),
        shuffle=False, drop_last=False,
    )


def _build_image_folder_loader(
    root: str, transform, batch_size: int, num_workers: int, distributed: bool,
    max_samples: Optional[int] = None,
) -> ClassificationLoader:
    dataset = _LabeledFolder(root=root, transform=transform)
    num_classes = len(dataset.classes)
    if max_samples is not None and max_samples < len(dataset):
        # Deterministic subset (seed 0) — for quick sanity checks; mirrors
        # c-radiov4 eval_knn --max-train-samples.
        rng = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(dataset), generator=rng)[:max_samples].tolist()
        dataset = Subset(dataset, indices)
    total_size = len(dataset)
    logger.info("ImageFolder backend: %d images, %d classes from %s",
                total_size, num_classes, root)
    sampler = _maybe_distributed_sampler(dataset, distributed)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False, collate_fn=_collate,
    )
    return ClassificationLoader(loader, num_classes, total_size)


def _build_webdataset_loader(
    shards: str, transform, batch_size: int, num_workers: int,
    num_classes: int, total_size: int, label_key: str = "cls",
) -> ClassificationLoader:
    """Thin WDS reader for ``.tar`` shards.

    WebDataset is iterable (no random sampler / length), so ``num_classes`` and
    ``total_size`` must be supplied by the caller (ImageNet-1k: 1000 / 1281167
    train / 50000 val). ``label_key`` is the per-sample label field — RADIO's
    shards use ``json``'s ``id``; classic WDS ImageNet uses ``cls``
    (see DESIGN open item #6 — verify the on-disk shard layout on the devbox).
    """
    try:
        import webdataset as wds
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "webdataset is required for dataset_type='webdataset'. "
            "It is available in the tao_pt container."
        ) from e

    def _map(sample):
        img = transform(sample["jpg"] if "jpg" in sample else sample["png"])
        label = int(sample[label_key])
        key = sample.get("__key__", "")
        return img, label, key

    dataset = wds.WebDataset(shards, shardshuffle=False).decode("pil").map(_map)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, collate_fn=_collate,
    )
    return ClassificationLoader(loader, num_classes, total_size)


def build_classification_loader(
    dataset_type: str,
    root: str,
    *,
    batch_size: int = 128,
    num_workers: int = 8,
    crop: int = 224,
    resize: Optional[int] = None,
    interpolation: str = "bicubic",
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    distributed: bool = False,
    max_samples: Optional[int] = None,
    num_classes: int = 1000,
    total_size: int = 0,
    label_key: str = "cls",
) -> ClassificationLoader:
    """Build a labeled eval loader for the requested backend.

    Args:
        dataset_type: ``"image_folder"`` or ``"webdataset"``.
        root: folder root (image_folder) or shard glob/braced pattern (webdataset).
        batch_size, num_workers: standard loader params.
        crop, resize, interpolation, mean, std: see :func:`build_eval_transform`.
        distributed: shard across ranks with a ``DistributedSampler``
            (image_folder only).
        max_samples: optional deterministic subset cap (image_folder only).
        num_classes, total_size: required for ``webdataset`` (iterable, no length).
        label_key: per-sample label field for ``webdataset``.

    Returns:
        A :class:`ClassificationLoader` (loader + num_classes + unpadded length).
    """
    transform = build_eval_transform(crop, resize, interpolation, mean, std)
    if dataset_type == "image_folder":
        return _build_image_folder_loader(
            root, transform, batch_size, num_workers, distributed, max_samples)
    if dataset_type == "webdataset":
        return _build_webdataset_loader(
            root, transform, batch_size, num_workers, num_classes, total_size, label_key)
    raise ValueError(
        f"Unknown dataset_type '{dataset_type}'. Use 'image_folder' or 'webdataset'."
    )
