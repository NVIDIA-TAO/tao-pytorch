# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ScanNet++ v2 dataset for NVPanoptix3Dv2 panoptic variant."""

import json
import os.path as osp
import random
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.utils import (
    center_crop_and_resize,
    normalize_resolution_arg,
    rgb2id,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.dataloader.panoptic.augmentations import (
    GeometrySafePhotometricAugmentation,
)

_MAX_GETITEM_RETRIES = 5


def orient_resolution_bucket(
    img_w: int, img_h: int, buckets: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """Pick the aspect-closest bucket and orient it to the image.

    Each bucket is treated as an unordered ``{long, short}`` aspect class
    (mirroring PanSt3R, which stores landscape buckets and transposes them
    for portrait inputs). We pick the bucket whose long/short ratio is
    closest to the image's, then orient it so the returned ``(H, W)`` matches
    the image's orientation -- landscape image -> wide target, portrait image
    -> tall target. This keeps the chosen frame aspect-matched to the source
    so the subsequent fit+pad in ``center_crop_and_resize`` adds only a
    small border (vs ~25-33% for a forced square), which both preserves the
    full field of view and minimises the padding the frozen VGGT backbone
    has to process.

    With a single bucket this just returns that bucket (oriented), so the
    single-resolution path is unchanged in aspect.
    """
    img_long = max(img_w, img_h)
    img_short = max(min(img_w, img_h), 1)
    img_ratio = img_long / img_short
    landscape = img_w >= img_h

    best = None
    best_d = float("inf")
    for (bh, bw) in buckets:
        b_long = max(bh, bw)
        b_short = max(min(bh, bw), 1)
        d = abs(b_long / b_short - img_ratio)
        if d < best_d:
            best_d = d
            best = (b_long, b_short)

    b_long, b_short = best
    # Orient: (H, W) = (short, long) for landscape, (long, short) for portrait.
    return (b_short, b_long) if landscape else (b_long, b_short)


class ScanNetppDataset(Dataset):
    """
    Multi-view ScanNet++ v2 dataset for NVPanoptix3Dv2 training.

    Reads preprocessed images, depth, panoptic maps and metadata from a
    single split-aware directory (``preprocessed_root``).

    Each sample returns a ``List[Dict]`` of ``num_views`` view dicts.

    Sampling strategy (scene-based, like PanSt3R):
      - Each epoch visits every scene ``pairs_per_scene`` times.
      - Each visit randomly samples a pair from the scene's pair pool.
      - During training, either anchor can become reference view 0 and the
        remaining views are shuffled (when ``randomize_view_order=True``).
      - Validation/test sampling is deterministic for every dataset index.
      - Epoch length = num_scenes * pairs_per_scene.
    """

    def __init__(
        self,
        preprocessed_root: str,
        split: str = "train",
        num_views: int = 5,
        resolution: Tuple[int, int] = (518, 518),
        pairs_per_scene: int = 50,
        randomize_view_order: bool = False,
        seed: int = 0,
        photometric_augmentation=None,
    ):
        super().__init__()
        self.num_views = num_views
        # ``resolution`` may be a single (H, W) tuple or a list of buckets.
        # All views of a given sample share one bucket (chosen per-sample by
        # the anchor view's aspect ratio); ``self.resolution`` keeps the
        # first bucket as the back-compat single-resolution fallback.
        self.res_buckets = normalize_resolution_arg(resolution)
        self.resolution = self.res_buckets[0]
        self.pairs_per_scene = pairs_per_scene
        self.split = split
        self.randomize_view_order = bool(randomize_view_order)
        self.seed = int(seed)
        self.photometric_augmentation = (
            GeometrySafePhotometricAugmentation.from_config(
                photometric_augmentation
            )
            if split == "train" else None
        )

        self.preprocessed_root = preprocessed_root
        if not osp.isfile(osp.join(self.preprocessed_root, "all_metadata.npz")):
            raise FileNotFoundError(
                f"Could not find all_metadata.npz under {preprocessed_root!r}."
            )

        # Class vocabulary from preprocessing
        cat_path = osp.join(self.preprocessed_root, "categories.json")
        with open(cat_path) as f:
            self.categories = json.load(f)
        self.classes = [c["name"] for c in self.categories]

        # Global metadata
        meta_path = osp.join(self.preprocessed_root, "all_metadata.npz")
        with np.load(meta_path, allow_pickle=True) as data:
            self.scenes = list(data["scenes"])
            self.sceneids = data["sceneids"].astype(int)
            self.images = list(data["images"])
            self.intrinsics = data["intrinsics"].astype(np.float32)
            self.trajectories = data["trajectories"].astype(np.float32)
            # pairs columns: [idx1, idx2, score]; we only need the indices
            self.pairs = data["pairs"][:, :2].astype(int)
            self.cls_sep = int(data["cls_sep"])

        # ── Validate image existence: drop images & pairs for missing files
        self.filter_missing_images()

        # Per-image pair lookup (for multi-view tuple selection)
        n_images = len(self.images)
        self.pairs_per_image: List[set] = [set() for _ in range(n_images)]
        for idx1, idx2 in self.pairs:
            self.pairs_per_image[idx1].add(idx2)
            self.pairs_per_image[idx2].add(idx1)

        # Per-scene pair index lists (for scene-based sampling)
        self._scene_pair_indices: List[List[int]] = [[] for _ in range(len(self.scenes))]
        for pair_idx, (i1, _) in enumerate(self.pairs):
            self._scene_pair_indices[self.sceneids[i1]].append(pair_idx)

    # Camera & path helpers

    def scene_dir(self, view_idx: int) -> str:
        """Return the scene directory containing ``view_idx``."""
        scene_id = self.scenes[self.sceneids[view_idx]]
        return osp.join(self.preprocessed_root, scene_id)

    def image_path(self, view_idx: int) -> str:
        """Return the on-disk image path for an image index."""
        stem = self.images[view_idx]
        return osp.join(self.scene_dir(view_idx), "images", stem + ".jpg")

    def depth_path(self, view_idx: int) -> Optional[str]:
        """Return the on-disk depth path for an image index, or None if missing.

        Both DSLR and iPhone share the same per-scene depth directory with a
        uint16 mm encoding.
        """
        stem = self.images[view_idx]
        path = osp.join(self.scene_dir(view_idx), "depth", stem + ".png")
        return path if osp.exists(path) else None

    def panoptic_path(self, view_idx: int) -> str:
        """Return the panoptic-label path for ``view_idx``."""
        scene_id = self.scenes[self.sceneids[view_idx]]
        stem = self.images[view_idx]
        return osp.join(self.preprocessed_root, scene_id, "panoptic", stem + ".png")

    def filter_missing_images(self):
        """Remove images whose files are absent and drop referencing pairs."""
        n_before = len(self.images)
        keep = np.array([osp.exists(self.image_path(i))
                         for i in range(n_before)])
        n_missing = int(n_before - keep.sum())
        if n_missing == 0:
            return

        old_to_new = np.full(n_before, -1, dtype=np.int64)
        new_idx = np.where(keep)[0]
        for new_i, old_i in enumerate(new_idx):
            old_to_new[old_i] = new_i

        self.images = [self.images[i] for i in new_idx]
        self.intrinsics = self.intrinsics[new_idx]
        self.trajectories = self.trajectories[new_idx]
        self.sceneids = self.sceneids[new_idx]

        new_pairs = []
        for i1, i2 in self.pairs:
            n1, n2 = old_to_new[i1], old_to_new[i2]
            if n1 >= 0 and n2 >= 0:
                new_pairs.append([n1, n2])
        self.pairs = np.array(new_pairs, dtype=np.int64) if new_pairs \
            else np.zeros((0, 2), dtype=np.int64)

    def __len__(self) -> int:
        """Return the number of multi-view samples the split exposes."""
        return len(self.scenes) * self.pairs_per_scene

    # Multi-view tuple selection

    def select_views(self, idx1: int, idx2: int, rng=None) -> List[int]:
        """Pick ``num_views`` image indices around the anchor pair.

        ``rng`` may be the module-level :mod:`random` generator for training
        or an index-seeded :class:`random.Random` instance for deterministic
        validation/test sampling.
        """
        rng = random if rng is None else rng
        selected = [idx1, idx2]
        selected_set = set(selected)

        # Sort before shuffling so an index-seeded RNG produces the same
        # order across processes and Python hash seeds.
        candidates = sorted(self.pairs_per_image[idx1] |
                            self.pairs_per_image[idx2])
        rng.shuffle(candidates)

        for c in candidates:
            if len(selected) >= self.num_views:
                break
            if c not in selected_set:
                selected.append(c)
                selected_set.add(c)

        while len(selected) < self.num_views:
            selected.append(selected[len(selected) % len(selected)])

        return selected[:self.num_views]

    def augment_training_view_order(self, view_indices: List[int], rng=None) -> List[int]:
        """Randomise reference/tail order without choosing a weak reference.

        The first two entries are the central, directly-overlapping anchor
        pair. Pick either anchor as reference view 0, then shuffle every
        remaining view. Auxiliary neighbours are never promoted to reference
        because each is only guaranteed to overlap at least one anchor.
        """
        ordered = list(view_indices)
        if (
            self.split != "train" or
            not self.randomize_view_order or
            len(ordered) < 2
        ):
            return ordered

        rng = random if rng is None else rng
        if rng.random() < 0.5:
            ordered[0], ordered[1] = ordered[1], ordered[0]
        tail = ordered[1:]
        rng.shuffle(tail)
        return [ordered[0], *tail]

    def sampling_rng(self, idx: int):
        """Return stochastic train RNG or a stable per-index eval RNG."""
        if self.split == "train":
            return random
        return random.Random(self.seed + int(idx))

    # View loading

    def load_view(
        self,
        view_idx: int,
        target_res: Optional[Tuple[int, int]] = None,
        photometric_recipe=None,
    ) -> Dict:
        """Load and spatially align all modalities for one view."""
        scene_id = self.scenes[self.sceneids[view_idx]]
        stem = self.images[view_idx]
        K = self.intrinsics[view_idx].copy()
        E = self.trajectories[view_idx].copy()

        tgt_h, tgt_w = target_res if target_res is not None else self.resolution

        # RGB
        img = Image.open(self.image_path(view_idx)).convert("RGB")
        if (
            self.photometric_augmentation is not None and
            photometric_recipe is not None
        ):
            # Applied before resize/padding so the geometry transform and K
            # update remain identical, while synthetic white padding stays
            # exactly white.
            img = self.photometric_augmentation.apply(img, photometric_recipe)

        # ScanNet++ depth PNGs are uint16 millimetres; convert to metres.
        depth_path = self.depth_path(view_idx)
        if depth_path is not None:
            depth = np.array(Image.open(depth_path), dtype=np.float32)
            depth /= 1000.0
            depth[~np.isfinite(depth)] = 0.0
        else:
            W, H = img.size
            depth = np.zeros((H, W), dtype=np.float32)

        # Panoptic map
        pan_path = self.panoptic_path(view_idx)
        if osp.exists(pan_path):
            pan_rgb = np.array(Image.open(pan_path))
            pan_id = rgb2id(pan_rgb)
        else:
            pan_id = None

        # PP-centred crop + uniform resize
        img, K, depth, pan_id = center_crop_and_resize(
            img, K, depth, pan_id, (tgt_h, tgt_w))

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        if pan_id is not None:
            inst_id = pan_id // self.cls_sep
            cls_id = pan_id % self.cls_sep
        else:
            inst_id = np.zeros((tgt_h, tgt_w), dtype=np.int32)
            cls_id = np.zeros((tgt_h, tgt_w), dtype=np.int32)

        view = {
            "img": img_tensor,
            "true_shape": torch.tensor([tgt_h, tgt_w], dtype=torch.int32),
            "depthmap": torch.from_numpy(depth),
            "camera_intrinsics": torch.from_numpy(K),
            "camera_pose": torch.from_numpy(E),
            "intrinsics": torch.from_numpy(K),
            "extrinsics": torch.from_numpy(E),
            "pan_inst_id": torch.from_numpy(inst_id.astype(np.int64)),
            "pan_cls_id": torch.from_numpy(cls_id.astype(np.int64)),
            "dataset": "ScanNet++",
            "label": f"{scene_id}_{stem}",
        }
        return view

    def __getitem__(self, idx: int) -> List[Dict]:
        """Return one multi-view sample as a list of per-view dicts."""
        # Training retains bounded retries for transient I/O errors. Eval is
        # deliberately fail-fast: silently replacing a validation sample
        # makes metrics stochastic and can hide corrupt manifests.
        max_attempts = _MAX_GETITEM_RETRIES if self.split == "train" else 1
        for attempt in range(max_attempts):
            sampling_rng = self.sampling_rng(idx)
            scene_idx = idx % len(self.scenes)

            try:
                pair_list = self._scene_pair_indices[scene_idx]
                if not pair_list:
                    fallback = [si for si in range(len(self.scenes))
                                if self._scene_pair_indices[si]]
                    if not fallback:
                        raise RuntimeError("All scenes have 0 pairs")
                    scene_idx = sampling_rng.choice(fallback)
                    pair_list = self._scene_pair_indices[scene_idx]

                pair_idx = sampling_rng.choice(pair_list)
                idx1, idx2 = self.pairs[pair_idx]

                view_indices = self.select_views(idx1, idx2, rng=sampling_rng)
                view_indices = self.augment_training_view_order(
                    view_indices, rng=sampling_rng,
                )

                # One resolution bucket for the whole sample, chosen from the
                # anchor view's aspect ratio (header-only read, no decode), so
                # the stacked multi-view tensor is consistent. Orient the bucket
                # even when there is only one, or portrait sources get forced
                # through a landscape target.
                with Image.open(self.image_path(view_indices[0])) as _im:
                    aw, ah = _im.size
                target_res = orient_resolution_bucket(aw, ah, self.res_buckets)

                photometric_recipe = (
                    self.photometric_augmentation.sample(sampling_rng)
                    if self.photometric_augmentation is not None else None
                )
                views = [
                    self.load_view(
                        view_index,
                        target_res=target_res,
                        photometric_recipe=photometric_recipe,
                    )
                    for view_index in view_indices
                ]

                return views
            except Exception:
                if attempt == max_attempts - 1:
                    traceback.print_exc(file=sys.stderr)
                    raise
                idx = random.randint(0, len(self) - 1)

        # Unreachable: the final attempt either returns or raises above. Kept so
        # every path out of the method is explicit.
        raise RuntimeError(
            f"__getitem__ exhausted {max_attempts} attempts for idx={idx}"
        )


class ScanNetppCollator:
    """Stack a ScanNet++ batch and attach its fixed class vocabulary."""

    def __init__(self, classes: List[str]):
        self.classes = list(classes)

    def __call__(self, batch_list):
        """Return a view-major batch with the native ScanNet++ vocabulary."""
        batch_size = len(batch_list)
        num_views = len(batch_list[0])
        collated = []

        for view_index in range(num_views):
            view_dicts = [batch_list[index][view_index] for index in range(batch_size)]
            out = {}
            for key in view_dicts[0]:
                values = [view[key] for view in view_dicts]
                if isinstance(values[0], torch.Tensor):
                    out[key] = torch.stack(values, dim=0)
                elif isinstance(values[0], str):
                    out[key] = values
                else:
                    out[key] = values
            collated.append(out)

        collated[0]["vocab"] = list(self.classes)
        return collated
