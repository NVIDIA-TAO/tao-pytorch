# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor for NVPanoptix-3D model datasets."""

import os
import numpy as np
import pyexr
from typing import Dict, Any, List, Optional, Tuple, Union

from PIL import Image, ImageOps
from fvcore.transforms.transform import Transform, CropTransform, HFlipTransform, VFlipTransform

import torch
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.augmentations import (
    Compose, ResizeShortestEdge, RandomCrop,
    Absolute, ColorAugSSDTransform, RandomFlip,
    ToBinaryMask, ToTDF, ResizeTrilinear, ToPILImage,
    Resize, ToTensor, ToDepthMap, ResizeMax, ModelInputResize
)

from nvidia_tao_pytorch.core.tlt_logging import logging


class DatasetConstants:
    """
    Constants used across dataset classes for 3D panoptic reconstruction.

    This class centralizes all shared constants including default image sizes,
    depth ranges, grid dimensions, category definitions, camera intrinsics,
    and padding values. These constants ensure consistency across different
    dataset implementations and preprocessing pipelines.
    """

    DEFAULT_IMG_SIZE = (240, 320)
    DEFAULT_DEPTH_RANGE = (0.4, 6.0)
    DEFAULT_GRID_DIMS = [256, 256, 256]
    DEFAULT_VOXEL_SIZE = 0.03
    MATTERPORT_DEPTH_DIVISOR = 4000
    DEFAULT_TRUNCATION = 12
    PADDING_RGB_VALUE = 128
    PADDING_DEPTH_VALUE = 0
    PADDING_ROOM_MASK_VALUE = 0
    PADDING_INST_SEG_VALUE = -1

    STUFF_CLASSES = [10, 11]

    INTRINSIC = torch.from_numpy(
        np.array(
            [
                [277.1281435, 0., 159.5, 0.],
                [0., 277.1281435, 119.5, 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]
            ]
        ).reshape((4, 4))
    )

    CATEGORIES = [
        {"color": (220, 20, 60), "isthing": 1, "id": 1, "trainId": 1, "name": "cabinet"},
        {"color": (255, 0, 0), "isthing": 1, "id": 2, "trainId": 2, "name": "bed"},
        {"color": (0, 0, 142), "isthing": 1, "id": 3, "trainId": 3, "name": "chair"},
        {"color": (0, 0, 70), "isthing": 1, "id": 4, "trainId": 4, "name": "sofa"},
        {"color": (0, 60, 100), "isthing": 1, "id": 5, "trainId": 5, "name": "table"},
        {"color": (0, 80, 100), "isthing": 1, "id": 6, "trainId": 6, "name": "desk"},
        {"color": (0, 0, 230), "isthing": 1, "id": 7, "trainId": 7, "name": "dresser"},
        {"color": (119, 11, 32), "isthing": 1, "id": 8, "trainId": 8, "name": "lamp"},
        {"color": (190, 50, 60), "isthing": 1, "id": 9, "trainId": 9, "name": "other"},
        {"color": (102, 102, 156), "isthing": 0, "id": 10, "trainId": 10, "name": "wall"},
        {"color": (128, 64, 128), "isthing": 0, "id": 11, "trainId": 11, "name": "floor"},
        {"color": (70, 70, 70), "isthing": 0, "id": 12, "trainId": 12, "name": "ceiling"},
    ]

    # Class ID to RGB color mapping from CATEGORIES in preprocessor.py
    SEMANTIC_CLASS_COLORS = {
        0: (0, 0, 0),           # background/void
        1: (220, 20, 60),       # cabinet
        2: (255, 0, 0),         # bed
        3: (0, 0, 142),         # chair
        4: (0, 0, 70),          # sofa
        5: (0, 60, 100),        # table
        6: (0, 80, 100),        # desk
        7: (0, 0, 230),         # dresser
        8: (119, 11, 32),       # lamp
        9: (190, 50, 60),       # other
        10: (102, 102, 156),    # wall
        11: (128, 64, 128),     # floor
        12: (70, 70, 70),       # ceiling
    }


class BasePreprocessor:
    """
    Base preprocessor for loading and transforming multi-modal 3D scene data.

    This class handles all preprocessing operations including data loading,
    augmentation transform configuration, 2D/3D data transformation, instance
    mask preparation, and coordinate system handling. It provides a unified
    interface for processing RGB images, depth maps, segmentations, and
    volumetric data across different dataset formats.
    """

    def __init__(
        self,
        cfg,
        is_flip: bool = False,
        is_matterport: bool = False,
        categories: List[Dict] = None,
        stuff_classes: List[int] = None,
        ignore_label: int = 255,
        min_instance_pixels: int = 1,
        resize_hw: Tuple[int, int] = None,
        depth_resize_hw: Tuple[int, int] = None,
        size_divisibility: int = 14,
        depth_bound: bool = False,
        iso_value: float = 1.0,
        truncation_range: List[float] = None,
        downsample_factor: int = 1,
        enable_mp_occ: bool = True,
        frustum_mask: torch.Tensor = None,
        intrinsic: torch.Tensor = None,
        **kwargs,
    ):
        """
        Constructor for BasePreprocessor.

        Args:
            cfg: Configuration object containing augmentation parameters.
            is_flip (bool): Whether to flip volumes along Y and Z axes for
                coordinate system alignment. Defaults to False.
            is_matterport (bool): Whether this is Matterport3D dataset, which
                requires instance ID rearrangement. Defaults to False.
            categories (List[Dict]): List of category definitions with id, name,
                color, and isthing flag. Defaults to None.
            stuff_classes (List[int]): List of stuff class IDs (non-countable
                categories like walls, floor). Defaults to None.
            ignore_label (int): Label ID to ignore in segmentation loss and
                evaluation. Defaults to 255.
            min_instance_pixels (int): Minimum pixel count for valid instances.
                Smaller instances are filtered out. Defaults to 1.
            resize_hw (Tuple[int, int]): Target (height, width) for 2D data.
                Defaults to None.
            depth_resize_hw (Tuple[int, int]): Target (height, width) for depth
                maps. Defaults to None.
            size_divisibility (int): Pad images to be divisible by this value.
                Defaults to 14.
            depth_bound (bool): Whether to clip depth values after augmentation.
                Defaults to False.
            iso_value (float): Isovalue threshold for extracting surfaces from
                signed distance fields. Defaults to 1.0.
            truncation_range (List[float]): [min, max] truncation distances for
                TSDFs. Defaults to None ([3, 12]).
            downsample_factor (int): Factor for downsampling 3D volumes during
                evaluation. Defaults to 1.
            enable_mp_occ (bool): Whether to load multiplane occupancy data.
                Defaults to True.
            frustum_mask (torch.Tensor): Precomputed viewing frustum mask.
                Defaults to None.
            intrinsic (torch.Tensor): Camera intrinsic matrix (4x4).
                Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.cfg = cfg
        self.is_flip = is_flip
        self.is_matterport = is_matterport
        self.categories = categories or []
        self.stuff_classes = stuff_classes or []
        self.ignore_label = ignore_label
        self.min_instance_pixels = min_instance_pixels
        self.resize_hw = resize_hw
        self.depth_resize_hw = depth_resize_hw
        self.size_divisibility = size_divisibility
        self._model_input_resize = ModelInputResize(size_divisibility)
        self.depth_bound = depth_bound
        self.iso_value = iso_value
        self.truncation_range = truncation_range or [3.0, 12.0]
        self.downsample_factor = downsample_factor

        # Set frustum_mask & intrinsic
        self.frustum_mask = frustum_mask
        self.intrinsic = intrinsic

        # Initialize transforms
        self.occ_truncation_lvl = cfg.occ_truncation_lvl
        self.img_format = cfg.img_format
        self.enable_mp_occ = enable_mp_occ
        self.prepare_transforms()

    def get_frustum_mask(self, **kwargs) -> torch.Tensor:
        """Return the per-sample 3D viewing frustum mask (if available).

        The frustum mask is a boolean voxel grid used to mark which voxels are
        considered valid/visible for a given sample (e.g., within the camera
        frustum and/or observed region). For datasets with a global/precomputed
        mask, this accessor simply returns the mask provided at construction.

        Args:
            **kwargs: Unused. Accepted for API compatibility with subclasses
                (e.g., Matterport3D computes/loads the mask from per-sample files).

        Returns:
            torch.Tensor: A boolean tensor (typically shaped like
                `DatasetConstants.DEFAULT_GRID_DIMS`) or None if not set.
        """
        return self.frustum_mask

    def get_intrinsic(self, **kwargs) -> torch.Tensor:
        """Get intrinsic.

        IMPORTANT:
        The stored intrinsics are typically calibrated for a *reference* image size
        (e.g., Front3D uses `DatasetConstants.DEFAULT_IMG_SIZE=(240,320)`).
        When the input image is resized for training/inference, intrinsics must be
        scaled accordingly to keep depth->3D projection consistent.
        """
        intr = self.intrinsic
        if intr is None:
            return intr

        # If caller provides a target image size, scale intrinsics to match it.
        # Expect kwargs width/height to be the (un-padded) image size in pixels.
        try:
            to_w = int(kwargs.get("width"))
            to_h = int(kwargs.get("height"))
        except Exception:
            return intr
        if to_w <= 0 or to_h <= 0:
            return intr

        # Determine the reference (calibration) size for these intrinsics.
        # - Front3D/synthetic/predict use the fixed DatasetConstants intrinsics.
        # - Matterport provides per-image intrinsics; use the original image size if provided.
        # - NYUv2 uses a fixed intrinsic calibrated at 640x480.
        name = str(getattr(getattr(self.cfg, "dataset", None), "name", "")).strip().lower()
        if name in {"matterport"}:
            try:
                from_w = int(kwargs.get("orig_width", to_w))
                from_h = int(kwargs.get("orig_height", to_h))
            except Exception:
                from_w, from_h = to_w, to_h
        else:
            # Front3D default (and fallback).
            from_h0 = int(DatasetConstants.DEFAULT_IMG_SIZE[0])
            from_w0 = int(DatasetConstants.DEFAULT_IMG_SIZE[1])
            from_w, from_h = from_w0, from_h0

        if from_w == to_w and from_h == to_h:
            return intr
        if from_w <= 1 or from_h <= 1 or to_w <= 1 or to_h <= 1:
            return intr

        return self.scale_intrinsic(intr, from_wh=(from_w, from_h), to_wh=(to_w, to_h))

    @staticmethod
    def update_intrinsic_for_2d_transform(
        intrinsic: Optional[torch.Tensor],
        tfm: Transform,
        *,
        prev_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        """Update camera intrinsics for 2D geometric transforms (crop/flip/resize).

        `update_intrinsic_for_2d_transform` and `scale_intrinsic` are **not** redundant:
        - `scale_intrinsic`: corrects intrinsics for **resizing** (focal length + principal point scaling)
        - this function: corrects intrinsics for **crop/flip** (principal point shift/reflection),
          and also supports our `ResizeShortestEdge` so call sites can stay uniform.
        """
        if intrinsic is None:
            return None
        if intrinsic.ndim != 2:
            # We only expect per-sample intrinsics here.
            return intrinsic

        prev_height, prev_width = int(prev_hw[0]), int(prev_hw[1])
        if prev_height <= 0 or prev_width <= 0:
            return intrinsic

        # Our augmentation wrappers (RandomCrop/RandomFlip) store the realized fvcore transform
        # in `.tfm`. If it's missing, fall back to the transform itself (e.g., ResizeShortestEdge).
        inner: Transform = getattr(tfm, "tfm", None) or tfm
        out = intrinsic

        # Resize: scale focal length + principal point. (Crop/flip must NOT use scaling.)
        if isinstance(inner, ResizeShortestEdge):
            try:
                # ResizeShortestEdge stores new_size as (W, H).
                new_w, new_h = int(inner.new_size[0]), int(inner.new_size[1])
            except Exception:
                new_h, new_w = -1, -1
            if new_w > 0 and new_h > 0 and (new_w, new_h) != (prev_width, prev_height):
                out = BasePreprocessor.scale_intrinsic(
                    intrinsic, from_wh=(prev_width, prev_height), to_wh=(new_w, new_h)
                )

        # Crop/flip: adjust principal point only.
        elif isinstance(inner, CropTransform):
            x0 = int(getattr(inner, "x0", 0))
            y0 = int(getattr(inner, "y0", 0))
            out = intrinsic.clone()
            out[0, 2] -= float(x0)
            out[1, 2] -= float(y0)

        elif isinstance(inner, HFlipTransform):
            out = intrinsic.clone()
            out[0, 2] = float(prev_width - 1) - out[0, 2]

        elif isinstance(inner, VFlipTransform):
            out = intrinsic.clone()
            out[1, 2] = float(prev_height - 1) - out[1, 2]

        return out

    @staticmethod
    def scale_intrinsic(
        intrinsic: torch.Tensor,
        *,
        from_wh: Tuple[int, int],
        to_wh: Tuple[int, int],
    ) -> torch.Tensor:
        """Scale a 3x3/4x4 intrinsic (optionally batched) from one image size to another.

        Args:
            intrinsic: Tensor [...,3,3] or [...,4,4]
            from_wh: (W,H) the calibration image size for `intrinsic`
            to_wh:   (W,H) the target image size we will project into
        """
        fw, fh = int(from_wh[0]), int(from_wh[1])
        tw, th = int(to_wh[0]), int(to_wh[1])
        width_scale = float(tw) / float(fw)
        height_scale = float(th) / float(fh)
        # Preserve pixel-center convention under resizing.
        width_offset_scale = float(tw - 1) / float(max(fw - 1, 1))
        height_offset_scale = float(th - 1) / float(max(fh - 1, 1))

        intr = intrinsic.clone()
        if intr.ndim == 2:
            # [3,3] or [4,4]
            intr[0, 0] *= width_scale
            intr[1, 1] *= height_scale
            intr[0, 2] *= width_offset_scale
            intr[1, 2] *= height_offset_scale
        else:
            # [...,3,3] or [...,4,4]
            intr[..., 0, 0] *= width_scale
            intr[..., 1, 1] *= height_scale
            intr[..., 0, 2] *= width_offset_scale
            intr[..., 1, 2] *= height_offset_scale
        return intr

    def prepare_transforms(self):
        """
        Initialize all data transformation pipelines.

        This method sets up separate transform pipelines for different data
        modalities and modes: room masks, depth maps, 2D training/testing
        augmentations, and 3D volumetric transforms. Called during initialization.
        """
        self.room_mask_transforms = self.get_room_mask_transforms()
        self.depth_transforms = self.get_depth_transforms()
        self.train2d_transforms = self.get_train_transforms_2d(self.resize_hw)
        self.test2d_transforms = self.get_test_transforms_2d(self.resize_hw)
        self.vol_transforms = self.get_3d_transforms()

    def get_3d_transforms(self):
        """
        Configure all 3D volumetric data transformations.

        Creates transform pipelines for geometry (TSDF), occupancy grids at
        multiple resolutions (64³, 128³, 256³), semantic segmentation volumes,
        weighting volumes, and multiplane occupancy. Each transform chain
        handles tensor conversion, resizing, truncation, and binarization as needed.

        Returns:
            dict: Dictionary mapping transform names to Compose objects containing
                transform sequences for different 3D data types and resolutions.
        """
        transforms = {}

        # transforms for geometry
        transforms["geometry"] = Compose(
            [
                ToTensor(dtype=torch.float),
                ToTDF(truncation=DatasetConstants.DEFAULT_TRUNCATION),
            ]
        )
        transforms["geometry_truncate"] = Compose(
            [
                ToTensor(dtype=torch.float),
                ToTDF(truncation=self.truncation_range[1]),
            ]
        )
        transforms["geometry_occ"] = transforms["geometry_truncate"]

        # transforms for occupancy
        transforms["occupancy_64"] = Compose(
            [
                ResizeTrilinear(0.25),
                ToBinaryMask(self.occ_truncation_lvl[0]),
                ToTensor(dtype=torch.float),
            ]
        )
        transforms["occupancy_128"] = Compose(
            [
                ResizeTrilinear(0.5),
                ToBinaryMask(self.occ_truncation_lvl[1]),
                ToTensor(dtype=torch.float),
            ]
        )
        transforms["occupancy_256"] = Compose(
            [
                ToBinaryMask(self.truncation_range[1]),
                ToTensor(dtype=torch.float),
            ]
        )

        # transform for weight volume
        transforms["weighting3d"] = Compose(
            [
                ToTensor(dtype=torch.float),
            ]
        )
        transforms["weighting3d_64"] = Compose([ResizeTrilinear(0.25)])
        transforms["weighting3d_128"] = Compose([ResizeTrilinear(0.5)])

        # transform for semantic volume
        transforms["semantic3d"] = Compose(
            [
                ToTensor(dtype=torch.long),
            ]
        )
        transforms["segmentation3d_64"] = Compose([ResizeMax(8, 4, 2)])
        transforms["segmentation3d_128"] = Compose([ResizeMax(4, 2, 1)])

        transforms["mp_occupancy"] = Compose(
            [
                Absolute(),
                ToBinaryMask(self.truncation_range[1]),
                ToTensor(dtype=torch.float),
            ]
        )

        return transforms

    def get_depth_transforms(self):
        """Build the depth preprocessing transform pipeline.

        Returns:
            Compose: A composed transform callable to be applied to depth arrays.

        Notes:
            This method assumes `self.depth_resize_hw` is set to (H, W).
        """
        return Compose(
            [
                ToPILImage(),
                Resize((self.depth_resize_hw[1], self.depth_resize_hw[0])),
                ToTensor(),
                ToDepthMap(None),
            ]
        )

    def get_room_mask_transforms(self):
        """Build the room-mask preprocessing transform pipeline.

        Room masks are used by some datasets (e.g., Matterport3D) to indicate
        valid/inside-room pixels. The transform resizes the mask to
        `self.depth_resize_hw` using nearest-neighbor semantics (via PIL),
        and converts it to a tensor.

        Returns:
            Compose: A composed transform callable to be applied to room-mask inputs.

        Notes:
            This method assumes `self.depth_resize_hw` is set to (H, W).
        """
        return Compose(
            [
                ToPILImage(),
                Resize((self.depth_resize_hw[1], self.depth_resize_hw[0])),
                ToTensor(),
            ]
        )

    def get_common_size_transform(self):
        """Build the shared spatial resize transform used across 2D modalities.

        The returned transform enforces a consistent resize policy (shortest-edge
        resize with max size cap) so RGB, semantic labels, instance labels, and
        other 2D maps can be transformed with identical geometry.

        Returns:
            Compose: A composed transform with a single `ResizeShortestEdge`.
        """
        return Compose(
            [
                ResizeShortestEdge(
                    orig_size=self.resize_hw,
                    short_edge_length=self.cfg.augmentation.train_min_size,
                    max_size=self.cfg.augmentation.train_max_size,
                )
            ]
        )

    def get_transforms_2d(self, mode: str = "test") -> List[Transform]:
        """Return the configured 2D augmentation pipeline for a given mode.

        Args:
            mode (str): One of {"train", "val", "test"}.
                - "train": returns `self.train2d_transforms` (may include random augments)
                - "val"/"test": returns `self.test2d_transforms` (deterministic)

        Returns:
            List[Transform]: Ordered list of transforms to apply to 2D inputs.

        Raises:
            ValueError: If `mode` is not one of the supported values.
        """
        if mode in ["test", "val"]:
            return self.test2d_transforms
        elif mode == "train":
            return self.train2d_transforms
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_train_transforms_2d(self, orig_size: Tuple[int, int]) -> List:
        """
        Configure 2D data augmentation pipeline for training.

        Builds a sequence of transforms including resize, random crop, color
        augmentation, and random flips based on configuration settings. Each
        augmentation can be individually enabled/disabled via config.

        Args:
            orig_size (Tuple[int, int]): Original image size as (height, width).

        Returns:
            List[Transform]: Ordered list of augmentation transforms to apply
                during training.
        """
        augs = []

        if self.cfg.augmentation.enable_crop:
            random_crop = RandomCrop(
                orig_size=orig_size,
                crop_size=self.cfg.augmentation.crop_size,
                mask=None,
                max_ratio=self.cfg.augmentation.single_category_max_area,
                ignored_category=self.ignore_label,
            )
            augs.append(random_crop)

        if self.cfg.augmentation.color_aug_ssd:
            color_aug = ColorAugSSDTransform(img_format=self.img_format)
            augs.append(color_aug)

        if self.cfg.augmentation.random_flip in ["horizontal", "vertical"]:
            random_flip = RandomFlip(
                orig_size=orig_size,
                prob=self.cfg.augmentation.random_flip_prob,  # default 0.3
                horizontal=self.cfg.augmentation.random_flip == "horizontal",
                vertical=self.cfg.augmentation.random_flip == "vertical",
            )
            augs.append(random_flip)

        return augs

    def get_test_transforms_2d(self, orig_size: Tuple[int, int]) -> List:
        """
        Get test transformation for 2D data.

        Builds a sequence of transforms including resize based on configuration
        settings. Each transform can be individually enabled/disabled via config.

        Args:
            orig_size (Tuple[int, int]): Original image size as (height, width).

        Returns:
            List[Transform]: Ordered list of augmentation transforms to apply
                during testing.
        """
        # Deterministic, no random augmentations at test time
        augs = []
        resize = ResizeShortestEdge(
            orig_size,
            self.cfg.augmentation.test_min_size,
            self.cfg.augmentation.test_max_size,
        )
        augs.append(resize)
        return augs

    @staticmethod
    def get_image(
        file_path: Union[str, np.ndarray, Image.Image], target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Load and optionally resize an RGB image.

        This method loads an image, handles EXIF orientation, converts to RGB
        format, and optionally resizes to a target size. EXIF orientation is
        respected to ensure images are displayed correctly.

        Args:
            file_path (Union[str, np.ndarray, Image.Image]): Path to the image file,
                or the image object itself (numpy array or PIL Image).
            target_size (Optional[Tuple[int, int]]): Target size as (width, height)
                to resize the image to. If None, original size is preserved.
                Defaults to None.

        Returns:
            np.ndarray: RGB image as numpy array with shape (H, W, 3) and
                dtype uint8.
        """
        if isinstance(file_path, (np.ndarray, Image.Image)):
            image = file_path
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
        else:
            image = Image.open(file_path)

        image = image.convert("RGB")
        image = ImageOps.exif_transpose(image)

        if target_size:
            image = image.resize(target_size)
        image = np.array(image)
        return image

    @staticmethod
    def get_room_mask(file_path: Union[str, np.ndarray]) -> np.ndarray:
        """Load a room mask from disk or pass-through an in-memory array.

        Args:
            file_path (Union[str, np.ndarray]): Path to an image file readable by PIL,
                or a pre-loaded mask array.

        Returns:
            np.ndarray: Room mask as a NumPy array cast to float64 ("double"),
                or None if the mask could not be loaded.
        """
        try:
            if isinstance(file_path, np.ndarray):
                room_mask = file_path
            else:
                room_mask = Image.open(file_path)
            return np.array(room_mask).astype("double", copy=False)
        except Exception:
            return None

    @staticmethod
    def get_mask_2d(
        file_path: Union[str, np.ndarray],
        target_size: Optional[Tuple[int, int]] = None,
        convert_rgb: bool = False,
    ) -> np.ndarray:
        """
        Load and resize a 2D mask to the target size if provided.

        Args:
            file_path (Union[str, np.ndarray]): Path to the mask file or the mask array.
            target_size (Optional[Tuple[int, int]]): Target size to resize the mask to.
            convert_rgb (bool): Whether to convert the mask to RGB.

        Returns:
            np.ndarray: Resized mask.
        """
        def resize_mask(mask_arr, size):
            pil_mask = Image.fromarray(mask_arr)
            pil_mask = pil_mask.resize(size, resample=Image.NEAREST)
            return np.array(pil_mask)

        if isinstance(file_path, np.ndarray):
            mask = file_path
            if target_size:
                mask = resize_mask(mask, target_size)
            return mask

        ext = os.path.splitext(file_path)[-1].lower()

        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            mask = Image.open(file_path)
            if convert_rgb:
                mask = mask.convert("RGB")
                mask = mask.resize(target_size, resample=Image.NEAREST)
            mask = np.array(mask)

        elif ext == ".npy":
            mask = np.load(file_path)
            if target_size:
                mask = resize_mask(mask, target_size)

        elif ext == ".npz":
            npz = np.load(file_path)
            mask = npz["data"] if "data" in npz else npz[npz.files[0]]
            if target_size:
                mask = resize_mask(mask, target_size)

        else:
            raise ValueError(f"Unsupported mask file extension: {ext}")

        return mask

    def get_mask_3d(self, file_path: Union[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load 3D semantic and instance segmentation volumes.

        This method loads volumetric segmentation data from NPZ files, handles
        coordinate system flipping if needed, and separates the two-channel
        volume into semantic and instance segmentation arrays.

        Args:
            file_path (Union[str, np.ndarray]): Path to NPZ file containing 3D segmentation data,
                or the segmentation array itself.
                Expected to have 'data' key with shape (2, D, H, W) or (D, H, W, 2).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (semantic3d, instance3d) where
                both are 3D volumes (D, H, W) containing class IDs and instance IDs
                respectively. Returns (None, None) if loading fails.

        Raises:
            ValueError: If the loaded volume doesn't have a channel dimension of size 2.
        """
        try:
            if isinstance(file_path, np.ndarray):
                segm_3d = file_path
            else:
                segm_3d = np.load(file_path)["data"]

            if self.is_flip:
                segm_3d = np.copy(np.flip(segm_3d, axis=[1, 2]))

            # Check if the first dimension is 2 (channels first) or last dimension is 2 (channels last)
            if segm_3d.shape[0] == 2:
                # [2, h, w, z] -> [h, w, z, 2]
                segm_3d = np.moveaxis(segm_3d, 0, -1)

            if segm_3d.shape[-1] == 2:
                semantic3d, instance3d = segm_3d[..., 0], segm_3d[..., 1]
            else:
                raise ValueError(
                    f"3D mask volume does not have a channel dimension of size 2: shape {segm_3d.shape}"
                )

            del segm_3d
            return semantic3d, instance3d

        except Exception as e:
            logging.warning(f'Warning: Got exception "{e}" when loading 3D mask at "{file_path}"')
            return None, None

    @staticmethod
    def get_depth(file_path: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load depth map from file or array.

        Supports multiple depth map formats including NumPy arrays (.npy) and
        OpenEXR files (.exr). Depth values are in meters (dataset-dependent).

        Args:
            file_path (Union[str, np.ndarray]): Path to depth file, or depth array.
                Supported extensions: .npy, .exr.

        Returns:
            np.ndarray: Depth map as 2D array (H, W) with float64 dtype, or None
                if loading fails or unsupported format.
        """
        try:
            if isinstance(file_path, np.ndarray):
                return file_path.astype("double")

            ext = os.path.splitext(file_path)[-1].lower()
            if ext == ".npy":
                depth = np.load(file_path).astype("double")
            elif ext == ".exr":
                depth = pyexr.read(file_path).squeeze().copy().astype("double")
            else:
                depth = None
        except Exception as e:
            logging.warning(f'Warning: Got exception "{e}" when loading depth at "{file_path}"')
            return None
        return depth

    def get_weight(self, file_path: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load 3D weighting volume for training.

        Weighting volumes assign importance weights to different voxels, typically
        based on distance to observed surfaces. This allows the loss function to
        focus on well-observed regions.

        Args:
            file_path (Union[str, np.ndarray]): Path to NPZ file containing weighting data with 'data' key,
                or the weighting array.

        Returns:
            np.ndarray: 3D weighting volume (D, H, W) with float values, or None if
                loading fails. Coordinates are flipped if self.is_flip is True.
        """
        try:
            if isinstance(file_path, np.ndarray):
                weight = file_path
            else:
                weight = np.load(file_path)["data"]

            if self.is_flip:
                weight = np.flip(weight, axis=[0, 1])
            return weight
        except Exception:
            return None

    def get_occupancy(self, file_path: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load 3D occupancy volume.

        Occupancy volumes indicate which voxels are occupied by surfaces (1) vs.
        empty space (0). Used for multiplane occupancy supervision.

        Args:
            file_path (Union[str, np.ndarray]): Path to NPZ file containing occupancy data with 'data' key,
                or the occupancy array.

        Returns:
            np.ndarray: 3D binary occupancy volume (D, H, W), or None if loading fails.
                Coordinates are flipped if self.is_flip is True.
        """
        try:
            if isinstance(file_path, np.ndarray):
                occupancy = file_path
            else:
                occupancy = np.load(file_path)["data"]

            if self.is_flip:
                occupancy = np.flip(occupancy, axis=[0, 1])
            return occupancy
        except Exception:
            return None

    def get_geometry(self, tsdf_file_path: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load truncated signed distance field (TSDF) geometry volume.

        TSDF volumes encode 3D geometry where each voxel stores the signed distance
        to the nearest surface. Negative values are inside surfaces, positive values
        outside. The distance is truncated beyond a certain range for efficiency.

        The file may also contain a 'mask' indicating valid/observed voxels vs.
        unobserved regions.

        Args:
            tsdf_file_path (Union[str, np.ndarray]): Path to NPZ file containing TSDF geometry,
                or the geometry array.
                Expected keys: 'data' (required), 'mask' (optional).

        Returns:
            dict: Dictionary with keys:
                - 'data': 3D TSDF volume (D, H, W) with float values
                - 'mask': Optional 3D boolean mask (D, H, W) indicating valid regions
                Returns None if loading fails. Coordinates are flipped if self.is_flip is True.
        """
        try:
            if isinstance(tsdf_file_path, np.ndarray):
                data = {"data": tsdf_file_path}
            else:
                data = np.load(tsdf_file_path)

            output = {"data": None}

            # Process data with optional flipping
            output["data"] = (
                np.ascontiguousarray(np.flip(data["data"], axis=[0, 1]))
                if self.is_flip
                else data["data"]
            )

            # Process mask if present
            if "mask" in data:
                output["mask"] = (
                    np.ascontiguousarray(np.flip(data["mask"], axis=[0, 1]))
                    if self.is_flip
                    else data["mask"]
                )

            if output["data"] is None:
                logging.warning(f"Warning: geometry data is None at {tsdf_file_path}")
            if "mask" in data:
                if output["mask"] is None:
                    logging.warning(f"Warning: geometry mask is None at {tsdf_file_path}")

            return output
        except Exception as e:
            logging.warning(f'Warning: Got exception "{e}" when loading geometry at "{tsdf_file_path}"')
            return None

    @staticmethod
    def convert_ins_seg(instance_seg: torch.Tensor, rearrange: bool = False) -> torch.Tensor:
        """
        Convert instance segmentation to appropriate index format.

        For Matterport3D dataset, instance IDs may be sparse and non-contiguous.
        This method optionally rearranges them to be contiguous starting from 1.
        For other datasets, it simply converts the dtype.

        Args:
            instance_seg (torch.Tensor or np.ndarray): Instance segmentation map where
                each unique value represents a distinct instance.
            rearrange (bool): If True, remap instance IDs to contiguous values
                [1, 2, 3, ...]. If False, keep original IDs. Defaults to False.

        Returns:
            torch.Tensor: Instance segmentation as int32 tensor with same spatial
                shape as input. IDs are contiguous if rearrange=True.
        """
        if isinstance(instance_seg, np.ndarray):
            instance_seg = torch.from_numpy(instance_seg)
        assert isinstance(instance_seg, torch.Tensor)

        if not rearrange:
            return instance_seg.to(dtype=torch.int32)

        unique_vals = torch.unique(instance_seg)
        ind_gt_rearranged = torch.zeros_like(instance_seg, dtype=torch.int32)
        for dst_i, orig_i in enumerate(unique_vals):
            ind_gt_rearranged[instance_seg == orig_i] = dst_i + 1
        return ind_gt_rearranged

    def apply_size_divisibility_padding(
        self, rgb: torch.Tensor, semantic_seg: Optional[torch.Tensor] = None,
        instance_seg: Optional[torch.Tensor] = None, depth: Optional[torch.Tensor] = None,
        room_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor]
    ]:
        """
        Pad all tensors to make dimensions divisible by specified value.

        Uses ModelInputResize to add padding to the right and bottom edges.
        Each tensor type uses appropriate padding values.

        Args:
            rgb (torch.Tensor): RGB image tensor (C, H, W).
            semantic_seg (Optional[torch.Tensor]): Semantic segmentation mask.
            instance_seg (Optional[torch.Tensor]): Instance segmentation mask.
            depth (Optional[torch.Tensor]): Depth map.
            room_mask (Optional[torch.Tensor]): Room mask for Matterport dataset.

        Returns:
            Tuple containing padded versions of all input tensors in the same
                order. None inputs remain None.
        """
        def _pad(x: Optional[torch.Tensor], val: float) -> Optional[torch.Tensor]:
            return self._model_input_resize.apply_image(x, pad_value=val) if x is not None else None

        rgb = _pad(rgb, DatasetConstants.PADDING_RGB_VALUE)
        semantic_seg = _pad(semantic_seg, self.ignore_label)
        instance_seg = _pad(instance_seg, DatasetConstants.PADDING_INST_SEG_VALUE)
        room_mask = _pad(room_mask, DatasetConstants.PADDING_ROOM_MASK_VALUE)
        depth = _pad(depth, DatasetConstants.PADDING_DEPTH_VALUE)

        return rgb, semantic_seg, instance_seg, depth, room_mask

    def prepare_instance_masks(
        self,
        instance_seg: np.ndarray,
        semantic_seg: np.ndarray,
        depth_gt: Optional[np.ndarray] = None,
    ):
        """
        Extract and organize instance-level annotations from segmentation masks.

        This method processes instance and semantic segmentation maps to create
        per-instance masks, class labels, and optional depth information. It
        filters instances by minimum pixel count, determines semantic labels via
        voting, and separates things (countable objects) from stuff (amorphous
        regions).

        Args:
            instance_seg (np.ndarray): Instance segmentation mask where each unique
                positive value represents a distinct instance.
            semantic_seg (np.ndarray): Semantic segmentation mask with class IDs.
            depth_gt (Optional[np.ndarray]): Ground truth depth map. If provided,
                per-instance depth maps and mean depths are computed.

        Returns:
            Tuple containing:
                - instances (dict): Dictionary with keys:
                    - image_size: (H, W) tuple
                    - gt_masks: Binary masks for each instance (N, H, W)
                    - gt_classes: Class ID for each instance (N,)
                    - gt_depths: Per-instance depth maps (N, H, W) if depth_gt provided
                    - mean_depths: Mean depth per instance (N,) if depth_gt provided
                - inst_ids (torch.Tensor): Instance IDs for things
                - stuff_ids (torch.Tensor): Class IDs for stuff categories
        """
        classes = []
        masks = []

        if depth_gt is not None:
            depths = []
            mean_depths = []

        def _add_results(_class_id, _seg_mask):
            classes.append(_class_id)
            masks.append(_seg_mask)

            if depth_gt is not None:
                seg_depth = torch.zeros_like(depth_gt)
                seg_depth[_seg_mask] = depth_gt[_seg_mask]
                depths.append(seg_depth)
                valid_seg_depth = seg_depth > 0
                mean_depths.append(seg_depth.sum() / valid_seg_depth.sum().clamp(1))

        instances = {"image_size": instance_seg.shape[:2]}
        indices = torch.unique(instance_seg)

        inst_ids = []
        for index in indices:
            if index <= 0:
                continue
            seg_mask = instance_seg == index
            if seg_mask.sum() <= self.min_instance_pixels:
                continue

            # determine semantic label of the current instance by voting
            semantic_labels = semantic_seg[seg_mask]
            unique_sem, counts = torch.unique(semantic_labels, return_counts=True)

            max_sem_label = torch.argmax(counts)
            class_id = unique_sem[max_sem_label]
            if class_id == self.ignore_label or int(class_id) in self.stuff_classes:
                continue

            inst_ids.append(index)
            _add_results(class_id, seg_mask)

        stuff_ids = []
        for class_id in torch.as_tensor(self.stuff_classes):
            seg_mask = semantic_seg == class_id
            if seg_mask.sum() == 0:
                continue
            stuff_ids.append(class_id)
            _add_results(class_id, seg_mask)

        if len(masks) == 0:
            # some images does not have annotation (all ignored)
            instances["gt_masks"] = torch.zeros(
                (0, instance_seg.shape[-2], instance_seg.shape[-1])
            )
            instances["gt_classes"] = torch.zeros(0, dtype=torch.long)
        else:
            instances["gt_masks"] = torch.stack(masks)
            instances["gt_classes"] = torch.stack(classes)

        if depth_gt is not None:
            if len(masks) == 0:
                instances["gt_depths"] = torch.zeros(
                    (0, depth_gt.shape[-2], depth_gt.shape[-1])
                )
                instances["mean_depths"] = torch.zeros(0)
            else:
                instances["gt_depths"] = depth_gt.unsqueeze(0).repeat(len(depths), 1, 1)
                instances["mean_depths"] = torch.stack(mean_depths)

        inst_ids = torch.stack(inst_ids) if len(inst_ids) else torch.empty(0)
        stuff_ids = torch.stack(stuff_ids) if len(stuff_ids) else torch.empty(0)

        return instances, inst_ids, stuff_ids

    def prepare_semantic_mapping(
        self, instances: torch.Tensor, semantics: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Create panoptic instance IDs and semantic class mapping for 3D volumes.

        This method converts per-voxel instance and semantic segmentations into
        panoptic format by assigning unique IDs to each instance and mapping
        them to semantic classes via majority voting. Stuff classes (walls, floor)
        are assigned fixed low IDs (1, 2, ...), while thing instances get higher
        sequential IDs.

        Args:
            instances (torch.Tensor): 3D instance segmentation volume (D, H, W)
                where each unique positive value represents a distinct thing instance.
                Background is 0.
            semantics (torch.Tensor): 3D semantic segmentation volume (D, H, W)
                with class IDs for each voxel.

        Returns:
            Tuple containing:
                - panoptic_instances (torch.Tensor): 3D volume (D, H, W) with panoptic
                  IDs where stuff classes have IDs 1, 2, ... and thing instances have
                  higher IDs.
                - semantic_mapping (Dict[int, int]): Dictionary mapping panoptic IDs
                  to semantic class IDs. Keys are panoptic instance IDs, values are
                  semantic class labels.
        """
        semantic_mapping = {}
        panoptic_instances = torch.zeros_like(instances).int()

        things_start_index = len(self.stuff_classes)  # map wall and floor to id 1 and 2

        non_thing_classes = torch.tensor(
            [0,] + self.stuff_classes,
            device=instances.device,
        )

        unique_instances = instances.unique()
        next_thing_id = things_start_index + 1
        for instance_id in unique_instances:
            # Ignore freespace
            if instance_id != 0:
                # Compute 3d instance surface mask
                instance_mask: torch.Tensor = instances == instance_id
                panoptic_instance_id = next_thing_id
                next_thing_id += 1
                panoptic_instances[instance_mask] = panoptic_instance_id

                # get semantic prediction
                semantic_region = torch.masked_select(semantics, instance_mask)
                thing_mask = torch.isin(semantic_region, non_thing_classes, invert=True)
                if thing_mask.sum() == 0:
                    continue
                semantic_things = semantic_region[thing_mask]

                unique_labels, semantic_counts = torch.unique(
                    semantic_things, return_counts=True
                )
                _, max_count_index = torch.max(semantic_counts, dim=0)
                selected_label = unique_labels[max_count_index]

                semantic_mapping[panoptic_instance_id] = selected_label.int().item()

        # Merge stuff classes
        for idx, stuff_class in enumerate(self.stuff_classes):
            stuff_mask = semantics == stuff_class
            if stuff_mask.sum() == 0:
                continue
            stuff_id = idx + 1
            panoptic_instances[stuff_mask] = stuff_id
            semantic_mapping[stuff_id] = stuff_class

        return panoptic_instances, semantic_mapping

    def thicken_grid(self, grid: torch.Tensor, grid_dims: List[int], frustum_mask: torch.Tensor):
        """
        Dilate a sparse 3D binary grid by expanding each occupied voxel to its 26-neighborhood.

        This method performs morphological dilation on a 3D occupancy grid by adding
        all 26-connected neighbors (3x3x3 cube minus center) for each occupied voxel.
        The result is then intersected with the viewing frustum mask to remove
        voxels outside the valid camera viewing volume. This thickening operation
        creates more robust surface representations for evaluation.

        Args:
            grid (torch.Tensor): Sparse binary 3D grid (D, H, W) indicating occupied
                voxels (True/1) and empty voxels (False/0).
            grid_dims (List[int]): Dimensions of the output grid as [depth, height, width].
                Used for bounds checking.
            frustum_mask (torch.Tensor): Boolean mask (D, H, W) indicating which voxels
                are within the camera's viewing frustum. Only voxels with True values
                are considered valid.

        Returns:
            torch.Tensor: Thickened binary grid (D, H, W) where each originally occupied
                voxel has been expanded to include its 26-neighbors, intersected with
                the frustum mask. Returned as boolean tensor.
        """
        device = frustum_mask.device
        offsets = torch.nonzero(torch.ones(3, 3, 3, device=device)).long()
        locs_grid = grid.nonzero(as_tuple=False)
        locs = locs_grid.unsqueeze(1).repeat(1, 27, 1)
        locs += offsets
        locs = locs.view(-1, 3)
        masks = ((locs >= 0) & (locs < torch.as_tensor(grid_dims, device=device))).all(-1)
        locs = locs[masks]

        thicken = torch.zeros(grid_dims, dtype=torch.bool, device=device)
        thicken[locs[:, 0], locs[:, 1], locs[:, 2]] = True
        # frustum culling
        thicken = thicken & frustum_mask

        return thicken

    def prepare_instance_masks_thicken(
        self,
        instances: torch.Tensor,
        semantic_mapping: Dict[int, int],
        distance_field: torch.Tensor,
        frustum_mask: torch.Tensor,
        iso_value: float = 1.0,
        truncation: float = 3.0,
        downsample_factor: int = 1,
    ) -> Dict[int, Tuple[torch.Tensor, int]]:
        """
        Generate thickened 3D instance masks for evaluation.

        This method creates per-instance 3D occupancy grids by extracting surfaces
        from a signed distance field (SDF), applying morphological dilation
        (thickening), and optionally downsampling.
        The thickening operation makes evaluation more robust to small alignment
        errors and discretization artifacts.

        Args:
            instances (torch.Tensor): 3D instance segmentation volume (D, H, W)
                where each unique ID represents a distinct instance. Background
                is typically ID 0.
            semantic_mapping (Dict[int, int]): Mapping from instance IDs to their
                semantic class labels. Keys are instance IDs, values are class IDs.
            distance_field (torch.Tensor): Truncated signed distance field (D, H, W)
                representing the geometry. Negative values inside surfaces, positive
                outside. Typically at 256³ resolution.
            frustum_mask (torch.Tensor): Boolean viewing frustum mask (D, H, W)
                indicating valid voxels within camera field of view.
            iso_value (float): Isovalue threshold for surface extraction. Voxels
                with |distance| < iso_value are considered on the surface.
                Defaults to 1.0.
            truncation (float): Fill value for voxels outside instance regions.
                Should match truncation distance used in distance field.
                Defaults to 3.0.
            downsample_factor (int): Factor for downsampling (1, 2, 4, etc.).
                Must evenly divide 256. Factor of 1 means no downsampling (256³),
                2 gives 128³, 4 gives 64³, etc. Defaults to 1.

        Returns:
            Dict[int, Tuple[torch.Tensor, int]]: Dictionary mapping instance IDs to
                tuples of (thickened_grid, semantic_class) where:
                - thickened_grid: Boolean occupancy grid at target resolution
                - semantic_class: Integer semantic class ID for the instance

        Raises:
            AssertionError: If downsample_factor is not an integer or doesn't
                evenly divide 256.
        """
        assert isinstance(downsample_factor, int) and 256 % downsample_factor == 0
        grid_dims = DatasetConstants.DEFAULT_GRID_DIMS.copy()
        need_rescale = downsample_factor != 1
        if need_rescale:
            grid_dims = (torch.as_tensor(grid_dims) // downsample_factor).tolist()
            frustum_mask = (
                F.interpolate(
                    frustum_mask[None, None].float(), size=grid_dims, mode="nearest"
                ).squeeze(0, 1).bool()
            )

        instance_information = {}

        for instance_id, semantic_class in semantic_mapping.items():
            instance_mask: torch.Tensor = instances == instance_id
            instance_distance_field = torch.full_like(
                instance_mask, dtype=torch.float, fill_value=truncation
            )
            instance_distance_field[instance_mask] = distance_field.squeeze()[instance_mask]
            instance_distance_field_masked = instance_distance_field.abs() < iso_value

            if need_rescale:
                instance_distance_field_masked = (
                    F.max_pool3d(
                        instance_distance_field_masked[None, None].float(),
                        kernel_size=downsample_factor + 1,
                        stride=downsample_factor,
                        padding=1,
                    ).squeeze(0, 1).bool()
                )

            instance_grid = self.thicken_grid(
                instance_distance_field_masked, grid_dims, frustum_mask
            )
            instance_grid = instance_grid.to(
                torch.device("cpu"), non_blocking=True
            )
            instance_information[instance_id] = instance_grid, semantic_class

        return instance_information

    def get_common_segmentation(self, dataset_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess segmentation masks from dataset files.

        Handles different dataset formats. Applies void masking and
        converts to appropriate data types.

        Args:
            dataset_dict (Dict[str, Any]): Dictionary containing file paths.
                Expected keys: 'seg_sem_label_file_name' and 'seg_ins_label_file_name'
                for synthetic, or 'segm_label_file_name' for Front3D/Matterport.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (semantic_seg, instance_seg) where
                void regions are marked with ignore_label in semantic segmentation.
        """
        # synthetic
        if "seg_sem_label_file_name" in dataset_dict and "seg_ins_label_file_name" in dataset_dict:
            semantic_seg_raw = self.get_mask_2d(
                dataset_dict["seg_sem_label_file_name"], self.resize_hw
            )
            instance_seg = self.get_mask_2d(
                dataset_dict["seg_ins_label_file_name"], self.resize_hw
            )
        # front3d & matterport
        elif "segm_label_file_name" in dataset_dict:
            panoptic_seg = self.get_mask_2d(dataset_dict["segm_label_file_name"])
            semantic_seg_raw, instance_seg = panoptic_seg[..., 0], panoptic_seg[..., 1]
            del panoptic_seg
        else:
            raise ValueError("No segmentation label file name found in dataset_dict")

        semantic_seg = semantic_seg_raw.copy()
        void_mask = semantic_seg == 0
        semantic_seg[void_mask] = self.ignore_label
        semantic_seg = semantic_seg.astype("double")
        instance_seg = instance_seg.astype("double")

        semantic_seg_raw[void_mask] = len(self.categories) + 1
        dataset_dict["raw_sem_seg"] = semantic_seg_raw.astype("int32")
        dataset_dict["raw_inst_seg"] = self.convert_ins_seg(instance_seg, rearrange=self.is_matterport)
        return semantic_seg, instance_seg

    def get_transformed_depth(
        self, common_size_transform: Compose,
        depth: np.ndarray = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process depth map after spatial transforms have been applied.

        The depth map should already be spatially transformed (resized, cropped, etc.)
        by the same transforms applied to the image and segmentation masks.
        This method handles depth value scaling based on the resize ratio and
        creates the output tensors.

        Args:
            common_size_transform (Compose): The transform that was applied to resize
                the depth. Used to compute the depth value scaling factor.
            depth (np.ndarray): Already spatially transformed depth map (H, W) in meters,
                or None.

        Returns:
            Tuple containing:
                - raw_depth (torch.Tensor): Depth scaled by 256 as int32 for evaluation
                - transformed_depth (torch.Tensor): Depth as float32 tensor (for collation)
                - depth (torch.Tensor): Depth as float32 tensor with optional depth
                  bound clipping applied
        """
        if depth is None:
            return None, None, None

        # Compute aug_scale from the resize transform
        # This scales depth values to maintain correct 3D geometry after resize
        # When image is resized, the apparent depth changes inversely with scale
        aug_scale = 1.0  # Metric depth, identical scale

        # Apply depth value scaling
        if self.depth_bound:
            depth_min, depth_max = depth.min(), depth.max()
            depth_scaled = np.clip(depth * aug_scale, depth_min, depth_max)
        else:
            depth_scaled = depth * aug_scale

        # raw_depth for evaluation: scaled by 256 as int32
        raw_depth = (depth_scaled * 256).astype(np.int32)

        # transformed_depth: as tensor (not DepthMap) so it can be properly collated/stacked
        # The model can wrap it in DepthMap using the intrinsic if needed
        transformed_depth = torch.as_tensor(depth_scaled.astype("float32"))

        # Training depth: float32 tensor
        depth_out = torch.as_tensor(depth_scaled.astype("float32", copy=False))

        return raw_depth, transformed_depth, depth_out

    @staticmethod
    def get_transformed_room_mask(room_mask: np.ndarray, tfms_2d: Compose) -> np.ndarray:
        """Apply 2D geometric transforms to a room mask.

        This is used to keep the room mask spatially aligned with the RGB/label
        tensors after the same resize/crop/flip transforms have been applied.

        Args:
            room_mask (np.ndarray): Room mask array (H, W) (or compatible shape)
                to be spatially transformed.
            tfms_2d (Compose): Transform pipeline whose `apply_segmentation`
                method applies the same geometric transforms used for labels.

        Returns:
            np.ndarray: The transformed room mask, aligned with transformed 2D inputs.
        """
        room_mask = tfms_2d.apply_segmentation(room_mask)
        return room_mask

    def get_mp_occ(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load multi-plane occupancy data.

        Multiplane occupancy provides intermediate 3D supervision at a coarser
        level than full TSDF.

        Args:
            dataset_dict (Dict[str, Any]): Dataset sample dictionary.

        Returns:
            Dict[str, Any]: Updated dataset_dict with 'mp_occ_256' key added if
                loading succeeded, otherwise unchanged.
        """
        if self.enable_mp_occ and "occupancy_file_name" in dataset_dict:
            mp_occ = self.get_occupancy(dataset_dict["occupancy_file_name"])
            dataset_dict["mp_occ_256"] = torch.from_numpy(mp_occ)
        return dataset_dict


class Front3DPreprocessor(BasePreprocessor):
    """
    Preprocessor specialized for Front3D dataset format.

    Extends BasePreprocessor to handle Front3D-specific multiplane occupancy
    loading from geometry files.
    """

    def get_mp_occ(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load multiplane occupancy from geometry file for Front3D.

        Front3D stores geometry (TSDF) which is converted to occupancy by
        thresholding.

        Args:
            dataset_dict (Dict[str, Any]): Dataset sample dictionary containing
                'geometry_file_name' key.

        Returns:
            Dict[str, Any]: Updated dataset_dict with 'mp_occ_256' key added.
        """
        if self.enable_mp_occ and "geometry_file_name" in dataset_dict:
            geometry_content = self.get_geometry(dataset_dict["geometry_file_name"])
            geometry = geometry_content["data"]

            geometry = self.vol_transforms["geometry"](geometry)
            mp_occ = self.vol_transforms["mp_occupancy"](geometry)
            dataset_dict["mp_occ_256"] = mp_occ
        return dataset_dict


class Matterport3DPreprocessor(Front3DPreprocessor):
    """
    Preprocessor specialized for Matterport3D dataset format.

    Extends Front3DPreprocessor to handle Matterport3D-specific data including:
    - Depth loading with PNG format and scaling by divisor
    - Per-image camera intrinsics
    - Frustum masks computed from geometry files
    - Depth value clamping to valid range
    """

    def __init__(
        self,
        cfg,
        is_flip: bool = False,
        is_matterport: bool = True,
        categories: List[Dict] = None,
        stuff_classes: List[int] = None,
        ignore_label: int = 255,
        min_instance_pixels: int = 1,
        resize_hw: Tuple[int, int] = None,
        depth_resize_hw: Tuple[int, int] = None,
        size_divisibility: int = 32,
        depth_bound: bool = False,
        iso_value: float = 1.0,
        truncation_range: List[float] = None,
        downsample_factor: int = 1,
        enable_mp_occ: bool = True,
        depth_min: float = 0.4,
        depth_max: float = 6.0,
        **kwargs
    ):
        """
        Constructor for Matterport3DPreprocessor.

        Args: Same as BasePreprocessor.__init__, plus:
            depth_min (float): Minimum valid depth value in meters. Depths
                below this are set to 0. Defaults to 0.4.
            depth_max (float): Maximum valid depth value in meters. Depths
                above this are set to 0. Defaults to 6.0.
        """
        super().__init__(
            cfg, is_flip, is_matterport, categories, stuff_classes, ignore_label,
            min_instance_pixels, resize_hw, depth_resize_hw, size_divisibility,
            depth_bound, iso_value, truncation_range, downsample_factor,
            enable_mp_occ, **kwargs
        )
        self.depth_min = depth_min
        self.depth_max = depth_max

    def get_depth(self, file_path: str) -> np.ndarray:
        """
        Load and preprocess depth map for Matterport3D.

        Matterport3D stores depth as PNG images with scaled integer values.
        This method loads the PNG, scales by the dataset-specific divisor,
        and clamps values to the valid depth range.

        Args:
            file_path (str): Path to depth PNG file.

        Returns:
            np.ndarray: Depth map in meters with invalid depths set to 0,
                or None if loading fails.
        """
        if file_path is None:
            return None
        try:
            depth = Image.open(file_path)
            dividor = DatasetConstants.MATTERPORT_DEPTH_DIVISOR
            depth = np.array(depth).astype("double", copy=False) / dividor
            depth[depth < self.depth_min] = 0
            depth[depth > self.depth_max] = 0
            return depth
        except Exception as e:
            logging.error(f"Error loading depth from {file_path}: {e}")
            return None

    def get_frustum_mask(self, geometry_file_name: str, **kwargs) -> torch.Tensor:
        """
        Get frustum mask from geometry file for Matterport3D.

        Matterport3D includes a valid region mask in geometry files. The
        frustum mask indicates which voxels are within the camera's viewing
        frustum and have valid data.

        Args:
            geometry_file_name (str): Path to geometry .npz file containing
                'mask' field.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Boolean frustum mask (256, 256, 256).
        """
        geometry_content = self.get_geometry(geometry_file_name)
        return ~torch.as_tensor(geometry_content["mask"], dtype=torch.bool)

    def get_intrinsic(self, intrinsic_label_file_name: str, **kwargs) -> torch.Tensor:
        """
        Load per-image camera intrinsic matrix for Matterport3D.

        Matterport3D has varying camera intrinsics across images, unlike
        Front3D which uses a fixed intrinsic.

        Args:
            intrinsic_label_file_name (str): Path to .npy file containing
                flattened 4x4 intrinsic matrix.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Camera intrinsic matrix (4, 4) as float32.
        """
        return torch.from_numpy(
            np.load(intrinsic_label_file_name).reshape(4, 4)
        ).float()
