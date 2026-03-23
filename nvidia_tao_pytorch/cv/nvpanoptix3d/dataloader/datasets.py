# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVPanoptix3D datasets.

Supports standard training and inference with arbitrary RGB images.
"""

import os
import json
import glob
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.augmentations import (
    Compose, apply_transform_multi
)
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.preprocessor import (
    BasePreprocessor, Front3DPreprocessor, Matterport3DPreprocessor,
    DatasetConstants
)
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.helper import create_frustum_mask


class BaseNVPanoptix3DDataset(Dataset):
    """
    Base class for NVPanoptix3D datasets.

    This class provides the foundational infrastructure for loading and processing
    multi-modal 3D scene understanding data, including RGB images, depth maps,
    2D/3D semantic and instance segmentations, and volumetric geometry. It handles
    data augmentation, preprocessing, and batch collation for both training and
    inference modes.

    The class supports multiple dataset formats (Front3D, Matterport3D, Predict)
    through a preprocessor pattern, allowing dataset-specific loading logic while
    maintaining a unified interface.
    """

    def __init__(self, cfg, **kwargs):
        """
        Constructor for BaseNVPanoptix3DDataset.

        Initializes dataset configuration parameters, sets up preprocessing
        arguments, and prepares the preprocessor mapping for different dataset
        types. Actual data loading is delegated to child classes.

        Args:
            cfg: Configuration object containing dataset parameters including:
                - downsample_factor: Factor for downsampling 3D volumes
                - iso_value: Isovalue for surface extraction
                - ignore_label: Label ID to ignore in segmentation
                - min_instance_pixels: Minimum pixels for valid instances
                - img_format: Image format (RGB or BGR)
                - target_size: Target 2D image dimensions
                - depth_bound: Depth value bounds
                - augmentation.size_divisibility: Padding divisibility
                - depth_size: Target depth map dimensions
                - enable_3d: Whether to load 3D data
                - occ_truncation_lvl: Occupancy truncation level
                - truncation_range: Range for SDF truncation
                - enable_mp_occ: Enable multiplane occupancy
            **kwargs: Additional keyword arguments.
        """
        self.cfg = cfg
        self.downsample_factor = cfg.downsample_factor
        self.iso_value = cfg.iso_value
        self.ignore_label = cfg.ignore_label
        self.min_instance_pixels = cfg.min_instance_pixels
        self.img_format = cfg.img_format
        self.resize_hw = cfg.target_size
        self.depth_bound = cfg.depth_bound
        self.size_divisibility = cfg.augmentation.size_divisibility
        self.gen_aug_weight = cfg.augmentation.gen_aug_weight
        self.depth_resize_hw = cfg.depth_size
        self.enable_3d = cfg.enable_3d
        self.occ_truncation_lvl = cfg.occ_truncation_lvl
        self.truncation_range = cfg.truncation_range
        self.enable_mp_occ = cfg.enable_mp_occ

        self.is_flip = None
        self.categories = None
        self.intrinsic = None
        self.frustum_mask = None
        self.stuff_classes = None
        self.thing_classes = None

        # Initialize dict of samples
        self.id2img = {}
        self.img_ids = []
        self.preprocessor_args = {
            "cfg": cfg,
            "is_flip": self.is_flip,
            "is_matterport": False,
            "categories": self.categories,
            "stuff_classes": self.stuff_classes,
            "ignore_label": self.ignore_label,
            "min_instance_pixels": self.min_instance_pixels,
            "resize_hw": self.resize_hw,
            "depth_resize_hw": self.depth_resize_hw,
            "size_divisibility": self.size_divisibility,
            "depth_bound": self.depth_bound,
            "iso_value": self.iso_value,
            "truncation_range": self.truncation_range,
            "downsample_factor": self.downsample_factor,
        }
        # Preprocessor will be initialized in child classes
        self.preprocessor_map = {
            "synthetic": BasePreprocessor,
            "front3d": Front3DPreprocessor,
            "matterport": Matterport3DPreprocessor,
            "predict": BasePreprocessor,
        }
        self.preprocessor = None

    def setup_preprocessor(self, preprocessor_name: str):
        """
        Initialize and return a dataset-specific preprocessor.

        Args:
            preprocessor_name (str): Name of the preprocessor type. Supported values:
                "synthetic", "front3d", "matterport", "predict".

        Returns:
            BasePreprocessor: An initialized preprocessor instance configured with
                the dataset's preprocessing arguments.
        """
        preprocessor = self.preprocessor_map[preprocessor_name](**self.preprocessor_args)
        return preprocessor

    def get_data_meta(self):
        """Get paths to load the dataset."""
        raise NotImplementedError("get_data_meta not implemented")

    def __len__(self):
        """Get dataset length."""
        return len(self.img_ids)

    def collate_fn(self, batch):
        """
        Collate a list of per-sample dicts into a batch dict.

        - Non-tensors: kept as lists
        - Same-shape tensors: stacked
        - Variable-shape tensors: padded to max shape then stacked
        """
        def _pad_value(key: str) -> int:
            if "sem_seg" in key:
                return self.ignore_label
            # Boolean masks should pad with 0/False (not -1, which becomes True when cast to bool)
            if key in {"room_mask", "room_mask_buol", "frustum_mask"}:
                return 0
            if "inst" in key or "mask" in key:
                return -1
            return 0

        def _pad_to_shape(v: torch.Tensor, max_shape: list, *, value: int) -> torch.Tensor:
            pad = []
            for cur, mx in zip(reversed(v.shape), reversed(max_shape)):
                pad.extend([0, mx - cur])
            return torch.nn.functional.pad(v, pad, value=value)

        out: Dict[str, list] = {}
        for item in batch:
            for key, value in item.items():
                out.setdefault(key, []).append(value)

        for key, values in out.items():
            if not values or not torch.is_tensor(values[0]):
                continue

            tensors = values  # list of torch.Tensors
            if all(t.shape == tensors[0].shape for t in tensors):
                out[key] = torch.stack(tensors, dim=0)
                continue

            if all(t.ndim <= 1 for t in tensors):
                out[key] = pad_sequence([t.view(-1) for t in tensors], batch_first=True, padding_value=-1)
                continue

            shapes = [t.shape for t in tensors]
            max_shape = [max(s[i] for s in shapes) for i in range(len(shapes[0]))]
            out[key] = torch.stack(
                [_pad_to_shape(t, max_shape, value=_pad_value(key)) for t in tensors],
                dim=0,
            )

        return out

    def get_category_mapping(self):
        """ Map category index in json to 1 based index. """
        self.thing_dataset_id_to_contiguous_id = {}
        self.stuff_dataset_id_to_contiguous_id = {}
        if self.categories:
            for i, cat in enumerate(self.categories):
                if cat["isthing"]:
                    self.thing_dataset_id_to_contiguous_id[cat["id"]] = i + 1

                # in order to use the semantic segmentation evaluator
                self.stuff_dataset_id_to_contiguous_id[cat["id"]] = i + 1
            return True
        else:
            return False

    def get_stuff_classes(self):
        """Get list of stuff class IDs."""
        return list(self.stuff_dataset_id_to_contiguous_id.keys())

    def get_thing_classes(self):
        """Get list of thing class IDs."""
        return list(self.thing_dataset_id_to_contiguous_id.keys())

    def load_frustum_mask(self, file_path: str = None) -> torch.Tensor:
        """
        Load pre-defined frustum mask for this dataset.

        Args:
            file_path (str): Path to NPZ file containing frustum mask.

        Returns:
            torch.Tensor: Frustum mask tensor. Returns None if loading fails.
        """
        try:
            if not file_path:
                return None
            frustum_mask = torch.from_numpy(np.load(file_path)["mask"]).bool()
            if self.is_flip:
                frustum_mask = torch.flip(frustum_mask, dims=[0, 1])
            return frustum_mask
        except Exception as e:
            logging.warning(
                f"Warning: Got exception '{e}' when loading frustum mask at '{file_path}'"
            )
            return None

    def _prepare_2d_data(
        self,
        dataset_dict: Dict[str, Any],
        mode: str = "train",
        *,
        batch_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load and preprocess 2D data including images, segmentations, and depth.

        This method handles the complete 2D data pipeline: loading RGB images,
        semantic/instance segmentations, depth maps, and optional room masks;
        applying augmentation transforms; and preparing instance-level annotations.
        The processed data is added to the dataset dictionary.

        Args:
            dataset_dict (Dict[str, Any]): Dictionary containing file paths and
                metadata for the sample. Will be updated in-place with processed data.
            mode (str): Processing mode, either "train" or "test". Determines which
                augmentation transforms are applied. Defaults to "train".

        Returns:
            Dict[str, Any]: The input dataset_dict enriched with processed 2D data:
                - image: Normalized RGB tensor (C, H, W)
                - semantic_seg: Semantic segmentation mask (stored under "sem_seg")
                - instances: Dict containing instance masks and metadata
                - inst_ids: List of instance IDs
                - stuff_ids: List of stuff class IDs
                - depth: Depth map (if available)
                - room_mask: Room mask (if available)
                - frustum_mask: Viewing frustum mask
                - intrinsic: Camera intrinsic matrix
        """
        rgb = self.preprocessor.get_image(dataset_dict["file_name"], self.resize_hw)
        semantic_seg, instance_seg = self.preprocessor.get_common_segmentation(dataset_dict)
        # Load depth before applying transforms so it can be resized together
        depth = self.preprocessor.get_depth(dataset_dict.get("depth_label_file_name", None))
        assert depth is not None, (
            f"Cannot load depth map from {dataset_dict.get('depth_label_file_name', None)}"
        )
        # Store raw_depth at original resolution (before any transforms)
        raw_depth = (depth * 256).astype(np.int32)
        # Depth resized to the fixed depth plane (for collation/visualization)
        transformed_depth = self.preprocessor.depth_transforms(depth.copy()).depth_map

        # load room mask if exists
        room_mask = self.preprocessor.get_room_mask(dataset_dict.get("room_mask_file_name", None))
        if room_mask is not None:
            # room mask for mp occ (at fixed resolution for multiplane occupancy)
            dataset_dict["room_mask_buol"] = (
                self.preprocessor.room_mask_transforms(room_mask.copy()) > 0
            )

        # Multi-scale resize (train only). This is the only place we randomize image scale.
        common_size_transform = self.preprocessor.get_common_size_transform()
        if mode == "train" and self.is_training:
            masks: Dict[str, np.ndarray] = {
                "sem_seg": semantic_seg,
                "inst_seg": instance_seg,
                "depth": depth,
            }
            if room_mask is not None:
                masks["room_mask"] = room_mask
            mask_keys = list(masks.keys())
            mask_vals = [masks[k] for k in mask_keys]
            rgb, mask_vals = common_size_transform.apply_multi(
                image=rgb, masks=mask_vals, random_select=True, seed=batch_seed
            )
            masks = dict(zip(mask_keys, mask_vals))
            semantic_seg = masks["sem_seg"]
            instance_seg = masks["inst_seg"]
            depth = masks["depth"]
            room_mask = masks.get("room_mask", None)

        # Apply other 2D augmentations (crop, color, flip) to rgb and all 2D maps.
        transforms_2d_list = self.preprocessor.get_transforms_2d(mode=mode)
        for transform_2d in transforms_2d_list:
            masks = {"sem_seg": semantic_seg, "inst_seg": instance_seg, "depth": depth}
            if room_mask is not None:
                masks["room_mask"] = room_mask
            mask_keys = list(masks.keys())
            mask_vals = [masks[k] for k in mask_keys]
            rgb, mask_vals = apply_transform_multi(rgb, mask_vals, transform_2d)
            masks = dict(zip(mask_keys, mask_vals))
            semantic_seg = masks["sem_seg"]
            instance_seg = masks["inst_seg"]
            depth = masks["depth"]
            room_mask = masks.get("room_mask", None)

        rgb = torch.as_tensor(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        semantic_seg = torch.as_tensor(semantic_seg.astype("long"))
        instance_seg = torch.as_tensor(instance_seg.astype("long"))
        depth = torch.as_tensor(depth.astype("float32", copy=False))
        if room_mask is not None:
            room_mask = torch.as_tensor(room_mask.astype("long"))

        # Track the true (H, W) before size-divisibility padding so 2D outputs can be
        # cropped correctly under multi-scale training.
        nopad_image_shape = rgb.shape[-2:]
        rgb, semantic_seg, instance_seg, depth, room_mask = self.preprocessor.apply_size_divisibility_padding(
            rgb, semantic_seg, instance_seg, depth, room_mask
        )

        instances, inst_ids, stuff_ids = self.preprocessor.prepare_instance_masks(
            instance_seg, semantic_seg, depth
        )
        # Propagate per-sample loss weight into the instance dict so the PL module can
        # forward it into SetCriterion targets.
        instances["loss_weight"] = float(dataset_dict.get("loss_weight", 1.0))

        dataset_dict["image"] = rgb
        dataset_dict["sem_seg"] = semantic_seg.long()
        dataset_dict["instances"] = instances
        dataset_dict["inst_ids"] = inst_ids
        dataset_dict["stuff_ids"] = stuff_ids
        dataset_dict["nopad_image_shape"] = nopad_image_shape
        # get mp occupancy if enabled
        dataset_dict = self.preprocessor.get_mp_occ(dataset_dict)

        if room_mask is not None:
            dataset_dict["room_mask"] = room_mask > 0

        if depth is not None:
            dataset_dict["depth"] = depth.float()
            dataset_dict["transformed_depth"] = transformed_depth
            dataset_dict["raw_depth"] = raw_depth

        # get frustum mask & intrinsic
        dataset_dict["frustum_mask"] = self.preprocessor.get_frustum_mask(**dataset_dict)
        dataset_dict["intrinsic"] = self.preprocessor.get_intrinsic(**dataset_dict)

        return dataset_dict

    def _prepare_3d_data_train(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and preprocess 3D volumetric data for training.

        This method loads 3D geometry (TSDF), 3D semantic/instance segmentations,
        and weighting volumes, applies transformations, and prepares multi-scale
        3D ground truth annotations. The data is processed at three resolution
        levels (256³, 128³, 64³) to support multi-scale supervision.

        Args:
            dataset_dict (Dict[str, Any]): Dictionary containing file paths and
                2D data from _prepare_2d_data. Will be updated in-place with 3D data.

        Returns:
            Dict[str, Any]: The input dataset_dict enriched with 3D data:
                - geometry: Truncated signed distance field at 256³
                - occupancy_256/128/64: Binary occupancy grids at multiple scales
                - instances.gt_masks_3d_256/128/64: Instance masks at multiple scales
                - weighting3d_256/128/64: Spatial weighting volumes
        """
        geometry_content = self.preprocessor.get_geometry(dataset_dict["geometry_file_name"])
        geometry_sdf = geometry_content["data"]
        sem_3d, ins_3d = self.preprocessor.get_mask_3d(dataset_dict["segm_3d_file_name"])
        weighting = self.preprocessor.get_weight(dataset_dict["weighting_file_name"])

        transforms_3d = self.preprocessor.vol_transforms

        # This matches the sparse-conv pipeline pattern: occupancy uses wider bands, geometry
        # stays in the fine truncation range.
        geometry_tdf = transforms_3d["geometry"](geometry_sdf)  # TUDF/TDF in [0, DEFAULT_TRUNCATION]
        geometry_occ = transforms_3d["geometry_occ"](geometry_sdf)  # TUDF/TDF in [0, truncation_range[1]]
        dataset_dict["geometry"] = geometry_tdf
        dataset_dict["occupancy_256"] = transforms_3d["occupancy_256"](geometry_occ)
        dataset_dict["occupancy_128"] = transforms_3d["occupancy_128"](geometry_occ)
        dataset_dict["occupancy_64"] = transforms_3d["occupancy_64"](geometry_occ)

        # Convert semantic + instance 3D label volumes into a per-mask representation.
        # For each thing instance id and stuff class id, we create a boolean 3D mask,
        # then cache the stacked masks at 256^3 and downsample to 128^3/64^3.
        sem_3d = transforms_3d["semantic3d"](sem_3d)
        ins_3d = transforms_3d["semantic3d"](ins_3d)

        # Build per-id boolean masks, then store full + downsampled stacks.
        segm_3d_masks = []
        for index in dataset_dict["inst_ids"]:
            segm_3d_masks.append(ins_3d == index)
        for class_id in dataset_dict["stuff_ids"]:
            segm_3d_masks.append(sem_3d == class_id)
        if len(segm_3d_masks) > 0:
            segm_3d_masks = torch.stack(segm_3d_masks)
            dataset_dict["instances"]["gt_masks_3d_256"] = segm_3d_masks
            dataset_dict["instances"]["gt_masks_3d_128"] = transforms_3d["segmentation3d_128"](segm_3d_masks)
            dataset_dict["instances"]["gt_masks_3d_64"] = transforms_3d["segmentation3d_64"](segm_3d_masks)
        else:
            # No instances/stuff classes; keep empty mask tensors per scale.
            dataset_dict["instances"]["gt_masks_3d_256"] = torch.zeros(
                (0, 256, 256, 256)
            )
            dataset_dict["instances"]["gt_masks_3d_128"] = torch.zeros(
                (0, 128, 128, 128)
            )
            dataset_dict["instances"]["gt_masks_3d_64"] = torch.zeros(
                (0, 64, 64, 64)
            )
        del segm_3d_masks

        # transform weighting
        weighting = transforms_3d["weighting3d"](weighting)
        dataset_dict["weighting3d_256"] = weighting
        dataset_dict["weighting3d_128"] = transforms_3d["weighting3d_128"](weighting)
        dataset_dict["weighting3d_64"] = transforms_3d["weighting3d_64"](weighting)

        del weighting

        return dataset_dict

    def _prepare_3d_data_test(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and preprocess 3D volumetric data for evaluation.

        This method loads 3D geometry and segmentations, applies transformations,
        and prepares ground truth instance information for evaluation. Unlike
        training, this extracts instance-level information and performs surface
        thickening for more robust evaluation metrics.

        Args:
            dataset_dict (Dict[str, Any]): Dictionary containing file paths and
                2D data from _prepare_2d_data. Will be updated in-place with 3D data.

        Returns:
            Dict[str, Any]: The input dataset_dict enriched with 3D evaluation data:
                - geometry: Truncated signed distance field
                - instance_info_gt: Dict containing ground truth instance information
                - downsample_factor: Factor used for downsampling
        """
        # Load volumes
        geometry_content = self.preprocessor.get_geometry(dataset_dict["geometry_file_name"])
        geometry = geometry_content["data"]
        sem_3d, ins_3d = self.preprocessor.get_mask_3d(dataset_dict["segm_3d_file_name"])

        # define transforms
        transforms_3d = self.preprocessor.vol_transforms

        # For evaluation, keep `geometry` in the model's truncation range (typically 3.0),
        # consistent with training-time geometry supervision and marching-cubes mesh extraction.
        # Occupancy bands (8/6/3) are only needed for training-time pruning supervision.
        geometry = transforms_3d["geometry"](geometry)
        dataset_dict["geometry"] = geometry

        # transform panoptic
        sem_3d = transforms_3d["semantic3d"](sem_3d)
        ins_3d = transforms_3d["semantic3d"](ins_3d)

        instances_gt, instance_semantic_classes_gt = self.preprocessor.prepare_semantic_mapping(
            ins_3d, sem_3d
        )

        instance_information_gt = self.preprocessor.prepare_instance_masks_thicken(
            instances_gt,
            instance_semantic_classes_gt,
            geometry,
            dataset_dict["frustum_mask"],
            iso_value=self.iso_value,
            downsample_factor=self.downsample_factor,
        )
        dataset_dict["instance_info_gt"] = instance_information_gt
        dataset_dict["downsample_factor"] = self.downsample_factor

        return dataset_dict

    def __getitem__(self, idx: Union[int, Any]) -> Dict[str, Any]:
        """
        Load and return a single sample from the dataset.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            Dict[str, Any]: Dictionary containing all data for the sample including
                images, segmentations, depth, and optionally 3D volumes.
        """
        batch_seed: Optional[int] = None
        # Allow samplers to pass (idx, seed) objects for batch-consistent resizing.
        if hasattr(idx, "idx") and hasattr(idx, "seed"):
            batch_seed = int(getattr(idx, "seed"))
            idx = int(getattr(idx, "idx"))
        elif isinstance(idx, (tuple, list)) and len(idx) == 2:
            idx, batch_seed = int(idx[0]), int(idx[1])

        sample_id = self.img_ids[int(idx)]
        dataset_dict = {k: v for k, v in self.id2img[sample_id].items()}

        if self.is_training:
            dataset_dict = self._prepare_2d_data(dataset_dict, mode="train", batch_seed=batch_seed)
            if self.enable_3d:
                dataset_dict = self._prepare_3d_data_train(dataset_dict)
        else:
            dataset_dict = self._prepare_2d_data(dataset_dict, mode="test", batch_seed=None)
            if self.enable_3d:
                dataset_dict = self._prepare_3d_data_test(dataset_dict)

        return dataset_dict


class Front3DDataset(BaseNVPanoptix3DDataset):
    """
    Dataset loader for Front3D dataset.

    This class handles loading RGB images,
    depth maps, 2D/3D panoptic segmentations, and volumetric geometry from
    the Front3D dataset format. The dataset uses Y-axis flipping for coordinate
    system alignment.
    """

    _NAME = "front3d"

    def __init__(
        self,
        json_path: str,
        base_dir: str,
        frustum_mask_path: str,
        is_training: bool,
        cfg,
        **kwargs,
    ):
        """
        Constructor for Front3DDataset.

        Args:
            json_path (str): Path to JSON file listing dataset samples with
                scene and image IDs.
            base_dir (str): Root directory containing Front3D data files.
            frustum_mask_path (str): Path to precomputed viewing frustum mask
                file (.npz format).
            is_training (bool): Whether dataset is used for training (True) or
                evaluation (False).
            cfg: Configuration object with dataset parameters.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(cfg, **kwargs)

        self.base_dir = base_dir
        self.json_path = json_path
        self.is_training = is_training
        self.enable_aug = self.gen_aug_weight > 0.0  # Enable only when having generative augmentation data

        self.is_flip = True
        self.downsample_factor = 1
        self.categories = DatasetConstants.CATEGORIES
        self.intrinsic = DatasetConstants.INTRINSIC
        self.frustum_mask = self.load_frustum_mask(frustum_mask_path)

        # Load data meta
        self.get_data_meta()
        self.get_category_mapping()
        self.stuff_classes = DatasetConstants.STUFF_CLASSES
        self.thing_classes = self.get_thing_classes()

        # Re-setup the preprocessor arguments
        self.preprocessor_args["is_flip"] = self.is_flip
        self.preprocessor_args["downsample_factor"] = self.downsample_factor
        self.preprocessor_args["categories"] = self.categories
        self.preprocessor_args["intrinsic"] = self.intrinsic
        self.preprocessor_args["frustum_mask"] = self.frustum_mask
        self.preprocessor_args["stuff_classes"] = self.stuff_classes
        self.preprocessor_args["thing_classes"] = self.thing_classes

        # Setup preprocessor
        self.preprocessor = self.setup_preprocessor(self._NAME)

    def get_data_meta(self):
        """Get paths to load the Front3D dataset."""
        # Load a list of sample dicts and enrich with absolute paths
        with open(self.json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # Process samples in a single loop
        for sample in loaded:
            scene_id, img_id = sample["scene_id"], sample["image_id"]
            base_dir = os.path.join(self.base_dir, "data", scene_id)
            sample_id = f"{scene_id}_{img_id}"

            # Create image metadata for original sample
            img = {
                "height": sample["height"],
                "width": sample["width"],
                "raw_image_id": img_id,
                "image_id": sample_id,
                "file_name": sample.get("file_name", os.path.join(base_dir, f"rgb_{img_id}.png")),
                "depth_label_file_name": os.path.join(base_dir, f"depth_{img_id}.exr"),
                "segm_label_file_name": os.path.join(base_dir, f"segmap_{img_id}.mapped.npz"),
                # Per-sample loss weight (used by SetCriterion). Originals get full weight.
                "loss_weight": 1.0,
            }

            # Add 3D paths if needed
            if self.enable_mp_occ or self.enable_3d:
                img["geometry_file_name"] = os.path.join(base_dir, f"geometry_{img_id}.npz")
            if self.enable_3d:
                img["segm_3d_file_name"] = os.path.join(base_dir, f"segmentation_{img_id}.mapped.npz")
                img["weighting_file_name"] = os.path.join(base_dir, f"weighting_{img_id}.npz")

            # Add original sample
            self.id2img[sample_id] = img
            self.img_ids.append(sample_id)

            # Process augmented samples if enabled
            if self.enable_aug:
                orig_file = os.path.join(base_dir, f"rgb_{img_id}.png")
                ref_path = orig_file.replace("front3d/data", "front3d/augment")
                ref_path = ref_path.split(".")[0]
                aug_files = glob.glob(ref_path + "*.png")

                for aug_file in aug_files:
                    aug_sample_id = f"{sample_id}_{os.path.basename(aug_file).split('.')[0].split('_')[-1]}"

                    # Create augmented image metadata (copy original and update file_name)
                    aug_img = img.copy()
                    aug_img["image_id"] = aug_sample_id
                    aug_img["file_name"] = aug_file
                    # Down-weight augmented samples so they contribute less to training loss.
                    aug_img["loss_weight"] = self.gen_aug_weight

                    self.id2img[aug_sample_id] = aug_img
                    self.img_ids.append(aug_sample_id)

    def load_frustum_mask(self, file_path: str = None) -> torch.Tensor:
        """
        Load pre-defined frustum mask for this dataset.

        Args:
            file_path (str): Path to NPZ file containing frustum mask.

        Returns:
            torch.Tensor: Frustum mask tensor. Returns None if loading fails.
        """
        if not file_path:
            return None
        with np.load(file_path, allow_pickle=False) as npz:
            frustum_mask = torch.from_numpy(npz["mask"]).bool()
        if self.is_flip:
            frustum_mask = torch.flip(frustum_mask, dims=[0, 1])
        return frustum_mask


class Matterport3DDataset(Front3DDataset):
    """
    Dataset loader for Matterport3D dataset.

    This class extends Front3DDataset to handle
    Matterport3D-specific data format, including panoramic images split into
    views, per-image intrinsics, and room masks. Uses 2x downsampling factor
    and specific stuff class IDs.
    """

    _NAME = "matterport"

    def __init__(
        self,
        json_path: str,
        base_dir: str,
        is_training: bool,
        cfg,
        **kwargs,
    ):
        """
        Constructor for Matterport3DDataset.

        Args:
            json_path (str): Path to JSON file listing dataset samples.
            base_dir (str): Root directory containing Matterport3D data files.
            is_training (bool): Whether dataset is used for training (True) or
                evaluation (False).
            cfg: Configuration object with dataset parameters including depth_min
                and depth_max for depth normalization.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        kwargs.pop("frustum_mask_path", None)
        super().__init__(
            json_path=json_path,
            base_dir=base_dir,
            frustum_mask_path=None,
            is_training=is_training,
            cfg=cfg,
            **kwargs
        )

        self.stuff_classes = [10, 11, 12]
        self.is_flip = False
        self.downsample_factor = int(getattr(cfg, "downsample_factor", 2))

        # Re-setup the preprocessor arguments
        self.preprocessor_args["depth_min"] = cfg.depth_min
        self.preprocessor_args["depth_max"] = cfg.depth_max
        self.preprocessor_args["is_matterport"] = True
        self.preprocessor_args["downsample_factor"] = self.downsample_factor
        self.preprocessor_args["is_flip"] = self.is_flip
        self.preprocessor_args["stuff_classes"] = self.stuff_classes

        # Setup preprocessor
        self.preprocessor = self.setup_preprocessor(self._NAME)

    def get_data_meta(self):
        """Get paths to load the Matterport3D dataset."""
        # Load a list of sample dicts and enrich with absolute paths
        with open(self.json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # get paths
        for sample in loaded:
            img = {}
            scene_id, img_id = sample["scene_id"], sample["image_id"]
            name, angle, rot = img_id.split("_")
            image_dir = os.path.join(self.base_dir, "data", scene_id)
            sample_id = f"{scene_id}_{img_id}"

            img["height"] = sample["height"]
            img["width"] = sample["width"]
            img["raw_image_id"] = img_id
            img["image_id"] = sample_id

            img["file_name"] = os.path.join(image_dir, f"{name}_i{angle}_{rot}.jpg")
            img["depth_label_file_name"] = os.path.join(
                self.base_dir, "depth_gen", scene_id, f"{name}_d{angle}_{rot}.png"
            )
            img["intrinsic_label_file_name"] = os.path.join(
                image_dir, f"{name}_intrinsics_{angle}.npy"
            )
            img["room_mask_file_name"] = os.path.join(
                self.base_dir, "room_mask", scene_id, f"{name}_rm{angle}_{rot}.png"
            )
            img["segm_label_file_name"] = os.path.join(
                image_dir, f"{name}_segmap{angle}_{rot}.mapped.npz"
            )

            # 3d paths (for loading frustum masks)
            img["geometry_file_name"] = os.path.join(
                image_dir, f"{name}_geometry{angle}_{rot}.npz"
            )
            if self.enable_3d:
                img["segm_3d_file_name"] = os.path.join(
                    image_dir, f"{name}_segmentation{angle}_{rot}.mapped.npz"
                )
                img["weighting_file_name"] = os.path.join(
                    image_dir, f"{name}_weighting{angle}_{rot}.npz"
                )

            self.id2img[sample_id] = img
            self.img_ids.append(sample_id)


class NVPanoptix3DPredictDataset(BaseNVPanoptix3DDataset):
    """
    Dataset loader for inference on arbitrary RGB images.

    This class provides a simplified dataset interface for running inference
    on a directory of RGB images without ground truth annotations. It loads
    images, applies preprocessing, and uses default camera intrinsics and
    frustum masks for 3D reconstruction.
    """

    _NAME = "predict"

    def __init__(self, image_dir: str, cfg):
        """
        Constructor for NVPanoptix3DPredictDataset.

        Args:
            image_dir (str): Directory path containing input images (.jpg or .png).
            cfg: Configuration object with dataset parameters.
        """
        super().__init__(cfg)
        jpg_files = glob.glob(image_dir + "/*.jpg")
        png_files = glob.glob(image_dir + "/*.png")
        self.img_list = sorted(jpg_files + png_files)

        # Get default intrinsic & frustum mask
        self.intrinsic = DatasetConstants.INTRINSIC
        self.frustum_mask = torch.from_numpy(create_frustum_mask(
            self.intrinsic,
            volume_shape=DatasetConstants.DEFAULT_GRID_DIMS,
            depth_range=DatasetConstants.DEFAULT_DEPTH_RANGE,
            voxel_size=DatasetConstants.DEFAULT_VOXEL_SIZE,
            image_shape=DatasetConstants.DEFAULT_IMG_SIZE,
            z_axis_reversed=False
        ))

        self.preprocessor_args["intrinsic"] = self.intrinsic
        self.preprocessor_args["frustum_mask"] = self.frustum_mask

        # Setup preprocessor
        self.preprocessor = self.setup_preprocessor(self._NAME)

    def __len__(self):
        """Dataset length."""
        return len(self.img_list)

    def __getitem__(self, idx):
        """Get item."""
        filename = self.img_list[idx]
        img_name = Path(filename).stem
        rgb = self.preprocessor.get_image(filename, self.resize_hw)
        height, width = rgb.shape[:2]

        # get transforms
        transforms_2d_list = self.preprocessor.test2d_transforms
        transforms_2d = Compose(transforms_2d_list)
        rgb = transforms_2d(rgb)
        rgb = torch.as_tensor(np.ascontiguousarray(rgb.transpose(2, 0, 1)))

        nopad_image_shape = rgb.shape[-2:]
        rgb, _, _, _, _ = self.preprocessor.apply_size_divisibility_padding(
            rgb, None, None, None, None
        )

        return {
            "image": rgb,
            "intrinsic": self.intrinsic,
            "frustum_mask": self.frustum_mask,
            "image_id": img_name,
            "height": height,
            "width": width,
            "nopad_image_shape": nopad_image_shape,
        }

    def collate_fn(self, batch):
        """Collate items in a batch."""
        out = {}
        for item in batch:
            for k, v in item.items():
                if k not in out:
                    out[k] = []
                out[k].append(v)
        for k, v in out.items():
            if len(v) > 0 and isinstance(v[0], torch.Tensor):
                out[k] = torch.stack(v)
        return out
