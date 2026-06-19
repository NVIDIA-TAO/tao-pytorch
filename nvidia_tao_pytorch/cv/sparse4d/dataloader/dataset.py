# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse4D Dataset for TAO PyTorch."""

import random
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from os import path as osp
import copy
from typing import List, Dict
from collections import OrderedDict
import sys
import subprocess
import cv2
import math
from importlib import import_module
import tempfile
import time
import tqdm
import json
import pyquaternion
from spatialai_data_utils.constants import FPS
from spatialai_data_utils.eval.detection.data_classes import DetectionConfig
from spatialai_data_utils.eval.tracking.data_classes import TrackingConfig
from spatialai_data_utils.visualization import COLOR_MAP
from spatialai_data_utils.core.boxes.aicity_box import AICityBox
from spatialai_data_utils.converters.nusc_results_to_nvschema import convert_sparse4d_to_nvschema

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.sparse4d.model.box3d import W, L, H, YAW


class Omniverse3DDetTrackDataset(Dataset):
    """Dataset class for Omniverse3D detection and tracking."""

    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    ID_COLOR_MAP = [c[::-1] for c in COLOR_MAP]  # rgb to bgr

    def __init__(self,
                 data_root: str,
                 anno_file: str,
                 classes: List[str],
                 class_config=None,
                 load_interval: int = 1,
                 max_frames: int = -1,
                 max_cameras: int = -1,
                 with_velocity: bool = True,
                 modality: Dict = None,
                 test_mode: bool = False,
                 use_valid_flag: bool = False,
                 augmentation: Dict = None,
                 sequences_split_num: int = 1,
                 with_seq_flag: bool = False,
                 keep_consistent_seq_aug: bool = True,
                 tracking: bool = False,
                 tracking_threshold: float = 0.2,
                 same_scene_in_batch: bool = True,
                 frame_drop_prob: float = 0,
                 fps_drop_prob: float = 0,
                 target_fps_choices: List[int] = None,
                 transforms=None,
                 train_dataset_cfg: Dict = None,
                 lazy_load: bool = False,
                 lazy_load_cache_size: int = 50,
                 pkl_sample_size: int = 0,
                 pkl_cam_counts_path: str = None,
                 eval_dist_fcn: str = "iou_3d",
                 eval_hota: bool = True):
        """Initialize Sparse4D dataset.

        Args:
            data_root: Root directory for data
            anno_file: Path to annotation file
            classes: List of class names
            load_interval: Interval of frames to load
            max_frames: Maximum number of frames to load (-1 for all)
            max_cameras: If >0, randomly subsample to this many cameras per frame (training only)
            with_velocity: Whether to use velocity
            modality: Dict for modality configuration
            test_mode: Whether in test mode
            use_valid_flag: Whether to use valid flags in data
            augmentation: Augmentation configuration
            sequences_split_num: Number of sequences to split into
            with_seq_flag: Whether to use sequence flags
            keep_consistent_seq_aug: Whether to keep consistent augmentation across sequence
            tracking: Whether to enable tracking
            tracking_threshold: Threshold for tracking confidence
            same_scene_in_batch: Whether to keep same scene in batch
            frame_drop_prob: Probability to drop frames
            fps_drop_prob: Probability to downsample a scene to a lower FPS during loading
            target_fps_choices: List of target FPS values for FPS drop augmentation
            transforms: Custom transforms
            train_dataset_cfg: Train dataset configuration
            lazy_load: Defer pkl loading to __getitem__ using a pre-built index
            lazy_load_cache_size: Max number of pkl files held in the LRU cache
            pkl_sample_size: If >0, sample this many pkl files per epoch (balanced by camera count)
            pkl_cam_counts_path: Path to pickle mapping pkl_path -> num_cameras
            eval_dist_fcn: Distance function for evaluation: "center_distance", "iou_3d", or "both"
            eval_hota: Whether to run HOTA tracking evaluation
        """
        self.data_root = data_root
        self.anno_file = anno_file
        self.load_interval = load_interval
        self.max_frames = max_frames
        self.max_cameras = max_cameras
        self.with_velocity = with_velocity
        self.modality = modality if modality else {'use_camera': True, 'use_lidar': False}
        self.test_mode = test_mode
        self.use_valid_flag = use_valid_flag
        self.augmentation = augmentation
        self.transforms = transforms
        self.sequences_split_num = sequences_split_num
        self.with_seq_flag = with_seq_flag
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.tracking = tracking
        self.tracking_threshold = tracking_threshold
        self.same_scene_in_batch = same_scene_in_batch
        self.frame_drop_prob = frame_drop_prob
        self.fps_drop_prob = fps_drop_prob
        self.target_fps_choices = target_fps_choices if target_fps_choices is not None else [30, 20, 15, 10, 6, 5, 3, 2, 1]
        self.train_dataset_cfg = train_dataset_cfg
        self.lazy_load = lazy_load
        self.lazy_load_cache_size = lazy_load_cache_size
        self.pkl_sample_size = pkl_sample_size
        self.pkl_cam_counts_path = pkl_cam_counts_path
        self.eval_dist_fcn = eval_dist_fcn
        self.eval_hota = eval_hota
        self.ann_file = self.anno_file

        # Initialize class-related attributes based on class_config or classes input
        if class_config is not None:
            self._set_global_configs(class_config)
        else:
            self.CLASSES = classes

            # Dynamically create DefaultAttribute
            self.DefaultAttribute = {}
            static_like_keywords = ["box", "pallet", "crate", "basket"]
            for cls_name in self.CLASSES:
                is_static = any(keyword in cls_name.lower() for keyword in static_like_keywords)
                self.DefaultAttribute[cls_name] = f"{cls_name}.{'static' if is_static else 'moving'}"

            # Dynamically create CLASS_RANGE (e.g., default all to 40)
            # use dict.fromkeys()
            self.CLASS_RANGE = dict.fromkeys(self.CLASSES, 40)
            # Dynamically create PRETTY_CLASS_NAMES (PascalCase from various inputs)
            self.PRETTY_CLASS_NAMES = {
                cls_name: "".join(word.capitalize() for word in cls_name.replace('-', '_').split('_'))
                for cls_name in self.CLASSES
            }
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        # Define DetConfigs and TrackConfigs as instance attributes using the now-defined class properties
        self.DetConfigs = {
            "class_range": self.CLASS_RANGE,
            "dist_fcn": "center_distance",
            "dist_ths": [0.5, 1.0, 2.0, 4.0],
            "dist_th_tp": 2.0,
            "min_recall": 0.1,
            "min_precision": 0.1,
            "max_boxes_per_sample": 500,
            "mean_ap_weight": 5
        }
        self.TrackConfigs = {
            "tracking_names": self.CLASSES,
            "pretty_tracking_names": self.PRETTY_CLASS_NAMES,
            "class_range": self.CLASS_RANGE,
            "dist_fcn": "center_distance",
            "dist_th_tp": 2.0,
            "min_recall": 0.1,
            "max_boxes_per_sample": 500,
            "metric_worst": {
                "amota": 0.0, "amotp": 2.0, "recall": 0.0, "motar": 0.0, "mota": 0.0, "motp": 2.0,
                "mt": 0.0, "ml": -1.0, "faf": 500, "gt": -1, "tp": 0.0, "fp": -1.0, "fn": -1.0,
                "ids": -1.0, "frag": -1.0, "tid": 20, "lgd": 20
            },
            "num_thresholds": 40
        }

        # Initialize lazy loading data structures
        if self.lazy_load:
            self._pkl_cache = OrderedDict()
            self._frame_index = []
            self.data_infos = self.load_annotations_lazy(self.ann_file)
        else:
            self.data_infos = self.load_annotations(self.ann_file)

        # Set up flags for batch sampling
        if self.with_seq_flag and self.same_scene_in_batch:
            self._set_sequence_group_flag()

        # Store results for testing & validation
        self.results = []
        self.det3d_eval_configs = DetectionConfig.deserialize(self.DetConfigs)
        self.track3d_eval_configs = TrackingConfig.deserialize(self.TrackConfigs)
        self.det3d_eval_configs_iou3d = DetectionConfig.deserialize(
            self._build_iou3d_det_config()
        )
        self.track3d_eval_configs_iou3d = TrackingConfig.deserialize(
            self._build_iou3d_track_config()
        )

    def __getstate__(self):
        """Exclude unpicklable LRU cache when forking DataLoader workers."""
        state = self.__dict__.copy()
        if "_pkl_cache" in state:
            state["_pkl_cache"] = OrderedDict()
        return state

    def __setstate__(self, state):
        """Restore state from pickle."""
        self.__dict__.update(state)

    def __len__(self):
        """Get dataset length."""
        return len(self.data_infos)

    def _set_global_configs(self, class_config):
        assert len(class_config["CLASS_LIST"]) > 0, "CLASS_LIST in class_config cannot be empty"
        self.CLASSES = class_config["CLASS_LIST"]
        self.CLASS_RANGE = class_config["CLASS_RANGE_DICT"]
        self.DefaultAttribute = class_config["ATTRIBUTE_DICT"]
        self.PRETTY_CLASS_NAMES = class_config["MAP_CLASS_NAMES"]

    def _build_iou3d_det_config(self):
        """Build 3D IoU-based detection eval config.

        dist_ths uses 1-IoU convention: 0.3 means IoU>=0.7, 0.5 means IoU>=0.5.
        """
        return {
            "class_range": self.CLASS_RANGE,
            "dist_fcn": "iou_3d",
            "dist_ths": [0.3, 0.5, 0.7],
            "dist_th_tp": 0.5,
            "min_recall": 0.1,
            "min_precision": 0.1,
            "max_boxes_per_sample": 500,
            "mean_ap_weight": 5,
        }

    def _build_iou3d_track_config(self):
        """Build 3D IoU-based tracking eval config.

        dist_th_tp uses 1-IoU convention: 0.5 means IoU>=0.5 for a TP match.
        """
        return {
            "tracking_names": self.CLASSES,
            "pretty_tracking_names": self.PRETTY_CLASS_NAMES,
            "class_range": self.CLASS_RANGE,
            "dist_fcn": "iou_3d",
            "dist_th_tp": 0.5,
            "min_recall": 0.1,
            "max_boxes_per_sample": 500,
            "metric_worst": {
                "amota": 0.0, "amotp": 1.0, "recall": 0.0, "motar": 0.0,
                "mota": 0.0, "motp": 1.0, "mt": 0.0, "ml": -1.0, "faf": 500,
                "gt": -1, "tp": 0.0, "fp": -1.0, "fn": -1.0, "ids": -1.0,
                "frag": -1.0, "tid": 20, "lgd": 20,
            },
            "num_thresholds": 40,
        }

    def _set_sequence_group_flag(self):
        """Set flag for sequence-based GroupSampler."""
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if "frame_idx" in self.data_infos[idx]:
                if idx != 0 and self.data_infos[idx]["frame_idx"] == 0:
                    # Not first frame and frame_idx == 0 -> new sequence
                    # NOTE: each scene in AICity'24 starts from frame 2
                    curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)
        if self.same_scene_in_batch:
            self.scene_flag = copy.deepcopy(self.flag)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag] / self.sequences_split_num
                                ),
                            )
                        ) + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag), "Error in splitting sequences"
                assert (
                    len(np.bincount(new_flags)) <=
                    len(np.bincount(self.flag)) * self.sequences_split_num
                ), (
                    f"{len(np.bincount(new_flags))} > {len(np.bincount(self.flag))} * {self.sequences_split_num}"
                )
                self.flag = np.array(new_flags, dtype=np.int64)

        for data_idx in range(len(self.data_infos)):
            self.data_infos[data_idx]["group_idx"] = self.flag[data_idx]
        if self.same_scene_in_batch:
            for data_idx in range(len(self.data_infos)):
                self.data_infos[data_idx]["scene_idx"] = self.scene_flag[data_idx]

    def get_augmentation(self):
        """Get augmentation configuration."""
        if self.augmentation is None:
            return None
        H, W = self.augmentation["image_size"]
        H, W = int(H), int(W)
        fH, fW = self.augmentation["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.augmentation["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (int((1 - np.random.uniform(* self.augmentation["bot_pct_lim"])) * newH) - fH)
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.augmentation["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.augmentation["rot_lim"])
            rotate_3d = np.random.uniform(*self.augmentation["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (int((1 - np.mean(self.augmentation["bot_pct_lim"])) * newH) - fH)
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
            "frame_drop_prob": self.frame_drop_prob,
        }
        return aug_config

    def camera_sampling(self, data_infos):
        """Randomly subsample cameras in each frame to max_cameras."""
        for data_info in data_infos:
            if len(data_info["cams"]) > self.max_cameras:
                sampled_cam_names = random.sample(
                    list(data_info["cams"].keys()), self.max_cameras
                )
                data_info["cams"] = {
                    cam_name: data_info["cams"][cam_name]
                    for cam_name in sampled_cam_names
                }
        return data_infos

    def _apply_fps_drop(self, infos):
        """Downsample frames to a randomly chosen lower FPS.

        Uses a periodic keep pattern: keep p of every q frames, where
        p/q = target_fps / base_fps reduced to lowest terms.
        """
        target_fps = random.choice(self.target_fps_choices)
        if target_fps >= FPS:
            return infos, int(FPS)
        g = math.gcd(int(FPS), int(target_fps))
        q = int(FPS) // g
        p = int(target_fps) // g
        return [info for i, info in enumerate(infos) if (i % q) < p], int(target_fps)

    def _get_ann_paths(self, ann_file):
        """Get list of pkl file paths from an annotation file.

        Supports three formats:
            - Directory: all .pkl files in the directory
            - .txt file: one pkl path per line
            - Single .pkl file: returns None (caller loads directly)
        """
        if osp.isdir(ann_file):
            ann_files = sorted([n for n in os.listdir(ann_file) if n.endswith(".pkl")])
            return [osp.join(ann_file, name) for name in ann_files]
        elif ann_file.endswith(".txt"):
            with open(ann_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            return [line.split()[0] for line in lines]
        return None

    def _load_pkl_files(self, ann_paths):
        """Load multiple pkl files and merge their infos."""
        data = {"infos": [], "metadata": {}}
        for ann_idx, ann_path in enumerate(ann_paths):
            start_time = time.time()
            with open(ann_path, 'rb') as f:
                data_scene = pickle.load(f)

            scene_infos = data_scene["infos"]
            if not self.test_mode and self.fps_drop_prob > 0 and random.random() < self.fps_drop_prob:
                scene_infos, local_fps = self._apply_fps_drop(scene_infos)
            else:
                local_fps = int(FPS)

            data["infos"].extend(scene_infos)
            data["metadata"] = data_scene.get("metadata", {})
            if (ann_idx + 1) % 1 == 0 or (ann_idx + 1) == len(ann_paths):
                logging.info(
                    f"[{ann_idx + 1}/{len(ann_paths)}] loaded {len(data['infos'])} frames so far "
                    f"@ {local_fps} fps ({time.time() - start_time:.2f}s last)"
                )
        return data

    def load_annotations(self, ann_file: str) -> List[Dict]:
        """Load annotations from file."""
        logging.info(f"** loading annotations {ann_file} ...")

        ann_paths = self._get_ann_paths(ann_file)
        if ann_paths is not None:
            data = self._load_pkl_files(ann_paths)
        else:
            with open(ann_file, 'rb') as f:
                data = pickle.load(f)

        data_infos = sorted(data["infos"], key=lambda e: (e['scene_name'], e['timestamp']))

        if self.max_frames > 0:
            data_infos = data_infos[:self.max_frames]
        if not self.test_mode and self.max_cameras > 0:
            data_infos = self.camera_sampling(data_infos)

        data_infos = data_infos[:: self.load_interval]
        self.metadata = data.get("metadata", {})
        self.version = self.metadata.get("version", "unknown")
        return data_infos

    def _get_cached_pkl(self, pkl_path):
        """Load a pkl file with LRU caching (single-threaded, no locking needed)."""
        if pkl_path in self._pkl_cache:
            self._pkl_cache.move_to_end(pkl_path)
            return self._pkl_cache[pkl_path]

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self._pkl_cache[pkl_path] = data

        while len(self._pkl_cache) > self.lazy_load_cache_size:
            self._pkl_cache.popitem(last=False)

        return data

    def _get_lazy_index_cache_path(self, ann_file):
        """Get the path to the cached lazy index file."""
        if ann_file.endswith(".txt"):
            return ann_file.replace(".txt", "_lazy_index.pkl")
        elif osp.isdir(ann_file):
            return osp.join(ann_file, "_lazy_index.pkl")
        return None

    def load_annotations_lazy(self, ann_file):
        """Load annotations lazily using a pre-built index file.

        The index file must be created beforehand using tools/build_lazy_index.py.
        """
        logging.info(f"** loading annotations (lazy mode) {ann_file} ...")

        if not (osp.isdir(ann_file) or ann_file.endswith(".txt")):
            logging.info("Single pkl file detected, falling back to normal loading")
            self.lazy_load = False
            return self.load_annotations(ann_file)

        cache_path = self._get_lazy_index_cache_path(ann_file)

        if not cache_path or not osp.exists(cache_path):
            raise FileNotFoundError(
                f"Lazy index cache not found at: {cache_path}\n"
                f"Please build the index first by running:\n"
                f"  python tools/build_lazy_index.py {ann_file}"
            )

        logging.info(f"** Loading lazy index from {cache_path}")
        start_time = time.time()
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
        frame_index = cached_data["frame_index"]
        self.metadata = cached_data.get("metadata", {})
        self.version = self.metadata.get("version", "unknown")
        logging.info(f"** Loaded index with {len(frame_index)} frames in {time.time() - start_time:.2f}s")

        frame_index = sorted(frame_index, key=lambda e: (e["scene_name"], e["timestamp"]))

        if self.max_frames > 0:
            frame_index = frame_index[:self.max_frames]
        frame_index = frame_index[::self.load_interval]

        self._full_frame_index = frame_index

        self._pkl_cam_counts = {}
        if self.pkl_sample_size > 0 and self.pkl_cam_counts_path:
            if osp.exists(self.pkl_cam_counts_path):
                with open(self.pkl_cam_counts_path, "rb") as f:
                    self._pkl_cam_counts = pickle.load(f)
                logging.info(f"** Loaded pkl cam counts: {len(self._pkl_cam_counts)} entries")
            else:
                logging.warning(f"pkl_cam_counts_path not found: {self.pkl_cam_counts_path}")

        if self.pkl_sample_size > 0:
            return self._apply_pkl_sampling(epoch=0)

        self._frame_index = frame_index
        return self._build_data_infos(frame_index)

    def _apply_pkl_sampling(self, epoch=0):
        """Sample a balanced subset of pkl files and rebuild data_infos.

        Args:
            epoch: Used as random seed offset for reproducible per-epoch variation.
        """
        from collections import defaultdict

        full_index = self._full_frame_index
        all_pkls = sorted({e["pkl_path"] for e in full_index})

        missing = [p for p in all_pkls if p not in self._pkl_cam_counts]
        if missing:
            raise KeyError(
                f"{len(missing)} pkl paths in lazy index are not in pkl_cam_counts_path. "
                f"First 5: {missing[:5]}. "
                f"Rebuild _pkl_cam_counts.pkl or use a matching ann_file."
            )

        target = min(self.pkl_sample_size, len(all_pkls))

        by_cams = defaultdict(list)
        for pkl_path in all_pkls:
            by_cams[self._pkl_cam_counts[pkl_path]].append(pkl_path)

        num_cam_groups = len(by_cams)
        base_per_cam = target // num_cam_groups
        leftover = target - base_per_cam * num_cam_groups

        rng = random.Random(42 + epoch)

        alloc = {}
        for nc in sorted(by_cams):
            alloc[nc] = min(base_per_cam, len(by_cams[nc]))

        for nc in sorted(by_cams, key=lambda x: -len(by_cams[x])):
            if leftover <= 0:
                break
            extra = min(len(by_cams[nc]) - alloc[nc], leftover)
            if extra > 0:
                alloc[nc] += extra
                leftover -= extra

        total_alloc = sum(alloc.values())
        if total_alloc < target:
            deficit = target - total_alloc
            for nc in sorted(by_cams, key=lambda x: -len(by_cams[x])):
                extra = min(len(by_cams[nc]) - alloc[nc], deficit)
                if extra > 0:
                    alloc[nc] += extra
                    deficit -= extra
                if deficit <= 0:
                    break

        sampled_pkls = set()
        for nc in sorted(by_cams):
            pool = by_cams[nc]
            n = alloc[nc]
            chosen = pool if n >= len(pool) else rng.sample(pool, n)
            sampled_pkls.update(chosen)

        sampled_index = [e for e in full_index if e["pkl_path"] in sampled_pkls]
        self._frame_index = sampled_index

        cam_dist = defaultdict(int)
        for p in sampled_pkls:
            cam_dist[self._pkl_cam_counts[p]] += 1
        dist_str = ", ".join(f"{nc}cam:{cnt}" for nc, cnt in sorted(cam_dist.items()))

        logging.info(
            f"** pkl sampling (epoch={epoch}): {len(sampled_pkls)}/{len(all_pkls)} pkls, "
            f"{len(sampled_index)} frames [{dist_str}]"
        )

        return self._build_data_infos(sampled_index)

    def resample_pkls(self, epoch):
        """Re-sample pkl files for a new epoch. Called by PklResampleCallback."""
        if self.pkl_sample_size <= 0:
            return

        if self.lazy_load:
            self.data_infos = self._apply_pkl_sampling(epoch)
            self._pkl_cache.clear()
        else:
            return

        self._set_sequence_group_flag()

    def _build_data_infos(self, frame_index):
        """Build lightweight data_infos from frame_index for lazy loading."""
        data_infos = []
        for idx, entry in enumerate(frame_index):
            data_infos.append({
                "scene_name": entry["scene_name"],
                "timestamp": entry["timestamp"],
                "frame_idx": entry["frame_idx"],
                "_lazy_idx": idx,
            })

        num_pkl_files = len({entry["pkl_path"] for entry in frame_index})
        logging.info(f"** lazy loading complete: {len(data_infos)} frames from {num_pkl_files} pkl files")
        return data_infos

    def _get_full_info_lazy(self, index):
        """Get the full info dict for a frame using lazy loading."""
        entry = self._frame_index[index]
        pkl_path = entry["pkl_path"]
        local_idx = entry["local_idx"]

        data = self._get_cached_pkl(pkl_path)
        info = data["infos"][local_idx]

        if not self.test_mode and self.max_cameras > 0 and len(info.get("cams", {})) > self.max_cameras:
            sampled_cam_names = random.sample(list(info["cams"].keys()), self.max_cameras)
            info = copy.copy(info)
            info["cams"] = {name: info["cams"][name] for name in sampled_cam_names}

        return info

    def get_data_info(self, index):
        """Get data info by index."""
        if self.lazy_load:
            info = self._get_full_info_lazy(index)
        else:
            info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info["token"],
            timestamp=info["timestamp"],
            scene_name=info["scene_name"],
        )
        if "group_idx" in info:
            input_dict["group_idx"] = info["group_idx"]
        else:
            input_dict["group_idx"] = -1
        if "scene_idx" in info:
            input_dict["scene_idx"] = info["scene_idx"]
        else:
            input_dict["scene_idx"] = -1

        if self.modality["use_camera"]:
            image_paths = []
            depthmap_paths = []
            lidar2img_rts = []
            cam_intrinsic = []
            cam2world_transforms = []
            cam_names = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                if "depth_map_path" not in cam_info:
                    depthmap_paths.append(None)
                else:
                    if isinstance(cam_info["depth_map_path"], tuple):
                        depth_map_absolute_path = (osp.join(self.data_root, cam_info["depth_map_path"][0]), cam_info["depth_map_path"][1])
                    else:
                        depth_map_absolute_path = osp.join(self.data_root, cam_info["depth_map_path"])
                    depthmap_paths.append(depth_map_absolute_path)
                cam_names.append(cam_type)
                # obtain lidar to image transformation matrix
                cam2world_transform = cam_info['sensor2world_transform']
                intrinsic = copy.deepcopy(cam_info["cam_intrinsic"])
                cam_intrinsic.append(intrinsic)
                cam2world_transforms.append(cam2world_transform)
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ cam2world_transform
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    depth_map_filename=depthmap_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsic,
                    cam_names=cam_names,
                )
            )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict.update(annos)
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info by index."""
        if self.lazy_load:
            info = self._get_full_info_lazy(index)
        else:
            info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0] * gt_velocity.shape[-1]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        if "instance_inds" in info:
            instance_inds = np.array(info["instance_inds"], dtype=np.int64)[mask]
            anns_results["instance_inds"] = instance_inds
        if "asset_inds" in info:
            asset_inds = np.array(info["asset_inds"], dtype=np.int64)[mask]
            anns_results["asset_inds"] = asset_inds

        if "gt_visibility" in info:
            visiblity_scores = []

            for object_visibility_dict in info["gt_visibility"]:
                object_visibility = []
                for cam_type, _ in info["cams"].items():
                    if cam_type not in object_visibility_dict.keys():
                        object_visibility.append(0)
                    else:
                        visibility_score = object_visibility_dict[cam_type]
                        object_visibility.append(visibility_score)
                visiblity_scores.append(object_visibility)

            gt_visibilities = np.array(visiblity_scores, dtype=np.float32)[mask]
            anns_results["gt_visibility"] = gt_visibilities
        return anns_results

    def __getitem__(self, idx):
        """Get item by index."""
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data = self.transforms(data)
        return data

    def update_results(self, results):
        """Update results for validation."""
        self.results.extend(results)

    def clear_results(self):
        """Clear results for validation."""
        self.results = []

    @staticmethod
    def output_to_nusc_box(detection, threshold=None):
        """Convert output to NuScenes box."""
        box3d = detection["boxes_3d"]
        scores = detection["scores_3d"].float().numpy()
        labels = detection["labels_3d"].numpy()
        if "instance_ids" in detection:
            ids = detection["instance_ids"].numpy()
        if "instance_feats" in detection:
            feats = detection["instance_feats"].numpy()
        if "reid_feats" in detection:
            reid_feats = detection["reid_feats"].numpy()
        if "visibility_scores" in detection:
            visibility_scores = detection["visibility_scores"].numpy()
        if threshold is not None:
            if "cls_scores" in detection:
                mask = detection["cls_scores"].float().numpy() >= threshold
            else:
                mask = scores >= threshold
            box3d = box3d[mask]
            scores = scores[mask]
            labels = labels[mask]
            if "instance_ids" in detection:
                ids = ids[mask]
            if "instance_feats" in detection:
                feats = feats[mask]
            if "reid_feats" in detection:
                reid_feats = reid_feats[mask]
            if "visibility_scores" in detection:
                visibility_scores = visibility_scores[mask]
        if hasattr(box3d, "gravity_center"):
            box_gravity_center = box3d.gravity_center.numpy()
            box_dims = box3d.dims.numpy()
            nus_box_dims = box_dims[:, [1, 0, 2]]
            box_yaw = box3d.yaw.numpy()
        else:
            box3d = box3d.numpy()
            box_gravity_center = box3d[..., :3].copy()
            box_dims = box3d[..., 3:6].copy()
            nus_box_dims = box_dims[..., [1, 0, 2]]
            box_yaw = box3d[..., 6].copy()

        box_list = []
        for i in range(len(box3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            if hasattr(box3d, "gravity_center"):
                velocity = (*box3d.tensor[i, 7:9], 0.0)
            else:
                velocity = (*box3d[i, 7:9], 0.0)
            box = AICityBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity,
            )
            if "instance_ids" in detection:
                box.token = ids[i]
            if "instance_feats" in detection:
                box.embed = feats[i]
            if "reid_feats" in detection:
                box.reid_embed = reid_feats[i]
            if "visibility_scores" in detection:
                box.visibility_scores = visibility_scores[i]
            box_list.append(box)
        return box_list

    @staticmethod
    def lidar_nusc_box_to_global(
        info,
        boxes,
        classes,
        eval_configs,
    ):
        """Convert NuScenes box to global box."""
        box_list = []
        for _, box in enumerate(boxes):
            # filter det in ego.
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
            box_list.append(box)
        return box_list

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False, output_nvschema=False):
        """Format bbox results."""
        nusc_annos = {}
        mapped_class_names = self.CLASSES
        for sample_id, det in enumerate(results):
            annos = []
            boxes = Omniverse3DDetTrackDataset.output_to_nusc_box(
                det, threshold=self.tracking_threshold if tracking else None
            )
            sample_token = self.data_infos[sample_id]["token"]
            boxes = Omniverse3DDetTrackDataset.lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
            )
            for _, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = self.DefaultAttribute.get(name, f"{name}.moving")
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = self.DefaultAttribute.get(name, f"{name}.static")

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )

                # detection results
                nusc_anno.update(
                    dict(
                        detection_name=name,
                        detection_score=box.score,
                        attribute_name=attr,
                    )
                )
                # tracking results
                nusc_anno.update(
                    dict(
                        tracking_name=name,
                        tracking_score=box.score,
                        tracking_id=str(box.token),
                    )
                )
                # save instance embeddings
                if tracking and box.embed is not None:
                    nusc_anno.update(
                        dict(
                            embedding=box.embed.tolist(),
                        )
                    )

                # save reid embeddings
                if tracking and box.reid_embed is not None:
                    nusc_anno.update(
                        dict(
                            reid_embedding=box.reid_embed.tolist(),
                        )
                    )
                annos.append(nusc_anno)

            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        os.makedirs(jsonfile_prefix, exist_ok=True)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        with open(res_path, 'w') as f:
            json.dump(nusc_submissions, f)

        if output_nvschema:
            # only convert to nvschema when in tracking mode
            res_nvschema_path = osp.join(jsonfile_prefix, "results_nvschema")
            os.makedirs(res_nvschema_path, exist_ok=True)
            convert_sparse4d_to_nvschema(res_path, res_nvschema_path, self.PRETTY_CLASS_NAMES)
        else:
            res_nvschema_path = None

        return res_path, res_nvschema_path

    def _evaluate_single(
        self, result_path, result_name="img_bbox", tracking=False,
        det_config=None, track_config=None, metric_prefix_override=None,
    ):
        """Evaluate single result.

        Args:
            result_path: Path to results JSON file.
            result_name: Name prefix for metrics.
            tracking: Whether to run tracking evaluation.
            det_config: Override detection config (for IoU3D mode).
            track_config: Override tracking config (for IoU3D mode).
            metric_prefix_override: Override metric prefix string.
        """
        output_dir = osp.join(*osp.split(result_path)[:-1])

        det_cfg = det_config if det_config is not None else self.det3d_eval_configs
        trk_cfg = track_config if track_config is not None else self.track3d_eval_configs

        if not tracking:
            from spatialai_data_utils.eval.detection.evaluate import AIC24DetEval

            eval_output_dir = output_dir
            if metric_prefix_override is not None:
                eval_output_dir = osp.join(output_dir, metric_prefix_override)
                os.makedirs(eval_output_dir, exist_ok=True)

            aic24_det_eval = AIC24DetEval(
                self.data_infos,
                config=det_cfg,
                result_path=result_path,
                output_dir=eval_output_dir,
                verbose=True,
            )
            aic24_det_eval.main(render_curves=False)

            metrics_path = osp.join(eval_output_dir, "metrics_summary.json")
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.error(f"Evaluation failed reading {metrics_path}: {e}")
                return {}

            detail = dict()
            metric_prefix = metric_prefix_override if metric_prefix_override else f"{result_name}_NuScenes"
            for name in self.CLASSES:
                for k, v in metrics["label_aps"][name].items():
                    val = float(f"{v:.4f}")
                    detail[f"{metric_prefix}/{name}_AP_dist_{k}"] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float(f"{v:.4f}")
                    detail[f"{metric_prefix}/{name}_{k}"] = val
                for k, v in metrics["tp_errors"].items():
                    val = float(f"{v:.4f}")
                    detail[f"{metric_prefix}/{self.ErrNameMapping[k]}"] = val

            detail[f"{metric_prefix}/NDS"] = metrics["nd_score"]
            detail[f"{metric_prefix}/mAP"] = metrics["mean_ap"]
        else:
            from spatialai_data_utils.eval.tracking.aic24_eval import AIC24TrackEval

            eval_output_dir = output_dir
            if metric_prefix_override is not None:
                eval_output_dir = osp.join(output_dir, metric_prefix_override)
                os.makedirs(eval_output_dir, exist_ok=True)

            aic24_track_eval = AIC24TrackEval(
                self.data_infos,
                config=trk_cfg,
                result_path=result_path,
                output_dir=eval_output_dir,
                verbose=True,
            )
            aic24_track_eval.main()

            metrics_path = osp.join(eval_output_dir, "metrics_summary.json")
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.error(f"Evaluation failed reading {metrics_path}: {e}")
                return {}

            detail = dict()
            metric_prefix = metric_prefix_override if metric_prefix_override else f"{result_name}_NuScenes"
            tracking_keys = [
                "amota", "amotp", "recall", "motar", "gt", "mota", "motp",
                "mt", "ml", "faf", "tp", "fp", "fn", "ids", "frag", "tid", "lgd",
            ]
            for key in tracking_keys:
                detail[f"{metric_prefix}/{key}"] = metrics[key]

        return detail

    def _evaluate_hota(self, result_path, output_dir, eval_dist_fcn="iou_3d",
                       metric_prefix="img_bbox_HOTA"):
        """Run per-class HOTA tracking evaluation.

        Args:
            result_path: Path to results_nusc.json.
            output_dir: Directory to write HOTA eval artifacts.
            eval_dist_fcn: "iou_3d" for 3D IoU matching, "center_distance" for center distance.
            metric_prefix: Prefix for metric keys in the returned dict.
        """
        from spatialai_data_utils.eval.tracking.hota.hota_eval import (
            evaluate_hota,
            HOTA_FIELDS,
        )

        if self.lazy_load:
            data_infos = [self._get_full_info_lazy(i) for i in range(len(self.data_infos))]
        else:
            data_infos = self.data_infos

        results = evaluate_hota(
            data_infos=data_infos,
            result_path=result_path,
            output_dir=output_dir,
            class_names=self.CLASSES,
            tracker_name="sparse4d",
            verbose=True,
            eval_dist_fcn=eval_dist_fcn,
        )

        detail = {}
        for class_name, metrics in results["per_class"].items():
            if metrics is None:
                continue
            for field in HOTA_FIELDS:
                detail[f"{metric_prefix}/{class_name}_{field}"] = metrics[field] * 100

        for field in HOTA_FIELDS:
            if field in results.get("average", {}):
                detail[f"{metric_prefix}/{field}"] = results["average"][field] * 100

        return detail

    def format_results(
        self,
        results,
        metrics=["detection", "tracking"],
        jsonfile_prefix=None,
        tracking=True,
        output_nvschema=False,
        show=False,
        out_dir=None,
        pipeline=None,
        vis_score_threshold=0.25,
        n_images_col=6,
        viz_down_sample=3,
    ):
        """Format results for evaluation."""
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files, _ = self._format_bbox(
                results, jsonfile_prefix, tracking=tracking, output_nvschema=output_nvschema
            )
        else:
            result_files = dict()
            for name in results[0]:
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files_, result_files_nvschema_ = self._format_bbox(
                    results_, tmp_file_, tracking=tracking, output_nvschema=output_nvschema
                )
                result_files.update(
                    {
                        name: result_files_,
                        f"{name}_nvschema": result_files_nvschema_,
                    }
                )

        if show:
            logging.info(f"\nPlotting results & saving the images to out_dir: {out_dir}\n")
            self.show(results, save_dir=out_dir, show=show,
                      tracking=tracking,
                      pipeline=pipeline,
                      vis_score_threshold=vis_score_threshold,
                      n_images_col=n_images_col, down_sample=viz_down_sample)

        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metrics=["detection"],
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        output_nvschema=False,
        show=False,
        out_dir=None,
        pipeline=None,
        vis_score_threshold=0.25,
        n_images_col=6,
        viz_down_sample=3,
    ):
        """Evaluate results."""
        result_files, tmp_dir = self.format_results(
            results, jsonfile_prefix=jsonfile_prefix, tracking=True,
            output_nvschema=output_nvschema,
            show=show, out_dir=out_dir, pipeline=pipeline,
            vis_score_threshold=vis_score_threshold,
            n_images_col=n_images_col, viz_down_sample=viz_down_sample
        )

        # Determine which distance functions to evaluate
        if self.eval_dist_fcn == "both":
            dist_fcn_modes = ["center_distance", "iou_3d"]
        elif self.eval_dist_fcn in ("center_distance", "iou_3d"):
            dist_fcn_modes = [self.eval_dist_fcn]
        else:
            raise ValueError(
                f"Unknown eval_dist_fcn: {self.eval_dist_fcn}. "
                "Must be 'center_distance', 'iou_3d', or 'both'."
            )

        results_dict = dict()
        for metric in metrics:
            tracking = metric == "tracking"
            if tracking and not self.tracking:
                continue

            for dist_mode in dist_fcn_modes:
                if dist_mode == "iou_3d":
                    det_cfg = self.det3d_eval_configs_iou3d
                    trk_cfg = self.track3d_eval_configs_iou3d
                    amota_prefix = "img_bbox_IoU3D"
                    hota_prefix = "img_bbox_HOTA_IoU3D"
                else:
                    det_cfg = self.det3d_eval_configs
                    trk_cfg = self.track3d_eval_configs
                    amota_prefix = None
                    hota_prefix = "img_bbox_HOTA"

                # AMOTA / detection evaluation
                if isinstance(result_files, dict):
                    for name in result_names:
                        try:
                            ret_dict = self._evaluate_single(
                                result_files[name],
                                tracking=tracking,
                                det_config=det_cfg,
                                track_config=trk_cfg,
                                metric_prefix_override=amota_prefix,
                            )
                        except Exception as e:
                            logging.warning(f"{dist_mode} evaluation failed: {e}")
                            ret_dict = dict()
                        results_dict.update(ret_dict)
                elif isinstance(result_files, str):
                    try:
                        ret_dict = self._evaluate_single(
                            result_files,
                            tracking=tracking,
                            det_config=det_cfg,
                            track_config=trk_cfg,
                            metric_prefix_override=amota_prefix,
                        )
                    except Exception as e:
                        logging.warning(f"{dist_mode} evaluation failed: {e}")
                        ret_dict = dict()
                    results_dict.update(ret_dict)

                # HOTA evaluation (same distance mode as AMOTA)
                if self.eval_hota and tracking:
                    try:
                        if isinstance(result_files, dict):
                            res_path = result_files[result_names[0]]
                        else:
                            res_path = result_files
                        hota_output_dir = osp.join(*osp.split(res_path)[:-1])
                        hota_detail = self._evaluate_hota(
                            res_path, hota_output_dir,
                            eval_dist_fcn=dist_mode,
                            metric_prefix=hota_prefix,
                        )
                        results_dict.update(hota_detail)
                    except Exception as e:
                        logging.warning(f"HOTA ({dist_mode}) evaluation failed: {e}")

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return results_dict

    def show(self, results, save_dir=None, show=False, tracking=False, pipeline=None,
             vis_score_threshold=0.25, n_images_col=6, down_sample=3, write_video=False):
        """Show results.

        Visualization is written as a per-frame JPEG image sequence (royalty-free). The
        legacy MPEG-4 Part 2 ``cv2.VideoWriter`` (fourcc) was removed (TAO-2183 / FF-5):
        OpenCV is built ``WITH_FFMPEG=OFF`` so it can no longer encode video. Set
        ``write_video=True`` to additionally stitch the JPEG sequence into a royalty-free
        VP9/WebM clip via the codec-disabled ffmpeg CLI; it defaults off.
        """
        save_dir = "./" if save_dir is None else save_dir
        if not tracking:
            save_dir = os.path.join(save_dir, "visual_det")
        else:
            save_dir = os.path.join(save_dir, "visual_trk")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        frame_paths = []

        for i, result in enumerate(tqdm.tqdm(results)):
            if "img_bbox" in result.keys():
                result = result["img_bbox"]
            data_info = pipeline(self.get_data_info(i))
            imgs = []

            raw_imgs = data_info["img"]
            lidar2img = data_info["lidar2img"]
            pred_bboxes_3d = result["boxes_3d"][
                result["scores_3d"] > vis_score_threshold
            ]
            pred_bboxes_3d_conf = result["scores_3d"][
                result["scores_3d"] > vis_score_threshold
            ]

            if "instance_ids" in result and self.tracking:
                pred_bboxes_3d_id = result["instance_ids"][
                    result["scores_3d"] > vis_score_threshold
                ]
                pred_bboxes_3d_text = [
                    f"{track_id} ({conf:.2f})"
                    for track_id, conf in zip(pred_bboxes_3d_id, pred_bboxes_3d_conf)
                ]
                color = []
                for obj_id in result["instance_ids"].cpu().numpy().tolist():
                    color.append(
                        self.ID_COLOR_MAP[int(obj_id % len(self.ID_COLOR_MAP))]
                    )
            elif "labels_3d" in result:
                pred_bboxes_3d_id = result["labels_3d"][
                    result["scores_3d"] > vis_score_threshold
                ]
                pred_bboxes_3d_text = [
                    f"{self.CLASSES[label_id]} ({conf:.2f})"
                    for label_id, conf in zip(pred_bboxes_3d_id, pred_bboxes_3d_conf)
                ]
                color = []
                for obj_id in result["labels_3d"].cpu().numpy().tolist():
                    color.append(self.ID_COLOR_MAP[int(obj_id % len(self.ID_COLOR_MAP))])
            else:
                pred_bboxes_3d_text = None
                color = (255, 0, 0)

            # ===== draw boxes_3d to images =====
            for j, img in enumerate(raw_imgs):
                img = img.astype(np.uint8)
                if len(pred_bboxes_3d) != 0:
                    img = Omniverse3DDetTrackDataset.draw_lidar_bbox3d_on_img(
                        pred_bboxes_3d,
                        pred_bboxes_3d_text,
                        img,
                        lidar2img[j],
                        img_metas=None,
                        color=color,
                        thickness=3,
                    )
                imgs.append(img)

            # ===== put text and concat =====
            for j, name in enumerate(data_info["cam_names"]):
                cam_name_viz = f"{name}"
                w, h = cv2.getTextSize(cam_name_viz, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                rect_br = (w + 64, h + 44)
                imgs[j] = cv2.rectangle(
                    imgs[j],
                    (0, 0),
                    rect_br,
                    color=(66, 66, 66),
                    thickness=-1,
                )

                text_x = 32
                text_y = 60
                imgs[j] = cv2.putText(
                    imgs[j],
                    cam_name_viz,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # resize image
                target_height, target_width = imgs[j].shape[:2]
                target_height = int(target_height / down_sample)
                target_width = int(target_width / down_sample)
                imgs[j] = cv2.resize(imgs[j], (target_width, target_height))

            # stack images from different cameras
            n_images = len(imgs)
            n_images_row = n_images // n_images_col + ((n_images % n_images_col) > 0)
            for _ in range(n_images_row * n_images_col - n_images):
                # add empty images
                imgs.append(np.zeros((target_height, target_width, 3), dtype=np.uint8))

            rows = []
            for rid in range(n_images_row):
                row1 = cv2.hconcat(imgs[rid * n_images_col: (rid + 1) * n_images_col])
                rows.append(row1)
            image = cv2.vconcat(rows)

            # ===== save per-frame visualization (royalty-free JPEG image sequence) =====
            frame_path = os.path.join(save_dir, f"{i:09}.jpg")
            cv2.imwrite(frame_path, image)
            frame_paths.append(frame_path)

        # Optional royalty-free video stitch, gated and OFF by default (TAO-2183 / FF-5).
        # The MPEG-4 Part 2 cv2.VideoWriter (fourcc) was removed; OpenCV is built
        # WITH_FFMPEG=OFF so it cannot encode video. When a video is explicitly requested,
        # stitch the JPEG sequence into royalty-free VP9/WebM via the codec-disabled ffmpeg CLI.
        if write_video and frame_paths:
            self._stitch_frames_to_webm(save_dir, fps=int(FPS))

    @staticmethod
    def _stitch_frames_to_webm(save_dir, fps=10, pattern="%09d.jpg", out_name="video.webm"):
        """Stitch the per-frame JPEG sequence into a royalty-free VP9/WebM clip.

        Uses the codec-disabled ffmpeg CLI (libvpx-vp9, no audio) over the ``{i:09}.jpg``
        image sequence written by :meth:`show`. Best-effort: on any failure (ffmpeg absent,
        encoder missing, etc.) the JPEG image sequence in ``save_dir`` remains the usable
        artifact and a warning is logged. No royalty-bearing codec is used (TAO-2183 / FF-5).
        """
        out_path = os.path.join(save_dir, out_name)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", os.path.join(save_dir, pattern),
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuv420p",
            "-an",
            out_path,
        ]
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            detail = e.stderr.decode(errors="ignore") if getattr(e, "stderr", None) else str(e)
            logging.warning(
                "VP9 video stitch failed (%s); per-frame JPEGs in %s remain the artifact.",
                detail, save_dir,
            )

    @staticmethod
    def load_class_config_from_file(config_path):
        """Load class config from file."""
        module_name = os.path.basename(config_path)[:-3]
        if '.' in module_name:
            raise ValueError('Dots are not allowed in config file path.')
        config_dir = os.path.dirname(config_path)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }

        CLASS_MAPPING_DICT = {}
        for cid, c in enumerate(cfg_dict["CLASS_LIST"]):
            CLASS_MAPPING_DICT[c] = cid
        cfg_dict["CLASS_MAPPING_DICT"] = CLASS_MAPPING_DICT

        MAP_SUB_CLASS_TO_CLASS_DICT = {}
        for c in cfg_dict["SUB_CLASS_DICT"].keys():
            for sub_c in cfg_dict["SUB_CLASS_DICT"][c]:
                MAP_SUB_CLASS_TO_CLASS_DICT[sub_c] = c
        cfg_dict["MAP_SUB_CLASS_TO_CLASS_DICT"] = MAP_SUB_CLASS_TO_CLASS_DICT

        return cfg_dict

    @staticmethod
    def box3d_to_corners(box3d):
        """Convert 3D box to 8 corners."""
        if isinstance(box3d, torch.Tensor):
            box3d = box3d.detach().cpu().numpy()
        corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 0.5, 0]
        corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
        corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        rot_cos = np.cos(box3d[:, YAW])
        rot_sin = np.sin(box3d[:, YAW])
        rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
        rot_mat[:, 0, 0] = rot_cos
        rot_mat[:, 0, 1] = -rot_sin
        rot_mat[:, 1, 0] = rot_sin
        rot_mat[:, 1, 1] = rot_cos
        corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
        corners += box3d[:, None, :3]
        return corners

    @staticmethod
    def plot_rect3d_on_img(
        img, num_rects, rect_corners, rect_texts=None, color=(0, 255, 0), thickness=1
    ):
        """Plot the boundary lines of 3D rectangular on 2D images.

        Args:
            img (numpy.array): The numpy array of image.
            num_rects (int): Number of 3D rectangulars.
            rect_corners (numpy.array): Coordinates of the corners of 3D
                rectangulars. Should be in the shape of [num_rect, 8, 2].
            color (tuple[int], optional): The color to draw bboxes.
                Default: (0, 255, 0).
            thickness (int, optional): The thickness of bboxes. Default: 1.
        """
        line_indices = (
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 5),
            (3, 2),
            (3, 7),
            (4, 5),
            (4, 7),
            (2, 6),
            (5, 6),
            (6, 7),
        )
        h, w = img.shape[:2]
        for i in range(num_rects):
            corners = np.clip(rect_corners[i], -1e4, 1e5).astype(np.int32)
            for start, end in line_indices:
                # Check if both start and end points are outside image boundaries
                start_y_outside = corners[start, 1] >= h or corners[start, 1] < 0
                start_x_outside = corners[start, 0] >= w or corners[start, 0] < 0
                start_is_outside = start_y_outside or start_x_outside

                end_y_outside = corners[end, 1] >= h or corners[end, 1] < 0
                end_x_outside = corners[end, 0] >= w or corners[end, 0] < 0
                end_is_outside = end_y_outside or end_x_outside

                if start_is_outside and end_is_outside:
                    continue

                # Draw the line if at least one point is inside
                line_color = color[i] if isinstance(color[0], (list, tuple)) else color
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    line_color,
                    thickness,
                    cv2.LINE_AA,
                )

            # print text for each box
            if rect_texts is not None:
                cv2.putText(
                    img,
                    rect_texts[i],
                    corners[3],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,  # scale
                    (255, 255, 255),
                    3,  # thickness
                    cv2.LINE_AA
                )

        return img.astype(np.uint8)

    @staticmethod
    def draw_lidar_bbox3d_on_img(
        bboxes3d, bboxes3d_text, raw_img, lidar2img_rt, img_metas=None, color=(0, 255, 0), thickness=1
    ):
        """Project the 3D bbox on 2D plane and draw on input image.

        Args:
            bboxes3d (:obj:`LiDARInstance3DBoxes`):
                3d bbox in lidar coordinate system to visualize.
            raw_img (numpy.array): The numpy array of image.
            lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
                according to the camera intrinsic parameters.
            img_metas (dict): Useless here.
            color (tuple[int], optional): The color to draw bboxes.
                Default: (0, 255, 0).
            thickness (int, optional): The thickness of bboxes. Default: 1.
        """
        img = raw_img.copy()
        # corners_3d = bboxes3d.corners
        corners_3d = Omniverse3DDetTrackDataset.box3d_to_corners(bboxes3d)
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate(
            [corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1
        )
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        pts_2d = pts_4d @ lidar2img_rt.T

        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

        return Omniverse3DDetTrackDataset.plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, bboxes3d_text, color, thickness)

    @staticmethod
    def draw_points_on_img(points, img, lidar2img_rt, color=(0, 255, 0), circle=4):
        """Draw points on image."""
        img = img.copy()
        N = points.shape[0]
        points = points.cpu().numpy()
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        pts_2d = (
            np.sum(points[:, :, None] * lidar2img_rt[:3, :3], axis=-1) + lidar2img_rt[:3, 3]
        )
        pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=1e5)
        pts_2d = pts_2d[..., :2] / pts_2d[..., 2:3]
        pts_2d = np.clip(pts_2d, -1e4, 1e4).astype(np.int32)

        for i in range(N):
            for point in pts_2d[i]:
                if isinstance(color[0], int):
                    color_tmp = color
                else:
                    color_tmp = color[i]
                cv2.circle(img, point.tolist(), circle, color_tmp, thickness=-1)
        return img.astype(np.uint8)

    @staticmethod
    def draw_lidar_bbox3d_on_bev(bboxes_3d, bev_size, bev_range=115, color=(255, 0, 0), thickness=3):
        """Draw 3D bounding boxes on BEV."""
        if isinstance(bev_size, (list, tuple)):
            bev_h, bev_w = bev_size
        else:
            bev_h, bev_w = bev_size, bev_size
        bev = np.zeros([bev_h, bev_w, 3])

        marking_color = (127, 127, 127)
        bev_resolution = bev_range / bev_h
        for cir in range(int(bev_range / 2 / 10)):
            cv2.circle(
                bev,
                (int(bev_h / 2), int(bev_w / 2)),
                int((cir + 1) * 10 / bev_resolution),
                marking_color,
                thickness=thickness,
            )
        cv2.line(
            bev,
            (0, int(bev_h / 2)),
            (bev_w, int(bev_h / 2)),
            marking_color,
        )
        cv2.line(
            bev,
            (int(bev_w / 2), 0),
            (int(bev_w / 2), bev_h),
            marking_color,
        )
        if len(bboxes_3d) != 0:
            bev_corners = Omniverse3DDetTrackDataset.box3d_to_corners(bboxes_3d)[:, [0, 3, 4, 7]][
                ..., [0, 1]
            ]
            xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
            ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2
            for obj_idx, (x, y) in enumerate(zip(xs, ys)):
                for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
                    if isinstance(color[0], (list, tuple)):
                        tmp = color[obj_idx]
                    else:
                        tmp = color
                    cv2.line(
                        bev,
                        (int(x[p1]), int(y[p1])),
                        (int(x[p2]), int(y[p2])),
                        tmp,
                        thickness=thickness,
                    )
        return bev.astype(np.uint8)

    @staticmethod
    def draw_lidar_bbox3d(bboxes_3d, imgs, lidar2imgs, color=(255, 0, 0)):
        """Draw 3D bounding boxes on images."""
        vis_imgs = []
        for _, (img, lidar2img) in enumerate(zip(imgs, lidar2imgs)):
            vis_imgs.append(
                Omniverse3DDetTrackDataset.draw_lidar_bbox3d_on_img(bboxes_3d, None, img, lidar2img, color=color)  # bboxes3d_text is None as it's not available here
            )

        num_imgs = len(vis_imgs)
        if num_imgs < 4 or num_imgs % 2 != 0:
            vis_imgs = np.concatenate(vis_imgs, axis=1)
        else:
            vis_imgs = np.concatenate([
                np.concatenate(vis_imgs[:num_imgs // 2], axis=1),
                np.concatenate(vis_imgs[num_imgs // 2:], axis=1)
            ], axis=0)

        bev = Omniverse3DDetTrackDataset.draw_lidar_bbox3d_on_bev(bboxes_3d, vis_imgs.shape[0], color=color)
        vis_imgs = np.concatenate([bev, vis_imgs], axis=1)
        return vis_imgs
