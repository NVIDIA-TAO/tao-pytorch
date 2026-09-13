# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transforms for Sparse4D dataset."""

import json
import os
import re
import time
import warnings
from collections import OrderedDict
from typing import Dict

import cv2
import h5py
import numpy as np
import torch
from PIL import Image

_H5_MAX_RETRIES = 10
_H5_RETRY_DELAY = 0.1


def _read_h5(h5_path, dataset_key, max_retries=_H5_MAX_RETRIES):
    """Read a dataset from an HDF5 file with retry logic for transient I/O errors."""
    for attempt in range(max_retries):
        try:
            with h5py.File(h5_path, "r") as f:
                return f[dataset_key][:]
        except KeyError:
            raise
        except Exception:
            if attempt >= max_retries - 1:
                raise
            time.sleep(_H5_RETRY_DELAY * (attempt + 1))
    raise RuntimeError(
        f"_read_h5: exhausted {max_retries} retries for {h5_path}:{dataset_key}"
    )


def _to_3ch(image):
    """Normalize grayscale, two-channel, or RGBA input to contiguous HWC RGB."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 0:
            raise ValueError(f"image has no channels: shape={image.shape}")
        if channels in (1, 2):
            image = np.repeat(image[:, :, :1], 3, axis=2)
        else:
            image = image[:, :, :3]
    else:
        raise ValueError(f"expected an HxW or HxWxC image, got shape={image.shape}")
    return np.ascontiguousarray(image)


class LoadMultiViewImageFromFiles:
    """Load multi-view images from files."""

    def __init__(self, to_float32=False, color_type="unchanged", h5_file=False):
        """Initialize transform.

        Args:
            to_float32: Whether to convert to float32
            color_type: Color type (unchanged, color, etc.)
            h5_file: Whether to load h5 file
        """
        self.to_float32 = to_float32
        self.color_type = color_type
        self.h5_file = h5_file

    def __call__(self, results: Dict) -> Dict:
        """Call function to load multi-view images.

        Args:
            results: Dict with image filenames

        Returns:
            Dict with loaded images
        """
        filename = results["img_filename"]
        # Load images (shape: h, w, c, num_views)
        if self.h5_file:
            img = []
            for name in filename:
                if isinstance(name, (tuple, list)):
                    try:
                        img.append(_to_3ch(_read_h5(name[0], name[1])))
                    except Exception as e:
                        raise RuntimeError(
                            f"Error loading {name[0]} {name[1]}: {e}"
                        ) from e
                else:
                    img.append(_to_3ch(np.array(Image.open(name).convert("RGB"))))
            img = np.stack(img, axis=-1)
        else:
            img = np.stack(
                [
                    _to_3ch(np.array(Image.open(name).convert("RGB")))
                    for name in filename
                ],
                axis=-1,
            )

        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        # Unravel to list, each image has shape (h, w, c)
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = [img[..., i].shape for i in range(img.shape[-1])]
        results["ori_shape"] = results["img_shape"]

        # Set default values
        results["pad_shape"] = results["img_shape"]
        results["scale_factor"] = 1.0

        return results


class LoadDepthMap:
    """Load depth maps from files."""

    def __init__(self, max_depth=100, default_shape=(1080, 1920), h5_file=False):
        """Initialize transform.

        Args:
            max_depth: Maximum depth value
            default_shape: Default shape of depth map
            h5_file: Whether to load h5 file
        """
        self.max_depth = max_depth
        self.default_shape = default_shape
        self.h5_file = h5_file

    def __call__(self, results: Dict) -> Dict:
        """Call function to load depth maps.

        Args:
            results: Dict with depth map filenames

        Returns:
            Dict with loaded depth maps
        """
        if "depth_map_filename" not in results:
            return results

        filename = results["depth_map_filename"]
        # Load depth maps (shape: h, w, num_views)
        depths = []

        if self.h5_file:
            for name in filename:
                if name is None:
                    depths.append(np.ones(self.default_shape) * -1)
                elif isinstance(name, (tuple, list)):
                    try:
                        depths.append(_read_h5(name[0], name[1]))
                    except KeyError:
                        if "CT1_distill__" in name[0]:
                            alt_h5_path = name[0].replace("CT1_distill__", "")
                            depths.append(_read_h5(alt_h5_path, name[1]))
                        else:
                            raise KeyError(
                                f"Depth key '{name[1]}' not found in {name[0]}"
                            )
                    except Exception as e:
                        raise RuntimeError(
                            f"Error loading {name[0]} {name[1]}: {e}"
                        ) from e
                else:
                    depth = Image.open(name)
                    depths.append(np.array(depth))
            depths = np.stack(depths, axis=-1)
        else:
            for name in filename:
                if name is None:
                    depth = np.ones(self.default_shape) * -1
                else:
                    depth = Image.open(name)
                    depth = np.array(depth)
                depths.append(depth)

            depths = np.stack(depths, axis=-1)
        depths = depths.astype(np.float32)
        # Missing maps use a negative sentinel. Preserve invalid pixels while
        # converting valid millimetre values to metres; clipping the entire
        # array would otherwise turn missing supervision into a fake 0.1 m
        # target.
        valid_depth = np.logical_and(np.isfinite(depths), depths > 0.0)
        depths = np.where(
            valid_depth,
            np.clip(depths / 1000.0, 0.1, self.max_depth),
            -1.0,
        ).astype(np.float32)

        # Convert to list of depth maps
        gt_depth = []
        for i, _ in enumerate(results["lidar2img"]):
            gt_depth.append(depths[..., i])

        results["gt_depth"] = gt_depth
        return results


class LoadLooseToTight2DGT:
    """Attach visible 2D boxes and occlusion weights from per-scene sidecars.

    The output arrays are aligned with ``instance_inds`` and ``cam_names`` and
    remain in original pixel coordinates. This transform must therefore run
    before :class:`ResizeCropFlipImage` and :class:`InstanceNameFilter`.
    """

    def __init__(
        self,
        sidecar_dir,
        frame_regex=None,
        dedup_regex=r"^CT[\w.]+?__",
        cache_size=8,
        warn_on_miss=True,
    ):
        """Initialize the sidecar loader."""
        self.sidecar_dir = os.fspath(sidecar_dir)
        self.frame_regex = re.compile(frame_regex) if frame_regex else None
        self.dedup_regex = dedup_regex
        self.cache_size = int(cache_size)
        self.warn_on_miss = bool(warn_on_miss)
        self._cache = OrderedDict()
        self._warned = set()

    def _frame_id(self, path):
        """Extract the integer frame identifier from an image path."""
        if isinstance(path, (tuple, list)):
            path = path[-1]
        base = os.path.splitext(os.path.basename(str(path)))[0]
        if self.frame_regex is not None:
            match = self.frame_regex.search(base)
            if match is None:
                return None
            return int(match.group(1) if match.groups() else match.group(0))
        numbers = re.findall(r"\d+", base)
        return int(numbers[-1]) if numbers else None

    def _sidecar_path(self, scene):
        """Resolve a scene name to its LTT sidecar path."""
        bare_scene = scene.split("+")[0]
        if self.dedup_regex:
            bare_scene = re.sub(self.dedup_regex, "", bare_scene)
        return os.path.join(self.sidecar_dir, f"{bare_scene}__ltt2dgt.npz")

    def _get_index(self, scene):
        """Load and cache a sidecar as a frame/instance/camera index."""
        if scene in self._cache:
            self._cache.move_to_end(scene)
            return self._cache[scene]

        index = None
        sidecar_path = self._sidecar_path(scene)
        if os.path.isfile(sidecar_path):
            with np.load(sidecar_path, allow_pickle=False) as sidecar:
                metadata = (
                    json.loads(bytes(sidecar["_meta"]).decode("utf-8"))
                    if "_meta" in sidecar
                    else {}
                )
                frame_ids = sidecar["frame_id"]
                instance_ids = sidecar["instance_id"]
                camera_ids = sidecar["cam"]
                boxes = sidecar["box3"]
                occlusion = sidecar["occ"]

            camera_names = metadata.get("cam_names", [])
            index = {}
            for row in range(len(frame_ids)):
                camera_idx = int(camera_ids[row])
                camera_name = (
                    camera_names[camera_idx] if camera_names else str(camera_idx)
                )
                per_frame = index.setdefault(int(frame_ids[row]), {})
                per_instance = per_frame.setdefault(int(instance_ids[row]), {})
                per_instance[camera_name] = (
                    boxes[row],
                    float(occlusion[row]),
                )

        self._cache[scene] = index
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return index

    def __call__(self, results: Dict) -> Dict:
        """Attach sidecar targets to one dataset sample."""
        scene = results.get("scene_name")
        camera_names = results.get("cam_names")
        image_files = results.get("img_filename")
        instance_ids = results.get("instance_inds")
        if scene is None or not camera_names or not image_files or instance_ids is None:
            return results

        num_gt, num_cameras = len(instance_ids), len(camera_names)
        boxes = np.zeros((num_gt, num_cameras, 4), dtype=np.float32)
        occlusion = np.zeros((num_gt, num_cameras), dtype=np.float32)

        index = self._get_index(scene)
        if index is not None and num_gt > 0:
            frame_id = self._frame_id(image_files[0])
            frame_map = index.get(frame_id, {}) if frame_id is not None else {}
            hits = 0
            for gt_idx, instance_id in enumerate(instance_ids):
                per_instance = frame_map.get(int(instance_id))
                if not per_instance:
                    continue
                for camera_idx, camera_name in enumerate(camera_names):
                    record = per_instance.get(camera_name)
                    if record is None:
                        continue
                    boxes[gt_idx, camera_idx] = record[0]
                    occlusion[gt_idx, camera_idx] = record[1]
                    hits += 1
            if hits == 0 and self.warn_on_miss and scene not in self._warned:
                self._warned.add(scene)
                warnings.warn(
                    "LoadLooseToTight2DGT matched no rows for "
                    f"scene={scene!r}, frame_id={frame_id}. Check frame and "
                    "camera naming between the annotation and sidecar.",
                    stacklevel=2,
                )

        results["gt_boxes_2d_visible"] = boxes
        results["gt_occ_weight"] = occlusion
        return results


class LoadRTDETR2D:
    """Attach per-camera RT-DETR detections from a safe NPZ sidecar."""

    def __init__(
        self,
        cache_dir=None,
        cache_path=None,
        dedup_regex=r"^CT[\w.]+?__",
        score_thr=0.0,
        per_class_score_thr=None,
        cache_size=4,
        mark_real=True,
        class_names=None,
    ):
        """Initialize the RT-DETR cache loader."""
        if not cache_dir and not cache_path:
            raise ValueError("LoadRTDETR2D requires cache_dir or cache_path")
        self.cache_dir = os.fspath(cache_dir) if cache_dir else None
        self.cache_path = os.fspath(cache_path) if cache_path else None
        self.dedup_regex = dedup_regex
        self.score_thr = float(score_thr)
        self.per_class_score_thr = dict(per_class_score_thr or {})
        self.cache_size = int(cache_size)
        self.mark_real = bool(mark_real)
        self.class_names = None if class_names is None else list(class_names)
        if self.class_names is not None:
            if not self.class_names or len(set(self.class_names)) != len(
                self.class_names
            ):
                raise ValueError(
                    "LoadRTDETR2D class_names must be non-empty and unique"
                )
        self._cache = OrderedDict()
        self._warned = set()

    def _warn_once(self, key, message):
        """Warn once per missing cache or frame join in each worker."""
        if key in self._warned:
            return
        self._warned.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _path(self, scene):
        """Resolve one scene to its RT-DETR sidecar."""
        if self.cache_path:
            return self.cache_path
        bare_scene = scene.split("+")[0]
        if self.dedup_regex:
            bare_scene = re.sub(self.dedup_regex, "", bare_scene)
        return os.path.join(self.cache_dir, f"{bare_scene}__rtdetr2d.npz")

    def _get_index(self, scene):
        """Load and cache a frame/camera detection index."""
        if scene in self._cache:
            self._cache.move_to_end(scene)
            return self._cache[scene]

        cache_record = None
        cache_path = self._path(scene)
        if os.path.isfile(cache_path):
            with np.load(cache_path, allow_pickle=False) as sidecar:
                metadata = (
                    json.loads(bytes(sidecar["_meta"]).decode("utf-8"))
                    if "_meta" in sidecar
                    else {}
                )
                frame_ids = sidecar["frame_id"]
                camera_ids = sidecar["cam"]
                class_ids = np.asarray(sidecar["class_id"])
                boxes = sidecar["box"]
                scores = np.asarray(sidecar["score"])
                has_valid_frame = "valid_frame_id" in sidecar
                has_valid_camera = "valid_cam" in sidecar
                if has_valid_frame != has_valid_camera:
                    raise ValueError(
                        f"{cache_path} must provide both valid_frame_id and "
                        "valid_cam"
                    )
                if has_valid_frame:
                    valid_frame_ids = sidecar["valid_frame_id"]
                    valid_camera_ids = sidecar["valid_cam"]
                else:
                    # Legacy caches cannot encode explicitly empty frames. Raw
                    # rows still prove that their own frame/camera joins exist;
                    # every other join remains conservatively invalid.
                    valid_frame_ids = frame_ids
                    valid_camera_ids = camera_ids

            if len(valid_frame_ids) != len(valid_camera_ids):
                raise ValueError(
                    f"{cache_path} has mismatched valid_frame_id/valid_cam lengths"
                )

            camera_names = metadata.get("cam_names", [])
            cache_class_names = list(metadata.get("class_names", []))
            if (
                self.class_names is not None and
                cache_class_names and
                cache_class_names != self.class_names
            ):
                raise ValueError(
                    f"{cache_path} class_names/order does not match "
                    f"dataset.classes: cache={cache_class_names!r}, "
                    f"dataset={self.class_names!r}"
                )
            active_class_names = cache_class_names or self.class_names or []
            canonical_shape = None
            virtual_camera = metadata.get("virtual_camera")
            if virtual_camera is not None:
                if not isinstance(virtual_camera, dict):
                    raise ValueError(
                        f"{cache_path} virtual_camera metadata must be an object"
                    )
                width = virtual_camera.get("width")
                height = virtual_camera.get("height")
                try:
                    width_value = int(width)
                    height_value = int(height)
                    dimensions_are_integral = (
                        not isinstance(width, (bool, np.bool_)) and
                        not isinstance(height, (bool, np.bool_)) and
                        float(width) == width_value and
                        float(height) == height_value
                    )
                except (TypeError, ValueError, OverflowError):
                    dimensions_are_integral = False
                if (
                    not dimensions_are_integral or
                    width_value <= 0 or
                    height_value <= 0
                ):
                    raise ValueError(
                        f"{cache_path} virtual_camera width/height must be "
                        "positive integers"
                    )
                canonical_shape = (height_value, width_value)

            def camera_name(camera_id):
                camera_idx = int(camera_id)
                return (
                    str(camera_names[camera_idx]) if camera_names else str(camera_idx)
                )

            valid_pairs = {
                (int(frame_id), camera_name(camera_id))
                for frame_id, camera_id in zip(valid_frame_ids, valid_camera_ids)
            }
            threshold_count = (
                len(active_class_names)
                if active_class_names
                else max(int(class_ids.max()) + 1 if len(class_ids) else 1, 1)
            )
            invalid_class = np.logical_or(class_ids < 0, class_ids >= threshold_count)
            if invalid_class.any():
                invalid_ids = sorted({int(value) for value in class_ids[invalid_class]})
                raise ValueError(
                    f"{cache_path} contains pseudo-label class IDs outside "
                    f"[0, {threshold_count - 1}]: {invalid_ids}"
                )
            thresholds = np.full(threshold_count, self.score_thr, dtype=np.float32)
            for class_name, threshold in self.per_class_score_thr.items():
                if class_name in active_class_names:
                    thresholds[active_class_names.index(class_name)] = float(threshold)
            keep = scores >= thresholds[class_ids]

            index = {}
            for frame_id, camera_id, class_id, box, score in zip(
                frame_ids[keep],
                camera_ids[keep],
                class_ids[keep],
                boxes[keep],
                scores[keep],
            ):
                current_camera_name = camera_name(camera_id)
                record = index.setdefault(int(frame_id), {}).setdefault(
                    current_camera_name, [[], [], []]
                )
                record[0].append(box)
                record[1].append(int(class_id))
                record[2].append(float(score))
            cache_record = (index, valid_pairs, canonical_shape)

        self._cache[scene] = cache_record
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cache_record

    @staticmethod
    def _frame_id(results):
        """Resolve the current frame identifier from stable sample metadata."""
        token = results.get("sample_idx") or results.get("token")
        if isinstance(token, str) and "__" in token:
            token_tail = token.rsplit("__", 1)[-1]
            if token_tail.isdigit():
                return int(token_tail)
        frame_idx = results.get("frame_idx")
        if isinstance(frame_idx, (int, np.integer)):
            return int(frame_idx)
        image_files = results.get("img_filename")
        if image_files:
            image_path = image_files[0]
            if isinstance(image_path, (tuple, list)):
                image_path = image_path[-1]
            numbers = re.findall(r"\d+", os.path.basename(str(image_path)))
            if numbers:
                return int(numbers[-1])
        return None

    def __call__(self, results: Dict) -> Dict:
        """Attach detections to one dataset sample."""
        scene = results.get("scene_name")
        camera_names = results.get("cam_names")
        if scene is None or not camera_names:
            return results

        cache_record = self._get_index(scene)
        frame_id = self._frame_id(results)
        if cache_record is None:
            index, valid_pairs = {}, set()
            cache_path = self._path(scene)
            self._warn_once(
                ("cache", os.path.abspath(cache_path)),
                "LoadRTDETR2D found no cache for "
                f"scene={scene!r}: {cache_path}. Pseudo-label loss will be "
                "skipped for this sample.",
            )
        else:
            index, valid_pairs, canonical_shape = cache_record
            images = results.get("img")
            if canonical_shape is not None and images is not None:
                mismatched_shapes = [
                    tuple(np.asarray(image).shape[:2])
                    for image in images
                    if tuple(np.asarray(image).shape[:2]) != canonical_shape
                ]
                if mismatched_shapes:
                    raise ValueError(
                        "LoadRTDETR2D cache coordinates require image shape "
                        f"{canonical_shape}, but received {mismatched_shapes[0]}. "
                        "Enable dataset.resize_to_canonical_2d and set "
                        "canonical_2d_height/canonical_2d_width to the cache "
                        "virtual_camera dimensions."
                    )

        has_2d_pseudo = False
        if frame_id is None:
            self._warn_once(
                ("frame-id", str(scene)),
                "LoadRTDETR2D could not resolve a frame ID for "
                f"scene={scene!r}. Pseudo-label loss will be skipped for this "
                "sample.",
            )
        elif cache_record is not None:
            missing_cameras = [
                str(camera_name)
                for camera_name in camera_names
                if (frame_id, str(camera_name)) not in valid_pairs
            ]
            has_2d_pseudo = not missing_cameras
            if missing_cameras:
                self._warn_once(
                    ("frame-join", str(scene)),
                    "LoadRTDETR2D matched no valid cache join for "
                    f"scene={scene!r}, frame_id={frame_id}, cameras="
                    f"{missing_cameras!r}. Pseudo-label loss will be skipped "
                    "for this sample.",
                )
        frame_map = index.get(frame_id, {}) if frame_id is not None else {}

        boxes, classes, scores = [], [], []
        for camera_name in camera_names:
            record = frame_map.get(str(camera_name))
            if record and record[0]:
                boxes.append(np.asarray(record[0], dtype=np.float32).reshape(-1, 4))
                classes.append(np.asarray(record[1], dtype=np.int64))
                scores.append(np.asarray(record[2], dtype=np.float32))
            else:
                boxes.append(np.zeros((0, 4), dtype=np.float32))
                classes.append(np.zeros((0,), dtype=np.int64))
                scores.append(np.zeros((0,), dtype=np.float32))

        results["det_boxes_2d"] = boxes
        results["det_classes_2d"] = classes
        results["det_scores_2d"] = scores
        results["has_2d_pseudo"] = has_2d_pseudo
        if self.mark_real and cache_record is not None:
            results["has_3d_gt"] = False
        return results


class ResizeToCanonical2D:
    """Resize explicitly 2D-only images to the calibration-free SV2D size.

    Calibrated 3D samples must keep their source resolution because their
    projection matrices and intrinsics describe that pixel coordinate system.
    The transform therefore requires ``has_3d_gt=False`` and is a no-op when
    the route marker is true or absent.
    """

    def __init__(self, height=1080, width=1920):
        """Initialize the canonical image size."""
        self.height = int(height)
        self.width = int(width)

    def __call__(self, results: Dict) -> Dict:
        """Resize marked 2D-only images, leaving calibrated samples untouched."""
        has_3d_gt = results.get("has_3d_gt")
        if has_3d_gt is None or bool(has_3d_gt):
            return results

        images = results.get("img")
        if images is None:
            return results

        resized_images = []
        for image in images:
            if image.shape[:2] == (self.height, self.width):
                resized_images.append(image)
            else:
                resized_images.append(
                    cv2.resize(
                        image,
                        (self.width, self.height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                )

        results["img"] = resized_images
        image_shapes = [image.shape for image in resized_images]
        results["img_shape"] = image_shapes
        results["ori_shape"] = image_shapes
        results["pad_shape"] = image_shapes
        return results


class InstanceNameFilter:
    """Filter instances by class names."""

    def __init__(self, classes):
        """Initialize transform.

        Args:
            classes: List of class names to keep
        """
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, results: Dict) -> Dict:
        """Filter objects by class names.

        Args:
            results: Dict with instances

        Returns:
            Dict with filtered instances
        """
        if "gt_labels_3d" not in results:
            return results

        gt_labels_3d = results["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )

        results["gt_bboxes_3d"] = results["gt_bboxes_3d"][gt_bboxes_mask]
        results["gt_labels_3d"] = results["gt_labels_3d"][gt_bboxes_mask]

        if "instance_inds" in results:
            results["instance_inds"] = results["instance_inds"][gt_bboxes_mask]

        if "asset_inds" in results:
            results["asset_inds"] = results["asset_inds"][gt_bboxes_mask]

        if "gt_visibility" in results:
            results["gt_visibility"] = results["gt_visibility"][gt_bboxes_mask]

        # Keep LTT targets aligned with the filtered 3D ground-truth rows.
        for key in ("gt_boxes_2d_visible", "gt_occ_weight"):
            if key in results and results[key] is not None:
                results[key] = results[key][gt_bboxes_mask]

        return results


class AICitySparse4DAdaptor:
    """Adapt data format for Sparse4D model."""

    def __call__(self, results: Dict) -> Dict:
        """Format data for Sparse4D model.

        Args:
            results: Dict with data

        Returns:
            Dict with formatted data
        """
        # Convert projection matrices
        if "lidar2img" in results:
            results["projection_mat"] = np.float32(np.stack(results["lidar2img"]))

        # Convert image dimensions
        if "img_shape" in results:
            results["image_wh"] = np.ascontiguousarray(
                np.array(results["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
            )

        if "cam2world_transform" in results:
            results["cam2world_transform"] = np.float32(
                np.stack(results["cam2world_transform"])
            )

        # Process camera intrinsics
        if "cam_intrinsic" in results:
            results["cam_intrinsic"] = np.float32(np.stack(results["cam_intrinsic"]))
            results["focal"] = results["cam_intrinsic"][..., 0, 0]

        # Process instance IDs
        if "instance_inds" in results:
            results["instance_id"] = results["instance_inds"]

        if "asset_inds" in results:
            results["asset_id"] = results["asset_inds"]

        # Process 3D bounding boxes
        if "gt_bboxes_3d" in results:
            # Normalize yaw angle
            results["gt_bboxes_3d"][:, 6] = self.limit_period(
                results["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )

            # Convert to tensor
            results["gt_bboxes_3d"] = torch.tensor(results["gt_bboxes_3d"]).float()

        # Process labels
        if "gt_labels_3d" in results:
            results["gt_labels_3d"] = torch.tensor(results["gt_labels_3d"]).long()

        if "gt_boxes_2d_visible" in results:
            results["gt_boxes_2d_visible"] = torch.as_tensor(
                results["gt_boxes_2d_visible"], dtype=torch.float32
            )

        if "gt_occ_weight" in results:
            results["gt_occ_weight"] = torch.as_tensor(
                results["gt_occ_weight"], dtype=torch.float32
            )

        if "det_boxes_2d" in results:
            results["det_boxes_2d"] = [
                torch.as_tensor(boxes, dtype=torch.float32)
                for boxes in results["det_boxes_2d"]
            ]
        if "det_classes_2d" in results:
            results["det_classes_2d"] = [
                torch.as_tensor(classes, dtype=torch.long)
                for classes in results["det_classes_2d"]
            ]
        if "det_scores_2d" in results:
            results["det_scores_2d"] = [
                torch.as_tensor(scores, dtype=torch.float32)
                for scores in results["det_scores_2d"]
            ]

        # Process images
        imgs = [img.transpose(2, 0, 1) for img in results["img"]]  # HWC -> CHW
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        results["img"] = torch.tensor(imgs)
        return results

    @staticmethod
    def limit_period(
        val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        """Limit angle to the range of [-offset * period, (1-offset) * period].

        Args:
            val: Angle in radians
            offset: Offset for the periodic boundary
            period: Period for the angle

        Returns:
            Angle after limiting to a period
        """
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val

    @staticmethod
    def limit_period_torch(
        val: torch.Tensor, offset: float = 0.5, period: float = np.pi
    ) -> torch.Tensor:
        """PyTorch version of limit_period.

        Args:
            val: Angle in radians
            offset: Offset for the periodic boundary
            period: Period for the angle

        Returns:
            Angle after limiting to a period
        """
        return val - torch.floor(val / period + offset) * period


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms):
        """Initialize transform.

        Args:
            transforms: List of transforms to apply
        """
        self.transforms = transforms

    def __call__(self, results: Dict) -> Dict:
        """Apply all transforms sequentially.

        Args:
            results: Dict with data

        Returns:
            Dict with transformed data
        """
        for t in self.transforms:
            results = t(results)
            if results is None:
                return None
        return results
