# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Augmentation functions for Sparse4D dataset."""

import cv2
import numpy as np
from numpy import random
from PIL import Image
from scipy.ndimage import zoom
from typing import Dict, Tuple


LTT_BOX2D_KEY = "gt_boxes_2d_visible"


def transform_boxes_xyxy(boxes_xyxy, transform, img_w=None, img_h=None):
    """Apply an affine homography to xyxy boxes and return their new AABBs."""
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes_xyxy.size == 0:
        return boxes_xyxy.astype(np.float32).reshape(-1, 4)

    matrix = np.asarray(transform, dtype=np.float64)[:3, :3]
    x_min, y_min, x_max, y_max = boxes_xyxy.T
    x_corners = np.stack([x_min, x_max, x_min, x_max], axis=1)
    y_corners = np.stack([y_min, y_min, y_max, y_max], axis=1)
    corners = np.stack([x_corners, y_corners, np.ones_like(x_corners)], axis=1)
    transformed = np.einsum("ij,njk->nik", matrix, corners)
    scale = np.clip(transformed[:, 2, :], 1e-6, None)
    x_out = transformed[:, 0, :] / scale
    y_out = transformed[:, 1, :] / scale
    boxes_out = np.stack(
        [x_out.min(1), y_out.min(1), x_out.max(1), y_out.max(1)], axis=1
    )
    if img_w is not None:
        boxes_out[:, [0, 2]] = np.clip(boxes_out[:, [0, 2]], 0.0, float(img_w))
    if img_h is not None:
        boxes_out[:, [1, 3]] = np.clip(boxes_out[:, [1, 3]], 0.0, float(img_h))
    return boxes_out.astype(np.float32)


def normalize_image(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.

    Args:
        img: Image to normalize (numpy array)
        mean: Mean values for normalization (numpy array)
        std: Standard deviation values for normalization (numpy array)
        to_rgb: Whether to convert BGR to RGB (not used, kept for API compatibility)

    Returns:
        Normalized image
    """
    # Apply normalization
    assert (
        img.dtype != np.uint8
    ), f"img.dtype: {img.dtype} != np.uint8, Image is not uint8"
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace

    return img


class ResizeCropFlipImage:
    """Resize, crop and flip images."""

    def __call__(self, results: Dict) -> Dict:
        """Apply resize, crop and flip to images.

        Args:
            results: Dict with images and augmentation config

        Returns:
            Dict with transformed images
        """
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results

        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        has_ltt_boxes = LTT_BOX2D_KEY in results and results[LTT_BOX2D_KEY] is not None
        ltt_boxes = (
            np.asarray(results[LTT_BOX2D_KEY], dtype=np.float32)
            if has_ltt_boxes
            else None
        )
        has_detections = (
            "det_boxes_2d" in results and results["det_boxes_2d"] is not None
        )

        for i in range(N):
            img, mat = self._img_transform(np.uint8(imgs[i]), aug_config)
            transformed_image = np.array(img).astype(np.float32)
            new_imgs.append(transformed_image)

            if "lidar2img" in results:
                results["lidar2img"][i] = mat @ results["lidar2img"][i]

            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= aug_config["resize"]

            image_height, image_width = transformed_image.shape[:2]
            if has_ltt_boxes and ltt_boxes.shape[0] > 0:
                ltt_boxes[:, i, :] = transform_boxes_xyxy(
                    ltt_boxes[:, i, :],
                    mat,
                    img_w=image_width,
                    img_h=image_height,
                )
            if has_detections and len(results["det_boxes_2d"][i]) > 0:
                results["det_boxes_2d"][i] = transform_boxes_xyxy(
                    results["det_boxes_2d"][i],
                    mat,
                    img_w=image_width,
                    img_h=image_height,
                )

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        if has_ltt_boxes:
            results[LTT_BOX2D_KEY] = ltt_boxes

        return results

    def _img_transform(
        self, img: np.ndarray, aug_configs: Dict
    ) -> Tuple[Image.Image, np.ndarray]:
        """Transform a single image.

        Args:
            img: Image to transform
            aug_configs: Augmentation configuration

        Returns:
            Transformed image and transform matrix
        """
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = aug_configs.get("resize_dims", (int(W * resize), int(H * resize)))
        crop = aug_configs.get("crop", [0, 0, resize_dims[0], resize_dims[1]])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        # Add frame drop augmentation
        frame_drop_prob = aug_configs.get("frame_drop_prob", 0)
        if frame_drop_prob > 0 and random.random() < frame_drop_prob:
            img = np.zeros_like(img)

        # Save original dtype for possible restoration
        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_value = img.max()
            scale = 255 / (max_value - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)

        # Apply transforms
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)

        # Restore original dtype if needed
        if origin_dtype != np.uint8:
            img = img / scale + min_value

        # Calculate transformation matrix
        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])

        if flip:
            flip_matrix = np.array([[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]])
            transform_matrix = flip_matrix @ transform_matrix

        # Apply rotation
        rotate_rad = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate_rad), np.sin(rotate_rad), 0],
                [-np.sin(rotate_rad), np.cos(rotate_rad), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix

        # Create 4x4 matrix for 3D transforms
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix

        return img, extend_matrix


class ResizeCropFlipMultiScaleDepthMap:
    """Resize, crop, flip and downsample depth maps."""

    def __init__(self, downsample=[4, 8, 16]):
        """Initialize transform.

        Args:
            downsample: Downsample factors for depth maps
        """
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample

    def __call__(self, results: Dict) -> Dict:
        """Apply transforms to depth maps.

        Args:
            results: Dict with depth maps and augmentation config

        Returns:
            Dict with transformed depth maps
        """
        aug_config = results.get("aug_config", None)
        gt_depths = results.get("gt_depth", None)

        if any([not aug_config, not gt_depths]):
            return results

        N = len(gt_depths)
        new_gt_depths = []

        for i in range(N):
            gt_depth = self._img_transform(gt_depths[i], aug_config)

            for j, downsample in enumerate(self.downsample):
                if len(new_gt_depths) < j + 1:
                    new_gt_depths.append([])

                # Apply downsampling with zoom
                gt_depth_scale = zoom(gt_depth, 1 / downsample, order=1)
                # Mark invalid depth values
                gt_depth_scale = np.where(gt_depth_scale < 0.5, -1, gt_depth_scale)
                new_gt_depths[j].append(gt_depth_scale)

        results["gt_depth"] = [np.stack(x) for x in new_gt_depths]
        return results

    def _img_transform(self, img: np.ndarray, aug_configs: Dict) -> np.ndarray:
        """Transform a single depth map.

        Args:
            img: Depth map to transform
            aug_configs: Augmentation configuration

        Returns:
            Transformed depth map
        """
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, resize_dims[0], resize_dims[1]])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        # Apply transforms
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate, fillcolor=0)
        img = np.array(img).astype(np.float32)

        return img


class BBoxRotation:
    """Apply rotation to 3D bounding boxes."""

    def __call__(self, results: Dict) -> Dict:
        """Apply 3D rotation to bounding boxes.

        Args:
            results: Dict with bounding boxes and augmentation config

        Returns:
            Dict with rotated bounding boxes
        """
        if "aug_config" not in results or "rotate_3d" not in results["aug_config"]:
            return results

        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        # Create rotation matrix
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        # Apply rotation to camera projections
        if "lidar2img" in results:
            num_view = len(results["lidar2img"])
            for view in range(num_view):
                results["lidar2img"][view] = results["lidar2img"][view] @ rot_mat_inv

        # This matrix is named after its source annotation field but follows the
        # same ego-to-camera convention as lidar2img. Rotate its ego frame too.
        if "cam2world_transform" in results:
            for view in range(len(results["cam2world_transform"])):
                results["cam2world_transform"][view] = (
                    results["cam2world_transform"][view] @ rot_mat_inv
                )

        # Apply rotation to global transform
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv

        # Apply rotation to bounding boxes
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = self.box_rotate(results["gt_bboxes_3d"], angle)

        return results

    @staticmethod
    def box_rotate(bbox_3d: np.ndarray, angle: float) -> np.ndarray:
        """Rotate 3D bounding boxes.

        Args:
            bbox_3d: 3D bounding boxes
            angle: Rotation angle

        Returns:
            Rotated 3D bounding boxes
        """
        if len(bbox_3d) == 0:
            return bbox_3d

        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array([[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]])

        # Rotate center coordinates
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T

        # Adjust yaw angle
        bbox_3d[:, 6] += angle

        # Rotate velocity components if present
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]

        return bbox_3d


class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        """Initialize transform."""
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        """Represent the class."""
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


class NormalizeMultiviewImage:
    """Normalize multi-view images."""

    def __init__(self, mean, std, to_rgb=True):
        """Initialize transform.

        Args:
            mean: Mean values for normalization (numpy array)
            std: Standard deviation values for normalization (numpy array)
            to_rgb: Whether to convert BGR to RGB (not used, kept for API compatibility)
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results: Dict) -> Dict:
        """Normalize images.

        Args:
            results: Dict with images

        Returns:
            Dict with normalized images
        """
        if "img" not in results:
            return results

        # Use the normalize_image function
        results["img"] = [
            normalize_image(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]

        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results
