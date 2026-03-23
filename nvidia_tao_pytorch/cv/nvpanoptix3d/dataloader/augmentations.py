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

"""NVPanoptix3D augmentations."""

import sys
import numpy as np
import random
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Any, Optional
from fvcore.transforms.transform import (
    CropTransform, HFlipTransform, VFlipTransform,
    NoOpTransform, Transform, TransformList
)

from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.depth_map import DepthMap


def apply_transform(image, mask, T: Transform):
    """
    Apply a transform to both an image and its corresponding mask.

    This utility function applies the same transformation to an image and
    a segmentation mask simultaneously, ensuring they remain aligned after
    geometric transformations like crops, flips, or resizes.

    Args:
        image: Input image array to be transformed. Type depends on the
            transform being applied (typically np.ndarray or PIL.Image).
        mask: Segmentation mask array corresponding to the image. Should
            have the same spatial dimensions as the image.
        T (Transform): A Transform object that implements apply_image and
            apply_segmentation methods.

    Returns:
        tuple: A tuple containing (transformed_image, transformed_mask) with
            both elements having undergone the same transformation.
    """
    transformed_image = T.apply_image(image)
    transformed_mask = T.apply_segmentation(mask)
    return transformed_image, transformed_mask


def apply_transform_multi(image: np.ndarray, segmentation_list: List[np.ndarray], T: Transform):
    """
    Apply a transform to an image and multiple segmentation masks.

    This utility function applies the same transformation to an image and
    a list of segmentation masks, useful when working with multiple annotation
    types (e.g., semantic segmentation, instance masks, depth maps) that all
    need to remain spatially aligned with the image.

    Args:
        image (np.ndarray): Input image array to be transformed.
        segmentation_list (List[np.ndarray]): List of segmentation mask arrays,
            each corresponding to the image. All masks should have compatible
            spatial dimensions with the image.
        T (Transform): A Transform object that implements apply_image and
            apply_segmentation methods.

    Returns:
        tuple: A tuple containing (transformed_image, transformed_masks) where
            transformed_masks is a list of transformed segmentation arrays in
            the same order as the input list.
    """
    transformed_image = T.apply_image(image)
    transformed_masks = [T.apply_segmentation(mask) for mask in segmentation_list]
    return transformed_image, transformed_masks


def get_output_shape(height: int,
                     width: int,
                     short_edge_length: int,
                     max_size: int):
    """
    Compute the output size for aspect-ratio-preserving resize operations.

    This function calculates the target dimensions for resizing an image such
    that the shorter edge matches a specified length while maintaining the
    original aspect ratio. An optional maximum size constraint prevents the
    longer edge from becoming too large.

    Args:
        height (int): Original image height in pixels.
        width (int): Original image width in pixels.
        short_edge_length (int): Target length for the shorter edge of the
            resized image.
        max_size (int): Maximum allowed length for the longer edge. If the
            computed longer edge exceeds this value, both dimensions are
            scaled down proportionally to satisfy the constraint.

    Returns:
        tuple: A tuple (new_height, new_width) containing the computed output
            dimensions as integers, rounded to the nearest pixel.
    """
    size = short_edge_length * 1.0
    scale = size / min(height, width)
    if height < width:
        new_height, new_width = size, scale * width
    else:
        new_height, new_width = scale * height, size
    if max(new_height, new_width) > max_size:
        scale = max_size * 1.0 / max(new_height, new_width)
        new_height = new_height * scale
        new_width = new_width * scale
    new_width = int(new_width + 0.5)
    new_height = int(new_height + 0.5)
    return (new_height, new_width)


class Compose:
    """Class to compose multiple transforms."""

    def __init__(self, transforms: List[Transform]):
        """
        Constructor for Compose class.

        This class allows you to chain multiple transformation operations
        that will be applied sequentially to images, segmentations, and
        coordinates. The composed transforms can be applied as a single
        unified operation.

        Args:
            transforms (List[Transform]): A list of Transform objects to be
                applied sequentially. Each transform in the list will be
                executed in order when the Compose object is called.
        """
        self._raw_transforms = transforms  # Keep reference to original transforms
        self.transforms = TransformList(transforms)

    def apply_coords(self, coords) -> Any:
        """ Apply the compose transform to the coordinates. """
        return self.transforms.apply_coords(coords)

    def apply_image(self, array: np.ndarray) -> torch.Tensor:
        """ Apply the compose transform to the image. """
        return self.transforms.apply_image(array)

    def apply_segmentation(self, array: np.ndarray) -> torch.Tensor:
        """ Apply the compose transform to the segmentation. """
        return self.transforms.apply_segmentation(array)

    def apply_multi(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        random_select: bool = True,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Apply transforms to an image and multiple masks with coordinated random selection.

        This method delegates to individual transforms that support apply_multi (like
        ResizeShortestEdge) for coordinated random selection, ensuring image and all
        masks get the same randomly selected size. For other transforms, it applies
        them sequentially.

        Args:
            image (np.ndarray): Input image array.
            masks (List[np.ndarray]): List of mask arrays (semantic_seg, instance_seg, depth, etc.; stored under "sem_seg"/"inst_seg" keys).
            random_select (bool): Whether to randomly select from size options.

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: Transformed image and list of transformed masks.
        """
        for tfm in self._raw_transforms:
            if hasattr(tfm, 'apply_multi'):
                # Some transforms (e.g., ResizeShortestEdge) accept a deterministic seed
                # to avoid mutating global RNG state.
                try:
                    image, masks = tfm.apply_multi(image, masks, random_select=random_select, seed=seed)
                except TypeError:
                    image, masks = tfm.apply_multi(image, masks, random_select=random_select)
            else:
                image = tfm.apply_image(image)
                masks = [tfm.apply_segmentation(mask) for mask in masks]
        return image, masks

    @property
    def new_size(self) -> Optional[Tuple[int, int]]:
        """Get the output size (W, H) from the first transform that has this attribute."""
        for tfm in self._raw_transforms:
            if hasattr(tfm, 'new_size'):
                return tfm.new_size
        return None

    def __call__(self, array: np.ndarray) -> torch.Tensor:
        """ Apply transformations when calling the compose transform. """
        return self.transforms.apply_image(array)


class ToPILImage(Transform):
    """Transforms numpy array to PIL Image."""

    def apply_coords(self, coords):
        """ Apply the to PIL image transform to the coordinates. """
        return coords

    def apply_image(self, array: np.ndarray) -> Image.Image:
        """ Apply the to PIL image transform to the image. """
        return Image.fromarray(array)

    def apply_segmentation(self, array: np.ndarray) -> Any:
        """ Apply the to PIL image transform to the segmentation. """
        return Image.fromarray(array)


class ModelInputResize(Transform):
    """Resize and pad the model input."""

    def __init__(self, size_divisibility: int = 0, pad_value: float = 0):
        """
        Constructor for ModelInputResize class.

        This transform ensures that the input image dimensions are divisible
        by a specified value by padding the image. This is commonly needed
        for neural networks that require input dimensions to be multiples
        of certain values (e.g., for pooling layers or stride requirements).

        Args:
            size_divisibility (int): The target divisibility for image dimensions.
                If greater than 1, the image will be padded so that both height
                and width are divisible by this value. If 0 or 1, no padding
                is applied. Defaults to 0.
            pad_value (float): The constant value to use for padding pixels.
                Defaults to 0.
        """
        super().__init__()
        self.size_divisibility = size_divisibility
        self.pad_value = pad_value

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        return coords

    def apply_image(
        self, array: torch.Tensor, pad_value: Optional[float] = None
    ) -> torch.Tensor:
        """
        Pad the image tensor to make dimensions divisible by the stride.

        This method computes the required padding to ensure both height and
        width are divisible by size_divisibility, then applies padding to the
        right and bottom edges.

        Args:
            array (torch.Tensor): Input image tensor with shape (..., H, W)
                where H and W are the height and width dimensions.
            pad_value (Optional[float]): Override padding value for this call.
                If None, uses self.pad_value.

        Returns:
            torch.Tensor: Padded image tensor where the last two dimensions
                (height and width) are divisible by size_divisibility.
        """
        if self.size_divisibility <= 1:
            return array

        assert len(array) > 0
        device = array.device
        image_size = [array.shape[-2], array.shape[-1]]

        stride = self.size_divisibility
        max_size = torch.tensor(image_size, device=device)
        max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

        u0 = int(max_size[-1] - image_size[1])
        u1 = int(max_size[-2] - image_size[0])
        padding_size = [0, u0, 0, u1]

        value = pad_value if pad_value is not None else self.pad_value
        return F.pad(array, padding_size, value=value)

    def apply_segmentation(
        self, array: torch.Tensor, pad_value: Optional[float] = None
    ) -> torch.Tensor:
        """ Apply transforms to the segmentation. """
        return self.apply_image(array, pad_value=pad_value)


class Resize(Transform):
    """Resize the image."""

    def __init__(self, size, mode=Image.NEAREST):
        """
        Constructor for Resize class.

        This transform resizes images and segmentation masks to a specified
        target size using PIL Image resize functionality. The resize operation
        can use different interpolation modes depending on the content type.

        Args:
            size (tuple): Target size as (width, height) to which the image
                will be resized.
            mode (PIL.Image resampling filter): The resampling filter to use
                for resizing. Common options include Image.NEAREST (for masks),
                Image.BILINEAR, Image.BICUBIC. Defaults to Image.NEAREST.
        """
        super().__init__()
        self.size = size
        self.mode = mode

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        return coords

    def apply_image(self, array: Image.Image) -> np.ndarray:
        """ Apply transforms to the image. """
        out_width, out_height = self.size
        array = array.resize((out_width, out_height), self.mode)
        return np.array(array)

    def apply_segmentation(self, array: np.ndarray) -> np.ndarray:
        """ Apply transforms to the segmentation. """
        # Convert np.ndarray to PIL Image before resizing, then back to np.ndarray
        ow, oh = self.size
        pil_image = Image.fromarray(array)
        resized = pil_image.resize((ow, oh), self.mode)
        return np.array(resized)


class ToNumpyFromPIL(Transform):
    """Transform PIL Image to numpy array."""

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        return coords

    def apply_image(self, array: Image.Image) -> np.ndarray:
        """ Apply transforms to the image. """
        return np.array(array)

    def apply_segmentation(self, array: np.ndarray) -> np.ndarray:
        """ Apply transforms to the segmentation. """
        return np.array(array)


class ToTensor(Transform):
    """Transform numpy array to tensor."""

    def __init__(self, dtype=None):
        """
        Constructor for ToTensor class.

        This transform converts numpy arrays or existing PyTorch tensors
        to PyTorch tensors with an optional dtype conversion. If the input
        is already a tensor, it will optionally convert the dtype. If the
        input is a numpy array, it creates a new tensor.

        Args:
            dtype (torch.dtype, optional): The desired data type for the
                output tensor. If None, the dtype is inferred from the
                input array. Common values include torch.float32, torch.int64,
                etc. Defaults to None.
        """
        super().__init__()
        self.dtype = dtype

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        return coords

    def apply_image(self, image: np.ndarray) -> torch.Tensor:
        """ Apply transforms to the image. """
        if isinstance(image, torch.Tensor):
            if self.dtype and image.dtype != self.dtype:
                return image.to(dtype=self.dtype)
            return image
        if isinstance(image, np.ndarray):
            image = image.copy()
        if self.dtype:
            return torch.as_tensor(image, dtype=self.dtype)
        return torch.as_tensor(image)

    def apply_segmentation(self, array: np.ndarray) -> torch.Tensor:
        """ Apply transforms to the segmentation. """
        if isinstance(array, torch.Tensor):
            if self.dtype and array.dtype != self.dtype:
                return array.to(dtype=self.dtype)
            return array
        if isinstance(array, np.ndarray):
            array = array.copy()
        if self.dtype:
            return torch.as_tensor(array, dtype=self.dtype)
        return torch.as_tensor(array)


class ToDepthMap(Transform):
    """Transform Torch tensor to DepthMap object."""

    def __init__(self, intrinsic):
        """
        Constructor for ToDepthMap class.

        This transform converts a PyTorch tensor representing depth values
        into a DepthMap object, which encapsulates both the depth information
        and camera intrinsic parameters. The DepthMap object enables various
        3D reconstruction and geometric operations.

        Args:
            intrinsic: Camera intrinsic parameters matrix (typically 3x3 or 4x4)
                containing focal lengths, principal point, and other calibration
                parameters necessary for projecting 2D depth maps into 3D space.
        """
        super().__init__()
        self.intrinsic = intrinsic

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        return coords

    def apply_image(self, tensor: torch.Tensor) -> DepthMap:
        """ Apply transforms to the image. """
        depth_map = DepthMap(tensor.float(), self.intrinsic)
        return depth_map

    def apply_segmentation(self, tensor: torch.Tensor) -> DepthMap:
        """ Apply transforms to the segmentation. """
        depth_map = DepthMap(tensor.float(), self.intrinsic)
        return depth_map


class ToTDF(Transform):
    """Transform distance field tensor to truncated distance field tensor."""

    def __init__(self, truncation):
        """
        Constructor for ToTDF class.

        This transform converts a signed distance field (SDF) into a truncated
        distance field (TDF) by taking the absolute value and clipping it to
        a maximum truncation distance. This is commonly used in 3D reconstruction
        to limit the influence region around surfaces and reduce memory/computation.

        Args:
            truncation (float): The maximum distance value for truncation.
                Distance values beyond this threshold will be clipped to this
                value. Typically measured in the same units as the distance field
                (e.g., meters or voxels).
        """
        super().__init__()
        self.truncation = truncation

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        return coords

    def apply_image(self, distance_field: torch.Tensor) -> torch.Tensor:
        """
        Convert a signed distance field to a truncated distance field.

        This method takes the absolute value of a signed distance field (where
        negative values indicate inside a surface and positive values indicate
        outside) and clips it to a maximum truncation distance.

        Args:
            distance_field (torch.Tensor): Input signed distance field tensor
                with arbitrary shape. Values can be positive or negative.

        Returns:
            torch.Tensor: Truncated distance field with the same shape as input.
                All values are non-negative and clipped to [0, truncation].
        """
        distance_field = torch.abs(distance_field)
        distance_field = torch.clip(distance_field, 0, self.truncation)
        return distance_field

    def apply_segmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        """ Apply transforms to the segmentation. """
        return tensor


class ToBinaryMask(Transform):
    """Transform distance field tensor to binary mask."""

    def __init__(self, threshold: float, compare_function=torch.lt):
        """
        Constructor for ToBinaryMask class.

        This transform converts a continuous-valued tensor (such as a distance
        field) into a binary mask by comparing each value against a threshold.
        The comparison function determines which values are considered True
        in the resulting mask.

        Args:
            threshold (float): The threshold value used for comparison. Values
                are compared against this threshold using the compare_function
                to determine mask membership.
            compare_function (callable): A PyTorch comparison function that takes
                two arguments (tensor, threshold) and returns a boolean mask.
                Common options include torch.lt (less than), torch.gt (greater than),
                torch.le (less or equal), torch.ge (greater or equal). Defaults to
                torch.lt (less than).
        """
        super().__init__()
        self.threshold = threshold
        self.compare_function = compare_function

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        return coords

    def apply_image(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a continuous tensor to a binary mask via thresholding.

        This method applies a comparison operation between the input tensor
        and the threshold value to produce a boolean mask.

        Args:
            tensor (torch.Tensor): Input continuous-valued tensor with arbitrary
                shape. Typically represents distance values, probabilities, or
                other continuous quantities.

        Returns:
            torch.Tensor: Binary mask with the same shape as input, containing
                boolean values (True/False) based on the comparison result.
        """
        mask = self.compare_function(tensor, self.threshold)
        return mask

    def apply_segmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        """ Apply transforms to the segmentation. """
        return tensor


class ResizeTrilinear(Transform):
    """Resize 3D volume."""

    # Required dimensionality for PyTorch 3D operations (N, C, D, H, W)
    REQUIRED_DIM = 5

    def __init__(self, factor: float, mode: str = "trilinear"):
        """
        Constructor for ResizeTrilinear class.

        This transform resizes 3D volumetric data using interpolation. It
        supports both trilinear (smooth) and nearest neighbor (sharp) interpolation
        modes. The resize is performed by scaling all three spatial dimensions
        by the same factor.

        Args:
            factor (float): The scaling factor for resizing. Values greater than 1.0
                will upsample (increase resolution), while values less than 1.0 will
                downsample (decrease resolution). For example, 0.5 will halve the
                dimensions, while 2.0 will double them.
            mode (str): The interpolation mode to use. Supported values are:
                - "trilinear": Smooth trilinear interpolation with aligned corners,
                  suitable for continuous data like distance fields or density.
                - "nearest": Nearest neighbor interpolation, suitable for discrete
                  data like semantic labels or instance IDs.
                Defaults to "trilinear".
        """
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.mode_args = {
            "trilinear": {
                "recompute_scale_factor": False,
                "align_corners": True
            },
            "nearest": {
                "recompute_scale_factor": True
            }
        }

    def apply_image(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Apply trilinear or nearest neighbor resizing to a 3D volume.

        This method resizes 3D volumetric data by the specified scaling factor.
        It handles volumes of varying dimensionality by temporarily expanding them
        to 5D format (batch, channel, depth, height, width) as required by
        PyTorch's F.interpolate, then restoring the original dimensionality.

        Args:
            volume (torch.Tensor): Input 3D volume tensor to be resized. Can have
                various dimensionalities (typically 3D, 4D, or 5D). The spatial
                dimensions will be scaled by the factor specified in __init__.

        Returns:
            torch.Tensor: Resized volume with the same dimensionality as the input.
                The spatial dimensions (last 3 dimensions) are scaled by self.factor,
                while other dimensions remain unchanged.
        """
        old_dim = volume.dim()
        while volume.dim() < self.REQUIRED_DIM:
            volume = volume.unsqueeze(0)

        mode_args = self.mode_args.get(self.mode, {})
        resized = F.interpolate(volume, scale_factor=self.factor, mode=self.mode, **mode_args)

        while resized.dim() > old_dim:
            resized.squeeze_(0)

        return resized

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        return coords

    def apply_segmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        """ Apply transforms to the segmentation. """
        return tensor


class ResizeMax(Transform):
    """Downsample a 3D volume using max pooling."""

    def __init__(self, kernel_size, stride, padding, required_dim: int = 5):
        """
        Constructor for ResizeMax class.

        This transform downsamples 3D volumetric data using 3D max pooling.
        Max pooling selects the maximum value within each pooling window,
        which is useful for preserving strong features while reducing spatial
        resolution. This is particularly effective for binary occupancy grids
        or preserving prominent features.

        Args:
            kernel_size (int or tuple): The size of the pooling window. Can be
                a single integer for cubic windows (e.g., 2 for 2x2x2) or a
                tuple of three integers (depth, height, width) for non-cubic
                windows.
            stride (int or tuple): The stride of the pooling operation, determining
                how much the window moves between applications. Can be a single
                integer or a tuple of three integers. Typically equals kernel_size
                for non-overlapping pooling.
            padding (int or tuple): The amount of zero-padding to add to the input
                volume before pooling. Can be a single integer or a tuple of three
                integers for different padding on each dimension.
            required_dim (int): The required dimensionality of the volume to apply max pooling.
                If the volume has fewer dimensions, it will be expanded to the
                required dimensionality by unsqueezing a dimension of size 1.
                If the volume has more dimensions, it will be downsampled to the
                required dimensionality by squeezing the extra dimensions.
                Defaults to 5 (B, C, D, H, W).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.required_dim = required_dim

    def apply_image(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Downsample a 3D volume using max pooling.

        This method applies 3D max pooling to downsample volumetric data while
        preserving prominent features. It handles volumes of varying dimensionality
        by temporarily expanding to 5D format, applies max pooling, then restores
        the original dimensionality and data type.

        Args:
            volume (torch.Tensor): Input 3D volume tensor to be downsampled.
                Can have various dimensionalities (typically 3D, 4D, or 5D).
                The data type is preserved through the operation.

        Returns:
            torch.Tensor: Downsampled volume with the same dimensionality and
                data type as the input. The spatial dimensions are reduced
                according to the kernel_size and stride parameters.
        """
        old_dtype = volume.type()
        old_dim = volume.dim()

        while volume.dim() < self.required_dim:
            volume = volume.unsqueeze(0)

        volume = volume.type(torch.float)

        resized = F.max_pool3d(volume, self.kernel_size, self.stride, self.padding)
        resized = resized.type(old_dtype)

        while resized.dim() > old_dim:
            resized.squeeze_(0)

        return resized

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        return coords

    def apply_segmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        """ Apply transforms to the segmentation. """
        return tensor


class Absolute(Transform):
    """Absolute transform."""

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        return coords

    def apply_image(self, image):
        """ Apply transforms to the image. """
        return torch.abs(image)

    def apply_segmentation(self, mask):
        """ Apply transforms to the segmentation. """
        return torch.abs(mask)


class RandomCrop(Transform):
    """Random crop transform."""

    def __init__(
        self,
        orig_size: Tuple[int, int],
        crop_size: Tuple[int, int],
        mask: np.ndarray = None,
        max_ratio=0.6,
        ignored_category=0,
        num_retry: int = 10,
    ):
        """
        Constructor for RandomCrop class.

        This transform performs random cropping with optional instance-aware
        sampling. When a segmentation mask is provided, it attempts to find
        crops that contain diverse semantic categories, avoiding crops dominated
        by a single class.

        Args:
            orig_size (Tuple[int, int]): The original image size as (height, width)
                before cropping.
            crop_size (Tuple[int, int]): The desired crop size as (height, width).
                If larger than orig_size in any dimension, that dimension will not
                be cropped.
            mask (np.ndarray, optional): Segmentation mask used to guide crop
                selection. If None, performs simple random cropping. If provided,
                attempts to find crops with diverse instance/category content.
                Defaults to None.
            max_ratio (float): Maximum allowed ratio for the most dominant category
                in a crop. Only applies when mask is provided and max_ratio < 1.0.
                For example, 0.6 means no single category should occupy more than
                60% of the crop area. Defaults to 0.6.
            ignored_category (int): Category ID to ignore when computing category
                diversity (typically background class). Defaults to 0.
            num_retry (int): Number of times to retry finding a valid crop. Defaults to 10.
        """
        super().__init__()
        self.orig_size = orig_size
        self.crop_size = crop_size
        self.max_ratio = max_ratio
        self.ignored_category = ignored_category
        self.num_retry = num_retry
        # Treat constructor mask as (optional) guidance; in this codebase it is usually None.
        self._guidance_mask = mask
        # Cache the deterministic transform sampled for the current image/mask pair.
        self.tfm: Optional[CropTransform] = None

    def _build_transform(self, mask):
        """
        Compute the crop transform based on segmentation guidance.

        This internal method determines the crop location. If a segmentation
        mask is provided, it attempts to find a crop with good category diversity
        by sampling up to 10 random locations and selecting one that meets the
        diversity criteria. If no segmentation is provided, it performs simple
        random cropping.

        Args:
            mask (np.ndarray or None): Segmentation mask to guide crop selection.
                If None, performs uniform random cropping.

        Returns:
            CropTransform: A fvcore CropTransform object configured with the
                selected crop location and size.
        """
        height, width = self.orig_size
        crop_height, crop_width = (min(self.crop_size[0], height), min(self.crop_size[1], width))

        # If no segmentation provided, do simple random crop
        if mask is None:
            height_start = np.random.randint(height - crop_height + 1)
            width_start = np.random.randint(width - crop_width + 1)
            return CropTransform(width_start, height_start, crop_width, crop_height)

        # Try to find a valid crop with instances
        # Attempt up to "num_retry" times to find a good crop
        height_start, width_start = None, None
        for _ in range(self.num_retry):
            height_start_temp = np.random.randint(height - crop_height + 1)
            width_start_temp = np.random.randint(width - crop_width + 1)
            mask_temp = mask[
                height_start_temp: height_start_temp + crop_height,
                width_start_temp: width_start_temp + crop_width
            ]

            # Get unique labels and their counts
            labels, counts = np.unique(mask_temp, return_counts=True)

            # Filter out ignored categories
            if self.ignored_category is not None:
                valid_mask = labels != self.ignored_category
                valid_counts = counts[valid_mask]
            else:
                valid_counts = counts

            # Check 1: Must have at least one valid instance (not just background)
            if len(valid_counts) == 0:
                continue  # Empty crop, try again

            # Check 2: If max_ratio < 1.0, ensure category diversity
            if self.max_ratio < 1.0:
                # Need at least 2 categories for diversity check
                if len(valid_counts) > 1 and np.max(valid_counts) < np.sum(valid_counts) * self.max_ratio:
                    height_start, width_start = height_start_temp, width_start_temp
                    break  # Found a good diverse crop
                if len(valid_counts) == 1:
                    # Only one category, but it's valid - keep as fallback
                    if height_start is None:
                        height_start, width_start = height_start_temp, width_start_temp
            else:
                # No diversity constraint, any crop with instances is good
                height_start, width_start = height_start_temp, width_start_temp
                break

        # If no valid crop found after 10 attempts, use the last attempt or fallback
        if height_start is None:
            height_start = np.random.randint(height - crop_height + 1)
            width_start = np.random.randint(width - crop_width + 1)

        return CropTransform(width_start, height_start, crop_width, crop_height)

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        if self.tfm is None:
            self.tfm = self._build_transform(self._guidance_mask)
        return self.tfm.apply_coords(coords)

    def apply_image(self, image):
        """ Apply transforms to the image. """
        # Re-sample per image (per sample) so this is truly random augmentation.
        self.orig_size = (int(image.shape[0]), int(image.shape[1]))
        self.tfm = self._build_transform(self._guidance_mask)
        return self.tfm.apply_image(image)

    def apply_segmentation(self, mask):
        """ Apply transforms to the segmentation. """
        if self.tfm is None:
            self.orig_size = (int(mask.shape[0]), int(mask.shape[1]))
            self.tfm = self._build_transform(self._guidance_mask)
        return self.tfm.apply_segmentation(mask)


class ResizeShortestEdge(Transform):
    """Resize shortest edge transform."""

    def __init__(
        self,
        orig_size: Tuple[int, int],
        short_edge_length,
        max_size=sys.maxsize,
        interp=cv2.INTER_LINEAR,
        prob=1.0
    ):
        """
        Constructor for ResizeShortestEdge class.

        This transform resizes an image such that its shorter edge matches
        a target length while maintaining aspect ratio. If multiple target
        lengths are provided, one is randomly selected. An optional maximum
        size constraint prevents the longer edge from becoming too large.

        Args:
            orig_size (Tuple[int, int]): The original image size as (height, width)
                before resizing.
            short_edge_length (int or tuple/list of int): Target length(s) for the
                shorter edge. If a single integer, that value is always used. If a
                tuple or list, a random value is sampled from the list for each call.
                If 0, no resizing is performed.
            max_size (int): Maximum allowed size for the longer edge. If the computed
                size exceeds this, the image is scaled down to respect this constraint
                while maintaining aspect ratio. Defaults to sys.maxsize (no limit).
            interp (int): OpenCV interpolation mode for image resizing. Common values
                include cv2.INTER_LINEAR (bilinear), cv2.INTER_CUBIC, cv2.INTER_AREA.
                Defaults to cv2.INTER_LINEAR.
            prob (float): Probability of applying this transform. Currently not
                actively used in the implementation. Defaults to 1.0.
        """
        super().__init__()
        self.orig_size = orig_size
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.interp = interp
        self.prob = prob
        self._update_output_shape()

    def _update_output_shape(self, *, seed: Optional[int] = None):
        """
        Compute the output shape for this resize operation.

        This internal method randomly selects a target short edge length (if
        multiple options are provided) and computes the corresponding output
        dimensions while maintaining aspect ratio and respecting the maximum
        size constraint. The result is stored in self.new_size.
        """
        height, width = self.orig_size
        self.new_size = (width, height)
        if seed is None:
            size = np.random.choice(self.short_edge_length)
        else:
            # Use a local RNG to avoid mutating global RNG state.
            rng = np.random.RandomState(int(seed))
            size = rng.choice(self.short_edge_length)
        if size != 0:
            new_height, new_width = get_output_shape(height, width, size, self.max_size)
            self.new_size = (new_width, new_height)

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        return coords

    def apply_multi(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        random_select: bool = True,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """ Apply transforms to a list of arrays. """
        if random_select:
            self._update_output_shape(seed=seed)
        image = self.apply_image(image)
        masks = [self.apply_segmentation(mask) for mask in masks]
        return image, masks

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the image to the computed target size.

        This method resizes the image to the dimensions computed during
        initialization, which maintain the aspect ratio while matching the
        target short edge length.

        Args:
            image (np.ndarray): Input image array to be resized.
            interp: Unused parameter, kept for interface compatibility.
                The interpolation method from __init__ is used instead.

        Returns:
            np.ndarray: Resized image with dimensions matching self.new_size.
        """
        new_height, new_width = self.new_size
        return cv2.resize(image, (new_width, new_height), interpolation=self.interp)

    def apply_segmentation(self, mask: np.ndarray) -> np.ndarray:
        """
        Resize the segmentation mask to the computed target size.

        This method resizes segmentation masks using nearest-neighbor
        interpolation to preserve discrete label values without introducing
        interpolated/blended values.

        Args:
            mask (np.ndarray): Input segmentation mask array.

        Returns:
            np.ndarray: Resized segmentation mask with dimensions matching
                self.new_size and all original label values preserved.
        """
        new_height, new_width = self.new_size
        return cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)


class ColorAugSSDTransform(Transform):
    """Color augmentation transform."""

    def __init__(
        self,
        img_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        """
        Constructor for ColorAugSSDTransform class.

        This transform applies SSD-style color augmentation to images, including
        random adjustments to brightness, contrast, saturation, and hue. The
        augmentations are applied with 50% probability each, and the order of
        contrast/saturation/hue is randomized to increase diversity.

        Args:
            img_format (str): Format of the input image. Must be either "BGR"
                (OpenCV default) or "RGB" (PIL/matplotlib default). This determines
                whether channel reordering is needed before HSV conversion.
            brightness_delta (int): Maximum absolute change in brightness. The
                actual brightness adjustment is uniformly sampled from
                [-brightness_delta, +brightness_delta]. Applied with 50% probability.
                Defaults to 32.
            contrast_low (float): Lower bound for contrast scaling factor. The
                contrast multiplier is uniformly sampled from [contrast_low, contrast_high].
                Applied with 50% probability. Defaults to 0.5.
            contrast_high (float): Upper bound for contrast scaling factor.
                Defaults to 1.5.
            saturation_low (float): Lower bound for saturation scaling factor. The
                saturation multiplier is uniformly sampled from [saturation_low, saturation_high].
                Applied with 50% probability. Defaults to 0.5.
            saturation_high (float): Upper bound for saturation scaling factor.
                Defaults to 1.5.
            hue_delta (int): Maximum absolute change in hue (in degrees). The
                actual hue adjustment is uniformly sampled from
                [-hue_delta, +hue_delta] and applied modulo 180. Applied with
                50% probability. Defaults to 18.
        """
        super().__init__()
        assert img_format in ["BGR", "RGB"]
        self.is_rgb = img_format == "RGB"
        self.brightness_delta = brightness_delta
        self.contrast_low = contrast_low
        self.contrast_high = contrast_high
        self.saturation_low = saturation_low
        self.saturation_high = saturation_high
        self.hue_delta = hue_delta
        del img_format

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        return coords

    def apply_segmentation(self, mask):
        """ Apply transforms to the segmentation. """
        return mask

    def apply_image(self, image):
        """
        Apply SSD-style color augmentation to an image.

        This method applies a sequence of color transformations including
        brightness, contrast, saturation, and hue adjustments. The order
        of contrast/saturation/hue is randomized (50/50) to increase
        augmentation diversity, while brightness is always applied first.
        RGB images are converted to BGR for processing, then back to RGB.

        Args:
            image (np.ndarray): Input image array. Format should match img_format
                specified in __init__ (either RGB or BGR).

        Returns:
            np.ndarray: Color-augmented image in the same format as input.
                The image has the same shape and dtype (uint8) as the input.
        """
        if self.is_rgb:
            image = image[:, :, [2, 1, 0]]
        image = self.brightness(image)
        if random.randrange(2):
            image = self.contrast(image)
            image = self.saturation(image)
            image = self.hue(image)
        else:
            image = self.saturation(image)
            image = self.hue(image)
            image = self.contrast(image)
        if self.is_rgb:
            image = image[:, :, [2, 1, 0]]
        return image

    def convert(self, image, alpha=1, beta=0):
        """
        Apply linear transformation to image pixel values.

        This helper method performs the transformation: output = alpha * input + beta,
        then clips the result to valid uint8 range [0, 255].

        Args:
            image (np.ndarray): Input image array to be transformed.
            alpha (float): Multiplicative factor (scaling). Used for contrast
                adjustment. Defaults to 1 (no scaling).
            beta (float): Additive constant (offset). Used for brightness
                adjustment. Defaults to 0 (no offset).

        Returns:
            np.ndarray: Transformed image as uint8 array with values clipped
                to [0, 255].
        """
        image = image.astype(np.float32) * alpha + beta
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)

    def brightness(self, image):
        """
        Randomly adjust image brightness.

        This method randomly adds or subtracts a value from all pixel intensities
        with 50% probability. The adjustment amount is uniformly sampled from
        [-brightness_delta, +brightness_delta].

        Args:
            image (np.ndarray): Input image array in BGR format.

        Returns:
            np.ndarray: Image with potentially adjusted brightness. If not
                applied (50% chance), returns the original image unchanged.
        """
        if random.randrange(2):
            return self.convert(
                image, beta=random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return image

    def contrast(self, image):
        """
        Randomly adjust image contrast.

        This method randomly scales all pixel intensities by a multiplication
        factor with 50% probability. The scaling factor is uniformly sampled
        from [contrast_low, contrast_high]. Values > 1.0 increase contrast,
        while values < 1.0 decrease contrast.

        Args:
            img (np.ndarray): Input image array in BGR format.

        Returns:
            np.ndarray: Image with potentially adjusted contrast. If not
                applied (50% chance), returns the original image unchanged.
        """
        if random.randrange(2):
            return self.convert(image, alpha=random.uniform(self.contrast_low, self.contrast_high))
        return image

    def saturation(self, image):
        """
        Randomly adjust image color saturation.

        This method randomly scales the saturation channel in HSV color space
        with 50% probability. The scaling factor is uniformly sampled from
        [saturation_low, saturation_high]. Values > 1.0 increase saturation
        (more vivid colors), while values < 1.0 decrease saturation (more gray).

        Args:
            img (np.ndarray): Input image array in BGR format.

        Returns:
            np.ndarray: Image in BGR format with potentially adjusted saturation.
                If not applied (50% chance), returns the original image unchanged.
        """
        if random.randrange(2):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:, :, 1] = self.convert(
                image[:, :, 1], alpha=random.uniform(self.saturation_low, self.saturation_high)
            )
            return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

    def hue(self, image):
        """
        Randomly shift image hue values.

        This method randomly shifts the hue channel in HSV color space with
        50% probability. The shift amount is uniformly sampled from
        [-hue_delta, +hue_delta] and applied modulo 180 (OpenCV's hue range).
        This effectively rotates colors around the color wheel.

        Args:
            img (np.ndarray): Input image array in BGR format.

        Returns:
            np.ndarray: Image in BGR format with potentially shifted hue values.
                If not applied (50% chance), returns the original image unchanged.
        """
        if random.randrange(2):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:, :, 0] = (
                image[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image


class RandomFlip(Transform):
    """
    Randomly flip the image horizontally or vertically with the given probability.
    """

    def __init__(
        self, orig_size: Tuple[int, int], prob: float = 0.5, *,
        horizontal: bool = True, vertical: bool = False
    ):
        """
        Constructor for RandomFlip class.

        This transform randomly flips images, segmentation masks, and coordinates
        either horizontally or vertically with a specified probability. This is
        a common data augmentation technique to increase dataset diversity and
        improve model robustness to orientation changes.

        Note: Only one flip direction (horizontal or vertical) can be specified
        per instance. To apply both types of flips, create two separate
        RandomFlip instances.

        Args:
            orig_size (Tuple[int, int]): The original image size as (height, width).
                This is needed to correctly compute the flip transformation for
                coordinates.
            prob (float): Probability of applying the flip. Value should be between
                0.0 (never flip) and 1.0 (always flip). Defaults to 0.5.
            horizontal (bool): If True, perform horizontal (left-right) flipping.
                Cannot be True simultaneously with vertical. Defaults to True.
            vertical (bool): If True, perform vertical (top-bottom) flipping.
                Cannot be True simultaneously with horizontal. Defaults to False.

        Raises:
            ValueError: If both horizontal and vertical are True, or if both are False.
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horizontal and vertical. Please create two RandomFlip instances.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horizontal or vertical has to be True!")

        self.orig_size = orig_size
        self.prob = float(prob)
        self.horizontal = bool(horizontal)
        self.vertical = bool(vertical)
        # Cache the deterministic transform sampled for the current image/mask pair.
        self.tfm: Optional[Transform] = None

    def _build_transform(self):
        """
        Build the underlying flip transform based on probability.

        This internal method randomly determines whether to apply the flip
        based on self.prob, and creates the appropriate fvcore transform
        (HFlipTransform, VFlipTransform, or NoOpTransform). The selected
        transform is stored in self._tfm and used by the apply_* methods.
        """
        height, width = self.orig_size
        apply_transform = np.random.rand() < self.prob
        if not apply_transform:
            return NoOpTransform()
        else:
            if self.horizontal:
                return HFlipTransform(width)
            else:  # vertical
                return VFlipTransform(height)

    def apply_image(self, image):
        """ Apply transforms to the image. """
        # Re-sample per image (per sample) so this is truly random augmentation.
        self.orig_size = (int(image.shape[0]), int(image.shape[1]))
        self.tfm = self._build_transform()
        return self.tfm.apply_image(image)

    def apply_coords(self, coords):
        """ Apply transforms to the coordinates. """
        if self.tfm is None:
            self.tfm = self._build_transform()
        return self.tfm.apply_coords(coords)

    def apply_segmentation(self, mask):
        """ Apply transforms to the segmentation. """
        if self.tfm is None:
            self.orig_size = (int(mask.shape[0]), int(mask.shape[1]))
            self.tfm = self._build_transform()
        return self.tfm.apply_segmentation(mask)
