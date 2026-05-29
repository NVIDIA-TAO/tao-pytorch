# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Transformation for Depth Network."""

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose


def build_mono_transforms(aug_config, split='train', resize_target=True):
    """Build Augmentations.

    Args:
        aug_config (dict): augmentation configuration.
        split (str): data split (train, val, eval, infer).
        resize_target (bool): whether to resize the target.

    Returns:
        transforms (Compose): Final built transforms.

    """
    net_h, net_w = aug_config['crop_size']
    mean = aug_config["input_mean"]
    std = aug_config["input_std"]
    transform = Compose([
        Resize(
            width=net_w,
            height=net_h,
            resize_target=resize_target,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC
        ),
        NormalizeImage(mean=mean, std=std),
        PrepareForNet(),
    ] + ([Crop(net_h)] if split == 'train' else []))
    return transform


class Resize(object):
    """ Resize sample to given size (width, height). """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        """constraint size to multiple of self.__multiple_of .

        Args:
            x (int): size to constrain.
            min_val (int): minimum value.
            max_val (int): maximum value.

        Returns:
            y (int): constrained size.
        """
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        """compute final size to resize.

        Args:
            width (int): width of the image.
            height (int): height of the image.

        Returns:
            new_width (int): new width.
            new_height (int): new height.
        """
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        """__call__ for Resize.

        Args:
            sample (dict): sample dictionary.

        Returns:
            sample (dict): sample dictionary after Resize.
        """
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method
        )

        if "right_image" in sample:
            sample["right_image"] = cv2.resize(
                sample["right_image"],
                (width, height),
                interpolation=self.__image_interpolation_method
            )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

            if "semseg_mask" in sample:
                sample["semseg_mask"] = F.interpolate(torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...], (height, width), mode='nearest').numpy()[0, 0]

            if "valid_mask" in sample:
                sample["valid_mask"] = cv2.resize(
                    sample["valid_mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

        return sample


class NormalizeImage(object):
    """Normalize image by given mean and std."""

    def __init__(self, mean, std):
        """__init__ for NormalizeImage.

        Args:
            mean (list): mean of the image.
            std (list): std of the image.
        """
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        """__call__ for NormalizeImage.

        Args:
            sample (dict): sample dictionary.

        Returns:
            sample (dict): sample dictionary after NormalizeImage.
        """
        sample["image"] = ((sample["image"] / 255) - self.__mean) / self.__std
        if "right_image" in sample:
            sample["right_image"] = ((sample["right_image"] / 255) - self.__mean) / self.__std
        return sample


class PrepareForNet(object):
    """ Prepare sample for usage as network input. """

    def __init__(self):
        """__init__ for PrepareForNet. """

    def __call__(self, sample):
        """__call__ for PrepareForNet.

        Args:
            sample (dict): sample dictionary.

        Returns:
            sample (dict): sample dictionary after PrepareForNet.
        """
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)
        if "right_image" in sample:
            image = np.transpose(sample["right_image"], (2, 0, 1))
            sample["right_image"] = np.ascontiguousarray(image).astype(np.float32)

        if "valid_mask" in sample:
            sample["valid_mask"] = sample["valid_mask"].astype(np.float32)
            sample["valid_mask"] = np.ascontiguousarray(sample["valid_mask"])

        if "disparity" in sample:
            if len(sample['disparity'].shape) > 2:
                sample['disparity'] = sample["disparity"][:, :, 0:1]
                disparity = np.transpose(sample["disparity"], (2, 0, 1))
            else:
                disparity = sample['disparity'][None, :, :]

            sample["disparity"] = np.ascontiguousarray(disparity).astype(np.float32)

        if "depth" in sample:
            if len(sample['depth'].shape) > 2:
                sample['depth'] = sample["depth"][:, :, 0:1]
                depth = np.transpose(sample["depth"], (2, 0, 1))
            else:
                depth = sample['depth'][None, :, :]
            sample["depth"] = np.ascontiguousarray(depth).astype(np.float32)

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        return sample


class Crop(object):
    """Crop sample for batch-wise training. Image is of shape CxHxW."""

    def __init__(self, size):
        """__init__ for Crop.

        Args:
            size (int): size of the crop.
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample):
        """__call__ for Crop.

        Args:
            sample (dict): sample dictionary.

        Returns:
            sample (dict): sample dictionary after Crop.
        """
        h, w = sample['image'].shape[-2:]

        assert h >= self.size[0] and w >= self.size[1], \
            f'Crop size needs to be smaller than image size {self.size} < {h, w}'

        h_start = np.random.randint(0, h - self.size[0] + 1)
        w_start = np.random.randint(0, w - self.size[1] + 1)
        h_end = h_start + self.size[0]
        w_end = w_start + self.size[1]

        sample['image'] = sample['image'][:, h_start: h_end, w_start: w_end]
        if "right_image" in sample:
            sample['right_image'] = sample['right_image'][:, h_start: h_end, w_start: w_end]

        if "disparity" in sample:
            sample["disparity"] = sample["disparity"][:, h_start: h_end, w_start: w_end]

        if "depth" in sample:
            sample["depth"] = sample["depth"][:, h_start: h_end, w_start: w_end]

        if "valid_mask" in sample:
            sample["valid_mask"] = sample["valid_mask"][h_start: h_end, w_start: w_end]

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"][h_start: h_end, w_start: w_end]

        return sample
