# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Transformation for Depth Network."""

import cv2
import numpy as np
from PIL import Image

from torchvision.transforms import Compose, ColorJitter
from nvidia_tao_pytorch.cv.depth_net.dataloader.utils.dataset_utils import Resize, NormalizeImage, PrepareForNet, Crop


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


def build_stereo_transforms(aug_config, max_disparity=None, split='train'):
    """Build Stereo Augmentations.

    Args:
        aug_config (dict): augmentation configuration
        max_disparity (int): maximum disparity
        split (str): data split (train, val, eval, infer)
        resize_target (bool): whether to resize the images. (Not used)

    Returns:
        transforms (Compose): Final built transforms.

    """
    mean = aug_config["input_mean"]
    std = aug_config["input_std"]
    if split in ['train', 'val']:
        transform = Compose([
            ColorTransform(
                color_aug_prob=aug_config["color_aug_prob"],
                color_aug_brightness=aug_config["color_aug_brightness"],
                color_aug_contrast=aug_config["color_aug_contrast"],
                color_aug_saturation=aug_config["color_aug_saturation"],
                color_aug_hue_range=aug_config["color_aug_hue_range"]),
            EraserTransform(eraser_aug_prob=aug_config["eraser_aug_prob"]),
            SpatialTransform(
                do_flip=aug_config["do_flip"],
                crop_size=aug_config["crop_size"],
                min_scale=aug_config["min_scale"],
                max_scale=aug_config["max_scale"],
                max_disparity=max_disparity,
                stretch_prob=aug_config["stretch_prob"],
                max_stretch=aug_config["max_stretch"],
                spatial_aug_prob=aug_config["spatial_aug_prob"],
                yjitter_prob=aug_config["yjitter_prob"],
                crop_min_valid_disp_ratio=aug_config["crop_min_valid_disp_ratio"]),
            NormalizeImage(mean=mean, std=std),
            PrepareForNet()])
    elif split in ['infer']:
        transform = Compose([
            NormalizeImage(mean=mean, std=std),
            PrepareForNet()])
    else:
        raise NotImplementedError(f'training method {split} not implemented for training transforms!')

    return transform


class ColorTransform(object):
    """
    Applies photometric augmentation (color jitter) to stereo images.
    It can apply different augmentations to left and right images (asymmetric)
    or the same augmentation (symmetric) based on a probability.

    Args:
        color_aug_prob (float): The probability of applying asymmetric color jitter.
                                If a random number is less than this, asymmetric
                                jitter is applied. Otherwise, symmetric jitter is applied.
        color_aug_brightness (float or tuple): How much to jitter brightness.
                                               Float value is a factor (e.g., 0.4 means a random
                                               factor in [1-0.4, 1+0.4] or [0.6, 1.4] is used).
                                               Tuple (min, max) specifies the range directly.
        color_aug_contrast (float or tuple): How much to jitter contrast.
                                             Similar to brightness.
        color_aug_saturation (float or tuple): How much to jitter saturation.
                                               Similar to brightness.
        color_aug_hue_range (float or tuple): How much to jitter hue.
                                               Float value specifies the maximum delta.
                                               Tuple (min, max) specifies the range.
                                               Values should be between -0.5 and 0.5.
    """

    def __init__(self, color_aug_prob, color_aug_brightness, color_aug_contrast, color_aug_saturation, color_aug_hue_range):

        self.color_aug_prob = color_aug_prob
        self.color_aug_brightness = color_aug_brightness
        self.color_aug_contrast = color_aug_contrast
        self.color_aug_saturation = tuple(color_aug_saturation)
        self.color_aug_hue_range = tuple(color_aug_hue_range)

    def __call__(self, sample):
        """
        Applies color jitter transformation to the input sample.

        Args:
            sample (dict): A dictionary containing 'image' (left image) and 'right_image'.
                           Images are expected to be NumPy arrays.

        Returns:
            dict: The transformed sample with augmented 'image' and 'right_image'.
                  Images are returned as NumPy arrays.

        Raises:
            NotImplementedError: If 'right_image' is not found in the sample when symmetric
                                 augmentation is attempted.
        """
        img1 = Image.fromarray(sample['image'])

        if np.random.rand() < self.color_aug_prob:
            transform = ColorJitter(brightness=0.4,
                                    contrast=0.4,
                                    saturation=(0.6, 1.4),
                                    hue=self.color_aug_hue_range)

            img1 = np.array(transform(img1))
            if 'right_image' in sample:
                img2 = Image.fromarray(sample['right_image'])
                img2 = np.array(transform(img2))

        # symmetric augmentation
        else:
            # Symmetric augmentation: Apply the same random jitter factor to both images
            # by concatenating them and transforming them together.
            transform = ColorJitter(brightness=self.color_aug_brightness,
                                    contrast=self.color_aug_contrast,
                                    saturation=self.color_aug_saturation,
                                    hue=self.color_aug_hue_range)

            if 'right_image' in sample:
                img2 = sample['right_image']
            else:
                raise NotImplementedError('Color transforms requires a right image pair')

            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = Image.fromarray(image_stack)
            image_stack = np.array(transform(image_stack))
            img1, img2 = np.split(image_stack, 2, axis=0)

        sample['image'] = img1
        sample['right_image'] = img2

        return sample


class EraserTransform(object):
    """
    Applies an occlusion augmentation to the right stereo image by filling random rectangular
    regions with the mean color of the right image. This simulates occlusions.

    Args:
        eraser_aug_prob (float): The probability of applying this augmentation.
        eraser_bounds (list or tuple): A two-element list/tuple [min_size, max_size]
                                       specifying the minimum and maximum dimensions (width/height)
                                       of the rectangular regions to be erased.
    """

    def __init__(self, eraser_aug_prob, eraser_bounds=[50, 100]):
        self.eraser_aug_prob = eraser_aug_prob
        self.eraser_bounds = eraser_bounds

    def __call__(self, sample):
        """
        Applies the eraser transformation to the input sample.

        Args:
            sample (dict): A dictionary containing 'image' (left image) and 'right_image'.

        Returns:
            dict: The transformed sample with a potentially augmented 'right_image'.

        Raises:
            NotImplementedError: If 'right_image' is not found in the sample.
        """
        img1 = sample['image']
        ht, wd = img1.shape[:2]
        if 'right_image' in sample:
            img2 = sample['right_image']
        else:
            raise NotImplementedError('EraserTransform requires a stereo right image!')

        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                img2[y0: y0 + dy, x0: x0 + dx, :] = mean_color

        sample['image'] = img1
        sample['right_image'] = img2

        return sample


class SpatialTransform(object):
    """
    Applies random spatial augmentations to stereo images and their corresponding
    disparity maps, including scaling, stretching, flipping, and random cropping.

    Args:
        do_flip (str): Specifies the type of flipping to perform.
                       'hf' for horizontal flip of image and flow (for optical flow tasks),
                       'h' for horizontal flip for stereo pairs (swaps left/right images and flips them),
                       'v' for vertical flip. Set to None or empty string to disable.
        crop_size (tuple): Desired output crop size (height, width).
        min_scale (float): Minimum factor for random scaling.
        max_scale (float): Maximum factor for random scaling.
        max_disparity (float): Maximum valid disparity value. Disparity values greater than this
                          are considered invalid (e.g., np.inf).
        stretch_prob (float): Probability of applying non-uniform stretching (different scales for x and y).
        spatial_aug_prob (float): Probability of applying spatial augmentations (scaling and stretching).
        yjitter_prob (float): Probability of applying a small vertical jitter to the crop start.
        max_stretch (float): Maximum factor for stretching in either x or y direction.
        crop_min_valid_disp_ratio (float): Minimum ratio of valid disparity pixels required within a crop.
                                           If a random crop has fewer valid pixels, it will try again.
        h_flip_prob (float): Probability of applying horizontal flip. Default is 0.5.
        v_flip_prob (float): Probability of applying vertical flip. Default is 0.1.
    """

    def __init__(self, do_flip, crop_size, min_scale, max_scale, max_disparity,
                 stretch_prob, spatial_aug_prob, yjitter_prob, max_stretch,
                 crop_min_valid_disp_ratio, h_flip_prob=0.5, v_flip_prob=0.1):
        self.crop_size = crop_size
        self.do_flip = do_flip
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.stretch_prob = stretch_prob
        self.spatial_aug_prob = spatial_aug_prob
        self.yjitter_prob = yjitter_prob
        self.max_disparity = max_disparity
        self.max_stretch = max_stretch
        self.crop_min_valid_disp_ratio = crop_min_valid_disp_ratio
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob

    def __call__(self, sample):
        """
        Applies the spatial transformations to the input sample.

        Args:
            sample (dict): A dictionary containing 'image' (left image), 'right_image', and 'disparity'.

        Returns:
            dict: The transformed sample with augmented 'image', 'right_image', and 'disparity'.

        Raises:
            NotImplementedError: If 'right_image' is not found in the sample.
        """
        img1 = sample['image']
        ht, wd = img1.shape[:2]
        if 'right_image' in sample:
            img2 = sample['right_image']
        else:
            raise NotImplementedError('SpatialTransorm requires a right stereo image pair!')

        flow = sample['disparity']

        # Ensure images are large enough for cropping by padding if necessary
        min_height_needed = self.crop_size[0] + 8  # +8 for jittering buffer
        min_width_needed = self.crop_size[1] + 8

        if ht < min_height_needed or wd < min_width_needed:
            pad_top = max(0, (min_height_needed - ht) // 2)
            pad_bottom = max(0, min_height_needed - ht - pad_top)
            pad_left = max(0, (min_width_needed - wd) // 2)
            pad_right = max(0, min_width_needed - wd - pad_left)

            # Pad images with reflection to avoid introducing artifacts
            img1 = np.pad(img1, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
            img2 = np.pad(img2, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')

            # Pad disparity/flow with zeros (invalid disparity)
            if flow.ndim == 2:
                flow = np.pad(flow, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            elif flow.ndim == 3:
                flow = np.pad(flow, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
            else:
                raise ValueError(f"Unexpected flow array dimensions: {flow.shape}")

            # Update dimensions
            ht, wd = img1.shape[:2]

        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, 1)
        scale_y = np.clip(scale_y, min_scale, 1)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            H, W = img1.shape[:2]
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img1 = cv2.resize(img1, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, dsize=(W, H), interpolation=cv2.INTER_LINEAR)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf':  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h':  # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v':
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        n_trial = -1
        x0_best = 0
        y0_best = 0
        valid_ratio_best = 0

        while 1:
            n_trial += 1
            if n_trial >= 100:
                img1 = img1[y0_best: y0_best + self.crop_size[0], x0_best: x0_best + self.crop_size[1]]
                img2 = img2[y0_best: y0_best + self.crop_size[0], x0_best: x0_best + self.crop_size[1]]
                flow = flow[y0_best: y0_best + self.crop_size[0], x0_best: x0_best + self.crop_size[1]]
                break

            if np.random.uniform(0, 1) < self.yjitter_prob:
                # Check if we have enough space for jittering (need at least 4 pixels buffer)
                y_max = img1.shape[0] - self.crop_size[0] - 2
                x_max = img1.shape[1] - self.crop_size[1] - 2

                if y_max > 2 and x_max > 2:
                    # We have enough space for jittering
                    y0 = np.random.randint(2, y_max)
                    x0 = np.random.randint(2, x_max)
                    y1 = y0 + np.random.randint(-2, 2 + 1)
                else:
                    # Fall back to regular cropping without jittering
                    y0 = np.random.randint(0, max(1, img1.shape[0] - self.crop_size[0]))
                    x0 = np.random.randint(0, max(1, img1.shape[1] - self.crop_size[1]))
                    y1 = y0
            else:
                y0 = np.random.randint(0, max(1, img1.shape[0] - self.crop_size[0]))
                x0 = np.random.randint(0, max(1, img1.shape[1] - self.crop_size[1]))
                y1 = y0

            flow_crop = flow[y0: y0 + self.crop_size[0], x0: x0 + self.crop_size[1]]
            valid_ratio = (flow_crop[..., 0] <= self.max_disparity).sum() / (self.crop_size[0] * self.crop_size[1])
            if valid_ratio < self.crop_min_valid_disp_ratio:
                if valid_ratio > valid_ratio_best:
                    valid_ratio_best = valid_ratio
                    x0_best = x0
                    y0_best = y0
                    continue

            img1 = img1[y0: y0 + self.crop_size[0], x0: x0 + self.crop_size[1]]
            img2 = img2[y1: y1 + self.crop_size[0], x0: x0 + self.crop_size[1]]
            flow = flow_crop
            break

        sample['image'] = img1
        sample['right_image'] = img2
        sample['disparity'] = flow
        return sample
