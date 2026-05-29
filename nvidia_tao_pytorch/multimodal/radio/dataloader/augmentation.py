# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RADIO Augmentation module."""
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter
from omegaconf import OmegaConf

import torchvision.transforms.functional as TF
from torchvision import transforms
from timm.data.random_erasing import RandomErasing
from nvidia_tao_pytorch.multimodal.radio.dataloader.rand_aug import RandAug
from nvidia_tao_pytorch.multimodal.radio.dataloader.spatial_transforms import (
    patch_aligned_random_resized_crop,
    continuous_random_rotation,
    random_perspective_transform,
)


class CLDataAugmentation:
    """
    Class for applying data augmentation to images.

    Args:
        img_size (int): The size of the images after resizing.
        random_flip (dict, optional): A dictionary containing keys: enable (bool), hflip_probability (float), vflip_probability (float).
        random_rotate (dict, optional): A dictionary containing keys: enable (bool), rotate_probability (float), angle_list (List[float]).
        random_color (dict, optional): A dictionary containing keys: enable (bool), brightness (float), contrast (float), saturation (float), hue (float).
        random_erase (dict, optional): A dictionary containing keys: enable (bool), erase_probability(float), erase_scale(List[float]), erase_ratio(List[float]), value (float).
        with_scale_random_crop (dict, optional): A dictionary containing keys: enable (bool), scale_range (List[float]).
        with_random_crop (bool, optional): Apply random resized crop.
        with_random_blur (bool, optional): Apply random Gaussian blur.
        mean (list, optional): Mean values used by RandAug (default is [0.5, 0.5, 0.5]).
        patch_size (int, optional): ViT patch size for patch-aligned crops (e.g. 14, 16). Enables alignment when set.
        use_continuous_rotation (bool, optional): Use continuous angle rotation instead of discrete (default False).
        perspective_distortion (dict, optional): Config for perspective transform: enable, scale, prob.
    """

    def __init__(
            self,
            img_size,
            random_flip=None,
            random_rotate=None,
            random_color=None,
            random_erase=None,
            random_aug=None,
            with_scale_random_crop=None,
            with_random_crop=False,
            with_random_blur=False,
            mean=[0.5, 0.5, 0.5],
            patch_size=None,
            use_continuous_rotation=False,
            perspective_distortion=None,
            is_training=True,
    ):
        """Initialize"""
        if img_size is None:
            raise ValueError("img_size must be specified")
        self.img_size = img_size
        self.is_training = is_training
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_color = random_color
        self.random_erase = random_erase
        self.random_aug = random_aug
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.mean = mean
        self.patch_size = patch_size if patch_size else None
        self.use_continuous_rotation = use_continuous_rotation
        self.perspective_distortion = perspective_distortion

        # transform function
        if self.with_random_crop and self.patch_size is None:
            self.randomcrop = transforms.RandomResizedCrop(size=self.img_size, scale=(0.08, 1.0))
        if self.random_color is not None and self.random_color.enable:
            self.colorjitter = transforms.ColorJitter(
                brightness=self.random_color.brightness,
                contrast=self.random_color.contrast,
                saturation=self.random_color.saturation,
                hue=self.random_color.hue
            )

        if self.random_erase is not None and self.random_erase.enable:
            # This to_container is to convert omegaconf to python type (dict, list); our using old version torch.transforms requires python type
            self.randomerase = OmegaConf.to_container(self.random_erase)
            self.randomerase = RandomErasing(
                probability=self.randomerase["erase_probability"], device="cpu",
            )
        if self.random_aug is not None and self.random_aug.enable:
            self.randomaug = RandAug({}, mean=self.mean)

    def _aspect_resize(self, img):
        """Resize so the shortest side equals img_size, preserving aspect ratio."""
        w, h = img.size  # PIL returns (width, height)
        short_side = min(w, h)
        scale = self.img_size / short_side
        new_w, new_h = round(w * scale), round(h * scale)
        return TF.resize(img, [new_h, new_w], interpolation=Image.BILINEAR)

    def _aspect_resize_and_center_crop(self, img):
        """Standard evaluation resize: scale shortest side to target, then center crop to square."""
        img = self._aspect_resize(img)
        return TF.center_crop(img, [self.img_size, self.img_size])

    def _aspect_resize_and_random_crop(self, img):
        """Training resize: scale shortest side to target, then random crop to square."""
        img = self._aspect_resize(img)
        w, h = img.size
        if w == self.img_size and h == self.img_size:
            return img
        top = random.randint(0, h - self.img_size)
        left = random.randint(0, w - self.img_size)
        return TF.crop(img, top, left, self.img_size, self.img_size)

    def transform(self, imgs, to_tensor=True):
        """
        Apply a sequence of data augmentation routines to a list of images.

        Args:
            imgs (list): A list of PIL images to apply data augmentation to.
            to_tensor (bool, optional): Convert images to PyTorch tensors (default is True).

        Returns:
            tuple: A tuple containing augmented images.

        Notes:
            This function performs a series of image augmentation operations on the input images. The following steps are performed in sequence:

            1. Aspect-preserving resize (shortest side to target) + random crop (train) or center crop (val).
            2. Apply horizontal and vertical flips based on the given probabilities.
            3. Apply Rand Aug.
            4. Apply random rotation based on the given probability and angle list.
            5. Apply random resized crop if specified.
            6. Apply scale and random crop transformations if enabled.
            7. Apply random Gaussian blur if specified.
            8. Apply random color jitter transformations.
            9. Convert images to PyTorch tensors (output range [0, 1]).

        """
        # resize image and covert to tensor
        # input already is PIL image

        if self.is_training:
            imgs = [self._aspect_resize_and_random_crop(img) for img in imgs]
        else:
            imgs = [self._aspect_resize_and_center_crop(img) for img in imgs]

        if self.random_flip is not None and self.random_flip.enable:
            hflip_probability = 1 - self.random_flip.hflip_probability
            vflip_probability = 1 - self.random_flip.vflip_probability

            if random.random() > hflip_probability:
                imgs = [TF.hflip(img) for img in imgs]

            if random.random() > vflip_probability:
                imgs = [TF.vflip(img) for img in imgs]

        if self.random_aug is not None and self.random_aug.enable:
            imgs = [self.randomaug.aug_image(img) for img in imgs]

        # Rotation: continuous or discrete
        if self.random_rotate is not None and self.random_rotate.enable:
            random_base = 1 - self.random_rotate.rotate_probability
            if random.random() > random_base:
                if self.use_continuous_rotation:
                    angle_range = getattr(
                        self.random_rotate, "angle_range", (-15, 15)
                    )
                    if angle_range is not None and isinstance(
                            angle_range, (list, tuple)) and len(angle_range) >= 2:
                        # Convert to tuple of floats (handles OmegaConf)
                        low, high = float(angle_range[0]), float(angle_range[1])
                        imgs = [continuous_random_rotation(img, (low, high))
                                for img in imgs]
                    else:
                        imgs = [continuous_random_rotation(img, (-15, 15))
                                for img in imgs]
                else:
                    angles = getattr(self.random_rotate, "angle_list", [0, 90, 180, 270]) or [0, 90, 180, 270]
                    index = random.randint(0, len(angles) - 1) if angles else 0
                    angle = angles[index]
                    imgs = [TF.rotate(img, angle) for img in imgs]

        # Crop: patch-aligned when patch_size set, else standard
        if self.patch_size is not None and (self.with_random_crop or
                                            (self.with_scale_random_crop is not None and self.with_scale_random_crop.enable)):
            if self.with_scale_random_crop is not None and self.with_scale_random_crop.enable:
                scale_range = self.with_scale_random_crop.scale_range
                target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
                imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            imgs = [patch_aligned_random_resized_crop(
                img, self.img_size, self.patch_size, scale=(0.08, 1.0)
            ) for img in imgs]
        elif self.with_random_crop:
            imgs = [self.randomcrop(img) for img in imgs]
        elif self.with_scale_random_crop is not None and self.with_scale_random_crop.enable:
            # rescale
            scale_range = self.with_scale_random_crop.scale_range
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]

        # Perspective transform
        if self.perspective_distortion is not None and getattr(
                self.perspective_distortion, "enable", False):
            scale = getattr(self.perspective_distortion, "scale", [0.01, 0.05])
            if hasattr(scale, '__len__') and len(scale) >= 2:
                dist = scale[0] + random.random() * (scale[1] - scale[0])
            else:
                dist = 0.1
            prob = getattr(self.perspective_distortion, "prob", 0.5)
            imgs = [random_perspective_transform(img, distortion_scale=dist, probability=prob)
                    for img in imgs]

        if self.with_random_blur:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        if self.random_color is not None and self.random_color.enable:
            imgs_tf = []
            for img in imgs:
                imgs_tf.append(self.colorjitter(img))
            imgs = imgs_tf

        if to_tensor:
            imgs = [TF.to_tensor(img) for img in imgs]

        if self.random_erase is not None and self.random_erase.enable:
            imgs_tf = []
            for img in imgs:
                imgs_tf.append(self.randomerase(img))
            imgs = imgs_tf

        return imgs


def pil_crop(image, box, cropsize, default_value):
    """
    Crop an image using the specified box coordinates.

    Args:
        image (PIL.Image.Image): The input image to be cropped.
        box (Tuple[int]): A tuple containing the crop box coordinates.
        cropsize (int): The desired size of the crop.
        default_value (int): The default value to fill the cropped image.

    Returns:
        PIL.Image.Image: The cropped image.
    """
    assert isinstance(image, Image.Image), "image should be PIL.Image.Image"
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    """
    Generate random crop box coordinates for cropping an image.

    Args:
        imgsize (Tuple[int, int]): The size of the original image (height, width).
        cropsize (int): The desired size of the crop.

    Returns:
        Tuple: A tuple containing the crop box coordinates.
    """
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top + ch, cont_left, cont_left + cw, img_top, img_top + ch, img_left, img_left + cw


def pil_rescale(img, scale, order):
    """
    Resize an image using a specified scale.

    Args:
        img (Image.Image): The input image to be rescaled.
        scale (float): The scaling factor.
        order (int): The interpolation order for resizing.

    Returns:
        Image.Image: The rescaled image.
    """
    assert isinstance(img, Image.Image), "image should be PIL.Image.Image"
    height, width = img.size
    target_size = (int(np.round(height * scale)), int(np.round(width * scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    """
    Resize an image using a specified scale.

    Args:
        img (Image.Image): The input image to be resized.
        size (Tuple[int, int]): The target size (height, width) of the resized image.
        order (int): The interpolation order for resizing.

    Returns:
        Image.Image: The resized image.
    """
    assert isinstance(img, Image.Image), "image should be PIL.Image.Image"
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)
