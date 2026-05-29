# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classification Augmentation module."""
import random
import numpy as np
import torch
from typing import Callable, Optional, Any
from PIL import Image
from PIL import ImageFilter
from omegaconf import OmegaConf

import torchvision.transforms.functional as TF
from torchvision import transforms
from timm.data.random_erasing import RandomErasing
from nvidia_tao_pytorch.cv.classification_pyt.dataloader.rand_aug import RandAug


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
        mean (list, optional): Mean values for normalization (default is [0.5, 0.5, 0.5]).
        std (list, optional): Standard deviation values for normalization (default is [0.5, 0.5, 0.5]).
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
            std=[0.5, 0.5, 0.5]
    ):
        """Initialize"""
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_color = random_color
        self.random_erase = random_erase
        self.random_aug = random_aug
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.mean = mean
        self.std = std
        # transform function
        if self.with_random_crop:
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

            1. Resize images to the specified image size (if not using dynamic resizing).
            2. Apply horizontal and vertical flips based on the given probabilities.
            3. Apply Rand Aug.
            4. Apply random rotation based on the given probability and angle list.
            5. Apply random resized crop if specified.
            6. Apply scale and random crop transformations if enabled.
            7. Apply random Gaussian blur if specified.
            8. Apply random color jitter transformations.
            9. Convert images to PyTorch tensors and normalize them.

        """
        # resize image and covert to tensor
        # input already is PIL image

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=Image.BILINEAR)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]

        if self.random_flip is not None and self.random_flip.enable:
            hflip_probability = 1 - self.random_flip.hflip_probability
            vflip_probability = 1 - self.random_flip.vflip_probability

            if random.random() > hflip_probability:
                imgs = [TF.hflip(img) for img in imgs]

            if random.random() > vflip_probability:
                imgs = [TF.vflip(img) for img in imgs]

        if self.random_aug is not None and self.random_aug.enable:
            imgs = [self.randomaug.aug_image(img) for img in imgs]

        if self.random_rotate is not None and self.random_rotate.enable:
            random_base = 1 - self.random_rotate.rotate_probability

            if random.random() > random_base:
                angles = self.random_rotate.angle_list
                index = random.randint(0, 2)
                angle = angles[index]
                imgs = [TF.rotate(img, angle) for img in imgs]

        if self.with_random_crop:
            imgs = [self.randomcrop(img) for img in imgs]

        if self.with_scale_random_crop is not None and self.with_scale_random_crop.enable:
            # rescale
            scale_range = self.with_scale_random_crop.scale_range
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
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
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            imgs = [TF.normalize(img, mean=self.mean, std=self.std)
                    for img in imgs]

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


# reference:
# https://github.com/mit-han-lab/efficientvit/blob/master/efficientvit/clscore/trainer/utils/mixup.py
# https://github.com/mit-han-lab/efficientvit/blob/master/efficientvit/apps/data_provider/augment/bbox.py
# https://github.com/mit-han-lab/efficientvit/blob/master/efficientvit/models/utils/random.py
def torch_randint(low: int, high: int, generator: Optional[torch.Generator] = None) -> int:
    """
    Generate a random integer between low and high.

    Args:
        low (int): The lower bound of the random integer.
        high (int): The upper bound of the random integer.
        generator (torch.Generator, optional): The random number generator.

    Returns:
        int: The random integer.
    """
    if low == high:
        return low
    else:
        assert low < high, "in torch_randint, low should be less than high"
        return int(torch.randint(low=low, high=high, generator=generator, size=(1,)))


def torch_shuffle(src_list: list[Any], generator: Optional[torch.Generator] = None) -> list[Any]:
    """
    Shuffle a list of items randomly.

    Args:
        src_list (list): The list of items to shuffle.
        generator (torch.Generator, optional): The random number generator.

    Returns:
        list: The shuffled list of items
    """
    rand_indexes = torch.randperm(len(src_list), generator=generator).tolist()
    return [src_list[i] for i in rand_indexes]


def rand_bbox(
    h: int,
    w: int,
    lam: float,
    rand_func: Callable = np.random.uniform,
) -> tuple[int, int, int, int]:
    """
    Generate a random bounding box for cutmix augmentation.

    Args:
        h (int): The height of the image.
        w (int): The width of the image.
        lam (float): The cutmix ratio.
        rand_func (Callable, optional): The random number generator function.

    Returns:
        tuple: A tuple containing the bounding box coordinates.
    """
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = w * cut_rat
    cut_h = h * cut_rat

    # uniform
    cx = rand_func(0, w)
    cy = rand_func(0, h)

    bbx1 = int(np.clip(cx - cut_w / 2, 0, w))
    bby1 = int(np.clip(cy - cut_h / 2, 0, h))
    bbx2 = int(np.clip(cx + cut_w / 2, 0, w))
    bby2 = int(np.clip(cy + cut_h / 2, 0, h))

    return bbx1, bby1, bbx2, bby2


def mixup(
    images: torch.Tensor,
    target: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mixup augmentation to a batch of images and labels.

    Args:
        images (torch.Tensor): The input images.
        target (torch.Tensor): The input labels.
        lam (float): The mixup ratio.

    Returns:
        tuple: A tuple containing the augmented images and labels.
    """
    rand_index = torch_shuffle(list(range(0, images.shape[0])))

    flipped_images = images[rand_index]
    flipped_target = target[rand_index]

    return (
        lam * images + (1 - lam) * flipped_images,
        lam * target + (1 - lam) * flipped_target,
    )


def cutmix(
    images: torch.Tensor,
    target: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply cutmix augmentation to a batch of images and labels.

    Args:
        images (torch.Tensor): The input images.
        target (torch.Tensor): The input labels.
        lam (float): The cutmix ratio.

    Returns:
        tuple: A tuple containing the augmented images and labels.
    """
    rand_index = torch_shuffle(list(range(0, images.shape[0])))
    flipped_images = images[rand_index]
    flipped_target = target[rand_index]

    b, _, h, w = images.shape
    lam_list = []
    for i in range(b):
        bbx1, bby1, bbx2, bby2 = rand_bbox(
            h=h,
            w=w,
            lam=lam,
            rand_func=torch_randint,
        )
        images[i, :, bby1:bby2, bbx1:bbx2] = flipped_images[i, :, bby1:bby2, bbx1:bbx2]
        lam_list.append(1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w)))
    lam = torch.Tensor(lam_list).to(images.device).view(b, 1)
    return images, lam * target + (1 - lam) * flipped_target


def apply_mixup_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    lam: float,
    mix_type="mixup",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mixup or cutmix augmentation to a batch of images and labels.

    Args:
        images (torch.Tensor): The input images.
        labels (torch.Tensor): The input labels.
        lam (float): The mixup or cutmix ratio.
        mix_type (str): The type of augmentation to apply (mixup or cutmix).

    Returns:
        tuple: A tuple containing the augmented images and labels.
    """
    if mix_type == "mixup":
        return mixup(images, labels, lam)
    elif mix_type == "cutmix":
        return cutmix(images, labels, lam)
    else:
        raise NotImplementedError("Only `mixup` and `cutmix` are supported.")
