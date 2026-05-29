# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group transformation for action recognition"""

import numpy as np
from PIL import Image
import random
import torch


class GroupWorker(object):
    """Wrapper for group transformation using torchvision."""

    def __init__(self, worker):
        """Init worker."""
        self.worker = worker

    def __call__(self, img_group):
        """img_group: PIL Images list."""
        return [self.worker(img) for img in img_group]


class GroupRandomCrop(object):
    """RandomCrop for the group of frames."""

    def __init__(self, size):
        """Init."""
        self.size = size

    def __call__(self, img_group):
        """img_group: PIL Images list."""
        w, h = img_group[0].size
        th, tw = self.size

        out_images = []

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class MultiScaleCrop(object):
    """
    Crop images with a list of randomly selected scales.

    Args:
        input_size (int | tuple[int]): (w, h) of network input.
        scales (list[float]): width and height scales to be selected.
    """

    def __init__(self,
                 input_size,
                 scales=[1, 0.875, 0.75, 0.66],
                 max_distort=0,
                 fix_crop=True,
                 more_fix_crop=True):
        """max_distort: introducing aspect-ratio augmentation."""
        self.scales = scales
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):
        """img_group: PIL Images list."""
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_patch(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
                          for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _fill_crop_size(self, img_w, img_h):
        """Generate crop size collections."""
        base_size = min(img_w, img_h)
        crop_sizes = [int(base_size * s) for s in self.scales]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]

        candidate_sizes = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    candidate_sizes.append((w, h))

        return candidate_sizes

    def _fill_fix_offset(self, image_w, image_h, crop_w, crop_h):
        """Generate crop offset collections."""
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = []
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if self.more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    def _sample_crop_patch(self, im_size):
        """Random choose crop patch."""
        img_w, img_h = im_size

        # find a crop size
        candidate_sizes = self._fill_crop_size(img_w, img_h)
        crop_width, crop_height = random.choice(candidate_sizes)

        if not self.fix_crop:
            w_offset = random.randint(0, img_w - crop_width)
            h_offset = random.randint(0, img_h - crop_height)
        else:
            offsets = self._fill_fix_offset(img_w, img_h, crop_width, crop_height)
            w_offset, h_offset = random.choice(offsets)

        return crop_width, crop_height, w_offset, h_offset


class GroupRandomHorizontalFlip(object):
    """Random horizontal flip group of frames."""

    def __init__(self, flip_prob=0.5, is_flow=False):
        """Init."""
        self.flip_prob = flip_prob
        self.is_flow = is_flow

    def __call__(self, img_group):
        """img_group: PIL Images list."""
        if random.random() < self.flip_prob:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            # @TODO(tylerz): figure out the right way to flip optical flow
        else:
            ret = img_group

        return ret


class GroupNormalize(object):
    """Normalize the group of frames. substract mean -> divide std."""

    def __init__(self, mean, std):
        """Init."""
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """tensor: torch tensor CTHW."""
        if len(self.mean) != 0 and len(self.std) != 0:
            rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
            rep_std = self.std * (tensor.size()[0] // len(self.std))

            # TODO: make efficient
            for t, m, s in zip(tensor, rep_mean, rep_std):
                t.sub_(m).div_(s)
        elif len(self.mean) != 0 and len(self.std) == 0:
            rep_mean = self.mean * (tensor.size()[0] // len(self.mean))

            # TODO: make efficient
            for t, m in zip(tensor, rep_mean):
                t.sub_(m)
        elif len(self.std) != 0 and len(self.mean) == 0:
            rep_std = self.std * (tensor.size()[0] // len(self.std))

            # TODO: make efficient
            for t, s in zip(tensor, rep_std):
                t.div_(s)

        return tensor


class GroupThreeCrop(object):
    """Crop group of frames. Crop three parts of each frames."""

    def __init__(self, size):
        """Init."""
        self.size = size

    def __call__(self, img_group):
        """img_group: PIL Images list."""
        w, h = img_group[0].size
        th, tw = self.size
        assert th == h or tw == w

        if th == h:
            w_step = (w - tw) // 2
            offsets = []
            offsets.append((0, 0))  # left
            offsets.append((2 * w_step, 0))  # right
            offsets.append((w_step, 0))  # middle
        elif tw == w:
            h_step = (h - th) // 2
            offsets = []
            offsets.append((0, 0))  # top
            offsets.append((0, 2 * h_step))  # down
            offsets.append((0, h_step))  # middle

        new_clips = []
        for ow, oh in offsets:
            for cur_img in img_group:
                # crop_img = cur_img[oh:oh+th, ow:ow+tw, :]
                crop_img = cur_img.crop((ow, oh, ow + tw, oh + th))
                new_clips.append(crop_img)
        return new_clips


class ToTorchFormatTensor(object):
    """ Converts numpy.ndarray (T x H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0]
    """

    def __init__(self, div=True):
        """Init."""
        self.div = div

    def __call__(self, pic):
        """pic: ndarray (THWC)"""
        if isinstance(pic, np.ndarray):
            # handle numpy array
            # put it from THWC to CTHW format
            imgs = torch.from_numpy(pic).permute(3, 0, 1, 2).contiguous()
        else:
            raise TypeError("pic should be numpy.ndarray")

        return imgs.float().div(255) if self.div else imgs.float()


class ToNumpyNDArray(object):
    """Convert PIL Images to nd array."""

    def __call__(self, img_group):
        """img_group: PIL Images list."""
        if img_group[0].mode == 'L':
            return np.array([np.stack((np.array(img_group[x]), np.array(img_group[x + 1])), axis=-1)
                             for x in range(0, len(img_group), 2)])
        if img_group[0].mode == 'RGB':
            return np.array([np.array(x) for x in img_group])

        return np.array([])


class GroupJointWorker(object):
    """Wrapper for joint group transformation using torchvision."""

    def __init__(self, worker):
        """Init."""
        self.worker = worker

    def __call__(self, img_group):
        """img_group: two PIL Images lists for rgb and of respectively."""
        rgb_group, of_group = img_group
        rgb_group = [self.worker(img) for img in rgb_group]
        of_group = [self.worker(img) for img in of_group]

        return [rgb_group, of_group]


class JointWorker(object):
    """Wrapper for joint group transformation using other group op."""

    def __init__(self, worker):
        """Init."""
        self.worker = worker

    def __call__(self, img_group):
        """img_group: two PIL Images lists or ndarray for rgb and of respectively."""
        rgb_group, of_group = img_group
        rgb_ret_group = self.worker(rgb_group)
        of_ret_group = self.worker(of_group)

        return [rgb_ret_group, of_ret_group]


class GroupJointRandomCrop(object):
    """Group random crop for joint training."""

    def __init__(self, size):
        """init."""
        self.size = size

    def __call__(self, img_group):
        """img_group: two PIL Images lists for rgb and of respectively."""
        rgb_group, of_group = img_group

        w, h = rgb_group[0].size
        th, tw = self.size

        out_rgb_images = []
        out_of_images = []

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in rgb_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_rgb_images.append(img)
            else:
                out_rgb_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        for img in of_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_of_images.append(img)
            else:
                out_of_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return [out_rgb_images, out_of_images]


class JointMultiScaleCrop(MultiScaleCrop):
    """MultiScaleCrop for joint training."""

    def __call__(self, img_group):
        """img_group: two PIL Images lists for rgb and of respectively."""
        rgb_group, of_group = img_group

        im_size = rgb_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_patch(im_size)

        rgb_crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
                              for img in rgb_group]
        rgb_ret_img_group = [img.resize((self.input_size[0], self.input_size[1]),
                                        self.interpolation) for img in rgb_crop_img_group]

        of_crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
                             for img in of_group]
        of_ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                            for img in of_crop_img_group]

        return [rgb_ret_img_group, of_ret_img_group]


class GroupJointRandomHorizontalFlip(object):
    """Group random horizontal flip for joint training."""

    def __init__(self, flip_prob=0.5):
        """Init."""
        self.flip_prob = flip_prob

    def __call__(self, img_group):
        """img_group: two PIL Images lists for rgb and of respectively."""
        rgb_group, of_group = img_group

        if random.random() < self.flip_prob:
            rgb_ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in rgb_group]
            of_ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in of_group]
        else:
            rgb_ret = rgb_group
            of_ret = of_group

        return [rgb_ret, of_ret]


class GroupJointNormalize(object):
    """Group normalization for joint training."""

    def __init__(self, rgb_input_mean, rgb_input_std,
                 of_input_mean, of_input_std):
        """Init"""
        self.rgb_normalize = GroupNormalize(rgb_input_mean,
                                            rgb_input_std)
        self.of_normalize = GroupNormalize(of_input_mean,
                                           of_input_std)

    def __call__(self, img_group):
        """img_group: two torch tensors for rgb and of respectively."""
        rgb_group, of_group = img_group

        rgb_ret_group = self.rgb_normalize(rgb_group)
        of_ret_group = self.of_normalize(of_group)

        return [rgb_ret_group, of_ret_group]
