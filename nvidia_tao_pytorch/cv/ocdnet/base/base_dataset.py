#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base_dataset module."""
import copy
import torch
from torch.utils.data import Dataset
from nvidia_tao_pytorch.cv.ocdnet.data_loader.modules import *  # pylint: disable=W0401,W0611,W0614
import cv2
import numpy as np
# flake8: noqa: F401, F403


class BaseDataSet(Dataset):
    """The base class for dataloader."""

    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None,
                 target_transform=None):
        """Initialize the dataset object with the given parameters.

        Args:
            data_path (str): The path to the dataset
            img_mode (str): The image mode of the images
            pre_process (dict): The preprocessing parameters to be used
            filter_keys (list): The keys not used in data dict
            ignore_tags (list): In lable file, the lines which contain ignore_tags will be ignored during training
        """
        if img_mode not in ['RGB', 'BGR', 'GRAY']:
            raise NotImplementedError(
                f"Unsupported image mode {img_mode} requested. Please set to any one of 'RGB', 'BGR', 'GRAY'."
            )
        self.ignore_tags = ignore_tags
        self.data_list = self.load_data(data_path)
        item_keys = ['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags']
        for item in item_keys:
            assert item in self.data_list[0], 'data_list from load_data must contain {}'.format(item_keys)
        self.img_mode = img_mode
        self.filter_keys = filter_keys
        self.transform = transform
        self.target_transform = target_transform
        self._init_pre_processes(pre_processes)

    def _init_pre_processes(self, pre_processes):
        """Initialize the preprocessing parameters.

        Args:
            IaaAugment (dict): Uses imgaug to perform augmentation. "Fliplr", "Affine", and "Resize" are used by default.
                               The "Fliplr" defines the probability of each image to be flipped.
                               The "rotate" defines the degree range when rotating images by a random value.
                               The "size" defines the range when resizing each image compared to its original size.
                               More methods can be implemented by using API in https://imgaug.readthedocs.io/en/latest/source/api.html
            EastRandomCropData (dict): The ramdom crop after augmentation. The "size" defines the cropped target size(width,height).
                                       The width and height should be multiples of 32. The "max_tries" defines the maximum times to try
                                       to crop since the cropped area may be too small or cropping may have failed.
                                       The "keep_ratio" specifies whether to keep the aspect ratio.
            MakeBorderMap (dict): Defines the parameter when generating a threshold map. The "shrink_ratio" is used to calculate the distance
                                  between expanding/shrinking polygons and the original text polygon. The "thresh_min" and "thresh_max" will
                                  set the threshold range when generating the threshold map.
            MakeShrinkMap (dict): Defines the parameter when generating a probability map. The "shrink_ratio" is used to generate shrunken
                                  polygons. The "min_text_size" specifies that the text will be ignored if its height or width is lower than this parameter.
        """
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']

                if isinstance(args, dict):
                    cls = globals()[aug['type']](**args)
                else:
                    cls = globals()[aug['type']](args)

                self.aug.append(cls)

    def load_data(self, data_path: str) -> list:
        """Load data into a dict

        Args:
            data_path (str): file or folder

        Returns:
            A dict (dict): contains 'img_path','img_name','text_polys','texts','ignore_tags'
        """
        raise NotImplementedError

    def apply_pre_processes(self, data):
        """Implement preprocessing for dataset."""
        for aug in self.aug:
            data = aug(data)
        return data

    def __getitem__(self, index):
        """Generate data dict per item."""
        try:
            data = copy.deepcopy(self.data_list[index])
            im = cv2.imread(data['img_path'], 1 if self.img_mode != 'GRAY' else 0).astype("float32")
            if self.img_mode == 'RGB':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            data['img'] = im
            data['shape'] = [im.shape[0], im.shape[1]]
            data = self.apply_pre_processes(data)
            rgb_mean = np.array([122.67891434, 116.66876762, 104.00698793])
            image = data['img']
            image -= rgb_mean
            image /= 255.
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            data['img'] = image
            data['text_polys'] = data['text_polys'].tolist()
            if len(self.filter_keys):
                data_dict = {}
                for k, v in data.items():
                    if k not in self.filter_keys:
                        data_dict[k] = v
                return data_dict
            return data
        except Exception:
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        """The length of data list."""
        return len(self.data_list)
