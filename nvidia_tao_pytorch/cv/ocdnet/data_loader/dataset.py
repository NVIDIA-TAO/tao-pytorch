#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset module."""
import os
import pathlib
import cv2
import multiprocessing
import numpy as np
import torch
from torch.utils.data import Dataset

from nvidia_tao_pytorch.cv.ocdnet.base.base_dataset import BaseDataSet
from nvidia_tao_pytorch.cv.ocdnet.utils import order_points_clockwise, get_datalist, get_datalist_uber
from nvidia_tao_pytorch.cv.ocdnet.utils.util import get_file_list


class UberDataset(BaseDataSet):
    """Uber Dataset class."""

    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        """Initialize."""
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """Load data."""
        pool = multiprocessing.Pool(processes=4)
        data_list = pool.apply_async(get_datalist_uber, args=(data_path,)).get()
        pool.close()
        pool.join()

        t_datalist = []
        pool = multiprocessing.Pool(processes=4)
        for img_path, label_path in data_list:
            tmp = pool.apply_async(self._get_annotation, args=(label_path,))
            data = tmp.get()
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_datalist.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        pool.close()
        pool.join()

        return t_datalist

    def _get_annotation(self, label_path: str) -> dict:
        polys = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f:
                content = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split('\t')
                params = content[0].split(" ")[:-2]
                try:
                    poly = np.array(list(map(float, params))).reshape(-1, 2).astype(np.float32)
                    if cv2.contourArea(poly) > 0:
                        polys.append(poly)
                        label = content[1]
                        if len(label.split(" ")) > 1:
                            label = "###"
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except Exception:
                    print('load label failed on {}'.format(label_path))
        data = {
            'text_polys': np.array(polys),
            'texts': texts,
            'ignore_tags': ignores,
        }

        return data


class ICDAR2015Dataset(BaseDataSet):
    """ICDAR2015 Dataset."""

    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, **kwargs):
        """Initialize."""
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """Load data."""
        data_list = get_datalist(data_path)
        t_datalist = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_datalist.append(item)
            else:
                print(f'there is no suit bbox in {label_path}')

        return t_datalist

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if not (box > -50).all():
                        continue
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = ','.join(params[8:])
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except Exception:
                    print(f'load label failed on {label_path}')
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }

        return data


class CustomImageDataset(Dataset):
    """Custom Image Dataset."""

    def __init__(self, img_dir, width, height, image_mode):
        """Initialize."""
        self.img_paths = get_file_list(img_dir, p_postfix=['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG', '.bmp'])
        self.width = width
        self.height = height
        self.image_mode = image_mode

    def __len__(self):
        """Length."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Get image."""
        img_path = self.img_paths[idx]
        assert os.path.exists(img_path), 'file is not exists'
        ori_img = cv2.imread(img_path, 1 if self.image_mode != 'GRAY' else 0).astype(np.float32)
        if self.image_mode == 'RGB':
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        # h, w = ori_img.shape[:2]
        ori_img = cv2.resize(ori_img, (self.width, self.height))
        rgb_mean = np.array([122.67891434, 116.66876762, 104.00698793])
        image = ori_img
        image -= rgb_mean
        image /= 255.
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        # change (w,h) to (1,img_channel,h,w)
        # tensor = tensor.unsqueeze_(0)

        return {'img': tensor, 'img_path': img_path}
