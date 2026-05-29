# Copyright (c) 2023 Chaminda Bandara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Original source taken from https://github.com/wgcban/ChangeFormer
#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual ChangeNet Segmentation Dataloader"""

import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

from nvidia_tao_pytorch.cv.visual_changenet.segmentation.dataloader.data_utils_cn import CDDataAugmentation
from nvidia_tao_pytorch.core.path_utils import expand_path


"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""

IGNORE = 255


def load_image_label_list_from_npy(npy_path, img_name_list):
    """
    Load image labels from a NumPy file and extract labels for the given image names.

    Parameters:
        npy_path (str): The file path to the NumPy file containing the class labels.
        img_name_list (List[str]): A list of image names for which the labels are required.

    Returns:
        List: A list containing the class labels corresponding to the provided image names.
    """
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


class ImageDataset(Dataset):
    """
    VOCdataloder
    A custom dataset for loading paired images with optional data augmentation.
    """

    def __init__(self, root_dir, augmentation=None, split='train', img_size=256, is_train=True, to_tensor=True,
                 a_dir='A', b_dir='B', label_dir='label', list_dir='list', label_suffix='.png'):
        """Initialize"""
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        self.list_path = expand_path(os.path.join(self.root_dir, list_dir, self.split + '.txt'))
        self.img_name_list = self.load_img_name_list(self.list_path)

        self.a_dir = a_dir
        self.b_dir = b_dir
        self.label_dir = label_dir
        self.label_suffix = label_suffix

        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augmentor = CDDataAugmentation(
                img_size=self.img_size,
                random_flip=augmentation['random_flip'],
                random_rotate=augmentation['random_rotate'],
                random_color=augmentation['random_color'],
                with_scale_random_crop=augmentation['with_scale_random_crop'],
                with_random_crop=augmentation['with_random_crop'],
                with_random_blur=augmentation['with_random_blur'],
                mean=augmentation['mean'],
                std=augmentation['std'],
            )
        else:
            self.augmentor = CDDataAugmentation(
                img_size=self.img_size
            )

    def __getitem__(self, index):
        """
        Get a pair of augmented images and their names from the dataset.

        Args:
            index (int): Index of the image pair to retrieve.

        Returns:
            dict: A dictionary containing two augmented images ('A' and 'B') and the image name ('name').
        """
        name = self.img_name_list[index]
        A_path = self.get_img_path(self.root_dir, self.img_name_list[index % self.A_size], self.a_dir)
        B_path = self.get_img_path(self.root_dir, self.img_name_list[index % self.A_size], self.b_dir)

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        [img, img_B], _ = self.augmentor.transform([img, img_B], [], to_tensor=self.to_tensor)
        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size

    def load_img_name_list(self, dataset_path):
        """
        Load the list of image filenames from a given file.
        """
        img_name_list = np.loadtxt(dataset_path, dtype=str)
        if img_name_list.ndim == 2:
            return img_name_list[:, 0]
        return img_name_list

    def get_img_path(self, root_dir, img_name, folder_name):
        """
        Get the full path of an image given its filename and folder name.
        """
        return expand_path(os.path.join(root_dir, folder_name, img_name))

    def get_label_path(self, root_dir, img_name, folder_name):
        """
        Get the full path of a label image given its filename and folder name.
        """
        return expand_path(os.path.join(root_dir, folder_name, img_name.replace('.jpg', self.label_suffix)))


class CNDataset(ImageDataset):
    """
    Custom dataset for loading paired images
    Inherits from ImageDataset for common functionality.

    Args:
        root_dir (str): Root directory of the dataset.
        img_size (int): Size to which the images will be resized.
        split (str, optional): Dataset split ('train', 'val', 'test', 'predict'). Default is 'train'.
        is_train (bool, optional): Whether the dataset is used for training. Default is True.
        label_transform (str, optional): Label transformation type ('norm'). Default is None.
        to_tensor (bool, optional): Convert images to PyTorch tensors. Default is True.
        a_dir (str, optional): Directory name for images 'A' (Test Images). Default is 'A'.
        b_dir (str, optional): Directory name for images 'B' (Compare Images). Default is 'B'.
        label_dir (str, optional): Directory name for labels. Default is 'label'.
        list_dir (str, optional): Directory name for image list. Default is 'list'.
        augmentation (Dict, optional): A dictionary containing parameters for each augmentation to be applied
        label_suffix (str, optional): Label image file suffix. Default is '.png'.
    """

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 to_tensor=True, a_dir='A', b_dir='B', label_dir='label', list_dir='list', augmentation=None, label_suffix='.png'):
        """Initialize"""
        super(CNDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor, a_dir=a_dir, b_dir=b_dir, label_dir=label_dir, list_dir=list_dir,
                                        augmentation=augmentation, label_suffix=label_suffix)
        self.label_transform = label_transform

    def __getitem__(self, index):
        """
        Get a pair of augmented images and their names from the dataset.

        Args:
            index (int): Index of the image pair to retrieve.

        Returns:
            dict: A dictionary containing two augmented images ('A' and 'B') and the image name ('name').
                It also returns the label during train, validation and evaluation and not during inference.
        """
        name = self.img_name_list[index]
        A_path = self.get_img_path(self.root_dir, self.img_name_list[index % self.A_size], self.a_dir)
        B_path = self.get_img_path(self.root_dir, self.img_name_list[index % self.A_size], self.b_dir)
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        if self.split != 'predict':
            L_path = self.get_label_path(self.root_dir, self.img_name_list[index % self.A_size], self.label_dir)
            label = np.array(Image.open(L_path), dtype=np.uint8)
            if self.label_transform == 'norm':
                label = label // 255

            [img, img_B], [label] = self.augmentor.transform([img, img_B], [label], to_tensor=self.to_tensor)
            return {'name': name, 'A': img, 'B': img_B, 'L': label}

        [img, img_B], _ = self.augmentor.transform([img, img_B], [], to_tensor=self.to_tensor)
        return {'name': name, 'A': img, 'B': img_B}
