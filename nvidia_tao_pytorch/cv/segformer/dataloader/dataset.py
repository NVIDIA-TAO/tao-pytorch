# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SegFormer Dataset"""

import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.cv.segformer.dataloader.augmentation import CDDataAugmentation


class SFDataset(Dataset):
    """
    SegFormer Dataset
    """

    def __init__(self, root_dir, augmentation=None, split='train', img_size=256, label_transform=None, to_tensor=True, color_map=None):
        """Initialize

        Args:
            root_dir (str): The root directory of the dataset.
            augmentation (dict): A dictionary containing the augmentation parameters.
            split (str): The dataset split (folder name) to load. valid values are 'train', 'val', 'test'.
            img_size (int): The size of the image to load.
            label_transform (str): The label transformation to apply. valid values are 'norm', None. If 'norm', the RGB will be normalized to [0, 1] before calling color_map.
            to_tensor (bool): Convert the image to tensor.
            color_map (dict): The color map to use for RGB to train label transformation.
        """
        super(SFDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | val | test
        self.label_transform = label_transform
        self.img_name_list = sorted(os.listdir(os.path.join(self.root_dir, "images", self.split)))

        if self.split in ["train", "val"]:
            self.mask_name_list = sorted(os.listdir(os.path.join(self.root_dir, "masks", self.split)))
            assert len(self.img_name_list) == len(self.mask_name_list), "Number of images and masks should be the same."

        self.dataset_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        self.color_map = color_map
        # inverse the color map to get color to id map
        if self.color_map is not None:
            self.color_to_id_map = {v: k for k, v in self.color_map.items()}
        else:
            raise ValueError("Color map is required for label transformation. Define palette in the config file dataset section.")

        cdd_kwargs = {
            "img_size": self.img_size,
            "mean": augmentation["mean"],
            "std": augmentation["std"],
        }

        if self.split == "train":
            cdd_kwargs.update({
                "random_flip": augmentation['random_flip'],
                "random_rotate": augmentation['random_rotate'],
                "random_color": augmentation['random_color'],
                "with_scale_random_crop": augmentation['with_scale_random_crop'],
                "with_random_crop": augmentation['with_random_crop'],
                "with_random_blur": augmentation['with_random_blur']
            })

        self.augmentor = CDDataAugmentation(**cdd_kwargs)

    def __getitem__(self, index):
        """
        Get a pair of augmented images and their names from the dataset.

        Args:
            index (int): Index of the image pair to retrieve.

        Returns:
            dict: A dictionary containing two augmented images ('A' and 'B') and the image name ('name').
        """
        name = self.img_name_list[index]
        img_path = self.get_img_path(self.root_dir, "images", self.split, name)
        img = np.asarray(Image.open(img_path).convert('RGB'))

        if self.split in ["train", "val"]:
            mask_name = self.mask_name_list[index]
            mask_path = self.get_img_path(self.root_dir, "masks", self.split, mask_name)

            # Load the mask image based on the color_to_id_map
            if self.color_to_id_map is not None:
                mask = Image.open(mask_path)
                # check whether the mask is in RGB format or not (if not then it's in grayscale)
                if len(list(self.color_to_id_map.keys())[0]) == 3:
                    mask = np.asarray(mask.convert('RGB'))
                else:
                    mask = np.expand_dims(np.asarray(mask), axis=2)
                if self.label_transform == 'norm':
                    mask = mask // 255
                # Convert the RGB mask to id map
                # method1: trasverse each id in the color_to_id_map
                id_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
                for color, id_value in self.color_to_id_map.items():
                    id_map[(mask == color).all(axis=-1)] = id_value

            mask = id_map

            [img], [mask] = self.augmentor.transform([img], [mask], to_tensor=self.to_tensor)
            return {'img': img, 'mask': mask, 'name': name}

        else:
            [img], _ = self.augmentor.transform([img], [], to_tensor=self.to_tensor)
            return {'img': img, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.dataset_size

    def get_img_path(self, root_dir, folder_name, split_name, img_name):
        """
        Get the full path of an image given its filename and folder name.
        """
        return expand_path(os.path.join(root_dir, folder_name, split_name, img_name))
