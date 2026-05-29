# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optical Inspection dataset."""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.utils import CDDataAugmentation


class SiameseNetworkTRIDataset(Dataset):
    """Siamese Model Dataset Class"""

    def __init__(self, data_frame=None, transform=None,
                 input_data_path=None, train=True, data_config=None):
        """Initialize the SiameseNetworkTRIDataset.

        Args:
            data_frame (pd.DataFrame): The input data frame.
            transform (transforms.Compose): transformation to be applied to the input image samples.
            input_data_path (str): The path to the input data root directory.
            train (bool): Flag indicating whether the dataset is for training.
            data_config (OmegaConf.DictConf): Configuration for the dataset.
        """
        self.data_frame = data_frame
        self.transform = transform
        self.input_image_root = input_data_path
        self.train = train
        self.num_inputs = data_config["num_input"]
        self.concat_type = data_config["concat_type"]
        self.lighting = data_config["input_map"]
        self.grid_map = data_config["grid_map"]
        self.ext = data_config["image_ext"]
        self.output_shape = (data_config["image_height"], data_config["image_width"])
        if self.concat_type == "grid":
            print("Using {} input types and {} type {} X {} for comparison ".format(
                self.num_inputs,
                self.concat_type,
                self.grid_map["x"],
                self.grid_map["y"]
            ))
        else:
            print("Using {} input types and {} type 1 X {} for comparison".format(
                self.num_inputs,
                self.concat_type,
                self.num_inputs
            ))
        augmentation = data_config["augmentation_config"]
        self.augment = augmentation['augment']
        if self.train and self.augment:
            self.augmentor = CDDataAugmentation(
                img_size=self.output_shape,
                random_flip=augmentation['random_flip'],
                random_rotate=augmentation['random_rotate'],
                random_color=augmentation['random_color'],
                with_random_crop=augmentation['with_random_crop'],
                with_random_blur=augmentation['with_random_blur'],
                mean=augmentation['rgb_input_mean'],
                std=augmentation['rgb_input_std'],
            )

    def get_absolute_image_path(self, prefix, light=None):
        """
        Get the absolute image path.

        Args:
            prefix (str): The prefix of the image path.
            light (str): The lighting condition suffix to be appended to image name.

        Returns:
            str: The absolute image path.
        """
        image_path = prefix
        if light:
            image_path += f"_{light}"
        image_path += self.ext
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file wasn't found at {image_path}")
        return image_path

    def __getitem__(self, index):
        """Get the item at a specific index."""
        img_tuple = self.data_frame.iloc[index, :]

        img0, img1 = [], []
        if self.lighting:
            for _, light in enumerate(self.lighting):
                img0.append(
                    Image.open(
                        self.get_absolute_image_path(
                            self.get_compare_paths_v1(img_tuple),
                            light
                        )
                    ).convert("RGB")
                )
                img1.append(
                    Image.open(
                        self.get_absolute_image_path(
                            self.get_golden_paths_v1(img_tuple),
                            light
                        )
                    ).convert("RGB")
                )
        else:
            img0.append(
                Image.open(
                    self.get_absolute_image_path(
                        self.get_compare_paths_v1(img_tuple)
                    )
                ).convert("RGB")
            )
            img1.append(
                Image.open(
                    self.get_absolute_image_path(
                        self.get_golden_paths_v1(img_tuple)
                    )
                ).convert("RGB")
            )

        if self.train:
            # Apply data augmentation
            if self.augment:
                img0T, img1T = self.augmentor.transform(img0, img1, to_tensor=True)
            else:
                img0T = [self.transform(img) for img in img0]
                img1T = [self.transform(img) for img in img1]
        else:
            if self.transform is not None:
                img0T = [self.transform(img) for img in img0]
                img1T = [self.transform(img) for img in img1]
        if self.concat_type == "grid" and int(self.num_inputs) % 2 == 0:
            img0 = self.get_grid_concat(img0T)
            img1 = self.get_grid_concat(img1T)
        else:
            img0 = torch.cat(img0T, 1)
            img1 = torch.cat(img1T, 1)

        label = torch.from_numpy(
            np.array([int(img_tuple['label'] != 'PASS')], dtype=np.float32)
        )
        return img0, img1, label

    def __len__(self):
        """Length of Dataset"""
        return len(self.data_frame)

    def get_grid_concat(self, img_list):
        """Grid Concat"""
        x, y = int(self.grid_map["x"]), int(self.grid_map["y"])
        combined_y = []
        cnt = 0
        for _ in range(0, y, 1):
            combined_x = []
            for j in range(0, x, 1):
                combined_x.append(img_list[cnt])
                cnt += 1
                if j == (x - 1):
                    combined_y.append(torch.cat(combined_x, 2))
        img_grid = torch.cat(combined_y, 1)
        return img_grid

    def get_tot_fail(self):
        """Total FAIL in dataset"""
        return len(self.data_frameFAIL)

    def get_tot_pass(self):
        """Total PASS in dataset"""
        return len(self.data_framePASS)

    def get_compare_paths_v1(self, img_tuple):
        """Get compare file Path"""
        return os.path.join(self.input_image_root, img_tuple['input_path'], img_tuple['object_name'])

    def get_golden_paths_v1(self, img_tuple):
        """Get golden file Path"""
        return os.path.join(self.input_image_root, img_tuple['golden_path'], img_tuple['object_name'])


class MultiGoldenDataset(Dataset):
    """Milti-golden Dataset Class"""

    def __init__(self, data_frame=None, transform=None,
                 input_data_path=None, train=True, data_config=None):
        """Initialize the MultiGoldenDataset.

        Args:
            data_frame (pd.DataFrame): The input data frame.
            transform (transforms.Compose): transformation to be applied to the input image samples.
            input_data_path (str): The path to the input data root directory.
            train (bool): Flag indicating whether the dataset is for training.
            data_config (OmegaConf.DictConf): Configuration for the dataset.
        """
        self.data_frame = data_frame
        self.transform = transform
        self.input_image_root = input_data_path
        self.train = train
        self.num_inputs = data_config["num_input"]
        self.concat_type = data_config["concat_type"]
        self.lighting = data_config["input_map"]
        self.grid_map = data_config["grid_map"]
        self.num_golden = data_config['num_golden']
        self.ext = data_config["image_ext"]
        self.output_shape = (data_config["image_height"], data_config["image_width"])
        if self.concat_type == "grid":
            print("Using {} input types and {} type {} X {} for comparison ".format(
                self.num_inputs,
                self.concat_type,
                self.grid_map["x"],
                self.grid_map["y"]
            ))
        else:
            print("Using {} input types and {} type 1 X {} for comparison".format(
                self.num_inputs,
                self.concat_type,
                self.num_inputs
            ))
        augmentation = data_config["augmentation_config"]
        self.augment = augmentation['augment']
        if self.train and self.augment:
            self.augmentor = CDDataAugmentation(
                img_size=self.output_shape,
                random_flip=augmentation['random_flip'],
                random_rotate=augmentation['random_rotate'],
                random_color=augmentation['random_color'],
                with_random_crop=augmentation['with_random_crop'],
                with_random_blur=augmentation['with_random_blur'],
                mean=augmentation['rgb_input_mean'],
                std=augmentation['rgb_input_std'],
            )

    def get_absolute_image_path(self, prefix, light=None):
        """
        Get the absolute image path.

        Args:
            prefix (str): The prefix of the image path.
            light (str): The lighting condition suffix to be appended to image name.

        Returns:
            str: The absolute image path.
        """
        image_path = prefix
        if light:
            image_path += f"_{light}"
        image_path += self.ext
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file wasn't found at {image_path}")
        return image_path

    def __getitem__(self, index):
        """Get the item at a specific index."""
        img_tuple = self.data_frame.iloc[index, :]

        # Precompute paths
        compare_path = self.get_compare_paths_v1(img_tuple)
        golden_paths = self.get_multi_golden_paths(img_tuple, self.num_golden)

        def load_images(base_path, lights=None):
            images = []
            if lights:
                for light in lights:
                    img_path = self.get_absolute_image_path(base_path, light)
                    images.append(Image.open(img_path).convert("RGB"))
            else:
                img_path = self.get_absolute_image_path(base_path)
                images.append(Image.open(img_path).convert("RGB"))
            return images

        img0 = load_images(compare_path, self.lighting)
        goldens = [load_images(path, self.lighting) for path in golden_paths]

        # Flatten the goldens list
        flat_goldens = [img for sublist in goldens for img in sublist]

        # Apply data augmentation if needed
        if self.augment and self.train:
            img0T, flat_goldensT = self.augmentor.transform(img0, flat_goldens, to_tensor=True)
        else:
            img0T = [self.transform(img) for img in img0] if self.transform else img0
            flat_goldensT = [self.transform(img) for img in flat_goldens] if self.transform else flat_goldens
        # Reshape flat_goldensT back to the original structure of goldens
        goldensT = []
        idx = 0
        for sublist in goldens:
            length = len(sublist)
            if length > 0:
                goldensT.append(flat_goldensT[idx:idx + length])
            idx += length
        # Concatenate images if required
        if self.concat_type == "grid" and int(self.num_inputs) % 2 == 0:
            img0 = self.get_grid_concat(img0T)
            goldens = torch.stack([self.get_grid_concat(g) for g in goldensT], dim=0)
        else:
            img0 = torch.cat(img0T, 1)
            goldens = torch.stack([torch.cat(g, 1) for g in goldensT], dim=0)

        label = torch.tensor([int(img_tuple['label'] != 'PASS')], dtype=torch.float32)

        return img0, goldens, label

    def __len__(self):
        """Length of Dataset"""
        return len(self.data_frame)

    def get_grid_concat(self, img_list):
        """Grid Concat"""
        x, y = int(self.grid_map["x"]), int(self.grid_map["y"])
        combined_y = []
        cnt = 0
        for _ in range(0, y, 1):
            combined_x = []
            for j in range(0, x, 1):
                combined_x.append(img_list[cnt])
                cnt += 1
                if j == (x - 1):
                    combined_y.append(torch.cat(combined_x, 2))
        img_grid = torch.cat(combined_y, 1)
        return img_grid

    def get_tot_fail(self):
        """Total FAIL in dataset"""
        return len(self.data_frameFAIL)

    def get_tot_pass(self):
        """Total PASS in dataset"""
        return len(self.data_framePASS)

    def get_compare_paths_v1(self, img_tuple):
        """Get compare file Path"""
        return os.path.join(self.input_image_root, img_tuple['input_path'], img_tuple['object_name'])

    def get_multi_golden_paths(self, img_tuple, N=1):
        """Get N golden file paths, sampled randomly without replacement.
        If fewer than N images exist, duplicate selected images to make up N.
        """
        golden_dir = os.path.join(self.input_image_root, img_tuple['golden_path'])

        golden_images = [os.path.join(golden_dir, os.path.splitext(f)[0]) for f in os.listdir(golden_dir) if f.endswith(self.ext)]
        if self.lighting:
            for light in self.lighting:
                for i in range(len(golden_images)):
                    # remove the suffix of lighting and remove duplicate golden paths
                    if golden_images[i].endswith(f"_{light}"):
                        golden_images[i] = golden_images[i].replace(f"_{light}", '')
            golden_images = list(set(golden_images))

        assert len(golden_images) > 0, "No golden images found in {}".format(golden_dir)

        if len(golden_images) < N:
            selected_images = random.choices(golden_images, k=N)
        else:
            selected_images = random.sample(golden_images, N)

        return selected_images


def get_sampler(dataset, train_sampl_ratio=0.1):
    """
    Returns a weighted sampler for imbalanced dataset.

    The weighted sampler increases the sampling rate of FAIL instances relative to PASS instances.

    Args:
        dataset (SiameseNetworkTRIDataset): The input dataset.
        train_sampl_ratio (float): The ratio to increase the sampling rate of FAIL instances.

    Returns:
        WeightedRandomSampler: The weighted random sampler object.
    """
    n = dataset.data_frame.shape[0]
    target_list = [0] * n
    df_ = dataset.data_frame.copy()
    fail_indices = [i for i in range(df_.shape[0]) if df_['label'].iloc[i] != 'PASS']

    for i in fail_indices:
        target_list[i] = 1
    num_pass = dataset.data_frame.shape[0] - len(fail_indices)
    pf_ratio = num_pass / len(fail_indices)
    fail_wt = pf_ratio * train_sampl_ratio

    logging.info('\nSampling Defective components at {:05.2f}:1 rate'.format(fail_wt))
    class_weights = torch.tensor([1, fail_wt], dtype=torch.float)
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True)

    return weighted_sampler
