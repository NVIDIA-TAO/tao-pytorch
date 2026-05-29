# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stereo Dataset Module """

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from nvidia_tao_pytorch.cv.depth_net.dataloader.utils.frame_utils import read_disparity, read_gt_crestereo, read_image


def get_file_paths(file_path):
    """
    Function to split or parse single line in a text file

    Args:
        file_path (str): A string containing a file paths sepatered by a ' '.

    Returns:
        the split paths
    """
    files = file_path.split(' ')
    if len(files) < 3:
        left_img_path, right_img_path = files
        disp_path = None
    elif len(files) == 3:
        left_img_path, right_img_path, disp_path = files
    else:
        raise NotImplementedError('Only 3 split paths are supported for generic file paths!')
    return left_img_path, right_img_path, disp_path


def apply_transforms(transform, left_image, right_image, disparity, max_disparity, left_img_path=None):
    """
    Applies transformations to the input images and disparity map, and converts them to PyTorch tensors.

    Args:
        transform (callable, optional): A composed transformation pipeline. If None, no transformations are applied.
        left_image (numpy.ndarray): The left stereo image.
        right_image (numpy.ndarray): The right stereo image.
        disparity (numpy.ndarray): The disparity map.
        left_img_path (str, optional): The file path of the left image. Used for debugging or tracking.

    Returns:
        dict: A dictionary containing the transformed images, disparity, image size, and image path.
              Keys include 'image', 'right_image', 'disparity', 'image_size', and 'image_path'.
    """
    if transform is not None:
        # Stack disparity with a zero-like array to match expected input for some transforms (e.g., for channels)
        disparity = np.stack([disparity, np.zeros_like(disparity)], axis=-1)
        sample = transform({'image': left_image, 'right_image': right_image, 'disparity': disparity})
    else:
        sample = {'image': left_image, 'right_image': right_image, 'disparity': disparity}

    image_size = left_image.shape
    sample['image'] = torch.from_numpy(sample['image'])
    sample['right_image'] = torch.from_numpy(sample['right_image'])
    sample['image_size'] = image_size
    sample['disparity'] = torch.from_numpy(sample['disparity'])
    sample['valid_mask'] = sample['disparity'] < max_disparity
    sample['image_path'] = left_img_path
    return sample


class StereoDataset(Dataset):
    """
    Base class for stereo datasets. Provides common initialization and
    length measurement functionality. Subclasses must implement `__getitem__`.

    Args:
        dataset_config (OmegaConf.DictConfig): Configuration for the dataset, expected to contain "data_file".
        transform (callable, optional): A composed transformation pipeline to be applied to the samples.
        max_disparity (float): The maximum disparity value in the dataset.
    """

    def __init__(self, data_file, transform, max_disparity):
        self.sample = {}
        self.max_disparity = max_disparity
        self.transform = transform

        with open(data_file, 'r') as f:
            self.filelist = f.read().splitlines()
            self.filelist = [x.strip() for x in self.filelist]

    def __getitem__(self, index):
        """
        Abstract method to be implemented by various datasets.
        This method should load and return a single sample (left image, right image, disparity).

        Args:
            index (int): The index of the sample to retrieve.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.filelist)


class GenericDataset(StereoDataset):
    """
    The generic class can process inference for datasets without groundtruth disparity
    Dataset class for any generic dataset.
    Inherits from StereoDataset.
    Initializes the GenericDataset class.
    """

    def __getitem__(self, index):
        """
        Retrieves a sample from the FSD dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the transformed images, disparity, image size, and image path.
                  Keys include 'image', 'right_image', 'disparity', 'image_size', and 'image_path'.
        """
        left_img_path, right_img_path, disp_path = get_file_paths(self.filelist[index])
        left_image = read_image(left_img_path)
        right_image = read_image(right_img_path)
        if disp_path is None:
            disparity = np.zeros([left_image.shape[0], left_image.shape[1]]) - 1
        else:
            disparity = read_disparity(disp_path)
        return apply_transforms(self.transform, left_image, right_image, disparity, self.max_disparity, left_img_path)


class FSD(StereoDataset):
    """
    Dataset class for the FSD (FoundationStereo Dataset) dataset.
    Inherits from StereoDataset.
    Initializes the FSD dataset.
    """

    def __getitem__(self, index):
        """
        Retrieves a sample from the FSD dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the transformed images, disparity, image size, and image path.
                  Keys include 'image', 'right_image', 'disparity', 'image_size', and 'image_path'.
        """
        left_img_path, right_img_path, disp_path = self.filelist[index].split(' ')
        left_image = read_image(left_img_path)
        right_image = read_image(right_img_path)
        disparity = read_disparity(disp_path)
        return apply_transforms(self.transform, left_image, right_image, disparity, self.max_disparity, left_img_path)


class IsaacRealDataset(FSD):
    """
    Dataset class for the Isaac stereo dataset.
    Inherits from FSD as it shares similar data loading logic.
    Initializes the IsaacReal dataset.
    """


class Crestereo(StereoDataset):
    """
    Dataset class for the CREStereo dataset.
    Inherits from StereoDataset.
    Initializes the CREStereo dataset.
    """

    def __getitem__(self, index):
        """
        Retrieves a sample from the CREStereo dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the transformed images, disparity, image size, and image path.
                  Keys include 'image', 'right_image', 'disparity', 'image_size', and 'image_path'.
        """
        left_img_path, right_img_path, disp_path = self.filelist[index].split(' ')
        left_image = read_image(left_img_path)
        right_image = read_image(right_img_path)
        disparity = read_gt_crestereo(disp_path)  # Specific disparity reading for CREStereo
        return apply_transforms(self.transform, left_image, right_image, disparity, self.max_disparity, left_img_path)


class Middlebury(StereoDataset):
    """
    Dataset class for the Middlebury stereo dataset.
    Inherits from StereoDataset. Handles optional non-occluded mask.
    Initializes the Middlebury dataset.
    """

    def __getitem__(self, index):
        """
        Retrieves a sample from the Middlebury dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the transformed images, disparity, image size, and image path.
                  Keys include 'image', 'right_image', 'disparity', 'image_size', and 'image_path'.
        """
        files = self.filelist[index].split(' ')
        if len(files) == 4:
            left_img_path, right_img_path, disp_path, nocc_mask_path = files
        elif len(files) == 3:
            left_img_path, right_img_path, disp_path = files
            nocc_mask_path = None
        else:
            raise ValueError(f"Unexpected number of files in line {index}: {len(files)}")

        left_image = read_image(left_img_path).copy()
        right_image = read_image(right_img_path).copy()
        disparity = read_disparity(disp_path).copy()

        if nocc_mask_path is not None:
            nocc_mask = read_image(nocc_mask_path)
            # Assuming nocc_mask is grayscale and we take the first channel
            nocc_mask = np.asarray(nocc_mask, dtype=np.float32)[:, :, 0]
            # Invalidate disparity where non-occluded mask is not 255
            disparity[nocc_mask != 255] = np.inf
        return apply_transforms(self.transform, left_image, right_image, disparity, self.max_disparity, left_img_path)


class Eth3d(Middlebury):
    """
    Dataset class for the ETH3D stereo dataset.
    Inherits from Middlebury as it shares similar data loading logic.
    Initializes the ETH3D dataset.
    """


class Kitti(StereoDataset):
    """
    Dataset class for the KITTI stereo dataset.
    Inherits from StereoDataset.
    Initializes the KITTI dataset.
    """

    def __getitem__(self, index):
        """
        Retrieves a sample from the KITTI dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the transformed images, disparity, image size, and image path.
                  Keys include 'image', 'right_image', 'disparity', 'image_size', and 'image_path'.
        """
        files = self.filelist[index].split(' ')
        if len(files) != 3:
            raise ValueError(f"Unexpected number of files in line {index} for KITTI: {len(files)}")
        image_path, right_image_path, disp_path = files

        left_image = read_image(image_path)
        right_image = read_image(right_image_path)

        # KITTI disparity is typically stored as 16-bit PNG, where value / 256.0 gives disparity in pixels
        disparity = cv2.imread(disp_path, -1) / 256.0

        # Invalidate disparity where it's 0 ( KITTI ground truth disparity convention)
        disparity[disparity == 0] = np.inf

        return apply_transforms(self.transform, left_image, right_image, disparity, self.max_disparity, image_path)
