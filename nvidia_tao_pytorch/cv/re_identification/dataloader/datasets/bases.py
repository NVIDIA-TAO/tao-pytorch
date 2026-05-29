# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base Module for all datasets."""

from tabulate import tabulate
from nvidia_tao_pytorch.cv.re_identification.utils.common_utils import read_image
from torch.utils.data import Dataset


class BaseDataset(object):
    """Base class for all datasets.

    This class serves as the base class for all re-identification datasets.
    It provides methods for retrieving metadata from the images.

    """

    def get_imagedata_info(self, data):
        """Return metadata from the images.

        Args:
            data (list): A list of tuples containing image data.

        Returns:
            tuple: A tuple containing the number of unique person IDs, the total number of images,
                and the number of unique camera IDs.

        """
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        """Base class for image re-identification datasets.

        This class inherits from BaseDataset and provides a method to print dataset statistics.

        """
        raise NotImplementedError("Printing dataset statistics is not implemented.")


class BaseImageDataset(BaseDataset):
    """Base class for image re-identification datasets.

    This class inherits from BaseDataset and provides a method to print dataset statistics.

    """

    def print_dataset_statistics(self, *args):
        """Print the dataset statistics.

        This method prints the number of person IDs, number of images, and number of cameras
        for each subset of the dataset.

        Args:
            *args: Variable length argument list of datasets.

        """
        table = []

        if len(args) == 3:
            dataset_type = ["Train", "Query", "Gallery"]
        elif len(args) == 2:
            dataset_type = ["Query", "Gallery"]
        for index, dataset in enumerate(args):
            num_pids, num_imgs, num_cams = self.get_imagedata_info(dataset)
            table.append([dataset_type[index], num_pids, num_imgs, num_cams])
        print(tabulate(table, headers=["Subset", "# IDs", "# Images", "# Cameras"], floatfmt=".4f", tablefmt="fancy_grid"))


class ImageDataset(Dataset):
    """Dataset class for images.

    This class stores images, object IDs, camera IDs, and image paths.

    """

    def __init__(self, dataset, transform=None):
        """Initialize the ImageDataset.

        Args:
            dataset (list): A list of tuples containing image data.
            transform (callable, optional): A function/transform to apply to the images. Defaults to None.

        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Return the image, person ID, camera ID, and image path for a given index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image, person ID, camera ID, and image path.

        """
        assert index < len(self.dataset), f"Index {index} out of bounds!"
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path
