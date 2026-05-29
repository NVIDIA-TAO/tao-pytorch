# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RADIO Dataset"""

import os
import glob
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.multimodal.radio.dataloader.augmentation import (
    CLDataAugmentation,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 9000000000
NOCLASS_IDX = -1


class CLDataset(Dataset):
    """
    RADIO Dataset

    Args:
        root_dir (str): The root directory of the dataset.
        data_path (str): The path of the image folders.
        augmentation (dict): A dictionary containing the augmentation parameters.
        split (str): The split of the dataset (train | val | test).
        nolabel_folder (str): Path to image folder with no labels(unstructured data)
        img_size (int): The size of the images after resizing.
        to_tensor (bool): Convert the images to tensors.
    """

    def __init__(
        self,
        root_dir,
        data_path,
        augmentation,
        nolabel_folder=None,
        split="train",
        img_size=256,
        to_tensor=True,
    ):
        """Initialize"""
        super(CLDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | val | test
        self.data_path = data_path
        self.nolabel_folder = nolabel_folder

        self.class_names = {}

        # read class.txt and each line is a class name
        # find wheter the class.txt is in the root_dir
        if os.path.exists(os.path.join(self.root_dir, "classes.txt")):
            with open(os.path.join(self.root_dir, "classes.txt")) as f:
                for idx, line in enumerate(f):
                    self.class_names[line.strip()] = idx
        else:
            if self.data_path:
                class_names = sorted(os.listdir(self.data_path))
                for idx, class_name in enumerate(class_names):
                    self.class_names[class_name] = idx
                # write the class.txt
                with open(os.path.join(self.root_dir, "classes.txt"), "w") as f:
                    for class_name in class_names:
                        f.write(f"{class_name}\n")

        if split == "train" or split == "val":
            self.img_name_list = self.get_image_file_names(inference=False)
        else:
            self.img_name_list = self.get_image_file_names(inference=True)

        self.dataset_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor

        if augmentation is None:
            raise ValueError("Augmentation is required for classification dataset. Define augmentation in the config file dataset section.")

        is_training = self.split == "train"
        aug_kwargs = {
            "img_size": self.img_size,
            "mean": augmentation["mean"],
            "is_training": is_training,
        }

        if is_training:
            aug_kwargs.update(
                {
                    "random_flip": augmentation.get("random_flip"),
                    "random_rotate": augmentation.get("random_rotate"),
                    "random_color": augmentation.get("random_color"),
                    "random_erase": augmentation.get("random_erase"),
                    "random_aug": augmentation.get("random_aug"),
                    "with_scale_random_crop": augmentation.get("with_scale_random_crop"),
                    "with_random_crop": augmentation.get("with_random_crop", False),
                    "with_random_blur": augmentation.get("with_random_blur", False),
                    "patch_size": (p if (p := augmentation.get("patch_size")) else None),
                    "use_continuous_rotation": augmentation.get("use_continuous_rotation", False),
                    "perspective_distortion": augmentation.get("perspective_distortion"),
                }
            )

        self.augmentor = CLDataAugmentation(**aug_kwargs)

    def __getitem__(self, index):
        """
        Get a pair of augmented images and their names from the dataset.

        Args:
            index (int): Index of the image pair to retrieve.

        Returns:
            dict: A dictionary containing two augmented images ('A' and 'B') and the image name ('name').
        """
        img_path = self.img_name_list[index]
        img_path = self.get_img_path(img_path)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {e}")

        # record the h,w of the image for visualization
        h, w = img.size

        if self.split == "train" or self.split == "val":

            # Check if the image is in the nolabel folder
            is_structured = (
                img_path.find(self.nolabel_folder) == -1
                if self.nolabel_folder and self.split == "train"
                else True
            )
            split_img_path = img_path.split("/")
            class_name = (
                split_img_path[-2]
                if len(split_img_path) >= 2 and is_structured
                else "nolabel"
            )
            c = self.class_names.get(class_name, NOCLASS_IDX)

            [img] = self.augmentor.transform([img], to_tensor=self.to_tensor)
            return {"img": img, "class": c, "name": img_path}

        else:
            [img] = self.augmentor.transform([img], to_tensor=self.to_tensor)
            return {"img": img, "name": img_path, "size": (h, w)}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.dataset_size

    def get_image_file_names(self, inference=False, suffix=["jpg", "png", "JPEG"]):
        """
        Get the list of image file names in the dataset.

        Args:
            inference (bool): If True, return the image file names for inference. If inference, then the class folders are ignored (Get every img files under the data_path folder).
            suffix (list): A list of image file suffixes.

        Returns:
            list: A list of image file names.
        """
        img_name_list = []
        if inference:
            for s in suffix:
                img_name_list.extend(
                    glob.glob(
                        os.path.join(self.data_path, f"**/*.{s}"),
                        recursive=True,
                    )
                )
        else:
            for s in suffix:
                for class_name in self.class_names:
                    img_name_list.extend(
                        glob.glob(
                            os.path.join(
                                self.data_path, class_name, f"*.{s}"
                            )
                        )
                    )

                if self.nolabel_folder:
                    img_name_list.extend(
                        glob.glob(
                            os.path.join(self.nolabel_folder, f"**/*.{s}"),
                            recursive=True,
                        )
                    )

        return img_name_list

    def get_img_path(self, path):
        """
        Get the full path of an image given its filename and folder name.
        """
        return expand_path(path)

    def collate_fn(self, batch):
        """Collate items in a batch."""
        out = {}
        images = []
        labels = []

        for item in batch:
            images.append(item["img"])
            labels.append(item["class"])
        out["img"] = torch.stack(images)
        out["class"] = torch.tensor(labels)
        return out
