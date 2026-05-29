# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Object Detection Dataset Class and Related Functions """

import torch
from torch.utils.data.dataset import Dataset

import os
import json
import glob
import numpy as np
from PIL import Image, ImageOps
from typing import Any, Tuple, List

from nvidia_tao_pytorch.cv.deformable_detr.utils.coco import COCO
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import read_h5_image_from_path

# List of valid image extensions
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".JPEG", ".JPG", ".PNG")


class ODDataset(Dataset):
    """Base Object Detection Dataset Class."""

    def __init__(self, json_file: str = None, dataset_dir: str = None, transforms=None, contiguous_labels=False):
        """Initialize the Object Detetion Dataset Class.

        Note that multiple loading of COCO type JSON files can lead to system memory OOM.
        In such case, use SerializedDatasetFromList.

        Args:
            json_file (str): json_file name to load the data.
            dataset_dir (str): dataset directory.
            transforms: augmentations to apply.
            contiguous_labels (bool): if True, remap category IDs to contiguous 0-based indices.
        """
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.contiguous_labels = contiguous_labels
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        self.coco = COCO(json_data)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.label_map = self.coco.dataset['categories']
        if self.contiguous_labels:
            cat_ids = sorted([cat['id'] for cat in self.label_map])
            self.cat2contiguous = {cat_id: i for i, cat_id in enumerate(cat_ids)}
            self.contiguous2cat = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    def _load_image(self, img_id: int) -> Image.Image:
        """Load image given image id.

        Args:
            img_id (int): image id to load.

        Returns:
            Loaded PIL Image.
        """
        path = self.coco.loadImgs(img_id)[0]["file_name"]

        if path.startswith("h5://"):
            # Handle h5 file format: h5://[h5_file_base_path]:image_file_name
            img, _ = read_h5_image_from_path(path, self.dataset_dir)  # no need to return full h5 file name
            return_output = (img, path)
        else:
            if not self.dataset_dir == "":
                img_path = os.path.join(self.dataset_dir, path)
            else:
                img_path = path
            img = Image.open(img_path).convert("RGB")
            return_output = (ImageOps.exif_transpose(img), img_path)

        return return_output

    def _load_target(self, img_id: int) -> List[Any]:
        """Load target (annotation) given image id.

        Args:
            img_id (int): image id to load.

        Returns:
            Loaded COCO annotation list
        """
        return self.coco.loadAnns(self.coco.getAnnIds(img_id))

    def _process_image_target(self, image: Image.Image, target: List[Any], img_id: int) -> Tuple[Any, Any]:
        """Process the image and target given image id.

        Args:
            image (PIL.Image): Loaded image given img_id.
            target (list): Loaded annotation given img_id.
            img_id (int): image id to load.

        Returns:
            (image, target): pre-processed image and target for the model.
        """
        width, height = image.size
        image_id = torch.tensor([img_id])

        boxes = [obj["bbox"] for obj in target]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        if self.contiguous_labels:
            classes = [self.cat2contiguous[obj["category_id"]] for obj in target]
        else:
            classes = [obj["category_id"] for obj in target]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        classes = classes[keep]

        area = torch.tensor([obj["area"] for obj in target])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in target])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(height), int(width)])
        target["size"] = torch.as_tensor([int(height), int(width)])

        return image, target

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """Get image, target, image_path given index.

        Args:
            index (int): index of the image id to load.

        Returns:
            (image, target, image_path): pre-processed image, target and image_path for the model.
        """
        img_id = self.ids[index]
        image, image_path = self._load_image(img_id)

        target = self._load_target(img_id)
        image, target = self._process_image_target(image, target, img_id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, image_path

    def __len__(self) -> int:
        """__len__"""
        return len(self.ids)


class ODPredictDataset(Dataset):
    """Base Object Detection Predict Dataset Class."""

    def __init__(self, dataset_list: List[Any], label_map_path: str, transforms=None, start_from_one=True):
        """Initialize the Object Detetion Dataset Class for inference.

        Unlike ODDataset, this class does not require COCO JSON file.

        Args:
            dataset_list (list): list of dataset directory.
            label_map_path (str): label mapping path.
            transforms: augmentations to apply.

        Raises:
            FileNotFoundErorr: If provided classmap, sequence, or image extension does not exist.
        """
        self.dataset_list = dataset_list
        self.transforms = transforms
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"Provided class map {label_map_path} does not exist!")

        # Load classmap and reformat it to COCO categories format
        with open(label_map_path, "r") as f:
            classmap = [line.rstrip() for line in f.readlines()]
        self.label_map = [{"id": i + int(start_from_one), "name": c} for i, c in enumerate(classmap)]

        self.ids = []
        for seq in dataset_list:
            if not os.path.exists(seq):
                raise FileNotFoundError(f"Provided inference directory {seq} does not exist!")

            for ext in VALID_IMAGE_EXTENSIONS:
                self.ids.extend(glob.glob(seq + f"/*{ext}"))
        if len(self.ids) == 0:
            raise FileNotFoundError(f"No valid image with extensions {VALID_IMAGE_EXTENSIONS} found in the provided directories")

    def _load_image(self, img_path: int) -> Image.Image:
        """Load image given image path.

        Args:
            img_path (str): image path to load.

        Returns:
            Loaded PIL.Image.
        """
        img = ImageOps.exif_transpose(Image.open(img_path).convert("RGB"))
        return_output = (img, img_path)

        return return_output

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """Get image, target, image_path given index.

        Args:
            index (int): index of the image id to load.

        Returns:
            (image, target, image_path): pre-processed image, target and image_path for the model.
        """
        img_path = self.ids[index]
        image, image_path = self._load_image(img_path)

        width, height = image.size
        target = {}
        target["orig_size"] = torch.as_tensor([int(height), int(width)])
        target["size"] = torch.as_tensor([int(height), int(width)])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, image_path

    def __len__(self) -> int:
        """__len__"""
        return len(self.ids)


def CoCoDataMerge(coco_list):
    """ Concatenate COCO Dataset.

    We assume that the sharded JSON files were generated using `deformable_detr convert`
    where the ids of the sharded JSON are ensured to be unique.
    We do not perform ID deduplication for faster data loading.

    Args:
        coco_list (list): list of COCO Datasets.

    Returns:
        merged_coco_data (dict) : Merged dictionary in COCO format.
    """
    merged_coco_data = {"images": [], "annotations": [], "categories": None}

    for idx, coco in enumerate(coco_list):
        # Merge all the annotations to single dict
        merged_coco_data["images"].extend(coco.dataset["images"])
        merged_coco_data["annotations"].extend(coco.dataset["annotations"])
        if idx == 0:
            merged_coco_data["categories"] = coco.dataset["categories"]

    return merged_coco_data


class ConcateODDataset(torch.utils.data.ConcatDataset):
    """ Concatenate ODDataset """

    def __init__(self, datasets):
        """Initialize the ConcateODDataset Class.

        Args:
            datasets (iterable): List of datasets to be concatenated.
        """
        super(ConcateODDataset, self).__init__(datasets)
        self.datasets = list(datasets)
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.cum_sizes = np.cumsum([len(x) for x in self.datasets])

        coco_list = []
        for dataset in datasets:
            coco_list.append(dataset.coco)

        self.coco = COCO(CoCoDataMerge(coco_list))
        self.label_map = self.coco.dataset['categories']

    def __len__(self) -> int:
        """Returns length of the concatenated dataset."""
        return self.cum_sizes[-1]

    def __getitem__(self, idx):
        """Get sub-dataset from ConcateODDataset.

        Args:
            idx (int): index to retrieve.

        Returns:
            Sub dataset from the list.
        """
        super(ConcateODDataset, self).__getitem__(idx)
        dataset_index = self.cum_sizes.searchsorted(idx, 'right')

        if dataset_index == 0:
            dataset_idx = idx
        else:
            dataset_idx = idx - self.cum_sizes[dataset_index - 1]

        return self.datasets[dataset_index][dataset_idx]
