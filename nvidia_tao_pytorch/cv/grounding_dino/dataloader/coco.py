# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object Detection COCO dataset."""

import glob
import os
from typing import Any, List, Tuple
from PIL import Image

import torch
import torch.utils.data
from torch.utils.data.dataset import Dataset

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.od_dataset import ODDataset


# List of valid image extensions
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".JPEG", ".JPG", ".PNG")


class CocoDetection(ODDataset):
    """COCO Evaluation Dataset.

    Note that category id must be different from the default COCO annotation for COCO dataset.
    The category id must start from 0 and ids should be contiguous.
    """

    def __init__(self, json_file: str = None, dataset_dir: str = None, transforms=None):
        """Initialize COCO ODVG """
        super().__init__(json_file, dataset_dir, transforms)
        category_dict = self.coco.loadCats(self.coco.getCatIds())
        cat_lists = [item['name'] for item in category_dict]
        self.cap_lists = cat_lists
        self.captions = " . ".join(cat_lists) + ' .'

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

        classes = [obj["category_id"] for obj in target]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        classes = classes[keep]

        area = torch.tensor([obj["area"] for obj in target])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in target])

        # Process categories as caption
        # classes, cap_list, caption = self._process_detection_classes(classes)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(height), int(width)])
        target["size"] = torch.as_tensor([int(height), int(width)])

        # # For ODVG
        # target["cap_list"] = cap_list
        # target["caption"] = caption
        return image, target


class ODPredictDataset(Dataset):
    """Base Object Detection Predict Dataset Class."""

    def __init__(self, dataset_list: List[Any], captions: list, transforms=None):
        """Initialize the Object Detetion Dataset Class for inference.

        Unlike ODDataset, this class does not require COCO JSON file.

        Args:
            dataset_list (list): list of dataset directory.
            captions (list): list of captions.
            transforms: augmentations to apply.

        Raises:
            FileNotFoundErorr: If provided classmap, sequence, or image extension does not exist.
        """
        self.dataset_list = dataset_list
        self.transforms = transforms
        self.cap_lists = [c.lower().strip() for c in captions]
        self.captions = ' . '.join(captions) + ' .'

        self.label_map = [{"id": i, "name": c} for i, c in enumerate(self.cap_lists)]

        self.ids = []
        for seq in dataset_list:
            if not os.path.exists(seq):
                raise FileNotFoundError(f"Provided inference directory {seq} does not exist!")

            for ext in VALID_IMAGE_EXTENSIONS:
                self.ids.extend(glob.glob(seq + f"/*{ext}"))
        if len(self.ids) == 0:
            raise FileNotFoundError(f"No valid image with extensions {VALID_IMAGE_EXTENSIONS} found in the provided directories")
        else:
            self.ids = sorted(self.ids)

    def _load_image(self, img_path: int) -> Image.Image:
        """Load image given image path.

        Args:
            img_path (str): image path to load.

        Returns:
            Loaded PIL.Image.
        """
        return_output = (Image.open(img_path).convert("RGB"), img_path)

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
        target["captions"] = self.captions
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, image_path

    def __len__(self) -> int:
        """__len__"""
        return len(self.ids)
