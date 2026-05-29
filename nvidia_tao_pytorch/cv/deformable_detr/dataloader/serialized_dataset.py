# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
List serialization code adopted from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py
"""
import os
import contextlib
import io
import torch

from typing import Dict, List, Any
from pycocotools.coco import COCO
from PIL import Image, ImageOps

from nvidia_tao_pytorch.cv.deformable_detr.utils.data_source_config import build_data_source_lists
from nvidia_tao_pytorch.core.distributed.comm import get_local_rank
from nvidia_tao_pytorch.core.distributed.serialized_object import TorchShmSerializedList


def load_coco_json(json_file: str, image_root: str) -> List[Dict]:
    """Load COCO json file and return list of dictionaries.

    Referenced from detectron2: https://detectron2.readthedocs.io/en/latest/_modules/detectron2/data/datasets/coco.html

    Args:
        json_file (str): Path to the JSON annotation file.
        image_root (str): Path to root directory of images from the annotations.

    Returns:
        List of COCO annotation dicts.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id", "area"]
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def build_shm_dataset(data_sources, transforms):
    """Preload the COCO ann lists to prevent memory leakage from Python.

    Args:
        data_sources (str): list of different data sources.
        transforms (dict): augmentations to apply.
    """
    # grab all the json files and concate them into one single dataset
    data_source_list = build_data_source_lists(data_sources)
    dataset_list = []
    for data_source in data_source_list:
        image_dir = data_source.image_dir
        for _json_file in data_source.dataset_files:
            if get_local_rank() == 0:
                dl = load_coco_json(_json_file, image_root=image_dir)
                dataset_list.extend(dl)
    dataset = SerializedDatasetFromList(dataset_list, transforms=transforms)
    return dataset


class SerializedDatasetFromList(torch.utils.data.Dataset):
    """
    Hold memory using serialized objects so that data loader workers can use
    shared RAM from master process instead of making a copy in each subprocess.
    """

    def __init__(self, lst, transforms=None):
        """Initialize the Serialized Shared Memory COCO-based Dataset.

        Reference from this blog: https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/

        Args:
            lst (list): list of dataset dicts.
            transforms (dict): augmentations to apply.
        """
        self.transforms = transforms
        self.metas = TorchShmSerializedList(lst)

    def __len__(self):
        """__len__"""
        return len(self.metas)

    def _process_image_target(self, image: Image.Image, target: List[Any], img_id: int):
        """Process the image and target given image id.

        Args:
            image (PIL.Image): Loaded Pillow Image .
            target (list): Loaded annotation given img_id.
            img_id (int): image id to load.

        Returns:
            target (dict): pre-processed target.
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

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(height), int(width)])
        target["size"] = torch.as_tensor([int(height), int(width)])

        return target

    def __getitem__(self, idx: int):
        """Get image, target, image_path given index,

        Args:
            index (int): index of the image id to load

        Returns:
            (image, target, image_path): pre-processed image, target and image_path for the model

        """
        record = self.metas[idx]
        image_path = record['file_name']
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        img_id = record["image_id"]

        target = record['annotations']
        target = self._process_image_target(image, target, img_id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, image_path
