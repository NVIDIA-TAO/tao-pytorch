# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Object Detection Visual Genome Dataset """

import json
import random
import os

from typing import Callable, Optional
from PIL import Image, ImageOps

import torch
from torch.utils.data import ConcatDataset
from torchvision.datasets.vision import VisionDataset

from nvidia_tao_pytorch.core.tlt_logging import logging


def build_odvg(data_sources, transforms, max_labels=50):
    """Load dataset

    Args:
        data_sources (str): list of different data sources.
        transforms (dict): augmentations to apply.
        max_labels (int): max number of labels to sample.
    """
    if type(data_sources).__name__ == "DictConfig":
        data_sources = [data_sources]

    dataset_list = []
    for data_source in data_sources:
        image_dir = data_source.image_dir
        json_file = data_source.json_file
        label_map = data_source.label_map if "label_map" in data_source else None
        dataset_list.append(ODVGDataset(image_dir, json_file, label_map,
                                        max_labels=max_labels,
                                        transforms=transforms))

        if len(dataset_list) > 1:
            train_dataset = ConcatDataset(dataset_list)
        else:
            train_dataset = dataset_list[0]
    return train_dataset


class ODVGDataset(VisionDataset):
    """Obejct Detection Visual Genome Dataset."""

    def __init__(
        self,
        root: str,
        anno: str,
        label_map_anno: str = None,
        max_labels: int = 80,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        """Initialize ODVG dataset.
        Args:
            root (string): Root directory where images are downloaded to.
            anno (string): Path to json annotation file.
            label_map_anno (string):  Path to json label mapping file. Only for Object Detection
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.PILToTensor``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            transforms (callable, optional): A function/transform that takes input sample and its target as entry
                and returns a transformed version.
        """
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.dataset_mode = "OD" if label_map_anno else "VG"
        self.max_labels = max_labels
        if self.dataset_mode == "OD":
            self.load_label_map(label_map_anno)
        self._load_metas(anno)
        self.get_dataset_info()

    def load_label_map(self, label_map_anno):
        """Load the label map json file for detection dataset."""
        with open(label_map_anno, 'r') as file:
            self.label_map = json.load(file)
        self.label_index = set(self.label_map.keys())

    def _load_metas(self, anno):
        """Load ODVG jsonl file"""
        with open(anno, 'r')as f:
            self.metas = [json.loads(line) for line in f]

    def get_dataset_info(self):
        """print dataset info."""
        logging.info(f"  == total images: {len(self)}")
        if self.dataset_mode == "OD":
            logging.info(f"  == total labels: {len(self.label_map)}")

    def __getitem__(self, index: int):
        """return image / target."""
        meta = self.metas[index]
        rel_path = meta["file_name"]
        abs_path = os.path.join(self.root, rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{self.root} {rel_path} {abs_path} not found.")
        image = Image.open(abs_path)
        image = ImageOps.exif_transpose(image).convert('RGB')

        w, h = image.size
        if self.dataset_mode == "OD":
            anno = meta["detection"]
            instances = [obj for obj in anno["instances"]]
            boxes = [obj["bbox"] for obj in instances]
            # generate vg_labels
            # pos bbox labels
            ori_classes = [str(obj["label"]) for obj in instances]
            pos_labels = set(ori_classes)
            # neg bbox labels
            neg_labels = self.label_index.difference(pos_labels)

            vg_labels = list(pos_labels)
            num_to_add = min(len(neg_labels), self.max_labels - len(pos_labels))
            if num_to_add > 0:
                vg_labels.extend(random.sample(neg_labels, num_to_add))

            # shuffle
            for i in range(len(vg_labels) - 1, 0, -1):
                j = random.randint(0, i)
                vg_labels[i], vg_labels[j] = vg_labels[j], vg_labels[i]

            caption_list = [self.label_map[lb] for lb in vg_labels]
            caption_dict = {item: index for index, item in enumerate(caption_list)}

            caption = ' . '.join(caption_list) + ' .'
            classes = [caption_dict[self.label_map[str(obj["label"])]] for obj in instances]

            boxes, classes = self.preprocess_boxes(boxes, classes, w, h)

        elif self.dataset_mode == "VG":
            anno = meta["grounding"]
            instances = [obj for obj in anno["regions"]]
            boxes = [obj["bbox"] for obj in instances]
            caption_list = [obj["phrase"] for obj in instances]
            c = list(zip(boxes, caption_list))
            random.shuffle(c)
            boxes[:], caption_list[:] = zip(*c)
            uni_caption_list = list(set(caption_list))
            label_map = {}
            for idx in range(len(uni_caption_list)):
                label_map[uni_caption_list[idx]] = idx
            classes = [label_map[cap] for cap in caption_list]
            caption = ' . '.join(uni_caption_list) + ' .'
            caption_list = uni_caption_list

            boxes, classes = self.preprocess_boxes(boxes, classes, w, h)

        target = {}
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["cap_list"] = caption_list
        target["caption"] = caption
        target["boxes"] = boxes
        target["labels"] = classes

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def preprocess_boxes(self, boxes, classes, w, h):
        """preprocess bounding boxes to given format."""
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # Clamp the coordinates to the image resolution
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Filter out invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        classes = torch.tensor(classes, dtype=torch.int64)
        classes = classes[keep]
        return boxes, classes

    def __len__(self) -> int:
        """return length of the dataset."""
        return len(self.metas)
