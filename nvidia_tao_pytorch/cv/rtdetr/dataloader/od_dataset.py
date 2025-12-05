# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original implementation: https://github.com/lyuwenyu/RT-DETR
# Licensed under the Apache License 2.0
# Copyright (c) 2023 lyuwenyu. All Rights Reserved.

""" Object Detection Dataset Class and Related Functions """

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import ConcatDataset
from torchvision import tv_tensors

import os
import glob
import h5py
from PIL import Image, ImageOps
from typing import Any, Tuple, List

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.od_dataset import ODDataset, VALID_IMAGE_EXTENSIONS
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import read_h5_image_from_path


def build_coco(data_sources, transforms, remap_mscoco_category):
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
        dataset_list.append(RTDataset(json_file, image_dir,
                                      transforms=transforms,
                                      remap_mscoco_category=remap_mscoco_category))  # removed hard-coded True

        if len(dataset_list) > 1:
            train_dataset = ConcatDataset(dataset_list)
        else:
            train_dataset = dataset_list[0]
    return train_dataset


class RTDataset(ODDataset):
    """RT-DETR Object Detection Dataset Class."""

    def __init__(self, json_file: str = None, dataset_dir: str = None, transforms=None, remap_mscoco_category=False):
        """Initialize the Object Detetion Dataset Class.

        Note that multiple loading of COCO type JSON files can lead to system memory OOM.
        In such case, use SerializedDatasetFromList.

        Args:
            json_file (str): json_file name to load the data.
            dataset_dir (str): dataset directory.
            transforms: augmentations to apply.
        """
        super(RTDataset, self).__init__(json_file, dataset_dir, transforms)
        self.remap_mscoco_category = remap_mscoco_category

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

        # RT-DETR original src code remap coco category from 1 to 81
        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in target]
        else:
            classes = [obj["category_id"] for obj in target]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format='XYXY',
            canvas_size=image.size[::-1]  # h w
        )

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


class ODPredictDataset(Dataset):
    """Base Object Detection Predict Dataset Class."""

    def __init__(self, dataset_list: List[Any], label_map_path: str,
                 transforms=None, start_from_one=False, fixed_resolution=None):
        """Initialize the Object Detetion Dataset Class for inference.

        Unlike ODDataset, this class does not require COCO JSON file.

        Args:
            dataset_list (list): list of dataset directory.
            label_map_path (str): label mapping path.
            transforms: augmentations to apply.
            start_from_one (bool): Whether to start the class_mapping index from 1 or not.
            fixed_resolution (tuple): Fixed resolution (h, w) for evaluation.
                Only needed when we resize with aspect ratio preserved.
            verbose (bool): Whether to print verbose messages during processing.

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

        self.fixed_resolution = fixed_resolution

        self.ids = []
        self.base_dirs = {}
        for seq in dataset_list:
            if not os.path.exists(seq):
                raise FileNotFoundError(f"Provided inference directory {seq} does not exist!")

            for ext in VALID_IMAGE_EXTENSIONS:
                image_paths = glob.glob(seq + f"/*{ext}")
                self.ids.extend(image_paths)
                for img_path in image_paths:
                    self.base_dirs[img_path] = seq

            h5_files = glob.glob(seq + "/*.h5")
            for h5_file in h5_files:
                try:
                    with h5py.File(h5_file, 'r') as f:
                        if 'rgb' in f:
                            rgb_group = f['rgb']
                            h5_base_name = os.path.splitext(os.path.basename(h5_file))[0]
                            for image_name in rgb_group.keys():
                                # Format: h5://<h5 file name without .h5>:<image name>
                                h5_path = f"h5://{h5_base_name}:{image_name}"
                                self.ids.append(h5_path)
                                self.base_dirs[h5_path] = seq
                        else:
                            print(f"Warning: No 'rgb' group found in {h5_file}")
                except Exception as e:
                    print(f"Warning: Could not read h5 file {h5_file}: {e}")

        if len(self.ids) == 0:
            raise FileNotFoundError(f"No valid image with extensions {VALID_IMAGE_EXTENSIONS} or h5 files found in the provided directories")

    def _load_image(self, img_path: int) -> Image.Image:
        """Load image given image path.

        Args:
            img_path (str): image path to load. Can be regular file path or
                          h5 format: h5://[h5_file_path]:image_file_name

        Returns:
            Loaded PIL.Image.
        """
        if img_path.startswith("h5://"):
            img, img_path = read_h5_image_from_path(img_path, self.base_dirs[img_path])
        else:
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
        if self.fixed_resolution:
            target["orig_size"] = torch.as_tensor(list(self.fixed_resolution))
        else:
            target["orig_size"] = torch.as_tensor([int(height), int(width)])
        target["size"] = torch.as_tensor([int(height), int(width)])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, image_path

    def __len__(self) -> int:
        """__len__"""
        return len(self.ids)


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
