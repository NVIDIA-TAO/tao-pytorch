# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""COCO dataset."""

from typing import Any, List, Tuple
from PIL import Image
from pycocotools import mask as coco_mask

import torch
import torch.utils.data

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.od_dataset import ODDataset


# List of valid image extensions
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".JPEG", ".JPG", ".PNG")


def convert_coco_poly_to_mask(segmentations, height, width):
    """Convert coco poly annotation to mask tensor."""
    masks = []
    for polygons in segmentations:
        if isinstance(polygons, list):
            rles = coco_mask.frPyObjects(polygons, height, width)
            rles = coco_mask.merge(rles)
        elif 'counts' in polygons:
            # e.g. {'counts': [6, 1, 40, 4, 5, 4, 5, 4, 21], 'size': [9, 10]}
            if isinstance(polygons['counts'], list):
                rles = coco_mask.frPyObjects(polygons, height, width)
            else:
                rles = polygons
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class CocoDetection(ODDataset):
    """COCO Evaluation Dataset.

    Note that category id must be different from the default COCO annotation for COCO dataset.
    The category id must start from 0 and ids should be contiguous.
    """

    def __init__(self, json_file: str = None, dataset_dir: str = None, transforms=None, has_mask=True):
        """Initialize COCO ODVG """
        super().__init__(json_file, dataset_dir, transforms)
        category_dict = self.coco.loadCats(self.coco.getCatIds())
        cat_lists = [item['name'] for item in category_dict]
        self.cap_lists = cat_lists
        self.captions = " . ".join(cat_lists) + ' .'
        self.has_mask = has_mask

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

        if self.has_mask:
            segmentations = [obj["segmentation"] for obj in target]
            masks = convert_coco_poly_to_mask(segmentations, height, width)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        classes = classes[keep]
        if self.has_mask:
            masks = masks[keep]

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

        if self.has_mask:
            target["masks"] = masks

        # # For ODVG
        # target["cap_list"] = cap_list
        # target["caption"] = caption
        return image, target
