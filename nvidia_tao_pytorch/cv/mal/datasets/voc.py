# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VOC dataset."""

import os
import logging
import json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

from torch.utils.data import Dataset
logger = logging.getLogger(__name__)


class DataWrapper:
    """Simple data wrapper."""

    def __init__(self, data):
        """Initialize DataWrapper.

        Args:
            data (np.array): numpy array
        """
        self.data = data


class BoxLabelVOC(Dataset):
    """Base class for loading COCO format labels."""

    def __init__(self, ann_path, img_data_dir,
                 min_obj_size=0, max_obj_size=1e10,
                 transform=None, cfg=None,
                 **kwargs):
        """Initialize dataset.

        Args:
            ann_path (str): annotation file in json format
            img_data_dir (str): raw image directory
            min_obj_size (float): min object size
            max_obj_size (float): max object size
            transform (transform.Compose): data augmentation methods
            cfg (Hydra config): Hydra configurations
        """
        self.cfg = cfg
        self.ann_path = ann_path
        self.img_data_dir = img_data_dir
        self.min_obj_size = min_obj_size
        self.max_obj_size = max_obj_size
        self.transform = transform
        self.coco = COCO(ann_path)
        self._filter_imgs()
        self.get_category_mapping()

    def get_category_mapping(self):
        """Map category index in json to 1 based index."""
        self.cat_mapping = dict([i, i] for i in range(1, 21))

    def _filter_imgs(self):
        """Filter out bboxes based on area and H/W range."""
        anns = self.coco.dataset['annotations']
        filtered_anns = []
        for ann in anns:
            # query image info
            image_info = self.coco.loadImgs(ann['image_id'])[0]
            # check if bbox is out of bound
            is_correct_bbox = ann['bbox'][0] >= 0 and ann['bbox'][1] >= 0 and \
                (ann['bbox'][0] + ann['bbox'][2]) <= image_info['width'] and \
                (ann['bbox'][1] + ann['bbox'][3]) <= image_info['height']
            area = ann['bbox'][2] * ann['bbox'][3]
            # check if bbox area is within range
            is_correct_area = self.max_obj_size > area > self.min_obj_size
            # additionally, check bbox w/h > 2
            if is_correct_bbox and is_correct_area and ann['bbox'][2] > 2 and ann['bbox'][3] > 2:
                filtered_anns.append(ann)
        self.coco.dataset['annotations'] = filtered_anns
        num_filtered = len(self.coco.dataset['annotations']) - len(filtered_anns)
        if num_filtered > 0:
            print("***********************************")
            print(f"WARNING: {num_filtered} bboxes were filtered out.")
            print("***********************************")

    def __len__(self):
        """Total number of bboxes."""
        return len(self.coco.getAnnIds())

    def __getitem__(self, idx):
        """Per item."""
        ann = self.coco.dataset['annotations'][idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        h, w, file_name = img_info['height'], img_info['width'], img_info['file_name']
        img = self.get_image(file_name)

        # box mask
        mask = np.zeros((h, w))
        bbox = ann['bbox']
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        mask[y0:y1 + 1, x0:x1 + 1] = 1

        data = {
            'image': img, 'mask': mask,
            'height': h, 'width': w,
            'category_id': ann['category_id'],
            'bbox': np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32),
            'compact_category_id': self.cat_mapping[int(ann['category_id'])],
            'id': ann['id']
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_image(self, file_name):
        """Load image.

        Args:
            file_name (str): relative path to an image file.
        Return:
            image (PIL image): loaded image
        """
        image = Image.open(os.path.join(self.img_data_dir, file_name)).convert('RGB')
        return image


class InstSegVOC(BoxLabelVOC):
    """Class for loading COCO format labels with instance segmentation masks."""

    def __init__(self, *args, load_mask=True, **kwargs):
        """Initialize dataset with segmentation groundtruth.

        Args:
            load_mask (bool): whether to load instance segmentation annotations
        """
        super().__init__(*args, **kwargs)
        self.load_mask = load_mask
        if load_mask:
            for ann in self.coco.dataset['annotations']:
                if not ann.get('segmentation', None):
                    raise ValueError(
                        "Please check your annotation file, "
                        "as not all annotations contain segmentation info. "
                        "Or set load_mask to False.")
        self.get_category_mapping()

    def __getitem__(self, idx):
        """Per item."""
        ann = self.coco.dataset['annotations'][idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        h, w, file_name = img_info['height'], img_info['width'], img_info['file_name']
        img = self.get_image(file_name)

        # box mask
        boxmask = np.zeros((h, w))
        bbox = ann['bbox']
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        boxmask[y0:y1 + 1, x0:x1 + 1] = 1

        data = {'image': img, 'boxmask': boxmask,
                'height': h, 'width': w,
                'category_id': ann['category_id'],
                'bbox': np.array(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32),
                'compact_category_id': self.cat_mapping[int(ann['category_id'])],
                'id': ann['id'],
                'image_id': ann['image_id']}

        if self.load_mask:
            # mask = np.ascontiguousarray(
            #     maskUtils.decode(maskUtils.frPyObjects(ann['segmentation'], h, w)))
            # polygons
            if isinstance(ann['segmentation'], list):
                rles = maskUtils.frPyObjects(ann['segmentation'], h, w)
                rle = maskUtils.merge(rles)
            elif 'counts' in ann['segmentation']:
                # e.g. {'counts': [6, 1, 40, 4, 5, 4, 5, 4, 21], 'size': [9, 10]}
                if isinstance(ann['segmentation']['counts'], list):
                    rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
                else:
                    rle = ann['segmentation']
            else:
                raise ValueError('Please check the segmentation format.')
            mask = np.ascontiguousarray(maskUtils.decode(rle))
            if len(mask.shape) > 2:
                mask = mask.transpose((2, 0, 1)).sum(0) > 0
            mask = mask.astype(np.uint8)

            data['gtmask'] = DataWrapper(mask)
            data['mask'] = mask

        if self.transform is not None:
            data = self.transform(data)

        return data


class InstSegVOCwithBoxInput(InstSegVOC):
    """Class for loading bbox inputs with instance segmentation masks."""

    def __init__(self,
                 ann_path,
                 img_data_dir,
                 min_obj_size=0,
                 max_obj_size=1e10,
                 transform=None,
                 load_mask=True,
                 box_inputs=None):
        """Init."""
        self.load_mask = load_mask
        self.ann_path = ann_path
        self.img_data_dir = img_data_dir
        self.min_obj_size = min_obj_size
        self.max_obj_size = max_obj_size
        self.transform = transform
        self.coco = COCO(ann_path)
        self._filter_imgs()
        self.get_category_mapping()
        with open(box_inputs, "r") as f:
            self.val_coco = json.load(f)

    def __len__(self):
        """Number of samples."""
        return len(self.val_coco)

    def __getitem__(self, idx):
        """Per item."""
        ann = self.val_coco[idx]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        h, w, file_name = img_info['height'], img_info['width'], img_info['file_name']
        img = self.get_image(file_name)

        # box mask
        boxmask = np.zeros((h, w))
        bbox = np.array(ann['bbox'])
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        boxmask[y0:y1 + 1, x0:x1 + 1] = 1

        if 'id' not in ann.keys():
            _id = hash(str(ann['image_id']) + ' ' + str(x0) + ' ' + str(x1) + ' ' + str(y0) + ' ' + str(y1))
        else:
            _id = ann['id']

        data = {'image': img, 'boxmask': boxmask,
                'height': h, 'width': w,
                'category_id': ann['category_id'],
                'bbox': np.array(
                    [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32),
                'compact_category_id': self.cat_mapping[int(ann['category_id'])],
                'id': _id,
                'image_id': ann['image_id'],
                'score': ann['score']}

        if self.load_mask:
            mask = np.ascontiguousarray(maskUtils.decode(ann['segmentation']))
            if len(mask.shape) > 2:
                mask = mask.transpose((2, 0, 1)).sum(0) > 0
            mask = mask.astype(np.uint8)

            data['gtmask'] = DataWrapper(mask)
            data['mask'] = mask

        if self.transform is not None:
            data = self.transform(data)

        return data
