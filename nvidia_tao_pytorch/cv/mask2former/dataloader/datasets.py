# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Mask2former datasets."""
from pathlib import Path
import glob
import os
import logging
import json
import numpy as np
from PIL import Image, ImageOps
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from panopticapi.utils import rgb2id
from fvcore.transforms.transform import PadTransform

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from nvidia_tao_pytorch.core.utilities import PIL_SUPPORTED_FORMATS
from nvidia_tao_pytorch.cv.mask2former.dataloader.augmentations import (
    RandomHorizontalFlip, RandomCrop, ResizeShortestEdge,
    ColorAugSSDTransform,
    apply_transform
)
logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Base Dataset."""

    def __init__(self, cfg):
        """Init."""
        self.cfg = cfg
        self.segm_downsampling_rate = 4
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = 2**5
        self.pixel_mean = np.array(cfg.pixel_mean)
        self.pixel_std = np.array(cfg.pixel_std)

        min_size = self.cfg.augmentation.train_min_size
        if len(min_size) <= 1:
            if len(min_size) == 0:
                min_size = min(self.cfg.augmentation.train_crop_size)
            else:
                min_size = int(min_size[0])
            self.cfg.augmentation.train_min_size = list(range(min_size // 2, min_size * 2 + 64, 64))

        assert len(self.cfg.test.target_size) <= 2, "The length of target_size must be less than 3."
        if len(self.cfg.test.target_size) == 1:
            self.cfg.test.target_size = self.cfg.test.target_size * 2

    def get_image(self, file_name, root_dir=None, target_size=None):
        """Load image.

        Args:
            file_name (str): absolute or relative path to an image file.
            root_dir (str): root path of the file_name if any
            target_size (list): [width, height]
        Return:
            image (PIL image): loaded image
        """
        root_dir = root_dir or ""
        image = Image.open(os.path.join(root_dir, file_name)).convert('RGB')
        image = ImageOps.exif_transpose(image)  # image.copy overhead
        if target_size:
            image = image.resize(target_size)
        image = np.array(image)
        return image

    def get_mask(self, file_name, root_dir=None, mode="L", target_size=None):
        """Load image.

        Args:
            file_name (str): relative path to an image file (.png).
        Return:
            image (PIL image): loaded image
        """
        root_dir = root_dir or ""
        if mode != "L":
            mode = "RGB"
        image = Image.open(os.path.join(root_dir, file_name)).convert(mode)
        if target_size:
            image = image.resize(target_size, resample=Image.NEAREST)
        image = np.array(image)
        return image

    def normalize(self, img):
        """Normalize image with mean and std."""
        img = np.float32(img) / 255.
        img = (img - self.pixel_mean) / self.pixel_std
        img = img.transpose((2, 0, 1))  # [c, h, w]
        return img

    def round2nearest_multiple(self, x, p):
        """Round x to the nearest multiple of p and x' >= x."""
        return ((x - 1) // p + 1) * p

    def get_padding_offset(self, orig_size, target_size=None):
        """Get padding offset."""
        h, w = target_size or orig_size
        new_h = int(self.round2nearest_multiple(h, self.padding_constant))
        new_w = int(self.round2nearest_multiple(w, self.padding_constant))
        return new_h - orig_size[0], new_w - orig_size[1]


class COCOPanopticDataset(BaseDataset):
    """Base class for loading COCO Panoptic format labels."""

    def __init__(self, ann_path, img_dir,
                 panoptic_dir,
                 cfg=None,
                 is_training=False):
        """Initialize dataset.

        Args:
            ann_path (str): annotation file in COCO panoptic format
            img_dir (str): raw image directory
            panoptic_dir (str): directory of panoptic segmentation images
            cfg (Hydra config): Dataset configuration.
        """
        super().__init__(cfg)
        self.ann_path = ann_path
        self.img_dir = img_dir
        self.panoptic_dir = panoptic_dir
        self.is_training = is_training
        self.contiguous_id = cfg.contiguous_id  # TODO(@yuw)

        self.load_json()
        self.get_category_mapping()

    def load_json(self):
        """Load json file in COCO panoptic format."""
        with open(self.ann_path, 'r', encoding='utf-8') as f:
            self.raw_annot = json.load(f)  # 'images', 'annotations', 'categories'

        self.id2img = {}
        for img in self.raw_annot['images']:
            self.id2img[img["id"]] = img

    def get_category_mapping(self):
        """Map category index in json to 1 based index."""
        self.thing_dataset_id_to_contiguous_id = {}
        self.stuff_dataset_id_to_contiguous_id = {}
        for i, cat in enumerate(self.raw_annot['categories']):
            if cat["isthing"]:
                self.thing_dataset_id_to_contiguous_id[cat["id"]] = i + 1

            # in order to use sem_seg evaluator
            self.stuff_dataset_id_to_contiguous_id[cat["id"]] = i + 1

    def _get_train_transforms(self, orig_size):
        random_flip = RandomHorizontalFlip(orig_size, prob=0.5)
        resize = ResizeShortestEdge(
            orig_size,
            self.cfg.augmentation.train_min_size,
            self.cfg.augmentation.train_max_size)
        random_color = ColorAugSSDTransform("RGB")
        return [random_flip, random_color, resize]

    def _get_test_transforms(self, orig_size):
        resize = ResizeShortestEdge(
            orig_size,
            self.cfg.augmentation.test_min_size,
            self.cfg.augmentation.test_max_size)
        return resize

    def __len__(self):
        """Total number of images/annotations."""
        return len(self.raw_annot['annotations'])

    def __getitem__(self, idx):
        """Per item."""
        ann = self.raw_annot['annotations'][idx]
        filename = self.id2img[ann['image_id']]['file_name']
        target_size = self.cfg.train.target_size if self.is_training else self.cfg.val.target_size
        img = self.get_image(filename, root_dir=self.img_dir,
                             target_size=target_size)
        pan_segm = self.get_mask(ann['file_name'], self.panoptic_dir, mode="RGB",
                                 target_size=target_size)
        pan_segm = rgb2id(pan_segm)

        # data augmentation
        orig_size = img.shape[:2]
        if self.is_training:
            transforms = self._get_train_transforms(orig_size)
            for transform in transforms:
                img, pan_segm = apply_transform(img, pan_segm, transform)
            random_crop = RandomCrop(img.shape[:2], self.cfg.augmentation.train_crop_size, pan_segm)
            img, pan_segm = apply_transform(img, pan_segm, random_crop)
            dh, dw = self.get_padding_offset(img.shape[:2], self.cfg.augmentation.train_crop_size)
        else:
            transform = self._get_test_transforms(orig_size)
            img, pan_segm = apply_transform(img, pan_segm, transform)
            dh, dw = self.get_padding_offset(img.shape[:2])
        # fixed padding
        if dh > 0 or dw > 0:
            pad = PadTransform(0, 0, dw, dh, pad_value=0, seg_pad_value=0)
            img, pan_segm = apply_transform(img, pan_segm, pad)

        labels = []
        masks = []
        segm = np.zeros_like(pan_segm)
        for segment_info in ann["segments_info"]:
            cat_id = segment_info["category_id"]
            if self.contiguous_id:
                cat_id = self.stuff_dataset_id_to_contiguous_id[cat_id]
            if not segment_info["iscrowd"]:
                labels.append(cat_id)
                masks.append(pan_segm == segment_info["id"])
                segm[pan_segm == segment_info["id"]] = cat_id

        h, w = segm.shape
        new_shape = (h // self.segm_downsampling_rate, w // self.segm_downsampling_rate)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            masks = torch.zeros((0, ) + new_shape)
            labels = np.array([0])
            segm = torch.zeros(new_shape).long()
        else:
            labels = np.array(labels)
            masks = torch.from_numpy(np.array(masks))
            masks = F.resize(
                masks,
                new_shape,
                interpolation=F.InterpolationMode.NEAREST)
            masks = masks.long()
            segm = torch.from_numpy(segm)
            segm = F.resize(
                segm.unsqueeze(0),
                new_shape,
                interpolation=F.InterpolationMode.NEAREST)[0]
            segm = segm.long()

        data = {}
        data['image'] = torch.from_numpy(self.normalize(img)).float()
        data['target'] = {
            'masks': masks,
            'labels':  torch.from_numpy(labels)}
        data['segm'] = segm
        return data

    def collate_fn(self, batch):
        """Collate items in a batch."""
        out = {}
        images = []
        targets = []
        segms = []

        for item in batch:
            images.append(item['image'])
            targets.append(item['target'])
            segms.append(item['segm'])

        out['images'] = torch.stack(images)
        out['targets'] = targets
        out['segms'] = torch.stack(segms)
        return out


class COCODataset(BaseDataset):
    """Base class for loading COCO Panoptic format labels."""

    def __init__(self, ann_path, img_dir,
                 cfg=None,
                 is_training=False):
        """Initialize dataset.

        Args:
            ann_path (str): annotation file in COCO panoptic format
            img_dir (str): raw image directory
            cfg (Hydra config): Dataset configuration.
        """
        super().__init__(cfg)
        self.ann_path = ann_path
        self.img_dir = img_dir
        self.is_training = is_training
        self.contiguous_id = cfg.contiguous_id  # TODO(@yuw)
        self.load_coco()
        self.get_category_mapping()

    def load_coco(self):
        """Load COCO annotation."""
        self.coco = COCO(self.ann_path)
        self.image_ids = self.coco.getImgIds()

    def get_category_mapping(self):
        """Map category index in json to 1 based index."""
        self.stuff_dataset_id_to_contiguous_id = {}
        for i, cat in enumerate(self.coco.dataset['categories']):
            self.stuff_dataset_id_to_contiguous_id[cat["id"]] = i + 1

    def _get_train_transforms(self, orig_size):
        random_flip = RandomHorizontalFlip(orig_size, prob=0.5)
        resize = ResizeShortestEdge(
            orig_size,
            self.cfg.augmentation.train_min_size,
            self.cfg.augmentation.train_max_size)
        random_color = ColorAugSSDTransform("RGB")
        return [random_flip, random_color, resize]

    def _get_test_transforms(self, orig_size):
        resize = ResizeShortestEdge(
            orig_size,
            self.cfg.augmentation.test_min_size,
            self.cfg.augmentation.test_max_size)
        return resize

    def __len__(self):
        """Total number of images/annotations."""
        return len(self.image_ids)

    def get_gt(self, image_info, annot_ids, target_size=None):
        """Get GT annotations."""
        labels = []
        ids = []

        h = image_info['height']
        w = image_info['width']
        segm = np.zeros((h, w))
        for i in annot_ids:
            ann = self.coco.loadAnns(ids=i)[0]
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
            assert len(mask.shape) == 2
            mask = mask.astype(np.uint8)

            cat_id = ann["category_id"]
            if self.contiguous_id:
                cat_id = self.stuff_dataset_id_to_contiguous_id[cat_id]
            segm[mask == 1] = ann['id']
            labels.append(cat_id)
            ids.append(ann['id'])

        if target_size:
            image = Image.fromarray(segm)
            image = image.resize(target_size, resample=Image.NEAREST)
            segm = np.array(image)
        return segm, labels, ids

    def __getitem__(self, idx):
        """Per item."""
        # get ground truth annotations
        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        filename = image_info['file_name']
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx], iscrowd=False)

        target_size = self.cfg.train.target_size if self.is_training else self.cfg.val.target_size
        img = self.get_image(filename, root_dir=self.img_dir, target_size=target_size)
        pan_segm, labels, ids = self.get_gt(image_info, annotations_ids, target_size=target_size)

        # data augmentation
        orig_size = img.shape[:2]
        if self.is_training:
            transforms = self._get_train_transforms(orig_size)
            for transform in transforms:
                img, pan_segm = apply_transform(img, pan_segm, transform)
            random_crop = RandomCrop(img.shape[:2], self.cfg.augmentation.train_crop_size, pan_segm)
            img, pan_segm = apply_transform(img, pan_segm, random_crop)
            dh, dw = self.get_padding_offset(img.shape[:2], self.cfg.augmentation.train_crop_size)
        else:
            transform = self._get_test_transforms(orig_size)
            img, pan_segm = apply_transform(img, pan_segm, transform)
            dh, dw = self.get_padding_offset(img.shape[:2])
        # fixed padding
        if dh > 0 or dw > 0:
            pad = PadTransform(0, 0, dw, dh, pad_value=0, seg_pad_value=0)
            img, pan_segm = apply_transform(img, pan_segm, pad)

        masks = []
        segm = np.zeros_like(pan_segm)
        for (l, i) in zip(labels, ids):
            masks.append(pan_segm == i)
            segm[pan_segm == i] = l

        h, w = segm.shape
        new_shape = (h // self.segm_downsampling_rate, w // self.segm_downsampling_rate)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            masks = torch.zeros((0, ) + new_shape)
            labels = np.array([0])
            segm = torch.zeros(new_shape).long()
        else:
            labels = np.array(labels)
            masks = torch.from_numpy(np.array(masks))
            masks = F.resize(
                masks,
                new_shape,
                interpolation=F.InterpolationMode.NEAREST)
            masks = masks.long()
            segm = torch.from_numpy(segm)
            segm = F.resize(
                segm.unsqueeze(0),
                new_shape,
                interpolation=F.InterpolationMode.NEAREST)[0]
            segm = segm.long()

        data = {}
        data['image'] = torch.from_numpy(self.normalize(img)).float()
        data['target'] = {
            'masks': masks,
            'labels':  torch.from_numpy(labels)}
        data['segm'] = segm
        return data

    def collate_fn(self, batch):
        """Collate function."""
        out = {}
        images = []
        targets = []
        segms = []

        for item in batch:
            images.append(item['image'])
            targets.append(item['target'])
            segms.append(item['segm'])

        out['images'] = torch.stack(images)
        out['targets'] = targets
        out['segms'] = torch.stack(segms)
        return out


class ADEDataset(BaseDataset):
    """ADE format dataset."""

    def __init__(self,
                 gt_list_or_file,
                 root_dir=None,
                 cfg=None,
                 is_training=False):
        """Init."""
        super().__init__(cfg)
        self.root_dir = root_dir or ''
        self.is_training = is_training
        # parse the input list
        self.parse_input_list(gt_list_or_file)

    def parse_input_list(self, gt_list_or_file, max_sample=-1, start_idx=-1, end_idx=-1):
        """Parse input list."""
        if isinstance(gt_list_or_file, list):
            self.sample_list = gt_list_or_file
        elif isinstance(gt_list_or_file, str):
            with open(gt_list_or_file, 'r', encoding='utf-8') as f:
                self.sample_list = [json.loads(x.rstrip()) for x in f]
            # self.sample_list = [json.loads(x.rstrip()) for x in open(gt_list_or_file, 'r')]

        if max_sample > 0:
            self.sample_list = self.sample_list[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.sample_list = self.sample_list[start_idx:end_idx]

        self.num_samples = len(self.sample_list)
        assert self.num_samples > 0
        logger.info(f'Number of samples: {self.num_samples}')

    def __len__(self):
        """Length of the dataset."""
        return self.num_samples

    def _get_train_transforms(self, orig_size):
        random_flip = RandomHorizontalFlip(orig_size, prob=0.5)
        resize = ResizeShortestEdge(
            orig_size,
            self.cfg.augmentation.train_min_size,
            self.cfg.augmentation.train_max_size)
        random_color = ColorAugSSDTransform("RGB")
        return [random_flip, random_color, resize]

    def _get_test_transforms(self, orig_size):
        resize = ResizeShortestEdge(
            orig_size,
            self.cfg.augmentation.test_min_size,
            self.cfg.augmentation.test_max_size)
        return resize

    def __getitem__(self, index):
        """Per item."""
        this_record = self.sample_list[index]
        # load image and label
        image_path = os.path.join(self.root_dir, this_record['img'])
        segm_path = os.path.join(self.root_dir, this_record['segm'])

        target_size = self.cfg.train.target_size if self.is_training else self.cfg.val.target_size
        img = self.get_image(image_path, target_size=target_size)
        segm = self.get_mask(segm_path, target_size=target_size)
        orig_size = img.shape[:2]
        assert segm.shape == orig_size, "Image and mask shape must match before augmentation!"
        # data augmentation
        if self.is_training:
            transforms = self._get_train_transforms(orig_size)
            for transform in transforms:
                img, segm = apply_transform(img, segm, transform)
            random_crop = RandomCrop(img.shape[:2], self.cfg.augmentation.train_crop_size, segm)
            img, segm = apply_transform(img, segm, random_crop)
            dh, dw = self.get_padding_offset(img.shape[:2], self.cfg.augmentation.train_crop_size)
        else:
            transform = self._get_test_transforms(orig_size)
            img, segm = apply_transform(img, segm, transform)
            dh, dw = self.get_padding_offset(img.shape[:2])
        # fixed padding
        if dh > 0 or dw > 0:
            pad = PadTransform(0, 0, dw, dh, pad_value=0, seg_pad_value=0)
            img, segm = apply_transform(img, segm, pad)

        data = {}
        data['image'] = torch.from_numpy(self.normalize(img)).float()

        cat_ids = np.unique(segm)[1:]  # skip 0
        masks = []
        for i in cat_ids:
            masks.append(segm == i)
        h, w = segm.shape
        new_shape = (h // self.segm_downsampling_rate, w // self.segm_downsampling_rate)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            h, w = segm.shape
            masks = torch.zeros((0,) + new_shape)
            labels = np.array([0])
            segm = torch.zeros(new_shape).long()
        else:
            labels = cat_ids
            h, w = segm.shape
            masks = torch.from_numpy(np.array(masks))
            masks = F.resize(
                masks,
                new_shape,
                interpolation=F.InterpolationMode.NEAREST)
            masks = masks.long()
            segm = torch.from_numpy(segm)
            segm = F.resize(
                segm.unsqueeze(0),
                new_shape,
                interpolation=F.InterpolationMode.NEAREST)[0]
            segm = segm.long()

        data['target'] = {
            'masks': masks,
            'labels':  torch.from_numpy(labels).long()}
        data['segm'] = segm
        return data

    def collate_fn(self, batch):
        """Collate function."""
        out = {}
        images = []
        targets = []
        segms = []

        for item in batch:
            images.append(item['image'])
            targets.append(item['target'])
            segms.append(item['segm'])

        out['images'] = torch.stack(images)
        out['targets'] = targets
        out['segms'] = torch.stack(segms)
        return out


class PredictDataset(BaseDataset):
    """Dataset for prediction only."""

    def __init__(self,
                 img_dir,
                 cfg=None):
        """Init dataset for prediction."""
        super().__init__(cfg)
        self.img_list = sorted([
            file
            for ext in PIL_SUPPORTED_FORMATS
            for file in glob.glob(self.cfg.test.img_dir + f"/*{ext}")
        ])

    def image_preprocess(self, img):
        """Preprocess image."""
        img_height, img_width = img.shape[0], img.shape[1]
        resize = ResizeShortestEdge(
            img.shape[:2],
            self.cfg.augmentation.test_min_size,
            self.cfg.augmentation.test_max_size)
        img = resize.apply_image(img)
        # fixed padding
        dh, dw = self.get_padding_offset(img.shape[:2])
        if dh > 0 or dw > 0:
            pad = PadTransform(0, 0, dw, dh, pad_value=0, seg_pad_value=0)
            img = pad.apply_image(img)
        img = self.normalize(img)

        info = {'padding': (dh, dw),
                'image_size': (img_height, img_width)}
        input_tensor = torch.from_numpy(img).float()
        return input_tensor, info

    def __len__(self):
        """Dataset length."""
        return len(self.img_list)

    def __getitem__(self, idx):
        """Per item."""
        filename = self.img_list[idx]
        img = self.get_image(filename, target_size=self.cfg.test.target_size)
        # img_height, img_width, _ = img.shape
        img_tensor, info = self.image_preprocess(img)
        info['filename'] = Path(filename).stem
        data = {}
        data['raw_image'] = img
        data['image'] = img_tensor
        data['info'] = info
        return data

    def collate_fn(self, batch):
        """Collate items in a batch."""
        out = {}
        images = []
        info = []
        raw_images = []
        for item in batch:
            images.append(item['image'])
            info.append(item['info'])
            raw_images.append(item['raw_image'])

        out['images'] = torch.stack(images)
        out['info'] = info
        out['raw_images'] = raw_images
        return out
