# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""OneFormer unified COCO dataset with advanced augmentations."""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from fvcore.transforms.transform import PadTransform
from panopticapi.utils import rgb2id
from PIL import Image, ImageOps
from PIL.Image import Resampling
from torch.utils.data import Dataset
import glob

from nvidia_tao_pytorch.core.utilities import PIL_SUPPORTED_FORMATS
from nvidia_tao_pytorch.cv.mask2former.utils.d2.structures import Instances  # pylint: disable=import-error
from nvidia_tao_pytorch.cv.mask2former.dataloader.augmentations import (
    RandomHorizontalFlip,
    RandomCrop,
    ResizeShortestEdge,
    ColorAugSSDTransform,
    apply_transform,
)
from nvidia_tao_pytorch.cv.oneformer.dataloader.augmentations import (
    RandomRotation,
    GaussianBlur,
    RandomErasing,
)
logger = logging.getLogger(__name__)


def masks_to_boxes(masks):
    """Convert masks to bounding boxes."""
    if not isinstance(masks, torch.Tensor) or masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    device = masks.device
    boxes = []
    for mask in masks:
        y, x = torch.where(mask)
        if len(y) == 0:
            boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0], device=device))
        else:
            boxes.append(
                torch.tensor(
                    [
                        x.min().float(),
                        y.min().float(),
                        x.max().float(),
                        y.max().float(),
                    ],
                    device=device,
                )
            )

    return torch.stack(boxes)


class COCOUnifiedDataset(Dataset):  # pylint: disable=too-many-instance-attributes
    """
    Integrated COCO dataset for unified segmentation tasks.
    This class combines the base dataset functionalities directly for a standalone implementation.
    """

    def __init__(self, ann_path, img_dir, panoptic_dir, cfg=None, is_training=False):  # pylint: disable=too-many-arguments
        # --- Start of Integrated BaseDataset Logic ---
        self.cfg = cfg
        self.segm_downsampling_rate = 4
        self.padding_constant = 2**5
        self.pixel_mean = np.array(cfg.dataset.pixel_mean)
        self.pixel_std = np.array(cfg.dataset.pixel_std)
        min_size_cfg = cfg.dataset.augmentation.train_min_size
        if len(min_size_cfg) <= 1:
            if len(min_size_cfg) == 0:
                min_size = min(cfg.dataset.augmentation.train_crop_size)
            else:
                min_size = int(min_size_cfg[0])
            cfg.dataset.augmentation.train_min_size = list(
                range(min_size // 2, min_size * 2 + 64, 64)
            )

        target_size = getattr(cfg.dataset, "image_size", 1024)
        if isinstance(target_size, int):
            self.target_size = [target_size, target_size]
        else:
            self.target_size = target_size
            assert (
                len(self.target_size) <= 2
            ), "The length of target_size must be less than 3."
            if len(self.target_size) == 1:
                self.target_size = self.target_size * 2
        # --- End of Integrated BaseDataset Logic ---

        self.ann_path = ann_path
        self.img_dir = img_dir
        self.panoptic_dir = panoptic_dir
        self.is_training = is_training

        self.contiguous_id = cfg.dataset.contiguous_id
        self.num_queries = (
            cfg.model.one_former.num_object_queries - cfg.model.text_encoder.n_ctx
        )
        self.max_seq_len = cfg.dataset.max_seq_len
        self.task_seq_len = cfg.dataset.task_seq_len
        if is_training:
            self.semantic_prob = cfg.dataset.task_prob_train.semantic
            self.instance_prob = cfg.dataset.task_prob_train.instance
            self.panoptic_prob = cfg.dataset.task_prob_train.panoptic
        else:
            self.semantic_prob = cfg.dataset.task_prob_val.semantic
            self.instance_prob = cfg.dataset.task_prob_val.instance
            self.panoptic_prob = cfg.dataset.task_prob_val.panoptic
        self.ignore_label = cfg.model.sem_seg_head.ignore_value

        if self.is_training:
            self.cutmix_prob = cfg.dataset.augmentation.get("cutmix_prob", 0.0)

        self.load_json()
        self.get_category_mapping()

    def get_image(self, file_name, root_dir=None, target_size=None):
        """Load and preprocess image from file."""
        root_dir = root_dir or ""
        image = Image.open(os.path.join(root_dir, file_name)).convert("RGB")
        image = ImageOps.exif_transpose(image)
        if target_size:
            image = image.resize(target_size)
        return np.array(image)

    def get_mask(self, file_name, root_dir=None, mode="L", target_size=None):
        """Load and preprocess mask from file."""
        root_dir = root_dir or ""
        mode = "RGB" if mode != "L" else "L"
        image = Image.open(os.path.join(root_dir, file_name)).convert(mode)
        if target_size:
            image = image.resize(target_size, resample=Resampling.NEAREST)
        return np.array(image)

    def normalize(self, img):
        """Normalize image using dataset statistics."""
        img = np.float32(img)
        img = (img - self.pixel_mean) / self.pixel_std
        return img.transpose((2, 0, 1))

    def round2nearest_multiple(self, x, p):
        """Round value to nearest multiple."""
        return ((x - 1) // p + 1) * p

    def get_padding_offset(self, orig_size, target_size=None):
        """Calculate padding offset for resizing."""
        h, w = target_size or orig_size
        new_h = int(self.round2nearest_multiple(h, self.padding_constant))
        new_w = int(self.round2nearest_multiple(w, self.padding_constant))
        return new_h - orig_size[0], new_w - orig_size[1]

    def load_json(self):
        """Load annotation data from JSON file."""
        with open(self.ann_path, "r", encoding="utf-8") as f:
            self.raw_annot = json.load(f)
        self.id2img = {img["id"]: img for img in self.raw_annot["images"]}

    def get_category_mapping(self):
        """Get category mapping for dataset."""
        self.thing_dataset_id_to_contiguous_id = {}
        self.stuff_dataset_id_to_contiguous_id = {}
        self.class_names = []
        self.things = []
        for i, cat in enumerate(self.raw_annot["categories"]):
            if cat["isthing"]:
                self.thing_dataset_id_to_contiguous_id[cat["id"]] = i
                self.things.append(i)
            self.stuff_dataset_id_to_contiguous_id[cat["id"]] = i
            self.class_names.append(cat["name"])

    def _get_train_transforms(self, orig_size):
        transforms = []
        if np.random.random() < 0.5:
            transforms.append(RandomHorizontalFlip(orig_size, prob=1.0))
        if np.random.random() < 0.5:
            transforms.append(RandomRotation(orig_size, angle_range=(-15, 15)))

        transforms.append(ColorAugSSDTransform("RGB"))

        if np.random.random() < 0.4:
            transforms.append(GaussianBlur())

        min_scale = self.cfg.dataset.min_scale
        max_scale = self.cfg.dataset.max_scale
        target_size = self.cfg.dataset.augmentation.train_crop_size[0]
        scale = np.random.uniform(min_scale, max_scale)
        new_size = int(target_size * scale)

        transforms.append(ResizeShortestEdge(orig_size, [new_size], new_size * 2))
        return transforms

    def _get_test_transforms(self, orig_size):
        return ResizeShortestEdge(
            orig_size,
            [self.cfg.dataset.augmentation.test_min_size],
            self.cfg.dataset.augmentation.test_max_size,
        )

    def _get_semantic_dict(self, pan_seg_gt, image_shape, segments_info, num_class_obj):
        instances = Instances(image_shape)
        classes, masks = [], []
        texts = ["a semantic photo"] * self.num_queries
        label = np.ones_like(pan_seg_gt) * self.ignore_label

        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if self.contiguous_id:
                class_id = self.stuff_dataset_id_to_contiguous_id[class_id]

            if not segment_info["iscrowd"]:
                mask = pan_seg_gt == segment_info["id"]
                if not np.all(mask is False):
                    if class_id not in classes:
                        cls_name = self.class_names[class_id]
                        classes.append(class_id)
                        masks.append(mask)
                        num_class_obj[cls_name] += 1
                    else:
                        masks[classes.index(class_id)] |= mask
                    label[mask] = class_id

        num = 0
        for cls_name in self.class_names:
            if num_class_obj.get(cls_name, 0) > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        instances.gt_classes = torch.tensor(np.array(classes), dtype=torch.int64)
        if len(masks) == 0:
            instances.gt_masks = torch.zeros(
                (0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])
            )
            instances.gt_bboxes = torch.zeros((0, 4))
        else:
            instances.gt_masks = torch.stack(
                [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]
            )
            instances.gt_bboxes = torch.stack(
                [torch.tensor([0.0, 0.0, 1.0, 1.0])] * instances.gt_masks.shape[0]
            )
        return instances, texts, label

    def _get_instance_dict(self, pan_seg_gt, image_shape, segments_info, num_class_obj):
        instances = Instances(image_shape)
        classes, masks = [], []
        texts = ["an instance photo"] * self.num_queries
        label = np.ones_like(pan_seg_gt) * self.ignore_label

        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if self.contiguous_id:
                class_id = self.stuff_dataset_id_to_contiguous_id[class_id]

            if class_id in self. things and not segment_info["iscrowd"]:
                mask = pan_seg_gt == segment_info["id"]
                if not np.all(mask is False):
                    cls_name = self.class_names[class_id]
                    classes.append(class_id)
                    masks.append(mask)
                    num_class_obj[cls_name] += 1
                    label[mask] = class_id

        num = 0
        for cls_name in self.class_names:
            if num_class_obj.get(cls_name, 0) > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        instances.gt_classes = torch.tensor(np.array(classes), dtype=torch.int64)
        if len(masks) == 0:
            instances.gt_masks = torch.zeros(
                (0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])
            )
            instances.gt_bboxes = torch.zeros((0, 4))
        else:
            instances.gt_masks = torch.stack(
                [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]
            )
            instances.gt_bboxes = masks_to_boxes(instances.gt_masks)
        return instances, texts, label

    def _get_panoptic_dict(self, pan_seg_gt, image_shape, segments_info, num_class_obj):  # pylint: disable=too-many-locals
        instances = Instances(image_shape)
        classes, masks = [], []
        texts = ["a panoptic photo"] * self.num_queries
        label = np.ones_like(pan_seg_gt) * self.ignore_label

        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if self.contiguous_id:
                class_id = self.stuff_dataset_id_to_contiguous_id[class_id]

            if not segment_info["iscrowd"]:
                mask = pan_seg_gt == segment_info["id"]
                if not np.all(mask is False):
                    cls_name = self.class_names[class_id]
                    classes.append(class_id)
                    masks.append(mask)
                    num_class_obj[cls_name] += 1
                    label[mask] = class_id

        num = 0
        for cls_name in self.class_names:
            if num_class_obj.get(cls_name, 0) > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        instances.gt_classes = torch.tensor(np.array(classes), dtype=torch.int64)
        if len(masks) == 0:
            instances.gt_masks = torch.zeros(
                (0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])
            )
            instances.gt_bboxes = torch.zeros((0, 4))
        else:
            instances.gt_masks = torch.stack(
                [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]
            )
            instances.gt_bboxes = masks_to_boxes(instances.gt_masks)
            for i in range(instances.gt_classes.shape[0]):
                if instances.gt_classes[i].item() not in self.things:
                    instances.gt_bboxes[i] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        return instances, texts, label

    def __len__(self):
        """Return length of dataset."""
        return len(self.raw_annot["annotations"])

    def __getitem__(self, idx):
        """Get item from dataset by index."""
        if not self.is_training or random.random() > self.cutmix_prob:
            return self._get_single_item(idx)

        item1 = self._get_single_item(idx)
        idx2 = random.randint(0, len(self) - 1)
        item2 = self._get_single_item(idx2)
        return self._apply_cutmix(item1, item2)

    def _apply_cutmix(self, item1, item2):  # pylint: disable=too-many-locals
        img_h, img_w = item1["image"].shape[1], item1["image"].shape[2]
        lam = np.random.beta(1.0, 1.0)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(img_w * cut_ratio), int(img_h * cut_ratio)
        cx, cy = np.random.randint(img_w), np.random.randint(img_h)
        bbx1, bby1 = np.clip(cx - cut_w // 2, 0, img_w), np.clip(
            cy - cut_h // 2, 0, img_h
        )
        bbx2, bby2 = np.clip(cx + cut_w // 2, 0, img_w), np.clip(
            cy + cut_h // 2, 0, img_h
        )

        item1["image"][:, bby1:bby2, bbx1:bbx2] = item2["image"][
            :, bby1:bby2, bbx1:bbx2
        ]
        item1["sem_seg"][bby1:bby2, bbx1:bbx2] = item2["sem_seg"][bby1:bby2, bbx1:bbx2]

        instances1, instances2 = item1["instances"], item2["instances"]
        cutout_mask = torch.zeros(
            (img_h, img_w), dtype=torch.bool, device=item1["image"].device
        )
        cutout_mask[bby1:bby2, bbx1:bbx2] = True

        new_masks, new_classes = [], []
        if instances1.has("gt_masks"):
            masks = instances1.gt_masks & ~cutout_mask
            areas = masks.sum(dim=(1, 2))
            keep = areas > 10
            if keep.any():
                new_masks.append(masks[keep])
                new_classes.append(instances1.gt_classes[keep])

        if instances2.has("gt_masks"):
            masks = instances2.gt_masks & cutout_mask
            areas = masks.sum(dim=(1, 2))
            keep = areas > 10
            if keep.any():
                new_masks.append(masks[keep])
                new_classes.append(instances2.gt_classes[keep])

        final_masks = (
            torch.cat(new_masks)
            if new_masks
            else torch.empty((0, img_h, img_w), dtype=torch.bool)
        )
        final_classes = (
            torch.cat(new_classes) if new_classes else torch.empty(0, dtype=torch.long)
        )

        item1["instances"].gt_masks = final_masks
        item1["instances"].gt_classes = final_classes
        item1["instances"].gt_bboxes = masks_to_boxes(final_masks)

        return item1

    def _get_single_item(self, idx):  # pylint: disable=too-many-locals
        ann = self.raw_annot["annotations"][idx]
        img_info = self.id2img[ann["image_id"]]

        img = self.get_image(
            img_info["file_name"],
            root_dir=self.img_dir,
            target_size=self.target_size if not self.is_training else None,
        )
        pan_seg_rgb = self.get_mask(
            ann["file_name"],
            self.panoptic_dir,
            mode="RGB",
            target_size=self.target_size if not self.is_training else None,
        )
        pan_segm = rgb2id(pan_seg_rgb)

        orig_size = img.shape[:2]
        if self.is_training:
            transforms = self._get_train_transforms(orig_size)
            for transform in transforms:
                img, pan_segm = apply_transform(img, pan_segm, transform)

            if np.random.random() < 0.25:
                erase_transform = RandomErasing(
                    img.shape[:2], seg_value=self.ignore_label
                )
                img, pan_segm = apply_transform(img, pan_segm, erase_transform)

            crop_size = self.cfg.dataset.augmentation.train_crop_size
            # Reverted to original RandomCrop call
            random_crop = RandomCrop(img.shape[:2], crop_size, pan_segm)
            img, pan_segm = apply_transform(img, pan_segm, random_crop)
            dh, dw = self.get_padding_offset(img.shape[:2], crop_size)
        else:
            transform = self._get_test_transforms(orig_size)
            img, pan_segm = apply_transform(img, pan_segm, transform)
            dh, dw = self.get_padding_offset(img.shape[:2])

        if dh > 0 or dw > 0:
            # Reverted to original seg_pad_value
            pad = PadTransform(0, 0, dw, dh, pad_value=0, seg_pad_value=0)
            img, pan_segm = apply_transform(img, pan_segm, pad)

        image_shape = img.shape[:2]
        segments_info = ann["segments_info"]
        prob_task = np.random.uniform(0, 1.0)
        num_class_obj = dict.fromkeys(self.class_names, 0)

        if prob_task < self.semantic_prob:
            task = "The task is semantic"
            instances, texts, sem_seg = self._get_semantic_dict(
                pan_segm, image_shape, segments_info, num_class_obj
            )
        elif prob_task < self.instance_prob + self.semantic_prob:
            task = "The task is instance"
            instances, texts, sem_seg = self._get_instance_dict(
                pan_segm, image_shape, segments_info, num_class_obj
            )
        else:
            task = "The task is panoptic"
            instances, texts, sem_seg = self._get_panoptic_dict(
                pan_segm, image_shape, segments_info, num_class_obj
            )

        data = {
            "image": torch.from_numpy(self.normalize(img)).float(),
            "sem_seg": torch.from_numpy(sem_seg).long(),
            "instances": instances,
            "orig_shape": image_shape,
            "task": task,
            "text": texts,
            "thing_ids": self.things,
            "file_name": img_info["file_name"],
            "image_id": ann["image_id"],
        }
        return data

    def collate_fn(self, batch):
        """Restored original collate function for consistent output."""
        out = {}
        images, all_instances, sem_segs, tasks, all_texts = [], [], [], [], []
        thing_ids, orig_shapes, file_names, image_ids = [], [], [], []

        for item in batch:
            images.append(item["image"])
            all_instances.append(item["instances"])
            sem_segs.append(item["sem_seg"])
            tasks.append(item["task"])
            all_texts.append(item["text"])
            thing_ids.append(item["thing_ids"])
            orig_shapes.append(item["orig_shape"])
            file_names.append(item["file_name"])
            image_ids.append(item["image_id"])

        out["images"] = torch.stack(images)
        out["instances"] = all_instances
        out["sem_segs"] = torch.stack(sem_segs)
        out["tasks"] = tasks
        out["texts"] = all_texts
        out["thing_ids"] = thing_ids
        out["orig_shapes"] = orig_shapes
        out["file_names"] = file_names
        out["image_ids"] = image_ids

        return out


class OneFormerPredictDataset(Dataset):
    """Dataset for prediction only."""

    def __init__(self, cfg=None):
        """Init dataset for prediction."""
        super().__init__()
        self.cfg = cfg
        self.img_list = sorted([
            file
            for ext in PIL_SUPPORTED_FORMATS
            for file in glob.glob(self.cfg.inference.images_dir + f"/*{ext}")
        ])
        self.mode = self.cfg.inference.mode.lower()
        self.padding_constant = 2**5
        self.pixel_mean = np.array(cfg.dataset.pixel_mean)
        self.pixel_std = np.array(cfg.dataset.pixel_std)

    def __len__(self):
        """Dataset length."""
        return len(self.img_list)

    def image_preprocess(self, img):
        """Preprocess image."""
        img_height, img_width = img.shape[0], img.shape[1]
        resize = ResizeShortestEdge(
            img.shape[:2],
            self.cfg.dataset.augmentation.test_min_size,
            self.cfg.dataset.augmentation.test_max_size)
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

    def round2nearest_multiple(self, x, p):
        """Round value to nearest multiple."""
        return ((x - 1) // p + 1) * p

    def get_padding_offset(self, orig_size, target_size=None):
        """Calculate padding offset for resizing."""
        h, w = target_size or orig_size
        new_h = int(self.round2nearest_multiple(h, self.padding_constant))
        new_w = int(self.round2nearest_multiple(w, self.padding_constant))
        return new_h - orig_size[0], new_w - orig_size[1]

    def normalize(self, img):
        """Normalize image using dataset statistics."""
        img = np.float32(img)
        img = (img - self.pixel_mean) / self.pixel_std
        return img.transpose((2, 0, 1))

    def get_image(self, file_name, root_dir=None, target_size=None):
        """Load image."""
        root_dir = root_dir or ""
        image = Image.open(os.path.join(root_dir, file_name)).convert('RGB')
        image = ImageOps.exif_transpose(image)
        if target_size:
            image = image.resize(target_size)
        image = np.array(image)
        return image

    def __getitem__(self, idx):
        """Per item."""
        filename = self.img_list[idx]
        img = self.get_image(filename, target_size=self.cfg.inference.image_size)
        # img_height, img_width, _ = img.shape
        img_tensor, info = self.image_preprocess(img)
        info['filename'] = Path(filename).stem
        data = {}
        data['raw_image'] = img
        data['image'] = img_tensor
        data['info'] = info
        if self.mode == "semantic":
            data['task'] = "The task is semantic"
        elif self.mode == "instance":
            data['task'] = "The task is instance"
        elif self.mode == "panoptic":
            data['task'] = "The task is panoptic"
        return data

    def collate_fn(self, batch):
        """Collate items in a batch."""
        out = {}
        images = []
        info = []
        raw_images = []
        tasks = []
        for item in batch:
            images.append(item['image'])
            info.append(item['info'])
            raw_images.append(item['raw_image'])
            tasks.append(item['task'])

        out['images'] = torch.stack(images)
        out['info'] = info
        out['raw_images'] = raw_images
        out['tasks'] = tasks
        return out
