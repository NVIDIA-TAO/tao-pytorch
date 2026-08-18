# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
import numpy as np
import tempfile
from PIL import Image
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.oneformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.oneformer.dataloader.pl_data_module import SemSegmDataModule


TEST_BATCH_SIZE = 4
TEST_WIDTH = 544
TEST_HEIGHT = 960
TEST_OBJ_WIDTH = 80
TEST_OBJ_HEIGHT = 80
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
tmp_panoptic_dir = os.path.join(tmp_top_dir, "panoptic")
json_file = os.path.join(tmp_top_dir, "sample_json.json")


def _generate_panoptic_mask(segments_info, sample_h, sample_w):
    """Generate a panoptic segmentation mask from segments info."""
    # Create RGB panoptic mask
    pan_mask = np.zeros((sample_h, sample_w, 3), dtype=np.uint8)

    for seg in segments_info:
        # Generate random mask for this segment
        mask = np.zeros((sample_h, sample_w), dtype=bool)
        if 'bbox' in seg:
            x1, y1, w, h = seg['bbox']
            x2, y2 = x1 + w, y1 + h
            # Ensure bounds are within image
            x1 = max(0, min(x1, sample_w - 1))
            x2 = max(0, min(x2, sample_w))
            y1 = max(0, min(y1, sample_h - 1))
            y2 = max(0, min(y2, sample_h))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True

        # Convert segment ID to RGB
        seg_id = seg['id']
        r = seg_id % 256
        g = (seg_id // 256) % 256
        b = (seg_id // (256 * 256)) % 256

        pan_mask[mask] = [r, g, b]

    return pan_mask


@pytest.fixture
def _test_sample_json():
    json_output = {
        "images": [],
        "annotations": [],
        "categories": [{"supercategory": "person", "id": 1, "name": "person", "isthing": 1},
                       {"supercategory": "face", "id": 2, "name": "face", "isthing": 1},
                       {"supercategory": "bag", "id": 3, "name": "bag", "isthing": 1}]
    }
    os.makedirs(tmp_top_dir, exist_ok=True)
    os.makedirs(tmp_panoptic_dir, exist_ok=True)

    for image_id in range(0, 10):
        sample_w = int(np.random.randint(low=TEST_WIDTH - 20, high=TEST_WIDTH + 20, size=1)[0])
        sample_h = int(np.random.randint(low=TEST_HEIGHT - 20, high=TEST_HEIGHT + 20, size=1)[0])
        img = Image.fromarray(np.random.randint(low=0, high=255, size=(sample_h, sample_w, 3), dtype=np.uint8))
        img_file = os.path.join(tmp_top_dir, f"test_{str(image_id)}.jpg")
        img.save(img_file)

        images_info = {
            "id": image_id,
            "file_name": f"test_{str(image_id)}.jpg",
            "height": sample_h,
            "width": sample_w
        }
        json_output["images"].append(images_info)

        segments_info = []
        for seg_idx in range(0, 5):
            segment_id = image_id * 100 + seg_idx  # Unique segment ID
            category_id = int(np.random.randint(low=1, high=4, size=1)[0])
            x1 = int(np.random.randint(low=10, high=max(11, sample_w - TEST_OBJ_WIDTH - 10), size=1)[0])
            y1 = int(np.random.randint(low=10, high=max(11, sample_h - TEST_OBJ_HEIGHT - 10), size=1)[0])
            w = int(np.random.randint(low=10, high=min(TEST_OBJ_WIDTH, sample_w - x1), size=1)[0])
            h = int(np.random.randint(low=10, high=min(TEST_OBJ_HEIGHT, sample_h - y1), size=1)[0])
            bbox = [x1, y1, w, h]
            area = bbox[2] * bbox[3]

            segment_info = {
                'id': segment_id,
                'category_id': category_id,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0
            }
            segments_info.append(segment_info)

        # Generate panoptic mask
        pan_mask = _generate_panoptic_mask(segments_info, sample_h, sample_w)
        pan_file = os.path.join(tmp_panoptic_dir, f"test_{str(image_id)}.png")
        Image.fromarray(pan_mask).save(pan_file)

        annotation_info = {
            'image_id': image_id,
            'file_name': f"test_{str(image_id)}.png",
            'segments_info': segments_info
        }
        json_output["annotations"].append(annotation_info)

    with open(json_file, 'w+') as outfile:
        json.dump(json_output, outfile)


@pytest.fixture
def _cfg():
    cfg = OmegaConf.structured(ExperimentConfig())

    # Dataset configuration
    cfg.dataset.contiguous_id = True
    cfg.dataset.train.annotations = json_file
    cfg.dataset.train.images = tmp_top_dir
    cfg.dataset.train.panoptic = tmp_panoptic_dir
    cfg.dataset.train.batch_size = 4
    cfg.dataset.train.num_workers = 0

    cfg.dataset.val.annotations = json_file
    cfg.dataset.val.images = tmp_top_dir
    cfg.dataset.val.panoptic = tmp_panoptic_dir
    cfg.dataset.val.batch_size = 2
    cfg.dataset.val.num_workers = 0

    cfg.dataset.test.annotations = json_file
    cfg.dataset.test.images = tmp_top_dir
    cfg.dataset.test.panoptic = tmp_panoptic_dir
    cfg.dataset.test.batch_size = 2
    cfg.dataset.test.num_workers = 0

    cfg.dataset.pin_memory = False
    cfg.dataset.image_size = 1024
    cfg.dataset.min_scale = 0.1
    cfg.dataset.max_scale = 2.0

    # Augmentation configuration
    cfg.dataset.augmentation.train_min_size = [320]
    cfg.dataset.augmentation.train_crop_size = [320, 320]
    cfg.dataset.augmentation.test_min_size = 640
    cfg.dataset.augmentation.test_max_size = 1333

    # Task probabilities
    cfg.dataset.task_prob_train = {"semantic": 0.33, "instance": 0.66, "panoptic": 0.01}
    cfg.dataset.task_prob_val = {"semantic": 0.33, "instance": 0.66, "panoptic": 0.01}

    # Model configuration (required by dataset)
    cfg.model.one_former.num_object_queries = 250
    cfg.model.text_encoder.n_ctx = 16
    cfg.model.sem_seg_head.num_classes = 3
    cfg.model.sem_seg_head.ignore_value = 255

    yield cfg


@pytest.mark.cv_unit
@pytest.mark.parametrize("train_min_size", [[320, 640], [200]])
@pytest.mark.parametrize("train_crop_size", [[320, 320], [128, 128]])
@pytest.mark.parametrize("val_batch_size", [2, 1])
def test_dataloader(_test_sample_json, _cfg, train_min_size, train_crop_size, val_batch_size):
    _cfg.dataset.augmentation.train_min_size = train_min_size
    _cfg.dataset.augmentation.train_crop_size = train_crop_size
    _cfg.dataset.augmentation.test_min_size = 640
    _cfg.dataset.val.batch_size = val_batch_size

    dm = SemSegmDataModule(_cfg)
    train_loader = dm.train_dataloader()
    for _, batch in enumerate(train_loader):
        b, c, h, w = batch['images'].shape
        assert b == 4
        assert c == 3
        assert [h, w] == train_crop_size
        break  # Only check first batch

    val_loader = dm.val_dataloader()
    for _, batch in enumerate(val_loader):
        b, c, h, w = batch['images'].shape
        assert b == val_batch_size
        assert c == 3
        break
