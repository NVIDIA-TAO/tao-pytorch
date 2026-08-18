# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import numpy as np
import pytest
from PIL import Image
from pytorch_lightning import Trainer

from omegaconf import OmegaConf
from nvidia_tao_pytorch.config.oneformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.oneformer.dataloader.pl_data_module import SemSegmDataModule
from nvidia_tao_pytorch.cv.oneformer.model.pl_oneformer import OneformerPlModule


FAST_DEV_RUN = 2
TEST_BATCH_SIZE = 4
TEST_WIDTH = 544
TEST_HEIGHT = 960
TEST_OBJ_WIDTH = 80
TEST_OBJ_HEIGHT = 80
tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
tmp_panoptic_dir = os.path.join(tmp_top_dir, "panoptic")
json_file = os.path.join(tmp_top_dir, "sample_json.json")
labelmap_file = os.path.join(tmp_top_dir, "labelmap.json")


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
    """Generate sample dataset including images, panoptic masks, and COCO json."""
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
    # Create label map file
    labelmap = [
        {"color": [255, 0, 0], "isthing": 1, "id": 1, "name": "person"},
        {"color": [0, 255, 0], "isthing": 1, "id": 2, "name": "face"},
        {"color": [0, 0, 255], "isthing": 1, "id": 3, "name": "bag"}
    ]
    with open(labelmap_file, "w") as f:
        json.dump(labelmap, f)


@pytest.fixture
def _train_spec():
    """Generate a structured OmegaConf object for training."""
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.results_dir = results_dir
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    experiment_config.train.num_epochs = 1

    # Dataset configuration
    experiment_config.dataset.label_map = labelmap_file
    experiment_config.dataset.contiguous_id = True
    experiment_config.dataset.train.annotations = json_file
    experiment_config.dataset.train.images = tmp_top_dir
    experiment_config.dataset.train.panoptic = tmp_panoptic_dir
    experiment_config.dataset.train.batch_size = TEST_BATCH_SIZE
    experiment_config.dataset.train.num_workers = 0

    experiment_config.dataset.val.annotations = json_file
    experiment_config.dataset.val.images = tmp_top_dir
    experiment_config.dataset.val.panoptic = tmp_panoptic_dir
    experiment_config.dataset.val.batch_size = 2
    experiment_config.dataset.val.num_workers = 0

    experiment_config.dataset.pin_memory = False
    experiment_config.dataset.image_size = 1024
    experiment_config.dataset.min_scale = 0.1
    experiment_config.dataset.max_scale = 2.0

    # Augmentation configuration
    experiment_config.dataset.augmentation.train_min_size = [320]
    experiment_config.dataset.augmentation.train_crop_size = [320, 320]
    experiment_config.dataset.augmentation.test_min_size = 640
    experiment_config.dataset.augmentation.test_max_size = 1333

    # Task probabilities
    experiment_config.dataset.task_prob_train = {"semantic": 0.33, "instance": 0.66, "panoptic": 0.01}
    experiment_config.dataset.task_prob_val = {"semantic": 0.33, "instance": 0.66, "panoptic": 0.01}

    # Model configuration (required by dataset)
    experiment_config.model.one_former.num_object_queries = 250
    experiment_config.model.text_encoder.n_ctx = 16
    experiment_config.model.sem_seg_head.num_classes = 3
    experiment_config.model.sem_seg_head.ignore_value = 255

    yield experiment_config


@pytest.fixture
def _eval_spec():
    """Generate a structured OmegaConf object for evaluation."""
    experiment_config = OmegaConf.structured(ExperimentConfig())
    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir
    experiment_config.evaluate.results_dir = results_dir
    experiment_config.evaluate.num_gpus = 1
    # Dataset configuration
    experiment_config.dataset.label_map = labelmap_file
    experiment_config.dataset.contiguous_id = True
    experiment_config.dataset.test.annotations = json_file
    experiment_config.dataset.test.images = tmp_top_dir
    experiment_config.dataset.test.panoptic = tmp_panoptic_dir
    experiment_config.dataset.test.batch_size = 2
    experiment_config.dataset.test.num_workers = 0
    experiment_config.dataset.pin_memory = False
    experiment_config.dataset.image_size = 1024
    experiment_config.dataset.min_scale = 0.1
    experiment_config.dataset.max_scale = 2.0
    # Augmentation configuration
    experiment_config.dataset.augmentation.test_min_size = 640
    experiment_config.dataset.augmentation.test_max_size = 1333
    # Task probabilities
    experiment_config.dataset.task_prob_val = {"semantic": 0.33, "instance": 0.66, "panoptic": 0.01}
    # Model configuration (required by dataset)
    experiment_config.model.one_former.num_object_queries = 250
    experiment_config.model.text_encoder.n_ctx = 16
    experiment_config.model.sem_seg_head.num_classes = 3
    experiment_config.model.sem_seg_head.ignore_value = 255

    yield experiment_config


@pytest.fixture
def _infer_spec():
    """Generate a structured OmegaConf object for inference."""
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.inference.results_dir = results_dir
    experiment_config.inference.images_dir = tmp_top_dir
    experiment_config.inference.image_size = [640, 640]
    # Dataset configuration
    experiment_config.dataset.label_map = labelmap_file
    experiment_config.dataset.contiguous_id = True
    experiment_config.dataset.test.batch_size = 2
    experiment_config.dataset.test.num_workers = 0
    experiment_config.dataset.pin_memory = False
    experiment_config.dataset.image_size = 1024
    # Model configuration (required by dataset)
    experiment_config.model.one_former.num_object_queries = 250
    experiment_config.model.text_encoder.n_ctx = 16
    experiment_config.model.sem_seg_head.num_classes = 3
    experiment_config.model.sem_seg_head.ignore_value = 255

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.oneformer
@pytest.mark.train
@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
@pytest.mark.skip(reason="flaky test pending fix")
def test_trainer_fit(_test_sample_json, _train_spec, precision):
    """Test the OneFormer training pipeline."""
    _train_spec.model.backbone.name = "D2SwinTransformer"

    dm = SemSegmDataModule(_train_spec)
    dm.setup(stage="fit")
    pt_model = OneformerPlModule(_train_spec)

    if not _train_spec.train.iters_per_epoch:
        _train_spec.train.iters_per_epoch = len(dm.train_dataloader())

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      num_nodes=_train_spec.train.num_nodes,
                      max_epochs=_train_spec.train.num_epochs,
                      check_val_every_n_epoch=1,
                      default_root_dir=_train_spec.results_dir,
                      accelerator='auto',
                      precision=precision,
                      gradient_clip_val=_train_spec.train.clip_grad_norm,
                      use_distributed_sampler=False,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.oneformer
@pytest.mark.evaluate
@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
@pytest.mark.skip(reason="flaky test pending fix")
def test_trainer_evaluate(_test_sample_json, _eval_spec, precision):
    """Test the OneFormer evaluation pipeline."""
    _eval_spec.model.backbone.name = "D2SwinTransformer"

    dm = SemSegmDataModule(_eval_spec)
    dm.setup(stage="test")
    pt_model = OneformerPlModule(_eval_spec)

    trainer = Trainer(devices=_eval_spec.evaluate.num_gpus,
                      default_root_dir=_eval_spec.results_dir,
                      accelerator='auto',
                      precision=precision,
                      fast_dev_run=FAST_DEV_RUN)
    # Test evaluate
    trainer.test(pt_model, dm)


@pytest.mark.cv_unit
@pytest.mark.oneformer
@pytest.mark.inference
@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
@pytest.mark.skip(reason="flaky test pending fix")
def test_trainer_inference(_test_sample_json, _infer_spec, precision):
    """Test the OneFormer inference pipeline."""
    _infer_spec.model.backbone.name = "D2SwinTransformer"

    dm = SemSegmDataModule(_infer_spec)
    dm.setup(stage="predict")
    pt_model = OneformerPlModule(_infer_spec)

    trainer = Trainer(devices=_infer_spec.inference.num_gpus,
                      default_root_dir=_infer_spec.results_dir,
                      accelerator='auto',
                      precision=precision,
                      fast_dev_run=FAST_DEV_RUN)
    # Test predict
    trainer.predict(pt_model, dm)

    tmp_top_obj.cleanup()
