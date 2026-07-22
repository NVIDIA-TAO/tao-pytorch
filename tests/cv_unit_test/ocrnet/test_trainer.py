# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import numpy as np
import lmdb
import os
from PIL import Image
import pytest
import tempfile
from pytorch_lightning import Trainer
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.ocrnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocrnet.dataloader.pl_ocr_data_module import OCRDataModule
from nvidia_tao_pytorch.cv.ocrnet.model.pl_ocrnet import OCRNetModel

DEFAULT_LABEL= "0123456789abcdefghijklmnopqrstuvwxyz"
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 200
FAST_DEV_RUN = 2  # Run dry run 2 times
batch_size = 32

# Per-fixture-invocation paths so that each parametrized test gets a unique
# lmdb directory. lmdb 2.x's process-wide registry rejects re-opening the
# same path, and Lightning's reference cycles can hold a prior test's
# OCRDataset (and its env) alive past gc.collect(). Unique paths sidestep
# the collision entirely.
tmp_top_dir = ""
lmdb_dir = ""
val_lmdb_dir = ""
gt_file = ""
character_file = ""


@pytest.fixture
def _test_data():
    global tmp_top_dir, lmdb_dir, val_lmdb_dir, gt_file, character_file
    gc.collect()
    tmp_top_dir = tempfile.mkdtemp(prefix="ocrnet_trainer_")
    lmdb_dir = os.path.join(tmp_top_dir, "lmdb")
    # Separate val dir so the train and val DataLoaders don't both open the
    # same lmdb path inside one process — lmdb 2.x rejects that.
    val_lmdb_dir = os.path.join(tmp_top_dir, "lmdb_val")
    gt_file = os.path.join(tmp_top_dir, "gt.txt")
    character_file = os.path.join(tmp_top_dir, "character_list")
    img = Image.fromarray(np.random.randint(low=0, high=255, size=(DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8))
    sample_cnt = batch_size
    tmp_img_path = os.path.join(tmp_top_dir, "tmp_img.png")
    img.save(tmp_img_path)
    with open(tmp_img_path, 'rb') as f:
        img_bin = f.read()

    cache = {}
    for i in range(1, sample_cnt + 1):
        imageKey = 'image-%09d'.encode() % i
        labelKey = 'label-%09d'.encode() % i
        cache[imageKey] = img_bin
        cache[labelKey] = DEFAULT_LABEL.encode()

    cache['num-samples'.encode()] = str(sample_cnt).encode()
    for path in (lmdb_dir, val_lmdb_dir):
        os.makedirs(path, exist_ok=True)
        env = lmdb.open(path, map_size=1099511627776)
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k, v)
        # lmdb 2.x raises "already open in this process" if the consumer test
        # tries to lmdb.open() the same path while this fixture's env is alive.
        env.close()

    os.makedirs(os.path.join(tmp_top_dir, 'images'), exist_ok=True)
    with open(gt_file, "w") as f:
        for idx in range(sample_cnt):
            tmp_img_path = os.path.join(tmp_top_dir, 'images', f"tmp_img_{idx}.png")
            f.write(f"{tmp_img_path} {DEFAULT_LABEL}\n")
            img.save(tmp_img_path)

    with open(character_file, "w") as f:
        for ch in DEFAULT_LABEL:
            f.write(f"{ch}\n")

    yield
    # Best-effort cleanup; ignore errors in case downstream still holds files.
    import shutil
    shutil.rmtree(tmp_top_dir, ignore_errors=True)


@pytest.fixture
def _train_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.train.results_dir = results_dir
    experiment_config.train.num_gpus = 1

    experiment_config.dataset.character_list_file = character_file
    experiment_config.dataset.max_label_length = len(DEFAULT_LABEL)

    experiment_config.model.TPS = True
    experiment_config.model.input_width = DEFAULT_WIDTH
    experiment_config.model.input_height = DEFAULT_HEIGHT

    yield experiment_config


@pytest.fixture
def _eval_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.evaluate.num_gpus = 1

    experiment_config.dataset.character_list_file = character_file
    experiment_config.dataset.max_label_length = len(DEFAULT_LABEL)

    experiment_config.model.TPS = True
    experiment_config.model.input_width = DEFAULT_WIDTH
    experiment_config.model.input_height = DEFAULT_HEIGHT

    yield experiment_config


@pytest.fixture
def _infer_spec():
    experiment_config = OmegaConf.structured(ExperimentConfig())

    results_dir = os.path.join(tmp_top_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    experiment_config.results_dir = results_dir

    experiment_config.inference.num_gpus = 1
    experiment_config.inference.inference_dataset_dir = tmp_top_dir

    experiment_config.dataset.character_list_file = character_file

    experiment_config.model.TPS = True
    experiment_config.model.input_width = DEFAULT_WIDTH
    experiment_config.model.input_height = DEFAULT_HEIGHT

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.ocrnet
@pytest.mark.train
@pytest.mark.parametrize("backbone", ["FAN_tiny_2X", "ResNet"])
@pytest.mark.parametrize("dataset", ["lmdb", "raw"])
def test_trainer_fit(_test_data, _train_spec, backbone, dataset):

    _train_spec.model.backbone = backbone
    if backbone == 'FAN_tiny_2X':
        _train_spec.model.prediction = "Attn"
    else:
        _train_spec.model.prediction = "CTC"

    if dataset == 'lmdb':
        _train_spec.dataset.train_dataset_dir = [lmdb_dir]
        _train_spec.dataset.val_dataset_dir = val_lmdb_dir
    else:
        _train_spec.dataset.train_dataset_dir = [tmp_top_dir]
        _train_spec.dataset.train_gt_file = gt_file
        _train_spec.dataset.val_dataset_dir = tmp_top_dir
        _train_spec.dataset.val_gt_file = gt_file

    dm = OCRDataModule(_train_spec)
    dm.setup(stage='fit')
    model = OCRNetModel(_train_spec, dm)
    clip_grad = _train_spec.train.clip_grad_norm

    trainer = Trainer(devices=_train_spec.train.num_gpus,
                      default_root_dir=_train_spec.results_dir,
                      gradient_clip_val=clip_grad,
                      fast_dev_run=FAST_DEV_RUN)

    # Test train
    trainer.fit(model, dm)


@pytest.mark.cv_unit
@pytest.mark.ocrnet
@pytest.mark.evaluate
@pytest.mark.parametrize("backbone", ["FAN_tiny_2X", "ResNet"])
@pytest.mark.parametrize("dataset", ["lmdb", "raw"])
def test_trainer_evaluate(_test_data, _eval_spec, backbone, dataset):

    _eval_spec.model.backbone = backbone
    if backbone == 'FAN_tiny_2X':
        _eval_spec.model.prediction = "Attn"
    else:
        _eval_spec.model.prediction = "CTC"

    if dataset == 'lmdb':
        _eval_spec.evaluate.test_dataset_dir = lmdb_dir
    else:
        _eval_spec.evaluate.test_dataset_dir = tmp_top_dir
        _eval_spec.evaluate.test_dataset_gt_file = gt_file

    dm = OCRDataModule(_eval_spec)
    dm.setup(stage='test')
    model = OCRNetModel(_eval_spec, dm)

    trainer = Trainer(devices=_eval_spec.train.num_gpus,
                      default_root_dir=_eval_spec.results_dir,
                      fast_dev_run=FAST_DEV_RUN)

    # Test evaluate
    trainer.test(model, dm)


@pytest.mark.cv_unit
@pytest.mark.ocrnet
@pytest.mark.inference
@pytest.mark.parametrize("backbone", ["FAN_tiny_2X", "ResNet"])
def test_trainer_inference(_test_data, _infer_spec, backbone):

    _infer_spec.model.backbone = backbone
    if backbone == 'FAN_tiny_2X':
        _infer_spec.model.prediction = "Attn"
    else:
        _infer_spec.model.prediction = "CTC"

    dm = OCRDataModule(_infer_spec)
    dm.setup(stage='predict')
    model = OCRNetModel(_infer_spec, dm)

    trainer = Trainer(devices=_infer_spec.train.num_gpus,
                      default_root_dir=_infer_spec.results_dir,
                      fast_dev_run=FAST_DEV_RUN)

    # Test predict
    trainer.predict(model, dm)
