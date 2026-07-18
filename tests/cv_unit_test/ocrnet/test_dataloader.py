# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import io
import lmdb
import os
import pytest
import numpy as np
import shutil
from PIL import Image
from nvidia_tao_pytorch.config.ocrnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocrnet.dataloader.ocr_dataset import (LmdbDataset,
                                                                 RawGTDataset,
                                                                 ResizeNormalize,
                                                                 NormalizePAD,
                                                                 AlignCollateVal,
                                                                 AlignCollate)

DEFAULT_LABEL= "0123456789abcdefghijklmnopqrstuvwxyz"
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 100

@pytest.fixture
def _test_lmdb():
    # Force GC of any leaked LmdbDataset from a prior test; lmdb 2.x rejects
    # double-open on the same path.
    gc.collect()
    img = Image.fromarray(np.random.randint(low=0, high=255, size=(DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8))
    sample_cnt = 16
    # img_bin = io.BytesIO()
    label = DEFAULT_LABEL
    # img.save(img_bin, format="png")
    tmp_img_path = "tmp_img.png"
    img.save(tmp_img_path)
    with open(tmp_img_path, 'rb') as f:
        img_bin = f.read()

    cache = {}
    for i in range(1, sample_cnt + 1):
        imageKey = 'image-%09d'.encode() % i
        labelKey = 'label-%09d'.encode() % i
        cache[imageKey] = img_bin
        cache[labelKey] = label.encode()

    lmdb_path = "tmp_lmdb/"
    cache['num-samples'.encode()] = str(sample_cnt).encode()
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=1099511627776)

    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)
    # lmdb 2.x raises "already open in this process" if the consumer test
    # tries to lmdb.open() the same path while this fixture's env is alive.
    # Close it before yielding so OCRDataset can open it cleanly.
    env.close()

    yield lmdb_path
    shutil.rmtree(lmdb_path)
    os.remove(tmp_img_path)


@pytest.fixture
def _test_raw():
    img = Image.fromarray(np.random.randint(low=0, high=255, size=(DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8))
    sample_cnt = 16
    # img_bin = io.BytesIO()
    label = DEFAULT_LABEL
    # img.save(img_bin, format="png")
    tmp_img_dir = "./tmp"
    os.makedirs(tmp_img_dir)
    gt_file_path = "./gt.txt"
    with open(gt_file_path, "w") as f:
        for idx in range(sample_cnt):
            tmp_img_path = f"tmp_img_{idx}.png"
            f.write(f"{tmp_img_path} {label}\n")
            img.save(os.path.join(tmp_img_dir, tmp_img_path))
    
    yield (tmp_img_dir, gt_file_path)
    shutil.rmtree(tmp_img_dir)
    os.remove(gt_file_path)


@pytest.fixture
def _test_opt():
    parser = argparse.ArgumentParser()
    opt, _ = parser.parse_known_args()
    opt.rgb = True
    opt.character = DEFAULT_LABEL
    opt.data_filtering_off = True
    opt.batch_max_length = 25
    opt.imgH=DEFAULT_HEIGHT
    opt.imgW=DEFAULT_WIDTH

    yield opt


@pytest.fixture
def _test_batch(_test_lmdb, _test_opt):
    opt = _test_opt
    root = _test_lmdb
    lmdb_dataset = LmdbDataset(root, opt)
    img, label = lmdb_dataset[0]
    
    yield [[img, label], [img, label]]


@pytest.fixture
def _test_img():
    img = Image.fromarray(np.random.randint(low=0, high=255, size=(32, 96, 3), dtype=np.uint8))
    yield img


@pytest.mark.ocrnet
def test_lmdb_datset(_test_lmdb, _test_opt):
    opt = _test_opt
    root = _test_lmdb
    lmdb_dataset = LmdbDataset(root, opt)
    assert len(lmdb_dataset) == 16, "Dummy dataset's length should match default value"
    
    img, label = lmdb_dataset[0]
    assert img.size == (DEFAULT_WIDTH, DEFAULT_HEIGHT), "Dummy dataset's image output shape should match default value"
    assert label == DEFAULT_LABEL, "Dummy dataset's label should match default value"

    # lmdb 2.x rejects opening the same path twice in one process. Drop the
    # first dataset and force GC so its lmdb.Environment is fully released
    # before the second LmdbDataset instantiation re-opens the same env.
    del lmdb_dataset
    gc.collect()

    opt.data_filtering_off = False
    lmdb_dataset = LmdbDataset(root, opt)
    assert len(lmdb_dataset) == 0, "Dummy dataset's length should match default value"


@pytest.mark.ocrnet
def test_raw_dataset(_test_raw, _test_opt):
    img_dir, gt_file = _test_raw
    opt = _test_opt
    raw_dataset = RawGTDataset(gt_file, img_dir, opt)
    assert len(raw_dataset) == 16, "Dummy dataset's length should match default value"

    img, label = raw_dataset[0]
    assert img.size == (DEFAULT_WIDTH, DEFAULT_HEIGHT), "Dummy dataset's image output shape should match default value"
    assert label == DEFAULT_LABEL, "Dummy dataset's label should match default value"


@pytest.mark.ocrnet
def test_resize_normalize(_test_img):
    op = ResizeNormalize(size=(100, 32))
    img = op(_test_img)
    assert img.max() <= 1.0, "Dummy augmented image's value should be less than 1.0"
    assert img.shape == (3, 32, 100), "Dummy augmented image should match default value"


#@TODO(tylerz): NormalizePAD cannot handle the orig image 
# is larger than the input size by itself. The transform is done in AlignCollate
@pytest.mark.ocrnet
def test_normalize_pad(_test_img):
    op = NormalizePAD(max_size=(3, 32, 100))
    img = op(_test_img)
    assert img.max() <= 1.0, "Dummy augmented image's value should be less than 1.0"
    assert img.shape == (3, 32, 100), "Dummy augmented image should match default value"


@pytest.mark.ocrnet
@pytest.mark.parametrize("keep_ratio_with_pad",
                         [(True),
                          (False)])
def test_align_collate_val(_test_batch, keep_ratio_with_pad):
    op = AlignCollateVal(imgH=32, imgW=100, keep_ratio_with_pad=keep_ratio_with_pad)
    _, _ = op(_test_batch)


@pytest.mark.ocrnet
@pytest.mark.parametrize("extra_aug_prob, color_reverse_prob, rotate_prob, blur_prob",
                         [(0, 0, 0, 0),
                          (0.5, 0, 0, 0),
                          (0.5, 0.5, 0.5, 0),
                          (0.5, 0.5, 0.5, 0.5)])
def test_align_collate(_test_batch, extra_aug_prob, color_reverse_prob, rotate_prob, blur_prob):
    exp = ExperimentConfig()
    exp.dataset.augmentation.aug_prob = extra_aug_prob
    exp.dataset.augmentation.reverse_color_prob = color_reverse_prob
    exp.dataset.augmentation.rotate_prob = rotate_prob
    exp.dataset.augmentation.blur_prob = blur_prob
    
    op = AlignCollate(imgH=32, imgW=100, keep_ratio_with_pad=False, experiment_spec=exp)
    _, _ = op(_test_batch)