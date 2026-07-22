# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import pickle
import numpy as np
import shutil
from nvidia_tao_pytorch.cv.pose_classification.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.pose_classification.dataloader.skeleton_feeder import (auto_pad, random_choose, random_move)


@pytest.fixture
def _test_dir():
    tmp_top_dir = "tmp"
    if not os.path.exists(tmp_top_dir):
        os.makedirs(tmp_top_dir)

    test_data = np.random.rand(3, 3, 300, 34, 1)
    np.save(file=os.path.join(tmp_top_dir, "test_data.npy"), 
            arr=test_data, allow_pickle=False)

    test_label = [["a", "b", "c"], [0, 1, 2]]
    with open(os.path.join(tmp_top_dir, "test_label.pkl"), "wb") as f:
        pickle.dump(test_label, f, protocol=4)

    yield tmp_top_dir
    shutil.rmtree(tmp_top_dir)


@pytest.fixture
def _test_spec():
    stage = "test"
    data_path = "tmp/test_data.npy"
    label_path = "tmp/test_label.pkl"
    label_map = {"a": 0, "b": 1, "c": 2}
    spec = {"stage": stage,
            "data_path": data_path,
            "label_path": label_path,
            "label_map": label_map}
    yield spec


@pytest.mark.cv_unit
def test_auto_pad():
    dummy_input = np.random.rand(3, 300, 34, 1)

    output = auto_pad(dummy_input, 200, False)
    assert np.array_equal(dummy_input, output) == True, "dummy_input and output should match."
    
    output = auto_pad(dummy_input, 400, False)
    assert output.shape == (3, 400, 34, 1), f"output.shape should be (3, 400, 34, 1). "\
        f"Got {output.shape}."

    output = auto_pad(dummy_input, 400, True)
    assert output.shape == (3, 400, 34, 1), f"output.shape should be (3, 400, 34, 1). "\
        f"Got {output.shape}."


@pytest.mark.cv_unit
def test_random_choose():
    dummy_input = np.random.rand(3, 300, 34, 1)

    output = random_choose(dummy_input, 300, True)
    assert np.array_equal(dummy_input, output) == True, "dummy_input and output should match."
    
    output = random_choose(dummy_input, 400, True)
    assert output.shape == (3, 400, 34, 1), f"output.shape should be (3, 400, 34, 1). "\
        f"Got {output.shape}."

    output = random_choose(dummy_input, 400, False)
    assert np.array_equal(dummy_input, output) == True, "dummy_input and output should match."

    output = random_choose(dummy_input, 200, True)
    assert output.shape == (3, 200, 34, 1), f"output.shape should be (3, 200, 34, 1). "\
        f"Got {output.shape}."


@pytest.mark.cv_unit
def test_random_move():
    dummy_input = np.random.rand(3, 300, 34, 1)

    output = random_move(dummy_input)
    assert output.shape == (3, 300, 34, 1), f"output.shape should be (3, 300, 34, 1). "\
        f"Got {output.shape}."


@pytest.mark.cv_unit
@pytest.mark.parametrize("random_choose, random_move",
                         [(False, False),
                          (False, True),
                          (True, False),
                          (True, True)])
def test_build_dataloader(_test_dir, _test_spec, 
                          random_choose, random_move):
    stage = _test_spec["stage"]
    data_path = _test_spec["data_path"]
    label_path = _test_spec["label_path"]
    label_map = _test_spec["label_map"]
    build_dataloader(stage=stage,
                     data_path=data_path,
                     label_path=label_path,
                     label_map=label_map,
                     random_choose=random_choose,
                     random_move=random_move)
