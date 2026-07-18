# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import pytest

from nvidia_tao_pytorch.cv.ml_recog.utils.common_utils import no_folders_in

TEST_DATA_DIR = "/home/scratch.metropolis2/tao_ci/tao_pytorch/data/metric_learning_recognition"
TEST_OUTPUT_DIR = "tests/cv_unit_test/ml_recog/test_outputs"


@pytest.fixture
def _test_dir():
    output_dir = TEST_OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


def test_no_folders_in(_test_dir):
    assert not no_folders_in(os.path.join(TEST_DATA_DIR, "train"))
    assert no_folders_in(os.path.join(TEST_DATA_DIR, "train", "c000001"))
    shutil.rmtree(TEST_OUTPUT_DIR)
