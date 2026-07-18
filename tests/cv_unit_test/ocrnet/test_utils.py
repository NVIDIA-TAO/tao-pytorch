# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import argparse
import torch
import os
from nvidia_tao_pytorch.cv.ocrnet.utils.utils import CTCLabelConverter, save_checkpoint, load_checkpoint

@pytest.fixture
def _test_characters():
    ch_list = "0123456789abcdefghijklmnopqrstuvwxyz"
    yield ch_list


@pytest.mark.ocrnet
def test_ctc_label_encode(_test_characters):
    ctc_label_converter = CTCLabelConverter(character=_test_characters)
    
    raw_input = ["nvidiarocks"]
    expected = [24, 32, 19, 14, 19, 11, 28, 25, 13, 21, 29,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0]
    expected = np.array(expected)
    encoded_out = ctc_label_converter.encode(raw_input)
    assert encoded_out[0][0].shape == (25,), "Dummy encoded label shape should match the default value"
    assert (encoded_out[0][0].cpu().numpy() == expected).all(), "Dummy encoded label should match the default value"
    assert encoded_out[1].shape == (1,), "Dummy max label length should match the default value"


@pytest.mark.ocrnet
@pytest.mark.parametrize("raw_index, expected",
                         [(np.array([[24, 32, 19, 14, 19, 11, 28, 25, 13, 21, 29,  0,  0,  0,  0,  0,  0,  0,
                            0,  0,  0,  0,  0,  0,  0]]), "nvidiarocks"),
                          (np.array([[24, 24, 32, 19, 14, 19, 11, 28, 25, 13, 21, 29,  0,  0,  0,  0,  0,  0,  0,
                            0,  0,  0,  0,  0,  0]]), "nvidiarocks"),
                          (np.array([[24, 0, 32, 19, 14, 19, 11, 28, 25, 13, 21, 29,  0,  0,  0,  0,  0,  0,  0,
                            0,  0,  0,  0,  0,  0]]), "nvidiarocks")])
def test_ctc_label_decode(_test_characters, raw_index, expected):
    ctc_label_converter = CTCLabelConverter(character=_test_characters)
    
    encoded_out = ctc_label_converter.decode(raw_index, [len(raw_index[0, :])])
    assert encoded_out[0] == expected, "Dummy decoded output should match the default value"


@pytest.mark.ocrnet
def test_save_load_ckpt():
    fake_ckpt = {"state_dict": {"a": 1, "b": 2, "c": 3}}
    temp_ckpt_path = "temp.tlt"
    key = "nvidia_tao"
    save_checkpoint(fake_ckpt, temp_ckpt_path, key)
    load_checkpoint(temp_ckpt_path, key)
    os.remove(temp_ckpt_path)
