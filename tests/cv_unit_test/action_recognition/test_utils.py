# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from PIL import Image
import pytest
import torchvision.transforms as transforms
from nvidia_tao_pytorch.core.utilities import patch_decrypt_checkpoint
from nvidia_tao_pytorch.cv.action_recognition.utils.group_transforms import (GroupWorker,
                                                          GroupRandomCrop,
                                                          MultiScaleCrop,
                                                          GroupRandomHorizontalFlip,
                                                          ToNumpyNDArray,
                                                          ToTorchFormatTensor,
                                                          GroupNormalize,
                                                          )
from nvidia_tao_pytorch.core.connectors.checkpoint_connector import encrypt_checkpoint


@pytest.fixture
def _test_pil_img_group_random():
    img = np.random.randint(low=0, high=255, size=(240, 320, 3), dtype=np.uint8)
    seq_len = 32
    tmp_im = [Image.fromarray(img) for _ in range(seq_len)]
    yield tmp_im


@pytest.fixture
def _test_pil_img_group_255():
    img = np.random.randint(low=255, high=256, size=(240, 320, 3), dtype=np.uint8)
    seq_len = 32
    tmp_im = [Image.fromarray(img) for _ in range(seq_len)]
    yield tmp_im


@pytest.mark.cv_unit
def test_patch_decrypt_ckpt():
    fake_ckpt = {"state_dict": {"a": 1, "b": 2, "c": 3}}

    encrypted_ckpt = encrypt_checkpoint(fake_ckpt, "tao")

    assert encrypted_ckpt["state_dict_encrypted"] is not False

    decrypted_ckpt = patch_decrypt_checkpoint(encrypted_ckpt, "tao")

    assert decrypted_ckpt["state_dict_encrypted"] is False


@pytest.mark.cv_unit
def test_group_worker(_test_pil_img_group_random):
    func = GroupWorker(transforms.Resize([224, 224]))

    ret_img = func(_test_pil_img_group_random)

    assert ret_img[0].size == (224, 224)


@pytest.mark.cv_unit
def test_group_random_crop(_test_pil_img_group_random):
    func = GroupRandomCrop((224, 224))

    ret_img = func(_test_pil_img_group_random)

    assert ret_img[0].size == (224, 224)


@pytest.mark.cv_unit
def test_group_multiscale_crop(_test_pil_img_group_random):
    func = MultiScaleCrop((224, 224))

    ret_img = func(_test_pil_img_group_random)

    assert ret_img[0].size == (224, 224)


@pytest.mark.cv_unit
def test_group_random_flip(_test_pil_img_group_random):
    func = GroupRandomHorizontalFlip(0.0)

    ret_img = func(_test_pil_img_group_random)

    img_array_a = np.array(_test_pil_img_group_random[0])
    img_array_b = np.array(ret_img[0])

    assert False not in (img_array_a == img_array_b)


@pytest.mark.cv_unit
def test_group_common_preprocess(_test_pil_img_group_255):

    ret_img = ToNumpyNDArray()(_test_pil_img_group_255)

    assert ret_img.shape == (32, 240, 320, 3)

    ret_img = ToTorchFormatTensor()(ret_img)

    assert ret_img.shape == (3, 32, 240, 320)
    assert ret_img[0, 0, 0, 0] == 1


@pytest.mark.cv_unit
def test_group_normalize(_test_pil_img_group_255):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    func = GroupNormalize(mean, std)

    test_input = ToNumpyNDArray()(_test_pil_img_group_255)
    test_input = ToTorchFormatTensor()(test_input)

    ret_img = func(test_input)

    assert ret_img.shape == (3, 32, 240, 320)
    assert ret_img[0, 0, 0, 0] == 1
