# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video frames sampler."""
import pytest
import random
import numpy as np


def random_interval_sample(max_sample_cnt, seq_length):
    """Randomly sample frames.

    This function takes the maximum sample count and sequence length as input. It randomly samples frames based on
    the specified parameters and returns the sampled image IDs as a sorted numpy array.

    Args:
        max_sample_cnt (int): The maximum sample count.
        seq_length (int): The sequence length.

    Returns:
        numpy.ndarray: The sampled image IDs as a sorted numpy array.
    """
    if max_sample_cnt < seq_length:
        # choose all the images and randomly duplicate
        img_ids = np.sort(np.random.randint(max_sample_cnt, size=seq_length))
    else:
        seq_interval = max_sample_cnt // seq_length
        img_ids = np.sort((np.arange(seq_length) * seq_interval +
                           np.random.randint(seq_interval, size=seq_length)))
    return img_ids


def random_consecutive_sample(max_sample_cnt, seq_length, sample_rate=1):
    """Randomly choose a start frame and pick up the continuous frames.

    This function takes the maximum sample count, sequence length, and sample rate as input. It randomly chooses
    a start frame and picks up the continuous frames based on the specified parameters. The function returns the
    selected image IDs as a numpy array.

    Args:
        max_sample_cnt (int): The maximum sample count.
        seq_length (int): The sequence length.
        sample_rate (int, optional): The sample rate. Defaults to 1.

    Returns:
        numpy.ndarray: The selected image IDs as a numpy array.
    """
    total_frames_req = seq_length * sample_rate
    average_duration = max_sample_cnt - total_frames_req + 1
    if average_duration > 0:
        start_idx = random.randint(0, average_duration)
    else:
        start_idx = 0

    img_ids = start_idx + np.arange(seq_length) * sample_rate

    # loop the video to form sequence:
    img_ids = np.mod(img_ids, max_sample_cnt)

    return img_ids


@pytest.mark.skip(reason="Not a unit test.")
def test_interval_sample(max_sample_cnt, seq_length):
    """Choose the middle frames of each clip with interval.

    It chooses the middle frames of each clip based on the specified parameters
    and returns the selected image IDs as a numpy array.

    Args:
        max_sample_cnt (int): The maximum sample count.
        seq_length (int): The sequence length.

    Returns:
        numpy.ndarray: The selected image IDs as a numpy array.
    """
    clip_interval = max_sample_cnt / float(seq_length)
    img_ids = np.array([int(clip_interval / 2.0 + clip_interval * x) for x in range(seq_length)])
    return img_ids


@pytest.mark.skip(reason="Not a unit test.")
def test_consecutive_sample(max_sample_cnt, seq_length, sample_rate=1, all_frames_3d=False):
    """Choose the middle consecutive frames of each video.

    This function takes the maximum sample count, sequence length, sample rate, and all frames 3D flag as input.
    It chooses the middle consecutive frames of each video based on the specified parameters and returns the
    selected image IDs as a numpy array.

    Args:
        max_sample_cnt (int): The maximum sample count.
        seq_length (int): The sequence length.
        sample_rate (int, optional): The sample rate. Defaults to 1.
        all_frames_3d (bool, optional): Use all frames for 3D model. Defaults to False.

    Returns:
        numpy.ndarray: The selected image IDs as a numpy array.
    """
    if all_frames_3d:
        # inference on all frames for 3D model
        img_ids = np.arange(max_sample_cnt)
    else:
        total_frames_req = seq_length * sample_rate
        average_duration = max_sample_cnt - total_frames_req + 1
        if average_duration > 0:
            start_idx = int(average_duration / 2.0)
        else:
            start_idx = 0

        img_ids = start_idx + np.arange(seq_length) * sample_rate
        # loop the video to form sequence:
        img_ids = np.mod(img_ids, max_sample_cnt)

    return img_ids


def segment_sample(id_list, seq_length):
    """Randomly choose frames out of an averagely segmented frames list.

    This function takes a list of image IDs and a sequence length as input. It randomly chooses frames out of an
    averagely segmented frames list based on the specified parameters and returns the selected image IDs as a list.

    Args:
        id_list (list): The list of image IDs.
        seq_length (int): The sequence length.

    Returns:
        list: The selected image IDs as a list.
    """
    candidate_id_list = []
    max_sample_cnt = len(id_list)
    start_idx = 0
    seg_length = max_sample_cnt // seq_length
    for _ in range(seq_length - 1):
        end_idx = start_idx + seg_length - 1
        img_idx = random.randint(start_idx, end_idx)
        start_idx = start_idx + seg_length
        candidate_id_list.append(id_list[img_idx])

    end_idx = max_sample_cnt - 1
    img_idx = random.randint(start_idx, end_idx)
    candidate_id_list.append(id_list[img_idx])

    return candidate_id_list


def joint_random_interval_sample(max_sample_cnt, rgb_seq_length, of_seq_length):
    """Randomly choose RGB and optical flow images for joint model training with random interval.

    This function takes the maximum sample count, RGB sequence length, and optical flow sequence length as input.
    It randomly chooses RGB and optical flow images for joint model training based on the specified parameters
    and returns the selected image IDs as a tuple.

    Args:
        max_sample_cnt (int): The maximum sample count.
        rgb_seq_length (int): The RGB sequence length.
        of_seq_length (int): The optical flow sequence length.

    Returns:
        tuple: The selected RGB and optical flow image IDs as a tuple.
    """
    if of_seq_length > rgb_seq_length:
        of_ids = random_interval_sample(max_sample_cnt, of_seq_length)
        rgb_ids = segment_sample(of_ids, rgb_seq_length)
    elif of_seq_length < rgb_seq_length:
        rgb_ids = random_interval_sample(max_sample_cnt, rgb_seq_length)
        of_ids = segment_sample(rgb_ids, of_seq_length)
    else:
        rgb_ids = random_interval_sample(max_sample_cnt, rgb_seq_length)
        of_ids = rgb_ids

    return rgb_ids, of_ids


def joint_random_consecutive_sample(max_sample_cnt, rgb_seq_length, of_seq_length,
                                    sample_rate=1):
    """Randomly choose consecutive RGB and optical flow images for joint model training.

    This function takes the maximum sample count, RGB sequence length, optical flow sequence length, and sample rate
    as input. It randomly chooses RGB and optical flow images for joint model training based on the specified
    parameters and returns the selected image IDs as a tuple.

    Args:
        max_sample_cnt (int): The maximum sample count.
        rgb_seq_length (int): The RGB sequence length.
        of_seq_length (int): The optical flow sequence length.
        sample_rate (int, optional): The sample rate. Defaults to 1.

    Returns:
        tuple: The selected RGB and optical flow image IDs as a tuple.
    """
    if of_seq_length > rgb_seq_length:
        of_ids = random_consecutive_sample(max_sample_cnt, of_seq_length, sample_rate)
        rgb_ids = []
        can_idx = test_consecutive_sample(len(of_ids), rgb_seq_length, sample_rate)
        for idx in can_idx:
            rgb_ids.append(of_ids[idx])
    elif of_seq_length < rgb_seq_length:
        rgb_ids = random_consecutive_sample(max_sample_cnt, rgb_seq_length, sample_rate)
        of_ids = []
        can_idx = test_consecutive_sample(len(rgb_ids), of_seq_length, sample_rate)
        for idx in can_idx:
            of_ids.append(rgb_ids[idx])
    else:
        rgb_ids = random_consecutive_sample(max_sample_cnt, rgb_seq_length, sample_rate)
        of_ids = rgb_ids

    return rgb_ids, of_ids


def joint_test_interval_sample(max_sample_cnt, rgb_seq_length, of_seq_length):
    """Choose RGB and optical flow images for joint model test with consistent interval.

    This function takes the maximum sample count, RGB sequence length, and optical flow sequence length as input.
    It chooses RGB and optical flow images for joint model test with consistent interval based on the specified
    parameters and returns the selected image IDs as a tuple.

    Args:
        max_sample_cnt (int): The maximum sample count.
        rgb_seq_length (int): The RGB sequence length.
        of_seq_length (int): The optical flow sequence length.

    Returns:
        tuple: The selected RGB and optical flow image IDs as a tuple.
    """
    if of_seq_length > rgb_seq_length:
        of_ids = test_interval_sample(max_sample_cnt, of_seq_length)
        rgb_ids = []
        can_idx = test_interval_sample(len(of_ids), rgb_seq_length)
        for idx in can_idx:
            rgb_ids.append(of_ids[idx])
    elif of_seq_length < rgb_seq_length:
        rgb_ids = test_interval_sample(max_sample_cnt, rgb_seq_length)
        of_ids = []
        can_idx = test_interval_sample(len(rgb_ids), of_seq_length)
        for idx in can_idx:
            of_ids.append(rgb_ids[idx])
    else:
        rgb_ids = test_interval_sample(max_sample_cnt, rgb_seq_length)
        of_ids = rgb_ids

    return rgb_ids, of_ids


def joint_test_consecutive_sample(max_sample_cnt, rgb_seq_length, of_seq_length,
                                  sample_rate=1, all_frames_3d=False):
    """Choose consecutive RGB and optical flow images for joint model test.

    This function takes the maximum sample count, RGB sequence length, optical flow sequence length, sample rate,
    and all_frames_3d as input. It chooses consecutive RGB and optical flow images for joint model test based on the
    specified parameters and returns the selected image IDs as a tuple.

    Args:
        max_sample_cnt (int): The maximum sample count.
        rgb_seq_length (int): The RGB sequence length.
        of_seq_length (int): The optical flow sequence length.
        sample_rate (int, optional): The sample rate. Defaults to 1.
        all_frames_3d (bool, optional): Whether to choose all frames for 3D model. Defaults to False.

    Returns:
        tuple: The selected RGB and optical flow image IDs as a tuple.
    """
    if all_frames_3d:
        rgb_ids = np.arange(max_sample_cnt)
        of_ids = rgb_ids
        return rgb_ids, of_ids

    if of_seq_length > rgb_seq_length:
        of_ids = test_consecutive_sample(max_sample_cnt, of_seq_length, sample_rate)
        rgb_ids = []
        can_idx = test_consecutive_sample(len(of_ids), rgb_seq_length, sample_rate)
        for idx in can_idx:
            rgb_ids.append(of_ids[idx])
    elif of_seq_length < rgb_seq_length:
        rgb_ids = test_consecutive_sample(max_sample_cnt, rgb_seq_length, sample_rate)
        of_ids = []
        can_idx = test_consecutive_sample(len(rgb_ids), of_seq_length, sample_rate)
        for idx in can_idx:
            of_ids.append(rgb_ids[idx])
    else:
        rgb_ids = test_consecutive_sample(max_sample_cnt, rgb_seq_length)
        of_ids = rgb_ids

    return rgb_ids, of_ids


if __name__ == "__main__":
    max_sample_cnt = 58
    seq_length = 64
    sample_rate = 1

    print(random_consecutive_sample(max_sample_cnt, seq_length, sample_rate))
    print(test_consecutive_sample(max_sample_cnt, seq_length, sample_rate))
