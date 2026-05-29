# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utils for pose classification."""

import json
import numpy as np


def load_json_from_file(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path of the JSON file to load data from.

    Returns:
        dict: The data loaded from the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_np_to_file(file_path, data):
    """
    Write a Numpy array to a file.

    Args:
        file_path (str): The path where the file should be written.
        data (numpy.ndarray): The Numpy array to write to the file.
    """
    np.save(file=file_path, arr=data, allow_pickle=False)
