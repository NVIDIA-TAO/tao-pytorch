# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter labels by LIDAR."""
import os
import sys
from nvidia_tao_pytorch.core.path_utils import expand_path


def filter_labels(label_dir, lidar_dir, val_label_dir):
    """filter labels by lidar."""
    labels, lidars = [], []
    if os.path.isdir(label_dir):
        labels = os.listdir(label_dir)
    if os.path.isdir(lidar_dir):
        lidars = os.listdir(lidar_dir)
    for label in labels:
        lidar = label[:-4] + ".bin"
        if lidar not in lidars:
            print("Moving to ", os.path.join(val_label_dir, label))
            os.rename(os.path.join(label_dir, label), os.path.join(val_label_dir, label))


if __name__ == "__main__":
    label_dir = expand_path(sys.argv[1])
    lidar_dir = expand_path(sys.argv[2])
    val_label_dir = expand_path(sys.argv[3])
    filter_labels(label_dir, lidar_dir, val_label_dir)
