# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utils for metric-learning recognition."""

import os


def no_folders_in(path_to_parent):
    """Checks whether folders exist in the directory.

    Args:
        path_to_parent (String): a directory for an image file or folder.

    Returns:
        no_folders (Boolean): If true, the directory is an image folder, otherwise it's a classifcation folder.
    """
    no_folders = True
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent, fname)):
            no_folders = False
            break

    return no_folders
