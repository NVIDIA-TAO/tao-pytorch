# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drop a class in dataset."""
import os
import sys
from nvidia_tao_pytorch.core.path_utils import expand_path


def drop_class(label_dir, classes):
    """drop label by class names."""
    labels = []
    if os.path.isdir(label_dir):
        labels = os.listdir(label_dir)
        labels = [os.path.join(label_dir, x) for x in labels]

    for gt in labels:
        print("Processing ", gt)
        with open(gt) as f:
            lines = f.readlines()
            lines_ret = []
            for line in lines:
                ls = line.strip()
                line = ls.split()
                if line[0] in classes:
                    print("Dropping ", line[0])
                    continue
                else:
                    lines_ret.append(ls)
        with open(gt, "w") as fo:
            out = '\n'.join(lines_ret)
            fo.write(out)


if __name__ == "__main__":
    label_dir = expand_path(sys.argv[1])
    drop_class(label_dir, sys.argv[2].split(','))
