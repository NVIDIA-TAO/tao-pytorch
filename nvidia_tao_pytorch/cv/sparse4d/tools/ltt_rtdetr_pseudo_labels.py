# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-compatible CLI alias for :mod:`rtdetr_pseudo_labels`."""

from nvidia_tao_pytorch.cv.sparse4d.tools.rtdetr_pseudo_labels import main


if __name__ == "__main__":
    raise SystemExit(main())
