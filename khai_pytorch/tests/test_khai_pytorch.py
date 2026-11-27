# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import khai_pytorch
from khai.core.cv import container


def test_import():
    assert khai_pytorch.__name__ == "khai_pytorch"
    assert container.CONTRACT_VERSION >= 1
