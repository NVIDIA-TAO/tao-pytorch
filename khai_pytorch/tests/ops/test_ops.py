# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import khai_data_services
import khai_pytorch


def test_ops_import():
    assert khai_pytorch.__name__ == "khai_pytorch"
    assert khai_data_services.__name__ == "khai_data_services"
