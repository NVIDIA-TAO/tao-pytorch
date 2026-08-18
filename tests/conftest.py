# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

def pytest_unconfigure(config):
    if hasattr(config, "testmon_data"):
        config.testmon_data.db.con.close()