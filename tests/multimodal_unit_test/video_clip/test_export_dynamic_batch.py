# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for dynamic ONNX export batch semantics."""

import pytest

from nvidia_tao_pytorch.multimodal.video_clip.scripts.export import (
    _resolve_export_batch_size,
)


@pytest.mark.multimodal_unit
def test_dynamic_batch_uses_non_specialized_dynamo_sample():
    assert _resolve_export_batch_size(-1, use_dynamo=True) == (2, True)
    assert _resolve_export_batch_size(None, use_dynamo=True) == (2, True)
    assert _resolve_export_batch_size(-1, use_dynamo=False) == (1, True)
    assert _resolve_export_batch_size(10, use_dynamo=True) == (10, False)
    for invalid in (0, -2):
        with pytest.raises(ValueError, match="must be -1 or a positive"):
            _resolve_export_batch_size(invalid, use_dynamo=True)
