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
#
# Vendored and adapted from OpenGVLab InternVideo2 (multi_modality), Apache-2.0:
# https://github.com/OpenGVLab/InternVideo

"""Vendored InternVideo2-CLIP (L14) architecture for the TAO video_clip module.

Trimmed to the L14 distillation model (``InternVideo2_CLIP_small`` with the
InternVideo2 vision tower and MobileCLIP text tower). The full 1B/6B CLIP
wrapper and its LLaMA/InternVL text tower are intentionally not vendored.
"""
from .internvideo2_clip_small import InternVideo2_CLIP_small

__all__ = ["InternVideo2_CLIP_small"]
