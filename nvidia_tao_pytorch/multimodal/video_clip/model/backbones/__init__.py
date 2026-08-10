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

"""Vendored backbone architectures for the TAO video_clip module.

Subpackages:
    internvideo2: InternVideo2-CLIP L14 (InternVideo2 vision tower + MobileCLIP
        text tower), vendored from OpenGVLab InternVideo.

This file is required for packaging: ``setup.py`` collects modules with
``setuptools.find_packages()``, which does not descend into directories without
an ``__init__.py``. Without it the whole ``model.backbones`` subtree is dropped
from the wheel, and the container fails at import even though the source tree
runs fine via implicit namespace packages.

Deliberately left free of imports so that importing the package does not pull in
the vision tower and its dependencies.
"""
