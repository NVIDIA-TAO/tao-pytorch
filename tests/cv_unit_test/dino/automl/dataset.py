# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
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

"""Synthetic DINO datasets for AutoML integration tests."""

import json

import numpy as np
from PIL import Image


def create_tiny_coco_dataset(tmp_path):
    """Create a tiny generated COCO dataset under pytest's temp directory."""
    dataset_root = tmp_path / "images"
    dataset_root.mkdir()
    rng = np.random.default_rng(1234)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "object", "id": 1, "name": "person"},
            {"supercategory": "object", "id": 2, "name": "face"},
            {"supercategory": "object", "id": 3, "name": "bag"},
        ],
    }

    ann_id = 0
    for image_id in range(6):
        image = Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
        file_name = f"sample_{image_id}.jpg"
        image.save(dataset_root / file_name)
        coco["images"].append({
            "id": image_id,
            "file_name": file_name,
            "height": 64,
            "width": 64,
        })
        for category_id in (1, 2):
            x1 = int(rng.integers(1, 32))
            y1 = int(rng.integers(1, 32))
            w = int(rng.integers(8, 24))
            h = int(rng.integers(8, 24))
            coco["annotations"].append({
                "image_id": image_id,
                "category_id": category_id,
                "id": ann_id,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    annotation_file = tmp_path / "annotations.json"
    annotation_file.write_text(json.dumps(coco), encoding="utf-8")
    return dataset_root, annotation_file
