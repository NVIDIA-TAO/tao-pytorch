# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for dataset.deterministic_label_order (ODVG caption token-order determinism).

Two complementary tests:
  1. Flag unit test (in-process): config default is True for grounding_dino and
     mask_grounding_dino; the serialized ODVG dataset honors the flag, iterates in
     both modes without error (also a regression guard for the Python 3.11+
     random.sample(set) crash), and yields only valid label names.
  2. Determinism test (subprocess): PYTHONHASHSEED is fixed at interpreter startup,
     so hash-independence can only be exercised across processes. Two subprocesses
     with PYTHONHASHSEED=0 and =1 dump caption order; with the flag True the order is
     identical across hash seeds (the guarantee), with it False the order differs
     (the legacy bug the flag closes).
"""

import os
import sys
import json
import subprocess

import pytest
from omegaconf import OmegaConf


def _make_synthetic_odvg(tmp_path, n_classes=12, n_images=8):
    """Write a tiny self-contained ODVG detection dataset (dummy images + jsonl + labelmap).

    n_classes is kept >= 12 so that the hash-seed-dependent set-iteration order is
    (with overwhelming probability) different from the sorted order and between two
    PYTHONHASHSEED values.
    """
    from PIL import Image

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    labelmap = {str(i): f"class_{i:02d}" for i in range(n_classes)}
    (tmp_path / "labelmap.json").write_text(json.dumps(labelmap))

    lines = []
    for k in range(n_images):
        Image.new("RGB", (8, 8)).save(img_dir / f"img{k}.jpg")
        # two positive classes per image -> non-empty neg set -> exercises both
        # list(pos_labels) order and the negative sampling order.
        labels = [k % n_classes, (k * 3 + 1) % n_classes]
        instances = [{"bbox": [0, 0, 4, 4], "label": lb} for lb in labels]
        lines.append(json.dumps({
            "file_name": f"img{k}.jpg", "height": 8, "width": 8,
            "detection": {"instances": instances},
        }))
    (tmp_path / "anno.jsonl").write_text("\n".join(lines))
    return f"{img_dir}/", str(tmp_path / "anno.jsonl"), str(tmp_path / "labelmap.json")


# Runs in a fresh interpreter (so PYTHONHASHSEED takes effect). Dumps the first-K
# captions for both flag values as one JSON line prefixed with RESULT:.
_HELPER = r"""
import sys, json
import torch  # noqa: F401
from pytorch_lightning import seed_everything
from nvidia_tao_pytorch.cv.grounding_dino.dataloader.serialized_dataset import (
    load_coco_jsonl, ODVGSerializedDatasetFromList,
)
IMG, ANNO, LMAP, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

def caps(flag):
    seed_everything(1234, workers=True)
    lst = load_coco_jsonl(ANNO, image_root=IMG, labelmap_file=LMAP)
    ds = ODVGSerializedDatasetFromList(lst, transforms=None, max_labels=80,
                                       deterministic_label_order=flag)
    return [ds[i][1]["caption"] for i in range(min(K, len(ds)))]

print("RESULT:" + json.dumps({"true": caps(True), "false": caps(False)}))
"""


def _run_helper(img, anno, lmap, hashseed, k=8):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    proc = subprocess.run(
        [sys.executable, "-c", _HELPER, img, anno, lmap, str(k)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"helper failed (hashseed={hashseed}):\n{proc.stderr[-3000:]}"
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")][-1]
    return json.loads(line[len("RESULT:"):])


@pytest.mark.cv_unit
def test_deterministic_label_order_default_true():
    """The flag defaults to True on grounding_dino and (inherited) mask_grounding_dino."""
    from nvidia_tao_pytorch.config.grounding_dino.default_config import ExperimentConfig as GDINOExp
    from nvidia_tao_pytorch.config.mask_grounding_dino.default_config import ExperimentConfig as MaskExp

    assert OmegaConf.structured(GDINOExp).dataset.deterministic_label_order is True
    assert OmegaConf.structured(MaskExp).dataset.deterministic_label_order is True


@pytest.mark.cv_unit
def test_dataset_honors_flag(tmp_path):
    """Dataset stores the flag and iterates in both modes without error (incl. the
    Python 3.11+ random.sample(set) regression), yielding only valid label names."""
    from pytorch_lightning import seed_everything
    from nvidia_tao_pytorch.cv.grounding_dino.dataloader.serialized_dataset import (
        load_coco_jsonl, ODVGSerializedDatasetFromList,
    )

    img, anno, lmap = _make_synthetic_odvg(tmp_path)
    valid = set(json.load(open(lmap)).values())

    for flag in (True, False):
        seed_everything(1234, workers=True)
        lst = load_coco_jsonl(anno, image_root=img, labelmap_file=lmap)
        ds = ODVGSerializedDatasetFromList(lst, transforms=None, max_labels=80,
                                           deterministic_label_order=flag)
        assert ds.deterministic_label_order is flag
        for i in range(min(4, len(ds))):
            _, target = ds[i]
            assert set(target["cap_list"]).issubset(valid)
            assert len(target["cap_list"]) > 0


@pytest.mark.cv_unit
def test_caption_order_hash_independent(tmp_path):
    """With the flag True, caption order is identical across PYTHONHASHSEED values;
    with it False, it differs (the legacy non-determinism the flag closes)."""
    img, anno, lmap = _make_synthetic_odvg(tmp_path)
    r0 = _run_helper(img, anno, lmap, hashseed=0)
    r1 = _run_helper(img, anno, lmap, hashseed=1)

    # The guarantee: sorted() canonicalization -> hash-seed independent.
    assert r0["true"] == r1["true"], "deterministic_label_order=True must be PYTHONHASHSEED-independent"
    # The bug it closes: legacy set-iteration order flips with the hash seed.
    assert r0["false"] != r1["false"], "expected legacy (flag off) caption order to vary with PYTHONHASHSEED"
