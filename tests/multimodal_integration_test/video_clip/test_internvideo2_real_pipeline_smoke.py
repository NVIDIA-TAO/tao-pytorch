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

"""Opt-in real InternVideo2-CLIP train/eval/inference smoke test."""

import json
import os
from pathlib import Path

import h5py
import pytest
import torch
from omegaconf import OmegaConf

# Use the canonical config the scripts import (nvidia_tao_pytorch.config), not the
# nvidia_tao_core mirror, so the smoke test validates against the real schema.
from nvidia_tao_pytorch.config.video_clip.default_config import VideoCLIPExperimentConfig
from nvidia_tao_pytorch.multimodal.video_clip.scripts import evaluate, inference, train


VISION_CKPT = Path(
    "/workspace/alicli/hub/models--OpenGVLab--InternVideo2_distillation_models/"
    "snapshots/449f7ea1d7d3b70b6b5630e70d238b44d3b7aaac/stage1/L14/"
    "L14_dist_1B_stage2/pytorch_model.bin"
)
TEXT_CKPT = Path("/workspace/alicli/models/internvideo2_clip_l14/mobileclip_blt.pt")
CLIP_HEAD_CKPT = Path(
    "/workspace/alicli/hub/models--OpenGVLab--InternVideo2_distillation_models/"
    "snapshots/449f7ea1d7d3b70b6b5630e70d238b44d3b7aaac/clip/L14/"
    "pytorch_model.bin"
)
REAL_VIDEO = Path(
    "/workspace/alicli/datasets/Vad-R1/Vad-Reasoning-SFT/NEW/"
    "28482404788-1-192-10.mp4"
)


def _require_real_pipeline_env():
    if os.getenv("RUN_INTERNVIDEO2_REAL_PIPELINE") != "1":
        pytest.skip("set RUN_INTERNVIDEO2_REAL_PIPELINE=1 to run the real smoke")
    if not torch.cuda.is_available():
        pytest.skip("real InternVideo2 smoke requires CUDA")
    required_paths = {
        "InternVideo2 vision checkpoint": VISION_CKPT,
        "MobileCLIP text checkpoint": TEXT_CKPT,
        "InternVideo2 CLIP head": CLIP_HEAD_CKPT,
        "Vad-R1 sample video": REAL_VIDEO,
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        pytest.skip(f"missing required real assets: {', '.join(missing)}")


def _write_smoke_metadata(tmp_path):
    metadata_path = tmp_path / "internvideo2_smoke_vadr1.json"
    video_path = str(REAL_VIDEO)
    rows = [
        {
            "split": "train",
            "dataset": "SMOKE",
            "video_id": "internvideo2-smoke-train",
            "anomaly_type": "vehicle stopped in roadway",
            "total_frames": 300,
            "video_duration_sec": 10.0,
            "fps": 30.0,
            "what": "a vehicle is stopped in the lane",
            "where": "center lane",
            "why": "a stopped vehicle blocks normal traffic flow",
            "how": "nearby vehicles need to slow down or change course",
            "video_path": video_path,
            "chunks": [
                {
                    "chunk_index": 0,
                    "start_time_sec": 0.0,
                    "end_time_sec": 5.0,
                    "start_frame": 0,
                    "end_frame": 150,
                    "is_anomaly": True,
                    "overlap_ratio": 1.0,
                },
                {
                    "chunk_index": 1,
                    "start_time_sec": 5.0,
                    "end_time_sec": 10.0,
                    "start_frame": 150,
                    "end_frame": 300,
                    "is_anomaly": True,
                    "overlap_ratio": 1.0,
                },
            ],
        },
        {
            "split": "test",
            "dataset": "SMOKE",
            "video_id": "internvideo2-smoke-test",
            "anomaly_type": "vehicle stopped in roadway",
            "total_frames": 300,
            "video_duration_sec": 10.0,
            "fps": 30.0,
            "what": "traffic is interrupted by a stopped vehicle",
            "where": "road ahead",
            "why": "the obstacle requires defensive driving",
            "how": "drivers must adjust speed and trajectory",
            "video_path": video_path,
            "chunks": [
                {
                    "chunk_index": 0,
                    "start_time_sec": 0.0,
                    "end_time_sec": 5.0,
                    "start_frame": 0,
                    "end_frame": 150,
                    "is_anomaly": True,
                    "overlap_ratio": 1.0,
                },
                {
                    "chunk_index": 1,
                    "start_time_sec": 5.0,
                    "end_time_sec": 10.0,
                    "start_frame": 150,
                    "end_frame": 300,
                    "is_anomaly": True,
                    "overlap_ratio": 1.0,
                },
            ],
        },
    ]
    metadata_path.write_text(json.dumps(rows), encoding="utf-8")
    return metadata_path


def _write_text_prompts(tmp_path):
    text_path = tmp_path / "prompts.txt"
    text_path.write_text(
        "a stopped vehicle blocks a road\n"
        "traffic flow is disrupted by an obstacle\n",
        encoding="utf-8",
    )
    return text_path


def _build_config(tmp_path, metadata_path):
    cfg = OmegaConf.structured(VideoCLIPExperimentConfig())
    OmegaConf.set_struct(cfg, False)

    cfg.results_dir = str(tmp_path / "results")
    cfg.encryption_key = "nvidia_tao"
    if hasattr(cfg, "wandb"):
        cfg.wandb.enable = False

    cfg.model.type = "internvideo2-clip-l14"
    cfg.model.internvideo2clip_hf_id = None
    cfg.model.vision_encoder = str(VISION_CKPT)
    cfg.model.text_encoder = str(TEXT_CKPT)
    cfg.model.clip_head = str(CLIP_HEAD_CKPT)
    cfg.model.image_size = 224
    cfg.model.num_frames = 8
    cfg.model.freeze_vision_encoder = True
    cfg.model.freeze_text_encoder = True

    cfg.train.num_epochs = 1
    cfg.train.validation_interval = 1
    cfg.train.checkpoint_interval = 1
    cfg.train.checkpoint_interval_unit = "epoch"
    cfg.train.precision = "bf16"
    cfg.train.loss_type = "internvideo2_vtc"
    cfg.train.optim.vision_lr = 1e-8
    cfg.train.optim.text_lr = 1e-8
    cfg.train.optim.warmup_steps = 0
    cfg.train.optim.scheduler = "constant"

    cfg.dataset.pin_memory = False
    cfg.dataset.train.type = "video_text"
    cfg.dataset.train.batch_size = 2
    cfg.dataset.train.num_workers = 0
    cfg.dataset.train.video_text.metadata = str(metadata_path)
    cfg.dataset.train.video_text.format = "vadr1_chunks"
    cfg.dataset.train.video_text.split = "train"
    cfg.dataset.train.video_text.caption_fields = ["what"]
    cfg.dataset.train.video_text.idx_mode = "sample_id"
    cfg.dataset.train.video_text.num_frames = 8

    cfg.dataset.val.type = "video_text"
    cfg.dataset.val.batch_size = 2
    cfg.dataset.val.num_workers = 0
    cfg.dataset.val.video_text.metadata = str(metadata_path)
    cfg.dataset.val.video_text.format = "vadr1_chunks"
    cfg.dataset.val.video_text.split = "test"
    cfg.dataset.val.video_text.caption_fields = ["what"]
    cfg.dataset.val.video_text.idx_mode = "sample_id"
    cfg.dataset.val.video_text.num_frames = 8

    cfg.evaluate.checkpoint = None
    cfg.evaluate.batch_size = 2
    cfg.evaluate.num_workers = 0

    cfg.inference.checkpoint = None
    cfg.inference.batch_size = 1
    cfg.inference.num_workers = 0
    cfg.dataset.inference.type = "video_text"
    cfg.dataset.inference.video_text.metadata = str(metadata_path)
    cfg.dataset.inference.video_text.format = "vadr1_chunks"
    cfg.dataset.inference.video_text.split = "test"
    cfg.dataset.inference.video_text.caption_fields = ["what"]
    cfg.dataset.inference.video_text.idx_mode = "sample_id"
    cfg.dataset.inference.video_text.num_frames = 8
    return cfg


def _latest_checkpoint(results_dir):
    checkpoints = sorted(Path(results_dir).glob("*.pth"))
    assert checkpoints, f"no .pth checkpoint written under {results_dir}"
    latest = Path(results_dir) / "clip_latest.pth"
    return latest if latest.exists() else checkpoints[-1]


def _assert_nonempty_embeddings(path):
    assert path.exists(), f"missing embedding output: {path}"
    with h5py.File(path, "r") as f:
        assert "embeddings" in f
        embeddings = f["embeddings"]
        assert embeddings.shape[0] > 0
        assert embeddings.shape[1] > 0


@pytest.mark.multimodal_integration
@pytest.mark.slow
def test_internvideo2_real_train_eval_inference_pipeline(tmp_path, monkeypatch):
    """Run one real train batch, evaluation, and video/text inference."""
    _require_real_pipeline_env()
    monkeypatch.setenv("TAO_VISIBLE_DEVICES", os.getenv("TAO_VISIBLE_DEVICES", "0"))

    metadata_path = _write_smoke_metadata(tmp_path)
    text_path = _write_text_prompts(tmp_path)
    cfg = _build_config(tmp_path, metadata_path)

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    train.run_experiment(cfg, key=cfg.encryption_key)
    checkpoint = _latest_checkpoint(cfg.results_dir)

    cfg.evaluate.checkpoint = str(checkpoint)
    evaluate.run_experiment(cfg, key=cfg.encryption_key)

    cfg.inference.checkpoint = str(checkpoint)
    cfg.inference.query.text_file = str(text_path)
    inference.run_experiment(cfg, key=cfg.encryption_key)

    results_dir = Path(cfg.results_dir)
    _assert_nonempty_embeddings(results_dir / "video_embeddings.h5")
    _assert_nonempty_embeddings(results_dir / "text_embeddings.h5")
