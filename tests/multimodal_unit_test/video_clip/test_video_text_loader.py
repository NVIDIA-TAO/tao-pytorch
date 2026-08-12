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

"""Tests for video-text metadata loading and batching."""

import json
import logging as py_logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from nvidia_tao_pytorch.multimodal.video_clip.dataloader import (
    pl_video_clip_data_module,
    video_text_loader,
)
from nvidia_tao_pytorch.multimodal.video_clip.dataloader.pl_video_clip_data_module import (
    VideoCLIPDataModule,
)
from nvidia_tao_pytorch.multimodal.video_clip.dataloader.video_text_loader import (
    VideoTextDataset,
    _load_with_ffmpeg,
    get_video_text_dataloader,
    load_video_frames,
    load_video_text_entries,
)


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _tokenizer(text):
    return [{
        "input_ids": torch.tensor([len(text), 1], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1], dtype=torch.long),
    }]


def _frame_loader(video_path, num_frames, start_time_sec, end_time_sec,
                  start_frame, end_frame):
    del video_path, start_time_sec, end_time_sec, start_frame, end_frame
    frames = []
    for idx in range(num_frames):
        array = np.full((4, 4, 3), idx, dtype=np.uint8)
        frames.append(Image.fromarray(array))
    return frames


def _transform(frame):
    array = np.asarray(frame, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _vadr1_records():
    return [{
        "split": "train",
        "dataset": "NEW",
        "video_id": "vid1",
        "anomaly_type": "Animals Obstructing Traffic",
        "total_frames": 300,
        "video_path": "/orig_dataset/Vad-Reasoning-SFT/NEW/vid1.mp4",
        "what": "an elk blocks the road",
        "where": "center",
        "why": "traffic is obstructed",
        "how": "drivers must stop",
        "chunks": [
            {
                "chunk_index": 0,
                "start_time_sec": 0.0,
                "end_time_sec": 5.0,
                "start_frame": 0,
                "end_frame": 150,
                "is_anomaly": True,
            },
            {
                "chunk_index": 1,
                "start_time_sec": 5.0,
                "end_time_sec": 10.0,
                "start_frame": 150,
                "end_frame": 300,
                "is_anomaly": False,
            },
        ],
    }]


@pytest.mark.multimodal_unit
class TestVideoTextEntries:
    """Test metadata normalization."""

    def test_vadr1_chunks_flatten_and_remap_paths(self, tmp_path):
        """Vad-R1 records should flatten nested chunks."""
        metadata = _write_json(tmp_path / "vadr1.json", _vadr1_records())

        entries = load_video_text_entries(
            metadata=str(metadata),
            metadata_format="vadr1_chunks",
            data_root=str(tmp_path),
            path_prefix_mapping={"/orig_dataset/": f"{tmp_path}/"},
            split="train",
            caption_fields=["anomaly_type", "what"],
            idx_mode="category",
        )

        assert len(entries) == 2
        assert entries[0]["caption"] == "Animals Obstructing Traffic"
        assert entries[0]["category"] == "Animals Obstructing Traffic"
        assert entries[1]["caption"] == "Normal"
        assert entries[1]["category"] == "Normal"
        assert entries[0]["video_path"] == str(
            tmp_path / "Vad-Reasoning-SFT/NEW/vid1.mp4"
        )
        assert entries[0]["idx"] != entries[1]["idx"]

    def test_vadr1_video_id_idx_groups_chunks_as_positives(self, tmp_path):
        """idx_mode=video_id should give chunks from one video the same idx."""
        metadata = _write_json(tmp_path / "vadr1.json", _vadr1_records())

        entries = load_video_text_entries(
            metadata=str(metadata),
            metadata_format="vadr1_chunks",
            data_root=str(tmp_path),
            split="train",
            caption_fields=["anomaly_type"],
            idx_mode="video_id",
        )

        assert entries[0]["idx"] == entries[1]["idx"]

    def test_multi_file_metadata_concatenates(self, tmp_path):
        """A list of metadata paths concatenates all files' records."""
        file_a = _write_json(tmp_path / "vadr1_a.json", _vadr1_records())
        records_b = _vadr1_records()
        records_b[0]["video_id"] = "vid2"
        records_b[0]["video_path"] = "/orig_dataset/Vad-Reasoning-SFT/NEW/vid2.mp4"
        file_b = _write_json(tmp_path / "vadr1_b.json", records_b)

        entries = load_video_text_entries(
            metadata=[str(file_a), str(file_b)],
            metadata_format="vadr1_chunks",
            data_root=str(tmp_path),
            path_prefix_mapping={"/orig_dataset/": f"{tmp_path}/"},
            split="train",
            caption_fields=["anomaly_type"],
            idx_mode="sample_id",
        )

        # 2 chunks/file x 2 files = 4 entries; both videos represented.
        assert len(entries) == 4
        assert {e["video_path"] for e in entries} == {
            str(tmp_path / "Vad-Reasoning-SFT/NEW/vid1.mp4"),
            str(tmp_path / "Vad-Reasoning-SFT/NEW/vid2.mp4"),
        }

    def test_multi_file_metadata_rejects_dict_file(self, tmp_path):
        """Dict-style (non-list) metadata cannot be concatenated in a list."""
        dict_file = _write_json(
            tmp_path / "msrvtt.json", {"videos": [], "sentences": []}
        )
        with pytest.raises(ValueError, match="each file to be a JSON array"):
            load_video_text_entries(
                metadata=[str(dict_file)],
                metadata_format="auto",
                data_root=str(tmp_path),
                split="train",
                caption_fields=["anomaly_type"],
                idx_mode="sample_id",
            )

    def test_flat_json_supports_common_video_text_rows(self, tmp_path):
        """Flat search metadata should normalize video/caption fields."""
        metadata = _write_json(tmp_path / "flat.json", [
            {
                "video": "clips/a.mp4",
                "captions": ["first caption", "second caption"],
                "video_id": "a",
                "split": "train",
                "start_sec": 1.0,
                "end_sec": 2.0,
            }
        ])

        entries = load_video_text_entries(
            metadata=str(metadata),
            metadata_format="flat_json",
            data_root=str(tmp_path),
            split="train",
            idx_mode="video_id",
        )

        assert len(entries) == 1
        assert entries[0]["video_path"] == str(tmp_path / "clips/a.mp4")
        assert entries[0]["captions"] == ["first caption", "second caption"]
        assert entries[0]["start_time_sec"] == 1.0
        assert entries[0]["end_time_sec"] == 2.0


@pytest.mark.multimodal_unit
class TestVideoTextDataset:
    """Test dataset and dataloader behavior."""

    def test_dataset_returns_video_text_and_idx(self):
        """Dataset item should match VideoCLIPPlModel internvideo2_vtc batch shape."""
        entries = [{
            "video_path": "/tmp/fake.mp4",
            "captions": ["caption"],
            "caption": "caption",
            "idx": 7,
            "start_time_sec": 0.0,
            "end_time_sec": 5.0,
            "start_frame": 0,
            "end_frame": 10,
            "metadata": {},
        }]
        dataset = VideoTextDataset(
            entries,
            transform=_transform,
            tokenizer=_tokenizer,
            num_frames=3,
            frame_loader=_frame_loader,
        )

        video, text, idx, row_position = dataset[0]

        assert video.shape == (3, 3, 4, 4)
        assert text["input_ids"].shape == (2,)
        assert idx == 7
        assert row_position == 0

    def test_pyav_requests_the_linspace_frame_indices(self, monkeypatch):
        """PyAV backend should decode exactly the sampled indices, in order."""
        requested = []

        def fake_decode(container, stream, rate, frame_indices):
            del container, stream, rate
            requested.append([int(index) for index in frame_indices])
            return {
                int(index): Image.new("RGB", (4, 4))
                for index in frame_indices
            }

        monkeypatch.setattr(
            video_text_loader, "_pyav_decode_indices", fake_decode
        )
        monkeypatch.setattr(
            video_text_loader, "_pyav_stream_rate", lambda stream: 30.0
        )
        monkeypatch.setattr(
            video_text_loader, "_pyav_stream_length", lambda stream, rate: 2
        )

        class FakeStream:
            thread_type = "AUTO"

        class FakeContainer:
            streams = SimpleNamespace(video=[FakeStream()])

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        monkeypatch.setitem(
            sys.modules, "av", SimpleNamespace(open=lambda path: FakeContainer())
        )

        frames = video_text_loader._load_with_pyav(
            "/tmp/fake.mp4",
            num_frames=2,
            start_time_sec=None,
            end_time_sec=None,
            start_frame=0,
            end_frame=2,
        )

        assert requested == [[0, 1]]
        assert len(frames) == 2
        assert all(isinstance(frame, Image.Image) for frame in frames)

    def test_pyav_repeats_indices_for_short_clips(self, monkeypatch):
        """A clip shorter than num_frames pads with the last index, in order."""
        requested = []

        def fake_decode(container, stream, rate, frame_indices):
            del container, stream, rate
            requested.append([int(index) for index in frame_indices])
            return {
                int(index): Image.new("RGB", (4, 4))
                for index in frame_indices
            }

        monkeypatch.setattr(
            video_text_loader, "_pyav_decode_indices", fake_decode
        )
        monkeypatch.setattr(
            video_text_loader, "_pyav_stream_rate", lambda stream: 30.0
        )
        monkeypatch.setattr(
            video_text_loader, "_pyav_stream_length", lambda stream, rate: 2
        )

        class FakeStream:
            thread_type = "AUTO"

        class FakeContainer:
            streams = SimpleNamespace(video=[FakeStream()])

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        monkeypatch.setitem(
            sys.modules, "av", SimpleNamespace(open=lambda path: FakeContainer())
        )

        frames = video_text_loader._load_with_pyav(
            "/tmp/fake.mp4",
            num_frames=4,
            start_time_sec=None,
            end_time_sec=None,
            start_frame=0,
            end_frame=2,
        )

        # _linspace_indices pads with the final index when the clip is short.
        assert requested == [[0, 1, 1, 1]]
        assert len(frames) == 4

    def test_opencv_frame_loader_decodes_video_clip(self, tmp_path):
        """OpenCV fallback should decode real video files in the TAO venv."""
        cv2 = pytest.importorskip("cv2")
        video_path = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            4.0,
            (8, 8),
        )
        if not writer.isOpened():
            pytest.skip("OpenCV mp4v writer is unavailable")
        try:
            for value in range(6):
                frame = np.full((8, 8, 3), value * 20, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

        frames = load_video_frames(
            str(video_path), num_frames=3, start_frame=1, end_frame=5
        )

        assert len(frames) == 3
        assert all(isinstance(frame, Image.Image) for frame in frames)

    def test_ffmpeg_loader_pads_short_decode(self, monkeypatch):
        """ffmpeg fallback should pad when selected tail frames do not decode."""
        frame_bytes = bytes([10, 20, 30] * 4)

        class Result:
            stdout = frame_bytes
            stderr = b""

        def fake_run(*args, **kwargs):
            del args, kwargs
            return Result()

        monkeypatch.setattr(
            video_text_loader,
            "_probe_video_with_ffmpeg",
            lambda video_path: (8, 1.0, 2, 2),
        )
        monkeypatch.setattr("subprocess.run", fake_run)

        frames = _load_with_ffmpeg(
            "/tmp/fake.mp4",
            num_frames=3,
            start_time_sec=None,
            end_time_sec=None,
            start_frame=0,
            end_frame=8,
        )

        assert len(frames) == 3
        assert all(isinstance(frame, Image.Image) for frame in frames)

    def test_dataloader_collates_to_b_t_c_h_w(self, tmp_path):
        """Dataloader should collate video clips and idx tensors."""
        metadata = _write_json(tmp_path / "vadr1.json", _vadr1_records())
        cfg = SimpleNamespace(
            metadata=str(metadata),
            format="vadr1_chunks",
            data_root=str(tmp_path),
            split="train",
            caption_fields=["anomaly_type"],
            caption_mode="first",
            anomaly_only=False,
            path_prefix_mapping={},
            idx_mode="video_id",
            idx_field=None,
            num_frames=2,
        )

        loader = get_video_text_dataloader(
            cfg=cfg,
            transform=_transform,
            tokenizer=_tokenizer,
            batch_size=2,
            num_workers=0,
            shuffle=False,
            mode="val",
            frame_loader=_frame_loader,
        )
        video, text, idx, row_positions = next(iter(loader))

        assert video.shape == (2, 2, 3, 4, 4)
        assert text["input_ids"].shape == (2, 2)
        assert torch.equal(idx, torch.tensor([0, 0]))
        assert torch.equal(row_positions, torch.tensor([0, 1]))


@pytest.mark.multimodal_unit
class TestVideoTextDataModule:
    """Test DataModule routing for video-text train/val configs."""

    def test_video_text_val_does_not_require_custom_datasets(
        self, monkeypatch, caplog
    ):
        """Validation should use val.video_text metadata, not val.datasets."""
        calls = []

        def fake_get_video_text_dataloader(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(dataset=[kwargs["mode"]])

        monkeypatch.setattr(
            pl_video_clip_data_module,
            "get_video_text_dataloader",
            fake_get_video_text_dataloader,
        )
        cfg = SimpleNamespace(
            seed=123,
            pin_memory=False,
            train=SimpleNamespace(
                type="video_text",
                video_text=SimpleNamespace(name="train"),
                batch_size=2,
                num_workers=0,
            ),
            val=SimpleNamespace(
                type="video_text",
                datasets=[],
                video_text=SimpleNamespace(
                    name="val", metadata="val.json"
                ),
                batch_size=2,
                num_workers=0,
            ),
        )
        module = VideoCLIPDataModule(
            dataset_config=cfg,
            tokenizer=_tokenizer,
            resume_step=0,
            preprocess=(_transform, _transform),
            world_size=1,
        )

        with caplog.at_level(py_logging.INFO, logger="TAO Toolkit"):
            module.setup(stage="fit")

        assert [call["mode"] for call in calls] == ["train", "val"]
        assert calls[0]["cfg"].name == "train"
        assert calls[1]["cfg"].name == "val"
        assert calls[0]["shuffle"] is True
        assert calls[1]["shuffle"] is False
        assert module.train_dataloader().dataset == ["train"]
        assert module.val_dataloader().dataset == ["val"]
        assert "Training video-text dataloader: 1 samples" in caplog.text


def test_explode_caption_entries_expands_and_shares_idx():
    """One entry per caption; copies of a chunk share idx and sample_id."""
    entries = [
        {"sample_id": "ds/v0#0", "idx": 0, "caption": "q1",
         "captions": ["q1", "q2", "q3"]},
        {"sample_id": "ds/v1#0", "idx": 1, "caption": "c", "captions": ["c"]},
    ]
    out = video_text_loader._explode_caption_entries(entries)
    # (a) length == sum of per-entry caption counts
    assert len(out) == 4
    src0 = [e for e in out if e["sample_id"] == "ds/v0#0"]
    # (b) all copies of one source share idx (multi-positive) and sample_id
    assert len(src0) == 3
    assert all(e["idx"] == 0 for e in src0)
    # (c) each output has exactly one caption, with caption == captions[0]
    for e in out:
        assert len(e["captions"]) == 1
        assert e["caption"] == e["captions"][0]
    assert sorted(e["caption"] for e in src0) == ["q1", "q2", "q3"]
    # (d) single-caption entry passes through unchanged
    src1 = [e for e in out if e["sample_id"] == "ds/v1#0"]
    assert len(src1) == 1
    assert src1[0]["captions"] == ["c"] and src1[0]["caption"] == "c"


def test_explode_caption_entries_does_not_mutate_source():
    entries = [{"sample_id": "ds/v0#0", "idx": 0, "caption": "q1",
                "captions": ["q1", "q2"]}]
    video_text_loader._explode_caption_entries(entries)
    assert entries[0]["captions"] == ["q1", "q2"]


def test_caption_candidate_groups_preserve_field_structure():
    """Groups keep per-field lists; flat candidates concatenate them in order."""
    chunk = {
        "queries": ["q1", "q2"],     # list field -> multi-caption group
        "dense_caption": "dc",        # string field -> single-caption group
        "scene_caption": "",          # empty -> dropped
    }
    fields = ["queries", "dense_caption", "scene_caption"]
    groups = video_text_loader._caption_candidate_groups(
        {}, chunk, fields, is_anomaly=True
    )
    assert groups == [["q1", "q2"], ["dc"]]
    # Flat view concatenates the groups in field order.
    flat = video_text_loader._caption_candidates(
        {}, chunk, fields, is_anomaly=True
    )
    assert flat == ["q1", "q2", "dc"]


def test_apply_caption_mode_dispatch():
    """first=unchanged, all=per-caption, one_per_field=per-field."""
    entries = [{
        "sample_id": "ds/v0#0", "idx": 0, "caption": "q1",
        "captions": ["q1", "q2", "dc"],
        "caption_groups": [["q1", "q2"], ["dc"]],
    }]
    first = video_text_loader._apply_caption_mode(list(entries), "first")
    assert len(first) == 1
    allm = video_text_loader._apply_caption_mode(list(entries), "all")
    assert len(allm) == 3 and all(len(e["captions"]) == 1 for e in allm)
    opf = video_text_loader._apply_caption_mode(list(entries), "one_per_field")
    # one entry per field group, each keeping that field's full caption list
    assert len(opf) == 2
    assert sorted(tuple(e["captions"]) for e in opf) == [("dc",), ("q1", "q2")]
    assert all(e["idx"] == 0 and e["sample_id"] == "ds/v0#0" for e in opf)
    assert all(e["caption"] == e["captions"][0] for e in opf)


def test_explode_by_field_falls_back_to_flat_when_no_groups():
    """Entries without caption_groups collapse to a single group."""
    entries = [{"sample_id": "ds/v0#0", "idx": 0, "caption": "c",
                "captions": ["c1", "c2"]}]
    out = video_text_loader._explode_caption_entries_by_field(entries)
    assert len(out) == 1 and out[0]["captions"] == ["c1", "c2"]
