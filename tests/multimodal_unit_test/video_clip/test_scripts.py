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

"""VideoCLIP scripts unit tests (CPU only, no model/GPU/weights).

Covers the pure helpers the train/evaluate/inference/export entrypoints rely
on: query loading/dedup, the embeddings cache gate, the HDF5 embeddings
round-trip + retrieval ranking, and the export encoder-type / num-frames
helpers.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from nvidia_tao_pytorch.multimodal.video_clip.model import pl_video_clip_model
from nvidia_tao_pytorch.multimodal.video_clip.scripts import export as export_module
from nvidia_tao_pytorch.multimodal.video_clip.scripts.inference import (
    _dedup_preserve_order,
    _embeddings_cache_ok,
    load_text_file,
)
from nvidia_tao_pytorch.multimodal.video_clip.scripts.export import (
    VALID_ENCODER_TYPES,
    _get_video_export_num_frames,
    run_export,
)
from nvidia_tao_pytorch.multimodal.video_clip.utils.embedding_io import (
    read_embeddings_h5,
    text_to_video_search,
    write_embeddings_h5,
)
from nvidia_tao_pytorch.multimodal.video_clip.utils.utils import (
    load_model_from_checkpoint,
    load_partial_pretrained_weights,
    validate_peft_state_dict,
)


@pytest.mark.multimodal_unit
class TestInferenceQueryHelpers:
    """Inline-query loading + dedup used by inference.query."""

    def test_load_text_file_strips_and_skips_blanks(self, tmp_path):
        p = tmp_path / "queries.txt"
        p.write_text("a person walking\n\n   a parked car  \n\n", encoding="utf-8")
        assert load_text_file(str(p)) == ["a person walking", "a parked car"]

    def test_dedup_preserve_order(self):
        assert _dedup_preserve_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


@pytest.mark.multimodal_unit
class TestEmbeddingsCacheGate:
    """inference._embeddings_cache_ok decides reuse-vs-regenerate."""

    def test_overwrite_forces_regenerate(self, tmp_path):
        p = tmp_path / "e.h5"
        write_embeddings_h5(str(p), ["v1"], np.zeros((1, 4), np.float32), "video", {})
        assert _embeddings_cache_ok(str(p), {}, overwrite=True) is False

    def test_missing_file_regenerates(self, tmp_path):
        assert _embeddings_cache_ok(
            str(tmp_path / "absent.h5"), {}, overwrite=False
        ) is False


@pytest.mark.multimodal_unit
class TestCheckpointLoading:
    """Checkpoint prefix normalization and restoration diagnostics."""

    def test_tao_model_prefix_loads_into_inner_model(self, tmp_path):
        source = nn.Linear(2, 2)
        target = nn.Linear(2, 2)
        checkpoint = tmp_path / "tao.pth"
        torch.save({
            "state_dict": {
                f"model.{key}": value
                for key, value in source.state_dict().items()
            }
        }, checkpoint)

        load_partial_pretrained_weights(
            target,
            str(checkpoint),
            prefixes=("module.", "model."),
        )

        for key, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value)

    def test_partial_load_reports_retained_and_ignored_layers(
        self, tmp_path, caplog
    ):
        target = nn.Linear(2, 2)
        checkpoint = tmp_path / "partial.pth"
        torch.save({
            "state_dict": {
                "model.weight": torch.ones(3, 2),
                "model.bias": torch.ones(2),
                "model.extra": torch.ones(1),
            }
        }, checkpoint)

        load_partial_pretrained_weights(
            target,
            str(checkpoint),
            prefixes=("model.",),
        )

        assert "Layers not loaded from checkpoint" in caplog.text
        assert "weight" in caplog.text
        assert "Layers skipped for shape mismatch" in caplog.text
        assert "Checkpoint-only layers ignored" in caplog.text
        assert "model.extra" in caplog.text

    def test_zero_compatible_tensors_raises(self, tmp_path):
        checkpoint = tmp_path / "wrong.pth"
        torch.save({"state_dict": {"other": torch.ones(1)}}, checkpoint)

        with pytest.raises(ValueError, match="No compatible tensors"):
            load_partial_pretrained_weights(
                nn.Linear(2, 2), str(checkpoint)
            )

    @pytest.mark.parametrize(
        ("state_dict", "peft_enabled"),
        [
            ({"model.layer.weight": torch.ones(1)}, False),
            ({"model.layer.lora_A": torch.ones(1)}, True),
        ],
    )
    def test_matching_peft_state_dict_does_not_warn(
        self, state_dict, peft_enabled, caplog
    ):
        experiment = SimpleNamespace(
            peft=SimpleNamespace(enabled=peft_enabled)
        )

        validate_peft_state_dict(state_dict, experiment)

        assert not caplog.text

    @pytest.mark.parametrize(
        ("state_dict", "peft_enabled", "message"),
        [
            (
                {"model.layer.lora_A": torch.ones(1)},
                False,
                "contains LoRA adapter weights",
            ),
            (
                {"model.layer.weight": torch.ones(1)},
                True,
                "has no LoRA adapter weights",
            ),
        ],
    )
    def test_mismatched_peft_state_dict_warns(
        self, state_dict, peft_enabled, message, caplog
    ):
        experiment = SimpleNamespace(
            peft=SimpleNamespace(enabled=peft_enabled)
        )

        validate_peft_state_dict(
            state_dict, experiment, checkpoint_label="unit checkpoint"
        )

        assert message in caplog.text
        assert "unit checkpoint" in caplog.text

    def test_lightning_hook_validates_loaded_state_dict(self, monkeypatch):
        state_dict = {"model.layer.lora_A": torch.ones(1)}
        experiment = SimpleNamespace(peft=SimpleNamespace(enabled=True))
        observed = {}

        def _super_hook(_model, actual_checkpoint):
            observed["super_checkpoint"] = actual_checkpoint

        def _validate(actual_state, actual_config, checkpoint_label):
            observed["state"] = actual_state
            observed["config"] = actual_config
            observed["label"] = checkpoint_label

        monkeypatch.setattr(
            pl_video_clip_model.TAOLightningModule,
            "on_load_checkpoint",
            _super_hook,
        )
        monkeypatch.setattr(
            pl_video_clip_model, "validate_peft_state_dict", _validate
        )
        model = pl_video_clip_model.VideoCLIPPlModel.__new__(
            pl_video_clip_model.VideoCLIPPlModel
        )
        nn.Module.__init__(model)
        model.experiment_spec = experiment
        checkpoint = {"state_dict": state_dict}

        pl_video_clip_model.VideoCLIPPlModel.on_load_checkpoint(
            model, checkpoint
        )

        assert observed == {
            "super_checkpoint": checkpoint,
            "state": state_dict,
            "config": experiment,
            "label": "Lightning checkpoint",
        }

    def test_task_checkpoint_uses_isolated_architecture_only_config(
        self, tmp_path, monkeypatch
    ):
        checkpoint = tmp_path / "task.pth"
        torch.save({"state_dict": {"weight": torch.ones(1)}}, checkpoint)
        model_cfg = SimpleNamespace(
            internvideo2clip_hf_id="repo",
            vision_encoder="vision",
            text_encoder="text",
            clip_head="head",
            pretrained_ckpt="base.pth",
        )
        experiment = SimpleNamespace(model=model_cfg, peft=None)

        class _Model:
            received = None

            @classmethod
            def load_from_checkpoint(cls, path, map_location, experiment_spec):
                del path, map_location
                cls.received = experiment_spec
                return cls()

        monkeypatch.setattr(
            torch,
            "load",
            lambda *args, **kwargs: pytest.fail(
                "load_model_from_checkpoint performed a preflight checkpoint read"
            ),
        )
        load_model_from_checkpoint(str(checkpoint), experiment, _Model)

        assert all(
            getattr(_Model.received.model, field) is None
            for field in (
                "internvideo2clip_hf_id",
                "vision_encoder",
                "text_encoder",
                "clip_head",
                "pretrained_ckpt",
            )
        )
        assert experiment.model.internvideo2clip_hf_id == "repo"
        assert experiment.model.pretrained_ckpt == "base.pth"

    def test_null_task_checkpoint_raises(self):
        with pytest.raises(ValueError, match="trained checkpoint is required"):
            load_model_from_checkpoint(None, SimpleNamespace(), object)


@pytest.mark.multimodal_unit
class TestEmbeddingsIO:
    """HDF5 embeddings round-trip + retrieval ranking (pure numpy)."""

    def test_write_read_roundtrip(self, tmp_path):
        p = tmp_path / "vid.h5"
        ids = ["v0", "v1", "v2"]
        emb = np.arange(12, dtype=np.float32).reshape(3, 4)
        write_embeddings_h5(str(p), ids, emb, "video", {"model_type": "internvideo2-clip-l14"})
        got_ids, got_emb, attrs = read_embeddings_h5(str(p))
        assert got_ids == ids
        np.testing.assert_allclose(got_emb, emb)
        assert attrs.get("embedding_type") == "video"

    def test_cosine_search_ranks_nearest_first(self):
        video_ids = ["v0", "v1", "v2"]
        video_emb = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
        text_emb = np.array([[1.0, 0.05]], dtype=np.float32)  # closest to v0
        results, scores = text_to_video_search(
            video_ids, video_emb, ["q0"], text_emb,
            metric="cosine", normalize=True, top_k=3,
        )
        assert len(results) == 1
        assert scores.shape == (1, 3)
        assert video_ids[int(np.argmax(scores[0]))] == "v0"


@pytest.mark.multimodal_unit
class TestExportHelpers:
    """export.py encoder-type + video num-frames helpers."""

    def test_valid_encoder_types(self):
        assert VALID_ENCODER_TYPES == {"combined", "separate"}

    def test_num_frames_requires_internvideo2_marker(self):
        class _M:
            pass

        m = _M()
        m.num_frames = 8  # no is_internvideo2 -> not a video export
        assert _get_video_export_num_frames(m) is None

        m.is_internvideo2 = True
        m.num_frames = 8
        assert _get_video_export_num_frames(m) == 8

        m.num_frames = 0  # non-positive -> None
        assert _get_video_export_num_frames(m) is None

    def test_export_requires_trained_checkpoint(self):
        """A null checkpoint fails before export setup begins."""
        experiment = SimpleNamespace(
            export=SimpleNamespace(checkpoint=None),
        )

        with pytest.raises(ValueError, match="trained checkpoint is required"):
            run_export(experiment)

    @pytest.mark.parametrize(
        ("encoder_type", "existing_suffix"),
        [
            ("combined", ".onnx"),
            ("separate", "_vision.onnx"),
            ("separate", "_text.onnx"),
        ],
    )
    def test_existing_output_fails_before_model_load(
        self,
        encoder_type,
        existing_suffix,
        tmp_path,
        monkeypatch,
    ):
        """Existing combined or separate outputs are rejected immediately."""
        output_file = tmp_path / "model.onnx"
        existing_file = tmp_path / f"model{existing_suffix}"
        existing_file.touch()
        experiment = SimpleNamespace(
            encryption_key="",
            export=SimpleNamespace(
                checkpoint="model.ckpt",
                gpu_id=0,
                on_cpu=True,
                onnx_file=str(output_file),
                encoder_type=encoder_type,
            ),
        )
        monkeypatch.setattr(
            export_module.VideoCLIPPlModel,
            "load_from_checkpoint",
            lambda *args, **kwargs: pytest.fail("model was loaded"),
        )

        with pytest.raises(FileExistsError, match=str(existing_file)):
            run_export(experiment)

    @pytest.mark.parametrize(
        ("encoder_type", "partial_suffixes"),
        [
            (
                "combined",
                (".onnx", ".onnx.data", "_tokenizer"),
            ),
            (
                "separate",
                (
                    "_vision.onnx",
                    "_vision_weights.bin",
                    "_text.onnx",
                    "_config.yaml",
                    "_tokenizer",
                ),
            ),
        ],
    )
    def test_failed_export_removes_only_new_expected_artifacts(
        self,
        encoder_type,
        partial_suffixes,
        tmp_path,
        monkeypatch,
    ):
        """A failed export is retryable without sweeping same-prefix files."""
        output_file = tmp_path / "model.onnx"
        unrelated_file = tmp_path / "model.notes"
        unrelated_file.write_text("keep", encoding="utf-8")
        experiment = SimpleNamespace(
            export=SimpleNamespace(
                checkpoint="model.ckpt",
                onnx_file=str(output_file),
                encoder_type=encoder_type,
            ),
        )

        def _fail_after_writes(_experiment):
            for suffix in partial_suffixes:
                artifact = tmp_path / f"model{suffix}"
                if suffix == "_tokenizer":
                    artifact.mkdir()
                    (artifact / "tokenizer.json").write_text(
                        "partial", encoding="utf-8"
                    )
                else:
                    artifact.write_text("partial", encoding="utf-8")
            raise RuntimeError("parity mismatch")

        monkeypatch.setattr(export_module, "_run_export_impl", _fail_after_writes)

        with pytest.raises(RuntimeError, match="parity mismatch"):
            run_export(experiment)

        assert unrelated_file.read_text(encoding="utf-8") == "keep"
        for suffix in partial_suffixes:
            assert not (tmp_path / f"model{suffix}").exists()

    def test_failed_export_preserves_preexisting_expected_artifact(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Rollback never removes an expected artifact it did not create."""
        output_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model_config.yaml"
        config_file.write_text("existing", encoding="utf-8")
        experiment = SimpleNamespace(
            export=SimpleNamespace(
                checkpoint="model.ckpt",
                onnx_file=str(output_file),
                encoder_type="combined",
            ),
        )

        def _fail_after_write(_experiment):
            output_file.write_text("partial", encoding="utf-8")
            raise RuntimeError("export failed")

        monkeypatch.setattr(export_module, "_run_export_impl", _fail_after_write)

        with pytest.raises(RuntimeError, match="export failed"):
            run_export(experiment)

        assert not output_file.exists()
        assert config_file.read_text(encoding="utf-8") == "existing"
