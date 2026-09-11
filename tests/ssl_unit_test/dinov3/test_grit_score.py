# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the frozen DINOv3 GRIT score."""

import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import numpy as np

from nvidia_tao_pytorch.ssl.dinov3.data_refinement.grit import score_grit_frame
from nvidia_tao_pytorch.ssl.dinov3.data_refinement import grit_pipeline
from nvidia_tao_pytorch.ssl.dinov3.data_refinement.grit_pipeline import (
    _archive_local_order,
    _blockwise_neighbors,
    _build_backbone,
    _faiss_neighbors,
    _model_patch_size,
    derive_grit_observations,
    relative_layer_numbers,
)


@pytest.mark.parametrize("column", ["global_consensus", "dense_consensus"])
@pytest.mark.parametrize("invalid", [float("inf"), float("-inf"), "invalid", 1j])
def test_grit_rejects_invalid_consensus_channels(column, invalid):
    """Invalid observations must not become plausible acquisition ranks."""
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "task": ["domain", "domain"],
            "global_consensus": [0.2, 0.8],
            "dense_consensus": [0.8, 0.2],
        }
    )
    frame[column] = [invalid, invalid]
    with pytest.raises(ValueError, match="finite real numbers"):
        score_grit_frame(frame)


def test_grit_backbone_reads_custom_attention_from_train_config(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("omegaconf")
    captured = {}

    class FakeBackbone:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "nvidia_tao_pytorch.ssl.dinov3.model.pl_model",
        SimpleNamespace(
            DinoV3PlModel=SimpleNamespace(
                load_backbone_weights=lambda _model, _checkpoint: None
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "nvidia_tao_pytorch.ssl.dinov3.model.vit",
        SimpleNamespace(
            DinoV3VisionTransformer=FakeBackbone,
            SwiGLUFusedFull=object,
        ),
    )
    spec = tmp_path / "dinov3.yaml"
    spec.write_text(
        "model:\n  backbone:\n    teacher_type: vit_s\n"
        "train:\n  use_custom_attention: false\n",
        encoding="utf-8",
    )

    _build_backbone(spec, tmp_path / "checkpoint.pth")

    assert captured["use_custom_attention"] is False


def test_grit_extraction_order_groups_archive_members() -> None:
    frame = pd.DataFrame(
        {
            "storage_type": ["tar", "tar", "tar", "file"],
            "path": ["b.tar", "a.tar", "b.tar", "/z.jpg"],
            "member": ["2.jpg", "1.jpg", "1.jpg", None],
        }
    )
    assert _archive_local_order(frame) == [3, 1, 2, 0]


def test_grit_resolves_tuple_patch_size_from_patch_embed() -> None:
    model = SimpleNamespace(patch_embed=SimpleNamespace(patch_size=(16, 16)))
    assert _model_patch_size(model) == 16


def test_cuda_faiss_failure_does_not_fall_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_faiss = SimpleNamespace(
        normalize_L2=lambda _values: None,
        IndexFlatIP=lambda _dimensions: object(),
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    values = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="GPU-enabled FAISS"):
        _faiss_neighbors(
            values,
            values,
            1,
            exclude_identity=False,
            device="cuda",
        )


def test_faiss_capability_is_checked_before_backbone_inference(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "targets.parquet"
    pd.DataFrame(
        {
            "sample_id": ["query", "reference"],
            "path": ["/missing/query.jpg", "/missing/reference.jpg"],
            "storage_type": ["file", "file"],
            "task": ["aoi", "aoi"],
            "role": ["query", "reference"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        }
    ).to_parquet(manifest, index=False)

    def fail_preflight(_device):
        raise RuntimeError("faiss preflight failed")

    monkeypatch.setattr(grit_pipeline, "_preflight_faiss", fail_preflight)
    monkeypatch.setattr(
        grit_pipeline,
        "_build_backbone",
        lambda *_args: pytest.fail("backbone built before FAISS preflight"),
    )
    with pytest.raises(RuntimeError, match="faiss preflight failed"):
        grit_pipeline.extract_grit_scores(
            {
                "input_parquet": str(manifest),
                "base_spec": "unused.yaml",
                "checkpoint": "unused.pth",
                "neighbor_backend": "faiss_exact",
            }
        )


def test_blockwise_neighbors_match_full_matrix() -> None:
    values = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    expected = values @ values.T
    np.fill_diagonal(expected, -np.inf)
    expected_indices = np.argsort(-expected, axis=1)[:, :2]
    observed = _blockwise_neighbors(
        values, values, 2, exclude_identity=True, block_rows=2
    )
    np.testing.assert_array_equal(observed, expected_indices)


def test_grit_uses_within_domain_midrank_and_hard_or() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "task": ["small", "small", "large", "large"],
            "global_consensus": [0.0, 1.0, 10.0, 20.0],
            "dense_consensus": [1.0, 0.0, 30.0, 2.0],
        }
    )
    result = score_grit_frame(frame).set_index("sample_id")
    assert result.loc["a", "grit_score"] == pytest.approx(0.75)
    assert result.loc["b", "grit_score"] == pytest.approx(0.75)
    assert result.loc["c", "grit_score"] == pytest.approx(0.75)
    assert result.loc["d", "grit_score"] == pytest.approx(0.75)
    assert set(result["grit_formula_version"]) == {"grit_v1"}


def test_grit_rejects_duplicate_identities() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "a"],
            "task": ["x", "x"],
            "global_consensus": [0.1, 0.2],
            "dense_consensus": [0.2, 0.3],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        score_grit_frame(frame)


def test_grit_observations_are_checkpoint_feature_conditioned() -> None:
    rng = np.random.default_rng(7)
    manifest = pd.DataFrame(
        {
            "sample_id": ["q0", "q1", "q2", "r0", "r1"],
            "task": ["aoi"] * 5,
            "role": ["query"] * 3 + ["reference"] * 2,
            "embedding": [rng.normal(size=3).tolist() for _ in range(5)],
        }
    )

    def normalized(shape: tuple[int, ...]) -> np.ndarray:
        values = rng.normal(size=shape).astype(np.float32)
        return values / np.linalg.norm(values, axis=-1, keepdims=True)

    result = derive_grit_observations(
        manifest,
        global_layers=normalized((5, 4, 6)),
        patch_layers=normalized((5, 4, 16, 6)),
        query_views={
            name: normalized((3, 6))
            for name in ("canonical", "view0", "view1")
        },
        patch_retrieval=np.array([0.1, 0.3, 0.2]),
        settling_k=1,
        view_ks=(1,),
    )
    assert list(result["sample_id"]) == ["q0", "q1", "q2"]
    assert result["grit_score"].between(0.0, 1.0).all()
    assert np.isfinite(
        result[
            [
                "global_depth_settling",
                "global_view_instability",
                "dense_depth_settling",
                "dense_view_retrieval",
            ]
        ].to_numpy()
    ).all()
    assert "embedding" in result


def test_grit_accepts_streamed_dense_settling() -> None:
    rng = np.random.default_rng(11)
    manifest = pd.DataFrame(
        {
            "sample_id": ["q0", "q1", "q2", "r0", "r1"],
            "task": ["aoi"] * 5,
            "role": ["query"] * 3 + ["reference"] * 2,
        }
    )

    def normalized(shape: tuple[int, ...]) -> np.ndarray:
        values = rng.normal(size=shape).astype(np.float32)
        return values / np.linalg.norm(values, axis=-1, keepdims=True)

    patch_layers = normalized((5, 4, 16, 6))
    common = {
        "global_layers": normalized((5, 4, 6)),
        "query_views": {
            name: normalized((3, 6))
            for name in ("canonical", "view0", "view1")
        },
        "patch_retrieval": np.array([0.1, 0.3, 0.2]),
        "settling_k": 1,
        "view_ks": (1,),
    }
    expected = derive_grit_observations(
        manifest, patch_layers=patch_layers, **common
    )
    streamed = derive_grit_observations(
        manifest,
        dense_settling=np.concatenate(
            [
                expected["dense_depth_settling"].to_numpy(),
                np.zeros(2, dtype=np.float32),
            ]
        ),
        **common,
    )
    np.testing.assert_allclose(
        streamed["grit_score"], expected["grit_score"], rtol=0.0, atol=0.0
    )


def test_view_neighbors_are_computed_once_at_max_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(17)
    manifest = pd.DataFrame(
        {
            "sample_id": ["q0", "q1", "q2", "r0", "r1", "r2"],
            "task": ["aoi"] * 6,
            "role": ["query"] * 3 + ["reference"] * 3,
        }
    )
    calls = []
    original = grit_pipeline._within_cohort_neighbors

    def counted(values, k, **kwargs):
        calls.append(k)
        return original(values, k, **kwargs)

    monkeypatch.setattr(grit_pipeline, "_within_cohort_neighbors", counted)
    values = rng.normal(size=(6, 4, 4)).astype(np.float32)
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    views = {
        name: rng.normal(size=(3, 4)).astype(np.float32)
        for name in ("canonical", "view0", "view1")
    }
    derive_grit_observations(
        manifest,
        global_layers=values,
        dense_settling=np.arange(6, dtype=np.float32),
        query_views=views,
        patch_retrieval=np.arange(3, dtype=np.float32),
        settling_k=1,
        view_ks=(1, 2),
    )
    assert calls == [2, 2, 2]


@pytest.mark.parametrize(
    ("variant", "depth", "expected"),
    [
        ("vit_s", 12, (3, 6, 9, 12)),
        ("vit_s_plus", 12, (3, 6, 9, 12)),
        ("vit_b", 12, (3, 6, 9, 12)),
        ("vit_l", 24, (6, 12, 18, 24)),
        ("vit_h_plus", 32, (8, 16, 24, 32)),
        ("vit_7b", 40, (10, 20, 30, 40)),
    ],
)
def test_grit_uses_relative_depth_across_every_dinov3_variant(
    variant: str, depth: int, expected: tuple[int, ...]
) -> None:
    assert variant.startswith("vit_")
    assert relative_layer_numbers(depth) == expected
