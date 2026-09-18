# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the frozen DINOv3 GRIT score."""

import sys
from types import SimpleNamespace
import tarfile
import zipfile

import pandas as pd
from PIL import Image
import pytest
import numpy as np
import torch

from nvidia_tao_pytorch.ssl.dinov3.data_refinement.grit import score_grit_frame
from nvidia_tao_pytorch.ssl.dinov3.data_refinement import grit_pipeline
from nvidia_tao_pytorch.ssl.dinov3.data_refinement.grit_pipeline import (
    _archive_local_order,
    _blockwise_neighbors,
    _faiss_neighbors,
    _ManifestImages,
    _model_patch_size,
    _selected_layers,
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
) -> None:
    from omegaconf import OmegaConf

    from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
    from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel

    spec = OmegaConf.structured(ExperimentConfig())
    spec.model.backbone.teacher_type = "vit_s"
    spec.train.use_custom_attention = False
    backbone = DinoV3PlModel.build_backbone(
        spec.model.backbone,
        spec.train,
        backbone_type="vit_s",
        img_size=512,
    )
    assert backbone.patch_embed.grid_size == (32, 32)
    assert backbone.blocks[0].use_custom_attention is False


def test_real_cpu_extraction_reads_file_tar_zip_and_excludes_registers(
    tmp_path,
) -> None:
    from nvidia_tao_pytorch.ssl.dinov3.model.vit import DinoV3VisionTransformer

    image = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "purple").save(image)
    tar_path = tmp_path / "images.tar"
    zip_path = tmp_path / "images.zip"
    with tarfile.open(tar_path, "w") as archive:
        archive.add(image, arcname="image.png")
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(image, arcname="image.png")
    frame = pd.DataFrame({
        "storage_type": ["file", "tar", "zip"],
        "path": [str(image), str(tar_path), str(zip_path)],
        "member": [None, "image.png", "image.png"],
    })
    dataset = _ManifestImages(frame, views=False, input_size=32, archive_cache_size=1)
    images = torch.stack([dataset[index]["canonical"] for index in range(3)])
    model = DinoV3VisionTransformer(
        img_size=32,
        patch_size=8,
        embed_dim=32,
        depth=4,
        num_heads=4,
        register_tokens=2,
        use_custom_attention=False,
    ).eval()
    with torch.inference_mode():
        global_features, patch_features = _selected_layers(model, images)
    assert global_features.shape == (3, 4, 32)
    assert patch_features.shape == (3, 4, 16, 32)
    assert torch.isfinite(global_features).all()
    assert torch.isfinite(patch_features).all()


def test_extract_grit_scores_cpu_end_to_end(tmp_path, monkeypatch):
    """Run both real image passes, scratch maps, neighbors, and embedding join."""
    from nvidia_tao_pytorch.ssl.dinov3.model.vit import DinoV3VisionTransformer

    rows = []
    for index in range(6):
        path = tmp_path / f"{index}.png"
        pixels = np.random.default_rng(index).integers(0, 256, (40, 40, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        rows.append({"sample_id": str(index), "storage_type": "file", "path": str(path),
                     "task": "tiny", "role": "query" if index < 3 else "reference",
                     "embedding": [float(index), 1.0]})
    manifest = tmp_path / "targets.parquet"
    pd.DataFrame(rows[::-1]).to_parquet(manifest, index=False)
    model = DinoV3VisionTransformer(
        img_size=512, patch_size=16, embed_dim=32, depth=4, num_heads=4,
        register_tokens=2, use_custom_attention=False,
    ).eval()
    model.grit_backbone_type = "tiny_test"
    monkeypatch.setattr(grit_pipeline, "_build_backbone", lambda *_: model)
    result = grit_pipeline.extract_grit_scores({
        "input_parquet": str(manifest), "base_spec": "unused", "checkpoint": "unused",
        "device": "cpu", "neighbor_backend": "torch_exact", "batch_size": 2,
        "workers": 0, "amp": False, "settling_k": 1, "view_ks": [1],
        "work_dir": str(tmp_path),
    })
    assert set(result.sample_id) == {"0", "1", "2"}
    assert np.isfinite(result.select_dtypes(include="number").to_numpy()).all()
    for row in result.itertuples():
        assert list(row.embedding) == [float(row.sample_id), 1.0]
    assert result.attrs["backbone_depth"] == 4
    assert result.attrs["patch_size"] == 16
    assert result.attrs["layer_numbers_one_based"] == [1, 2, 3, 4]
    assert result.attrs["required_scratch_bytes"] > 0
    assert result.attrs["realized_neighbor_device"] == "cpu"


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
            "sample_id": ["query-1", "query-2", "reference-1", "reference-2"],
            "path": [
                "/missing/query-1.jpg",
                "/missing/query-2.jpg",
                "/missing/reference-1.jpg",
                "/missing/reference-2.jpg",
            ],
            "storage_type": ["file"] * 4,
            "task": ["aoi"] * 4,
            "role": ["query", "query", "reference", "reference"],
            "embedding": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
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
                "work_dir": str(tmp_path),
                "device": "cpu",
                "neighbor_device": "cpu",
                "neighbor_backend": "faiss_exact",
                "settling_k": 1,
                "view_ks": [1],
            }
        )


@pytest.mark.parametrize(
    ("config_updates", "error"),
    [
        ({"settling_k": 0}, "settling_k"),
        ({"settling_k": "1"}, "settling_k"),
        ({"view_ks": []}, "view_ks"),
        ({"view_ks": [1.5]}, "view_ks"),
        ({"settling_k": 2}, "more than 2 references"),
        ({"view_ks": [2]}, "more than 2 query samples"),
    ],
)
def test_grit_rejects_invalid_cohorts_before_backbone_inference(
    tmp_path, monkeypatch: pytest.MonkeyPatch, config_updates, error,
) -> None:
    manifest = tmp_path / "targets.parquet"
    pd.DataFrame({
        "sample_id": ["query-1", "query-2", "reference-1", "reference-2"],
        "path": [f"/missing/{index}.jpg" for index in range(4)],
        "storage_type": ["file"] * 4,
        "task": ["aoi"] * 4,
        "role": ["query", "query", "reference", "reference"],
    }).to_parquet(manifest, index=False)
    monkeypatch.setattr(
        grit_pipeline,
        "_build_backbone",
        lambda *_args: pytest.fail("backbone built before cohort validation"),
    )
    config = {
        "input_parquet": str(manifest),
        "base_spec": "unused.yaml",
        "checkpoint": "unused.pth",
        "device": "cpu",
        "neighbor_device": "cpu",
        "settling_k": 1,
        "view_ks": [1],
    }
    config.update(config_updates)
    with pytest.raises(ValueError, match=error):
        grit_pipeline.extract_grit_scores(config)


@pytest.mark.parametrize(
    ("config_updates", "error"),
    [
        ({"work_dir": None}, "requires work_dir"),
        ({"work_dir": "$DEFT_UNRESOLVED/scratch"}, "unresolved variable"),
        ({"scratch_headroom_fraction": -0.1}, "scratch_headroom_fraction"),
        ({"scratch_headroom_fraction": float("nan")}, "scratch_headroom_fraction"),
        ({"scratch_headroom_fraction": float("inf")}, "scratch_headroom_fraction"),
        ({"scratch_headroom_fraction": float("-inf")}, "scratch_headroom_fraction"),
        ({"batch_size": 0}, "batch_size"),
        ({"workers": -1}, "workers"),
        ({"neighbor_block_rows": 0}, "neighbor_block_rows"),
        ({"device": "bogus"}, "device"),
        ({"device": "cuda:999"}, "device"),
        ({"neighbor_device": "bogus"}, "neighbor_device"),
        ({"neighbor_device": "cuda:999"}, "neighbor_device"),
    ],
)
def test_grit_rejects_invalid_runtime_settings_before_backbone_inference(
    tmp_path, monkeypatch: pytest.MonkeyPatch, config_updates, error,
) -> None:
    manifest = tmp_path / "targets.parquet"
    pd.DataFrame({
        "sample_id": ["query-1", "query-2", "reference-1", "reference-2"],
        "path": [f"/missing/{index}.jpg" for index in range(4)],
        "storage_type": ["file"] * 4,
        "task": ["aoi"] * 4,
        "role": ["query", "query", "reference", "reference"],
    }).to_parquet(manifest, index=False)
    monkeypatch.delenv("TAO_LOCAL_SCRATCH", raising=False)
    monkeypatch.delenv("DEFT_UNRESOLVED", raising=False)
    monkeypatch.setattr(
        grit_pipeline,
        "_build_backbone",
        lambda *_args: pytest.fail("backbone built before runtime validation"),
    )
    config = {
        "input_parquet": str(manifest),
        "base_spec": "unused.yaml",
        "checkpoint": "unused.pth",
        "work_dir": str(tmp_path),
        "device": "cpu",
        "neighbor_device": "cpu",
        "settling_k": 1,
        "view_ks": [1],
    }
    config.update(config_updates)
    with pytest.raises(ValueError, match=error):
        grit_pipeline.extract_grit_scores(config)


@pytest.mark.parametrize("invalid_roles", [
    ["query", "typo"],
    ["query", None],
])
def test_grit_rejects_unknown_or_null_roles_before_inference(
    tmp_path, monkeypatch: pytest.MonkeyPatch, invalid_roles
) -> None:
    manifest = tmp_path / "invalid-roles.parquet"
    pd.DataFrame(
        {
            "sample_id": ["query", "other"],
            "path": ["/missing/query.jpg", "/missing/other.jpg"],
            "storage_type": ["file", "file"],
            "task": ["aoi", "aoi"],
            "role": invalid_roles,
        }
    ).to_parquet(manifest, index=False)
    monkeypatch.setattr(
        grit_pipeline,
        "_build_backbone",
        lambda *_args: pytest.fail("backbone built before role validation"),
    )
    with pytest.raises(ValueError, match="role"):
        grit_pipeline.extract_grit_scores(
            {
                "input_parquet": str(manifest),
                "base_spec": "unused.yaml",
                "checkpoint": "unused.pth",
            }
        )


def test_grit_rejects_sample_reused_across_domains() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["same", "same"],
            "task": ["domain-a", "domain-b"],
            "global_consensus": [0.1, 0.2],
            "dense_consensus": [0.2, 0.3],
        }
    )
    with pytest.raises(ValueError, match="globally unique"):
        score_grit_frame(frame)


def test_checkpoint_conditioned_manifest_read_disables_prebuffer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "incomplete.parquet"
    pd.DataFrame({"sample_id": ["query"]}).to_parquet(manifest, index=False)
    calls = []
    real_read_parquet = pd.read_parquet

    def observed_read_parquet(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_read_parquet(*args, **kwargs)

    monkeypatch.setattr(grit_pipeline.pd, "read_parquet", observed_read_parquet)
    with pytest.raises(ValueError, match="missing columns"):
        grit_pipeline.extract_grit_scores({"input_parquet": str(manifest)})
    assert calls
    assert all(options.get("pre_buffer") is False for options in calls)


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


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("sample_id", "   "),
        ("sample_id", None),
        ("task", ""),
        ("task", None),
        ("path", " "),
        ("path", None),
        ("storage_type", ""),
        ("role", " "),
    ],
)
def test_grit_rejects_empty_identity_domain_and_locator_before_inference(
    tmp_path, monkeypatch: pytest.MonkeyPatch, field, invalid,
) -> None:
    rows = {
        "sample_id": ["query", "reference"],
        "path": ["/missing/query.jpg", "/missing/reference.jpg"],
        "storage_type": ["file", "file"],
        "task": ["aoi", "aoi"],
        "role": ["query", "reference"],
    }
    rows[field][1] = invalid
    manifest = tmp_path / f"invalid-{field}.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    monkeypatch.setattr(
        grit_pipeline,
        "_build_backbone",
        lambda *_args: pytest.fail("backbone built before manifest validation"),
    )
    with pytest.raises(ValueError, match=field):
        grit_pipeline.extract_grit_scores({
            "input_parquet": str(manifest),
            "base_spec": "unused.yaml",
            "checkpoint": "unused.pth",
        })


def test_grit_rejects_empty_archive_member_before_inference(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "invalid-member.parquet"
    pd.DataFrame({
        "sample_id": ["query", "reference"],
        "path": ["images.tar", "images.tar"],
        "storage_type": ["tar", "tar"],
        "member": ["query.jpg", "  "],
        "task": ["aoi", "aoi"],
        "role": ["query", "reference"],
    }).to_parquet(manifest, index=False)
    monkeypatch.setattr(
        grit_pipeline,
        "_build_backbone",
        lambda *_args: pytest.fail("backbone built before member validation"),
    )
    with pytest.raises(ValueError, match="member"):
        grit_pipeline.extract_grit_scores({
            "input_parquet": str(manifest),
            "base_spec": "unused.yaml",
            "checkpoint": "unused.pth",
        })
