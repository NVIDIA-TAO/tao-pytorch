# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the NVPanoptix3Dv2 config schema and shipped specs."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import nvidia_tao_pytorch.cv.nvpanoptix3d_v2 as nvpanoptix3d_v2_pkg
from nvidia_tao_pytorch.config.nvpanoptix3d_v2.default_config import ExperimentConfig
from nvidia_tao_pytorch.config.nvpanoptix3d_v2.model import NVPanoptix3Dv2ModelConfig
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.build_pl_model import (
    PANOPTIC,
    REASONING,
    SUPPORTED_MODEL_TYPES,
    get_pl_module,
)


SPEC_DIR = Path(nvpanoptix3d_v2_pkg.__file__).parent / "experiment_specs"


@pytest.fixture
def experiment_spec():
    """Return the default structured experiment config."""
    return OmegaConf.structured(ExperimentConfig())


@pytest.mark.cv_unit
def test_experiment_config_instantiates(experiment_spec):
    """The full schema must build with the expected defaults."""
    assert experiment_spec.model.model_type == PANOPTIC
    assert experiment_spec.model.backbone.metric_depth_head.predict_shift is True


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    "spec_name,model_type",
    (
        ("spec_panoptic.yaml", PANOPTIC),
        ("spec_reasoning.yaml", REASONING),
    ),
)
def test_shipped_specs_match_schema_and_format(
    spec_name, model_type, experiment_spec
):
    """Each shipped spec must be well formatted and merge with the schema."""
    spec_path = SPEC_DIR / spec_name
    text = spec_path.read_text(encoding="utf-8")
    spec = OmegaConf.load(spec_path)
    merged = OmegaConf.merge(experiment_spec, spec)

    assert all(line.strip() for line in text.splitlines())
    assert "nvpanoptix3dv2" not in text
    assert "nvpanoptix3d_v2" in text
    assert merged.model.model_type == model_type


@pytest.mark.cv_unit
def test_console_entrypoint_uses_the_package_name():
    """The installed CLI and import target must use the same package name."""
    repo_root = SPEC_DIR.parents[3]
    setup_text = (repo_root / "setup.py").read_text(encoding="utf-8")
    target = (
        "nvpanoptix3d_v2="
        "nvidia_tao_pytorch.cv.nvpanoptix3d_v2.entrypoint.nvpanoptix3d_v2:main"
    )

    assert target in setup_text
    assert (SPEC_DIR.parent / "entrypoint" / "nvpanoptix3d_v2.py").is_file()


@pytest.mark.cv_unit
def test_specs_keep_the_streamlined_public_contract():
    """Specs should contain required overrides, not schema defaults."""
    panoptic = OmegaConf.load(SPEC_DIR / "spec_panoptic.yaml")
    reasoning = OmegaConf.load(SPEC_DIR / "spec_reasoning.yaml")
    panoptic_data = panoptic.dataset.panoptic
    reasoning_data = reasoning.dataset.reasoning

    assert panoptic_data.train_preprocessed_root.endswith("/pre_scannetpp_v2")
    assert panoptic_data.val_preprocessed_root.endswith("/pre_scannetpp_v2_val")
    assert not {
        "num_workers",
        "train_num_workers",
        "val_num_workers",
        "train_pairs_root",
        "val_pairs_root",
    }.intersection(panoptic_data)
    assert "type" not in panoptic.model.panoptic.upscaler

    assert not {
        "num_workers",
        "val_num_workers",
        "depth_scale",
    }.intersection(reasoning_data)
    assert "predict_shift" not in reasoning.model.backbone.metric_depth_head

    for spec in (panoptic, reasoning):
        assert spec.results_dir
        assert "results_dir" not in spec.train
        assert "wandb" not in spec


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    "spec_name,monitor",
    (
        ("spec_panoptic.yaml", "val/mAP"),
        ("spec_reasoning.yaml", "val/mIoU"),
    ),
)
def test_specs_use_the_common_tao_checkpointer(spec_name, monitor):
    """Both variants configure metric-best retention through TAO's API."""
    checkpointer = OmegaConf.load(SPEC_DIR / spec_name).train.checkpointer

    assert checkpointer.enable_topk is True
    assert checkpointer.monitor == monitor
    assert OmegaConf.select(checkpointer, "mode") is None
    assert checkpointer.save_top_k == 2


@pytest.mark.cv_unit
def test_spec_component_shapes_are_compatible():
    """Catch config combinations that would fail at the first model forward."""
    panoptic = OmegaConf.load(SPEC_DIR / "spec_panoptic.yaml")
    reasoning = OmegaConf.load(SPEC_DIR / "spec_reasoning.yaml")
    model = panoptic.model
    fusion = model.panoptic.feature_fusion
    decoder = model.panoptic.panoptic_decoder
    upscaler = model.panoptic.upscaler

    assert model.patch_size == upscaler.patch_size
    assert model.embed_dim == fusion.dino_dim
    assert fusion.hidden_dim == decoder.hidden_dim == upscaler.input_dim
    assert decoder.mask_dim == upscaler.dim
    assert all(
        height % model.patch_size == 0 and width % model.patch_size == 0
        for height, width in panoptic.dataset.panoptic.resolution
    )
    assert panoptic.export.input_height % model.patch_size == 0
    assert panoptic.export.input_width % model.patch_size == 0

    height, width = reasoning.dataset.reasoning.resolution
    assert height % reasoning.model.patch_size == 0
    assert width % reasoning.model.patch_size == 0
    context_views = reasoning.model.backbone.metric_depth_head.metric_context_views
    assert context_views <= reasoning.dataset.reasoning.num_views


@pytest.mark.cv_unit
def test_model_type_contract(experiment_spec):
    """The schema and builder must agree on the supported variants."""
    field = NVPanoptix3Dv2ModelConfig.__dataclass_fields__["model_type"]
    assert set(field.metadata["valid_options"].split(",")) == set(
        SUPPORTED_MODEL_TYPES
    )

    experiment_spec.model.model_type = "not-a-variant"
    with pytest.raises(ValueError, match="Unsupported model.model_type"):
        get_pl_module(experiment_spec)
