# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NVPanoptix3Dv2 ONNX export helpers."""

import json

import onnx
import pytest
import torch
from torch import nn

from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.layers.attention import (
    Attention,
    MemEffAttention,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.export.onnx_exporter import (
    FrozenTextEmbeddings,
    NVPanoptix3Dv2PanopticExportWrapper,
    depth_stats_export_safe,
    export_safe_metric_scale_head,
    flatten_outputs,
    patch_xformers_attention_for_export,
    write_vocabulary_sidecar,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.export.symbolic_funcs import (
    register_symbolic_functions,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.metric_scale_head import (
    MetricScaleHead,
)


BATCH = 2
VIEWS = 3
HW = 8


class _Expm1Model(nn.Module):
    """Minimal graph exercising the unsupported aten::expm1 operation."""

    def forward(self, value):
        """Apply the activation that requires the custom symbolic."""
        return torch.expm1(value)


class _TriuIndicesModel(nn.Module):
    """Minimal graph exercising the triangular index factory."""

    def forward(self, value):
        """Gather strict upper-triangular values using traced indices."""
        row, col = torch.triu_indices(
            value.shape[0],
            value.shape[1],
            offset=1,
            device=value.device,
        )
        return value[row, col]


class _FakeDecoder(nn.Module):
    """Provide the decoder attributes modified for export."""

    def __init__(self):
        super().__init__()
        self.text_encoder = None
        self.deep_supervision = True


class _FakePanopticModel(nn.Module):
    """Provide the panoptic forward contract expected by the wrapper."""

    def __init__(self, with_objectness=False):
        super().__init__()
        self.panoptic_decoder = _FakeDecoder()
        self.with_objectness = with_objectness
        self.seen_classes = "unset"

    def forward(self, images, true_shape, classes):
        """Return representative panoptic and geometry outputs."""
        self.seen_classes = classes
        self.seen_true_shape = true_shape
        batch = images.shape[0]
        panoptic = {
            "pred_logits": torch.zeros(batch, 4, 5),
            "pred_masks": torch.zeros(batch, VIEWS, 4, HW, HW),
            "aux_outputs": [{"pred_logits": torch.zeros(batch, 4, 5)}],
            "out_queries": torch.zeros(batch, 4, 8),
        }
        if self.with_objectness:
            panoptic["pred_objectness"] = torch.zeros(batch, 4)
        geometry = {
            "depth": torch.zeros(batch, VIEWS, HW, HW, 1),
            "intrinsics": torch.zeros(batch, VIEWS, 3, 3),
            "metric_scale_params": {"scale": torch.ones(batch, 1)},
        }
        return panoptic, geometry


def _depth(depth_rank=5, seed=0):
    """Return positive, non-degenerate relative depth."""
    generator = torch.Generator().manual_seed(seed)
    depth = torch.rand(
        BATCH, VIEWS, HW, HW, 1, generator=generator
    ) * 4.0 + 0.25
    return depth if depth_rank == 5 else depth.squeeze(-1)


@pytest.mark.cv_unit
@pytest.mark.parametrize("depth_rank", (4, 5))
def test_export_safe_depth_stats_matches_training(depth_rank):
    """The ONNX-safe implementation must preserve depth statistics."""
    head = MetricScaleHead(scene_token_dim=8, hidden_dims=(8, 4))
    depth = _depth(depth_rank)

    assert torch.allclose(
        head.depth_stats(depth),
        depth_stats_export_safe(head, depth),
        atol=1e-6,
    )


@pytest.mark.cv_unit
def test_export_safe_depth_stats_respects_clamp_range():
    """Both implementations must apply the configured depth bounds."""
    head = MetricScaleHead(
        scene_token_dim=8,
        hidden_dims=(8, 4),
        depth_min=1.0,
        depth_max=2.0,
    )
    depth = _depth() * 100.0

    assert torch.allclose(
        head.depth_stats(depth),
        depth_stats_export_safe(head, depth),
        atol=1e-6,
    )


@pytest.mark.cv_unit
def test_export_safe_context_restores_after_failure():
    """A failed export must restore the training implementation."""
    original = MetricScaleHead.depth_stats

    with pytest.raises(RuntimeError, match="export failed"):
        with export_safe_metric_scale_head():
            assert MetricScaleHead.depth_stats is depth_stats_export_safe
            raise RuntimeError("export failed")

    assert MetricScaleHead.depth_stats is original


@pytest.mark.cv_unit
def test_flatten_outputs_filters_and_orders_tensors():
    """Only supported tensors cross the ONNX boundary, in canonical order."""
    panoptic = {
        "pred_masks": torch.zeros(1),
        "aux_outputs": [{"pred_logits": torch.zeros(1)}],
        "out_queries": torch.zeros(1),
        "pred_logits": torch.zeros(1),
    }
    geometry = {
        "metric_scale_params": {"scale": torch.ones(1)},
        "pose_enc": torch.zeros(1),
        "depth": torch.zeros(1),
    }

    assert list(flatten_outputs(panoptic, geometry)) == [
        "pred_logits",
        "pred_masks",
        "depth",
        "pose_enc",
    ]


@pytest.mark.cv_unit
def test_frozen_text_embeddings_are_exportable():
    """Frozen vocabulary embeddings ignore input and remain model state."""
    embeddings = torch.randn(5, 16)
    frozen = FrozenTextEmbeddings(embeddings)

    assert torch.equal(frozen(None), embeddings)
    assert torch.equal(frozen(["ignored"]), embeddings)
    assert "embeddings" in frozen.state_dict()


@pytest.mark.cv_unit
def test_export_uses_the_shared_xformers_patch():
    """The legacy ONNX tracer must not receive xFormers operators."""
    from nvidia_tao_pytorch.cv.nvpanoptix3d.export.utils import (
        patch_xformers_attention_for_export as shared_patch,
    )

    assert patch_xformers_attention_for_export is shared_patch
    model = nn.Sequential(MemEffAttention(dim=8, num_heads=2))
    attention = model[0]

    patch_xformers_attention_for_export(model)

    assert attention.forward.__func__ is Attention.forward


@pytest.mark.cv_unit
def test_wrapper_freezes_and_flattens_the_model():
    """The wrapper disables training paths and exposes a flat tensor tuple."""
    model = _FakePanopticModel()
    images = torch.zeros(BATCH, VIEWS, 3, HW, HW)
    wrapper = NVPanoptix3Dv2PanopticExportWrapper(
        model,
        torch.randn(5, 16),
        probe_input=images,
    )
    outputs = wrapper(images)

    assert isinstance(model.panoptic_decoder.text_encoder, FrozenTextEmbeddings)
    assert model.panoptic_decoder.deep_supervision is False
    assert wrapper.output_names == [
        "pred_logits",
        "pred_masks",
        "depth",
        "intrinsics",
    ]
    assert isinstance(outputs, tuple)
    assert len(outputs) == len(wrapper.output_names)
    assert all(isinstance(value, torch.Tensor) for value in outputs)
    assert model.seen_true_shape.shape == (BATCH, VIEWS, 2)
    assert model.seen_classes is None


@pytest.mark.cv_unit
def test_wrapper_accepts_explicit_output_names_without_probing():
    """Explicit names allow construction without a probe forward."""
    model = _FakePanopticModel()
    wrapper = NVPanoptix3Dv2PanopticExportWrapper(
        model,
        torch.randn(5, 16),
        output_names=["pred_logits"],
    )

    assert wrapper.output_names == ["pred_logits"]
    assert model.seen_classes == "unset"


@pytest.mark.cv_unit
def test_wrapper_requires_probe_or_output_names():
    """The wrapper needs one way to determine its output signature."""
    with pytest.raises(ValueError, match="probe_input or output_names"):
        NVPanoptix3Dv2PanopticExportWrapper(
            _FakePanopticModel(),
            torch.randn(5, 16),
        )


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    "categories",
    (
        None,
        [
            {"name": "wall", "id": 0},
            {"name": "floor", "id": 1},
            {"name": "chair", "id": 2},
        ],
    ),
)
def test_write_vocabulary_sidecar_records_channel_order(tmp_path, categories):
    """The sidecar records the class-channel order and optional metadata."""
    classes = ["wall", "floor", "chair"]
    sidecar = write_vocabulary_sidecar(
        str(tmp_path / "model.onnx"),
        classes,
        categories,
    )

    assert sidecar == str(tmp_path / "model.classes.json")
    with open(sidecar, "r", encoding="utf-8") as handle:
        assert json.load(handle) == {
            "classes": classes,
            "categories": categories,
        }


@pytest.mark.cv_unit
def test_expm1_symbolic_lowers_to_standard_onnx_ops(tmp_path):
    """VGGT's Expm1 activation must not remain as an aten operation."""
    register_symbolic_functions(17)
    output = tmp_path / "expm1.onnx"

    torch.onnx.export(
        _Expm1Model(),
        torch.tensor([-1.0, 0.0, 1.0]),
        str(output),
        opset_version=17,
        dynamo=False,
    )

    graph = onnx.load(str(output)).graph
    op_types = [node.op_type for node in graph.node]
    assert {"Exp", "CastLike", "Sub"} <= set(op_types)


@pytest.mark.cv_unit
def test_triu_indices_symbolic_lowers_to_constant(tmp_path):
    """Fixed view-pair indices must be embedded without aten operations."""
    register_symbolic_functions(17)
    output = tmp_path / "triu_indices.onnx"

    torch.onnx.export(
        _TriuIndicesModel(),
        torch.arange(25, dtype=torch.float32).reshape(5, 5),
        str(output),
        opset_version=17,
        dynamo=False,
    )

    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    assert all(node.domain != "aten" for node in model.graph.node)
