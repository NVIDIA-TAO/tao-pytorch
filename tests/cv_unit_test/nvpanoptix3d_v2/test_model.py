# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NVPanoptix3Dv2 model components."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.metric_scale_head import (
    MetricScaleHead,
    apply_metric_scale,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.panoptic import (
    feature_fusion,
    loftup,
    mask_transformer,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.panoptic.panoptic_model import (
    NVPanoptix3Dv2Panoptic,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.reasoning.qwen_vlm import (
    bind_seg_to_views,
    extract_seg_hidden,
    find_answer_start,
    resolve_model_source,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.reasoning.reasoning_model import (
    NVPanoptix3Dv2Reasoning,
    SegToSAMPromptProjector,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.vggt.models import (
    aggregator,
    vggt,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.vggt.models.patch_embed import (
    PatchEmbed,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.metric_depth import (
    MetricDepthLoss,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.panoptic.criterion import (
    masked_dice_loss,
    masked_sigmoid_ce_loss,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.panoptic.engine_eval import (
    panoptic_inference,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.panoptic.losses import (
    NVPanoptix3Dv2PanopticLoss,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.panoptic.mAP import (
    evaluate_instance_map_segvggt,
    prepare_instance_map_sample,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.reasoning.losses import (
    NVPanoptix3Dv2ReasoningLoss,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.utils.reasoning.score import (
    CanonicalPointCloudMetrics,
    score_batch_on_canonical_points,
)


BATCH = 2
VIEWS = 4
PATCHES = 6
TOKEN_DIM = 8
HW = 8


class _Backbone(nn.Module):
    """Small VGGT stand-in exposing the tensors used by the model."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, images):
        """Return deterministic geometry and feature tensors."""
        batch, views, _, height, width = images.shape
        return {
            "dino_feats": torch.ones(batch, views, 2, 3),
            "vggt_feats": torch.ones(batch, views, 2, 4),
            "depth": torch.ones(batch, views, height, width, 1),
            "depth_conf": torch.ones(batch, views, height, width),
            "world_points": torch.ones(batch, views, height, width, 3),
            "world_points_conf": torch.ones(batch, views, height, width),
            "pose_enc": torch.zeros(batch, views, 9),
        }


class _Decoder(nn.Module):
    """Record the model-to-decoder contract."""

    def forward(
        self,
        dino_feats,
        vggt_feats,
        images,
        classes,
        outdevice=None,
    ):
        """Return representative panoptic outputs."""
        self.call = (dino_feats, vggt_feats, images, classes, outdevice)
        batch, views = images.shape[:2]
        return {
            "pred_logits": images.new_zeros(
                (batch, 2, len(classes))
            ),
            "pred_masks": images.new_zeros((batch, views, 2, 2, 2)),
        }


class _MetricHead(nn.Module):
    """Return a deterministic scene scale and intrinsics."""

    def forward(
        self,
        vggt_feats,
        rel_depth,
        pose_enc,
        image_size_hw,
    ):
        """Return scale two for every scene."""
        self.call = (vggt_feats, rel_depth, pose_enc, image_size_hw)
        batch, views = rel_depth.shape[:2]
        scale = rel_depth.new_full((batch, 1), 2.0)
        return {
            "log_s": scale.log(),
            "scale": scale,
            "intrinsics": torch.eye(3).expand(
                batch, views, 3, 3
            ).clone(),
        }


class _Qwen(nn.Module):
    """Minimal language-model output contract."""

    def forward(self, inputs):
        """Return tensors supplied by the test."""
        return SimpleNamespace(
            lm_logits=inputs["lm_logits"],
            labels=inputs.get("labels"),
            hidden=inputs["hidden"],
            input_ids=inputs["input_ids"],
        )

    @staticmethod
    def extract_seg_hidden(output):
        """Gather hidden states at the synthetic SEG token."""
        return extract_seg_hidden(
            output.hidden,
            output.input_ids,
            seg_token_id=9,
        )


class _Sam(nn.Module):
    """Parameter-only SAM stand-in."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))


def _pose_enc(batch=BATCH, views=VIEWS, seed=0):
    """Build a plausible pose-encoding tensor."""
    generator = torch.Generator().manual_seed(seed)
    translation = torch.randn(
        batch, views, 3, generator=generator
    ) * 0.5
    quaternion = torch.randn(batch, views, 4, generator=generator)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    fov = torch.full((batch, views, 2), 0.9)
    return torch.cat([translation, quaternion, fov], dim=-1)


def _metric_inputs(views=VIEWS):
    """Return small inputs accepted by MetricScaleHead."""
    features = torch.randn(BATCH, views, PATCHES, TOKEN_DIM)
    depth = torch.rand(BATCH, views, HW, HW, 1) + 0.5
    return features, depth, _pose_enc(views=views), (HW, HW)


def _metric_head(**kwargs):
    """Build a small metric head."""
    kwargs.setdefault("scene_token_dim", TOKEN_DIM)
    kwargs.setdefault("hidden_dims", (8, 4))
    return MetricScaleHead(**kwargs)


@pytest.mark.cv_unit
def test_panoptic_forward():
    """The top-level model connects features and metric geometry correctly."""
    decoder = _Decoder()
    metric_head = _MetricHead()
    model = NVPanoptix3Dv2Panoptic(
        _Backbone(),
        decoder,
        metric_head,
    )
    images = torch.zeros(2, 3, 3, 4, 5)
    true_shape = torch.tensor([[[4, 5]] * 3] * 2)

    panoptic, geometry = model(
        images,
        true_shape,
        ["chair", "table"],
    )

    assert panoptic["pred_logits"].shape == (2, 2, 2)
    assert decoder.call[0].shape == (2, 3, 2, 3)
    assert decoder.call[1].shape == (2, 3, 2, 4)
    assert torch.equal(
        geometry["metric_depth"],
        2.0 * geometry["depth"],
    )
    assert torch.equal(
        geometry["metric_points"],
        2.0 * geometry["world_points"],
    )
    assert metric_head.call[-1] == (4, 5)


@pytest.mark.cv_unit
def test_panoptic_input_and_freezing():
    """Invalid metadata fails early and freezing affects only VGGT."""
    backbone = _Backbone()
    decoder = nn.Linear(2, 2)
    model = NVPanoptix3Dv2Panoptic(backbone, decoder)
    images = torch.zeros(2, 3, 3, 4, 5)

    with pytest.raises(ValueError, match="true_shape must have shape"):
        model(images, torch.zeros(2, 2, 2), ["chair"])

    model.freeze_vggt_weights()
    assert all(
        not parameter.requires_grad
        for parameter in backbone.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in decoder.parameters()
    )


@pytest.mark.cv_unit
@pytest.mark.parametrize("predict_shift", (False, True))
def test_metric_head_identity(predict_shift):
    """A new metric head preserves depth in scale and shift modes."""
    head = _metric_head(predict_shift=predict_shift)
    features, depth, pose_enc, image_size = _metric_inputs()
    params = head(features, depth, pose_enc, image_size)

    assert params["scale"].shape == (BATCH, 1)
    assert params["intrinsics"].shape == (BATCH, VIEWS, 3, 3)
    assert torch.allclose(params["scale"], torch.ones_like(params["scale"]))
    assert torch.allclose(apply_metric_scale(depth, params), depth)
    assert ("b" in params) is predict_shift


@pytest.mark.cv_unit
def test_metric_head_context_and_gradients():
    """Only the configured view prefix contributes, without VGGT gradients."""
    head = _metric_head(metric_context_views=2)
    torch.nn.init.normal_(head.mlp[-1].weight, std=0.1)
    features, depth, pose_enc, image_size = _metric_inputs()

    baseline = head(features, depth, pose_enc, image_size)["scale"]
    perturbed = features.clone()
    perturbed[:, 2:] += 10.0
    assert torch.allclose(
        baseline,
        head(perturbed, depth, pose_enc, image_size)["scale"],
    )

    features.requires_grad_(True)
    depth.requires_grad_(True)
    head(features, depth, pose_enc, image_size)["scale"].sum().backward()
    assert features.grad is None
    assert depth.grad is None
    assert head.mlp[-1].weight.grad is not None


@pytest.mark.cv_unit
def test_metric_head_validation():
    """Invalid context and mismatched view counts are rejected."""
    with pytest.raises(ValueError, match="must be positive"):
        _metric_head(metric_context_views=0)

    head = _metric_head()
    features, depth, pose_enc, image_size = _metric_inputs()
    with pytest.raises(ValueError, match="view counts must match"):
        head(features, depth[:, :-1], pose_enc, image_size)
    with pytest.raises(ValueError, match="must be 4D or 5D"):
        apply_metric_scale(
            torch.ones(BATCH, VIEWS, HW),
            {"scale": torch.ones(BATCH, 1)},
        )


@pytest.mark.cv_unit
def test_panoptic_loss_contract():
    """Public loss weights drive matching and invalid pixels are ignored."""
    loss = NVPanoptix3Dv2PanopticLoss(
        class_weight=2.5,
        mask_weight=7.0,
        dice_weight=3.0,
        deep_supervision=False,
    )
    matcher = loss.criterion.matcher
    assert matcher.cost_class == pytest.approx(2.5)
    assert matcher.cost_mask == pytest.approx(7.0)
    assert matcher.cost_dice == pytest.approx(3.0)

    target = torch.tensor([[[1.0, 0.0]]])
    valid = torch.tensor([[[1.0, 0.0]]])
    baseline = torch.tensor([[[10.0, -10.0]]])
    changed_void = torch.tensor([[[10.0, 10.0]]])
    for loss_fn in (masked_sigmoid_ce_loss, masked_dice_loss):
        assert torch.allclose(
            loss_fn(baseline, target, valid, 1.0),
            loss_fn(changed_void, target, valid, 1.0),
        )


@pytest.mark.cv_unit
@pytest.mark.parametrize("gate", (None, "objectness"))
def test_panoptic_inference(gate):
    """Inference creates disjoint segments and honors confidence gating."""
    kwargs = {}
    if gate:
        kwargs[f"{gate}_logits"] = torch.full((1, 2), -10.0)
    result = panoptic_inference(
        torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]]),
        torch.tensor(
            [[[[[10.0, 10.0], [-10.0, -10.0]],
               [[-10.0, -10.0], [10.0, 10.0]]]]]
        ),
        target_hw=(2, 2),
        mask_threshold=0.5,
        **kwargs,
    )[0]

    if gate:
        assert not result["segments_info"]
    else:
        assert torch.equal(
            result["pan"],
            torch.tensor([[[1, 1], [2, 2]]], dtype=torch.int32),
        )


@pytest.mark.cv_unit
def test_instance_map():
    """An exactly aligned instance has perfect AP."""
    sample = prepare_instance_map_sample(
        np.asarray([[[1, 1, 0, 0]]]),
        [{"id": 1, "category_id": 0, "score": 0.9}],
        np.asarray([[[7, 7, 0, 0]]]),
        np.asarray([[[0, 0, -1, -1]]]),
        num_categories=2,
        evaluated_category_ids=[0],
        min_points=1,
    )

    assert evaluate_instance_map_segvggt([sample], [0]) == {
        "mAP": 100.0,
        "mAP50": 100.0,
        "mAP25": 100.0,
    }


@pytest.mark.cv_unit
def test_qwen_token_binding():
    """SEG hidden states preserve sample order and bind to requested views."""
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    token_ids = torch.tensor([[9, 2, 9, 0], [1, 9, 0, 0]])
    selected, batch_index = extract_seg_hidden(
        hidden,
        token_ids,
        seg_token_id=9,
    )
    with pytest.warns(RuntimeWarning, match="dropped 1"):
        sample, view, keep = bind_seg_to_views(
            batch_index,
            [[1], [0]],
            num_views=2,
        )

    assert torch.equal(
        selected,
        torch.stack([hidden[0, 0], hidden[0, 2], hidden[1, 1]]),
    )
    assert sample.tolist() == [0, 1]
    assert view.tolist() == [1, 0]
    assert keep.tolist() == [0, 2]
    assert find_answer_start(
        torch.tensor([5, 1, 2, 6, 1, 2, 7]),
        [1, 2],
    ) == 6


@pytest.mark.cv_unit
def test_qwen_model_source(tmp_path):
    """Local Qwen sources require processor and tokenizer metadata."""
    for name in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    ):
        (tmp_path / name).touch()
    source, is_local = resolve_model_source(str(tmp_path))
    assert source == str(tmp_path)
    assert is_local

    with pytest.raises(FileNotFoundError, match="Check the container mount"):
        resolve_model_source(str(tmp_path / "missing"))


@pytest.mark.cv_unit
def test_reasoning_projector():
    """The projector keeps one prompt per sample and marks missing prompts."""
    projector = SegToSAMPromptProjector(
        d_llm=4,
        d_sam=3,
        hidden=5,
    )
    hidden = torch.randn(3, 4)
    batch_index = torch.tensor([0, 0, 2])

    prompt, valid = projector(hidden, batch_index, batch_size=4)
    projected = projector.project_all(hidden)

    assert prompt.shape == (1, 4, 3)
    assert valid.tolist() == [True, False, True, False]
    assert torch.allclose(prompt[0, 0], projected[0])
    assert torch.allclose(prompt[0, 2], projected[2])
    assert torch.allclose(prompt[0, 1], projector.null_prompt[0])


@pytest.mark.cv_unit
def test_reasoning_forward(monkeypatch):
    """Each SEG prompt is sent to its requested image view."""
    sam = _Sam()
    model = NVPanoptix3Dv2Reasoning(
        qwen=_Qwen(),
        sam3_image_model=sam,
        projector=SegToSAMPromptProjector(
            d_llm=4,
            d_sam=3,
            hidden=5,
        ),
    )
    captured = {}

    def fake_sam_forward(images, prompt):
        captured["images"] = images
        captured["prompt"] = prompt
        count = images.shape[0]
        return {
            "pred_masks": images.new_zeros((count, 1, 1, 1)),
            "pred_logits": images.new_zeros((count, 1)),
            "presence_logits": None,
        }

    monkeypatch.setattr(
        model,
        "sam_forward_with_prompt",
        fake_sam_forward,
    )
    images = torch.arange(
        2 * 2 * 3 * 2 * 2,
        dtype=torch.float32,
    ).reshape(2, 2, 3, 2, 2)
    qwen_inputs = {
        "hidden": torch.randn(2, 3, 4),
        "input_ids": torch.tensor([[1, 9, 2], [9, 2, 3]]),
        "lm_logits": torch.randn(2, 3, 5),
        "labels": None,
    }

    output = model(
        images,
        qwen_inputs,
        seg_view_indices=[[1], [0]],
    )

    assert output["seg_sample_idx"].tolist() == [0, 1]
    assert output["seg_view_idx"].tolist() == [1, 0]
    assert torch.equal(captured["images"], images[[0, 1], [1, 0]])
    assert captured["prompt"].shape == (1, 2, 3)
    assert not sam.weight.requires_grad


@pytest.mark.cv_unit
def test_reasoning_lift_and_loss():
    """Reasoning lifts bound views and shares the metric-depth objective."""
    world_points = torch.arange(
        2 * 2 * 1 * 2 * 3,
        dtype=torch.float32,
    ).reshape(2, 2, 1, 2, 3)
    pred_masks = torch.tensor(
        [
            [[[10.0, -10.0]], [[-10.0, 10.0]]],
            [[[-10.0, 10.0]], [[10.0, -10.0]]],
        ]
    )
    pred_logits = torch.tensor(
        [[10.0, -10.0], [-10.0, 10.0]]
    )
    clouds, _, selected, _ = (
        NVPanoptix3Dv2Reasoning.lift_bindings_to_point_clouds(
            pred_masks,
            pred_logits,
            world_points,
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
            batch_size=2,
        )
    )
    expected = torch.stack(
        [world_points[0, 0, 0, 0], world_points[0, 1, 0, 0]]
    )
    assert torch.equal(clouds[0], expected)
    assert clouds[1].shape == (0, 3)
    assert selected.tolist() == [0, 1]

    criterion = NVPanoptix3Dv2ReasoningLoss(metric_weight=1.0)
    gt_depth = torch.linspace(0.5, 4.0, 16).reshape(1, 1, 4, 4)
    metric_depth = (gt_depth * 1.1).unsqueeze(-1)
    output = {
        "pred_masks": torch.zeros(1, 1, 4, 4),
        "metric_depth": metric_depth,
    }
    assert torch.allclose(
        criterion.metric_depth_loss(output, gt_depth),
        MetricDepthLoss()(metric_depth, gt_depth)["loss_metric_total"],
    )


@pytest.mark.cv_unit
def test_reasoning_metric():
    """Canonical scoring uses every valid view."""
    meter = CanonicalPointCloudMetrics()
    output = {
        "seg_sample_idx": torch.tensor([0]),
        "seg_view_idx": torch.tensor([0]),
        "pred_masks": torch.tensor([[[[10.0, -10.0]]]]),
        "pred_logits": torch.tensor([[1.0]]),
    }
    panoptic = torch.tensor([[[[7, 7]], [[0, 0]]]])
    score_batch_on_canonical_points(
        meter,
        output,
        panoptic,
        torch.tensor([7]),
        torch.ones_like(panoptic, dtype=torch.bool),
        0.5,
    )

    assert meter.compute() == {
        "mIoU": pytest.approx(0.5),
        "mAP50": pytest.approx(1.0),
        "mAP25": pytest.approx(1.0),
    }


LAYER_BINDINGS = (
    (aggregator, "Block", "layers", "block"),
    (aggregator, "PositionGetter", "layers", "rope"),
    (aggregator, "vit_base", "layers", "vision_transformer"),
    (feature_fusion, "Block", "layers", "block"),
    (feature_fusion, "RotaryPositionEmbedding2D", "layers", "rope"),
    (vggt, "CameraHead", "heads", "camera_head"),
    (vggt, "DPTHead", "heads", "dpt_head"),
)


@pytest.mark.cv_unit
@pytest.mark.parametrize(
    "consumer,name,subpackage,module",
    LAYER_BINDINGS,
)
def test_shared_model_imports(consumer, name, subpackage, module):
    """Shared VGGT objects are imported rather than copied."""
    source = importlib.import_module(
        "nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt."
        f"{subpackage}.{module}"
    )
    assert getattr(consumer, name) is getattr(source, name)


@pytest.mark.cv_unit
def test_shared_model_helpers():
    """Generic model helpers bind to their shared TAO implementations."""
    from nvidia_tao_pytorch.cv.mask2former.model.transformer_decoder.position_encoding import (
        PositionEmbeddingSine,
    )
    from nvidia_tao_pytorch.cv.nvpanoptix3d.model.transformer_decoder.joint_depth_transformer_decoder import (
        FFNLayer,
        MLP,
        SelfAttentionLayer,
    )
    from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.layers.mlp import Mlp
    from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.models.aggregator import (
        slice_expand_and_flatten,
    )

    assert mask_transformer.FFNLayer is FFNLayer
    assert mask_transformer.MLP is MLP
    assert mask_transformer.PositionEmbeddingSine is PositionEmbeddingSine
    assert mask_transformer.SelfAttentionLayer is SelfAttentionLayer
    assert loftup.Mlp is Mlp
    assert aggregator.slice_expand_and_flatten is (
        slice_expand_and_flatten
    )


@pytest.mark.cv_unit
def test_local_patch_embed():
    """PatchEmbed remains local because the V2 version differs."""
    assert aggregator.PatchEmbed is PatchEmbed
    assert PatchEmbed.__module__.startswith(
        "nvidia_tao_pytorch.cv.nvpanoptix3d_v2"
    )
