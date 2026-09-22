# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused loss and distributed contracts for Sparse4D co-training."""

import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.sparse4d.model.criterion import (
    DenseDepthLoss,
    FocalLoss,
    SetCriterion,
    SparseBox3DLoss,
)
from nvidia_tao_pytorch.cv.sparse4d.model.loose_to_tight_mlp import (
    LooseToTightMLP,
    WAREHOUSE_V4_CLASSES,
)
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d_pl_model import Sparse4DPlModel


pytestmark = pytest.mark.cv_unit


def _bare_criterion() -> SetCriterion:
    """Create a criterion shell for testing helpers that do not need configuration."""
    criterion = SetCriterion.__new__(SetCriterion)
    torch.nn.Module.__init__(criterion)
    criterion.scrub_nan_gradients = False
    return criterion


def _bare_lightning_model() -> Sparse4DPlModel:
    """Create a Lightning shell for the standalone synchronization helper."""
    model = Sparse4DPlModel.__new__(Sparse4DPlModel)
    torch.nn.Module.__init__(model)
    model.cotrain_loss_keys = ("loss_a", "loss_m", "loss_z")
    return model


def test_focal_loss_supervises_background_and_ignores_negative_labels():
    """Class index C is background, while negative labels remain ignored."""
    logits = torch.zeros((2, 3), requires_grad=True)
    targets = torch.tensor([3, -1])
    loss = FocalLoss(gamma=2.0, alpha=0.25)(
        logits,
        targets,
        avg_factor=1.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    loss.backward()
    assert torch.all(logits.grad[0] > 0.0)
    torch.testing.assert_close(logits.grad[1], torch.zeros(3))


def test_nonfinite_loss_uses_backward_safe_upstream_zero():
    """Replacing an invalid loss must not retain its invalid local Jacobian."""
    prediction = torch.tensor(0.0, requires_grad=True)
    depth = torch.tensor(2.0, requires_grad=True)
    invalid_loss = prediction / prediction
    criterion = _bare_criterion()
    criterion.scrub_nan_gradients = True
    zero_ref = criterion._prediction_zero(
        [prediction],
        [],
        [depth],
    )

    with pytest.warns(RuntimeWarning, match="non-finite"):
        losses = criterion._finite_losses(
            {"loss_invalid": invalid_loss},
            zero_ref=zero_ref,
        )

    assert losses["loss_invalid"] is zero_ref
    assert losses["loss_invalid"].item() == 0.0
    losses["loss_invalid"].backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad)
    assert prediction.grad.item() == 0.0
    assert depth.grad is not None
    assert depth.grad.item() == 0.0


@pytest.mark.parametrize("velocity_weight", [-1, 2])
def test_empty_positive_regression_is_a_finite_graph_zero(velocity_weight):
    """Empty GT is supported without enabling numerical recovery."""
    boxes = torch.empty((0, 11), requires_grad=True)
    quality = torch.empty((0, 2), requires_grad=True)
    losses = SparseBox3DLoss(valid_vel_weight=velocity_weight)(
        boxes, torch.empty_like(boxes), quality=quality, avg_factor=1.0
    )
    assert all(loss.ndim == 0 and loss.item() == 0 for loss in losses.values())
    sum(losses.values()).backward()
    assert boxes.grad is not None
    assert quality.grad is not None


def test_dense_depth_loss_masks_invalid_targets_and_keeps_valid_depth():
    """Negative/NaN targets have no gradient while valid metric depth trains."""
    prediction = torch.tensor([[[[2.0, 4.0, 6.0]]]], requires_grad=True)
    target = torch.tensor([[[[-1.0, float("nan"), 5.0]]]])

    loss = DenseDepthLoss(loss_weight=0.2)([prediction], [target])

    torch.testing.assert_close(loss, torch.tensor(0.2))
    loss.backward()
    torch.testing.assert_close(
        prediction.grad,
        torch.tensor([[[[0.0, 0.0, 0.2]]]]),
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_raises_by_default(value):
    """Baseline training must not silently replace divergence with zero."""
    with pytest.raises(FloatingPointError, match="loss_bad is non-finite"):
        _bare_criterion()._finite_losses({"loss_bad": torch.tensor(value)})


@pytest.mark.parametrize("scrub", [False, True])
def test_extreme_classification_logits_clip_only_when_opted_in(scrub):
    """Default focal gradients remain active beyond the old clipping bounds."""
    criterion = _bare_criterion()
    criterion.scrub_nan_gradients = scrub
    logits = torch.tensor([[80.0, -80.0]], requires_grad=True)
    loss = FocalLoss()(criterion._classification_input(logits), torch.tensor([1]))
    loss.backward()
    if scrub:
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
    else:
        assert torch.all(logits.grad.abs() > 0)


@pytest.mark.parametrize("flags", [[True, False], [False, None], torch.tensor([0, 1])])
def test_mixed_supervision_batch_fails_fast(flags):
    """Mixed routes cannot silently omit pseudo-label supervision."""
    criterion = _bare_criterion()
    criterion.ltt_has3dgt_key = "has_3d_gt"
    with pytest.raises(ValueError, match="Mixed 2D/3D"):
        criterion._is_real_batch({"has_3d_gt": flags})


def test_loss_schema_rejects_unconfigured_keys():
    """A new loss must be added to the shared schema, not silently dropped."""
    with pytest.raises(ValueError, match="absent from configured schema"):
        _bare_lightning_model()._sync_cotrain_loss_keys({"loss_new": _loss(1.0)}, _loss(0.0))


def test_calibration_free_route_skips_dense_depth_supervision():
    """SV2D returns a graph zero even if a caller supplies a positive target."""
    criterion = _bare_criterion()

    class UnexpectedDepthLoss(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            raise AssertionError("calibration-free route invoked DenseDepthLoss")

    criterion.loss_depth = UnexpectedDepthLoss()
    prediction = torch.ones((1, 1, 1, 1), requires_grad=True)
    output = {}

    criterion._add_dense_depth_loss(
        output,
        [prediction],
        [torch.ones((1, 1, 1, 1))],
        skip=True,
    )

    torch.testing.assert_close(output["loss_dense_depth"], torch.tensor(0.0))
    output["loss_dense_depth"].backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_parameter_touch_zero_is_finite_for_poisoned_parameters():
    """DDP graph touching must not turn a skipped bad batch back into NaN."""
    parameter = torch.nn.Parameter(torch.tensor([float("nan"), float("inf"), 2.0]))

    zero = Sparse4DPlModel._parameter_touch_zero([parameter])

    assert torch.isfinite(zero)
    assert zero.item() == 0.0
    zero.backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))


def test_enabled_ltt_requires_a_trained_mlp_checkpoint():
    """Production LTT must fail before training when its frozen MLP is absent."""
    model_config = OmegaConf.structured(ExperimentConfig()).model
    model_config.head.loose_to_tight.enable = True
    model_config.head.loose_to_tight.mlp_ckpt = ""

    with pytest.raises(FileNotFoundError, match="mlp_ckpt is empty"):
        SetCriterion(model_config, instance_bank=None)


def test_released_taxonomy_is_derived_when_optional_heads_are_disabled():
    """The released four-class experiment remains valid with auto taxonomy."""
    experiment = OmegaConf.structured(ExperimentConfig())
    class_names = list(experiment.dataset.classes)

    criterion = SetCriterion(
        experiment.model,
        instance_bank=None,
        class_names=class_names,
    )

    assert class_names == [
        "person",
        "gr1_t2",
        "agility_digit",
        "nova_carter",
    ]
    assert criterion.ltt_num_classes == len(class_names)
    assert criterion.sv_aux_num_classes == len(class_names)


def test_sv_aux_auto_taxonomy_uses_dataset_classes():
    """A zero SV class count resolves to the detector output taxonomy."""
    experiment = OmegaConf.structured(ExperimentConfig())
    experiment.model.sv_aux_head.enable = True
    experiment.model.sv_aux_head.in_channels = 2
    experiment.model.sv_aux_head.roi_size = 1
    experiment.model.sv_aux_head.hidden_dim = 2
    experiment.model.sv_aux_head.fpn_strides = [4]
    experiment.model.sv_aux_head.use_level = 0

    criterion = SetCriterion(
        experiment.model,
        instance_bank=None,
        class_names=list(experiment.dataset.classes),
    )

    assert criterion.sv_aux_head.num_classes == 4


def test_sv_aux_explicit_class_count_must_match_dataset():
    """An active SV head rejects a count that diverges from detector logits."""
    experiment = OmegaConf.structured(ExperimentConfig())
    experiment.model.sv_aux_head.enable = True
    experiment.model.sv_aux_head.num_classes = 7

    with pytest.raises(ValueError, match="must be zero or match dataset.classes"):
        SetCriterion(
            experiment.model,
            instance_bank=None,
            class_names=list(experiment.dataset.classes),
        )


@pytest.mark.parametrize(
    "checkpoint_classes",
    [
        None,
        ["person", "gr1_t2", "nova_carter", "agility_digit"],
    ],
)
def test_ltt_checkpoint_must_record_exact_dataset_taxonomy(
    tmp_path,
    checkpoint_classes,
):
    """Missing or reordered MLP class metadata fails before training."""
    checkpoint = tmp_path / "ltt.pth"
    LooseToTightMLP(
        num_classes=4,
        hidden_dim=8,
        class_names=checkpoint_classes,
    ).save(str(checkpoint))
    experiment = OmegaConf.structured(ExperimentConfig())
    experiment.model.head.loose_to_tight.enable = True
    experiment.model.head.loose_to_tight.mlp_ckpt = str(checkpoint)

    with pytest.raises(ValueError, match=r"checkpoint .*class_names"):
        SetCriterion(
            experiment.model,
            instance_bank=None,
            class_names=list(experiment.dataset.classes),
        )


def test_sv_gradient_mask_preserves_only_observable_components():
    """SV projection masks depth/yaw and scales size gradients as configured."""
    criterion = _bare_criterion()
    criterion.ltt_sv_depth_weight = 0.0
    criterion.ltt_sv_size_weight = 0.25
    criterion.ltt_sv_yaw_weight = 0.0
    boxes = torch.tensor(
        [[1.0, 2.0, 8.0, 2.0, 4.0, 1.5, 0.3, 9.0]],
        requires_grad=True,
    )

    masked = criterion._sv_mask_boxes(boxes, torch.eye(4))
    torch.testing.assert_close(masked, boxes)
    masked.sum().backward()
    torch.testing.assert_close(
        boxes.grad,
        torch.tensor([[1.0, 1.0, 0.0, 0.25, 0.25, 0.25, 0.0, 1.0]]),
    )


@pytest.mark.parametrize(
    ("parameter_touch", "expected_checkpointing"),
    [(False, True), (True, True)],
)
def test_parameter_touch_preserves_nonreentrant_gradient_checkpointing(
    monkeypatch,
    parameter_touch,
    expected_checkpointing,
):
    """TAO's non-reentrant checkpointing stays enabled with parameter touch."""
    from nvidia_tao_pytorch.cv.sparse4d.model import sparse4d as sparse4d_module

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.global_pool = torch.nn.Identity()
            self.fc = torch.nn.Identity()
            self.checkpointing_enabled = None

        def set_grad_checkpointing(self, enabled):
            self.checkpointing_enabled = enabled

    backbone = FakeBackbone()
    monkeypatch.setattr(
        sparse4d_module,
        "SPARSE4D_BACKBONE_REGISTRY",
        {"fake": lambda **kwargs: backbone},
    )
    monkeypatch.setattr(
        sparse4d_module,
        "build_neck",
        lambda config: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        sparse4d_module,
        "build_head",
        lambda config: torch.nn.Identity(),
    )
    model_config = {
        "backbone": {"type": "fake"},
        "neck": {},
        "head": {},
        "depth_branch": None,
        "cotrain_param_touch": parameter_touch,
        "use_grid_mask": False,
        "use_deformable_func": False,
        "sv_aux_head": {"enable": False},
    }

    sparse4d_module.Sparse4D(SimpleNamespace(model=model_config))

    assert backbone.checkpointing_enabled is expected_checkpointing


def _ltt_criterion(tmp_path) -> SetCriterion:
    """Build the real criterion with a valid frozen LTT checkpoint."""
    checkpoint = tmp_path / "ltt.pth"
    class_names = list(WAREHOUSE_V4_CLASSES)
    LooseToTightMLP(
        num_classes=len(class_names),
        hidden_dim=8,
        class_names=class_names,
    ).save(str(checkpoint))
    model_config = OmegaConf.structured(ExperimentConfig()).model
    model_config.head.loose_to_tight.enable = True
    model_config.head.loose_to_tight.pseudo_enable = True
    model_config.head.loose_to_tight.mlp_ckpt = str(checkpoint)
    model_config.head.loose_to_tight.giou_thr = -1.0
    model_config.head.loose_to_tight.min_cams = 1
    model_config.sv_scene_keywords = []
    return SetCriterion(
        model_config,
        instance_bank=None,
        class_names=class_names,
    )


def _encoded_test_box(requires_grad=True) -> torch.Tensor:
    """Return one valid encoded Sparse4D box centered in front of a camera."""
    return torch.tensor(
        [[[0.0, 0.0, 10.0, 0.6931, 0.6931, 0.6931, 0.0, 1.0, 0.0, 0.0, 0.0]]],
        requires_grad=requires_grad,
    )


def _camera_contract() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projection = torch.tensor(
        [
            [
                [
                    [100.0, 0.0, 50.0, 0.0],
                    [0.0, 100.0, 40.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ]
        ]
    )
    return projection, torch.tensor([[[100.0, 80.0]]]), torch.eye(4)[None, None]


def test_real_pseudo_helper_backpropagates_through_box_and_class(tmp_path):
    """The TAO criterion consumes RT-DETR camera lists end to end."""
    criterion = _ltt_criterion(tmp_path)
    reg = _encoded_test_box()
    cls = torch.tensor(
        [[[4.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]]], requires_grad=True
    )
    projection, image_wh, ego_to_camera = _camera_contract()
    data = {
        "det_boxes_2d": [[torch.tensor([[40.0, 30.0, 60.0, 50.0]])]],
        "det_classes_2d": [[torch.tensor([0])]],
        "det_scores_2d": [[torch.tensor([0.9])]],
        "projection_mat": projection,
        "image_wh": image_wh,
        "cam2world_transform": ego_to_camera,
    }

    losses = criterion._loss_2d_pseudo(0, reg, cls, data)
    total = sum(losses.values())

    assert set(losses) == {"loss_box_2d_pseudo_0", "loss_cls_pseudo_0"}
    assert torch.isfinite(total)
    total.backward()
    assert reg.grad is not None and torch.isfinite(reg.grad).all()
    assert cls.grad is not None and torch.isfinite(cls.grad).all()
    assert reg.grad.abs().sum() > 0
    assert cls.grad.abs().sum() > 0


def test_real_pseudo_helper_rejects_out_of_range_class(tmp_path):
    """A cache taxonomy mismatch must not silently discard supervision."""
    criterion = _ltt_criterion(tmp_path)
    reg = _encoded_test_box()
    cls = torch.zeros((1, 1, 7), requires_grad=True)
    projection, image_wh, ego_to_camera = _camera_contract()
    data = {
        "det_boxes_2d": [[torch.tensor([[40.0, 30.0, 60.0, 50.0]])]],
        "det_classes_2d": [[torch.tensor([7])]],
        "det_scores_2d": [[torch.tensor([0.9])]],
        "projection_mat": projection,
        "image_wh": image_wh,
        "cam2world_transform": ego_to_camera,
    }

    with pytest.raises(ValueError, match="outside the detector taxonomy"):
        criterion._loss_2d_pseudo(0, reg, cls, data)


def test_invalid_pseudo_frame_returns_graph_connected_zeros(tmp_path):
    """Missing cache joins skip both pseudo losses without losing the graph."""
    criterion = _ltt_criterion(tmp_path)
    reg = _encoded_test_box()
    cls = torch.zeros((1, 1, 7), requires_grad=True)

    losses = criterion._loss_2d_pseudo(
        0,
        reg,
        cls,
        {"has_2d_pseudo": torch.tensor([False])},
    )
    total = sum(losses.values())

    torch.testing.assert_close(total, torch.zeros_like(total))
    total.backward()
    assert reg.grad is not None and torch.isfinite(reg.grad).all()
    assert cls.grad is not None and torch.isfinite(cls.grad).all()
    torch.testing.assert_close(reg.grad, torch.zeros_like(reg.grad))
    torch.testing.assert_close(cls.grad, torch.zeros_like(cls.grad))


def test_valid_empty_pseudo_frame_supervises_background(tmp_path):
    """An explicitly processed empty frame still trains class background."""
    criterion = _ltt_criterion(tmp_path)
    reg = _encoded_test_box()
    cls = torch.zeros((1, 1, 7), requires_grad=True)
    projection, image_wh, ego_to_camera = _camera_contract()
    data = {
        "det_boxes_2d": [[torch.zeros((0, 4))]],
        "det_classes_2d": [[torch.zeros((0,), dtype=torch.long)]],
        "det_scores_2d": [[torch.zeros((0,))]],
        "projection_mat": projection,
        "image_wh": image_wh,
        "cam2world_transform": ego_to_camera,
        "has_2d_pseudo": torch.tensor([True]),
    }

    losses = criterion._loss_2d_pseudo(0, reg, cls, data)

    assert losses["loss_box_2d_pseudo_0"].item() == 0.0
    assert losses["loss_cls_pseudo_0"].item() > 0.0
    sum(losses.values()).backward()
    assert reg.grad is not None and torch.isfinite(reg.grad).all()
    assert cls.grad is not None and torch.isfinite(cls.grad).all()
    assert cls.grad.abs().sum() > 0


def _normalization_criterion(sync_positive_counts):
    """Build the normal 3D path with observable loss denominators."""
    criterion = _bare_criterion()
    criterion.sync_positive_counts = sync_positive_counts
    criterion.use_reid_sampling = False
    criterion.reg_weights = [1.0]
    criterion.cls_threshold_to_reg = 0.0
    criterion.ltt_enable = False
    criterion.ltt_pseudo_enable = False
    criterion.use_temporal_align = False
    criterion.num_single_frame_decoder = 1
    criterion.ltt_has3dgt_key = "has_3d_gt"
    criterion.sv_scene_keywords = []
    criterion.sv_aux_head = None
    criterion.cls_avg_factors = []
    criterion.reg_avg_factors = []

    class FakeSampler:
        def sample(self, *args, **kwargs):
            return (
                torch.tensor([[0, -1]]),
                torch.tensor([[[1.0], [0.0]]]),
                torch.ones((1, 2, 1)),
                None,
                None,
                None,
                None,
            )

    class RecordingClassificationLoss(torch.nn.Module):
        def forward(self, prediction, target, avg_factor):
            del target
            criterion.cls_avg_factors.append(float(avg_factor))
            return prediction.sum() * 0.0

    class RecordingRegressionLoss(torch.nn.Module):
        valid_vel_weight = 0.0

        def forward(
            self,
            prediction,
            target,
            *,
            avg_factor,
            suffix="",
            **kwargs,
        ):
            del target, kwargs
            criterion.reg_avg_factors.append(float(avg_factor))
            return {f"loss_box{suffix}": prediction.sum() * 0.0}

    criterion.sampler = FakeSampler()
    criterion.loss_cls = RecordingClassificationLoss()
    criterion.loss_reg = RecordingRegressionLoss()
    return criterion


@pytest.mark.parametrize(
    ("sync_positive_counts", "expected_factor", "expected_collectives"),
    [(True, 2.0, 2), (False, 1.0, 0)],
)
def test_main_positive_count_preserves_baseline_global_weighting(
    monkeypatch,
    sync_positive_counts,
    expected_factor,
    expected_collectives,
):
    """Baseline averages counts globally; mixed-route mode stays local."""
    criterion = _normalization_criterion(sync_positive_counts)
    collectives = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(value, op=None):
        del op
        collectives.append(value.detach().clone())
        if value.ndim == 0:
            value.add_(3.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    cls = torch.zeros((1, 2, 1), requires_grad=True)
    reg = torch.zeros((1, 2, 1), requires_grad=True)
    model_outs = {
        "classification": [cls],
        "prediction": [reg],
        "quality": [None],
    }

    losses = criterion(
        (model_outs, None),
        {
            "gt_labels_3d": [torch.tensor([0])],
            "gt_bboxes_3d": [torch.tensor([[1.0]])],
        },
    )

    assert torch.isfinite(sum(losses.values()))
    assert criterion.cls_avg_factors == [expected_factor]
    assert criterion.reg_avg_factors == [expected_factor]
    assert len(collectives) == expected_collectives
    if sync_positive_counts:
        torch.testing.assert_close(collectives[0], torch.tensor(1.0))
        torch.testing.assert_close(collectives[1], torch.tensor([0.0, 0.0]))


def _dn_model_outs():
    """Return one local valid DN item and no temporal DN tensors."""
    return {
        "dn_valid_mask": torch.tensor([[True, False]]),
        "dn_cls_target": torch.tensor([[0, -1]]),
        "dn_reg_target": torch.tensor([[[1.0], [0.0]]]),
    }


def test_dn_positive_counts_use_fixed_participation_global_vector(monkeypatch):
    """Missing local temporal DN still occupies a collective vector slot."""
    criterion = _bare_criterion()
    criterion.sync_positive_counts = True
    criterion.reg_weights = [1.0]
    collectives = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(value, op=None):
        del op
        collectives.append(value.detach().clone())
        value.add_(value.new_tensor([3.0, 2.0]))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    model_outs = _dn_model_outs()
    factors = criterion._synchronized_dn_positive_counts(
        model_outs,
        torch.zeros((1, 1, 1)),
    )
    prepared = criterion.prepare_for_dn_loss(
        model_outs,
        avg_factor=factors[""],
    )

    assert len(collectives) == 1
    torch.testing.assert_close(collectives[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(factors[""], torch.tensor(2.0))
    torch.testing.assert_close(factors["temp_"], torch.tensor(1.0))
    torch.testing.assert_close(prepared[-1], torch.tensor(2.0))


def test_mixed_route_dn_positive_count_is_local_and_collective_free(monkeypatch):
    """Active unsynchronized routes cannot enter a DN-only collective."""
    criterion = _bare_criterion()
    criterion.sync_positive_counts = False
    criterion.reg_weights = [1.0]

    def fail_collective(*args, **kwargs):
        del args, kwargs
        raise AssertionError("mixed-route DN path issued a collective")

    monkeypatch.setattr(torch.distributed, "all_reduce", fail_collective)
    model_outs = _dn_model_outs()

    assert (
        criterion._synchronized_dn_positive_counts(
            model_outs,
            torch.zeros((1, 1, 1)),
        )
        == {}
    )
    prepared = criterion.prepare_for_dn_loss(model_outs)
    torch.testing.assert_close(prepared[-1], torch.tensor(1.0))


def test_synthetic_ltt_helper_backpropagates_through_matched_box(tmp_path):
    """The TAO criterion joins matched instance IDs to visible 2D targets."""
    criterion = _ltt_criterion(tmp_path)
    reg = _encoded_test_box()
    cls = torch.tensor([[[4.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]]])
    projection, image_wh, ego_to_camera = _camera_contract()
    data = {
        "gt_boxes_2d_visible": [torch.tensor([[[40.0, 30.0, 60.0, 50.0]]])],
        "gt_occ_weight": [torch.tensor([[1.0]])],
        "instance_id": [torch.tensor([7])],
        "projection_mat": projection,
        "image_wh": image_wh,
        "cam2world_transform": ego_to_camera,
    }

    losses = criterion._loss_box_2d(
        0,
        reg,
        cls,
        torch.tensor([[True]]),
        torch.tensor([[7]]),
        data,
        torch.tensor(1.0),
    )
    loss = losses["loss_box_2d_0"]

    assert torch.isfinite(loss)
    loss.backward()
    assert reg.grad is not None and torch.isfinite(reg.grad).all()
    assert reg.grad.abs().sum() > 0


def _loss(value: float) -> torch.Tensor:
    """Create a scalar leaf loss."""
    return torch.tensor(value, requires_grad=True)


def _gloo_sync_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_dir: str,
) -> None:
    """Exercise canonical ordering and graph-connected fill on one Gloo rank."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _bare_lightning_model()
        def unexpected_collective(*_args, **_kwargs):
            raise AssertionError("loss-key filling must not use an object collective")
        dist.all_gather_object = unexpected_collective

        if rank == 0:
            same_keys = {"loss_z": _loss(2.0), "loss_a": _loss(1.0)}
        else:
            same_keys = {"loss_a": _loss(4.0), "loss_z": _loss(5.0)}
        same_synced = model._sync_cotrain_loss_keys(
            same_keys,
            _loss(rank + 1.0) * 0.0,
        )

        zero_source = _loss(rank + 10.0)
        zero_ref = zero_source * 0.0
        if rank == 0:
            heterogeneous = {"loss_z": _loss(2.0), "loss_a": _loss(1.0)}
            missing_key = "loss_m"
        else:
            heterogeneous = {"loss_m": _loss(3.0), "loss_a": _loss(4.0)}
            missing_key = "loss_z"
        heterogeneous_synced = model._sync_cotrain_loss_keys(
            heterogeneous,
            zero_ref,
        )
        sum(heterogeneous_synced.values()).backward()

        result = {
            "same_keys": list(same_synced),
            "heterogeneous_keys": list(heterogeneous_synced),
            "missing_value": heterogeneous_synced[missing_key].item(),
            "missing_is_zero_ref": heterogeneous_synced[missing_key] is zero_ref,
            "zero_source_grad": zero_source.grad.item(),
        }
        Path(result_dir, f"rank{rank}.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="the distributed Gloo backend is unavailable",
)
def test_cotrain_loss_keys_are_canonical_across_two_ranks(tmp_path):
    """Two ranks agree on loss-key order and fill route-specific keys safely."""
    world_size = 2
    mp.spawn(
        _gloo_sync_worker,
        args=(world_size, str(tmp_path / "gloo_init"), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    results = [
        json.loads((tmp_path / f"rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    for result in results:
        assert result["same_keys"] == ["loss_a", "loss_m", "loss_z"]
        assert result["heterogeneous_keys"] == ["loss_a", "loss_m", "loss_z"]
        assert result["missing_value"] == 0.0
        assert result["missing_is_zero_ref"]
        assert result["zero_source_grad"] == 0.0
