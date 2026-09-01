# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightning Trainer tests for both NVPanoptix3Dv2 variants."""

from types import MethodType

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nvidia_tao_pytorch.config.nvpanoptix3d_v2.default_config import (
    ExperimentConfig,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.panoptic import (
    pl_model as panoptic_pl,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.reasoning import (
    pl_model as reasoning_pl,
)


class _SingleBatchDataset(Dataset):
    """Dataset containing one already-batched multi-view sample."""

    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        """Return one trainer batch."""
        return 1

    def __getitem__(self, index):
        """Return the synthetic sample."""
        del index
        return self.sample


def _first_sample(samples):
    """Remove the DataLoader's outer list without recollating tensors."""
    return samples[0]


class _SingleBatchDataModule(pl.LightningDataModule):
    """Expose one batch to every Lightning lifecycle."""

    def __init__(self, sample):
        super().__init__()
        self.dataset = _SingleBatchDataset(sample)

    def _loader(self):
        """Build a fresh single-process loader."""
        return DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=_first_sample,
        )

    def train_dataloader(self):
        """Return the training loader."""
        return self._loader()

    def val_dataloader(self):
        """Return the validation loader."""
        return self._loader()

    def test_dataloader(self):
        """Return the evaluation loader."""
        return self._loader()

    def predict_dataloader(self):
        """Return the inference loader."""
        return self._loader()


class _TinyPanopticDecoder(nn.Module):
    """Trainable decoder surface required by the optimizer."""

    label_mode = "sigmoid"

    def __init__(self):
        super().__init__()
        self.score = nn.Parameter(torch.tensor(2.0))


class _TinyPanopticModel(nn.Module):
    """Small differentiable replacement for VGGT and the panoptic decoder."""

    def __init__(self):
        super().__init__()
        self.panoptic_decoder = _TinyPanopticDecoder()
        self.metric_depth_head = None
        self.classes = []

    def freeze_vggt_weights(self):
        """No-op because the stand-in has no backbone parameters."""

    def set_vocab(self, classes, device=None):
        """Record the configured class vocabulary."""
        del device
        self.classes = list(classes)

    def forward(self, images, true_shape, classes):
        """Return one confident query connected to the decoder parameter."""
        del true_shape
        batch, views, _, height, width = images.shape
        class_count = len(classes or self.classes)
        score = self.panoptic_decoder.score
        return {
            "pred_logits": score.expand(batch, 1, class_count),
            "pred_masks": score.expand(
                batch,
                views,
                1,
                height,
                width,
            ),
        }, {}


class _TinyPanopticCriterion(nn.Module):
    """Return a differentiable scalar from the fake decoder output."""

    def forward(
        self,
        batch,
        panoptic,
        classes,
        geometry_output=None,
        gt_depth=None,
    ):
        """Compute the synthetic training loss."""
        del batch, classes, geometry_output, gt_depth
        loss = panoptic["pred_logits"].mean()
        return loss, {"loss_total": loss.detach()}


class _TinyReasoningModel(nn.Module):
    """Single-parameter reasoning model used by its optimizer."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))


class _UnusedCriterion(nn.Module):
    """Criterion placeholder; forward_batch is replaced in these tests."""

    def forward(self, *args, **kwargs):
        """Fail if the trainer crosses the intended test boundary."""
        raise AssertionError("The trainer test supplies forward_batch directly")


def _experiment_config(tmp_path, variant):
    """Build a minimal structured config for one trainer lifecycle."""
    config = OmegaConf.structured(ExperimentConfig())
    results_dir = tmp_path / variant
    results_dir.mkdir()
    config.results_dir = str(results_dir)
    config.model.model_type = variant
    config.train.num_epochs = 1
    config.train.num_gpus = 1
    config.train.num_nodes = 1
    config.train.checkpointer.enable_topk = False
    config.inference.output_dir = str(results_dir / "predictions")
    return config


def _trainer(root):
    """Build a deterministic CPU trainer that executes one batch."""
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        fast_dev_run=True,
        default_root_dir=str(root),
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        use_distributed_sampler=False,
    )


def _panoptic_sample():
    """Return one two-view panoptic trainer batch."""
    views = []
    for view_index in range(2):
        view = {
            "img": torch.zeros(1, 3, 16, 16),
            "true_shape": torch.tensor([[16, 16]]),
            "pan_inst_id": torch.full(
                (1, 16, 16),
                7,
                dtype=torch.long,
            ),
            "pan_cls_id": torch.zeros(
                1,
                16,
                16,
                dtype=torch.long,
            ),
            "label": ["scene_000"],
        }
        if view_index == 0:
            view["vocab"] = ["chair"]
        views.append(view)
    return views


def _reasoning_sample():
    """Return one reasoning trainer batch."""
    return [
        {
            "img": torch.zeros(1, 3, 4, 4),
            "pan_inst_id": torch.full(
                (1, 4, 4),
                7,
                dtype=torch.long,
            ),
            "depthmap": torch.ones(1, 4, 4),
            "instruction": ["Segment the chair."],
            "target_inst_id": torch.tensor([7]),
        }
    ]


def _reasoning_forward_batch(module, batch):
    """Return differentiable loss and perfect synthetic predictions."""
    loss = module.model.weight.square()
    logs = {
        "loss_total": loss.detach(),
        "loss_text": loss.detach(),
        "loss_mask": torch.zeros_like(loss),
        "loss_dice": torch.zeros_like(loss),
        "loss_score": torch.zeros_like(loss),
        "n_mask_valid": torch.ones_like(loss),
    }
    height, width = batch[0]["pan_inst_id"].shape[-2:]
    output = {
        "seg_sample_idx": torch.tensor(
            [0],
            device=module.device,
        ),
        "seg_view_idx": torch.tensor(
            [0],
            device=module.device,
        ),
        "pred_masks": torch.full(
            (1, 1, height, width),
            10.0,
            device=module.device,
        ),
        "pred_logits": torch.full(
            (1, 1),
            10.0,
            device=module.device,
        ),
        "point_mask_prob": torch.ones(
            1,
            height,
            width,
            device=module.device,
        ),
        "segmented_point_clouds": [
            torch.zeros(1, 3, device=module.device)
        ],
    }
    return loss, logs, output


@pytest.mark.cv_unit
@pytest.mark.train
@pytest.mark.evaluate
@pytest.mark.infer
def test_panoptic_trainer(monkeypatch, tmp_path):
    """The panoptic module completes fit, test, and predict lifecycles."""
    monkeypatch.setattr(
        panoptic_pl,
        "build_model_from_config",
        lambda config: _TinyPanopticModel(),
    )
    monkeypatch.setattr(
        panoptic_pl,
        "build_criterion_from_config",
        lambda config: _TinyPanopticCriterion(),
    )
    config = _experiment_config(tmp_path, "panoptic")
    module = panoptic_pl.NVPanoptix3Dv2PanopticPlModule(config)
    module.set_classes(
        ["chair"],
        [{"id": 0, "name": "chair", "isthing": True}],
    )
    data_module = _SingleBatchDataModule(_panoptic_sample())
    initial_score = module.model.panoptic_decoder.score.detach().clone()

    fit_trainer = _trainer(config.results_dir)
    fit_trainer.fit(module, datamodule=data_module)
    assert fit_trainer.global_step == 1
    assert not torch.equal(
        module.model.panoptic_decoder.score.detach(),
        initial_score,
    )

    _trainer(config.results_dir).test(
        module,
        datamodule=data_module,
    )
    assert (tmp_path / "panoptic" / "instance_map.json").is_file()

    _trainer(config.results_dir).predict(
        module,
        datamodule=data_module,
    )
    assert list(
        (tmp_path / "panoptic" / "predictions").glob("*.npz")
    )


@pytest.mark.cv_unit
@pytest.mark.train
@pytest.mark.evaluate
@pytest.mark.infer
def test_reasoning_trainer(monkeypatch, tmp_path):
    """The reasoning module completes fit, test, and predict lifecycles."""
    monkeypatch.setattr(
        reasoning_pl,
        "build_reasoning_model_from_config",
        lambda config: _TinyReasoningModel(),
    )
    monkeypatch.setattr(
        reasoning_pl,
        "build_sam_criterion",
        lambda config: _UnusedCriterion(),
    )
    config = _experiment_config(tmp_path, "reasoning")
    module = reasoning_pl.NVPanoptix3Dv2ReasoningPlModule(config)
    module.forward_batch = MethodType(
        _reasoning_forward_batch,
        module,
    )
    data_module = _SingleBatchDataModule(_reasoning_sample())
    initial_weight = module.model.weight.detach().clone()

    fit_trainer = _trainer(config.results_dir)
    fit_trainer.fit(module, datamodule=data_module)
    assert fit_trainer.global_step == 1
    assert not torch.equal(
        module.model.weight.detach(),
        initial_weight,
    )

    _trainer(config.results_dir).test(
        module,
        datamodule=data_module,
    )
    metrics = tmp_path / "reasoning" / "reasoning_point_cloud_metrics.json"
    assert metrics.is_file()

    _trainer(config.results_dir).predict(
        module,
        datamodule=data_module,
    )
    assert list(
        (tmp_path / "reasoning" / "predictions").glob("*.npz")
    )
