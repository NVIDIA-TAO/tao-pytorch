# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for SegFormer optimizer and scheduler wiring."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("pytorch_lightning")
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel


def make_optimizer_only_model(policy="cosine", momentum=0.83):
    model = SegFormerPlModel.__new__(SegFormerPlModel)
    torch.nn.Module.__init__(model)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    model.lr = 1e-3
    model.max_epochs = 10
    model.lr_policy = policy
    model.monitor_name = "val_miou"
    model.optimizer = SimpleNamespace(
        optim="adamw", momentum=momentum, weight_decay=0.01
    )
    model._get_parameter_groups = lambda lr, weight_decay: [
        {"params": [parameter], "lr": lr, "weight_decay": weight_decay}
    ]
    model._trainer = SimpleNamespace(estimated_stepping_batches=100)
    return model


def test_adamw_uses_configured_beta1_and_cosine_scheduler():
    model = make_optimizer_only_model()
    configured = model.configure_optimizers()

    assert configured["optimizer"].defaults["betas"] == (0.83, 0.999)
    schedule = configured["lr_scheduler"]["scheduler"]
    assert configured["lr_scheduler"]["interval"] == "step"
    assert schedule.lr_lambdas[0](50) < schedule.lr_lambdas[0](10)


def test_unknown_policy_fails_instead_of_returning_an_exception_object():
    with pytest.raises(NotImplementedError, match="unknown"):
        make_optimizer_only_model(policy="unknown").configure_optimizers()
