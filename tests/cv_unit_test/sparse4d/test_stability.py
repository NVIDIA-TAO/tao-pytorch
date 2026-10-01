# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Sparse4D numerical and output-format stability helpers."""

import threading

import pytest
import torch

from nvidia_tao_pytorch.config.sparse4d.inference import (
    Sparse4DInferenceConfig,
)
from nvidia_tao_pytorch.config.sparse4d.train import Sparse4DTrainConfig
from nvidia_tao_pytorch.cv.sparse4d.utils.stability import (
    nvschema_fps_override,
    scrub_nan_gradients,
)


def test_scrub_nan_gradients_preserves_infinities_and_sparse_layout():
    """NaNs are dropped while overflow sentinels remain visible to AMP."""
    dense = torch.nn.Parameter(torch.zeros(4))
    dense.grad = torch.tensor([float("nan"), float("inf"), -float("inf"), 3.0])

    sparse = torch.nn.Parameter(torch.zeros(4))
    sparse.grad = torch.sparse_coo_tensor(
        torch.tensor([[0, 2]]),
        torch.tensor([float("nan"), float("inf")]),
        size=(4,),
    )
    unused = torch.nn.Parameter(torch.zeros(1))

    assert scrub_nan_gradients([dense, sparse, unused]) == 2
    torch.testing.assert_close(
        dense.grad,
        torch.tensor([0.0, float("inf"), -float("inf"), 3.0]),
    )
    torch.testing.assert_close(
        sparse.grad._values(),
        torch.tensor([0.0, float("inf")]),
    )


def test_nvschema_fps_override_is_scoped_and_restored_on_error():
    """The shared converter constant is restored even if conversion fails."""
    from spatialai_data_utils.converters import nusc_results_to_nvschema

    original_fps = nusc_results_to_nvschema.FPS
    with pytest.raises(RuntimeError, match="conversion failed"):
        with nvschema_fps_override(12.5):
            assert nusc_results_to_nvschema.FPS == 12.5
            raise RuntimeError("conversion failed")
    assert nusc_results_to_nvschema.FPS == original_fps

    with nvschema_fps_override(0.0):
        assert nusc_results_to_nvschema.FPS == original_fps


def test_nvschema_fps_override_serializes_overlapping_threads():
    """Concurrent conversions cannot observe or restore another override."""
    from spatialai_data_utils.converters import nusc_results_to_nvschema

    original_fps = nusc_results_to_nvschema.FPS
    first_entered = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    observations = []

    def first_conversion():
        with nvschema_fps_override(10.0):
            observations.append(("first", nusc_results_to_nvschema.FPS))
            first_entered.set()
            assert release_first.wait(timeout=5.0)

    def second_conversion():
        assert first_entered.wait(timeout=5.0)
        second_attempted.set()
        with nvschema_fps_override(0.0):
            observations.append(("second", nusc_results_to_nvschema.FPS))
            second_entered.set()

    first = threading.Thread(target=first_conversion)
    second = threading.Thread(target=second_conversion)
    first.start()
    second.start()
    assert first_entered.wait(timeout=5.0)
    assert second_attempted.wait(timeout=5.0)
    assert not second_entered.wait(timeout=0.2)
    release_first.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert observations == [("first", 10.0), ("second", original_fps)]
    assert nusc_results_to_nvschema.FPS == original_fps


@pytest.mark.parametrize("fps", [-1.0, float("nan"), float("inf")])
def test_nvschema_fps_override_rejects_invalid_values(fps):
    """Only zero-as-default or a finite positive override is accepted."""
    with pytest.raises(ValueError, match="finite and positive"):
        with nvschema_fps_override(fps):
            pass


def test_stability_options_are_disabled_by_default():
    """Released Sparse4D behavior remains unchanged without explicit opt-in."""
    assert Sparse4DTrainConfig().scrub_nan_gradients is False
    assert Sparse4DInferenceConfig().nvschema_fps == 0.0
