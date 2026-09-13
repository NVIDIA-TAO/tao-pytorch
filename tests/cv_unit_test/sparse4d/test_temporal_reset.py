# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Sparse4D temporal boundary resets."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nvidia_tao_pytorch.cv.sparse4d.model.instance_bank import InstanceBank
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d_head import Sparse4DHead


def _bank(reset_on_time_gap=False):
    """Build a tiny InstanceBank without filesystem-backed anchors."""
    return InstanceBank(
        num_anchor=2,
        embed_dims=3,
        anchor=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        num_temp_instances=1,
        max_time_interval=1.0,
        reset_on_time_gap=reset_on_time_gap,
    )


def _populate_temporal_state(bank, batch_size=2):
    """Populate every recurrent field with a recognizable value."""
    bank.cached_feature = torch.ones(batch_size, 1, bank.embed_dims)
    bank.cached_anchor = torch.ones(batch_size, 1, bank.anchor.shape[-1])
    bank.metas = {"timestamp": torch.zeros(batch_size)}
    bank.mask = torch.ones(batch_size, dtype=torch.bool)
    bank.confidence = torch.ones(batch_size, 1)
    bank.temp_confidence = torch.ones(batch_size, bank.num_anchor)
    bank.instance_id = torch.arange(batch_size).reshape(batch_size, 1)
    bank.gt_index_mapping = [{0: (0, 10)} for _ in range(batch_size)]
    bank.cached_gt_index_mapping = [{0: (0, 10)} for _ in range(batch_size)]
    bank.cached_query_indices = [torch.tensor([0]) for _ in range(batch_size)]


def _bare_head(bank):
    """Build only the state needed by the boundary-reset helper."""
    head = Sparse4DHead.__new__(Sparse4DHead)
    nn.Module.__init__(head)
    head.instance_bank = bank
    head.sampler = SimpleNamespace(dn_metas={"dn_anchor": torch.ones(1)})
    return head


def _assert_temporal_state_cleared(bank):
    for name in (
        "cached_feature",
        "cached_anchor",
        "metas",
        "mask",
        "confidence",
        "temp_confidence",
        "instance_id",
        "gt_index_mapping",
        "cached_gt_index_mapping",
        "cached_query_indices",
    ):
        assert getattr(bank, name) is None


@pytest.mark.parametrize("changed_key", ["group_idx", "scene_idx"])
def test_head_resets_all_temporal_state_on_any_data_boundary(changed_key):
    """A boundary in one batch slot invalidates batch-wide and DN caches."""
    bank = _bank()
    bank.prev_id = 41
    bank.set_data_indices([3, 3], [7, 7])
    _populate_temporal_state(bank)
    head = _bare_head(bank)
    metas = {"group_idx": [3, 3], "scene_idx": [7, 7]}
    metas[changed_key][1] += 1

    assert head._reset_at_data_boundary(metas)

    _assert_temporal_state_cleared(bank)
    assert head.sampler.dn_metas is None
    assert bank.prev_id == 41
    assert bank.get_data_indices() == (
        metas["group_idx"],
        metas["scene_idx"],
    )


def test_head_supports_legacy_nested_metadata_without_false_reset():
    """Legacy img_metas indices are tracked and equal indices retain caches."""
    bank = _bank()
    bank.prev_id = 9
    bank.set_data_indices([2], [5])
    _populate_temporal_state(bank, batch_size=1)
    head = _bare_head(bank)

    assert not head._reset_at_data_boundary(
        {"img_metas": [{"group_idx": 2, "scene_idx": 5}]}
    )
    assert bank.cached_feature is not None
    assert head.sampler.dn_metas is not None
    assert bank.prev_id == 9


def test_time_gap_uses_temporal_reset_without_recycling_instance_ids():
    """Timestamp discontinuities clear recurrent state but preserve IDs."""
    bank = _bank(reset_on_time_gap=True)
    bank.prev_id = 73
    bank.set_data_indices([4, 4], [8, 8])
    _populate_temporal_state(bank)

    outputs = bank.get(2, {"timestamp": torch.full((2,), 3.0)})

    _assert_temporal_state_cleared(bank)
    assert bank.prev_id == 73
    assert bank.get_data_indices() == ([4, 4], [8, 8])
    assert outputs[2] is None
    assert outputs[3] is None
    torch.testing.assert_close(outputs[4], torch.full((2,), 0.5))
