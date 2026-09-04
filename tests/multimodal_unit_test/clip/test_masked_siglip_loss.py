# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for metadata-masked SigLIP loss."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from nvidia_tao_pytorch.multimodal.clip.loss import (
    masked_siglip_loss as loss_module,
)
from nvidia_tao_pytorch.multimodal.clip.loss.masked_siglip_loss import (
    MetadataMaskedSigLipLoss,
)


def _features():
    """Build deterministic local image/text features."""
    image_features = torch.tensor([
        [0.10, 0.20, 0.30],
        [0.30, 0.20, 0.10],
        [0.20, 0.40, 0.10],
    ])
    text_features = torch.tensor([
        [0.20, 0.10, 0.30],
        [0.10, 0.30, 0.20],
        [0.40, 0.10, 0.20],
    ])
    return image_features, text_features


def _manual_siglip_loss(
    image_features,
    text_features,
    logit_scale,
    logit_bias,
    valid_terms=None,
):
    """Compute SigLIP loss terms directly for expected values."""
    batch_size = image_features.shape[0]
    logits = logit_scale * image_features @ text_features.T + logit_bias
    labels = -torch.ones_like(logits)
    labels = labels + 2 * torch.eye(batch_size, dtype=logits.dtype)
    loss_terms = -F.logsigmoid(labels * logits)
    if valid_terms is not None:
        loss_terms = loss_terms.masked_select(valid_terms)
    return loss_terms.sum() / batch_size


def _manual_siglip_loss_rectangular(
    image_features,
    text_features,
    logit_scale,
    logit_bias,
    positive_text_indices,
    valid_terms,
):
    """Compute SigLIP loss for local images against gathered text rows."""
    batch_size = image_features.shape[0]
    logits = logit_scale * image_features @ text_features.T + logit_bias
    labels = -torch.ones_like(logits)
    labels[
        torch.arange(batch_size),
        positive_text_indices,
    ] = 1
    loss_terms = -F.logsigmoid(labels * logits).masked_select(valid_terms)
    return loss_terms.sum() / batch_size


def _mock_distributed_context(monkeypatch, rank=0, world_size=2):
    """Mock a live default process group without launching worker processes."""
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda: world_size
    )


def _run_two_rank_gloo_gradient_check(rank, world_size, init_method):
    """Verify gathered text features receive gradients from every rank."""
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        image_features = torch.tensor([
            [0.40, -0.20],
            [0.10, 0.50],
        ])
        all_text_features = torch.tensor([
            [0.30, 0.70],
            [-0.60, 0.20],
        ])
        attr_values = torch.tensor([[1], [2]])
        logit_scale = torch.tensor(1.7)
        logit_bias = torch.tensor(-0.2)

        local_text_features = (
            all_text_features[rank:rank + 1].clone().requires_grad_()
        )
        loss = MetadataMaskedSigLipLoss(
            dist_impl="gather",
            world_size=world_size,
            rank=rank,
        )(
            image_features[rank:rank + 1],
            local_text_features,
            logit_scale,
            logit_bias,
            image_attr_values=attr_values[rank:rank + 1],
            text_attr_values=attr_values[rank:rank + 1],
        )
        loss.backward()
        actual_grad = local_text_features.grad.detach()

        reference_text_features = all_text_features.clone().requires_grad_()
        per_rank_losses = []
        for image_rank in range(world_size):
            logits = logit_scale * (
                image_features[image_rank:image_rank + 1]
                @ reference_text_features.T
            ) + logit_bias
            labels = -torch.ones_like(logits)
            labels[0, image_rank] = 1
            per_rank_losses.append(-F.logsigmoid(labels * logits).sum())

        expected_grad = torch.autograd.grad(
            sum(per_rank_losses),
            reference_text_features,
            retain_graph=True,
        )[0][rank:rank + 1]
        local_only_grad = torch.autograd.grad(
            per_rank_losses[rank],
            reference_text_features,
        )[0][rank:rank + 1]

        assert torch.allclose(actual_grad, expected_grad)
        assert not torch.allclose(actual_grad, local_only_grad)
    finally:
        dist.destroy_process_group()


def _run_two_rank_gloo_uneven_batch_check(rank, world_size, init_method):
    """Verify uneven local batches fail before feature/metadata gathers."""
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        local_batch_size = rank + 1
        image_features = torch.ones(local_batch_size, 2)
        text_features = torch.ones(local_batch_size, 2)
        attr_values = torch.arange(local_batch_size).reshape(-1, 1)

        with pytest.raises(
            RuntimeError,
            match="equal local text batch sizes",
        ):
            MetadataMaskedSigLipLoss(
                dist_impl="gather",
                world_size=world_size,
                rank=rank,
            )(
                image_features,
                text_features,
                torch.tensor(1.0),
                torch.tensor(0.0),
                image_attr_values=attr_values,
                text_attr_values=attr_values,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.multimodal_unit
class TestMetadataMaskedSigLipLoss:
    """Test metadata-masked SigLIP loss behavior."""

    def test_matches_regular_siglip_without_off_diagonal_metadata_matches(
        self,
    ):
        """Test no mask is applied when only paired metadata matches."""
        image_features, text_features = _features()
        logit_scale = torch.tensor(2.0)
        logit_bias = torch.tensor(-0.5)
        attr_values = torch.tensor([
            [1, 1],
            [2, 2],
            [3, 3],
        ])

        loss = MetadataMaskedSigLipLoss()(
            image_features,
            text_features,
            logit_scale,
            logit_bias,
            image_attr_values=attr_values,
            text_attr_values=attr_values,
        )

        expected = _manual_siglip_loss(
            image_features,
            text_features,
            logit_scale,
            logit_bias,
        )
        assert torch.allclose(loss, expected)

    def test_legacy_scalar_mode_ignores_optional_accessory_tensors(self):
        """Test accessory support does not change attribute_match_ignore."""
        attr_values = torch.tensor([
            [1, 1],
            [1, 1],
            [3, 3],
        ])
        loss = MetadataMaskedSigLipLoss(accessory_aware=False)

        legacy_mask = loss.get_valid_term_mask(
            image_attr_values=attr_values,
            text_attr_values=attr_values,
        )
        mask_with_unused_accessories = loss.get_valid_term_mask(
            image_attr_values=attr_values,
            text_attr_values=attr_values,
            image_accessory_ids=torch.tensor([[11], [12], [13]]),
            text_accessory_ids=torch.tensor([[11], [12], [13]]),
        )

        assert torch.equal(legacy_mask, mask_with_unused_accessories)

    def test_ignores_off_diagonal_metadata_match(self):
        """Test metadata-compatible off-diagonal terms are removed."""
        image_features, text_features = _features()
        logit_scale = torch.tensor(2.0)
        logit_bias = torch.tensor(-0.5)
        image_attr_values = torch.tensor([
            [1, 1],
            [2, 2],
            [1, 3],
        ])
        text_attr_values = torch.tensor([
            [1, 1],
            [2, 2],
            [1, -1],
        ])

        loss = MetadataMaskedSigLipLoss()(
            image_features,
            text_features,
            logit_scale,
            logit_bias,
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
        )

        valid_terms = torch.tensor([
            [True, True, False],
            [True, True, True],
            [True, True, True],
        ])
        expected = _manual_siglip_loss(
            image_features,
            text_features,
            logit_scale,
            logit_bias,
            valid_terms=valid_terms,
        )
        assert torch.allclose(loss, expected)

    def test_keeps_diagonal_positive_even_when_metadata_matches(self):
        """Test diagonal positives remain valid when metadata matches."""
        attr_values = torch.tensor([
            [1, 1],
            [1, 1],
            [1, 1],
        ])

        valid_terms = MetadataMaskedSigLipLoss().get_valid_term_mask(
            image_attr_values=attr_values,
            text_attr_values=attr_values,
        )

        assert torch.equal(
            valid_terms,
            torch.eye(3, dtype=torch.bool),
        )

    def test_positive_mode_builds_targets_mask_and_per_pair_weights(self):
        """Test compatible off-diagonals are weighted, not paired positives."""
        labels = torch.tensor([
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ])
        metadata_match = torch.tensor([
            [True, True, False],
            [False, True, False],
            [True, False, True],
        ])
        valid_terms = ~metadata_match | torch.eye(3, dtype=torch.bool)
        loss_fn = MetadataMaskedSigLipLoss(
            compatible_as_positive=True,
            compatible_positive_weight=0.5,
        )

        targets, valid, weights, promoted = (
            loss_fn.get_targets_and_term_weights(
                labels=labels,
                valid_terms=valid_terms,
                metadata_match=metadata_match,
                positive_text_indices=torch.arange(3),
                query_has_evidence=torch.ones(3, dtype=torch.bool),
            )
        )

        expected_promoted = torch.tensor([
            [False, True, False],
            [False, False, False],
            [True, False, False],
        ])
        expected_targets = torch.tensor([
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
        ])
        expected_weights = torch.tensor([
            [1.0, 0.5, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 1.0, 1.0],
        ])
        assert torch.equal(promoted, expected_promoted)
        assert torch.equal(targets, expected_targets)
        assert torch.all(valid)
        assert torch.equal(weights, expected_weights)
        assert torch.equal(targets.diag(), torch.ones(3))
        assert torch.equal(weights.diag(), torch.ones(3))

    def test_positive_mode_per_query_normalizes_total_promoted_weight(self):
        """Test each query shares one configured weight across its positives."""
        labels = -torch.ones(3, 4)
        positive_text_indices = torch.tensor([0, 1, 2])
        labels[torch.arange(3), positive_text_indices] = 1
        metadata_match = torch.tensor([
            [True, True, True, False],
            [True, True, False, True],
            [True, False, True, True],
        ])
        loss_fn = MetadataMaskedSigLipLoss(
            compatible_as_positive=True,
            compatible_positive_weight=0.6,
            compatible_positive_normalization="per_query",
        )

        _, _, weights, promoted = loss_fn.get_targets_and_term_weights(
            labels=labels,
            valid_terms=~metadata_match,
            metadata_match=metadata_match,
            positive_text_indices=positive_text_indices,
            query_has_evidence=torch.ones(4, dtype=torch.bool),
        )

        expected_promoted_weights = torch.tensor([
            [0.0, 0.6, 0.6, 0.0],
            [0.3, 0.0, 0.0, 0.3],
            [0.3, 0.0, 0.0, 0.3],
        ])
        assert torch.allclose(weights * promoted, expected_promoted_weights)
        assert torch.allclose(
            (weights * promoted).sum(dim=0),
            torch.full((4,), 0.6),
        )

    def test_positive_mode_with_no_compatible_pairs_keeps_diagonal(self):
        """Test zero promoted pairs are safe and retain paired positives."""
        image_features, text_features = _features()
        attr_values = torch.tensor([[1], [2], [3]])
        loss_fn = MetadataMaskedSigLipLoss(
            compatible_as_positive=True,
            compatible_positive_weight=0.5,
            compatible_positive_normalization="per_query",
        )

        actual = loss_fn(
            image_features,
            text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
            image_attr_values=attr_values,
            text_attr_values=attr_values,
        )

        expected = _manual_siglip_loss(
            image_features,
            text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
        )
        assert torch.allclose(actual, expected)
        assert loss_fn.last_compatible_positive_pairs.item() == 0

    def test_positive_mode_metadata_free_queries_keep_only_diagonal(self):
        """Test unconstrained queries never promote vacuous metadata matches."""
        image_features, text_features = _features()
        image_attr_values = torch.tensor([[1], [2], [3]])
        text_attr_values = -torch.ones((3, 1), dtype=torch.long)
        image_accessory_ids = torch.zeros((3, 1), dtype=torch.long)
        text_accessory_ids = torch.zeros((3, 1), dtype=torch.long)
        loss_fn = MetadataMaskedSigLipLoss(
            accessory_aware=True,
            compatible_as_positive=True,
            compatible_positive_weight=0.5,
            compatible_positive_normalization="per_query",
        )

        metadata_match = loss_fn.get_metadata_match_mask(
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
            image_accessory_ids=image_accessory_ids,
            text_accessory_ids=text_accessory_ids,
        )
        query_has_evidence = loss_fn.get_query_evidence_mask(
            text_attr_values=text_attr_values,
            text_accessory_ids=text_accessory_ids,
        )
        labels = loss_fn.get_ground_truth(
            device=image_features.device,
            dtype=image_features.dtype,
            batch_size=3,
        )
        valid_terms = loss_fn.get_valid_term_mask(
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
            image_accessory_ids=image_accessory_ids,
            text_accessory_ids=text_accessory_ids,
        )
        targets, valid, weights, promoted = (
            loss_fn.get_targets_and_term_weights(
                labels=labels,
                valid_terms=valid_terms,
                metadata_match=metadata_match,
                positive_text_indices=torch.arange(3),
                query_has_evidence=query_has_evidence,
            )
        )

        assert torch.all(metadata_match)
        assert not torch.any(query_has_evidence)
        assert not torch.any(promoted)
        assert torch.equal(targets, labels)
        assert torch.equal(valid, torch.eye(3, dtype=torch.bool))
        assert torch.equal(weights, torch.ones_like(labels))

        actual = loss_fn(
            image_features,
            text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
            image_accessory_ids=image_accessory_ids,
            text_accessory_ids=text_accessory_ids,
        )
        expected = _manual_siglip_loss(
            image_features,
            text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
            valid_terms=torch.eye(3, dtype=torch.bool),
        )
        assert torch.allclose(actual, expected)
        assert loss_fn.last_compatible_positive_pairs.item() == 0

    def test_ignore_mode_keeps_existing_targets_mask_and_weights(self):
        """Test normalized-positive controls do not alter legacy ignore mode."""
        labels = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        metadata_match = torch.ones(2, 2, dtype=torch.bool)
        valid_terms = torch.eye(2, dtype=torch.bool)
        loss_fn = MetadataMaskedSigLipLoss(
            compatible_as_positive=False,
            compatible_positive_weight=0.5,
            compatible_positive_normalization="per_query",
        )

        targets, valid, weights, promoted = (
            loss_fn.get_targets_and_term_weights(
                labels=labels,
                valid_terms=valid_terms,
                metadata_match=metadata_match,
                positive_text_indices=torch.arange(2),
            )
        )

        assert targets is labels
        assert valid is valid_terms
        assert torch.equal(weights, torch.ones_like(labels))
        assert not torch.any(promoted)

    def test_unknown_image_metadata_keeps_paired_positive(self):
        """Test metadata uncertainty never changes the paired label."""
        image_features, text_features = _features()
        image_features = image_features[:2]
        text_features = text_features[:2]
        logit_scale = torch.tensor(2.0)
        logit_bias = torch.tensor(-0.5)

        loss = MetadataMaskedSigLipLoss()(
            image_features,
            text_features,
            logit_scale,
            logit_bias,
            image_attr_values=torch.tensor([[-1], [2]]),
            text_attr_values=torch.tensor([[1], [2]]),
        )

        expected = _manual_siglip_loss(
            image_features,
            text_features,
            logit_scale,
            logit_bias,
        )
        assert torch.allclose(loss, expected)

    def test_accessory_aware_mask_keeps_mismatches_as_negatives(self):
        """Test only scalar-plus-accessory compatible terms are ignored."""
        attr_values = torch.tensor([
            [1, 1],
            [1, 1],
            [1, 1],
        ])
        image_accessory_ids = torch.tensor([
            [11, 12],
            [11, 0],
            [13, 0],
        ])
        text_accessory_ids = torch.tensor([
            [11, 0],
            [11, 12],
            [0, 0],
        ])

        valid_terms = MetadataMaskedSigLipLoss(
            accessory_aware=True
        ).get_valid_term_mask(
            image_attr_values=attr_values,
            text_attr_values=attr_values,
            image_accessory_ids=image_accessory_ids,
            text_accessory_ids=text_accessory_ids,
        )

        assert torch.equal(valid_terms, torch.tensor([
            [True, False, False],
            [False, True, False],
            [True, True, True],
        ]))

    def test_accessory_aware_mask_requires_accessory_tensors(self):
        """Test the opt-in mode fails clearly with legacy metadata."""
        attr_values = torch.tensor([[1, 1], [2, 2]])

        with pytest.raises(ValueError, match="requires image_accessory_ids"):
            MetadataMaskedSigLipLoss(
                accessory_aware=True
            ).get_valid_term_mask(
                image_attr_values=attr_values,
                text_attr_values=attr_values,
            )

    def test_all_wildcard_query_ignores_all_off_diagonal_images(self):
        """Test all-wildcard query masks every off-diagonal image."""
        image_features, text_features = _features()
        logit_scale = torch.tensor(2.0)
        logit_bias = torch.tensor(-0.5)
        image_attr_values = torch.tensor([
            [1, 1],
            [2, 2],
            [3, 3],
        ])
        text_attr_values = torch.tensor([
            [1, 1],
            [-1, -1],
            [3, 3],
        ])

        loss = MetadataMaskedSigLipLoss()(
            image_features,
            text_features,
            logit_scale,
            logit_bias,
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
        )

        valid_terms = torch.tensor([
            [True, False, True],
            [True, True, True],
            [True, False, True],
        ])
        expected = _manual_siglip_loss(
            image_features,
            text_features,
            logit_scale,
            logit_bias,
            valid_terms=valid_terms,
        )
        assert torch.allclose(loss, expected)

    def test_loss_uses_sum_over_local_batch_size_not_valid_mean(self):
        """Test normalization is OpenCLIP-compatible sum divided by batch."""
        image_features, text_features = _features()
        logit_scale = torch.tensor(2.0)
        logit_bias = torch.tensor(-0.5)
        image_attr_values = torch.tensor([
            [1, 1],
            [2, 2],
            [3, 3],
        ])
        text_attr_values = torch.tensor([
            [1, 1],
            [-1, -1],
            [3, 3],
        ])

        loss = MetadataMaskedSigLipLoss()(
            image_features,
            text_features,
            logit_scale,
            logit_bias,
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
        )

        valid_terms = torch.tensor([
            [True, False, True],
            [True, True, True],
            [True, False, True],
        ])
        logits = logit_scale * image_features @ text_features.T + logit_bias
        labels = -torch.ones_like(logits) + 2 * torch.eye(3)
        selected_terms = -F.logsigmoid(labels * logits).masked_select(
            valid_terms
        )

        assert torch.allclose(loss, selected_terms.sum() / 3)
        assert not torch.allclose(loss, selected_terms.mean())

    def test_output_dict_matches_open_clip_shape(self):
        """Test output_dict returns contrastive_loss key."""
        image_features, text_features = _features()
        attr_values = torch.tensor([
            [1, 1],
            [2, 2],
            [3, 3],
        ])

        output = MetadataMaskedSigLipLoss()(
            image_features,
            text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
            image_attr_values=attr_values,
            text_attr_values=attr_values,
            output_dict=True,
        )

        assert set(output) == {"contrastive_loss"}
        assert output["contrastive_loss"].ndim == 0

    def test_gather_masks_cross_rank_metadata_matches(self, monkeypatch):
        """Test cross-rank metadata-compatible terms are not negatives."""
        image_features = torch.tensor([
            [0.10, 0.20, 0.30],
            [0.30, 0.20, 0.10],
        ])
        text_features = torch.tensor([
            [0.20, 0.10, 0.30],
            [0.10, 0.30, 0.20],
        ])
        remote_text_features = torch.tensor([
            [0.40, 0.10, 0.20],
            [0.30, 0.40, 0.10],
        ])
        image_attr_values = torch.tensor([[1], [2]])
        text_attr_values = torch.tensor([[1], [2]])
        remote_text_attr_values = torch.tensor([[1], [3]])

        _mock_distributed_context(monkeypatch)
        monkeypatch.setattr(
            loss_module.dist_nn,
            "all_gather",
            lambda tensor: (tensor, remote_text_features),
        )

        def fake_all_gather(gathered, tensor):
            if tensor.ndim == 1:
                gathered[0].fill_(text_features.shape[0])
                gathered[1].fill_(remote_text_features.shape[0])
                return
            gathered[0].copy_(tensor)
            gathered[1].copy_(remote_text_attr_values)

        monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

        loss = MetadataMaskedSigLipLoss(
            dist_impl="gather",
            world_size=2,
            rank=0,
        )(
            image_features,
            text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
        )

        all_text_features = torch.cat(
            [text_features, remote_text_features], dim=0
        )
        valid_terms = torch.tensor([
            [True, True, False, True],
            [True, True, True, True],
        ])
        expected = _manual_siglip_loss_rectangular(
            image_features,
            all_text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
            positive_text_indices=torch.tensor([0, 1]),
            valid_terms=valid_terms,
        )
        assert torch.allclose(loss, expected)

    def test_gather_promotes_cross_rank_compatible_pairs(self, monkeypatch):
        """Test gathered compatible pairs use rectangular weighted targets."""
        image_features = torch.tensor([
            [0.10, 0.20, 0.30],
            [0.30, 0.20, 0.10],
        ])
        text_features = torch.tensor([
            [0.20, 0.10, 0.30],
            [0.10, 0.30, 0.20],
        ])
        remote_text_features = torch.tensor([
            [0.40, 0.10, 0.20],
            [0.30, 0.40, 0.10],
        ])
        image_attr_values = torch.tensor([[1], [2]])
        text_attr_values = torch.tensor([[1], [2]])
        remote_text_attr_values = torch.tensor([[1], [3]])

        _mock_distributed_context(monkeypatch)
        monkeypatch.setattr(
            loss_module.dist_nn,
            "all_gather",
            lambda tensor: (tensor, remote_text_features),
        )

        def fake_all_gather(gathered, tensor):
            if tensor.ndim == 1:
                gathered[0].fill_(text_features.shape[0])
                gathered[1].fill_(remote_text_features.shape[0])
                return
            gathered[0].copy_(tensor)
            gathered[1].copy_(remote_text_attr_values)

        monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

        loss_fn = MetadataMaskedSigLipLoss(
            dist_impl="gather",
            world_size=2,
            rank=0,
            compatible_as_positive=True,
            compatible_positive_weight=0.5,
        )
        output = loss_fn(
            image_features,
            text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
            output_dict=True,
        )

        all_text_features = torch.cat(
            [text_features, remote_text_features], dim=0
        )
        logits = 2.0 * image_features @ all_text_features.T - 0.5
        labels = torch.tensor([
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, -1.0],
        ])
        weights = torch.ones_like(labels)
        weights[0, 2] = 0.5
        expected = (-F.logsigmoid(labels * logits) * weights).sum() / 2

        assert output["contrastive_loss"].ndim == 0
        assert torch.allclose(output["contrastive_loss"], expected)
        assert output["compatible_positive_pairs"].item() == 1
        assert loss_fn.last_compatible_positive_pairs.item() == 1

    def test_gather_per_query_normalizes_with_global_compatible_counts(
        self, monkeypatch
    ):
        """Test per-query weights use global counts with rectangular shapes."""
        labels = -torch.ones(2, 4)
        positive_text_indices = torch.tensor([2, 3])
        labels[torch.arange(2), positive_text_indices] = 1
        metadata_match = torch.tensor([
            [True, False, True, False],
            [False, True, False, True],
        ])
        reduced_shapes = []

        def fake_all_reduce(tensor, op):
            assert op == dist.ReduceOp.SUM
            reduced_shapes.append(tuple(tensor.shape))
            tensor.add_(torch.tensor([1.0, 0.0, 2.0, 0.0]))

        monkeypatch.setattr(loss_module.dist, "all_reduce", fake_all_reduce)
        loss_fn = MetadataMaskedSigLipLoss(
            dist_impl="gather",
            world_size=2,
            rank=1,
            compatible_as_positive=True,
            compatible_positive_weight=0.5,
            compatible_positive_normalization="per_query",
        )

        targets, valid, weights, promoted = (
            loss_fn.get_targets_and_term_weights(
                labels=labels,
                valid_terms=~metadata_match,
                metadata_match=metadata_match,
                positive_text_indices=positive_text_indices,
                query_has_evidence=torch.ones(4, dtype=torch.bool),
            )
        )

        assert reduced_shapes == [(4,)]
        assert targets.shape == valid.shape == weights.shape == promoted.shape
        assert targets.shape == (2, 4)
        assert weights[0, 0] == 0.25
        assert weights[1, 1] == 0.5
        assert targets[0, 2] == 1 and targets[1, 3] == 1
        assert weights[0, 2] == 1 and weights[1, 3] == 1

    @pytest.mark.skipif(
        not dist.is_available() or not dist.is_gloo_available(),
        reason="Two-rank gradient test requires torch.distributed with Gloo.",
    )
    def test_gather_propagates_cross_rank_text_gradients(self, tmp_path):
        """Test real Gloo gather backpropagates remote-rank loss terms."""
        world_size = 2
        init_method = f"file://{tmp_path / 'gloo_init'}"

        mp.spawn(
            _run_two_rank_gloo_gradient_check,
            args=(world_size, init_method),
            nprocs=world_size,
            join=True,
        )

    @pytest.mark.skipif(
        not dist.is_available() or not dist.is_gloo_available(),
        reason="Two-rank batch-size test requires torch.distributed with Gloo.",
    )
    def test_gather_rejects_uneven_cross_rank_batches(self, tmp_path):
        """Test unequal per-rank batches fail before shaped all_gather."""
        world_size = 2
        init_method = f"file://{tmp_path / 'gloo_uneven_init'}"

        mp.spawn(
            _run_two_rank_gloo_uneven_batch_check,
            args=(world_size, init_method),
            nprocs=world_size,
            join=True,
        )

    def test_gather_masks_cross_rank_accessory_matches(self, monkeypatch):
        """Test cross-rank scalar+accessory matches are also ignored."""
        image_features = torch.tensor([
            [0.10, 0.20, 0.30],
            [0.30, 0.20, 0.10],
        ])
        text_features = torch.tensor([
            [0.20, 0.10, 0.30],
            [0.10, 0.30, 0.20],
        ])
        remote_text_features = torch.tensor([
            [0.40, 0.10, 0.20],
            [0.30, 0.40, 0.10],
        ])
        image_attr_values = torch.tensor([[1], [1]])
        text_attr_values = torch.tensor([[1], [1]])
        remote_text_attr_values = torch.tensor([[1], [1]])
        image_accessory_ids = torch.tensor([[11, 12], [13, 0]])
        text_accessory_ids = torch.tensor([[11, 0], [13, 0]])
        remote_text_accessory_ids = torch.tensor([[11, 12], [14, 0]])

        _mock_distributed_context(monkeypatch)
        monkeypatch.setattr(
            loss_module.dist_nn,
            "all_gather",
            lambda tensor: (tensor, remote_text_features),
        )

        def fake_all_gather(gathered, tensor):
            if tensor.ndim == 1:
                gathered[0].fill_(text_features.shape[0])
                gathered[1].fill_(remote_text_features.shape[0])
            elif tensor.shape[1] == 1:
                gathered[0].copy_(tensor)
                gathered[1].copy_(remote_text_attr_values)
            else:
                gathered[0].copy_(tensor)
                gathered[1].copy_(remote_text_accessory_ids)

        monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

        loss = MetadataMaskedSigLipLoss(
            dist_impl="gather",
            world_size=2,
            rank=0,
            accessory_aware=True,
        )(
            image_features,
            text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
            image_accessory_ids=image_accessory_ids,
            text_accessory_ids=text_accessory_ids,
        )

        all_text_features = torch.cat(
            [text_features, remote_text_features], dim=0
        )
        valid_terms = torch.tensor([
            [True, True, False, True],
            [True, True, True, True],
        ])
        expected = _manual_siglip_loss_rectangular(
            image_features,
            all_text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
            positive_text_indices=torch.tensor([0, 1]),
            valid_terms=valid_terms,
        )
        assert torch.allclose(loss, expected)

    def test_gather_places_rank_one_positives_in_its_text_chunk(
        self, monkeypatch
    ):
        """Test nonzero ranks use their gathered text chunk for positives."""
        image_features = torch.tensor([
            [0.10, 0.20, 0.30],
            [0.30, 0.20, 0.10],
        ])
        text_features = torch.tensor([
            [0.20, 0.10, 0.30],
            [0.10, 0.30, 0.20],
        ])
        remote_text_features = torch.tensor([
            [0.40, 0.10, 0.20],
            [0.30, 0.40, 0.10],
        ])
        image_attr_values = torch.tensor([[1], [2]])
        text_attr_values = torch.tensor([[1], [2]])
        remote_text_attr_values = torch.tensor([[3], [4]])

        _mock_distributed_context(monkeypatch, rank=1)
        monkeypatch.setattr(
            loss_module.dist_nn,
            "all_gather",
            lambda tensor: (remote_text_features, tensor),
        )

        def fake_all_gather(gathered, tensor):
            if tensor.ndim == 1:
                gathered[0].fill_(remote_text_features.shape[0])
                gathered[1].fill_(text_features.shape[0])
            else:
                gathered[0].copy_(remote_text_attr_values)
                gathered[1].copy_(tensor)

        monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

        loss = MetadataMaskedSigLipLoss(
            dist_impl="gather",
            world_size=2,
            rank=1,
        )(
            image_features,
            text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
        )

        expected = _manual_siglip_loss_rectangular(
            image_features,
            torch.cat([remote_text_features, text_features], dim=0),
            torch.tensor(2.0),
            torch.tensor(-0.5),
            positive_text_indices=torch.tensor([2, 3]),
            valid_terms=torch.ones((2, 4), dtype=torch.bool),
        )
        assert torch.allclose(loss, expected)

    def test_gather_rejects_mismatched_process_group(self, monkeypatch):
        """Test stale trainer coordinates fail before entering a collective."""
        image_features, text_features = _features()
        attr_values = torch.tensor([[1], [2], [3]])
        _mock_distributed_context(monkeypatch, rank=0, world_size=4)

        with pytest.raises(RuntimeError, match="distributed context mismatch"):
            MetadataMaskedSigLipLoss(
                dist_impl="gather",
                world_size=2,
                rank=0,
            )(
                image_features,
                text_features,
                torch.tensor(2.0),
                torch.tensor(-0.5),
                image_attr_values=attr_values,
                text_attr_values=attr_values,
            )

    def test_rejects_unsupported_dist_impl(self):
        """Test unsupported distributed modes fail clearly."""
        image_features, text_features = _features()
        attr_values = torch.tensor([
            [1, 1],
            [2, 2],
            [3, 3],
        ])

        with pytest.raises(ValueError, match="local.*gather"):
            MetadataMaskedSigLipLoss(dist_impl="bidir", world_size=2)(
                image_features,
                text_features,
                torch.tensor(2.0),
                torch.tensor(-0.5),
                image_attr_values=attr_values,
                text_attr_values=attr_values,
            )

    def test_positive_mode_validates_weight_and_normalization(self):
        """Test invalid normalized-positive controls fail at construction."""
        with pytest.raises(ValueError, match="non-negative"):
            MetadataMaskedSigLipLoss(compatible_positive_weight=-0.1)
        with pytest.raises(ValueError, match="per_pair.*per_query"):
            MetadataMaskedSigLipLoss(
                compatible_positive_normalization="unsupported"
            )
