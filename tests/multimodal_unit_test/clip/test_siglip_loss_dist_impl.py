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

"""Unit tests for SigLIP distributed loss implementation selection."""

from types import SimpleNamespace

import pytest
import torch
from open_clip.loss import SigLipLoss

from nvidia_tao_pytorch.config.clip.default_config import CLIPTrainConfig
from nvidia_tao_pytorch.multimodal.clip.loss.masked_siglip_loss import (
    MetadataMaskedSigLipLoss,
)
from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import CLIPPlModel


def _build_siglip_loss(
    dist_impl,
    world_size=16,
    rank=3,
    mask_mode="none",
    include_attribute_metadata=True,
    dataset_type="custom",
):
    """Build the SigLIP criterion without constructing the full model."""
    model = SimpleNamespace(
        loss_type="siglip",
        siglip_loss_dist_impl=dist_impl,
        siglip_loss_mask_mode=mask_mode,
        global_rank=rank,
        trainer=SimpleNamespace(world_size=world_size),
        experiment_spec=SimpleNamespace(
            dataset=SimpleNamespace(
                train=SimpleNamespace(
                    include_attribute_metadata=include_attribute_metadata,
                    type=dataset_type,
                ),
            ),
        ),
    )
    CLIPPlModel._build_criterion(model)
    return model.loss


@pytest.mark.multimodal_unit
class TestSigLipLossDistImpl:
    """Test SigLIP loss distributed implementation selection."""

    def test_config_defaults_to_gather_and_allows_local(self):
        """Test default and valid options for SigLIP loss distribution."""
        field = CLIPTrainConfig.__dataclass_fields__["siglip_loss_dist_impl"]

        assert CLIPTrainConfig().siglip_loss_dist_impl == "gather"
        assert "local" in field.metadata["valid_options"].split(",")
        mask_field = CLIPTrainConfig.__dataclass_fields__[
            "siglip_loss_mask_mode"
        ]
        assert "attribute_plus_accessory_match_ignore" in (
            mask_field.metadata["valid_options"].split(",")
        )

    def test_local_siglip_loss_uses_single_rank_loss_world(self):
        """Test local mode disables cross-rank negative exchange."""
        loss = _build_siglip_loss("local")

        assert isinstance(loss, SigLipLoss)
        assert loss.dist_impl == "local"
        assert loss.world_size == 1
        assert loss.rank == 0

    def test_gather_siglip_loss_uses_trainer_world(self):
        """Test gather mode keeps the trainer world size."""
        loss = _build_siglip_loss("gather")

        assert isinstance(loss, SigLipLoss)
        assert loss.dist_impl == "gather"
        assert loss.world_size == 16
        assert loss.rank == 3

    def test_local_masked_siglip_loss_uses_metadata_masked_loss(self):
        """Test masked SigLIP local mode selects the TAO masked loss."""
        loss = _build_siglip_loss(
            "local",
            mask_mode="attribute_match_ignore",
        )

        assert isinstance(loss, MetadataMaskedSigLipLoss)
        assert loss.dist_impl == "local"
        assert loss.world_size == 1

    def test_gather_masked_siglip_loss_uses_trainer_world(
        self,
    ):
        """Test masked gather mode keeps cross-rank negatives."""
        loss = _build_siglip_loss(
            "gather",
            mask_mode="attribute_match_ignore",
        )

        assert isinstance(loss, MetadataMaskedSigLipLoss)
        assert loss.dist_impl == "gather"
        assert loss.world_size == 16
        assert loss.rank == 3

    def test_accessory_mask_mode_selects_accessory_aware_loss(self):
        """Test the new mode enables accessory-aware metadata matching."""
        loss = _build_siglip_loss(
            "local",
            mask_mode="attribute_plus_accessory_match_ignore",
        )

        assert isinstance(loss, MetadataMaskedSigLipLoss)
        assert loss.accessory_aware is True

    @pytest.mark.parametrize(
        "mask_mode",
        [
            "attribute_match_ignore",
            "attribute_plus_accessory_match_ignore",
        ],
    )
    def test_masked_siglip_requires_training_metadata(self, mask_mode):
        """Test masked loss rejects a dataset that omits batch metadata."""
        with pytest.raises(
            ValueError,
            match="dataset.train.include_attribute_metadata=True",
        ):
            _build_siglip_loss(
                "local",
                mask_mode=mask_mode,
                include_attribute_metadata=False,
            )

    def test_masked_siglip_requires_custom_dataset(self):
        """Test masked loss rejects loaders that cannot emit metadata."""
        with pytest.raises(ValueError, match="dataset.train.type='custom'"):
            _build_siglip_loss(
                "local",
                mask_mode="attribute_match_ignore",
                dataset_type="wds",
            )

    def test_unmasked_siglip_does_not_require_training_metadata(self):
        """Test regular SigLIP remains valid without metadata batches."""
        loss = _build_siglip_loss(
            "local",
            include_attribute_metadata=False,
            dataset_type="wds",
        )

        assert isinstance(loss, SigLipLoss)

    def test_masked_siglip_loss_rejects_unsupported_distributed_mode(self):
        """Test masked SigLIP rejects modes without metadata exchange."""
        with pytest.raises(NotImplementedError, match="local.*gather"):
            _build_siglip_loss(
                "bidir",
                mask_mode="attribute_match_ignore",
            )

    def test_masked_siglip_backward_requires_metadata(self):
        """Test masked SigLIP training batches must include metadata."""
        model = SimpleNamespace(loss=MetadataMaskedSigLipLoss())
        outputs = (
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.tensor(1.0),
            torch.tensor(0.0),
        )

        with pytest.raises(ValueError, match="requires batch metadata"):
            CLIPPlModel._backward(model, outputs, batch=(None, None))

    def test_masked_siglip_backward_uses_batch_metadata(self):
        """Test _backward passes batch metadata into the masked loss."""
        model = SimpleNamespace(loss=MetadataMaskedSigLipLoss())
        image_features = torch.tensor([
            [0.10, 0.20, 0.30],
            [0.30, 0.20, 0.10],
        ])
        text_features = torch.tensor([
            [0.20, 0.10, 0.30],
            [0.10, 0.30, 0.20],
        ])
        outputs = (
            image_features,
            text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
        )
        metadata = {
            "image_attr_values": torch.tensor([
                [1, 1],
                [2, 2],
            ]),
            "text_attr_values": torch.tensor([
                [1, 1],
                [2, 2],
            ]),
        }

        loss, logit_scale, output_image_features, output_text_features = (
            CLIPPlModel._backward(model, outputs, batch=(None, None, metadata))
        )
        expected = model.loss(
            image_features,
            text_features,
            torch.tensor(2.0),
            torch.tensor(-0.5),
            image_attr_values=metadata["image_attr_values"],
            text_attr_values=metadata["text_attr_values"],
        )

        assert torch.allclose(loss, expected)
        assert logit_scale is outputs[2]
        assert output_image_features is image_features
        assert output_text_features is text_features
