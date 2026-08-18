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

"""Unit tests for CLIP metadata match masks."""

import pytest
import torch

from nvidia_tao_pytorch.multimodal.clip.loss.metadata_mask import (
    build_accessory_match_mask,
    build_attribute_match_mask,
)


@pytest.mark.multimodal_unit
class TestBuildAttributeMatchMask:
    """Test text-query to image-attribute matching."""

    def test_wildcards_match_only_specified_attributes(self):
        """Test negative text values are treated as wildcards."""
        image_attr_values = torch.tensor([
            [1, 6, 2, 5, 8, 5, 2],
            [2, 3, 2, 5, 12, 7, 3],
            [1, 6, 1, 3, 8, 5, 1],
        ])
        text_attr_values = torch.tensor([
            [1, 6, 2, 5, -1, -1, -1],
            [2, 3, -1, 5, -1, -1, -1],
            [1, 6, -1, -1, 8, -1, -1],
        ])

        mask = build_attribute_match_mask(
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
        )

        expected = torch.tensor([
            [True, False, False],
            [False, True, False],
            [True, False, True],
        ])
        assert mask.dtype == torch.bool
        assert torch.equal(mask, expected)

    def test_no_wildcards_requires_exact_match(self):
        """Test fully specified queries require exact row equality."""
        image_attr_values = torch.tensor([
            [1, 2, 3],
            [1, 2, 4],
        ])
        text_attr_values = torch.tensor([
            [1, 2, 3],
            [1, 2, 5],
        ])

        mask = build_attribute_match_mask(
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
        )

        assert torch.equal(mask, torch.tensor([
            [True, False],
            [False, False],
        ]))

    def test_all_wildcard_query_matches_all_images(self):
        """Test a query with no specified attributes matches every image."""
        image_attr_values = torch.tensor([
            [1, 2, 3],
            [4, 5, 6],
        ])
        text_attr_values = torch.tensor([[-1, -1, -1]])

        mask = build_attribute_match_mask(
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
        )

        assert torch.equal(mask, torch.tensor([[True, True]]))

    def test_unknown_image_does_not_satisfy_specified_query(self):
        """Test missing image metadata is not a symmetric wildcard."""
        image_attr_values = torch.tensor([
            [-1, 2],
            [1, 2],
        ])
        text_attr_values = torch.tensor([
            [1, 2],
            [-1, 2],
        ])

        mask = build_attribute_match_mask(
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
        )

        assert torch.equal(mask, torch.tensor([
            [False, True],
            [True, True],
        ]))

    def test_requires_rank_two_tensors(self):
        """Test input tensors must be rank two."""
        with pytest.raises(ValueError, match="image_attr_values"):
            build_attribute_match_mask(
                image_attr_values=torch.tensor([1, 2, 3]),
                text_attr_values=torch.tensor([[1, 2, 3]]),
            )
        with pytest.raises(ValueError, match="text_attr_values"):
            build_attribute_match_mask(
                image_attr_values=torch.tensor([[1, 2, 3]]),
                text_attr_values=torch.tensor([1, 2, 3]),
            )

    def test_requires_matching_attribute_width(self):
        """Test image/text attribute widths must match."""
        with pytest.raises(ValueError, match="same attribute width"):
            build_attribute_match_mask(
                image_attr_values=torch.tensor([[1, 2, 3]]),
                text_attr_values=torch.tensor([[1, 2]]),
            )


@pytest.mark.multimodal_unit
class TestBuildAccessoryMatchMask:
    """Test required-accessory subset matching."""

    def test_requires_every_positive_query_accessory(self):
        """Test padding is ignored and all required IDs must be present."""
        image_accessory_ids = torch.tensor([
            [11, 12, 0],
            [11, 0, 0],
            [12, 13, 0],
        ])
        text_accessory_ids = torch.tensor([
            [11, 0],
            [11, 12],
            [0, 0],
        ])

        mask = build_accessory_match_mask(
            image_accessory_ids=image_accessory_ids,
            text_accessory_ids=text_accessory_ids,
        )

        assert torch.equal(mask, torch.tensor([
            [True, True, False],
            [True, False, False],
            [True, True, True],
        ]))

    def test_rejects_invalid_shapes_and_negative_ids(self):
        """Test accessory tensors must be padded rank-two non-negative IDs."""
        with pytest.raises(ValueError, match="image_accessory_ids"):
            build_accessory_match_mask(
                torch.tensor([1, 2]),
                torch.tensor([[1, 2]]),
            )
        with pytest.raises(ValueError, match="non-negative"):
            build_accessory_match_mask(
                torch.tensor([[1, -1]]),
                torch.tensor([[1, 0]]),
            )
