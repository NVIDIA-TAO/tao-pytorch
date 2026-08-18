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

"""Metadata matching utilities for CLIP/SigLIP training."""

import torch


def build_attribute_match_mask(
    image_attr_values: torch.Tensor,
    text_attr_values: torch.Tensor,
) -> torch.Tensor:
    """Build text-to-image attribute match mask.

    Args:
        image_attr_values: Integer image attributes with shape [B_image, A].
            Values below zero mean the image attribute is unknown and do not
            satisfy a query that specifies that attribute.
        text_attr_values: Integer text query attributes with shape [B_text, A].
            Values below zero mean the query does not constrain that attribute
            and are treated as wildcards.

    Returns:
        Bool tensor with shape [B_text, B_image]. Entry [i, j] is True when
        every specified attribute in text query i matches image j.
    """
    if image_attr_values.ndim != 2:
        raise ValueError(
            "image_attr_values must have shape [B_image, A], got "
            f"{tuple(image_attr_values.shape)}."
        )
    if text_attr_values.ndim != 2:
        raise ValueError(
            "text_attr_values must have shape [B_text, A], got "
            f"{tuple(text_attr_values.shape)}."
        )
    if image_attr_values.shape[1] != text_attr_values.shape[1]:
        raise ValueError(
            "image_attr_values and text_attr_values must have the same "
            f"attribute width, got {image_attr_values.shape[1]} and "
            f"{text_attr_values.shape[1]}."
        )

    specified_query_attrs = text_attr_values >= 0
    same = text_attr_values[:, None, :] == image_attr_values[None, :, :]
    ok = same | ~specified_query_attrs[:, None, :]
    return ok.all(dim=-1)


def build_accessory_match_mask(
    image_accessory_ids: torch.Tensor,
    text_accessory_ids: torch.Tensor,
) -> torch.Tensor:
    """Build text-to-image required-accessory match mask.

    Accessory tensors contain positive vocabulary IDs and use zero for
    padding. A query matches an image when every positive query accessory ID
    is present in the image. Queries with no required accessories match every
    image.

    Args:
        image_accessory_ids: Padded image accessory IDs with shape
            [B_image, K_image].
        text_accessory_ids: Padded query accessory IDs with shape
            [B_text, K_text].

    Returns:
        Bool tensor with shape [B_text, B_image].

    Note:
        The broadcast comparison below materializes an intermediate with shape
        [B_text, B_image, K_text, K_image]. This is acceptable for current PAS
        batch sizes and accessory widths. If those grow substantially, reduce
        over text accessory slots or chunks to bound peak memory.
    """
    if image_accessory_ids.ndim != 2:
        raise ValueError(
            "image_accessory_ids must have shape [B_image, K_image], got "
            f"{tuple(image_accessory_ids.shape)}."
        )
    if text_accessory_ids.ndim != 2:
        raise ValueError(
            "text_accessory_ids must have shape [B_text, K_text], got "
            f"{tuple(text_accessory_ids.shape)}."
        )
    if torch.any(image_accessory_ids < 0):
        raise ValueError("image_accessory_ids must contain non-negative IDs.")
    if torch.any(text_accessory_ids < 0):
        raise ValueError("text_accessory_ids must contain non-negative IDs.")

    required = text_accessory_ids > 0
    same = torch.eq(
        text_accessory_ids[:, None, :, None],
        image_accessory_ids[None, :, None, :],
    )
    present = same.any(dim=-1)
    return (present | ~required[:, None, :]).all(dim=-1)
