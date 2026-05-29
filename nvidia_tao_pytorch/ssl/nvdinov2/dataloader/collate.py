# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVDINOv2 Collate"""

import math
import random
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch


class DinoV2Collate:
    """Collate for NVDINOv2 to combine and transform global and local crops, applying masking strategies for efficient training."""

    def __init__(
        self,
        patch_size: int,
        mask_probability: float = 0.5,
        mask_ratio_range: Tuple[float, float] = (0.1, 0.5),
        mask_type: Literal["block", "random"] = "block",
        # The following configs are used for block mask
        max_mask_ratio_per_iter: float = 0.5,
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: Optional[float] = None,
    ) -> None:
        """Init of Collate for NVDINOv2.

        Args:
            patch_size (int): Size of the patches for masking.
            mask_probability (float, optional): Probability of masking crops. Defaults to 0.5.
            mask_ratio_range (Tuple[float, float], optional): Range of mask ratios. Defaults to (0.1, 0.5).
            mask_type (Literal["block", "random"], optional): Type of masking to apply. Defaults to "block".
            max_mask_ratio_per_iter (float, optional): Maximum mask ratio per iteration. Defaults to 0.5.
            min_aspect_ratio (float, optional): Minimum aspect ratio for the masked area. Defaults to 0.3.
            max_aspect_ratio (Optional[float], optional): Maximum aspect ratio for the masked area. Defaults to None.
        """
        self.patch_size = patch_size
        self.mask_probability = mask_probability
        self.mask_ratio_range = mask_ratio_range
        self.mask_type = mask_type
        self.max_mask_ratio_per_iter = max_mask_ratio_per_iter

        if max_aspect_ratio is None:
            max_aspect_ratio = 1 / min_aspect_ratio
        self.log_aspect_ratio = (math.log(min_aspect_ratio), math.log(max_aspect_ratio))

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """collate function to a batch of data, combining and transforming global and local crops

        Args:
            batch (List[Dict[str, Any]]): The batch of data to be collated and transformed.

        Returns:
            Dict[str, Any]: The output dictionary containing the combined global and local crops, generated masks, and mask properties.
        """
        batch = [b for b in batch if b is not None]

        n_global_crops = len(batch[0]["global_crops"])
        n_local_crops = len(batch[0]["local_crops"])

        assert all(len(b["global_crops"]) == n_global_crops for b in batch), "Mismatch in the number of global crops across samples in the batch."
        assert all(len(b["local_crops"]) == n_local_crops for b in batch), "Mismatch in the number of local crops across samples in the batch."

        # (B * 2, C, H, W)
        # b0_c0, b1_c0, ..., b0_c1, b1_c1, ...
        global_crops = torch.stack(
            [b["global_crops"][i] for i in range(n_global_crops) for b in batch], dim=0
        )
        local_crops = torch.stack(
            [b["local_crops"][i] for i in range(n_local_crops) for b in batch], dim=0
        )

        # assert dividable by patch_size
        assert (
            global_crops.shape[-1] % self.patch_size == 0 and global_crops.shape[-2] % self.patch_size == 0
        ), "Global crop dimensions must be divisible by the patch size."
        assert (
            local_crops.shape[-1] % self.patch_size == 0 and local_crops.shape[-2] % self.patch_size == 0
        ), "Local crop dimensions must be divisible by the patch size."

        mask_shape = (
            int(global_crops.shape[-2] // self.patch_size),
            int(global_crops.shape[-1] // self.patch_size),
        )
        num_global_crops_tokens = math.prod(mask_shape)

        if self.mask_probability > 0:
            global_masks = []
            n_masked_samples = int(global_crops.shape[0] * self.mask_probability)
            mask_probabilities = torch.linspace(
                start=self.mask_ratio_range[0],
                end=self.mask_ratio_range[1],
                steps=n_masked_samples + 1,
            )
            upperbound = 0

            for prob_min, prob_max in zip(mask_probabilities[:-1], mask_probabilities[1:]):
                upperbound += int(prob_max * num_global_crops_tokens)

                mask_prob = float(random.uniform(prob_min, prob_max))
                global_masks.append(self._get_mask(mask_shape, mask_prob))

            global_masks = torch.stack(global_masks, dim=0)

            # Add unpadded mask (B, H, W)
            global_masks = torch.cat(
                [
                    global_masks,
                    torch.zeros(
                        (global_crops.shape[0] - global_masks.shape[0], *mask_shape),
                        dtype=torch.bool,
                    ),
                ],
                dim=0,
            )

            # Flatten (B, H * W)
            global_masks = global_masks.flatten(1)
        else:
            global_masks = torch.zeros(
                (global_crops.shape[0], *mask_shape), dtype=torch.bool
            ).flatten(1)
            upperbound = 0

        # Shuffle
        global_masks = global_masks[torch.randperm(global_masks.shape[0])]
        assert global_masks.shape == (global_crops.shape[0], math.prod(mask_shape)), "Global masks shape does not match the expected shape after reshaping and shuffling."

        global_masks_indices = global_masks.flatten().nonzero().flatten()
        global_masks_weight = (
            (1 / global_masks.float().sum(-1).clamp(min=1.0))
            .unsqueeze(-1)
            .expand_as(global_masks)[global_masks]
        )

        return {
            "global_crops": global_crops,
            "local_crops": local_crops,
            "global_masks": global_masks,
            "global_masks_indices": global_masks_indices,
            "global_masks_weight": global_masks_weight,
            "upperbound": upperbound,
        }

    def _get_mask(self, mask_shape: Tuple[int, int], mask_prob: float) -> torch.Tensor:
        """Generate a mask based on the specified shape and probability.

        Args:
            mask_shape (Tuple[int, int]): The shape of the mask to be generated.
            mask_prob (float):  The probability used to determine the extent of masking within the mask shape.

        Returns:
            torch.Tensor: The mask generated based on the specified shape and probability.
        """
        total_tokens = mask_shape[0] * mask_shape[1]
        to_mask_count = round(total_tokens * mask_prob)

        if self.mask_type == "random":
            mask = torch.randperm(total_tokens)[:to_mask_count]
            mask = torch.zeros(total_tokens).scatter_(0, mask, 1).reshape(mask_shape)
        elif self.mask_type == "block":
            mask = torch.zeros(mask_shape)
            masked_count = 0
            H, W = mask_shape

            while masked_count < to_mask_count:
                max_mask_patches = to_mask_count - masked_count
                # Shouldn't mask more than half of the image in one go
                max_mask_patches = min(
                    max_mask_patches, total_tokens * self.max_mask_ratio_per_iter
                )

                delta = 0
                for _ in range(10):
                    target_area = random.uniform(4, max_mask_patches)
                    aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))

                    if w < W and h < H:
                        top = random.randint(0, H - h)
                        left = random.randint(0, W - w)

                        num_masked = mask[top:top + h, left:left + w].sum()

                        if 0 < h * w - num_masked <= max_mask_patches:
                            for i in range(top, top + h):
                                for j in range(left, left + w):
                                    if mask[i, j] == 0:
                                        mask[i, j] = 1
                                        delta += 1

                    if delta > 0:
                        break

                if delta == 0:
                    break

                masked_count += delta

        return mask.bool()
