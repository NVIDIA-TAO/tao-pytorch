# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partial-negative Alignment loss for image-text retrieval."""

import torch
import torch.nn.functional as F


def partial_negative_alignment_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    margin: float,
    inverse_temperature: float,
    top_ratio: float,
    compatible_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute symmetric PA loss over the hardest valid negatives.

    The loss follows the partial-negative component of Song et al., *Dual
    alignment: Partial negative and soft-label alignment for text-to-image
    person retrieval*. ``compatible_mask`` generalizes identity positives to
    PAS metadata-compatible pairs; those pairs participate as positives and
    are never mined as negatives. ``inverse_temperature`` is ``1 / tau`` in
    the paper's notation.
    """
    batch_size = image_features.shape[0]
    if batch_size != text_features.shape[0]:
        raise ValueError(
            "PA requires equal image and text batch sizes, got "
            f"{batch_size} and {text_features.shape[0]}."
        )
    if batch_size < 2:
        return image_features.new_zeros(())
    if margin < 0:
        raise ValueError("PA margin must be non-negative")
    if inverse_temperature <= 0:
        raise ValueError("PA inverse temperature must be positive")
    if not 0.0 < top_ratio <= 1.0:
        raise ValueError("PA top_ratio must be in (0, 1]")

    image_norm = F.normalize(image_features, dim=-1)
    text_norm = F.normalize(text_features, dim=-1)
    similarity = image_norm @ text_norm.mT
    positive = torch.eye(
        batch_size, device=similarity.device, dtype=torch.bool
    )
    if compatible_mask is not None:
        compatible_mask = compatible_mask.to(
            device=similarity.device, dtype=torch.bool
        )
        if compatible_mask.shape != similarity.shape:
            raise ValueError(
                "compatible_mask must match the PA similarity matrix, got "
                f"{tuple(compatible_mask.shape)} and {tuple(similarity.shape)}."
            )
        positive = positive | compatible_mask

    def _direction(
        scores: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Eq. 3: similarity-weighted mean over valid positives.
        positive_logits = (inverse_temperature * scores).masked_fill(
            ~positive_mask, float("-inf")
        )
        positive_weights = torch.softmax(positive_logits, dim=1)
        positive_score = (positive_weights * scores).sum(dim=1)

        # Eq. 4--5: retain the largest top-ratio fraction of valid negatives.
        negative_mask = ~positive_mask
        negative_counts = negative_mask.sum(dim=1)
        selected_counts = torch.ceil(
            negative_counts.float() * top_ratio
        ).long()
        max_selected = max(1, int(selected_counts.max().item()))
        negative_scores = scores.masked_fill(
            ~negative_mask, float("-inf")
        )
        top_negative_scores = torch.topk(
            negative_scores, k=max_selected, dim=1
        ).values
        positions = torch.arange(
            max_selected, device=scores.device
        ).unsqueeze(0)
        top_negative_scores = top_negative_scores.masked_fill(
            positions >= selected_counts.unsqueeze(1), float("-inf")
        )
        smooth_hard_negative = torch.logsumexp(
            inverse_temperature * top_negative_scores, dim=1
        ) / inverse_temperature
        per_anchor = F.relu(
            margin - positive_score + smooth_hard_negative
        )
        # A row with no valid negatives has no PA supervision.
        return per_anchor.masked_fill(negative_counts == 0, 0.0).mean()

    return 0.5 * (
        _direction(similarity, positive) +
        _direction(similarity.mT, positive.mT)
    )
