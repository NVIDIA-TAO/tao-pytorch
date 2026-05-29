# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anchor Selector for Sparse4D."""
import torch

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.sparse4d.model.detection3d.decoder import decode_box


class AnchorSelector:
    """Anchor Selector for Sparse4D."""

    def __init__(self, grid_size=0.5):
        """Initialize the Anchor Selector."""
        self.grid_size = grid_size

    def create_grid(self, anchors_batch):
        """
        Create grid representation for a batch of anchors
        Args:
            anchors_batch: (batch_size, n_anchors, 11)
        Returns:
            list of grids for each batch
        """
        batch_size = anchors_batch.size(0)
        batch_grids = []

        for b in range(batch_size):
            grid = {}
            anchors = anchors_batch[b]  # (n_anchors, 11)
            if isinstance(self.grid_size, tuple) and len(self.grid_size) > 3:
                anchors_decoded = decode_box(anchors)
            else:
                anchors_decoded = anchors
            for i, anchor in enumerate(anchors_decoded):
                if torch.isnan(anchor).any():
                    logging.warning(f"[warning] anchors {i} is NaN in AnchorSelector.create_grid")
                    continue
                if isinstance(self.grid_size, tuple):
                    grid_pos = tuple(
                        [
                            int(anchor[i] // self.grid_size[i])
                            for i in range(len(self.grid_size))
                        ]
                    )
                else:
                    grid_pos = tuple((anchor[:3] // self.grid_size).int().tolist())
                if grid_pos not in grid:
                    grid[grid_pos] = []
                grid[grid_pos].append(i)
            batch_grids.append(grid)

        return batch_grids

    def filter_new_anchors(self, new_anchors_batch, existing_grids):
        """
        Filter out new anchors that are in the same or nearby grid cells as existing anchors
        Args:
            new_anchors_batch: (batch_size, n_new_anchors, 11)
            existing_grids: list of grids for each batch
        Returns:
            list of filtered indices for each batch
        """
        batch_size = new_anchors_batch.size(0)
        batch_filtered_indices = []

        for b in range(batch_size):
            filtered_indices = []
            new_anchors = new_anchors_batch[b]  # (n_new_anchors, 11)
            if isinstance(self.grid_size, tuple) and len(self.grid_size) > 3:
                new_anchors_decoded = decode_box(new_anchors)
            else:
                new_anchors_decoded = new_anchors
            existing_grid = existing_grids[b]

            for i, anchor in enumerate(new_anchors_decoded):
                if torch.isnan(anchor).any():
                    logging.warning(f"[warning] new_anchors {i} is NaN in AnchorSelector.filter_new_anchors")
                    continue
                if isinstance(self.grid_size, tuple):
                    grid_pos = tuple(
                        [
                            int(anchor[i] // self.grid_size[i])
                            for i in range(len(self.grid_size))
                        ]
                    )
                else:
                    grid_pos = tuple((anchor[:3] // self.grid_size).int().tolist())
                if grid_pos not in existing_grid:
                    filtered_indices.append(i)

            batch_filtered_indices.append(filtered_indices)

        return batch_filtered_indices

    def select_top_anchors(self, new_anchors, existing_anchors, new_anchor_features, confidence, top_k):
        """
        Select top new anchors by avoiding duplications with existing anchors
        Args:
            new_anchors: (batch_size, n_new_anchors, 11)
            existing_anchors: (batch_size, n_existing_anchors, 11)
            new_anchor_features: (batch_size, n_new_anchors, feature_dim)
            confidence: (batch_size, n_new_anchors)
            top_k: number of anchors to select
        Returns:
            selected_anchors: (batch_size, top_k, 11)
            selected_features: (batch_size, top_k, feature_dim)
            selected_confidences: (batch_size, top_k)
        """
        batch_size = new_anchors.size(0)
        device = new_anchors.device

        # Create grids for existing anchors
        existing_grids = self.create_grid(existing_anchors)

        # Filter new anchors for each batch
        batch_filtered_indices = self.filter_new_anchors(new_anchors, existing_grids)

        # Process each batch
        selected_anchors = []
        selected_features = []
        selected_confidences = []

        for b in range(batch_size):
            filtered_indices = torch.tensor(batch_filtered_indices[b], device=device)

            if len(filtered_indices) > 0:
                # Get filtered anchors and their corresponding confidence scores and features
                filtered_new_anchors = new_anchors[b, filtered_indices]
                filtered_confidence = confidence[b, filtered_indices]
                filtered_new_features = new_anchor_features[b, filtered_indices]

                # Sort filtered anchors by confidence
                sorted_indices = filtered_confidence.argsort(descending=True)
                filtered_new_anchors = filtered_new_anchors[sorted_indices]
                filtered_new_features = filtered_new_features[sorted_indices]
                filtered_confidence = filtered_confidence[sorted_indices]
            else:
                filtered_new_anchors = torch.empty((0, new_anchors.size(2)), device=device)
                filtered_new_features = torch.empty((0, new_anchor_features.size(2)), device=device)
                filtered_confidence = torch.empty(0, device=device)

            # If not enough filtered anchors, fill with remaining top anchors by confidence
            if filtered_new_anchors.size(0) < top_k:
                # Get remaining anchors (not in filtered_indices)
                all_indices = torch.arange(new_anchors.size(1), device=device)
                mask = torch.ones(new_anchors.size(1), dtype=torch.bool, device=device)
                mask[filtered_indices] = False
                remaining_indices = all_indices[mask]

                remaining_anchors = new_anchors[b, remaining_indices]
                remaining_features = new_anchor_features[b, remaining_indices]
                remaining_confidence = confidence[b, remaining_indices]

                # Sort remaining anchors by confidence
                sorted_remaining_indices = remaining_confidence.argsort(descending=True)
                remaining_anchors = remaining_anchors[sorted_remaining_indices]
                remaining_features = remaining_features[sorted_remaining_indices]
                remaining_confidence = remaining_confidence[sorted_remaining_indices]

                # Combine filtered and remaining anchors
                num_remaining_needed = top_k - filtered_new_anchors.size(0)
                batch_anchors = torch.cat([filtered_new_anchors, remaining_anchors[:num_remaining_needed]])
                batch_features = torch.cat([filtered_new_features, remaining_features[:num_remaining_needed]])
                batch_confidence = torch.cat([filtered_confidence, remaining_confidence[:num_remaining_needed]])
            else:
                # Take top_k from filtered anchors
                batch_anchors = filtered_new_anchors[:top_k]
                batch_features = filtered_new_features[:top_k]
                batch_confidence = filtered_confidence[:top_k]

            # Pad if necessary to ensure top_k size
            if batch_anchors.size(0) < top_k:
                pad_size = top_k - batch_anchors.size(0)
                batch_anchors = torch.cat([batch_anchors, torch.zeros(pad_size, new_anchors.size(2), device=device)])
                batch_features = torch.cat([batch_features, torch.zeros(pad_size, new_anchor_features.size(2), device=device)])
                batch_confidence = torch.cat([batch_confidence, torch.zeros(pad_size, device=device)])

            selected_anchors.append(batch_anchors)
            selected_features.append(batch_features)
            selected_confidences.append(batch_confidence)

        return (torch.stack(selected_anchors),
                torch.stack(selected_features),
                torch.stack(selected_confidences))
