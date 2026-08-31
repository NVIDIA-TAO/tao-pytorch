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

"""Metadata-masked SigLIP loss for CLIP training."""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.multimodal.clip.loss.metadata_mask import (
    build_accessory_match_mask,
    build_attribute_match_mask,
)


class MetadataMaskedSigLipLoss(nn.Module):
    """SigLIP loss with metadata-aware off-diagonal pair handling.

    The normalization intentionally mirrors OpenCLIP's SigLipLoss: valid loss
    terms are summed and divided by local image batch size, not by the number
    of valid terms. In gather mode, each rank keeps local image rows, gathers
    text rows from every rank, and masks metadata-compatible cross-rank image
    text pairs instead of treating them as negatives.
    """

    def __init__(
        self,
        dist_impl: str = "local",
        world_size: int = 1,
        rank: int = 0,
        accessory_aware: bool = False,
        compatible_as_positive: bool = False,
        compatible_positive_weight: float = 1.0,
        compatible_positive_normalization: str = "per_pair",
    ):
        """Initialize metadata-masked SigLIP loss.

        Args:
            dist_impl: Distributed loss implementation. Supports "local" and
                "gather".
            world_size: Distributed world size used by gather mode.
            rank: Distributed rank used to place the local positive diagonal
                inside gathered text chunks.
            accessory_aware: Also require query accessories to be present in
                an image before ignoring a metadata-compatible off-diagonal.
            compatible_as_positive: Promote compatible off-diagonal pairs to
                positives instead of only ignoring them.
            compatible_positive_weight: Weight assigned to promoted terms.
            compatible_positive_normalization: ``per_pair`` applies the full
                weight to every promoted pair. ``per_query`` divides the
                weight by the global promoted-image count for each text.
        """
        super().__init__()
        if dist_impl not in ("local", "gather"):
            raise ValueError(
                "MetadataMaskedSigLipLoss supports dist_impl='local' or "
                f"'gather', got {dist_impl!r}."
            )
        if world_size < 1:
            raise ValueError(f"world_size must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"rank must be in [0, world_size), got rank={rank} and "
                f"world_size={world_size}."
            )
        if dist_impl == "local" and (world_size != 1 or rank != 0):
            raise ValueError(
                "Local metadata-masked SigLIP loss requires world_size=1 and "
                f"rank=0, got world_size={world_size} and rank={rank}."
            )
        self.dist_impl = dist_impl
        self.world_size = world_size
        self.rank = rank
        self.accessory_aware = accessory_aware
        if compatible_positive_weight < 0:
            raise ValueError(
                "compatible_positive_weight must be non-negative, got "
                f"{compatible_positive_weight}."
            )
        if compatible_positive_normalization not in ("per_pair", "per_query"):
            raise ValueError(
                "compatible_positive_normalization must be 'per_pair' or "
                f"'per_query', got {compatible_positive_normalization!r}."
            )
        self.compatible_as_positive = compatible_as_positive
        self.compatible_positive_weight = compatible_positive_weight
        self.compatible_positive_normalization = compatible_positive_normalization
        self.last_compatible_positive_pairs = torch.tensor(0)

    @staticmethod
    def _resolve_positive_text_indices(
        batch_size: int,
        num_texts: int,
        device: torch.device,
        positive_text_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Validate or build the positive text column for each image row."""
        if positive_text_indices is None:
            if num_texts != batch_size:
                raise ValueError(
                    "positive_text_indices is required when text and image "
                    f"batch sizes differ, got {num_texts} and {batch_size}."
                )
            return torch.arange(batch_size, device=device)

        positive_text_indices = positive_text_indices.to(device=device)
        if positive_text_indices.shape != (batch_size,):
            raise ValueError(
                "positive_text_indices must have shape [B_image], got "
                f"{tuple(positive_text_indices.shape)}."
            )
        if torch.any(positive_text_indices < 0) or torch.any(
            positive_text_indices >= num_texts
        ):
            raise ValueError("positive_text_indices must be in [0, B_text).")
        return positive_text_indices

    def get_logits(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute SigLIP image-text logits."""
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits = logits + logit_bias
        return logits

    def get_ground_truth(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        num_texts: int | None = None,
        positive_text_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build rectangular SigLIP labels for local images and texts.

        Labels are negative except at each image's paired text column. Explicit
        positive indices are required when the text and image counts differ,
        as they do after gathering texts across ranks.
        """
        if num_texts is None:
            num_texts = batch_size
        labels = -torch.ones(
            (batch_size, num_texts), device=device, dtype=dtype
        )
        positive_text_indices = self._resolve_positive_text_indices(
            batch_size=batch_size,
            num_texts=num_texts,
            device=device,
            positive_text_indices=positive_text_indices,
        )
        labels[
            torch.arange(batch_size, device=device),
            positive_text_indices,
        ] = 1
        return labels

    def get_metadata_match_mask(
        self,
        image_attr_values: torch.Tensor,
        text_attr_values: torch.Tensor,
        image_accessory_ids: torch.Tensor | None = None,
        text_accessory_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build metadata compatibility aligned to ``[B_image, B_text]``."""
        metadata_match = build_attribute_match_mask(
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
        )
        if self.accessory_aware:
            if image_accessory_ids is None or text_accessory_ids is None:
                raise ValueError(
                    "Accessory-aware SigLIP masking requires "
                    "image_accessory_ids and text_accessory_ids."
                )
            accessory_match = build_accessory_match_mask(
                image_accessory_ids=image_accessory_ids,
                text_accessory_ids=text_accessory_ids,
            )
            metadata_match = metadata_match & accessory_match
        return metadata_match.T

    def get_valid_term_mask(
        self,
        image_attr_values: torch.Tensor,
        text_attr_values: torch.Tensor,
        image_accessory_ids: torch.Tensor | None = None,
        text_accessory_ids: torch.Tensor | None = None,
        positive_text_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a valid-term mask aligned to logits ``[B_image, B_text]``.

        Metadata-compatible off-diagonal pairs are excluded from the loss. In
        accessory-aware mode, both scalar attributes and required accessories
        must match before a pair is excluded. Paired positives always remain
        valid.
        """
        metadata_match = self.get_metadata_match_mask(
            image_attr_values=image_attr_values,
            text_attr_values=text_attr_values,
            image_accessory_ids=image_accessory_ids,
            text_accessory_ids=text_accessory_ids,
        )
        batch_size = image_attr_values.shape[0]
        valid_terms = ~metadata_match
        positive_text_indices = self._resolve_positive_text_indices(
            batch_size=batch_size,
            num_texts=text_attr_values.shape[0],
            device=metadata_match.device,
            positive_text_indices=positive_text_indices,
        )
        valid_terms[
            torch.arange(batch_size, device=metadata_match.device),
            positive_text_indices,
        ] = True
        return valid_terms

    def get_targets_and_term_weights(
        self,
        labels: torch.Tensor,
        valid_terms: torch.Tensor,
        metadata_match: torch.Tensor,
        positive_text_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Promote compatible pairs and return weighted SigLIP targets."""
        term_weights = torch.ones_like(labels)
        promoted = torch.zeros_like(valid_terms)
        if not self.compatible_as_positive:
            return labels, valid_terms, term_weights, promoted

        promoted = metadata_match.clone()
        promoted[
            torch.arange(labels.shape[0], device=labels.device),
            positive_text_indices,
        ] = False
        labels = labels.clone()
        valid_terms = valid_terms.clone()
        labels[promoted] = 1
        valid_terms[promoted] = True

        if self.compatible_positive_normalization == "per_pair":
            term_weights[promoted] = self.compatible_positive_weight
        else:
            per_query_count = promoted.sum(dim=0, dtype=labels.dtype)
            if self.dist_impl == "gather" and self.world_size > 1:
                dist.all_reduce(per_query_count, op=dist.ReduceOp.SUM)
            normalized = (
                self.compatible_positive_weight / per_query_count.clamp_min(1)
            ).expand(labels.shape[0], -1)
            term_weights[promoted] = normalized[promoted]
        return labels, valid_terms, term_weights, promoted

    def _validate_distributed_context(self) -> None:
        """Ensure configured gather coordinates match the default group."""
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "MetadataMaskedSigLipLoss gather mode requires an initialized "
                "torch.distributed process group."
            )
        actual_world_size = dist.get_world_size()
        actual_rank = dist.get_rank()
        if (actual_world_size, actual_rank) != (self.world_size, self.rank):
            raise RuntimeError(
                "MetadataMaskedSigLipLoss distributed context mismatch: "
                f"configured world_size={self.world_size}, rank={self.rank}; "
                f"process group world_size={actual_world_size}, "
                f"rank={actual_rank}."
            )

    def _gather_no_grad(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Gather metadata tensors without autograd."""
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        return tuple(gathered)

    def _validate_equal_gather_batch_size(
        self,
        local_batch_size: int,
        device: torch.device,
    ) -> None:
        """Fail before tensor gathers when ranks have unequal local batches."""
        local_size = torch.tensor(
            [local_batch_size],
            device=device,
            dtype=torch.long,
        )
        gathered_sizes = [
            torch.empty_like(local_size) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_sizes, local_size)
        batch_sizes = tuple(
            int(batch_size.item()) for batch_size in gathered_sizes
        )
        if len(set(batch_sizes)) != 1:
            raise RuntimeError(
                "MetadataMaskedSigLipLoss gather mode requires equal local "
                "text batch sizes on every rank before all_gather; got "
                f"{batch_sizes}. Use a distributed loader with full batches "
                "(for example, drop_last=True)."
            )

    def _gather_text_with_metadata(
        self,
        text_features: torch.Tensor,
        text_attr_values: torch.Tensor,
        text_accessory_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Gather text candidates and aligned metadata in rank order.

        Feature gathering preserves autograd so text towers receive cross-rank
        gradients. Integer metadata is gathered without autograd.
        """
        if self.dist_impl == "local" or self.world_size == 1:
            return text_features, text_attr_values, text_accessory_ids
        self._validate_distributed_context()
        self._validate_equal_gather_batch_size(
            local_batch_size=text_features.shape[0],
            device=text_features.device,
        )

        gathered_text_features = dist_nn.all_gather(text_features)
        all_text_features = torch.cat(gathered_text_features, dim=0)
        all_text_attr_values = torch.cat(
            self._gather_no_grad(text_attr_values),
            dim=0,
        )
        all_text_accessory_ids = None
        if text_accessory_ids is not None:
            all_text_accessory_ids = torch.cat(
                self._gather_no_grad(text_accessory_ids),
                dim=0,
            )
        return all_text_features, all_text_attr_values, all_text_accessory_ids

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor | None,
        image_attr_values: torch.Tensor,
        text_attr_values: torch.Tensor,
        image_accessory_ids: torch.Tensor | None = None,
        text_accessory_ids: torch.Tensor | None = None,
        output_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute metadata-masked SigLIP loss for a local training batch.

        Feature and metadata inputs describe aligned local image-text pairs.
        Gather mode keeps image rows local while collecting text features and
        metadata across ranks and requires every rank to contribute the same
        local text batch size. Metadata-compatible off-diagonal pairs are
        excluded, paired positives are retained, and the valid loss terms are
        normalized by the local image batch size. When ``output_dict`` is
        true, the scalar is returned under ``contrastive_loss``.

        Args:
            image_features: Local image features with shape ``[B, D]``.
            text_features: Paired local text features with shape ``[B, D]``.
            logit_scale: Scalar multiplier applied to image-text similarities.
            logit_bias: Optional scalar bias added to image-text similarities.
            image_attr_values: Image attributes with shape ``[B, A]``.
            text_attr_values: Paired text attributes with shape ``[B, A]``.
            image_accessory_ids: Optional padded image accessory IDs with shape
                ``[B, K_image]``.
            text_accessory_ids: Optional padded text accessory IDs with shape
                ``[B, K_text]``.
            output_dict: Return the loss under ``contrastive_loss`` when true.
        """
        if image_features.shape[0] != text_features.shape[0]:
            raise ValueError(
                "image_features and text_features must have the same local "
                f"batch size, got {image_features.shape[0]} and "
                f"{text_features.shape[0]}."
            )
        if image_attr_values.shape[0] != image_features.shape[0]:
            raise ValueError(
                "image_attr_values batch size must match image_features, got "
                f"{image_attr_values.shape[0]} and {image_features.shape[0]}."
            )
        if text_attr_values.shape[0] != text_features.shape[0]:
            raise ValueError(
                "text_attr_values batch size must match text_features, got "
                f"{text_attr_values.shape[0]} and {text_features.shape[0]}."
            )

        image_attr_values = image_attr_values.to(device=image_features.device)
        text_attr_values = text_attr_values.to(device=image_features.device)
        if image_accessory_ids is not None:
            if image_accessory_ids.shape[0] != image_features.shape[0]:
                raise ValueError(
                    "image_accessory_ids batch size must match "
                    "image_features, "
                    f"got {image_accessory_ids.shape[0]} and "
                    f"{image_features.shape[0]}."
                )
            image_accessory_ids = image_accessory_ids.to(
                device=image_features.device
            )
        if text_accessory_ids is not None:
            if text_accessory_ids.shape[0] != text_features.shape[0]:
                raise ValueError(
                    "text_accessory_ids batch size must match text_features, "
                    f"got {text_accessory_ids.shape[0]} and "
                    f"{text_features.shape[0]}."
                )
            text_accessory_ids = text_accessory_ids.to(
                device=image_features.device
            )

        (
            candidate_text_features,
            candidate_text_attr_values,
            candidate_text_accessory_ids,
        ) = self._gather_text_with_metadata(
            text_features=text_features,
            text_attr_values=text_attr_values,
            text_accessory_ids=text_accessory_ids,
        )
        local_batch = image_features.shape[0]
        positive_text_indices = torch.arange(
            local_batch,
            device=image_features.device,
        )
        if self.dist_impl == "gather" and self.world_size > 1:
            positive_text_indices = (
                positive_text_indices + self.rank * text_features.shape[0]
            )

        logits = self.get_logits(
            image_features=image_features,
            text_features=candidate_text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )
        labels = self.get_ground_truth(
            device=image_features.device,
            dtype=image_features.dtype,
            batch_size=local_batch,
            num_texts=candidate_text_features.shape[0],
            positive_text_indices=positive_text_indices,
        )
        valid_terms = self.get_valid_term_mask(
            image_attr_values=image_attr_values,
            text_attr_values=candidate_text_attr_values,
            image_accessory_ids=image_accessory_ids,
            text_accessory_ids=candidate_text_accessory_ids,
            positive_text_indices=positive_text_indices,
        )
        metadata_match = self.get_metadata_match_mask(
            image_attr_values=image_attr_values,
            text_attr_values=candidate_text_attr_values,
            image_accessory_ids=image_accessory_ids,
            text_accessory_ids=candidate_text_accessory_ids,
        )
        labels, valid_terms, term_weights, promoted = (
            self.get_targets_and_term_weights(
                labels=labels,
                valid_terms=valid_terms,
                metadata_match=metadata_match,
                positive_text_indices=positive_text_indices,
            )
        )
        self.last_compatible_positive_pairs = promoted.sum().detach()
        loss_terms = -F.logsigmoid(labels * logits) * term_weights
        loss = loss_terms.masked_select(valid_terms).sum()
        loss = loss / local_batch
        if output_dict:
            output = {"contrastive_loss": loss}
            if self.compatible_as_positive:
                output["compatible_positive_pairs"] = promoted.sum()
            return output
        return loss
