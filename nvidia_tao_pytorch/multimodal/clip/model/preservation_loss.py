# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Geometry-preserving regularization losses for CLIP domain adaptation.

Creates a frozen teacher copy of the pretrained model and computes
preservation losses that constrain embedding drift during fine-tuning.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.core.tlt_logging import logging


class PreservationLoss(nn.Module):
    """Computes preservation losses between student and frozen teacher embeddings.

    Three loss components:
        1. Embedding MSE: L2 distance between student and teacher embeddings.
        2. Cosine preservation: 1 - cosine_similarity(student, teacher).
        3. Similarity matrix preservation: MSE between student and teacher
           image-text similarity matrices (preserves retrieval geometry).

    Args:
        teacher_model: Frozen copy of the pretrained model.
        reg_config: CLIPRegularizationConfig with loss weights.
    """

    def __init__(self, teacher_model, reg_config):
        """Initialize PreservationLoss with frozen teacher and config."""
        super().__init__()
        self.teacher = teacher_model
        self.embedding_mse_weight = reg_config.embedding_mse_weight
        self.cosine_weight = reg_config.cosine_weight
        self.similarity_weight = reg_config.similarity_weight

    @torch.no_grad()
    def _teacher_forward(self, image, text):
        """Run teacher forward pass without gradients.

        Returns:
            Tuple of (teacher_image_features, teacher_text_features).
        """
        teacher_out = self.teacher(image=image, text=text)
        # teacher forward returns (img_feat, txt_feat, logit_scale, logit_bias)
        return teacher_out[0], teacher_out[1]

    def forward(self, student_image_feat, student_text_feat, image, text):
        """Compute preservation losses.

        Args:
            student_image_feat: Student image embeddings (B, D), normalized.
            student_text_feat: Student text embeddings (B, D), normalized.
            image: Raw image input (for teacher forward).
            text: Raw text input (for teacher forward).

        Returns:
            Dict with keys:
                - 'preservation_total': Weighted sum of all preservation losses.
                - 'embedding_mse': Unweighted embedding MSE loss.
                - 'cosine': Unweighted cosine preservation loss.
                - 'similarity': Unweighted similarity matrix preservation loss.
        """
        teacher_image_feat, teacher_text_feat = self._teacher_forward(
            image, text
        )

        losses = {}

        # 1. Embedding MSE loss
        emb_mse = (
            F.mse_loss(student_image_feat, teacher_image_feat) +
            F.mse_loss(student_text_feat, teacher_text_feat)
        ) / 2.0
        losses['embedding_mse'] = emb_mse

        # 2. Cosine preservation loss (1 - cosine similarity)
        cos_img = 1.0 - F.cosine_similarity(
            student_image_feat, teacher_image_feat, dim=-1
        ).mean()
        cos_txt = 1.0 - F.cosine_similarity(
            student_text_feat, teacher_text_feat, dim=-1
        ).mean()
        cosine_loss = (cos_img + cos_txt) / 2.0
        losses['cosine'] = cosine_loss

        # 3. Similarity matrix preservation loss
        student_sim = student_image_feat @ student_text_feat.T
        teacher_sim = teacher_image_feat @ teacher_text_feat.T
        sim_loss = F.mse_loss(student_sim, teacher_sim)
        losses['similarity'] = sim_loss

        # Weighted total
        total = (
            self.embedding_mse_weight * emb_mse +
            self.cosine_weight * cosine_loss +
            self.similarity_weight * sim_loss
        )
        losses['preservation_total'] = total

        return losses


def build_preservation_loss(model, reg_config):
    """Create a PreservationLoss with a frozen teacher copy of the model.

    The teacher is a deep copy of the model with all parameters frozen
    and set to eval mode. This must be called BEFORE LoRA injection
    so the teacher retains the original pretrained weights.

    Args:
        model: The pretrained BaseCLIPAdapter (before LoRA injection).
        reg_config: CLIPRegularizationConfig.

    Returns:
        PreservationLoss instance, or None if reg_config.enabled is False.
    """
    if not reg_config.enabled:
        return None

    logging.info("Creating frozen teacher model for preservation losses...")
    teacher = copy.deepcopy(model)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    teacher_params = sum(p.numel() for p in teacher.parameters())
    logging.info(
        f"Frozen teacher created: {teacher_params:,} params. "
        f"Weights: emb_mse={reg_config.embedding_mse_weight}, "
        f"cosine={reg_config.cosine_weight}, "
        f"sim={reg_config.similarity_weight}"
    )

    return PreservationLoss(teacher, reg_config)
