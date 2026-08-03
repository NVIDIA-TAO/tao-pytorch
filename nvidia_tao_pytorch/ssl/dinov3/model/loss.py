# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 losses.

The only genuinely new DINOv3 loss term is :class:`GramLoss` (Gram anchoring). All other
terms (DINO/iBOT/KoLeo) are inherited from ``nvdinov2`` and reused unchanged. The Gram
term regularizes the student's patch-token feature *geometry* toward a frozen Gram teacher
(initialized from the loaded DINOv3 weights), which DINOv3 uses to stabilize dense features
during long pre-training. See the DINOv3 paper, "Gram anchoring".
"""

from torch import nn
from torch.nn import functional as F


class GramLoss(nn.Module):
    """Gram-anchoring loss: MSE between L2-normalized patch-token Gram matrices.

    For a batch of patch tokens ``X`` shaped ``[B, N, C]``, the (cosine) Gram matrix is
    ``G = X_hat @ X_hat^T`` where ``X_hat`` is ``X`` L2-normalized along the channel axis,
    so ``G`` shaped ``[B, N, N]`` holds pairwise patch cosine similarities and is invariant
    to per-token scale. The loss is the mean-squared error between the student's and the
    frozen teacher's Gram matrices. The computation is done in fp32 (normalize *before* the
    Gram product) for numerical stability under autocast.
    """

    def forward(self, student_patch_tokens, teacher_patch_tokens):
        """Compute the Gram-anchoring loss.

        Args:
            student_patch_tokens (torch.Tensor): Student patch tokens ``[B, N, C]``.
            teacher_patch_tokens (torch.Tensor): Frozen-teacher patch tokens ``[B, N, C]``
                (already detached by the caller; same ``[B, N]`` layout as the student).

        Returns:
            torch.Tensor: Scalar Gram-anchoring loss.
        """
        student = F.normalize(student_patch_tokens.float(), p=2, dim=-1)
        teacher = F.normalize(teacher_patch_tokens.float(), p=2, dim=-1)

        gram_student = student @ student.transpose(-2, -1)
        gram_teacher = teacher @ teacher.transpose(-2, -1)

        return F.mse_loss(gram_student, gram_teacher)
