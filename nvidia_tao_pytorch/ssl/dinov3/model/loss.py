# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 losses.

The genuinely new DINOv3 loss terms are the *preservation* terms; all others (DINO/iBOT/
KoLeo) are inherited from ``nvdinov2`` and reused unchanged.

* :class:`GramLoss` (Gram anchoring) regularizes the student's **patch-token** feature
  geometry toward a frozen anchor teacher (initialized from the loaded DINOv3 weights),
  which DINOv3 uses to stabilize dense features during long pre-training. See the DINOv3
  paper, "Gram anchoring".
* :class:`ClsPreservationLoss` regularizes the student's **CLS/global embedding** toward the
  same frozen anchor teacher, guarding the geometry that k-NN / linear-probe / retrieval
  consumers depend on. It reads the ``x_norm_clstoken`` of the anchor forward that Gram
  anchoring already runs, so it costs no extra teacher pass.
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


class ClsPreservationLoss(nn.Module):
    """CLS-token preservation: MSE + cosine distance against a frozen anchor teacher.

    Two complementary views of the same drift. The **MSE** term pins the absolute embedding
    (magnitude included); the **cosine** term pins only the direction, which is what k-NN and
    retrieval actually consume. Both are computed in fp32 for stability under autocast, and
    both are exactly zero when the student equals the anchor -- the property gate G2.1 checks
    at step 0 (LoRA is an identity at init, so the run must start at ~0 preservation loss).

    The weights are applied by the caller, so a weight of 0 drops that term from the total
    while its value is still logged as a drift diagnostic.
    """

    def forward(self, student_cls_token, teacher_cls_token):
        """Compute the CLS MSE and cosine preservation terms.

        Args:
            student_cls_token (torch.Tensor): Student CLS embeddings ``[B, C]``.
            teacher_cls_token (torch.Tensor): Frozen-anchor CLS embeddings ``[B, C]``
                (already detached by the caller).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(cls_mse, cls_cosine)`` scalars, where
            ``cls_cosine`` is ``1 - mean cosine similarity`` (0 when perfectly aligned).
        """
        student = student_cls_token.float()
        teacher = teacher_cls_token.float()

        cls_mse = F.mse_loss(student, teacher)
        cls_cosine = 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()

        return cls_mse, cls_cosine
