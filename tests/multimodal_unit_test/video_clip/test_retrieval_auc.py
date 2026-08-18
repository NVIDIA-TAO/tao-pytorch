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

"""Tests that retrieval AUC pairs each score with its own item relevance.

``compute_auc`` sorts its ``scores`` argument internally and indexes ``labels``
with the result, so the two arrays must be in the same order. Ranked labels
alone are not enough -- a gallery-order/rank-order mix silently inverts AUC.
"""

import numpy as np
import pytest

from nvidia_tao_pytorch.multimodal.video_clip.model.evaluation.retrieval import (
    RetrievalEvaluator,
)


@pytest.mark.multimodal_unit
class TestRetrievalAUCAlignment:
    """Gallery is the identity basis, so ranks are exact and hand-checkable."""

    def setup_method(self):
        """Build a query whose ranking order differs from gallery order."""
        self.gallery = np.eye(4, dtype=np.float32)
        # sims proportional to [1, 2, 3, 4] -> ranking order [3, 2, 1, 0],
        # which is NOT gallery order, so a misaligned call cannot coincide.
        self.query = np.array([[1, 2, 3, 4]], dtype=np.float32)
        self.evaluator = RetrievalEvaluator(
            k_values=(1,), compute_auc=True, device="cpu"
        )

    def test_only_lowest_scoring_clip_relevant_is_auc_zero(self):
        """The sole relevant clip ranks last, so AUC is 0."""
        metrics = self.evaluator.evaluate(self.query, self.gallery, [[0]])
        assert abs(metrics.auc - 0.0) < 1e-6

    def test_only_highest_scoring_clip_relevant_is_auc_one(self):
        """The sole relevant clip ranks first, so AUC is 1."""
        metrics = self.evaluator.evaluate(self.query, self.gallery, [[3]])
        assert abs(metrics.auc - 1.0) < 1e-6
