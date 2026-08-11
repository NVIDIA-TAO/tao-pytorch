# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
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

"""Bayesian AutoML coverage for DINO."""

from .harness import AUTOML_TEST_MARKS, AutoMLBrainHarness


pytestmark = AUTOML_TEST_MARKS


def test_dino_automl_bayesian_runs_two_real_training_experiments(dino_case):
    runner = AutoMLBrainHarness(
        dino_case,
        "bayesian",
        {"automl_max_recommendations": 2},
    )

    first = runner.next_recommendations()
    assert len(first) == 1
    first_record = dino_case.train(first[0], target_epoch=1)
    assert first_record.started_epochs == [0]

    second = runner.next_recommendations()
    assert len(second) == 1
    second_record = dino_case.train(second[0], target_epoch=1)
    assert second_record.started_epochs == [0]
    for rec in (first[0], second[0]):
        assert 0.00001 <= rec.specs["train.optim.lr"] <= 0.0001

    best = runner.best_record()
    assert best.checkpoint_path.exists()
    assert len(dino_case.records) == 2
    assert runner.is_complete()
