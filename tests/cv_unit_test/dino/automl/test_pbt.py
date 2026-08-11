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

"""Population Based Training AutoML coverage for DINO."""

from pathlib import Path

from .harness import AUTOML_TEST_MARKS, AutoMLBrainHarness


pytestmark = AUTOML_TEST_MARKS


def test_dino_automl_pbt_resumes_population_and_exploits_best_checkpoint(dino_case):
    runner = AutoMLBrainHarness(
        dino_case,
        "pbt",
        {
            "automl_population_size": 2,
            "automl_max_generations": 2,
            "automl_eval_interval": 1,
            "automl_perturbation_factor": 1.1,
        },
    )

    first_generation = runner.next_recommendations()
    assert len(first_generation) == 2
    for rec in first_generation:
        dino_case.train(rec, target_epoch=1)
    best_first_generation = min(first_generation, key=lambda rec: rec.result)

    next_generation = runner.next_recommendations()
    assert len(next_generation) == 2
    assert all(rec.resume_from_job_id for rec in next_generation)
    exploited = [rec for rec in next_generation if rec.id != best_first_generation.id]
    assert any(rec.resume_from_job_id == best_first_generation.job_id for rec in exploited)

    resumed_records = [
        dino_case.train(
            rec,
            target_epoch=2,
            resume_checkpoint=Path(rec.resume_from_job_id),
        )
        for rec in next_generation
    ]
    assert all(record.started_epochs == [1] for record in resumed_records)

    runner.finish_if_ready()
    assert runner.is_complete()
    assert runner.best_record().checkpoint_path.exists()
