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

"""Hyperband AutoML coverage for DINO."""

from pathlib import Path

from .harness import AUTOML_TEST_MARKS, AutoMLBrainHarness


pytestmark = AUTOML_TEST_MARKS


def test_dino_automl_hyperband_promotes_best_rung_from_checkpoint(dino_case):
    runner = AutoMLBrainHarness(
        dino_case,
        "hyperband",
        {
            "automl_max_epochs": 2,
            "automl_reduction_factor": 2,
            "epoch_multiplier": 1,
        },
    )

    first_rung = runner.next_recommendations()
    assert len(first_rung) == 2
    assert [rec.early_stop_epoch for rec in first_rung] == [1, 1]
    first_records = [dino_case.train(rec, target_epoch=rec.early_stop_epoch) for rec in first_rung]
    assert all(record.started_epochs == [0] for record in first_records)

    best_first_rung = min(first_rung, key=lambda rec: rec.result)
    promotion = runner.next_recommendations()
    assert len(promotion) == 1
    promoted = promotion[0]
    assert promoted.id == best_first_rung.id
    assert promoted.resume_from_job_id == best_first_rung.job_id
    assert promoted.early_stop_epoch == 2

    resumed_record = dino_case.train(
        promoted,
        target_epoch=promoted.early_stop_epoch,
        resume_checkpoint=Path(promoted.resume_from_job_id),
    )
    assert resumed_record.started_epochs == [1]
    assert resumed_record.resume_checkpoint == Path(best_first_rung.job_id)

    runner.finish_if_ready()
    assert runner.is_complete()
    assert runner.best_record().checkpoint_path.exists()
