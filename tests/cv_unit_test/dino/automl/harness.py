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

"""Reusable DINO AutoML integration-test harness."""

import copy
import os
import site
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[4]
MINIMAL_DINO_SPEC = Path(__file__).resolve().parent / "specs" / "dino_minimal.yaml"
DEFORMABLE_ATTN_OPS = Path("nvidia_tao_pytorch/cv/deformable_detr/model/ops")

if os.getenv("TAO_AUTOML_TESTS") != "1":
    pytest.skip(
        "DINO AutoML integration tests are opt-in; set TAO_AUTOML_TESTS=1 to run them.",
        allow_module_level=True,
    )


def _add_tao_automl_source_path():
    """Prefer an explicit or sibling tao-automl checkout for local repo testing."""
    candidates = []
    env_src = os.getenv("TAO_AUTOML_SRC")
    if env_src:
        candidates.append(Path(env_src).expanduser())
    candidates.append(REPO_ROOT.parent / "tao-automl" / "src")

    for candidate in candidates:
        if (candidate / "tao_automl").exists():
            sys.path.append(str(candidate))
            return candidate
    return None


torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")
pl_callbacks = pytest.importorskip("pytorch_lightning.callbacks")
Trainer = pl.Trainer
Callback = pl_callbacks.Callback

from nvidia_tao_core.config.dino.default_config import ExperimentConfig  # noqa: E402
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule  # noqa: E402
from nvidia_tao_pytorch.cv.deformable_detr.model.ops import functions as deformable_ops_functions  # noqa: E402
from nvidia_tao_pytorch.cv.deformable_detr.model.ops import modules as deformable_ops_modules  # noqa: E402
from nvidia_tao_pytorch.cv.dino.model.pl_dino_model import DINOPlModel  # noqa: E402

pytest.importorskip(
    "tao_automl",
    reason="Install tao-automl in the image, set TAO_AUTOML_SRC, or place ../tao-automl/src next to tao-pytorch.",
)
from tao_automl.brain.factory import AlgorithmParams, BrainFactory  # noqa: E402
from tao_automl.controller.controller import Controller  # noqa: E402
from tao_automl.search_space.params import generate_hyperparams_to_search  # noqa: E402
from tao_automl.state.state_store import StateStore  # noqa: E402
from tao_automl.types import AutoMLContext, JobStates, Recommendation, ResumeRecommendation  # noqa: E402


AUTOML_TEST_MARKS = [
    pytest.mark.automl,
    pytest.mark.integration,
    pytest.mark.dino,
    pytest.mark.train,
]


def _patch_deformable_attention_loader_for_release_image():
    original_load_ops = deformable_ops_functions.load_ops
    loaded_paths = set()

    def load_ops_from_source_or_release_image(ops_dir, lib_name):
        source_candidate = Path(ops_dir) / lib_name
        if source_candidate.exists():
            selected_library = source_candidate
        else:
            selected_library = _find_release_image_deformable_attention_lib(lib_name)

        if selected_library is None:
            return original_load_ops(ops_dir, lib_name)

        selected_library = str(selected_library)
        if selected_library not in loaded_paths:
            torch.ops.load_library(selected_library)
            loaded_paths.add(selected_library)
        return None

    deformable_ops_functions.load_ops = load_ops_from_source_or_release_image
    deformable_ops_modules.load_ops = load_ops_from_source_or_release_image


def _find_release_image_deformable_attention_lib(lib_name):
    search_roots = []
    for package_path in site.getsitepackages() + [site.getusersitepackages()]:
        if package_path:
            search_roots.append(Path(package_path))
    search_roots.extend(Path(path) for path in sys.path if path)

    for search_root in search_roots:
        candidate = search_root / DEFORMABLE_ATTN_OPS / lib_name
        if candidate.exists() and not candidate.resolve().is_relative_to(REPO_ROOT):
            return candidate
    return None


@dataclass
class TrainingRecord:
    """Result from one short training launch."""

    rec_id: int
    metric: float
    reported_metric: float
    checkpoint_path: Path
    started_epochs: list[int]
    target_epoch: int
    resume_checkpoint: Path | None = None


class EpochTraceCallback(Callback):
    """Record epoch numbers actually entered by Lightning."""

    def __init__(self):
        self.started_epochs = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.started_epochs.append(int(trainer.current_epoch))


class DINOAutoMLHarness:
    """DINO implementation of the reusable AutoML training harness."""

    network = "dino"
    metric = "loss"

    def __init__(self, workspace, dataset_root, annotation_file, spec_path=MINIMAL_DINO_SPEC):
        self.workspace = Path(workspace)
        self.dataset_root = Path(dataset_root)
        self.annotation_file = Path(annotation_file)
        self.spec_path = Path(spec_path)
        self.spec = OmegaConf.load(self.spec_path)
        self.automl_hyperparameters = tuple(OmegaConf.to_container(self.spec.automl.hyperparameters, resolve=True))
        self.custom_param_ranges = OmegaConf.to_container(self.spec.automl.custom_param_ranges, resolve=True)
        self.records: list[TrainingRecord] = []

    def base_spec(self, results_dir, num_epochs=1):
        cfg = OmegaConf.merge(
            OmegaConf.structured(ExperimentConfig()),
            copy.deepcopy(self.spec.experiment),
        )
        cfg.results_dir = str(results_dir)
        cfg.train.num_epochs = num_epochs
        cfg.dataset.train_data_sources = [
            {"image_dir": str(self.dataset_root), "json_file": str(self.annotation_file)}
        ]
        cfg.dataset.val_data_sources = [
            {"image_dir": str(self.dataset_root), "json_file": str(self.annotation_file)}
        ]
        cfg.dataset.test_data_sources = {
            "image_dir": str(self.dataset_root),
            "json_file": str(self.annotation_file),
        }
        return cfg

    def train(self, rec, target_epoch, resume_checkpoint=None):
        run_dir = self.workspace / f"rec_{rec.id}_run_{len(self.records):02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.base_spec(run_dir, num_epochs=target_epoch)
        _apply_flat_overrides(cfg, rec.specs)

        dm = ODDataModule(cfg.dataset)
        dm.setup(stage="fit")
        model = DINOPlModel(cfg)
        epoch_trace = EpochTraceCallback()

        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            num_nodes=1,
            max_epochs=target_epoch,
            default_root_dir=str(run_dir),
            logger=False,
            enable_model_summary=False,
            gradient_clip_val=cfg.train.clip_grad_norm,
            use_distributed_sampler=False,
            num_sanity_val_steps=0,
            limit_train_batches=1,
            limit_val_batches=0,
            callbacks=[epoch_trace],
        )
        trainer.fit(model, dm, ckpt_path=str(resume_checkpoint) if resume_checkpoint else None)

        metric = _metric_from_trainer(trainer)
        checkpoint_path = _latest_checkpoint(run_dir)
        # Keep ranking deterministic for algorithm-contract assertions while
        # still deriving the objective from a real training result.
        # Make the objective depend on the reported metric, not insertion
        # order: this minimization objective rewards later recommendation IDs.
        reported_metric = metric - (float(rec.id) * 1e6)
        rec.assign_job_id(str(checkpoint_path))
        rec.update_result(reported_metric)
        rec.update_status(JobStates.success)

        record = TrainingRecord(
            rec_id=rec.id,
            metric=metric,
            reported_metric=reported_metric,
            checkpoint_path=checkpoint_path,
            started_epochs=list(epoch_trace.started_epochs),
            target_epoch=target_epoch,
            resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint else None,
        )
        self.records.append(record)
        return record

    def spec_dict(self):
        return OmegaConf.to_container(
            self.base_spec(self.workspace / "base", num_epochs=2),
            resolve=True,
            throw_on_missing=False,
        )


class AutoMLBrainHarness:
    """Small local runner for AutoML brains used by Pytorch integration tests."""

    def __init__(self, model_case, algorithm, settings):
        self.model_case = model_case
        self.algorithm = algorithm
        self.workspace = model_case.workspace / algorithm
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.state_store = StateStore(str(self.workspace))
        self.context = AutoMLContext(
            id=f"dino_{algorithm}",
            network=model_case.network,
            action="train",
            workspace_path=str(self.workspace),
            metric=model_case.metric,
            handler_id=f"dino_{algorithm}",
        )
        self.train_specs = model_case.spec_dict()
        self.state_store.save_job_specs(self.context.id, copy.deepcopy(self.train_specs))
        self.state_store.save_custom_param_ranges(
            self.context.handler_id,
            copy.deepcopy(model_case.custom_param_ranges),
        )
        params, self.param_names = generate_hyperparams_to_search(
            network=model_case.network,
            action="train",
            train_specs=self.train_specs,
            automl_hyperparameters=list(model_case.automl_hyperparameters),
        )
        self.settings = AlgorithmParams.from_dict({"algorithm": algorithm, **settings})
        self.brain = BrainFactory.create_brain(
            algorithm=algorithm,
            context=self.context,
            state_store=self.state_store,
            network=model_case.network,
            parameters=params,
            params=self.settings,
            metric=model_case.metric,
        )
        self.controller = Controller(
            brain=self.brain,
            context=self.context,
            state_store=self.state_store,
            settings=self.settings,
            metric=model_case.metric,
            algorithm=algorithm,
            parameter_names=self.param_names,
        )
        self.history: list[Recommendation] = []
        # Controller.is_complete() owns the Bayesian/BFBO max-recommendation
        # contract and reads controller.history.  Keep it aliased to the
        # harness history that receives each generated recommendation.
        self.controller.history = self.history
        self._next_id = 0

    def next_recommendations(self):
        raw_recs = self.brain.generate_recommendations(self.history)
        recs = []
        for raw in raw_recs:
            if isinstance(raw, ResumeRecommendation):
                rec = Recommendation(raw.id, raw.specs, self.model_case.metric)
                rec.resume_from_job_id = raw.resume_from_job_id or raw.job_id
                rec.early_stop_epoch = getattr(self.brain, "epoch_number", None)
            else:
                rec = Recommendation(self._next_id, raw, self.model_case.metric)
                rec.early_stop_epoch = getattr(self.brain, "epoch_number", None)
                self._next_id += 1
            self.history.append(rec)
            recs.append(rec)
        self.brain.save_state()
        return recs

    def finish_if_ready(self):
        self.brain.generate_recommendations(self.history)
        self.brain.save_state()

    def is_complete(self):
        brain_done = getattr(self.brain, "done", None)
        if callable(brain_done):
            return brain_done()
        return self.controller.is_complete()

    def best_record(self):
        return min(self.model_case.records, key=lambda record: record.reported_metric)


def _apply_flat_overrides(cfg, overrides):
    for key, value in overrides.items():
        if isinstance(value, np.generic):
            value = value.item()
        OmegaConf.update(cfg, key, value, merge=True)


def _metric_from_trainer(trainer):
    for key in ("train_loss_epoch", "train_loss"):
        value = trainer.callback_metrics.get(key)
        if value is not None:
            if hasattr(value, "detach"):
                return float(value.detach().cpu())
            return float(value)
    raise AssertionError(f"No train loss metric found in callback metrics: {trainer.callback_metrics}")


def _latest_checkpoint(results_dir):
    latest = Path(results_dir) / "dino_model_latest.pth"
    if latest.exists():
        return latest
    checkpoints = sorted(Path(results_dir).glob("*.pth"))
    assert checkpoints, f"No checkpoint written under {results_dir}"
    return checkpoints[-1]
