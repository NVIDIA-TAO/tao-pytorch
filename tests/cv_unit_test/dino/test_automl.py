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

"""DINO AutoML integration tests.

These tests intentionally run real short DINO training jobs on a tiny COCO
dataset.  The harness is written so new TAO Pytorch models can copy the DINO
case and provide their own spec/model/datamodule pieces before a matching
tao-skills model exists.
"""

import copy
import json
import os
import site
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")
pl_callbacks = pytest.importorskip("pytorch_lightning.callbacks")
Trainer = pl.Trainer
Callback = pl_callbacks.Callback

from nvidia_tao_core.config.dino.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.deformable_detr.model.ops import functions as deformable_ops_functions
from nvidia_tao_pytorch.cv.deformable_detr.model.ops import modules as deformable_ops_modules
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.dino.model.pl_dino_model import DINOPlModel


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFORMABLE_ATTN_OPS = Path("nvidia_tao_pytorch/cv/deformable_detr/model/ops")
SIBLING_AUTOML_SRC = REPO_ROOT.parent / "tao-automl" / "src"
if SIBLING_AUTOML_SRC.exists():
    sys.path.insert(0, str(SIBLING_AUTOML_SRC))

pytest.importorskip("tao_automl")
from tao_automl.brain.factory import AlgorithmParams, BrainFactory  # noqa: E402
from tao_automl.controller.controller import Controller  # noqa: E402
from tao_automl.search_space.params import generate_hyperparams_to_search  # noqa: E402
from tao_automl.state.state_store import StateStore  # noqa: E402
from tao_automl.types import AutoMLContext, JobStates, Recommendation, ResumeRecommendation  # noqa: E402


pytestmark = [
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


_patch_deformable_attention_loader_for_release_image()


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
    automl_hyperparameters = ("train.optim.lr",)
    custom_param_ranges = {
        "train.optim.lr": {
            "valid_min": 1e-5,
            "valid_max": 1e-4,
        },
    }

    def __init__(self, workspace, dataset_root, annotation_file):
        self.workspace = Path(workspace)
        self.dataset_root = Path(dataset_root)
        self.annotation_file = Path(annotation_file)
        self.records: list[TrainingRecord] = []

    def base_spec(self, results_dir, num_epochs=1):
        cfg = OmegaConf.structured(ExperimentConfig())
        cfg.results_dir = str(results_dir)

        cfg.train.num_gpus = 1
        cfg.train.num_nodes = 1
        cfg.train.num_epochs = num_epochs
        cfg.train.checkpoint_interval = 1
        cfg.train.validation_interval = 1
        cfg.train.activation_checkpoint = False
        cfg.train.precision = "fp32"
        cfg.train.seed = 1234
        cfg.train.optim.lr = 2e-5
        cfg.train.optim.lr_backbone = 2e-6
        cfg.train.optim.lr_linear_proj_mult = 0.1
        cfg.train.optim.lr_scheduler = "StepLR"
        cfg.train.optim.lr_step_size = 1
        cfg.train.optim.lr_decay = 0.9

        cfg.dataset.dataset_type = "default"
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
        cfg.dataset.num_classes = 4
        cfg.dataset.eval_class_ids = [1, 2, 3]
        cfg.dataset.batch_size = 1
        cfg.dataset.workers = 0
        cfg.dataset.pin_memory = False
        cfg.dataset.augmentation.scales = [64]
        cfg.dataset.augmentation.train_random_resize = [64]
        cfg.dataset.augmentation.random_resize_max_size = 64
        cfg.dataset.augmentation.test_random_resize = 64
        cfg.dataset.augmentation.train_random_crop_min = 32
        cfg.dataset.augmentation.train_random_crop_max = 64
        cfg.dataset.augmentation.horizontal_flip_prob = 0.0
        cfg.dataset.augmentation.fixed_padding = True
        cfg.dataset.augmentation.fixed_random_crop = None

        cfg.model.backbone = "resnet_34"
        cfg.model.pretrained_backbone_path = None
        cfg.model.train_backbone = False
        cfg.model.num_feature_levels = 1
        cfg.model.return_interm_indices = [1]
        cfg.model.hidden_dim = 256
        cfg.model.nheads = 4
        cfg.model.enc_layers = 1
        cfg.model.dec_layers = 1
        cfg.model.dim_feedforward = 256
        cfg.model.dec_n_points = 2
        cfg.model.enc_n_points = 2
        cfg.model.num_queries = 8
        cfg.model.num_select = 8
        cfg.model.aux_loss = False
        cfg.model.two_stage_type = "no"
        cfg.model.use_dn = False
        cfg.model.dn_number = 4
        cfg.model.dropout_ratio = 0.0

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
        reported_metric = metric + (float(rec.id) * 1e6)
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
        self.controller.history = self.history
        self.controller._next_id = self._next_id
        return self.controller.is_complete()

    def best_record(self):
        return min(self.model_case.records, key=lambda record: record.reported_metric)


@pytest.fixture(autouse=True)
def _require_training_device():
    if torch.cuda.is_available() or os.getenv("TAO_AUTOML_ALLOW_CPU") == "1":
        return
    pytest.skip("DINO AutoML integration tests run real training; set TAO_AUTOML_ALLOW_CPU=1 to run on CPU.")


@pytest.fixture()
def tiny_coco_dataset(tmp_path):
    dataset_root = tmp_path / "images"
    dataset_root.mkdir()
    rng = np.random.default_rng(1234)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "object", "id": 1, "name": "person"},
            {"supercategory": "object", "id": 2, "name": "face"},
            {"supercategory": "object", "id": 3, "name": "bag"},
        ],
    }

    ann_id = 0
    for image_id in range(6):
        image = Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
        file_name = f"sample_{image_id}.jpg"
        image.save(dataset_root / file_name)
        coco["images"].append({
            "id": image_id,
            "file_name": file_name,
            "height": 64,
            "width": 64,
        })
        for category_id in (1, 2):
            x1 = int(rng.integers(1, 32))
            y1 = int(rng.integers(1, 32))
            w = int(rng.integers(8, 24))
            h = int(rng.integers(8, 24))
            coco["annotations"].append({
                "image_id": image_id,
                "category_id": category_id,
                "id": ann_id,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    annotation_file = tmp_path / "annotations.json"
    annotation_file.write_text(json.dumps(coco))
    return dataset_root, annotation_file


@pytest.fixture()
def dino_case(tmp_path, tiny_coco_dataset):
    dataset_root, annotation_file = tiny_coco_dataset
    return DINOAutoMLHarness(tmp_path / "automl_workspace", dataset_root, annotation_file)


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
        assert 1e-5 <= rec.specs["train.optim.lr"] <= 1e-4

    best = runner.best_record()
    assert best.checkpoint_path.exists()
    assert len(dino_case.records) == 2
    assert runner.is_complete()


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


def test_dino_automl_asha_promotes_best_async_rung_from_checkpoint(dino_case):
    runner = AutoMLBrainHarness(
        dino_case,
        "asha",
        {
            "automl_max_epochs": 2,
            "automl_reduction_factor": 2,
            "epoch_multiplier": 1,
            "automl_max_concurrent": 2,
            "automl_max_trials": 2,
            "automl_min_top_configs": 1,
        },
    )

    first_rung = runner.next_recommendations()
    assert len(first_rung) == 2
    assert [rec.early_stop_epoch for rec in first_rung] == [1, 1]
    [dino_case.train(rec, target_epoch=rec.early_stop_epoch) for rec in first_rung]

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

    runner.finish_if_ready()
    assert runner.is_complete()
    assert runner.best_record().checkpoint_path.exists()


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
    [dino_case.train(rec, target_epoch=1) for rec in first_generation]
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
