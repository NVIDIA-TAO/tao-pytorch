# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MAE evaluation script.

Two modes, dispatched by config (backward-compatible):

- **Embedding-quality suite** — when any of ``evaluate.{knn,segmentation,retrieval}.enabled``
  is set, the trained MAE encoder is wrapped in the shared ``core/evaluation`` model
  adapter and the enabled evaluators run, writing ``results.json`` (same path as the
  NV-DINOv2 ``evaluate`` action).
- **Supervised classification test** (default, unchanged) — otherwise the original
  ``trainer.test`` path runs the finetune-stage classification metrics.
"""
import json
import logging
import os
import warnings

from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.distributed.comm import get_local_rank, is_dist_avail_and_initialized
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment

from nvidia_tao_pytorch.config.mae.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.mae.dataloader.pl_data_module import MAEDataModule
from nvidia_tao_pytorch.ssl.mae.model.pl_model import MAEPlModule

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
logger = logging.getLogger(__name__)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SUITE_BLOCKS = ("knn", "segmentation", "retrieval")


def _suite_enabled(cfg) -> bool:
    """True if any embedding-suite evaluator block is enabled in the evaluate config."""
    ev = cfg.evaluate
    return any(getattr(getattr(ev, b, None), "enabled", False) for b in _SUITE_BLOCKS)


def _run_embedding_suite(cfg, model_path) -> None:
    """Run the shared embedding-quality suite on the MAE encoder → results.json."""
    import torch

    from nvidia_tao_pytorch.core.evaluation import (
        EvalContext, build_adapter, build_enabled_evaluators,
    )
    from nvidia_tao_pytorch.core.evaluation.datasets.classification import (
        build_classification_loader,
    )

    device = torch.device(
        f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
    pl_model = MAEPlModule.load_from_checkpoint(model_path, map_location="cpu", cfg=cfg)
    pl_model = pl_model.to(device).eval()

    adapter = build_adapter("mae", pl_model).to(device)
    eval_cfg = cfg.evaluate
    ctx = EvalContext(
        model=adapter, network="mae", device=device,
        distributed=is_dist_avail_and_initialized(),
        build_loader=build_classification_loader, cfg=eval_cfg,
        results_dir=cfg.results_dir, cache_dir=eval_cfg.cache_dir,
    )

    results = {}
    for evaluator in build_enabled_evaluators(eval_cfg):
        logger.info("Running evaluator: %s", evaluator.name)
        results.update(evaluator.run(ctx))

    if get_local_rank() == 0:
        os.makedirs(cfg.results_dir, exist_ok=True)
        out_path = os.path.join(cfg.results_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote %s: %s", out_path, results)


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="eval", schema=ExperimentConfig
)
@monitor_status(name="MAE", mode="evaluate")
def run_experiment(cfg: ExperimentConfig) -> None:
    """Run MAE evaluation (embedding suite if enabled, else classification test)."""
    model_path, trainer_kwargs = initialize_evaluation_experiment(cfg)

    if _suite_enabled(cfg):
        logger.info("Running embedding-quality evaluation suite...")
        _run_embedding_suite(cfg, model_path)
        return

    logger.info("Setting up dataloader...")
    data_module = MAEDataModule(cfg=cfg)

    logger.info("Building MAE models...")
    pl_model = MAEPlModule.load_from_checkpoint(
        model_path, map_location="cpu", cfg=cfg)

    trainer = Trainer(**trainer_kwargs)
    trainer.test(pl_model, data_module)


if __name__ == '__main__':
    run_experiment()
