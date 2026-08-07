# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate (embedding-quality) action for DINOv3 SSL.

Offline front-end onto the shared ``core/evaluation`` library, mirroring the
``nvdinov2`` evaluate action: loads a trained checkpoint, wraps the teacher
backbone in the ``(summary, features)`` model adapter, builds an
:class:`EvalContext`, runs every enabled evaluator (currently KNN;
segmentation + retrieval follow), and writes ``results.json``.

The checkpoint follows the sibling ``inference`` / ``convert`` conventions —
either a stripped backbone file (``student_*.pth`` / ``teacher_*.pth``) or
timm-format DINOv3 weights (``.pth`` / ``.safetensors``), loaded via
``restore_pretrained_weights`` (which remaps timm keys and mirrors
student → teacher in the non-distillation case).
"""

import json
import os

import torch

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.distributed.comm import (
    get_local_rank,
    is_dist_avail_and_initialized,
)
from nvidia_tao_pytorch.core.evaluation import (
    EvalContext,
    build_adapter,
    build_enabled_evaluators,
)
from nvidia_tao_pytorch.core.evaluation.datasets.classification import (
    build_classification_loader,
)
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_evaluation_experiment
from nvidia_tao_pytorch.core.tlt_logging import logging, obfuscate_logs
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel


def run_experiment(experiment_config, key):
    """Run the embedding-quality evaluation suite and write ``results.json``."""
    model_path, _ = initialize_evaluation_experiment(experiment_config, key)
    results_dir = experiment_config.results_dir
    eval_cfg = experiment_config.evaluate

    device = torch.device(
        f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")

    # Build the LightningModule and restore the trained backbone weights.
    model = DinoV3PlModel(experiment_config)
    if model_path and model_path.endswith((".tlt", ".pth", ".safetensors")):
        model.pretrained_weights = model_path
        model.restore_pretrained_weights()
        logging.info("Loaded checkpoint: %s", model_path)
    else:
        raise NotImplementedError(
            "evaluate.checkpoint must be a .tlt/.pth/.safetensors backbone weights file.")
    model = model.to(device).eval()

    # Keep DINOv3's memory-efficient (xformers) attention as built. Its RoPE
    # attention path emits fp16 (q/k/v are hard-cast to half); the evaluators
    # run the backbone under bf16 autocast (the evaluate blocks default
    # ``amp=True``), which reconciles the dtype for the fp32 linear layers —
    # the same arrangement the nvdinov2 evaluate action uses. (Therefore keep
    # ``evaluate.*.amp=True`` for DINOv3.)

    # Wrap the teacher backbone in the (summary, features) adapter.
    adapter = build_adapter(
        "dinov3", model,
        patch_size=model.patch_size,
        feature_dim=model.teacher_embed_dim,
    ).to(device)

    distributed = is_dist_avail_and_initialized()
    ctx = EvalContext(
        model=adapter,
        network="dinov3",
        device=device,
        distributed=distributed,
        build_loader=build_classification_loader,
        cfg=eval_cfg,
        results_dir=results_dir,
        cache_dir=eval_cfg.cache_dir,
    )

    evaluators = build_enabled_evaluators(eval_cfg)
    if not evaluators:
        logging.warning("No evaluators enabled in the evaluate config — nothing to run.")

    results = {}
    for evaluator in evaluators:
        logging.info("Running evaluator: %s", evaluator.name)
        results.update(evaluator.run(ctx))

    if get_local_rank() == 0:
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logging.info("Wrote %s: %s", out_path, results)
    return results


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --config_path and --config_name are provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment_spec", schema=ExperimentConfig,
)
@monitor_status(name="DINOv3", mode="evaluate")
def main(cfg: ExperimentConfig) -> None:
    """Run the DINOv3 embedding-quality evaluation."""
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg, key=cfg.encryption_key)


if __name__ == "__main__":
    main()
