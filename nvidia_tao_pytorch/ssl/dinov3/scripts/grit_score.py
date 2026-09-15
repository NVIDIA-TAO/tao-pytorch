# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run GRIT scoring through the standard TAO DINOv3 experiment interface."""

import os
from pathlib import Path

from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.ssl.dinov3.data_refinement.cli import run


spec_root = Path(__file__).resolve().parents[1] / "experiment_specs"


@hydra_runner(config_path=str(spec_root), config_name="experiment_spec", schema=ExperimentConfig)
def main(cfg: ExperimentConfig) -> None:
    """Score a manifest and publish normal TAO status and experiment artifacts."""
    # GRIT is not distributed: an externally launched worker group would score
    # the entire manifest on every rank and race on one output.
    # Validate before monitor_status creates or changes any shared artifacts.
    sizes = ("WORLD_SIZE", "LOCAL_WORLD_SIZE")
    ranks = ("RANK", "LOCAL_RANK", "NODE_RANK")
    if any(int(os.environ.get(name, "1")) != 1 for name in sizes) or any(
        int(os.environ.get(name, "0")) != 0 for name in ranks
    ):
        raise ValueError("DINOv3 grit_score requires a single process on one node")
    _run_experiment(cfg)


@monitor_status(name="DINOv3", mode="grit_score")
def _run_experiment(cfg: ExperimentConfig) -> None:
    """Publish scoring artifacts after validating the launcher topology."""
    obfuscate_logs(cfg)
    config = OmegaConf.to_container(cfg.grit_score, resolve=True)
    config["output_dir"] = config.pop("results_dir")
    # Preserve adapter defaults for optional values omitted from the spec.
    config = {key: value for key, value in config.items() if value is not None}
    if not config["input_parquet"]:
        raise ValueError("grit_score.input_parquet must name a target manifest")
    run(config)


if __name__ == "__main__":
    main()
