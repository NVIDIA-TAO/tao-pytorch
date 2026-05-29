# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantize a OneFormer model using the configured backend."""

import os

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs, logging

from nvidia_tao_pytorch.config.oneformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.quantization import ModelQuantizer
from nvidia_tao_pytorch.cv.oneformer.model.pl_oneformer import OneformerPlModule
from nvidia_tao_pytorch.cv.oneformer.dataloader.pl_data_module import SemSegmDataModule


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="quantize",
    schema=ExperimentConfig,
)
@monitor_status(name="OneFormer", mode="quantize")
def main(cfg: ExperimentConfig) -> None:
    """Run the quantization process."""
    obfuscate_logs(cfg)

    logging.info("Starting OneFormer quantization")

    logging.debug("Loading OneFormer checkpoint")
    if not cfg.quantize.model_path.endswith(".onnx"):
        pl_model = OneformerPlModule.load_from_checkpoint(
            cfg.quantize.model_path,
            map_location="cpu",
            experiment_spec=cfg,
        )
        orig_model = pl_model.model
    else:
        orig_model = None

    if cfg.quantize.mode != "weight_only_ptq" and cfg.dataset.quant_calibration_dataset.images_dir:
        dm = SemSegmDataModule(cfg)
        dm.setup(stage="calibration")
        calibration_loader = dm.calib_dataloader()
    else:
        calibration_loader = None

    quantizer = ModelQuantizer(cfg.quantize)
    quantized_model = quantizer.quantize_model(orig_model, calibration_loader)
    logging.info("Quantization finished; saving model")
    quantizer.save_model(quantized_model, cfg.quantize.results_dir)
    logging.info("OneFormer quantization completed successfully")


if __name__ == "__main__":
    main()
