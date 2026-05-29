# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantize a SegFormer model using the configured backend."""

import os

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs, logging

from nvidia_tao_pytorch.config.segformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.quantization import ModelQuantizer
from nvidia_tao_pytorch.cv.segformer.model.segformer_pl_model import SegFormerPlModel
from nvidia_tao_pytorch.cv.segformer.dataloader.pl_segformer_data_module import SFDataModule


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="quantize",
    schema=ExperimentConfig,
)
@monitor_status(name="SegFormer", mode="quantize")
def main(cfg: ExperimentConfig) -> None:
    """Run the quantization process."""
    obfuscate_logs(cfg)

    logging.info("Starting SegFormer quantization")

    logging.debug("Loading SegFormer checkpoint")
    if not cfg.quantize.model_path.endswith(".onnx"):
        pl_model = SegFormerPlModel.load_from_checkpoint(
            cfg.quantize.model_path,
            map_location="cpu",
            experiment_spec=cfg,
        )
        orig_model = pl_model.model
    else:
        orig_model = None

    if cfg.quantize.mode != "weight_only_ptq" and cfg.dataset.segment.quant_calibration_dataset.images_dir:
        dm = SFDataModule(cfg.dataset.segment)
        dm.setup(stage="calibration")
        calibration_loader = dm.calib_dataloader()
    else:
        calibration_loader = None

    quantizer = ModelQuantizer(cfg.quantize)
    quantized_model = quantizer.quantize_model(orig_model, calibration_loader)
    logging.info("Quantization finished; saving model")
    quantizer.save_model(quantized_model, cfg.quantize.results_dir)
    logging.info("SegFormer quantization completed successfully")


if __name__ == "__main__":
    main()
