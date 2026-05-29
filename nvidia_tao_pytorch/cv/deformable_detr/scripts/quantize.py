# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantize a Deformable DETR model using the configured backend.

This script loads a trained Deformable DETR checkpoint, prepares the calibration data loader
from the dataset specified in ``quant_calibration_data_sources``, runs quantization via
``ModelQuantizer``, and saves the quantized model.
"""

import os
import logging

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs

from nvidia_tao_pytorch.config.deformable_detr.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.quantization import ModelQuantizer
from nvidia_tao_pytorch.cv.deformable_detr.model.pl_dd_model import DeformableDETRModel
from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additionally using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="quantize",
    schema=ExperimentConfig,
)
@monitor_status(name="Deformable-DETR", mode="quantize")
def main(cfg: ExperimentConfig) -> None:
    """Run the quantization process.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration including the ``quantize`` section.
    """
    # Obfuscate logs.
    obfuscate_logs(cfg)

    logger = logging.getLogger(__name__)
    logger.info("Starting Deformable DETR quantization")

    # Build the Lightning model and extract the underlying nn.Module
    logger.debug("Loading Deformable DETR checkpoint")
    if not cfg.quantize.model_path.endswith(".onnx"):
        pl_model = DeformableDETRModel.load_from_checkpoint(
            cfg.quantize.model_path,
            map_location="cpu",
            experiment_spec=cfg,
        )
        orig_model = pl_model.model
    else:
        orig_model = None  # ModelOpt ONNX backend loads the model from the file.

    # Prepare calibration dataloader via DataModule
    if cfg.quantize.mode != "weight_only_ptq" and cfg.dataset.quant_calibration_data_sources is not None:
        dm = ODDataModule(cfg.dataset)
        dm.setup(stage="calibration")
        calibration_loader = dm.calib_dataloader()
    else:
        calibration_loader = None

    # Create quantizer and quantize the model
    quantizer = ModelQuantizer(cfg.quantize)
    quantized_model = quantizer.quantize_model(orig_model, calibration_loader)
    logger.info("Quantization finished; saving model")
    quantizer.save_model(quantized_model, cfg.quantize.results_dir)
    logger.info("Deformable DETR quantization completed successfully")


if __name__ == "__main__":
    main()
