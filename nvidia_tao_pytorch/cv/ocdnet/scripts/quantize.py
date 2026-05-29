# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantize an OCDNet model using the configured backend.

This script loads a trained OCDNet checkpoint, prepares the calibration data loader
from the dataset specified in ``quant_calibration_dataset``, runs quantization via
``ModelQuantizer``, and saves the quantized model.
"""

import os

from omegaconf import OmegaConf

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs, logging

from nvidia_tao_pytorch.config.ocdnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.quantization import ModelQuantizer
from nvidia_tao_pytorch.cv.ocdnet.model.pl_ocd_model import OCDnetModel
from nvidia_tao_pytorch.cv.ocdnet.data_loader.pl_ocd_data_module import OCDDataModule
from nvidia_tao_pytorch.cv.ocdnet.utils.util import load_checkpoint


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additionally using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="quantize",
    schema=ExperimentConfig,
)
@monitor_status(name="OCDNet", mode="quantize")
def main(cfg: ExperimentConfig) -> None:
    """Run the quantization process.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration including the ``quantize`` section.
    """
    # Obfuscate logs.
    obfuscate_logs(cfg)

    logging.info("Starting OCDNet quantization")

    # Build the Lightning model and extract the underlying nn.Module
    logging.debug("Loading OCDNet checkpoint")
    if not cfg.quantize.model_path.endswith(".onnx"):
        experiment_config = OmegaConf.to_container(cfg, resolve=True)
        dm = OCDDataModule(experiment_config)
        pl_model = OCDnetModel(experiment_config, dm, "predict")
        checkpoint = load_checkpoint(cfg.quantize.model_path, to_cpu=True)
        pl_model.model.load_state_dict(checkpoint)
        orig_model = pl_model.model
    else:
        orig_model = None  # ModelOpt ONNX backend loads the model from the file.

    # Prepare calibration dataloader via DataModule
    calib_cfg = cfg.dataset.quant_calibration_dataset
    calib_images_dir = getattr(calib_cfg, "images_dir", "")
    if cfg.quantize.mode != "weight_only_ptq" and calib_images_dir:
        dm = OCDDataModule(cfg)
        dm.setup(stage="calibration")
        calibration_loader = dm.calib_dataloader()
    else:
        calibration_loader = None

    # Create quantizer and quantize the model
    quantizer = ModelQuantizer(cfg.quantize)
    quantized_model = quantizer.quantize_model(orig_model, calibration_loader)
    logging.info("Quantization finished; saving model")
    quantizer.save_model(quantized_model, cfg.quantize.results_dir)
    logging.info("OCDNet quantization completed successfully")


if __name__ == "__main__":
    main()
