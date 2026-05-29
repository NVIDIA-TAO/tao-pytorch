# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantize a Sparse4D model using the configured backend.

This script loads a trained Sparse4D checkpoint, prepares the calibration data loader
from the dataset specified in ``quant_calibration_dataset``, runs quantization via
``ModelQuantizer``, and saves the quantized model.
"""

import os

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs, logging

from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.quantization import ModelQuantizer
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d_pl_model import Sparse4DPlModel
from nvidia_tao_pytorch.cv.sparse4d.dataloader.pl_sparse4d_data_module import Sparse4DDataModule
from nvidia_tao_pytorch.cv.sparse4d.utils.misc import load_pretrained_weights


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additionally using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="quantize",
    schema=ExperimentConfig,
)
@monitor_status(name="Sparse4D", mode="quantize")
def main(cfg: ExperimentConfig) -> None:
    """Run the quantization process.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration including the ``quantize`` section.
    """
    # Obfuscate logs.
    obfuscate_logs(cfg)

    logging.info("Starting Sparse4D quantization")

    # Build the Lightning model and extract the underlying nn.Module
    logging.debug("Loading Sparse4D checkpoint")
    if not cfg.quantize.model_path.endswith(".onnx"):
        pl_model = Sparse4DPlModel(cfg)
        state_dict = load_pretrained_weights(cfg.quantize.model_path)
        pl_model.load_state_dict(state_dict, strict=False)
        orig_model = pl_model.model
    else:
        orig_model = None  # ModelOpt ONNX backend loads the model from the file.

    # Prepare calibration dataloader via DataModule
    calib_cfg = cfg.dataset.quant_calibration_dataset
    calib_images_dir = getattr(calib_cfg, "images_dir", "")
    if cfg.quantize.mode != "weight_only_ptq" and calib_images_dir:
        dm = Sparse4DDataModule(cfg)
        dm.setup(stage="calibration")
        calibration_loader = dm.calib_dataloader()
    else:
        calibration_loader = None

    # Create quantizer and quantize the model
    quantizer = ModelQuantizer(cfg.quantize)
    quantized_model = quantizer.quantize_model(orig_model, calibration_loader)
    logging.info("Quantization finished; saving model")
    quantizer.save_model(quantized_model, cfg.quantize.results_dir)
    logging.info("Sparse4D quantization completed successfully")


if __name__ == "__main__":
    main()
