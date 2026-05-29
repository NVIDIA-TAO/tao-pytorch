# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantize a Grounding DINO model using the configured backend.

This script loads a trained Grounding DINO checkpoint, prepares the calibration data loader
from the dataset specified in ``quant_calibration_data_sources``, runs quantization via
``ModelQuantizer``, and saves the quantized model.
"""

import os
import logging
import tempfile

import torch
import pytorch_lightning as pl

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs

from nvidia_tao_pytorch.config.grounding_dino.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.quantization import ModelQuantizer
from nvidia_tao_pytorch.cv.grounding_dino.model.pl_gdino_model import GDINOPlModel
from nvidia_tao_pytorch.cv.grounding_dino.model.utils import grounding_dino_parser
from nvidia_tao_pytorch.cv.grounding_dino.dataloader.pl_odvg_data_module import ODVGDataModule


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additionally using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="quantize",
    schema=ExperimentConfig,
)
@monitor_status(name="Grounding-DINO", mode="quantize")
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
    logger.info("Starting Grounding DINO quantization")

    calibration_loader = None
    cap_lists = None
    if cfg.quantize.mode != "weight_only_ptq" and cfg.dataset.quant_calibration_data_sources is not None:
        dm = ODVGDataModule(cfg.dataset)
        dm.setup(stage="calibration")
        calibration_loader = dm.calib_dataloader()
        cap_lists = dm.calib_dataset.cap_lists

    if cap_lists is None:
        dm = ODVGDataModule(cfg.dataset)
        dm.setup(stage="test")
        cap_lists = dm.test_dataset.cap_lists

    # Build the Lightning model and extract the underlying nn.Module
    logger.debug("Loading Grounding DINO checkpoint")
    if not cfg.quantize.model_path.endswith(".onnx"):
        model_path = cfg.quantize.model_path
        original = torch.load(model_path, map_location="cpu")
        if "pytorch-lightning_version" not in original:
            final = grounding_dino_parser(original["model"])
            tmp = tempfile.NamedTemporaryFile()
            model_path = tmp.name
            torch.save({"state_dict": final, "pytorch-lightning_version": pl.__version__}, model_path)
        pl_model = GDINOPlModel.load_from_checkpoint(
            model_path,
            map_location="cpu",
            experiment_spec=cfg,
            cap_lists=cap_lists,
            strict=False,
        )
        orig_model = pl_model.model
    else:
        orig_model = None  # ModelOpt ONNX backend loads the model from the file.

    # Create quantizer and quantize the model
    quantizer = ModelQuantizer(cfg.quantize)
    quantized_model = quantizer.quantize_model(orig_model, calibration_loader)
    logger.info("Quantization finished; saving model")
    quantizer.save_model(quantized_model, cfg.quantize.results_dir)
    logger.info("Grounding DINO quantization completed successfully")


if __name__ == "__main__":
    main()
