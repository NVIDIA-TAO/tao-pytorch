# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantize a Visual ChangeNet model using the configured backend."""

import os

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs, logging

from nvidia_tao_pytorch.config.visual_changenet.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.quantization import ModelQuantizer
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlSegment
from nvidia_tao_pytorch.cv.visual_changenet.segmentation.dataloader.pl_changenet_data_module import CNDataModule
from nvidia_tao_pytorch.cv.visual_changenet.classification.models.cn_pl_model import ChangeNetPlModel as ChangeNetPlClassify
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.pl_oi_data_module import OIDataModule


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="quantize",
    schema=ExperimentConfig,
)
@monitor_status(name="Visual ChangeNet", mode="quantize")
def main(cfg: ExperimentConfig) -> None:
    """Run the quantization process."""
    obfuscate_logs(cfg)

    task = cfg.task
    logging.info(f"Starting Visual ChangeNet {task} quantization")

    logging.debug("Loading Visual ChangeNet checkpoint")
    calibration_loader = None

    if not cfg.quantize.model_path.endswith(".onnx"):
        if task == "segment":
            pl_model = ChangeNetPlSegment.load_from_checkpoint(
                cfg.quantize.model_path,
                map_location="cpu",
                experiment_spec=cfg,
            )
            orig_model = pl_model.model

            if cfg.quantize.mode != "weight_only_ptq" and cfg.dataset.segment.quant_calibration_dataset.images_dir:
                dm = CNDataModule(cfg.dataset.segment)
                dm.setup(stage="calibration")
                calibration_loader = dm.calib_dataloader()

        elif task == "classify":
            dm = OIDataModule(cfg, changenet=True)
            pl_model = ChangeNetPlClassify.load_from_checkpoint(
                cfg.quantize.model_path,
                map_location="cpu",
                experiment_spec=cfg,
                dm=dm,
            )
            orig_model = pl_model.model

            if cfg.quantize.mode != "weight_only_ptq" and cfg.dataset.classify.quant_calibration_dataset.images_dir:
                dm.setup(stage="calibration")
                calibration_loader = dm.calib_dataloader()

        else:
            raise ValueError(f"Unsupported task: {task}. Must be 'segment' or 'classify'.")
    else:
        orig_model = None

    quantizer = ModelQuantizer(cfg.quantize)
    quantized_model = quantizer.quantize_model(orig_model, calibration_loader)
    logging.info("Quantization finished; saving model")
    quantizer.save_model(quantized_model, cfg.quantize.results_dir)
    logging.info(f"Visual ChangeNet {task} quantization completed successfully")


if __name__ == "__main__":
    main()
