# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantize an RT-DETR model using the configured backend.

This script loads a trained RT-DETR checkpoint, prepares the calibration data loader
from the dataset specified in ``quant_calibration_data_sources``, runs quantization via
``ModelQuantizer``, and saves the quantized model.
"""

import os
import logging

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs

from nvidia_tao_core.config.rtdetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.quantization import ModelQuantizer
from nvidia_tao_pytorch.cv.rtdetr.model.pl_rtdetr_model import RTDETRPlModel
from nvidia_tao_pytorch.cv.rtdetr.dataloader.pl_od_data_module import ODDataModule


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additionally using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="quantize",
    schema=ExperimentConfig,
)
@monitor_status(name="RT-DETR", mode="quantize")
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
    logger.info("Starting RT-DETR quantization")

    # Build the Lightning model and extract the underlying nn.Module
    logger.debug("Loading RT-DETR checkpoint")
    if not cfg.quantize.model_path.endswith(".onnx"):
        pl_model = RTDETRPlModel.load_from_checkpoint(
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
    logger.info("RT-DETR quantization completed successfully")


if __name__ == "__main__":
    main()
