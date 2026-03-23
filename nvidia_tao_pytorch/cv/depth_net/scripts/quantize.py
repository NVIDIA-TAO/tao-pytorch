# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Quantize a DepthNet model using the configured backend.

This script loads a trained DepthNet checkpoint, prepares the calibration data loader
from the dataset specified in ``quant_calibration_dataset``, runs quantization via
``ModelQuantizer``, and saves the quantized model.
"""

import os

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs, logging

from nvidia_tao_core.config.depth_net.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.quantization import ModelQuantizer
from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import build_pl_model
from nvidia_tao_pytorch.cv.depth_net.dataloader.pl_mono_data_module import MonoDepthNetDataModule


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additionally using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="quantize",
    schema=ExperimentConfig,
)
@monitor_status(name="DepthNet", mode="quantize")
def main(cfg: ExperimentConfig) -> None:
    """Run the quantization process.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration including the ``quantize`` section.
    """
    # Obfuscate logs.
    obfuscate_logs(cfg)

    logging.info("Starting DepthNet quantization")

    # Build the Lightning model and extract the underlying nn.Module
    logging.debug("Loading DepthNet checkpoint")
    if not cfg.quantize.model_path.endswith(".onnx"):
        pl_model = build_pl_model(cfg)
        pl_model.load_state_dict_from_checkpoint(cfg.quantize.model_path)
        orig_model = pl_model.model
    else:
        orig_model = None  # ModelOpt ONNX backend loads the model from the file.

    # Prepare calibration dataloader via DataModule
    calib_cfg = cfg.dataset.quant_calibration_dataset
    calib_images_dir = getattr(calib_cfg, "images_dir", "")
    if cfg.quantize.mode != "weight_only_ptq" and calib_images_dir:
        dm = MonoDepthNetDataModule(cfg.dataset)
        dm.setup(stage="calibration")
        calibration_loader = dm.calib_dataloader()
    else:
        calibration_loader = None

    # Create quantizer and quantize the model
    quantizer = ModelQuantizer(cfg.quantize)
    quantized_model = quantizer.quantize_model(orig_model, calibration_loader)
    logging.info("Quantization finished; saving model")
    quantizer.save_model(quantized_model, cfg.quantize.results_dir)
    logging.info("DepthNet quantization completed successfully")


if __name__ == "__main__":
    main()
