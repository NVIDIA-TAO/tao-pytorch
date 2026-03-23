# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""
Export of Segformer model.
"""
import os
from glob import glob

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_core.config.segformer.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.segformer_old.utils.onnx_export import ONNXExporter
from nvidia_tao_pytorch.cv.segformer_old.utils.config import MMSegmentationConfig

# Triggers build of custom modules
from nvidia_tao_pytorch.cv.segformer_old.model import * # noqa pylint: disable=W0401, W0614
from nvidia_tao_pytorch.cv.segformer_old.dataloader import * # noqa pylint: disable=W0401, W0614

from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.registry.utils import init_default_scope


def run_experiment(experiment_config):
    """Start the Export.
    Args:
        experiment_config (Dict): Config dictionary containing epxeriment parameters
        results_dir (str): Results dir to save the exported ONNX.

    """
    results_dir = experiment_config.results_dir
    status_logger = status_logging.get_status_logger()
    status_logger.write(message="**********************Start logging for Export**********************.")

    mmseg_config = MMSegmentationConfig(experiment_config, phase="export")
    model_cfg = Config(mmseg_config.updated_config)
    deploy_cfg = Config(mmseg_config.deploy_config)

    print_log(model_cfg)
    print_log(deploy_cfg)

    init_default_scope(model_cfg.default_scope)

    # experiment_config is a cfg, so use . notation
    # export_cfg is a dict

    model_path = experiment_config.export.checkpoint

    output_file = experiment_config.export.onnx_file
    if not output_file:
        onnx_path = model_path.replace(".pth", ".onnx")
    else:
        onnx_path = output_file

    pipeline_cfg = model_cfg.test_pipeline
    # Loading annotations is also not applicable (see inference.py)
    for i, transform in enumerate(pipeline_cfg):
        if transform['type'] == "TAOLoadAnnotations":
            del pipeline_cfg[i]

    # Instead of taking in img dims from config, uses a sample image from the test dataset
    test_dir = model_cfg["test_dataloader"]["dataset"]["data_prefix"]["img_path"]
    imgs = glob(test_dir + "/*.jpg", recursive=True)
    imgs += glob(test_dir + "/*.png", recursive=True)

    img = imgs[0]
    onnx_export = ONNXExporter()
    onnx_export.export_model(img=img, work_dir=results_dir, save_file=onnx_path.split('/')[-1],
                             deploy_cfg=deploy_cfg, model_cfg=model_cfg, model_checkpoint=model_path)

    status_logger.write(message="Completed Export.", status_level=status_logging.Status.RUNNING)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="export_isbi", schema=ExperimentConfig
)
@monitor_status(name="Segformer", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """Run the Export."""
    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
