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
Train OCRNet script.
"""
import os
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.initialize_experiments import initialize_train_experiment
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_core.config.ocrnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.ocrnet.dataloader.pl_ocr_data_module import OCRDataModule
from nvidia_tao_pytorch.cv.ocrnet.model.pl_ocrnet import OCRNetModel
from nvidia_tao_pytorch.cv.ocrnet.utils.utils import quantize_model


def run_experiment(experiment_spec: ExperimentConfig):
    """run experiment."""
    if "train_gt_file" in experiment_spec["dataset"]:
        if experiment_spec["dataset"]["train_gt_file"] == "":
            experiment_spec["dataset"]["train_gt_file"] = None
    if "val_gt_file" in experiment_spec["dataset"]:
        if experiment_spec["dataset"]["val_gt_file"] == "":
            experiment_spec["dataset"]["val_gt_file"] = None
    resume_ckpt, trainer_kwargs = initialize_train_experiment(experiment_spec)

    dm = OCRDataModule(experiment_spec)
    dm.setup(stage='fit')
    ocrnet_model = OCRNetModel(experiment_spec, dm)
    if experiment_spec.model.quantize:
        quantize_model(ocrnet_model, dm)
    print(ocrnet_model.model)
    clip_grad = experiment_spec.train.clip_grad_norm
    distributed_strategy = 'auto'

    if len(trainer_kwargs['devices']) > 1:
        distributed_strategy = experiment_spec.train.distributed_strategy

    use_distributed_sampler = experiment_spec.train.use_distributed_sampler

    trainer = Trainer(**trainer_kwargs,
                      strategy=distributed_strategy,
                      num_sanity_val_steps=0,
                      use_distributed_sampler=use_distributed_sampler,
                      gradient_clip_val=clip_grad)

    trainer.fit(ocrnet_model, dm, ckpt_path=resume_ckpt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="OCRNet", mode="train")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_spec=cfg)


if __name__ == '__main__':
    main()
