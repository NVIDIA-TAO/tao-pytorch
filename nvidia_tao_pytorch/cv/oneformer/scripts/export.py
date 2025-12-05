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
"""Export Oneformer model to ONNX."""

import os
import torch
import torch.nn as nn

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.oneformer.model.pl_oneformer import OneformerPlModule
from nvidia_tao_pytorch.cv.oneformer.export.onnx_exporter import ONNXExporter
from nvidia_tao_core.config.oneformer.default_config import ExperimentConfig

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec_ade", schema=ExperimentConfig
)
@monitor_status(name="OneFormer", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """CLI wrapper to run export.

    Args:
        cl_args(list): Arguments to parse.

    Returns:
        No explicit returns.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_export(cfg)


class OneFormerOnnxWrapper(nn.Module):
    """ONNX wrapper for OneFormer model.
    This wrapper prepares the OneFormer model for ONNX export by handling
    the forward pass with the required inputs.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, task_tokens):
        """
        Args:
            images: A batched tensor of *pre-normalized* images,
                    shape [B, 3, H, W].
            task_tokens: The pre-computed task tokens. [B, D]
        """
        task_embedding = self.model.task_mlp(task_tokens)
        features = self.model.backbone(images)
        outputs = self.model.sem_seg_head(features, task_embedding)
        return outputs['pred_logits'], outputs['pred_masks']


def run_export(experiment_config):
    """Wrapper to run export of tlt models.

    Args:
        args (dict): Dictionary of parsed arguments to run export.

    Returns:
        No explicit returns.
    """
    gpu_id = experiment_config.export.gpu_id
    torch.cuda.set_device(gpu_id)

    # Parsing command line arguments.
    experiment_config.model.export = True
    model_path = experiment_config.export.checkpoint
    output_file = experiment_config.export.onnx_file
    input_channel = experiment_config.export.input_channel
    input_width = experiment_config.export.input_width
    input_height = experiment_config.export.input_height
    opset_version = experiment_config.export.opset_version
    batch_size = experiment_config.export.batch_size
    on_cpu = experiment_config.export.on_cpu

    if batch_size is None or batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    # Set default output filename if the filename isn't provided over the command line.
    if not output_file:
        split_name = os.path.splitext(model_path)[0]
        output_file = f"{split_name}.onnx"

    # Warn the user if an exported file already exists.
    assert not os.path.exists(output_file), f"Default onnx file {output_file} already exists"

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    os.makedirs(output_root, exist_ok=True)

    device = 'cpu' if on_cpu else 'cuda'

    # load model
    pl_model = OneformerPlModule.load_from_checkpoint(
        model_path,
        map_location=device,
        cfg=experiment_config,
    )

    # model wrapper
    model = pl_model.model
    model.eval()
    model.to(device)

    wrapper = OneFormerOnnxWrapper(model)
    wrapper.eval()
    wrapper.to(device)

    dummy_image_input = torch.randn(input_batch_size, input_channel, input_height, input_width, device=device)
    dummy_task_input = torch.randn(input_batch_size, experiment_config.dataset.task_seq_len, device=device)

    dummy_input = (dummy_image_input, dummy_task_input)

    exporter = ONNXExporter()
    exporter.export_model(
        wrapper,
        batch_size,
        output_file,
        dummy_input,
        input_names=['images', 'task_tokens'],
        opset_version=opset_version,
        output_names=['pred_logits', 'pred_masks'],
        do_constant_folding=False,
        verbose=experiment_config.export.verbose)
    exporter.check_onnx(output_file)

    print(f"ONNX file stored at {output_file}")


if __name__ == "__main__":
    main()
