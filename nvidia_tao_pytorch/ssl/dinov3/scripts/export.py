# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export DINOv3 backbone to ONNX.

Thin wrapper that builds :class:`DinoV3PlModel`, restores weights, and exports the teacher
backbone to ONNX. Mirrors the nvdinov2 export flow. (RoPE / patch-16 ONNX validation is a
later, deploy-path phase.)
"""

import os
import torch

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel
from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import (
    merge_lora_state_dict,
    TAO_ONLY_KEYS,
    extract_backbone_state_dict,
    is_full_checkpoint,
)

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _restore_export_checkpoint(model, model_path):
    """Restore a DINOv3 export checkpoint while preserving the stripped-checkpoint path.

    A full Lightning checkpoint contains both student and teacher branches, so export selects
    and loads only ``teacher.backbone``. For all other supported checkpoint shapes, call the
    existing restore flow unchanged.

    Args:
        model (DinoV3PlModel): Constructed DINOv3 Lightning model.
        model_path (str): Export checkpoint path.
    """
    state_dict = model._load_pretrained_state_dict(model_path)
    # Export builds a plain backbone with no adapters, so any lora_* keys would simply be
    # dropped as unexpected and the ONNX would be the frozen base model rather than the
    # adapted one. Fold them in first; no-op for full-fine-tune checkpoints. This also keeps
    # the traced graph free of LoRA ops, i.e. stock DINOv3 topology.
    state_dict = merge_lora_state_dict(state_dict)
    if not is_full_checkpoint(state_dict):
        model.restore_pretrained_weights(preloaded_state_dict=state_dict)
        return

    teacher_state_dict = extract_backbone_state_dict(state_dict, source="teacher")
    teacher_backbone = model.teacher.backbone
    reference_state_dict = teacher_backbone.state_dict()
    remapped, unmapped = model._validate_and_remap_pretrained_state_dict(
        teacher_state_dict,
        reference_state_dict,
        model_path,
    )
    missing_keys, unexpected_keys = teacher_backbone.load_state_dict(remapped, strict=False)

    logging.info(
        "DINOv3 export: selected 'teacher.backbone' from full Lightning checkpoint "
        f"'{model_path}'."
    )
    logging.info(
        f"DINOv3 export remap: loaded {len(remapped)}/{len(remapped) + len(unmapped)} "
        "checkpoint tensors into the teacher ViT backbone."
    )
    residual_missing = [key for key in missing_keys if key not in TAO_ONLY_KEYS]
    if residual_missing:
        logging.info(f"DINOv3 export remap missing keys (kept as initialized): {residual_missing}")
    if unexpected_keys:
        logging.info(f"DINOv3 export remap unexpected keys: {unexpected_keys}")
    if unmapped:
        logging.info(f"DINOv3 export checkpoint keys with no matching backbone param: {unmapped}")


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="DINOv3", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """CLI wrapper to run export.

    Args:
        cfg (ExperimentConfig): Hydra-composed DINOv3 experiment config.

    Returns:
        No explicit returns.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_export(cfg)


def run_export(experiment_config):
    """Wrapper to run export of tlt models.

    Args:
        experiment_config (ExperimentConfig): Parsed config to run export.

    Returns:
        No explicit returns.
    """
    gpu_id = experiment_config.export.gpu_id
    torch.cuda.set_device(gpu_id)

    # Parsing command line arguments.
    model_path = experiment_config.export.checkpoint
    key = experiment_config.encryption_key

    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

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

    # Set default output filename if the filename
    # isn't provided over the command line.
    if output_file is None:
        split_name = os.path.splitext(model_path)[0]
        output_file = "{}.onnx".format(split_name)

    # Warn the user if an exported file already exists.
    assert not os.path.exists(output_file), "Default onnx file {} already "\
        "exists".format(output_file)

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    experiment_config.train.use_custom_attention = False
    model = DinoV3PlModel(experiment_config)

    model.pretrained_weights = model_path
    _restore_export_checkpoint(model, model_path)
    model = model.teacher.backbone

    input_names = ['input']
    output_names = ["output"]

    model.eval()
    if not on_cpu:
        model.cuda()
    model.float()

    # create dummy input
    if on_cpu:
        dummy_input = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cpu').float()
    else:
        dummy_input = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda').float()

    if batch_size is None or batch_size == -1:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    else:
        dynamic_axes = None

    torch.onnx.export(model,
                      dummy_input,
                      output_file,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=opset_version,
                      do_constant_folding=False,
                      verbose=True,
                      dynamo=False)

    print(f"ONNX file stored at {output_file}")


if __name__ == "__main__":
    main()
