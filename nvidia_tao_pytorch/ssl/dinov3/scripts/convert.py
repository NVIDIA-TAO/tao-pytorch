# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a DINOv3 SSL checkpoint to the cv/backbone_v2 (timm) layout.

CPU-only subtask. Takes a checkpoint produced by ``dinov3 train`` and writes a timm-format
``vit_*_dinov3`` backbone state dict, so the domain-adapted backbone can be consumed by the
``cv/backbone_v2`` ``dinov3_vitb16`` registry entry (and any downstream supervised task) via
``pretrained_backbone_path``. See ``ssl/dinov3/README.md`` (Downstream / backbone_v2 interop).
"""

import os

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging, obfuscate_logs
from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import (
    backbone_registry_name_for_arch,
    convert_ssl_to_timm,
    timm_model_name_for_arch,
)

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_convert(experiment_config):
    """Convert the SSL checkpoint to a timm-format backbone file.

    Args:
        experiment_config (ExperimentConfig): Parsed DINOv3 experiment config.
    """
    convert_config = experiment_config.convert

    checkpoint = convert_config.checkpoint
    assert checkpoint, "convert.checkpoint must be set to the SSL DINOv3 checkpoint to convert."
    assert os.path.exists(checkpoint), f"convert.checkpoint does not exist: {checkpoint}"

    source = convert_config.source
    # The exported architecture follows the source sub-model's backbone type.
    backbone = experiment_config.model.backbone
    arch = backbone.student_type if source.startswith("student") else backbone.teacher_type
    timm_model_name = timm_model_name_for_arch(arch)
    registry_name = backbone_registry_name_for_arch(arch)

    output_path = convert_config.output_path
    if not output_path:
        results_dir = convert_config.results_dir or experiment_config.results_dir or "."
        output_path = os.path.join(results_dir, f"dinov3_{arch}_backbone.safetensors")

    assert not os.path.exists(output_path), (
        f"Output file already exists: {output_path}. Remove it or set a different convert.output_path."
    )

    logging.info(
        f"Converting DINOv3 SSL checkpoint '{checkpoint}' (source={source}, arch={arch}) "
        f"-> timm-format backbone '{output_path}'."
    )
    convert_ssl_to_timm(
        checkpoint,
        output_path,
        source=source,
        validate=convert_config.validate,
        timm_model_name=timm_model_name,
    )
    logging.info(
        f"Backbone written to {output_path}. Load it downstream via "
        f"BACKBONE_REGISTRY.get('{registry_name}')(pretrained_backbone_path='{output_path}')."
    )


# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="DINOv3", mode="convert")
def main(cfg: ExperimentConfig) -> None:
    """Run the backbone conversion.

    Args:
        cfg (ExperimentConfig): Hydra-composed DINOv3 experiment config.
    """
    obfuscate_logs(cfg)
    run_convert(cfg)


if __name__ == "__main__":
    main()
