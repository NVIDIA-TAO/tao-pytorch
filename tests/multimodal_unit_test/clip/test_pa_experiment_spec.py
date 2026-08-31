# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


def test_fixed_pa_reproduction_profile_preserves_recorded_protocol():
    """The checked-in profile must remain PA-only and protocol-compatible."""
    spec_path = (
        Path(__file__).parents[3]
        / "nvidia_tao_pytorch/multimodal/clip/experiment_specs/pas_v3_1"
        / "experiment_siglip2_pas_v3_1_dual_tower_lora_pa.yaml"
    )
    with spec_path.open(encoding="utf-8") as spec_file:
        spec = yaml.safe_load(spec_file)

    assert spec["model"]["type"] == "siglip2-so400m-patch16-256"
    assert spec["peft"]["enabled"] is True
    assert spec["peft"]["method"] == "lora"
    assert spec["train"]["num_epochs"] == 20
    assert spec["train"]["num_gpus"] == 8
    assert spec["train"]["siglip_loss_dist_impl"] == "gather"
    assert (
        spec["train"]["siglip_loss_mask_mode"]
        == "attribute_plus_accessory_match_positive"
    )
    assert spec["train"]["triplet_loss_weight"] == 0.0
    assert spec["train"]["pa_loss_weight"] > 0.0
    assert spec["train"]["checkpointer"]["monitor"] == "val/pas/overall_mAP"
    assert spec["dataset"]["val"]["metadata_match_mode"] == (
        "scalar_plus_accessories"
    )
    assert spec["evaluate"]["pas_ground_truth_mode"] == (
        "scalar_plus_accessories"
    )
