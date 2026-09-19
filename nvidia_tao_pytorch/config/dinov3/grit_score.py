# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the DINOv3 GRIT scoring subtask."""

from dataclasses import dataclass
from typing import List, Optional

from nvidia_tao_pytorch.config.utils.types import (
    BOOL_FIELD, FLOAT_FIELD, INT_FIELD, LIST_FIELD, STR_FIELD,
)


@dataclass
class GRITScoreConfig:
    """Model-owned scoring options shared with the external DEFT adapter."""

    results_dir: Optional[str] = STR_FIELD(
        None,
        display_name="results dir",
        default_value=None,
        description="Directory for scores and status logs.",
    )
    input_parquet: str = STR_FIELD(
        "",
        display_name="input parquet",
        description="Target image or consensus-channel manifest.",
        required="yes",
    )
    checkpoint: str = STR_FIELD(
        "",
        display_name="checkpoint",
        description="DINOv3 checkpoint; required unless precomputed_consensus=true.",
    )
    base_spec: str = STR_FIELD(
        "",
        display_name="base spec",
        description="TAO DINOv3 backbone spec; required unless precomputed_consensus=true.",
    )
    precomputed_consensus: bool = BOOL_FIELD(
        False,
        display_name="precomputed consensus",
        description="Score existing consensus channels without inference.",
    )
    domain_column: str = STR_FIELD(
        "task",
        display_name="domain column",
        description="Column defining independent ranking domains.",
    )
    global_column: str = STR_FIELD(
        "global_consensus",
        display_name="global column",
        description="Precomputed global consensus column.",
    )
    dense_column: str = STR_FIELD(
        "dense_consensus",
        display_name="dense column",
        description="Precomputed dense consensus column.",
    )
    device: str = STR_FIELD(
        "cuda",
        display_name="device",
        description="Backbone inference device: cpu, cuda, or cuda:N (a visible device index).",
    )
    neighbor_device: Optional[str] = STR_FIELD(
        None,
        display_name="neighbor device",
        default_value=None,
        description="Neighbor device: cpu, cuda, or cuda:N; defaults to the inference device.",
    )
    neighbor_backend: str = STR_FIELD(
        "auto",
        display_name="neighbor backend",
        valid_options="auto,torch_exact,faiss_exact",
        description="auto selects exact Torch for small cohorts and exact FAISS above the cohort-size limit; explicit backends do not fall back.",
    )
    neighbor_block_rows: int = INT_FIELD(
        2048,
        display_name="neighbor block rows",
        valid_min=1,
        description="Rows per neighbor-search block.",
    )
    batch_size: int = INT_FIELD(
        12,
        display_name="batch size",
        valid_min=1,
        description="Images per scoring batch.",
    )
    workers: int = INT_FIELD(
        8,
        display_name="workers",
        valid_min=0,
        description="Image-loader workers.",
    )
    input_size: int = INT_FIELD(
        512,
        display_name="input size",
        valid_options="512",
        description=(
            "Square resize used for GRIT feature extraction. Images are resized without "
            "preserving aspect ratio to match the DEFT scoring contract."
        ),
    )
    amp: bool = BOOL_FIELD(
        True,
        display_name="amp",
        description="Enable mixed-precision inference.",
    )
    archive_cache_size: int = INT_FIELD(
        8,
        display_name="archive cache size",
        valid_min=1,
        description="Open archives retained per worker.",
    )
    settling_k: int = INT_FIELD(
        50,
        display_name="settling k",
        valid_min=1,
        description="Neighbors used for settling consensus.",
    )
    view_ks: List[int] = LIST_FIELD(
        [8, 16, 32],
        display_name="view ks",
        description="Neighborhood sizes for view consensus.",
    )
    work_dir: Optional[str] = STR_FIELD(
        None,
        display_name="work dir",
        default_value=None,
        description="Scratch directory for intermediate representations.",
    )
    scratch_headroom_fraction: float = FLOAT_FIELD(
        0.10,
        display_name="scratch headroom fraction",
        valid_min=0,
        description="Additional scratch-space allowance.",
    )
