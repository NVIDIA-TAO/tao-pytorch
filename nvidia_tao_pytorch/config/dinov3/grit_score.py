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

    results_dir: Optional[str] = STR_FIELD(None, description="Directory for scores and status logs.")
    input_parquet: str = STR_FIELD("", description="Target image or consensus-channel manifest.")
    checkpoint: str = STR_FIELD("", description="DINOv3 checkpoint used to score the targets.")
    base_spec: str = STR_FIELD("", description="TAO DINOv3 experiment spec defining the backbone.")
    precomputed_consensus: bool = BOOL_FIELD(False, description="Score existing consensus channels without inference.")
    domain_column: str = STR_FIELD("task", description="Column defining independent ranking domains.")
    global_column: str = STR_FIELD("global_consensus", description="Precomputed global consensus column.")
    dense_column: str = STR_FIELD("dense_consensus", description="Precomputed dense consensus column.")
    device: str = STR_FIELD("cuda", description="Backbone inference device.")
    neighbor_device: Optional[str] = STR_FIELD(None, description="Neighbor device; defaults to the inference device.")
    neighbor_backend: str = STR_FIELD("auto", valid_options="auto,torch_exact,faiss_exact", description="Neighbor-search implementation.")
    neighbor_block_rows: int = INT_FIELD(2048, valid_min=1, description="Rows per neighbor-search block.")
    batch_size: int = INT_FIELD(12, valid_min=1, description="Images per scoring batch.")
    workers: int = INT_FIELD(8, valid_min=0, description="Image-loader workers.")
    amp: bool = BOOL_FIELD(True, description="Enable mixed-precision inference.")
    archive_cache_size: int = INT_FIELD(8, valid_min=1, description="Open archives retained per worker.")
    settling_k: int = INT_FIELD(50, valid_min=1, description="Neighbors used for settling consensus.")
    view_ks: List[int] = LIST_FIELD([8, 16, 32], description="Neighborhood sizes for view consensus.")
    work_dir: Optional[str] = STR_FIELD(None, description="Scratch directory for intermediate representations.")
    scratch_headroom_fraction: float = FLOAT_FIELD(0.10, valid_min=0, description="Additional scratch-space allowance.")
