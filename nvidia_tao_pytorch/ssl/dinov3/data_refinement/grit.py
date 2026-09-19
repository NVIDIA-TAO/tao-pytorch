# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen GRIT score used by the DINOv3 SSL refinement workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd


GRIT_FORMULA_VERSION = "grit_v1"


def _midrank_ecdf(values: pd.Series) -> pd.Series:
    return (values.rank(method="average") - 0.5) / len(values)


def score_grit_frame(
    frame: pd.DataFrame,
    *,
    domain_column: str = "task",
    global_column: str = "global_consensus",
    dense_column: str = "dense_consensus",
) -> pd.DataFrame:
    """Return the within-domain hard-OR rank of two DINO consensus channels.

    The result is an ordinal acquisition score. It is not a calibrated error
    probability, epistemic uncertainty estimate, or universal quality metric.
    """
    required = {"sample_id", domain_column, global_column, dense_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing GRIT input columns: {sorted(missing)}")
    if frame[list(required)].isnull().any().any():
        raise ValueError("GRIT inputs contain null values")
    for column in (global_column, dense_column):
        values = frame[column]
        if (
            not pd.api.types.is_numeric_dtype(values) or
            pd.api.types.is_complex_dtype(values) or
            not np.isfinite(values.to_numpy(dtype=np.float64)).all()
        ):
            raise ValueError(f"GRIT channel {column} must contain finite real numbers")
    if frame["sample_id"].duplicated().any():
        raise ValueError(
            "GRIT inputs contain duplicate sample_id values; sample_id must be "
            "globally unique"
        )

    output = frame.copy()
    output["global_consensus_rank"] = output.groupby(
        domain_column, sort=True, group_keys=False
    )[global_column].transform(_midrank_ecdf)
    output["dense_consensus_rank"] = output.groupby(
        domain_column, sort=True, group_keys=False
    )[dense_column].transform(_midrank_ecdf)
    output["grit_score"] = output[
        ["global_consensus_rank", "dense_consensus_rank"]
    ].max(axis=1)
    output["grit_formula_version"] = GRIT_FORMULA_VERSION
    return output
