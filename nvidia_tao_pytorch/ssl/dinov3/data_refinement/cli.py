# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute the frozen DINOv3 GRIT score from consensus-channel Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from .grit import GRIT_FORMULA_VERSION, score_grit_frame
from .grit_pipeline import (
    INPUT_SIZE,
    NEIGHBOR_BLOCK_ROWS,
    PATCH_GRID,
    PATCH_RETRIEVAL_CVAR_FRACTION,
    PATCH_RETRIEVAL_GRID,
    PATCH_RETRIEVAL_TEMPERATURE,
    extract_grit_scores,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("model.safetensors", "pytorch_model.bin", "model.pth"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"Cannot fingerprint DINOv3 checkpoint: {path}")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_digest(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def run(config: dict) -> dict:
    """Execute scoring from a plain configuration dictionary."""
    input_path = Path(config["input_parquet"]).resolve()
    output_dir = Path(config["output_dir"]).resolve()
    if (output_dir / "_SUCCESS").exists():
        raise RuntimeError(f"Refusing to overwrite committed output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.get("precomputed_consensus", False):
        # Avoid asynchronous prefetch teardown races in the approved image.
        frame = pd.read_parquet(input_path, pre_buffer=False)
        result = score_grit_frame(
            frame,
            domain_column=config.get("domain_column", "task"),
            global_column=config.get("global_column", "global_consensus"),
            dense_column=config.get("dense_column", "dense_consensus"),
        )
        observation_mode = "precomputed_consensus"
    else:
        for required in ("checkpoint", "base_spec"):
            if not config.get(required):
                raise ValueError(f"GRIT image scoring requires {required}")
        result = extract_grit_scores(config)
        observation_mode = "checkpoint_conditioned_images"
    output_path = output_dir / "grit_scores.parquet"
    temporary = output_dir / "grit_scores.tmp.parquet"
    result.to_parquet(temporary, index=False)
    temporary.replace(output_path)
    request_sha256 = config.get("request_sha256") or os.environ.get(
        "TAO_REFINEMENT_REQUEST_SHA256"
    )
    metadata = {
        "schema_version": "1.0",
        "action": "dinov3_grit_score",
        "formula_version": GRIT_FORMULA_VERSION,
        "semantics": "within_domain_ordinal_instability_rank",
        "observation_mode": observation_mode,
        "input_uri": input_path.as_uri(),
        "input_sha256": _sha256(input_path),
        "output_uri": output_path.as_uri(),
        "output_sha256": _sha256(output_path),
        "row_count": len(result),
        "domain_counts": result.groupby(config.get("domain_column", "task"))
        .size()
        .to_dict(),
        "settings_digest": _canonical_digest(config),
        "implementation_sha256": _canonical_digest(
            {
                path.name: _sha256(path)
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("grit.py"),
                    Path(__file__).with_name("grit_pipeline.py"),
                )
            }
        ),
        "entrypoint_sha256": _sha256(Path(__file__)),
        "request_sha256": request_sha256,
    }
    if observation_mode == "checkpoint_conditioned_images":
        checkpoint = Path(config["checkpoint"]).resolve()
        base_spec = Path(config["base_spec"]).resolve()
        metadata.update(
            {
                "checkpoint_uri": checkpoint.as_uri(),
                "checkpoint_sha256": _sha256(_checkpoint_file(checkpoint)),
                "base_spec_uri": base_spec.as_uri(),
                "base_spec_sha256": _sha256(base_spec),
                "input_size": INPUT_SIZE,
                "backbone_type": result.attrs["backbone_type"],
                "backbone_depth": result.attrs["backbone_depth"],
                "observation_contract_version": result.attrs[
                    "observation_contract_version"
                ],
                "layers_one_based": result.attrs["layer_numbers_one_based"],
                "layer_policy": "relative_depth_quartiles",
                "views": ["canonical", "top_left_90pct", "bottom_right_90pct"],
                "settling_k": int(config.get("settling_k", 50)),
                "view_ks": list(map(int, config.get("view_ks", (8, 16, 32)))),
                "dense_pool_grid": PATCH_GRID,
                "patch_retrieval_grid": PATCH_RETRIEVAL_GRID,
                "patch_retrieval_temperature": PATCH_RETRIEVAL_TEMPERATURE,
                "patch_retrieval_cvar_fraction": PATCH_RETRIEVAL_CVAR_FRACTION,
                "rank_tie_policy": "average_midrank",
                "device": str(config.get("device", "cuda")),
                "neighbor_device": str(
                    config.get("neighbor_device", config.get("device", "cuda"))
                ),
                "neighbor_block_rows": int(
                    config.get("neighbor_block_rows", NEIGHBOR_BLOCK_ROWS)
                ),
                "neighbor_backend": str(
                    config.get("neighbor_backend", "auto")
                ),
                "realized_neighbor_backends": result.attrs[
                    "realized_neighbor_backends"
                ],
                "realized_neighbor_device": result.attrs[
                    "realized_neighbor_device"
                ],
                "work_dir": str(config.get("work_dir", "")),
                "scratch_path": result.attrs["scratch_path"],
                "required_scratch_bytes": result.attrs[
                    "required_scratch_bytes"
                ],
                "available_scratch_bytes_at_start": result.attrs[
                    "available_scratch_bytes_at_start"
                ],
                "patch_size": result.attrs["patch_size"],
                "archive_cache_size": int(config.get("archive_cache_size", 8)),
                "amp": bool(config.get("amp", True)),
            }
        )
    commit_path = output_dir / "score_commit.json"
    _atomic_json(output_dir / "grit_score_metadata.json", metadata)
    _atomic_json(commit_path, metadata)
    success = output_dir / "_SUCCESS"
    success.write_text(_sha256(commit_path) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    print(json.dumps(run(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
