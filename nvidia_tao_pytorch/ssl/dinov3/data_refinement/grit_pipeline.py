# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-conditioned observations for the frozen DINOv3 GRIT rank."""

from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
import math
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.parse import unquote, urlparse
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .grit import score_grit_frame


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 512
PATCH_GRID = 4
PATCH_RETRIEVAL_GRID = 8
PATCH_RETRIEVAL_TEMPERATURE = 0.07
PATCH_RETRIEVAL_CVAR_FRACTION = 0.2
NEIGHBOR_BLOCK_ROWS = 2048
EXACT_NEIGHBOR_ROW_LIMIT = 50_000


def relative_layer_numbers(depth: int) -> tuple[int, int, int, int]:
    """Map GRIT's four probes to backbone-relative depth quartiles."""
    if depth < 4:
        raise ValueError("GRIT requires a DINOv3 backbone with at least four blocks")
    return tuple(max(1, int(math.ceil(depth * fraction))) for fraction in (0.25, 0.5, 0.75, 1.0))


def _model_patch_size(model: torch.nn.Module) -> int:
    """Resolve and validate the DINO patch size used by retrieval geometry."""
    patch_size_value = model.patch_embed.patch_size
    patch_sizes = (
        tuple(map(int, patch_size_value))
        if isinstance(patch_size_value, (tuple, list))
        else (int(patch_size_value), int(patch_size_value))
    )
    if len(patch_sizes) != 2 or patch_sizes[0] != patch_sizes[1]:
        raise ValueError(f"GRIT requires square DINO patches: {patch_sizes}")
    return patch_sizes[0]


def _archive_local_order(frame: pd.DataFrame) -> list[int]:
    """Order extraction by storage path/member while preserving row indices."""
    members = frame["member"].tolist() if "member" in frame else [None] * len(frame)
    storage_types = frame["storage_type"].astype(str).tolist()
    paths = frame["path"].astype(str).tolist()
    return sorted(
        range(len(frame)),
        key=lambda index: (
            storage_types[index],
            paths[index],
            "" if members[index] is None else str(members[index]),
            index,
        ),
    )


def _rank(values: np.ndarray) -> np.ndarray:
    return ((pd.Series(values).rank(method="average") - 0.5) / len(values)).to_numpy()


def _normalize(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), dim=-1)


def _jaccard_deficit(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    values = np.empty(len(left), dtype=np.float64)
    width = left.shape[1]
    for index, (left_row, right_row) in enumerate(zip(left, right)):
        intersection = np.intersect1d(left_row, right_row, assume_unique=True).size
        values[index] = 1.0 - intersection / float(2 * width - intersection)
    return values


def _blockwise_neighbors(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    *,
    exclude_identity: bool,
    block_rows: int = NEIGHBOR_BLOCK_ROWS,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Return exact cosine neighbors with bounded pairwise memory."""
    if block_rows <= 0:
        raise ValueError("GRIT neighbor block_rows must be positive")
    compute_device = torch.device(device)
    query_values = query.astype(np.float32, copy=False)
    reference_values = reference.astype(np.float32, copy=False)
    output = np.empty((len(query), k), dtype=np.int64)
    allow_tf32 = None
    if compute_device.type == "cuda":
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for query_start in range(0, len(query), block_rows):
            query_stop = min(len(query), query_start + block_rows)
            query_block = torch.from_numpy(
                query_values[query_start:query_stop]
            ).to(compute_device)
            best_values = torch.full(
                (len(query_block), k), -torch.inf, device=compute_device
            )
            best_indices = torch.full(
                (len(query_block), k),
                -1,
                dtype=torch.int64,
                device=compute_device,
            )
            for reference_start in range(0, len(reference), block_rows):
                reference_stop = min(
                    len(reference), reference_start + block_rows
                )
                reference_block = torch.from_numpy(
                    reference_values[reference_start:reference_stop]
                ).to(compute_device)
                similarities = query_block @ reference_block.T
                if exclude_identity:
                    left = max(query_start, reference_start)
                    right = min(query_stop, reference_stop)
                    if left < right:
                        diagonal = torch.arange(
                            left, right, device=compute_device
                        )
                        similarities[
                            diagonal - query_start,
                            diagonal - reference_start,
                        ] = -torch.inf
                indices = torch.arange(
                    reference_start,
                    reference_stop,
                    device=compute_device,
                ).expand(len(query_block), -1)
                merged_values = torch.cat((best_values, similarities), dim=1)
                merged_indices = torch.cat((best_indices, indices), dim=1)
                best_values, positions = merged_values.topk(k, dim=1)
                best_indices = merged_indices.gather(1, positions)
            output[query_start:query_stop] = best_indices.cpu().numpy()
    finally:
        if allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    return output


def _faiss_neighbors(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    *,
    exclude_identity: bool,
    device: str | torch.device,
) -> np.ndarray:
    """Run exact FAISS cosine search while retaining the reference index once."""
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "FAISS is required for GRIT cohorts above "
            f"{EXACT_NEIGHBOR_ROW_LIMIT:,} rows"
        ) from error
    queries = np.ascontiguousarray(query, dtype=np.float32)
    references = np.ascontiguousarray(reference, dtype=np.float32)
    faiss.normalize_L2(queries)
    faiss.normalize_L2(references)
    index = faiss.IndexFlatIP(references.shape[1])
    if torch.device(device).type == "cuda":
        try:
            resources = faiss.StandardGpuResources()
            gpu_index = torch.device(device).index
            index = faiss.index_cpu_to_gpu(
                resources,
                torch.cuda.current_device() if gpu_index is None else gpu_index,
                index,
            )
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                "CUDA GRIT neighbor search requires GPU-enabled FAISS; refusing "
                "an unbounded exact CPU fallback"
            ) from error
    index.add(references)
    depth = k + 1 if exclude_identity else k
    _, neighbors = index.search(queries, depth)
    if not exclude_identity:
        return neighbors.astype(np.int64, copy=False)
    output = np.empty((len(queries), k), dtype=np.int64)
    for row, candidates in enumerate(neighbors):
        filtered = candidates[candidates != row]
        if len(filtered) < k:
            raise RuntimeError("FAISS self-exclusion underfilled the neighbor set")
        output[row] = filtered[:k]
    return output


def _search_neighbors(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    *,
    exclude_identity: bool,
    device: str | torch.device,
    block_rows: int,
    backend: str,
) -> np.ndarray:
    if backend not in {"auto", "torch_exact", "faiss_exact"}:
        raise ValueError(f"Unsupported GRIT neighbor backend: {backend}")
    large_cohort = max(len(query), len(reference)) > EXACT_NEIGHBOR_ROW_LIMIT
    use_faiss = any(
        (
            backend == "faiss_exact",
            all(
                (
                    backend == "auto",
                    large_cohort,
                )
            ),
        )
    )
    if use_faiss:
        return _faiss_neighbors(
            query,
            reference,
            k,
            exclude_identity=exclude_identity,
            device=device,
        )
    return _blockwise_neighbors(
        query,
        reference,
        k,
        exclude_identity=exclude_identity,
        device=device,
        block_rows=block_rows,
    )


def _realized_neighbor_backend(
    query_rows: int, reference_rows: int, backend: str
) -> str:
    if backend == "faiss_exact" or (
        backend == "auto" and
        max(query_rows, reference_rows) > EXACT_NEIGHBOR_ROW_LIMIT
    ):
        return "faiss_exact"
    return "torch_exact"


def _preflight_faiss(device: str | torch.device) -> None:
    """Fail before feature extraction when requested FAISS is unavailable."""
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError("Exact large-cohort GRIT requires FAISS") from error
    if torch.device(device).type != "cuda":
        return
    try:
        resources = faiss.StandardGpuResources()
        gpu_index = torch.device(device).index
        faiss.index_cpu_to_gpu(
            resources,
            torch.cuda.current_device() if gpu_index is None else gpu_index,
            faiss.IndexFlatIP(1),
        )
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(
            "CUDA GRIT neighbor search requires GPU-enabled FAISS"
        ) from error


def _manifest_requires_faiss(manifest: pd.DataFrame, backend: str) -> bool:
    if backend not in {"auto", "torch_exact", "faiss_exact"}:
        raise ValueError(f"Unsupported GRIT neighbor backend: {backend}")
    if backend == "faiss_exact":
        return True
    if backend == "torch_exact":
        return False
    for _, task_rows in manifest.groupby(manifest["task"].astype(str)):
        roles = task_rows["role"].astype(str)
        query_rows = int((roles == "query").sum())
        reference_rows = int((roles == "reference").sum())
        if max(query_rows, reference_rows) > EXACT_NEIGHBOR_ROW_LIMIT:
            return True
    return False


def _neighbors(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    *,
    device: str | torch.device,
    block_rows: int,
    backend: str = "auto",
) -> np.ndarray:
    if len(reference) <= k:
        raise ValueError(f"GRIT requires more than {k} references per domain")
    return _search_neighbors(
        query,
        reference,
        k,
        exclude_identity=False,
        device=device,
        block_rows=block_rows,
        backend=backend,
    )


def _within_cohort_neighbors(
    values: np.ndarray,
    k: int,
    *,
    device: str | torch.device,
    block_rows: int,
    backend: str = "auto",
) -> np.ndarray:
    if len(values) <= k:
        raise ValueError(f"GRIT requires more than {k} query samples per domain")
    return _search_neighbors(
        values,
        values,
        k,
        exclude_identity=True,
        device=device,
        block_rows=block_rows,
        backend=backend,
    )


def _patch_settling(block9: np.ndarray, block12: np.ndarray) -> np.ndarray:
    left = _normalize(torch.from_numpy(block9.astype(np.float32)))
    right = _normalize(torch.from_numpy(block12.astype(np.float32)))
    left_gram = left @ left.transpose(-1, -2)
    right_gram = right @ right.transpose(-1, -2)
    squared = (left_gram - right_gram).square()
    diagonal = torch.eye(squared.shape[-1], dtype=torch.bool)
    rows = squared.masked_fill(diagonal[None], 0.0).sum(dim=-1)
    rows = rows / float(squared.shape[-1] - 1)
    return rows.topk(4, dim=1).values.mean(dim=1).numpy()


def derive_grit_observations(
    manifest: pd.DataFrame,
    *,
    global_layers: np.ndarray,
    patch_layers: np.ndarray | None = None,
    dense_settling: np.ndarray | None = None,
    query_views: dict[str, np.ndarray],
    patch_retrieval: np.ndarray,
    settling_k: int = 50,
    view_ks: tuple[int, ...] = (8, 16, 32),
    neighbor_device: str | torch.device = "cpu",
    neighbor_block_rows: int = NEIGHBOR_BLOCK_ROWS,
    neighbor_backend: str = "auto",
) -> pd.DataFrame:
    """Derive the frozen four-probe GRIT rank from checkpoint features."""
    required = {"sample_id", "task", "role"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"GRIT manifest is missing columns: {sorted(missing)}")
    if global_layers.shape[:2] != (len(manifest), 4):
        raise ValueError("Global layer features do not align with the manifest")
    if patch_layers is None and dense_settling is None:
        raise ValueError("GRIT requires patch layers or precomputed dense settling")
    if patch_layers is not None and patch_layers.shape[:3] != (
        len(manifest), 4, 16
    ):
        raise ValueError("Patch layer features must use a 4x4 grid at four depths")

    query_rows = np.flatnonzero(manifest["role"].astype(str).to_numpy() == "query")
    queries = manifest.iloc[query_rows].reset_index(drop=True)
    if queries.empty:
        raise ValueError("GRIT manifest has no query rows")
    for name in ("canonical", "view0", "view1"):
        if query_views[name].shape[0] != len(queries):
            raise ValueError(f"{name} query features do not align with query rows")

    global_settling = np.empty(len(queries), dtype=np.float64)
    view_instability = np.empty(len(queries), dtype=np.float64)
    realized_backends = set()
    if dense_settling is None:
        dense_values = _patch_settling(
            patch_layers[query_rows, 2], patch_layers[query_rows, 3]
        )
    else:
        dense_values = np.asarray(dense_settling)
        if dense_values.shape == (len(manifest),):
            dense_values = dense_values[query_rows]
        if dense_values.shape != (len(queries),):
            raise ValueError("Dense settling values do not align with query rows")
    task_values = queries["task"].astype(str).to_numpy()
    manifest_tasks = manifest["task"].astype(str).to_numpy()
    for task in sorted(set(task_values)):
        output_rows = np.flatnonzero(task_values == task)
        task_query_rows = query_rows[output_rows]
        references = np.flatnonzero(
            (manifest_tasks == task) &
            (manifest["role"].astype(str).to_numpy() == "reference")
        )
        realized_backends.add(
            _realized_neighbor_backend(
                len(output_rows), len(references), neighbor_backend
            )
        )
        layer_neighbors = [
            _neighbors(
                global_layers[task_query_rows, layer],
                global_layers[references, layer],
                settling_k,
                device=neighbor_device,
                block_rows=neighbor_block_rows,
                backend=neighbor_backend,
            )
            for layer in range(4)
        ]
        global_settling[output_rows] = np.stack(
            [
                _jaccard_deficit(layer_neighbors[layer], layer_neighbors[-1])
                for layer in range(3)
            ],
            axis=1,
        ).mean(axis=1)

        spaces = [query_views[name][output_rows] for name in ("canonical", "view0", "view1")]
        realized_backends.add(
            _realized_neighbor_backend(
                len(output_rows), len(output_rows), neighbor_backend
            )
        )
        max_view_k = max(view_ks)
        max_neighbor_sets = [
            _within_cohort_neighbors(
                space,
                max_view_k,
                device=neighbor_device,
                block_rows=neighbor_block_rows,
                backend=neighbor_backend,
            )
            for space in spaces
        ]
        deficits = []
        for k in view_ks:
            neighbor_sets = [values[:, :k] for values in max_neighbor_sets]
            deficits.extend(
                _jaccard_deficit(neighbor_sets[left], neighbor_sets[right])
                for left, right in ((0, 1), (0, 2), (1, 2))
            )
        view_instability[output_rows] = np.stack(deficits, axis=1).mean(axis=1)

    observations = queries.copy()
    observations["global_depth_settling"] = global_settling
    observations["global_view_instability"] = view_instability
    observations["dense_depth_settling"] = dense_values
    observations["dense_view_retrieval"] = patch_retrieval
    observations["global_consensus"] = 0.0
    observations["dense_consensus"] = 0.0
    for task in sorted(set(task_values)):
        rows = np.flatnonzero(task_values == task)
        observations.loc[rows, "global_consensus"] = np.sqrt(
            _rank(global_settling[rows]) * _rank(view_instability[rows])
        )
        observations.loc[rows, "dense_consensus"] = np.sqrt(
            _rank(dense_values[rows]) * _rank(patch_retrieval[rows])
        )
    result = score_grit_frame(observations)
    result.attrs["realized_neighbor_backends"] = sorted(realized_backends)
    result.attrs["realized_neighbor_device"] = str(torch.device(neighbor_device))
    return result


class _ManifestImages(Dataset):
    def __init__(
        self, frame: pd.DataFrame, *, views: bool, archive_cache_size: int = 8
    ):
        self.storage_types = frame["storage_type"].astype(str).tolist()
        self.paths = frame["path"].astype(str).tolist()
        self.members = (
            frame["member"].tolist()
            if "member" in frame
            else [None] * len(frame)
        )
        self.views = views
        if archive_cache_size <= 0:
            raise ValueError("GRIT archive_cache_size must be positive")
        self.archive_cache_size = archive_cache_size
        self.archives: OrderedDict[tuple[str, str], object] = OrderedDict()
        self.archive_member_indices = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["archives"] = OrderedDict()
        state["archive_member_indices"] = {}
        return state

    def __del__(self):
        """Close process-owned archive descriptors."""
        for archive in getattr(self, "archives", {}).values():
            try:
                archive.close()
            except Exception:  # pragma: no cover - interpreter shutdown safety
                pass

    @staticmethod
    def _path(value: str) -> Path:
        parsed = urlparse(value)
        if not parsed.scheme:
            return Path(value)
        if any(
            (
                parsed.scheme != "file",
                parsed.netloc not in {"", "localhost"},
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        ):
            raise ValueError(f"GRIT requires a local file locator: {value}")
        return Path(unquote(parsed.path))

    def _archive(self, storage: str, path: Path):
        key = (storage, str(path))
        archive = self.archives.pop(key, None)
        if archive is None:
            if storage == "tar":
                if path.suffix.lower() != ".tar":
                    raise ValueError(
                        "Random GRIT archive access requires uncompressed .tar "
                        f"shards; stage or reshard {path}"
                    )
                archive = tarfile.open(path, "r:")
                self.archive_member_indices[key] = {
                    member.name: member for member in archive.getmembers()
                }
            else:
                archive = zipfile.ZipFile(path)
            while len(self.archives) >= self.archive_cache_size:
                stale_key, stale = self.archives.popitem(last=False)
                self.archive_member_indices.pop(stale_key, None)
                stale.close()
        self.archives[key] = archive
        return archive

    def _image(self, index: int) -> Image.Image:
        storage = self.storage_types[index]
        configured = self.paths[index]
        path = self._path(configured)
        if storage == "file":
            return Image.open(path).convert("RGB")
        member = str(self.members[index])
        if storage == "zip":
            archive = self._archive(storage, path)
            return Image.open(BytesIO(archive.read(member))).convert("RGB")
        if storage == "tar":
            archive = self._archive(storage, path)
            member_info = self.archive_member_indices[(storage, str(path))].get(
                member
            )
            extracted = (
                None if member_info is None else archive.extractfile(member_info)
            )
            if extracted is None:
                raise ValueError(f"Missing tar member {member} in {path}")
            return Image.open(extracted).convert("RGB")
        raise ValueError(f"Unsupported GRIT storage type: {storage}")

    @staticmethod
    def _tensor(image: Image.Image) -> torch.Tensor:
        value = TF.pil_to_tensor(image).float().div_(255.0)
        return TF.normalize(value, IMAGE_MEAN, IMAGE_STD)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        image = self._image(index).resize(
            (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC
        )
        output: dict[str, torch.Tensor | int] = {
            "canonical": self._tensor(image),
            "index": index,
        }
        if self.views:
            crop_size = int(round(INPUT_SIZE * 0.9))
            for corner, name in ((0, "view0"), (INPUT_SIZE - crop_size, "view1")):
                crop = image.crop(
                    (corner, corner, corner + crop_size, corner + crop_size)
                ).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)
                output[name] = self._tensor(crop)
        return output


def _build_backbone(base_spec: str | Path, checkpoint: str | Path) -> torch.nn.Module:
    from omegaconf import OmegaConf
    from timm.layers import Mlp

    from nvidia_tao_pytorch.config.dinov3.default_config import (
        ExperimentConfig,
        map_params,
    )
    from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel
    from nvidia_tao_pytorch.ssl.dinov3.model.vit import (
        DinoV3VisionTransformer,
        SwiGLUFusedFull,
    )

    spec = OmegaConf.merge(
        OmegaConf.structured(ExperimentConfig), OmegaConf.load(base_spec)
    )
    backbone_cfg = spec.model.backbone
    backbone_type = str(backbone_cfg.teacher_type)
    arch = {
        name: map_params[name][backbone_type]
        for name in (
            "embed_dim",
            "depth",
            "num_heads",
            "init_values",
            "drop_path_schedule",
            "num_classes",
            "mlp_ratio",
        )
    }
    backbone = DinoV3VisionTransformer(
        img_size=INPUT_SIZE,
        patch_size=int(backbone_cfg.patch_size),
        embed_dim=arch["embed_dim"],
        depth=arch["depth"],
        num_heads=arch["num_heads"],
        init_values=arch["init_values"],
        drop_path_schedule=arch["drop_path_schedule"],
        num_classes=arch["num_classes"],
        drop_path_rate=0.0,
        mlp_layer={"mlp": Mlp, "swiglu": SwiGLUFusedFull}[
            map_params["mlp_layer"][backbone_type]
        ],
        mlp_ratio=arch["mlp_ratio"],
        norm_layer=torch.nn.LayerNorm,
        act_layer=torch.nn.GELU,
        qkv_bias=False,
        register_tokens=int(backbone_cfg.num_register_tokens),
        use_custom_attention=bool(spec.train.use_custom_attention),
        rope_theta=float(backbone_cfg.rope_theta),
    )
    DinoV3PlModel.load_backbone_weights(backbone, str(checkpoint))
    backbone.grit_backbone_type = backbone_type
    return backbone


def _pool_patches(patches: torch.Tensor) -> torch.Tensor:
    side = int(round(math.sqrt(patches.shape[1])))
    feature_map = patches.transpose(1, 2).reshape(
        patches.shape[0], patches.shape[2], side, side
    )
    pooled = F.adaptive_avg_pool2d(feature_map.float(), PATCH_GRID)
    return _normalize(pooled.flatten(2).transpose(1, 2))


def _selected_layers(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    selected = model.forward_selected_layers(
        images, relative_layer_numbers(model.n_blocks)
    )
    registers = int(model.num_register_tokens)
    global_features = torch.stack([_normalize(value[:, 0]) for value in selected], dim=1)
    patch_features = torch.stack(
        [
            _pool_patches(value[:, 1:-registers] if registers else value[:, 1:])
            for value in selected
        ],
        dim=1,
    )
    return global_features, patch_features


def _mapped_indices(
    corner: int, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if patch_size <= 0 or INPUT_SIZE % patch_size:
        raise ValueError("GRIT patch size must divide the input size")
    grid = INPUT_SIZE // patch_size
    coords = torch.linspace(1, grid - 2, PATCH_RETRIEVAL_GRID).round().long()
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    query = (yy * grid + xx).reshape(-1)
    query_y = torch.div(query, grid, rounding_mode="floor")
    query_x = query.remainder(grid)
    offset = 0.0 if corner == 0 else 0.1
    canonical_x = torch.floor((offset + 0.9 * ((query_x + 0.5) / grid)) * grid)
    canonical_y = torch.floor((offset + 0.9 * ((query_y + 0.5) / grid)) * grid)
    anchor = canonical_y.long().clamp(0, grid - 1) * grid
    anchor += canonical_x.long().clamp(0, grid - 1)
    return query, anchor


def _patch_retrieval(
    canonical: torch.Tensor,
    view: torch.Tensor,
    corner: int,
    patch_size: int,
) -> torch.Tensor:
    query_indices, anchor_indices = _mapped_indices(corner, patch_size)
    query_indices = query_indices.to(view.device)
    anchor_indices = anchor_indices.to(view.device)
    query = view[:, query_indices]
    anchor = canonical[:, anchor_indices]
    logits = torch.einsum("bqd,bkd->bqk", query, anchor) / (
        PATCH_RETRIEVAL_TEMPERATURE
    )
    positive = logits.diagonal(dim1=1, dim2=2)
    nll = torch.logsumexp(logits, dim=-1) - positive
    count = max(
        1, int(math.ceil(nll.shape[-1] * PATCH_RETRIEVAL_CVAR_FRACTION))
    )
    return nll.topk(count, dim=-1).values.mean(dim=-1)


@torch.inference_mode()
def extract_grit_scores(config: dict) -> pd.DataFrame:
    """Load one checkpoint and compute all GRIT probes from image manifests."""
    import pyarrow.parquet as pq

    available = set(
        pq.ParquetFile(config["input_parquet"]).schema_arrow.names
    )
    manifest = pd.read_parquet(
        config["input_parquet"],
        pre_buffer=False,
        columns=[
            name
            for name in (
                "sample_id",
                "path",
                "storage_type",
                "member",
                "task",
                "role",
            )
            if name in available
        ],
    )
    required = {"sample_id", "path", "storage_type", "task", "role", "embedding"}
    missing = required.difference(available)
    if missing:
        raise ValueError(f"GRIT manifest is missing columns: {sorted(missing)}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("GRIT sample_id values must be unique")
    storage_types = set(manifest["storage_type"].astype(str))
    unsupported = storage_types.difference({"file", "tar", "zip"})
    if unsupported:
        raise ValueError(f"Unsupported GRIT storage types: {sorted(unsupported)}")
    archive_rows = manifest["storage_type"].astype(str).isin({"tar", "zip"})
    if archive_rows.any() and (
        "member" not in manifest or manifest.loc[archive_rows, "member"].isnull().any()
    ):
        raise ValueError("Archive-backed GRIT rows require a non-null member")
    device = torch.device(config.get("device", "cuda"))
    batch_size = int(config.get("batch_size", 12))
    workers = int(config.get("workers", 8))
    amp = bool(config.get("amp", True))
    archive_cache_size = int(config.get("archive_cache_size", 8))
    if archive_cache_size <= 0:
        raise ValueError("GRIT archive_cache_size must be positive")

    queries = manifest[manifest["role"].astype(str) == "query"].reset_index(drop=True)
    if queries.empty:
        raise ValueError("GRIT manifest has no query rows")
    neighbor_device = config.get("neighbor_device", str(device))
    neighbor_backend = str(config.get("neighbor_backend", "auto"))
    if _manifest_requires_faiss(manifest, neighbor_backend):
        _preflight_faiss(neighbor_device)
    model = _build_backbone(config["base_spec"], config["checkpoint"])
    model.to(device).eval()
    configured_work_dir = config.get("work_dir") or os.environ.get(
        "TAO_LOCAL_SCRATCH"
    )
    if not configured_work_dir:
        raise ValueError(
            "GRIT requires work_dir or TAO_LOCAL_SCRATCH for disk-backed features"
        )
    expanded_work_dir = os.path.expandvars(str(configured_work_dir))
    if "$" in expanded_work_dir:
        raise ValueError(
            f"GRIT work_dir has an unresolved variable: {expanded_work_dir}"
        )
    temporary_parent = str(Path(expanded_work_dir).expanduser().resolve())
    Path(temporary_parent).mkdir(parents=True, exist_ok=True)
    embedding_dim = int(model.embed_dim)
    feature_bytes = 4 * (
        len(manifest) * 4 * embedding_dim +
        len(manifest) +
        len(queries) * 3 * embedding_dim +
        len(queries)
    )
    scratch_headroom = float(config.get("scratch_headroom_fraction", 0.10))
    if scratch_headroom < 0:
        raise ValueError("GRIT scratch_headroom_fraction must be nonnegative")
    required_scratch_bytes = int(math.ceil(feature_bytes * (1 + scratch_headroom)))
    available_scratch_bytes = int(shutil.disk_usage(temporary_parent).free)
    if available_scratch_bytes < required_scratch_bytes:
        raise ValueError(
            "Insufficient GRIT scratch capacity: "
            f"required={required_scratch_bytes}, available={available_scratch_bytes}, "
            f"path={temporary_parent}"
        )
    patch_size = _model_patch_size(model)
    with tempfile.TemporaryDirectory(
        prefix="dinov3-grit-", dir=temporary_parent
    ) as temporary_dir:
        work_dir = Path(temporary_dir)
        layer_global: np.memmap | None = None
        dense_settling = np.memmap(
            work_dir / "dense_settling.f32",
            mode="w+",
            dtype=np.float32,
            shape=(len(manifest),),
        )
        loader = DataLoader(
            _ManifestImages(
                manifest,
                views=False,
                archive_cache_size=archive_cache_size,
            ),
            batch_size=batch_size,
            sampler=_archive_local_order(manifest),
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
        for batch in loader:
            images = batch["canonical"].to(device, non_blocking=True)
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                global_features, patch_features = _selected_layers(model, images)
            global_numpy = global_features.float().cpu().numpy()
            if layer_global is None:
                layer_global = np.memmap(
                    work_dir / "global_layers.f32",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(manifest), *global_numpy.shape[1:]),
                )
            indices = batch["index"].numpy()
            layer_global[indices] = global_numpy
            dense_settling[indices] = _patch_settling(
                patch_features[:, 2].float().cpu().numpy(),
                patch_features[:, 3].float().cpu().numpy(),
            )
        if layer_global is None:
            raise ValueError("GRIT manifest is empty")
        layer_global.flush()
        dense_settling.flush()

        view_features: dict[str, np.memmap] = {}
        retrieval = np.memmap(
            work_dir / "patch_retrieval.f32",
            mode="w+",
            dtype=np.float32,
            shape=(len(queries),),
        )
        query_loader = DataLoader(
            _ManifestImages(
                queries,
                views=True,
                archive_cache_size=archive_cache_size,
            ),
            batch_size=batch_size,
            sampler=_archive_local_order(queries),
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
        )
        for batch in query_loader:
            outputs = []
            indices = batch["index"].numpy()
            for name in ("canonical", "view0", "view1"):
                images = batch[name].to(device, non_blocking=True)
                with torch.autocast(
                    device.type,
                    dtype=torch.bfloat16,
                    enabled=amp and device.type == "cuda",
                ):
                    output = model(images)
                cls = _normalize(output["x_norm_clstoken"]).float().cpu().numpy()
                patches = _normalize(output["x_norm_patchtokens"])
                if name not in view_features:
                    view_features[name] = np.memmap(
                        work_dir / f"query_{name}.f32",
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(queries), cls.shape[1]),
                    )
                view_features[name][indices] = cls
                outputs.append(patches)
            retrieval[indices] = (
                torch.stack(
                    [
                        _patch_retrieval(
                            outputs[0], outputs[1], 0, patch_size
                        ),
                        _patch_retrieval(
                            outputs[0], outputs[2], 1, patch_size
                        ),
                    ],
                    dim=1,
                )
                .mean(dim=1)
                .float()
                .cpu()
                .numpy()
            )
        for values in view_features.values():
            values.flush()
        retrieval.flush()

        result = derive_grit_observations(
            manifest,
            global_layers=layer_global,
            dense_settling=dense_settling,
            query_views=view_features,
            patch_retrieval=retrieval,
            settling_k=int(config.get("settling_k", 50)),
            view_ks=tuple(map(int, config.get("view_ks", (8, 16, 32)))),
            neighbor_device=config.get("neighbor_device", str(device)),
            neighbor_block_rows=int(
                config.get("neighbor_block_rows", NEIGHBOR_BLOCK_ROWS)
            ),
            neighbor_backend=str(config.get("neighbor_backend", "auto")),
        )
        query_embeddings = pd.read_parquet(
            config["input_parquet"],
            pre_buffer=False,
            columns=["sample_id", "embedding"],
            filters=[("role", "==", "query")],
        )
        if query_embeddings["sample_id"].duplicated().any():
            raise ValueError("GRIT query embedding identities are not unique")
        embedding_by_id = query_embeddings.set_index("sample_id")["embedding"]
        result["embedding"] = result["sample_id"].map(embedding_by_id)
        if result["embedding"].isnull().any():
            raise ValueError("GRIT query embeddings do not cover every scored row")
    result.attrs["backbone_type"] = model.grit_backbone_type
    result.attrs["backbone_depth"] = model.n_blocks
    result.attrs["layer_numbers_one_based"] = list(
        relative_layer_numbers(model.n_blocks)
    )
    result.attrs["observation_contract_version"] = (
        "grit_vitb_blocks_3_6_9_12_v1"
        if model.grit_backbone_type == "vit_b"
        else "grit_relative_depth_quartiles_v1"
    )
    result.attrs["required_scratch_bytes"] = required_scratch_bytes
    result.attrs["available_scratch_bytes_at_start"] = available_scratch_bytes
    result.attrs["scratch_path"] = temporary_parent
    result.attrs["patch_size"] = patch_size
    return result
