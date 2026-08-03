# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Direct retrieval evaluation for TAO-FT NVIDIA PAS exports."""

from __future__ import annotations

import csv
import inspect
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.multimodal.clip.utils.attribute_metadata import (
    load_missing_match_value_ids,
    normalize_missing_match_values,
)


PAS_QUERY_TYPES: Tuple[str, ...] = (
    "easy",
    "medium",
    "hard",
    "natural_caption",
    "original_captions",
)
PAS_METADATA_QUERY_TYPES: Tuple[str, ...] = ("easy", "medium", "hard")
PAS_GROUND_TRUTH_MODES: Tuple[str, ...] = (
    "paired_caption",
    "scalar_attributes",
    "scalar_plus_accessories",
)
PAS_K = 5
PAS_EVAL_QUERY_BATCH = int(os.environ.get("TAO_PAS_EVAL_QUERY_BATCH", "512"))


@dataclass(frozen=True)
class PasPair:
    """One row from a TAO-FT ``*_pairs.json`` export."""

    dataset: str
    query_type: str
    caption: str
    unique_name: str
    image_path: str
    prepared_image_path: Path
    image_attr_values: Tuple[int, ...] = ()
    text_attr_values: Tuple[int, ...] = ()
    image_accessory_ids: Optional[Tuple[int, ...]] = None
    text_accessory_ids: Optional[Tuple[int, ...]] = None


class _ImageEmbeddingDataset(Dataset):
    """Minimal image-only dataset for embedding extraction."""

    def __init__(self, items: Sequence[Tuple[str, Path]], transform):
        self.items = list(items)
        self.transform = transform

    def __len__(self):
        """Return the number of unique gallery images."""
        return len(self.items)

    def __getitem__(self, idx):
        """Load one prepared image and its stable source-image key."""
        key, path = self.items[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, key


def _cfg_get(obj, key: str, default=None):
    """Read a key from a mapping, DictConfig, or dataclass-like object."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def _normalize_pas_ground_truth_mode(value) -> str:
    """Return a validated PAS ground-truth mode."""
    mode = str(value or "paired_caption").strip().lower()
    if mode not in PAS_GROUND_TRUTH_MODES:
        raise ValueError(
            "evaluate.pas_ground_truth_mode must be one of "
            f"{PAS_GROUND_TRUTH_MODES}, got {value!r}."
        )
    return mode


def _pairs_path_from_image_list(
    image_list_file: Optional[str],
) -> Optional[Path]:
    """Infer a split pairs path from its aligned image-list path."""
    if not image_list_file:
        return None
    path = Path(image_list_file)
    suffix = "_list.txt"
    if path.name.endswith(suffix):
        return path.with_name(path.name[: -len(suffix)] + "_pairs.json")
    return None


def _direct_pas_dataset_configs(experiment_config) -> List:
    """Return datasets that explicitly opt into direct PAS evaluation."""
    eval_cfg = getattr(experiment_config, "evaluate", None)
    eval_datasets = _cfg_get(eval_cfg, "datasets", None)
    return list(eval_datasets or [])


def find_pas_pairs_file(experiment_config) -> Optional[Path]:
    """Return a pairs file from explicit direct-PAS evaluation datasets."""
    for dataset in _direct_pas_dataset_configs(experiment_config):
        explicit_path = _cfg_get(dataset, "attribute_pairs_file")
        pairs_file = (
            Path(explicit_path)
            if explicit_path
            else _pairs_path_from_image_list(_cfg_get(dataset, "image_list_file"))
        )
        if pairs_file and pairs_file.is_file():
            return pairs_file
    return None


def _pas_ground_truth_mode(experiment_config) -> str:
    eval_cfg = getattr(experiment_config, "evaluate", None)
    return _normalize_pas_ground_truth_mode(
        _cfg_get(eval_cfg, "pas_ground_truth_mode", "paired_caption")
    )


def _iter_json_records(path: Path) -> Iterator[Dict]:
    """Iterate compact or pretty-printed TAO-FT pairs JSON arrays."""
    with open(path, "r", encoding="utf-8") as file:
        file.readline()
        second_line = file.readline()

    compact_lines = second_line.lstrip().startswith(
        "{"
    ) and second_line.rstrip().rstrip(",").endswith("}")
    if compact_lines:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                value = line.strip()
                if not value or value in ("[", "]"):
                    continue
                if value.endswith(","):
                    value = value[:-1]
                if value:
                    yield json.loads(value)
        return

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    yield from data


def _infer_dataset(image_path: str, explicit_dataset: str = "") -> str:
    """Infer the source dataset from a TAO-FT PAS image path."""
    if explicit_dataset:
        return str(explicit_dataset).strip()

    parts = [part for part in str(image_path).replace("\\", "/").split("/") if part]
    for marker in ("images", "data"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1].strip()
    if len(parts) > 1:
        return parts[0].strip()
    return ""


def _load_attribute_vocab_width(pairs_file: Path) -> int:
    """Load and validate the scalar attribute vocabulary contract."""
    vocab_path = pairs_file.with_name("attribute_vocab.json")
    if not vocab_path.is_file():
        raise ValueError("PAS scalar metadata matching requires " f"{vocab_path}.")
    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)
    if not isinstance(vocab, dict):
        raise ValueError(f"{vocab_path} must contain a JSON object.")
    attributes = vocab.get("attributes")
    if (
        not isinstance(attributes, list)
        or not attributes
        or any(
            not isinstance(attribute, str) or not attribute.strip()
            for attribute in attributes
        )
        or len(attributes) != len(set(attributes))
    ):
        raise ValueError(
            f"{vocab_path} must contain a non-empty ordered list of unique "
            "attribute names."
        )
    return len(attributes)


def _attr_tuple(
    value,
    *,
    field_name: str,
    context: str,
    expected_width: Optional[int],
    missing_ids_by_attr=None,
    required: bool,
) -> Tuple[int, ...]:
    """Validate and normalize one exported scalar attribute vector."""
    if value is None and not required:
        return ()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)):
        if required:
            raise ValueError(f"{context}: {field_name} must be a list.")
        return ()
    if not value:
        if required:
            raise ValueError(f"{context}: {field_name} must not be empty.")
        return ()
    if expected_width is not None and len(value) != expected_width:
        raise ValueError(
            f"{context}: {field_name} has length {len(value)}, expected "
            f"{expected_width}."
        )
    if any(type(attribute_id) is not int for attribute_id in value):
        raise ValueError(f"{context}: {field_name} must contain actual integer IDs.")
    return tuple(normalize_missing_match_values(value, missing_ids_by_attr))


def _validate_accessory_ids(
    value,
    *,
    field_name: str,
    context: str,
    valid_ids: Optional[Set[int]] = None,
) -> Tuple[int, ...]:
    """Validate one serialized accessory-ID list."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context}: {field_name} must be a list of integer IDs.")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{context}: {field_name} must contain only integer IDs.")
    ids = tuple(value)
    if any(accessory_id <= 0 for accessory_id in ids):
        raise ValueError(f"{context}: {field_name} must contain only positive IDs.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{context}: {field_name} must contain unique IDs.")
    if valid_ids is not None:
        unknown_ids = sorted(set(ids) - valid_ids)
        if unknown_ids:
            raise ValueError(
                f"{context}: {field_name} contains IDs absent from "
                f"accessory_vocab.json: {unknown_ids}."
            )
    return ids


def _optional_accessory_tuple(value) -> Optional[Tuple[int, ...]]:
    """Best-effort accessory parsing for paired/scalar-only evaluation."""
    if not isinstance(value, (list, tuple)):
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    return tuple(value)


def _load_pas_accessory_vocab(pairs_file: Path) -> Set[int]:
    """Load and validate the split-complete accessory vocabulary."""
    vocab_path = pairs_file.with_name("accessory_vocab.json")
    if not vocab_path.is_file():
        raise ValueError(
            "PAS accessory matching is required, but accessory_vocab.json "
            f"is missing next to {pairs_file}."
        )
    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)
    if not isinstance(vocab, Mapping):
        raise ValueError(f"{vocab_path} must be a JSON object.")

    source_splits = {
        str(split).strip().lower() for split in vocab.get("source_splits", [])
    }
    missing_splits = {"train", "val", "test"} - source_splits
    if missing_splits:
        raise ValueError(
            f"{vocab_path} does not cover required splits: "
            f"{sorted(missing_splits)}."
        )

    value_to_id = vocab.get("value_to_id")
    id_to_value = vocab.get("id_to_value")
    unknown_id = vocab.get("unknown_id")
    if not isinstance(value_to_id, Mapping) or not isinstance(id_to_value, list):
        raise ValueError(f"{vocab_path} must contain value_to_id and id_to_value.")
    if (
        isinstance(unknown_id, bool)
        or not isinstance(unknown_id, int)
        or unknown_id != 0
    ):
        raise ValueError(f"{vocab_path} must define integer unknown_id 0.")

    valid_ids: Set[int] = set()
    for value, accessory_id in value_to_id.items():
        if (
            isinstance(accessory_id, bool)
            or not isinstance(accessory_id, int)
            or accessory_id < 0
            or accessory_id >= len(id_to_value)
        ):
            raise ValueError(
                f"{vocab_path} maps {value!r} to invalid ID " f"{accessory_id!r}."
            )
        if str(id_to_value[accessory_id]) != str(value):
            raise ValueError(
                f"{vocab_path} has inconsistent mappings at ID " f"{accessory_id}."
            )
        if accessory_id > 0:
            valid_ids.add(accessory_id)
    return valid_ids


def _pair_image_key(pair: PasPair) -> str:
    """Return the stable source-image identity for a PAS row."""
    image_path = str(pair.image_path).replace("\\", "/").strip()
    return f"{pair.dataset}\t{image_path}"


def _validate_required_scalar_pairs(
    pairs: Sequence[PasPair],
    expected_width: int,
) -> None:
    """Validate scalar widths and repeated-image metadata consistency."""
    image_attributes: Dict[str, Tuple[int, ...]] = {}
    for pair in pairs:
        if pair.query_type not in PAS_METADATA_QUERY_TYPES:
            continue
        context = f"PAS pair {pair.unique_name!r} ({pair.query_type})"
        if len(pair.image_attr_values) != expected_width:
            raise ValueError(
                f"{context}: image_attr_values has length "
                f"{len(pair.image_attr_values)}, expected {expected_width}."
            )
        if len(pair.text_attr_values) != expected_width:
            raise ValueError(
                f"{context}: text_attr_values has length "
                f"{len(pair.text_attr_values)}, expected {expected_width}."
            )
        image_key = _pair_image_key(pair)
        previous = image_attributes.setdefault(image_key, pair.image_attr_values)
        if previous != pair.image_attr_values:
            raise ValueError(
                f"{context}: repeated image {image_key!r} has inconsistent "
                "scalar metadata."
            )


def _validate_required_accessory_pairs(
    pairs: Sequence[PasPair],
) -> None:
    """Validate accessory invariants across direct or file-loaded pairs."""
    image_accessories: Dict[str, Tuple[int, ...]] = {}
    for pair in pairs:
        context = f"PAS pair {pair.unique_name!r} ({pair.query_type})"
        if pair.image_accessory_ids is None or pair.text_accessory_ids is None:
            raise ValueError(
                f"{context}: accessory metadata is required but fields are " "missing."
            )
        image_ids = _validate_accessory_ids(
            pair.image_accessory_ids,
            field_name="image_accessory_ids",
            context=context,
        )
        text_ids = _validate_accessory_ids(
            pair.text_accessory_ids,
            field_name="text_accessory_ids",
            context=context,
        )
        if pair.query_type in ("easy", "medium") and text_ids:
            raise ValueError(
                f"{context}: easy/medium queries must not require " "accessories."
            )
        if pair.query_type == "hard" and not set(text_ids).issubset(image_ids):
            raise ValueError(
                f"{context}: hard-query accessories are not contained in "
                "the paired image."
            )
        image_key = _pair_image_key(pair)
        previous = image_accessories.setdefault(image_key, image_ids)
        if previous != image_ids:
            raise ValueError(
                f"{context}: repeated image {image_key!r} has inconsistent "
                "accessory metadata."
            )


def load_pas_pairs(
    dataset_cfg,
    pairs_file: Path,
    ground_truth_mode: str = "paired_caption",
) -> List[PasPair]:
    """Load and validate direct PAS evaluation rows."""
    ground_truth_mode = _normalize_pas_ground_truth_mode(ground_truth_mode)
    scalar_required = ground_truth_mode != "paired_caption"
    accessory_required = ground_truth_mode == "scalar_plus_accessories"
    attribute_width = (
        _load_attribute_vocab_width(pairs_file) if scalar_required else None
    )
    valid_accessory_ids = (
        _load_pas_accessory_vocab(pairs_file) if accessory_required else None
    )
    image_dir_value = _cfg_get(dataset_cfg, "image_dir")
    if not image_dir_value:
        raise ValueError("PAS evaluation requires dataset image_dir.")
    image_dir = Path(image_dir_value)
    missing_ids_by_attr = load_missing_match_value_ids(pairs_file)

    pairs: List[PasPair] = []
    for row_idx, row in enumerate(
        tqdm(
            _iter_json_records(pairs_file),
            desc=f"Loading {pairs_file.name}",
            unit="pair",
        ),
        start=1,
    ):
        if not isinstance(row, dict):
            raise ValueError(f"{pairs_file}:{row_idx} must be a JSON object.")
        caption = str(row.get("caption") or "").strip()
        query_type = str(row.get("query_type") or "").strip()
        unique_name = str(row.get("unique_name") or "").strip()
        image_path = str(row.get("image_path") or "").strip()
        if not caption or not query_type or not unique_name or not image_path:
            raise ValueError(
                f"{pairs_file}:{row_idx} requires caption, query_type, "
                "unique_name, and image_path."
            )
        if query_type not in PAS_QUERY_TYPES:
            raise ValueError(
                f"{pairs_file}:{row_idx} query_type must be one of "
                f"{PAS_QUERY_TYPES}, got {query_type!r}."
            )
        dataset = _infer_dataset(image_path, str(row.get("dataset") or ""))
        if not dataset:
            raise ValueError(
                f"{pairs_file}:{row_idx} could not infer the source dataset "
                f"from image_path {image_path!r}."
            )
        context = f"{pairs_file}:{row_idx} ({unique_name!r}, {query_type})"
        row_requires_scalar = scalar_required and query_type in PAS_METADATA_QUERY_TYPES
        image_attr_values = _attr_tuple(
            row.get("image_attr_values"),
            field_name="image_attr_values",
            context=context,
            expected_width=attribute_width if row_requires_scalar else None,
            missing_ids_by_attr=missing_ids_by_attr,
            required=row_requires_scalar,
        )
        text_attr_values = _attr_tuple(
            row.get("text_attr_values"),
            field_name="text_attr_values",
            context=context,
            expected_width=attribute_width if row_requires_scalar else None,
            missing_ids_by_attr=missing_ids_by_attr,
            required=row_requires_scalar,
        )

        if accessory_required:
            missing_fields = [
                field
                for field in (
                    "image_accessory_ids",
                    "text_accessory_ids",
                )
                if field not in row
            ]
            if missing_fields:
                raise ValueError(
                    f"{context}: accessory matching is required but fields "
                    f"are missing: {missing_fields}."
                )
            image_accessory_ids = _validate_accessory_ids(
                row["image_accessory_ids"],
                field_name="image_accessory_ids",
                context=context,
                valid_ids=valid_accessory_ids,
            )
            text_accessory_ids = _validate_accessory_ids(
                row["text_accessory_ids"],
                field_name="text_accessory_ids",
                context=context,
                valid_ids=valid_accessory_ids,
            )
        else:
            image_accessory_ids = _optional_accessory_tuple(
                row.get("image_accessory_ids")
            )
            text_accessory_ids = _optional_accessory_tuple(
                row.get("text_accessory_ids")
            )

        pairs.append(
            PasPair(
                dataset=dataset,
                query_type=query_type,
                caption=caption,
                unique_name=unique_name,
                image_path=image_path,
                prepared_image_path=image_dir / unique_name,
                image_attr_values=image_attr_values,
                text_attr_values=text_attr_values,
                image_accessory_ids=image_accessory_ids,
                text_accessory_ids=text_accessory_ids,
            )
        )

    if scalar_required:
        _validate_required_scalar_pairs(pairs, attribute_width)
    if accessory_required:
        _validate_required_accessory_pairs(pairs)
    return pairs


def resolve_pas_eval_data(
    experiment_config,
    pairs_file: Optional[Path] = None,
) -> Optional[Tuple[List[PasPair], Path]]:
    """Resolve and load the configured PAS export, if present."""
    pairs_file = pairs_file or find_pas_pairs_file(experiment_config)
    if pairs_file is None:
        return None
    datasets = _direct_pas_dataset_configs(experiment_config)
    if not datasets:
        raise ValueError(
            "Direct PAS evaluation requires explicit evaluate.datasets."
        )

    selected_dataset = None
    for dataset in datasets:
        explicit_path = _cfg_get(dataset, "attribute_pairs_file")
        candidate = (
            Path(explicit_path)
            if explicit_path
            else _pairs_path_from_image_list(_cfg_get(dataset, "image_list_file"))
        )
        if candidate == pairs_file:
            selected_dataset = dataset
            break
    if selected_dataset is None:
        selected_dataset = datasets[0]

    pairs = load_pas_pairs(
        selected_dataset,
        pairs_file,
        ground_truth_mode=_pas_ground_truth_mode(experiment_config),
    )
    if not pairs:
        raise ValueError(f"No PAS benchmark pairs loaded from {pairs_file}.")
    logging.info(
        "Loaded %s PAS pairs from %s",
        f"{len(pairs):,}",
        pairs_file,
    )
    return pairs, pairs_file


def _move_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _feature_array(features: torch.Tensor) -> np.ndarray:
    features = torch.nn.functional.normalize(features.float(), dim=-1)
    return features.detach().cpu().numpy().astype(np.float32, copy=False)


def build_pas_embedding_maps_from_rows(
    pairs: Sequence[PasPair],
    row_indices,
    image_features,
    text_features,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Deduplicate aligned normalized rows into PAS image/caption embeddings."""
    row_indices = np.asarray(row_indices)
    if row_indices.ndim != 1 or not np.issubdtype(
        row_indices.dtype, np.integer
    ):
        raise ValueError("PAS validation row indices must be a 1-D integer array.")
    image_features = (
        torch.as_tensor(image_features)
        .float()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    text_features = (
        torch.as_tensor(text_features)
        .float()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    row_count = len(row_indices)
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("PAS validation embeddings must be 2-D arrays.")
    if len(image_features) != row_count or len(text_features) != row_count:
        raise ValueError(
            "PAS validation row/embedding counts must match, got "
            f"{row_count}, {len(image_features)}, and {len(text_features)}."
        )

    invalid = sorted(
        {
            int(index)
            for index in row_indices
            if int(index) < 0 or int(index) >= len(pairs)
        }
    )
    if invalid:
        raise ValueError(
            "PAS validation row indices are outside the pairs export: "
            f"{invalid[:5]}."
        )
    observed = {int(index) for index in row_indices}
    missing = sorted(set(range(len(pairs))) - observed)
    if missing:
        raise ValueError(
            "PAS validation must cover the complete pairs export; missing "
            f"{len(missing)} rows, including {missing[:5]}."
        )

    image_embeddings: Dict[str, np.ndarray] = {}
    text_embeddings: Dict[str, np.ndarray] = {}
    for position in np.argsort(row_indices, kind="stable"):
        pair = pairs[int(row_indices[position])]
        image_embeddings.setdefault(
            _pair_image_key(pair),
            image_features[position],
        )
        text_embeddings.setdefault(
            pair.caption,
            text_features[position],
        )
    return image_embeddings, text_embeddings


def _call_with_supported_kwargs(function, kwargs):
    """Call a feature method with only the keyword arguments it accepts."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(**kwargs)
    accepted = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return function(**accepted)


def _as_feature_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        hidden_state = output.last_hidden_state
        return hidden_state[:, 0] if hidden_state.dim() == 3 else hidden_state
    raise TypeError(f"Expected tensor or model feature output, got {type(output)}.")


def _inner_hf_model(adapter):
    backbone = getattr(adapter, "backbone", None)
    return getattr(backbone, "inner", None)


def _encode_pas_image_features(adapter, images):
    inner = _inner_hf_model(adapter)
    feature_function = getattr(inner, "get_image_features", None)
    if callable(feature_function):
        kwargs = (
            dict(images) if isinstance(images, Mapping) else {"pixel_values": images}
        )
        return _as_feature_tensor(_call_with_supported_kwargs(feature_function, kwargs))
    output = adapter(image=images)
    return output["image_features"] if isinstance(output, dict) else output[0]


def _encode_pas_text_features(adapter, tokens):
    inner = _inner_hf_model(adapter)
    feature_function = getattr(inner, "get_text_features", None)
    if callable(feature_function):
        kwargs = dict(tokens) if isinstance(tokens, Mapping) else {"input_ids": tokens}
        return _as_feature_tensor(_call_with_supported_kwargs(feature_function, kwargs))
    output = adapter(text=tokens)
    return output["text_features"] if isinstance(output, dict) else output[1]


def _unique_image_candidates(
    pairs: Sequence[PasPair],
) -> List[Tuple[str, Path]]:
    """Return one prepared path per unique source image."""
    by_key: Dict[str, Path] = {}
    for pair in pairs:
        by_key.setdefault(_pair_image_key(pair), pair.prepared_image_path)
    return sorted(by_key.items())


def _resolve_existing_image_items(
    items: Sequence[Tuple[str, Path]],
    pairs: Sequence[PasPair],
) -> List[Tuple[str, Path]]:
    """Resolve a later prepared path when the first duplicate is missing."""
    by_key: Dict[str, Path] = dict(items)
    missing = {key for key, path in by_key.items() if not path.is_file()}
    if missing:
        for pair in pairs:
            key = _pair_image_key(pair)
            if key in missing and pair.prepared_image_path.is_file():
                by_key[key] = pair.prepared_image_path
                missing.remove(key)
                if not missing:
                    break
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(
            "PAS prepared images are missing for source keys: " f"{examples}."
        )
    return sorted(by_key.items())


def _extract_image_embeddings(
    model,
    pairs: Sequence[PasPair],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Extract one normalized embedding per source image."""
    items = _resolve_existing_image_items(
        _unique_image_candidates(pairs),
        pairs,
    )
    output: Dict[str, np.ndarray] = {}

    dataset = _ImageEmbeddingDataset(items, model.preprocess_val)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    with torch.no_grad():
        for images, keys in tqdm(
            dataloader,
            desc="Embedding PAS images",
            unit="batch",
        ):
            images = _move_to_device(images, device)
            features = _encode_pas_image_features(model.model, images)
            for key, embedding in zip(keys, _feature_array(features)):
                output[str(key)] = embedding
    return output


def _unique_captions(pairs: Sequence[PasPair]) -> List[str]:
    return sorted({pair.caption for pair in pairs})


def _tokenize_batch(tokenizer, captions: Sequence[str]):
    tokens = tokenizer(list(captions))
    if isinstance(tokens, list):
        tokens = tokens[0]
    return tokens


def _extract_text_embeddings(
    model,
    captions: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Extract one normalized embedding per exact caption."""
    output: Dict[str, np.ndarray] = {}

    model.eval()
    with torch.no_grad():
        starts = range(0, len(captions), batch_size)
        for start in tqdm(starts, desc="Embedding PAS text", unit="batch"):
            batch_captions = list(captions[start : start + batch_size])
            tokens = _move_to_device(
                _tokenize_batch(model.tokenizer, batch_captions),
                device,
            )
            features = _encode_pas_text_features(model.model, tokens)
            for caption, embedding in zip(batch_captions, _feature_array(features)):
                output[caption] = embedding
    return output


def _dataset_names_from_pairs(
    pairs: Sequence[PasPair],
) -> Tuple[str, ...]:
    seen: Set[str] = set()
    names: List[str] = []
    for pair in pairs:
        if pair.dataset and pair.dataset not in seen:
            seen.add(pair.dataset)
            names.append(pair.dataset)
    return tuple(names)


def _dataset_names_from_rows(
    rows: Sequence[Mapping],
) -> Tuple[str, ...]:
    seen: Set[str] = set()
    names: List[str] = []
    for row in rows:
        dataset = str(row.get("Dataset") or "")
        if not dataset or dataset.startswith("AVG_") or dataset.startswith("WAVG_"):
            continue
        if dataset not in seen:
            seen.add(dataset)
            names.append(dataset)
    return tuple(names)


def _pas_dataset_eval_order(
    pairs: Sequence[PasPair],
) -> Tuple[str, ...]:
    names = list(_dataset_names_from_pairs(pairs))
    names.sort(key=lambda name: (name != "RSTPReid", name))
    return tuple(names)


def _compute_query_metrics(
    similarities: np.ndarray,
    gt_indices: Sequence[int],
    k: int = PAS_K,
) -> Optional[Dict]:
    """Compute PAS metrics from the ranks of positive gallery items."""
    if not gt_indices:
        return None
    gallery_size = len(similarities)
    ground_truth = np.array(
        sorted({int(index) for index in gt_indices}),
        dtype=np.int64,
    )
    if ground_truth.size == 0:
        return None
    if np.any(ground_truth < 0) or np.any(ground_truth >= gallery_size):
        raise ValueError("PAS ground-truth index is outside the gallery.")

    # Compute every metric from one explicit ranking. Similarities are sorted
    # descending, with the original gallery index as a deterministic,
    # label-independent tie-break.
    ranking = np.lexsort(
        (
            np.arange(gallery_size, dtype=np.int64),
            -similarities,
        )
    )
    is_positive = np.zeros(gallery_size, dtype=bool)
    is_positive[ground_truth] = True
    positive_ranks = (
        np.flatnonzero(is_positive[ranking]).astype(np.float64) + 1.0
    )
    precision_at_hits = (
        np.arange(
            1,
            len(positive_ranks) + 1,
            dtype=np.float64,
        )
        / positive_ranks
    )
    average_precision = float(np.mean(precision_at_hits))

    matches_in_top_k = int(np.count_nonzero(positive_ranks <= k))
    top_k_rate = matches_in_top_k / min(k, len(positive_ranks))
    num_positives = len(positive_ranks)
    num_negatives = gallery_size - num_positives
    if num_negatives == 0:
        auc = 1.0
    else:
        ascending_positive_ranks = gallery_size - positive_ranks + 1
        positive_rank_sum = float(np.sum(ascending_positive_ranks))
        auc = (positive_rank_sum - num_positives * (num_positives + 1) / 2) / (
            num_positives * num_negatives
        )

    return {
        "ap": average_precision,
        "rank1": float(positive_ranks[0] <= 1),
        "rank5": float(positive_ranks[0] <= 5),
        "auc": float(auc),
        "top_k_rate": float(top_k_rate),
        "zero_match": float(matches_in_top_k == 0),
        "first_pos": float(positive_ranks[0]),
        "num_gt": num_positives,
    }


def _finalize_metrics(
    raw_metrics: Sequence[Dict],
    gallery_size: int,
    k: int,
) -> Optional[Dict]:
    if not raw_metrics:
        return None
    total_ground_truth = sum(float(metrics["num_gt"]) for metrics in raw_metrics)
    return {
        "num_queries": len(raw_metrics),
        "gallery_size": gallery_size,
        "avg_gt_per_query": (total_ground_truth / len(raw_metrics)),
        "mAP": float(np.mean([metrics["ap"] for metrics in raw_metrics])),
        "Rank-1": float(np.mean([metrics["rank1"] for metrics in raw_metrics])),
        "Rank-5": float(np.mean([metrics["rank5"] for metrics in raw_metrics])),
        "Separability": float(np.mean([metrics["auc"] for metrics in raw_metrics])),
        f"Match@{k}": float(
            np.mean([metrics["top_k_rate"] for metrics in raw_metrics])
        ),
        f"Zero@{k}": float(np.mean([metrics["zero_match"] for metrics in raw_metrics])),
        "First Pos": float(
            np.median([metrics["first_pos"] for metrics in raw_metrics])
        ),
    }


def _gallery_by_dataset(
    pairs: Sequence[PasPair],
    image_embeddings: Mapping[str, np.ndarray],
) -> Dict[str, List[str]]:
    """Build isolated source-image galleries for each PAS dataset."""
    gallery: Dict[str, List[str]] = {
        dataset: [] for dataset in _dataset_names_from_pairs(pairs)
    }
    seen: Dict[str, Set[str]] = {dataset: set() for dataset in gallery}
    for pair in pairs:
        image_key = _pair_image_key(pair)
        if image_key not in image_embeddings:
            continue
        gallery.setdefault(pair.dataset, [])
        seen.setdefault(pair.dataset, set())
        if image_key in seen[pair.dataset]:
            continue
        seen[pair.dataset].add(image_key)
        gallery[pair.dataset].append(image_key)
    return gallery


def _group_text_queries(
    pairs: Sequence[PasPair],
) -> Dict[Tuple[str, str, str], Set[str]]:
    groups: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    for pair in pairs:
        if pair.query_type in PAS_QUERY_TYPES:
            groups[(pair.dataset, pair.query_type, pair.caption)].add(
                _pair_image_key(pair)
            )
    return groups


def _evaluate_query_items(
    query_items: Sequence[Tuple[str, List[int]]],
    gallery_embeddings: np.ndarray,
    text_embeddings: Mapping[str, np.ndarray],
    description: str,
    k: int,
) -> Optional[Dict]:
    """Score one dataset/query-type slice in bounded query batches."""
    raw_metrics: List[Dict] = []
    for start in tqdm(
        range(0, len(query_items), PAS_EVAL_QUERY_BATCH),
        desc=description,
        leave=False,
    ):
        batch_items = query_items[start : start + PAS_EVAL_QUERY_BATCH]
        query_embeddings = np.stack(
            [text_embeddings[caption] for caption, _ in batch_items]
        )
        similarities_batch = query_embeddings @ gallery_embeddings.T
        for similarities, (_, ground_truth) in zip(
            similarities_batch, batch_items
        ):
            metrics = _compute_query_metrics(similarities, ground_truth, k=k)
            if metrics:
                raw_metrics.append(metrics)
    return _finalize_metrics(raw_metrics, len(gallery_embeddings), k)


def evaluate_text_to_image(
    pairs: Sequence[PasPair],
    image_embeddings: Mapping[str, np.ndarray],
    text_embeddings: Mapping[str, np.ndarray],
    k: int = PAS_K,
) -> List[Dict]:
    """Evaluate exact-caption positives for each dataset/query type."""
    query_groups = _group_text_queries(pairs)
    gallery = _gallery_by_dataset(pairs, image_embeddings)
    rows: List[Dict] = []

    for dataset in _pas_dataset_eval_order(pairs):
        gallery_names = gallery.get(dataset, [])
        if not gallery_names:
            continue
        gallery_embeddings = np.stack(
            [image_embeddings[name] for name in gallery_names]
        )
        gallery_index = {name: index for index, name in enumerate(gallery_names)}

        for query_type in PAS_QUERY_TYPES:
            query_items: List[Tuple[str, List[int]]] = []
            for (
                query_dataset,
                query_kind,
                caption,
            ), image_keys in query_groups.items():
                if (
                    query_dataset != dataset
                    or query_kind != query_type
                    or caption not in text_embeddings
                ):
                    continue
                ground_truth = sorted(
                    gallery_index[key] for key in image_keys if key in gallery_index
                )
                if ground_truth:
                    query_items.append((caption, ground_truth))

            metrics = _evaluate_query_items(
                query_items,
                gallery_embeddings,
                text_embeddings,
                f"PAS {dataset}/{query_type}",
                k,
            )
            if metrics is not None:
                rows.append(
                    {
                        "Dataset": dataset,
                        "QueryType": query_type,
                        "EasyAttribute": "",
                        **metrics,
                    }
                )
    return rows


def _metadata_gt_indices(
    pair: PasPair,
    gallery_attributes: np.ndarray,
    gallery_indices_by_accessory: Mapping[int, Set[int]],
    accessory_required: bool,
) -> List[int]:
    """Return metadata-compatible gallery indices for one query."""
    if (
        not pair.text_attr_values
        or len(pair.text_attr_values) != gallery_attributes.shape[1]
    ):
        return []
    mask = np.ones(gallery_attributes.shape[0], dtype=bool)
    for attribute_index, value in enumerate(pair.text_attr_values):
        if value < 0:
            continue
        mask &= gallery_attributes[:, attribute_index] == value
        if not mask.any():
            return []

    ground_truth = set(np.flatnonzero(mask).tolist())
    if accessory_required:
        for accessory_id in pair.text_accessory_ids or ():
            ground_truth.intersection_update(
                gallery_indices_by_accessory.get(accessory_id, set())
            )
            if not ground_truth:
                return []
    return sorted(ground_truth)


def evaluate_text_to_image_by_metadata(
    pairs: Sequence[PasPair],
    image_embeddings: Mapping[str, np.ndarray],
    text_embeddings: Mapping[str, np.ndarray],
    k: int = PAS_K,
    ground_truth_mode: str = "scalar_attributes",
) -> List[Dict]:
    """Evaluate caption-deduplicated scalar/accessory PAS positives."""
    ground_truth_mode = _normalize_pas_ground_truth_mode(ground_truth_mode)
    if ground_truth_mode == "paired_caption":
        raise ValueError("Metadata evaluation requires a scalar ground-truth mode.")
    accessory_required = ground_truth_mode == "scalar_plus_accessories"
    if accessory_required:
        _validate_required_accessory_pairs(pairs)

    gallery = _gallery_by_dataset(pairs, image_embeddings)
    image_attributes: Dict[str, Tuple[int, ...]] = {}
    image_accessories: Dict[str, Tuple[int, ...]] = {}
    for pair in pairs:
        image_key = _pair_image_key(pair)
        if pair.image_attr_values:
            previous = image_attributes.setdefault(image_key, pair.image_attr_values)
            if previous != pair.image_attr_values:
                raise ValueError(
                    f"Repeated PAS image {image_key!r} has inconsistent "
                    "scalar metadata."
                )
        if accessory_required:
            image_accessories.setdefault(image_key, pair.image_accessory_ids or ())

    query_groups: Dict[
        Tuple[str, str, str],
        Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], PasPair],
    ] = defaultdict(dict)
    for pair in pairs:
        if pair.query_type not in PAS_METADATA_QUERY_TYPES:
            continue
        signature = (
            pair.text_attr_values,
            pair.text_accessory_ids or (),
        )
        query_groups[(pair.dataset, pair.query_type, pair.caption)].setdefault(
            signature, pair
        )

    rows: List[Dict] = []
    for dataset in _pas_dataset_eval_order(pairs):
        gallery_names = [
            name for name in gallery.get(dataset, []) if name in image_attributes
        ]
        if not gallery_names:
            continue
        widths = {len(image_attributes[name]) for name in gallery_names}
        if len(widths) != 1:
            raise ValueError(f"PAS dataset {dataset!r} has inconsistent scalar widths.")
        gallery_embeddings = np.stack(
            [image_embeddings[name] for name in gallery_names]
        )
        gallery_attributes = np.asarray(
            [image_attributes[name] for name in gallery_names],
            dtype=np.int64,
        )
        gallery_indices_by_accessory: Dict[int, Set[int]] = defaultdict(set)
        if accessory_required:
            for gallery_index, name in enumerate(gallery_names):
                for accessory_id in image_accessories[name]:
                    gallery_indices_by_accessory[accessory_id].add(gallery_index)

        for query_type in PAS_METADATA_QUERY_TYPES:
            query_items: List[Tuple[str, List[int]]] = []
            for (
                query_dataset,
                query_kind,
                caption,
            ), signatures in query_groups.items():
                if (
                    query_dataset != dataset
                    or query_kind != query_type
                    or caption not in text_embeddings
                ):
                    continue
                ground_truth: Set[int] = set()
                for pair in signatures.values():
                    ground_truth.update(
                        _metadata_gt_indices(
                            pair,
                            gallery_attributes,
                            gallery_indices_by_accessory,
                            accessory_required,
                        )
                    )
                if ground_truth:
                    query_items.append((caption, sorted(ground_truth)))

            metrics = _evaluate_query_items(
                query_items,
                gallery_embeddings,
                text_embeddings,
                f"PAS metadata {dataset}/{query_type}",
                k,
            )
            if metrics is not None:
                rows.append(
                    {
                        "Dataset": dataset,
                        "QueryType": query_type,
                        "EasyAttribute": "",
                        **metrics,
                    }
                )
    return rows


def evaluate_pas_metadata_embeddings(
    pairs: Sequence[PasPair],
    image_embeddings: Mapping[str, np.ndarray],
    text_embeddings: Mapping[str, np.ndarray],
    ground_truth_mode: str,
    k: int = PAS_K,
) -> Dict[str, List[Dict]]:
    """Evaluate and aggregate metadata-compatible PAS embeddings."""
    rows = evaluate_text_to_image_by_metadata(
        pairs,
        image_embeddings,
        text_embeddings,
        k,
        ground_truth_mode=ground_truth_mode,
    )
    dataset_names = _dataset_names_from_pairs(pairs)
    return {
        "metadata": rows,
        "metadata_aggregate": aggregate_metric_rows(
            rows,
            k,
            dataset_names=dataset_names,
        ),
        "metadata_weighted_aggregate": aggregate_metric_rows(
            rows,
            k,
            dataset_names=dataset_names,
            weighted=True,
        ),
    }


def _metric_fieldnames(k: int) -> List[str]:
    return [
        "Dataset",
        "QueryType",
        "EasyAttribute",
        "num_queries",
        "gallery_size",
        "avg_gt_per_query",
        "mAP",
        "Rank-1",
        "Rank-5",
        "Separability",
        f"Match@{k}",
        f"Zero@{k}",
        "First Pos",
    ]


def _format_row(
    row: Mapping,
    fieldnames: Sequence[str],
) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for key in fieldnames:
        value = row.get(key, "")
        if isinstance(value, float):
            output[key] = "" if math.isnan(value) else f"{value:.16g}"
        else:
            output[key] = str(value)
    return output


def _write_metrics_csv(
    path: Path,
    rows: Sequence[Mapping],
    k: int,
) -> None:
    fieldnames = _metric_fieldnames(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(_format_row(row, fieldnames))
    logging.info("Saved PAS metrics to %s", path)


def _mean_numeric(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _weighted_mean_numeric(
    values: Sequence[Tuple[float, float]],
) -> float:
    total_weight = sum(weight for _, weight in values)
    if not values or total_weight <= 0:
        return float("nan")
    return float(sum(value * weight for value, weight in values) / total_weight)


def _row_weight(row: Mapping) -> float:
    try:
        return float(row.get("num_queries") or 0)
    except (TypeError, ValueError):
        return 0.0


def aggregate_metric_rows(
    rows: Sequence[Mapping],
    k: int = PAS_K,
    dataset_names: Optional[Sequence[str]] = None,
    dataset_label: Optional[str] = None,
    weighted: bool = False,
) -> List[Dict]:
    """Aggregate each query type across PAS source datasets."""
    dataset_names = tuple(dataset_names or _dataset_names_from_rows(rows))
    prefix = "WAVG" if weighted else "AVG"
    dataset_label = dataset_label or f"{prefix}_{len(dataset_names)}_DATASETS"
    metric_keys = _metric_fieldnames(k)[3:]
    slices = [(query_type, "") for query_type in PAS_QUERY_TYPES]
    output: List[Dict] = []
    for query_type, easy_attribute in slices:
        subset = [
            row
            for row in rows
            if (
                row.get("Dataset") in dataset_names
                and row.get("QueryType") == query_type
                and str(row.get("EasyAttribute") or "") == easy_attribute
            )
        ]
        if not subset:
            continue
        aggregate: Dict = {
            "Dataset": dataset_label,
            "QueryType": query_type,
            "EasyAttribute": easy_attribute,
        }
        for key in metric_keys:
            if weighted and key == "num_queries":
                aggregate[key] = sum(_row_weight(row) for row in subset)
                continue
            if weighted and key == "First Pos":
                aggregate[key] = float("nan")
                continue

            values: List[float] = []
            weighted_values: List[Tuple[float, float]] = []
            for row in subset:
                try:
                    value = float(row.get(key))
                except (TypeError, ValueError):
                    continue
                if weighted:
                    weight = _row_weight(row)
                    if weight > 0:
                        weighted_values.append((value, weight))
                else:
                    values.append(value)
            aggregate[key] = (
                _weighted_mean_numeric(weighted_values)
                if weighted
                else _mean_numeric(values)
            )
        output.append(aggregate)
    return output


def _write_pas_metric_outputs(
    results_dir: Path,
    stem: str,
    rows: Sequence[Mapping],
    dataset_names: Sequence[str],
    k: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Write detailed, dataset-average, and query-weighted PAS metrics."""
    aggregate = aggregate_metric_rows(
        rows,
        k,
        dataset_names=dataset_names,
    )
    weighted = aggregate_metric_rows(
        rows,
        k,
        dataset_names=dataset_names,
        weighted=True,
    )
    for suffix, output_rows in (
        ("", rows),
        ("_aggregate", aggregate),
        ("_weighted_aggregate", weighted),
    ):
        _write_metrics_csv(
            results_dir / f"{stem}{suffix}.csv",
            output_rows,
            k,
        )
    return aggregate, weighted


def _pas_results_dir(experiment_config) -> Path:
    eval_cfg = getattr(experiment_config, "evaluate", None)
    results_dir = _cfg_get(eval_cfg, "results_dir")
    if results_dir:
        return Path(results_dir)
    root = getattr(experiment_config, "results_dir", None)
    return Path(root or "results") / "pas_eval"


def _pas_eval_batch_size(experiment_config) -> int:
    eval_cfg = getattr(experiment_config, "evaluate", None)
    dataset_cfg = getattr(experiment_config, "dataset", None)
    val_cfg = getattr(dataset_cfg, "val", None)
    value = _cfg_get(
        eval_cfg,
        "batch_size",
        _cfg_get(val_cfg, "batch_size", 16),
    )
    return int(value or 16)


def _pas_eval_num_workers(experiment_config) -> int:
    eval_cfg = getattr(experiment_config, "evaluate", None)
    dataset_cfg = getattr(experiment_config, "dataset", None)
    val_cfg = getattr(dataset_cfg, "val", None)
    value = _cfg_get(
        eval_cfg,
        "num_workers",
        _cfg_get(val_cfg, "num_workers", 4),
    )
    return int(value or 0)


def _pas_eval_device(experiment_config) -> torch.device:
    eval_cfg = getattr(experiment_config, "evaluate", None)
    gpu_ids = list(_cfg_get(eval_cfg, "gpu_ids", [0]) or [0])
    if torch.cuda.is_available():
        gpu_id = int(gpu_ids[0]) if gpu_ids else 0
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def run_pas_evaluation(
    experiment_config,
    model,
    pairs: Sequence[PasPair],
) -> Dict[str, List[Dict]]:
    """Embed a PAS export, evaluate it, and write result CSVs."""
    results_dir = _pas_results_dir(experiment_config)
    results_dir.mkdir(parents=True, exist_ok=True)
    batch_size = _pas_eval_batch_size(experiment_config)
    num_workers = _pas_eval_num_workers(experiment_config)
    device = _pas_eval_device(experiment_config)
    ground_truth_mode = _pas_ground_truth_mode(experiment_config)
    dataset_names = _dataset_names_from_pairs(pairs)

    model.to(device)
    image_embeddings = _extract_image_embeddings(
        model,
        pairs,
        batch_size,
        num_workers,
        device,
    )
    text_embeddings = _extract_text_embeddings(
        model,
        _unique_captions(pairs),
        batch_size,
        device,
    )

    paired_rows = evaluate_text_to_image(
        pairs, image_embeddings, text_embeddings, PAS_K
    )
    paired_aggregate, paired_weighted = _write_pas_metric_outputs(
        results_dir,
        "nvidia_pas_metrics",
        paired_rows,
        dataset_names,
        PAS_K,
    )

    output = {
        "paired": paired_rows,
        "paired_aggregate": paired_aggregate,
        "paired_weighted_aggregate": paired_weighted,
    }
    if ground_truth_mode != "paired_caption":
        metadata_output = evaluate_pas_metadata_embeddings(
            pairs,
            image_embeddings,
            text_embeddings,
            ground_truth_mode=ground_truth_mode,
            k=PAS_K,
        )
        for suffix, key in (
            ("", "metadata"),
            ("_aggregate", "metadata_aggregate"),
            ("_weighted_aggregate", "metadata_weighted_aggregate"),
        ):
            _write_metrics_csv(
                results_dir / f"nvidia_pas_metadata_metrics{suffix}.csv",
                metadata_output[key],
                PAS_K,
            )
        output.update(metadata_output)
    logging.info("PAS TAO-FT evaluation finished.")
    return output
