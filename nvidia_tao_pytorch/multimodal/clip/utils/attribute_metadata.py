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

"""Utilities for PAS scalar attribute metadata."""

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple


_MISSING_MATCH_VALUE = -1
_MISSING_MATCH_LABELS = {"not visible"}


def _normalize_label(label: str) -> str:
    """Normalize exported vocab labels for matching policy checks."""
    return " ".join(str(label).strip().lower().replace("_", " ").split())


def load_missing_match_value_ids(pairs_file: Path) -> Tuple[Set[int], ...]:
    """Load per-attribute vocab IDs that should behave like missing values.

    PAS metadata stores scalar attribute values as integer IDs. Values such as
    ``not visible`` are normalized to ``-1`` on both sides before matching.
    On the text side, a negative value means the query does not constrain that
    attribute. On the image side, it means the attribute is unknown and cannot
    satisfy a specified query value.
    """
    vocab_path = Path(pairs_file).with_name("attribute_vocab.json")
    if not vocab_path.is_file():
        return ()

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    attributes = vocab.get("attributes") or []
    value_to_id = vocab.get("value_to_id") or {}
    id_to_value = vocab.get("id_to_value") or {}
    missing_ids: List[Set[int]] = []

    for attr_name in attributes:
        ids: Set[int] = set()
        by_value = value_to_id.get(attr_name) or {}
        if isinstance(by_value, dict):
            for label, value_id in by_value.items():
                if _normalize_label(label) in _MISSING_MATCH_LABELS:
                    ids.add(int(value_id))

        by_id = id_to_value.get(attr_name) or {}
        if isinstance(by_id, list):
            for value_id, label in enumerate(by_id):
                if _normalize_label(label) in _MISSING_MATCH_LABELS:
                    ids.add(int(value_id))
        elif isinstance(by_id, dict):
            for value_id, label in by_id.items():
                if _normalize_label(label) in _MISSING_MATCH_LABELS:
                    ids.add(int(value_id))

        missing_ids.append(ids)

    return tuple(missing_ids)


def normalize_missing_match_values(
    values: Sequence[int],
    missing_ids_by_attr: Optional[Sequence[Iterable[int]]] = None,
) -> List[int]:
    """Convert match-missing vocab values to the negative missing sentinel."""
    normalized = list(values)
    if not missing_ids_by_attr:
        return normalized

    for attr_idx, missing_ids in enumerate(missing_ids_by_attr):
        if attr_idx >= len(normalized):
            break
        if normalized[attr_idx] in missing_ids:
            normalized[attr_idx] = _MISSING_MATCH_VALUE
    return normalized
