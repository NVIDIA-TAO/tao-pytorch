# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Video-text dataset support for CLIP-compatible training.

The loader normalizes common video retrieval metadata into samples with
``video_path``, ``caption``, temporal bounds, and an integer ``idx`` used by
InternVideo2 VTC multi-positive targets.
"""

import json
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from omegaconf import ListConfig
from PIL import Image
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    distributed,
)

from nvidia_tao_pytorch.core.tlt_logging import logging


def _cfg_get(cfg, key, default=None):
    """Read a config field from dict/OmegaConf/dataclass-like objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_dict(value):
    """Convert OmegaConf containers to plain Python containers when needed."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {k: v for k, v in value.items()}
    return {}


def _is_path_list(value):
    """True for a sequence of paths (list/tuple/OmegaConf ListConfig).

    A single ``str``/``bytes``/``Path`` (or a dict-style metadata object) is not
    treated as a path list.
    """
    if isinstance(value, (str, bytes, Path, dict)):
        return False
    return isinstance(value, (list, tuple, ListConfig))


def _read_single_metadata(path):
    """Load JSON or JSONL metadata from a single path."""
    metadata_path = Path(path)
    if metadata_path.suffix.lower() == ".jsonl":
        rows = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_metadata(metadata):
    """Load video-text metadata from a single path or a list of paths.

    A single ``str``/``Path`` is read as-is and may be a JSON array, JSONL, or a
    dict-style file (e.g. MSR-VTT). A list/tuple of paths reads each file and
    concatenates their records; every listed file must parse to a ``list`` (JSON
    array or JSONL), since dict-style metadata cannot be concatenated.
    """
    if not _is_path_list(metadata):
        return _read_single_metadata(metadata)
    combined = []
    per_file = []
    for path in metadata:
        data = _read_single_metadata(path)
        if not isinstance(data, list):
            raise ValueError(
                f"Multi-file video_text.metadata requires each file to be a JSON "
                f"array or JSONL (a list of records); {str(path)!r} parsed as "
                f"{type(data).__name__}. Provide a single path for msrvtt-style "
                f"dict metadata, or merge the files into one JSON array."
            )
        combined.extend(data)
        per_file.append((str(path), len(data)))
    logging.info(
        "video_text: concatenated %d metadata files -> %d records (%s)",
        len(per_file), len(combined),
        ", ".join(f"{p}:{n}" for p, n in per_file),
    )
    return combined


def _resolve_video_path(video_path, data_root=None, path_prefix_mapping=None):
    """Resolve original dataset paths to local paths."""
    if video_path is None:
        return ""
    video_path = str(video_path)
    mapping = path_prefix_mapping or {}

    for prefix, replacement in mapping.items():
        if prefix and video_path.startswith(prefix):
            return video_path.replace(prefix, str(replacement), 1)

    path = Path(video_path)
    if path.is_absolute():
        return str(path)
    if data_root:
        return str(Path(data_root) / path)
    return video_path


def _pick_first(row, keys, default=None):
    """Return the first present, non-empty value from a row."""
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _normalize_caption_fields(caption_fields):
    """Normalize caption field config."""
    if isinstance(caption_fields, str):
        return [caption_fields]
    fields = list(caption_fields or [])
    return fields or ["caption"]


def _caption_candidate_groups(row, chunk, caption_fields, is_anomaly):
    """Per-field candidate captions, preserving which field each came from.

    Returns a list of non-empty caption lists, one per ``caption_field`` that
    yielded any caption. ``caption_mode='one_per_field'`` explodes on these
    groups so every field gets equal weight regardless of how many captions it
    contributes; :func:`_caption_candidates` flattens them for the other modes.
    """
    groups = []
    for field in caption_fields:
        value = None
        if chunk is not None:
            value = chunk.get(field)
        if not value:
            value = row.get(field)
            if value and not is_anomaly and field in {
                "anomaly_type", "what", "where", "why", "how",
            }:
                value = "Normal"
        group = []
        if value:
            if isinstance(value, (list, tuple)):
                group.extend(str(v).strip() for v in value if str(v).strip())
            else:
                group.append(str(value))
        elif not is_anomaly:
            group.append("Normal")
        if group:
            groups.append(group)
    return groups


def _caption_candidates(row, chunk, caption_fields, is_anomaly):
    """Flat candidate captions across all fields (order preserved)."""
    return [
        cap
        for group in _caption_candidate_groups(
            row, chunk, caption_fields, is_anomaly
        )
        for cap in group
    ]


def _idx_source(entry, idx_mode, idx_field):
    """Return the raw idx source before integer encoding."""
    if idx_mode == "sample_id":
        return entry["sample_id"]
    if idx_mode == "video_id":
        return entry.get("video_id") or entry["sample_id"]
    if idx_mode == "category":
        return entry.get("category") or entry.get("caption") or entry["sample_id"]
    if idx_mode == "field":
        if not idx_field:
            raise ValueError("idx_mode='field' requires idx_field")
        return entry.get(idx_field) or entry["metadata"].get(idx_field)
    raise ValueError(
        f"Unsupported idx_mode: {idx_mode}. "
        "Choose from sample_id, video_id, category, field."
    )


def _assign_integer_idx(entries, idx_mode, idx_field):
    """Map string idx sources to stable integer ids."""
    mapping = {}
    for entry in entries:
        raw = _idx_source(entry, idx_mode, idx_field)
        if isinstance(raw, bool):
            raw = str(raw)
        if isinstance(raw, int):
            entry["idx"] = raw
            continue
        raw = str(raw)
        if raw not in mapping:
            mapping[raw] = len(mapping)
        entry["idx"] = mapping[raw]
    return entries


def _explode_caption_entries(entries):
    """Train-only: one entry per caption candidate, sharing the source idx.

    Each source entry already carries a flat ``captions`` list and an integer
    ``idx`` (assigned by :func:`_assign_integer_idx`). Emit one shallow copy
    per caption, collapsing ``captions`` to that single caption and setting
    ``caption`` to match. ``idx`` and ``sample_id`` are preserved verbatim, so
    all copies of a chunk stay mutual positives under the loss's
    ``torch.eq(idx, idx.T)`` target mask.
    """
    exploded = []
    for entry in entries:
        caps = entry.get("captions") or [entry.get("caption")]
        for cap in caps:
            new = dict(entry)
            new["caption"] = cap
            new["captions"] = [cap]
            exploded.append(new)
    return exploded


def _explode_caption_entries_by_field(entries):
    """Train-only: one entry per caption FIELD, sharing the source idx.

    Like :func:`_explode_caption_entries` but emits one copy per field group
    (``entry["caption_groups"]``, set by :func:`_flatten_vadr1_chunks`) instead
    of per caption. Each copy keeps that field's full caption list so the
    dataset samples one caption from it per epoch, giving every field equal
    weight regardless of cardinality. Falls back to a single group (the flat
    ``captions``) for formats that do not populate ``caption_groups``.
    """
    exploded = []
    for entry in entries:
        groups = entry.get("caption_groups") or [
            entry.get("captions") or [entry.get("caption")]
        ]
        for group in groups:
            new = dict(entry)
            new["captions"] = list(group)
            new["caption"] = group[0]
            exploded.append(new)
    return exploded


def _apply_caption_mode(entries, caption_mode):
    """Train-time caption expansion for ``caption_mode``.

    Single source of truth shared by build_video_text_dataset (the actual
    dataloader) and _entries_from_cfg (train.py schedule sizing) so the two
    can never disagree on the entry count.
    """
    if caption_mode == "all":
        return _explode_caption_entries(entries)
    if caption_mode == "one_per_field":
        return _explode_caption_entries_by_field(entries)
    return entries


def _coerce_number(value, caster, *, field, context):
    """Cast a metadata field to int/float, with a contextual error on failure."""
    try:
        return caster(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not parse {field}={value!r} as {caster.__name__} "
            f"for {context}."
        ) from exc


def _log_flatten_summary(fmt, kept, dropped, detail):
    """Log how many entries a flattener produced and why rows were dropped.

    Emits one ``info`` line with the per-reason breakdown, and escalates to
    ``warning`` when nothing usable was produced or when drops dominate the
    kept set (a strong signal of a wrong ``caption_fields``/``split`` or a
    schema mismatch that would otherwise pass silently).
    """
    logging.info(
        "video_text[%s]: kept %d entries, dropped %d (%s)",
        fmt, kept, dropped, detail,
    )
    if kept == 0:
        logging.warning(
            "video_text[%s]: 0 usable entries after flattening (%s). Check "
            "caption_fields, split, and that the metadata matches the '%s' "
            "schema.",
            fmt, detail, fmt,
        )
    elif dropped >= kept:
        logging.warning(
            "video_text[%s]: dropped %d rows (>= %d kept) while flattening "
            "(%s) -- verify caption_fields/split are correct.",
            fmt, dropped, kept, detail,
        )


def _flatten_vadr1_chunks(
    data,
    data_root=None,
    split=None,
    caption_fields=None,
    anomaly_only=False,
    path_prefix_mapping=None,
):
    """Flatten Vad-R1 video records with nested chunks."""
    fields = _normalize_caption_fields(caption_fields or ["anomaly_type"])
    entries = []
    n_split_filtered = 0
    n_skipped_anomaly_only = 0
    n_skipped_no_caption = 0
    for record in data:
        if split and record.get("split") != split:
            n_split_filtered += 1
            continue
        video_path = _resolve_video_path(
            record.get("video_path"),
            data_root=data_root,
            path_prefix_mapping=path_prefix_mapping,
        )
        anomaly_type = str(record.get("anomaly_type") or "Unknown")
        for chunk in record.get("chunks", []):
            is_anomaly = bool(chunk.get("is_anomaly"))
            if anomaly_only and not is_anomaly:
                n_skipped_anomaly_only += 1
                continue
            caption_groups = _caption_candidate_groups(
                record, chunk, fields, is_anomaly=is_anomaly
            )
            captions = [cap for group in caption_groups for cap in group]
            if not captions:
                n_skipped_no_caption += 1
                continue
            category = anomaly_type if is_anomaly else "Normal"
            chunk_index = chunk.get("chunk_index", 0)
            sample_id = (
                f"{record.get('dataset', '')}/"
                f"{record.get('video_id', '')}#{chunk_index}"
            )
            entries.append({
                "sample_id": sample_id,
                "split": record.get("split"),
                "dataset": record.get("dataset"),
                "video_id": record.get("video_id"),
                "video_path": video_path,
                "captions": captions,
                "caption": captions[0],
                "caption_groups": caption_groups,
                "category": category,
                "anomaly_type": anomaly_type,
                "is_anomaly": is_anomaly,
                "chunk_index": chunk_index,
                "start_time_sec": _coerce_number(
                    chunk.get("start_time_sec", 0.0), float,
                    field="start_time_sec", context=sample_id),
                "end_time_sec": _coerce_number(
                    chunk.get("end_time_sec", 0.0), float,
                    field="end_time_sec", context=sample_id),
                "start_frame": _coerce_number(
                    chunk.get("start_frame", 0), int,
                    field="start_frame", context=sample_id),
                "end_frame": _coerce_number(
                    chunk.get("end_frame", 0), int,
                    field="end_frame", context=sample_id),
                "total_frames": int(record.get("total_frames", 0) or 0),
                "metadata": {**record, "chunk": chunk},
            })
    _log_flatten_summary(
        "vadr1_chunks", kept=len(entries),
        dropped=n_split_filtered + n_skipped_anomaly_only + n_skipped_no_caption,
        detail=(
            f"caption_fields={fields}, split_filtered_records={n_split_filtered}, "
            f"skipped_non_anomaly_chunks={n_skipped_anomaly_only}, "
            f"skipped_no_caption_chunks={n_skipped_no_caption}"
        ),
    )
    return entries


def _flatten_msrvtt(data, data_root=None, split=None, path_prefix_mapping=None):
    """Normalize MSR-VTT style metadata with videos/sentences sections."""
    video_meta = {}
    n_split_filtered = 0
    n_missing_video_id = 0
    for row in data.get("videos", []):
        if split and row.get("split") != split:
            n_split_filtered += 1
            continue
        video_id = row.get("video_id") or row.get("id") or row.get("video")
        if video_id is None:
            n_missing_video_id += 1
            continue
        video_path = _pick_first(row, ["video_path", "path", "video"])
        if video_path is None:
            video_path = f"{video_id}.mp4"
        video_meta[str(video_id)] = {
            "video_id": str(video_id),
            "video_path": _resolve_video_path(
                video_path,
                data_root=data_root,
                path_prefix_mapping=path_prefix_mapping,
            ),
            "metadata": row,
            "captions": [],
        }

    for row in data.get("sentences", []):
        video_id = row.get("video_id")
        if video_id in video_meta:
            caption = _pick_first(row, ["caption", "text", "sentence"])
            if caption:
                video_meta[video_id]["captions"].append(str(caption))

    entries = []
    n_no_caption = 0
    for index, item in enumerate(video_meta.values()):
        captions = item["captions"]
        if not captions:
            n_no_caption += 1
            continue
        entries.append({
            "sample_id": item["video_id"] or str(index),
            "video_id": item["video_id"],
            "video_path": item["video_path"],
            "captions": captions,
            "caption": captions[0],
            "category": captions[0],
            "start_time_sec": None,
            "end_time_sec": None,
            "start_frame": None,
            "end_frame": None,
            "metadata": item["metadata"],
        })
    _log_flatten_summary(
        "msrvtt", kept=len(entries),
        dropped=n_split_filtered + n_missing_video_id + n_no_caption,
        detail=(
            f"split_filtered_videos={n_split_filtered}, "
            f"missing_video_id={n_missing_video_id}, "
            f"videos_without_captions={n_no_caption}"
        ),
    )
    return entries


def _flatten_flat_rows(
    data,
    data_root=None,
    split=None,
    path_prefix_mapping=None,
):
    """Normalize flat JSON/JSONL rows used by traditional retrieval sets."""
    entries = []
    n_split_filtered = 0
    n_missing_fields = 0
    for index, row in enumerate(data):
        if split and row.get("split") != split:
            n_split_filtered += 1
            continue
        video_path = _pick_first(row, ["video_path", "video", "path", "file"])
        caption = _pick_first(row, ["caption", "text", "query", "sentence"])
        captions = row.get("captions")
        if captions is None:
            captions = [caption] if caption else []
        elif isinstance(captions, str):
            captions = [captions]
        else:
            captions = [str(item) for item in captions if item]
        if not video_path or not captions:
            n_missing_fields += 1
            continue
        video_id = row.get("video_id") or Path(str(video_path)).stem
        category = row.get("category") or row.get("label") or captions[0]
        entries.append({
            "sample_id": str(row.get("sample_id") or f"{video_id}#{index}"),
            "split": row.get("split"),
            "video_id": video_id,
            "video_path": _resolve_video_path(
                video_path,
                data_root=data_root,
                path_prefix_mapping=path_prefix_mapping,
            ),
            "captions": captions,
            "caption": captions[0],
            "category": str(category),
            "start_time_sec": row.get("start_time_sec", row.get("start_sec")),
            "end_time_sec": row.get("end_time_sec", row.get("end_sec")),
            "start_frame": row.get("start_frame"),
            "end_frame": row.get("end_frame"),
            "metadata": row,
        })
    _log_flatten_summary(
        "flat_json", kept=len(entries),
        dropped=n_split_filtered + n_missing_fields,
        detail=(
            f"split_filtered_rows={n_split_filtered}, "
            f"rows_missing_video_or_caption={n_missing_fields}"
        ),
    )
    return entries


def _resolve_format(data, metadata, metadata_format):
    """Resolve the concrete metadata format (handling 'auto').

    For an explicit ``metadata_format`` the value is returned verbatim. For
    ``auto`` the schema is sniffed positively; if nothing matches a known
    schema a descriptive ``ValueError`` is raised instead of silently treating
    the file as ``flat_json`` (which previously surfaced only as a misleading
    "No valid video-text samples found").
    """
    fmt = metadata_format or "auto"
    if fmt != "auto":
        return fmt
    if (
        isinstance(data, list) and data and
        isinstance(data[0], dict) and "chunks" in data[0]
    ):
        return "vadr1_chunks"
    if isinstance(data, dict) and {"videos", "sentences"} <= set(data):
        return "msrvtt"
    if Path(str(metadata)).suffix.lower() == ".jsonl":
        return "jsonl"
    if isinstance(data, list):
        # A list of flat rows is the legitimate flat_json/JSON-array case.
        return "flat_json"
    seen = (
        f"dict with top-level keys {sorted(data)[:10]}"
        if isinstance(data, dict) else type(data).__name__
    )
    raise ValueError(
        f"Could not auto-detect the video-text metadata format of {metadata!r} "
        f"(saw {seen}). Supported: 'vadr1_chunks' (list of records each with a "
        f"'chunks' list), 'msrvtt' (dict with 'videos'+'sentences'), "
        f"'flat_json'/'jsonl' (list of flat rows). Set video_text.format "
        f"explicitly to override."
    )


def _build_entries_for_format(
    fmt,
    data,
    data_root=None,
    split=None,
    caption_fields=None,
    anomaly_only=False,
    path_prefix_mapping=None,
):
    """Flatten raw metadata into normalized entries for a resolved format."""
    if fmt == "vadr1_chunks":
        return _flatten_vadr1_chunks(
            data,
            data_root=data_root,
            split=split,
            caption_fields=caption_fields,
            anomaly_only=anomaly_only,
            path_prefix_mapping=path_prefix_mapping,
        )
    if fmt == "msrvtt":
        return _flatten_msrvtt(
            data,
            data_root=data_root,
            split=split,
            path_prefix_mapping=path_prefix_mapping,
        )
    if fmt in {"flat_json", "jsonl"}:
        return _flatten_flat_rows(
            data,
            data_root=data_root,
            split=split,
            path_prefix_mapping=path_prefix_mapping,
        )
    raise ValueError(f"Unsupported video-text metadata format: {fmt}")


def load_video_text_entries(
    metadata,
    metadata_format="auto",
    data_root=None,
    split=None,
    caption_fields=None,
    anomaly_only=False,
    path_prefix_mapping=None,
    idx_mode="sample_id",
    idx_field=None,
):
    """Load and normalize video-text metadata entries."""
    data = _read_metadata(metadata)
    fmt = _resolve_format(data, metadata, metadata_format)
    entries = _build_entries_for_format(
        fmt,
        data,
        data_root=data_root,
        split=split,
        caption_fields=caption_fields,
        anomaly_only=anomaly_only,
        path_prefix_mapping=path_prefix_mapping,
    )
    return _assign_integer_idx(entries, idx_mode, idx_field)


def load_eval_queries(path):
    """Load an explicit-relevance retrieval-eval query file.

    Standard shape (e.g. a frozen ``domain_test.json``): an object with a
    top-level ``queries`` list, or a bare list of query records. Each record
    needs ``query``, ``chunk_id`` (the home clip id, matching a gallery
    ``sample_id``), ``slice`` and ``relevant_clip_ids``. ``near_universal`` is
    optional (default False) and lets the evaluator drop near-gallery-wide
    queries from the headline aggregation.

    Relevance is taken verbatim from ``relevant_clip_ids`` (the gallery is the
    shared corpus the queries are scored against); it is NOT derived from any
    idx grouping. Returns the normalized list of query records.
    """
    data = _read_metadata(path)
    queries = data.get("queries") if isinstance(data, dict) else data
    if not queries:
        raise ValueError(f"No 'queries' found in eval query file: {path}")
    out = []
    for i, q in enumerate(queries):
        try:
            cid = q["chunk_id"]
            query_text = q["query"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Malformed query record #{i} in eval query file {path}: "
                f"missing required key {exc}. Each record needs 'query' and "
                f"'chunk_id'."
            ) from exc
        out.append({
            "query": query_text,
            "chunk_id": cid,
            "slice": q.get("slice") or "all",
            "relevant_clip_ids": list(q.get("relevant_clip_ids") or [cid]),
            "near_universal": bool(q.get("near_universal", False)),
        })
    return out


def _linspace_indices(total_frames, num_frames):
    """Select a fixed number of frames from a clip."""
    if total_frames <= 0:
        raise ValueError("Cannot sample from an empty video clip")
    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)
    indices = np.full((num_frames,), total_frames - 1, dtype=int)
    indices[:total_frames] = np.arange(total_frames)
    return indices


def _load_with_decord(video_path, num_frames, start_time_sec, end_time_sec,
                      start_frame, end_frame):
    """Load frames with decord when available."""
    import decord

    decord.logging.set_level(decord.logging.FATAL)
    reader = decord.VideoReader(str(video_path))
    actual_frames = len(reader)
    start = 0
    end = actual_frames
    if start_frame is not None and end_frame is not None:
        start = max(0, min(int(start_frame), actual_frames - 1))
        end = min(int(end_frame), actual_frames)
    elif start_time_sec is not None and end_time_sec is not None:
        fps = reader.get_avg_fps()
        start = max(0, min(int(round(float(start_time_sec) * fps)), actual_frames - 1))
        end = min(int(round(float(end_time_sec) * fps)), actual_frames)
    if end <= start:
        raise ValueError(
            f"Invalid video range for {video_path}: [{start}, {end})"
        )
    frame_indices = _linspace_indices(end - start, num_frames) + start
    frames = reader.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


def _clip_frame_range(total_frames, fps, start_time_sec, end_time_sec,
                      start_frame, end_frame):
    """Resolve temporal metadata to a frame range."""
    start = 0
    end = total_frames
    if start_frame is not None and end_frame is not None:
        start = int(start_frame)
        end = int(end_frame)
    elif start_time_sec is not None and end_time_sec is not None and fps > 0:
        start = int(round(float(start_time_sec) * fps))
        end = int(round(float(end_time_sec) * fps))
    start = max(0, min(start, total_frames - 1))
    end = max(start + 1, min(end, total_frames))
    return start, end


def _parse_frame_rate(value):
    """Parse ffprobe frame-rate strings like 30000/1001."""
    if not value or value == "0/0":
        return 0.0
    if "/" in str(value):
        numerator, denominator = value.split("/", 1)
        denominator = float(denominator)
        if denominator == 0.0:
            return 0.0
        return float(numerator) / denominator
    return float(value)


def _probe_video_with_ffmpeg(video_path):
    """Read basic video stream metadata with ffprobe."""
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {video_path}")
    stream = streams[0]
    fps = _parse_frame_rate(
        stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    )
    duration = float(stream.get("duration") or 0.0)
    total_frames = stream.get("nb_frames")
    total_frames = int(total_frames) if total_frames else 0
    # pylint: disable=chained-comparison  # three independent > 0 tests
    if total_frames <= 0 and fps > 0.0 and duration > 0.0:
        total_frames = int(round(fps * duration))
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if total_frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Could not probe video geometry for {video_path}")
    return total_frames, fps, width, height


def _load_with_ffmpeg(video_path, num_frames, start_time_sec, end_time_sec,
                      start_frame, end_frame):
    """Load frames with the ffmpeg CLI as a dependency-light fallback."""
    import subprocess

    total_frames, fps, width, height = _probe_video_with_ffmpeg(video_path)
    start, end = _clip_frame_range(
        total_frames,
        fps,
        start_time_sec,
        end_time_sec,
        start_frame,
        end_frame,
    )
    frame_indices = _linspace_indices(end - start, num_frames) + start
    unique_indices = []
    for index in frame_indices:
        index = int(index)
        if index not in unique_indices:
            unique_indices.append(index)
    selector = "+".join(f"eq(n\\,{index})" for index in unique_indices)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select={selector}",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frame_bytes = height * width * 3
    decoded_frames = len(result.stdout) // frame_bytes
    if decoded_frames <= 0:
        raise ValueError(f"No frames decoded from {video_path}")
    if len(result.stdout) % frame_bytes != 0:
        logging.warning(
            "Dropping partial decoded frame bytes from %s: got %d bytes",
            video_path,
            len(result.stdout),
        )
    if decoded_frames != len(unique_indices):
        logging.warning(
            "Decoded %d/%d requested frames from %s; padding with the last "
            "decoded frame.",
            decoded_frames,
            len(unique_indices),
            video_path,
        )
    usable_bytes = decoded_frames * frame_bytes
    array = np.frombuffer(result.stdout[:usable_bytes], dtype=np.uint8)
    array = array.reshape(decoded_frames, height, width, 3)
    frames = [Image.fromarray(frame).convert("RGB") for frame in array]
    frames_by_index = {
        frame_index: frames[min(pos, decoded_frames - 1)]
        for pos, frame_index in enumerate(unique_indices)
    }
    return [frames_by_index[int(index)] for index in frame_indices]


def _load_with_opencv(video_path, num_frames, start_time_sec, end_time_sec,
                      start_frame, end_frame):
    """Load frames with OpenCV when decord and ffmpeg are unavailable."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if total_frames <= 0:
            raise ValueError(f"No frames found in {video_path}")

        start, end = _clip_frame_range(
            total_frames,
            fps,
            start_time_sec,
            end_time_sec,
            start_frame,
            end_frame,
        )
        frame_indices = _linspace_indices(end - start, num_frames) + start
        frames = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(
                    f"Could not decode frame {frame_index} from {video_path}"
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))
        return frames
    finally:
        cap.release()


def load_video_frames(video_path, num_frames, start_time_sec=None,
                      end_time_sec=None, start_frame=None, end_frame=None):
    """Load a fixed number of PIL RGB frames from a video clip."""
    decord_error = None
    try:
        return _load_with_decord(
            video_path,
            num_frames,
            start_time_sec,
            end_time_sec,
            start_frame,
            end_frame,
        )
    except Exception as exc:
        decord_error = exc
        logging.debug("decord decode failed for %s: %s", video_path, exc)

    ffmpeg_error = None
    try:
        return _load_with_ffmpeg(
            video_path,
            num_frames,
            start_time_sec,
            end_time_sec,
            start_frame,
            end_frame,
        )
    except Exception as exc:
        ffmpeg_error = exc
        logging.debug("ffmpeg decode failed for %s: %s", video_path, exc)

    try:
        return _load_with_opencv(
            video_path,
            num_frames,
            start_time_sec,
            end_time_sec,
            start_frame,
            end_frame,
        )
    except ImportError as exc:
        raise ImportError(
            "Video decoding requires decord, ffmpeg/ffprobe, or OpenCV in "
            "the TAO environment."
        ) from decord_error or ffmpeg_error or exc
    except Exception as exc:
        raise RuntimeError(
            "All video decoding backends failed. Check that decord is "
            "installed and can read the video: the container ffmpeg is "
            "codec-disabled and its fallback path cannot pipe raw frames."
        ) from decord_error or ffmpeg_error or exc


def _stack_processed_frames(processed_frames):
    """Stack transformed frame outputs into a video tensor."""
    if isinstance(processed_frames[0], dict):
        return {
            key: torch.stack([frame[key] for frame in processed_frames], dim=0)
            for key in processed_frames[0]
        }
    return torch.stack(processed_frames, dim=0)


def _pil_to_tensor(frame):
    """Convert a PIL frame to CHW float tensor."""
    array = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class BaseVideoTextDataset(Dataset):
    """Base video-text dataset with InternVideo2 idx support.

    Holds already-normalized ``entries`` and provides frame loading, caption
    selection and ``(video, text, idx)`` access. ``task_type`` records whether
    the dataset is used for ``classification`` (category-grouped /
    supervised-contrastive) or ``retrieval`` (instance / N-to-N) work. Per-format
    subclasses override :meth:`_category_for` to define the category label.
    """

    #: Metadata format this subclass handles (set by subclasses).
    DATASET_FORMAT = None

    #: How many subsequent samples to try when one fails to decode, before
    #: giving up. Keeps a single corrupt video from crashing the whole epoch.
    MAX_LOAD_RETRIES = 10

    def __init__(
        self,
        entries,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        num_frames: int = 8,
        caption_random: bool = False,
        frame_loader: Optional[Callable] = None,
        task_type: str = "retrieval",
        allow_substitution: bool = True,
    ):
        """Initialize the dataset.

        ``allow_substitution`` controls the decode-failure behavior of
        :meth:`__getitem__`. When True (training) a corrupt sample falls
        forward to a neighbor so one bad video cannot crash the epoch. It must
        be False for evaluation/retrieval, where silently substituting a
        different clip would misalign gallery/query positions and corrupt the
        metrics.
        """
        self.entries = list(entries)
        self.transform = transform
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.caption_random = caption_random
        self.frame_loader = frame_loader or load_video_frames
        self.task_type = task_type
        self.allow_substitution = allow_substitution
        if not self.entries:
            raise ValueError("No valid video-text samples found")
        # Normalize the category label through the subclass hook. Default and
        # all current subclasses return entry["category"], so this is a no-op
        # for existing data (behavior-preserving) while giving subclasses a
        # single place to define the classification label.
        for entry in self.entries:
            entry["category"] = self._category_for(entry)

    def _category_for(self, entry):
        """Return the category label for an entry (override per format)."""
        return entry.get("category")

    def __len__(self):
        """Return number of samples."""
        return len(self.entries)

    def __getitem__(self, idx):
        """Load a sample, falling forward to the next on a decode failure.

        In training (``allow_substitution=True``) a corrupt or unreadable video
        would otherwise raise and kill the epoch, so we log the failure and
        advance to the next entry, up to ``MAX_LOAD_RETRIES`` times, before
        giving up. In evaluation (``allow_substitution=False``) substitution is
        disabled: returning a neighbor would misalign the gallery/query indices
        used for retrieval, so a bad sample fails loudly instead.
        """
        if not self.allow_substitution:
            return self._load_item(idx)
        num = len(self.entries)
        for offset in range(self.MAX_LOAD_RETRIES + 1):
            cur = (idx + offset) % num
            try:
                return self._load_item(cur)
            except Exception as exc:  # noqa: BLE001 - skip bad samples, don't crash
                logging.warning(
                    "Failed to load sample %d (%s): %s",
                    cur, self.entries[cur].get("video_path"), exc,
                )
        raise RuntimeError(
            f"Could not load a valid sample within {self.MAX_LOAD_RETRIES + 1} "
            f"attempts starting at index {idx}."
        )

    def _load_item(self, idx):
        """Load a video clip, caption, and InternVideo2 positive id."""
        entry = self.entries[idx]
        frames = self.frame_loader(
            entry["video_path"],
            self.num_frames,
            entry.get("start_time_sec"),
            entry.get("end_time_sec"),
            entry.get("start_frame"),
            entry.get("end_frame"),
        )
        if len(frames) != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames, got {len(frames)}"
            )

        if self.transform:
            processed = [self.transform(frame) for frame in frames]
        else:
            processed = [_pil_to_tensor(frame) for frame in frames]
        video = _stack_processed_frames(processed)

        captions = entry.get("captions") or [entry["caption"]]
        caption = (
            random.choice(captions)
            if self.caption_random and len(captions) > 1
            else captions[0]
        )
        text = self.tokenizer(caption)[0] if self.tokenizer else caption
        # 4th element: unique dataset row position, used to drop the
        # DistributedSampler padding repeats during multi-GPU eval.
        return video, text, int(entry["idx"]), int(idx)


class VadR1Dataset(BaseVideoTextDataset):
    """Vad-R1 chunked dataset.

    For ``task_type='classification'`` the category is the anomaly type
    (``anomaly_type`` for anomaly chunks, ``"Normal"`` otherwise). That mapping
    is applied during flattening (:func:`_flatten_vadr1_chunks`), so the hook
    just surfaces the precomputed ``category`` (falling back to ``anomaly_type``).
    """

    DATASET_FORMAT = "vadr1_chunks"

    def _category_for(self, entry):
        """Vad-R1 category label: anomaly_type (or 'Normal'), set at flatten time."""
        return entry.get("category") or entry.get("anomaly_type")


class MsrVttDataset(BaseVideoTextDataset):
    """MSR-VTT style video-text dataset (retrieval)."""

    DATASET_FORMAT = "msrvtt"


class FlatRowsDataset(BaseVideoTextDataset):
    """Flat JSON / JSONL video-text dataset."""

    DATASET_FORMAT = "flat_json"


# Backward-compatible alias: external callers (e.g. inference.py) construct a
# generic dataset directly from pre-built entries.
VideoTextDataset = BaseVideoTextDataset

_FORMAT_TO_DATASET = {
    "vadr1_chunks": VadR1Dataset,
    "msrvtt": MsrVttDataset,
    "flat_json": FlatRowsDataset,
    "jsonl": FlatRowsDataset,
}


def build_video_text_dataset(
    cfg,
    transform: Optional[Callable] = None,
    tokenizer: Optional[Callable] = None,
    num_frames: int = 8,
    frame_loader: Optional[Callable] = None,
    mode: str = "train",
):
    """Build the per-format dataset subclass selected by ``cfg.format``.

    Honors ``task_type``: for ``classification`` with the default
    ``idx_mode='sample_id'`` the grouping defaults to ``category`` (an explicit
    ``idx_mode`` always wins). ``mode`` controls decode-failure handling:
    neighbor substitution is enabled only for ``train`` (see
    :class:`BaseVideoTextDataset`).
    """
    metadata = _cfg_get(cfg, "metadata")
    if not metadata:
        raise ValueError("video_text.metadata is required")
    metadata_format = _cfg_get(cfg, "format", "auto")
    task_type = _cfg_get(cfg, "task_type", "retrieval")
    idx_mode = _cfg_get(cfg, "idx_mode", "sample_id")
    idx_field = _cfg_get(cfg, "idx_field")
    if task_type == "classification" and idx_mode == "sample_id":
        idx_mode = "category"

    data = _read_metadata(metadata)
    fmt = _resolve_format(data, metadata, metadata_format)
    entries = _build_entries_for_format(
        fmt,
        data,
        data_root=_cfg_get(cfg, "data_root"),
        split=_cfg_get(cfg, "split"),
        # No hard ['caption'] default: only vadr1_chunks consumes caption_fields,
        # and it supplies its own ['anomaly_type'] default. Injecting ['caption']
        # here used to silently shadow that and drop every anomaly chunk.
        caption_fields=_cfg_get(cfg, "caption_fields"),
        anomaly_only=_cfg_get(cfg, "anomaly_only", False),
        path_prefix_mapping=_as_dict(_cfg_get(cfg, "path_prefix_mapping", {})),
    )
    entries = _assign_integer_idx(entries, idx_mode, idx_field)
    # Train-only caption expansion ('all' = one entry per caption,
    # 'one_per_field' = one entry per field). Exploded copies keep the chunk's
    # idx, so the InternVideo2 VTC loss treats them as positives.
    caption_mode = _cfg_get(cfg, "caption_mode", "first")
    if mode == "train":
        entries = _apply_caption_mode(entries, caption_mode)
    # Sample a caption per epoch only for the random-within-pool/field modes,
    # and only in training (eval/retrieval must be deterministic = captions[0]).
    caption_random = mode == "train" and caption_mode in {
        "random", "one_per_field",
    }
    dataset_cls = _FORMAT_TO_DATASET.get(fmt, FlatRowsDataset)
    return dataset_cls(
        entries=entries,
        transform=transform,
        tokenizer=tokenizer,
        num_frames=num_frames,
        caption_random=caption_random,
        frame_loader=frame_loader,
        task_type=task_type,
        allow_substitution=(mode == "train"),
    )


def _entries_from_cfg(cfg):
    """Build normalized entries from a video_text config."""
    metadata = _cfg_get(cfg, "metadata")
    if not metadata:
        raise ValueError("video_text.metadata is required")
    entries = load_video_text_entries(
        metadata=metadata,
        metadata_format=_cfg_get(cfg, "format", "auto"),
        data_root=_cfg_get(cfg, "data_root"),
        split=_cfg_get(cfg, "split"),
        caption_fields=_cfg_get(cfg, "caption_fields"),
        anomaly_only=_cfg_get(cfg, "anomaly_only", False),
        path_prefix_mapping=_as_dict(_cfg_get(cfg, "path_prefix_mapping", {})),
        idx_mode=_cfg_get(cfg, "idx_mode", "sample_id"),
        idx_field=_cfg_get(cfg, "idx_field"),
    )
    # Mirror build_video_text_dataset's train-time expansion so schedule sizing
    # (train.py limit_train_batches) counts exploded entries, not base chunks.
    return _apply_caption_mode(entries, _cfg_get(cfg, "caption_mode", "first"))


def get_video_text_dataloader(
    cfg,
    batch_size: int = 32,
    transform: Callable = None,
    tokenizer: Callable = None,
    num_workers: int = 0,
    seed: int = 42,
    shuffle=True,
    pin_memory=True,
    is_distributed=None,
    mode='train',
    frame_loader: Optional[Callable] = None,
):
    """Create a DataLoader for normalized video-text metadata."""
    # Keep randomness local to this loader. Reseeding the global RNGs here used
    # to reset whatever seed_everything() established, once per loader, in the
    # middle of datamodule setup.
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = build_video_text_dataset(
        cfg,
        transform=transform,
        tokenizer=tokenizer,
        num_frames=_cfg_get(cfg, "num_frames", 8),
        frame_loader=frame_loader,
        mode=mode,
    )
    dataloader_kwargs = {}
    batch_sampler = None
    if mode == 'train':
        if is_distributed:
            batch_sampler = distributed.DistributedSampler(
                dataset, shuffle=True, seed=seed)
        else:
            batch_sampler = RandomSampler(dataset, generator=generator)
    if batch_sampler:
        dataloader_kwargs['batch_sampler'] = BatchSampler(
            batch_sampler, batch_size, drop_last=True)
    elif is_distributed:
        # Shard validation/eval across ranks (each rank embeds ~1/world_size of
        # the clips); the model's on_validation_epoch_end all-gathers the per-rank
        # embeddings back into the full gallery. Avoids every rank redundantly
        # decoding the whole eval set.
        dataloader_kwargs['sampler'] = distributed.DistributedSampler(
            dataset, shuffle=False, drop_last=False)
        dataloader_kwargs['batch_size'] = batch_size
    else:
        dataloader_kwargs['batch_size'] = batch_size
        dataloader_kwargs['shuffle'] = shuffle

    return DataLoader(
        dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        **dataloader_kwargs,
    )
