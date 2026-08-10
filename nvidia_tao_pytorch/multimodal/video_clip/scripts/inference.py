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

"""Inference for the video_clip model.

Two modes (``inference.mode``):
- ``embeddings``: extract embeddings for whatever sources are present -
  the ``dataset.inference`` corpus -> ``video_embeddings.h5`` (multi-GPU/DDP,
  cache-aware), inline text queries -> ``text_embeddings.h5``, and inline video
  queries -> ``query_video_embeddings.h5``.
- ``retrieval``: embed the queries + the corpus and write the top-``top_k``
  corpus matches per query to ``retrieval_results.json`` (cosine or kNN).

The corpus is owned by ``dataset.inference`` (no copying into ``dataset.val``).
Ad-hoc queries come from ``inference.query`` (inline texts/videos + optional
``text_file``).
"""

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import Trainer
from tqdm import tqdm

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import (
    initialize_inference_experiment,
)
from nvidia_tao_pytorch.core.tlt_logging import logging, obfuscate_logs

from nvidia_tao_pytorch.config.video_clip.default_config import (
    VideoCLIPExperimentConfig as ExperimentConfig,
)
from nvidia_tao_pytorch.multimodal.video_clip.model.pl_video_clip_model import (
    VideoCLIPPlModel,
)
from nvidia_tao_pytorch.multimodal.video_clip.dataloader.pl_video_clip_data_module import (
    VideoCLIPDataModule,
)
from nvidia_tao_pytorch.multimodal.video_clip.dataloader.video_text_loader import (
    get_video_text_dataloader,
    load_video_frames,
    _stack_processed_frames,
)
from nvidia_tao_pytorch.multimodal.video_clip.utils.utils import (
    load_model_from_checkpoint,
    to_lightning_precision,
)
from nvidia_tao_pytorch.multimodal.video_clip.utils.embedding_io import (
    build_provenance,
    provenance_compatible,
    read_embeddings_h5,
    resolve_embeddings_path,
    text_to_video_search,
    write_embeddings_h5,
    write_similarity_stats,
)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
def extract_visual_features(model, batch) -> torch.Tensor:
    """Extract visual (video frame) features. batch: (B, T, C, H, W) or dict."""
    output = model.model(image=batch)
    if isinstance(output, dict):
        return output["image_features"]
    return output[0]


def extract_text_features(
    model, texts: List[str], device: torch.device,
) -> torch.Tensor:
    """Extract text features for a list of strings."""
    tokenized = model.tokenizer(texts)[0]
    if isinstance(tokenized, torch.Tensor) or hasattr(tokenized, "to"):
        tokenized = tokenized.to(device)
    elif isinstance(tokenized, dict):
        tokenized = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in tokenized.items()
        }
    text_output = model.model(text=tokenized)
    if isinstance(text_output, dict):
        return text_output["text_features"]
    return text_output[1]


def load_text_file(text_file: str) -> List[str]:
    """Load text prompts (one per line, non-empty)."""
    with open(text_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------
def _embeddings_cache_ok(path: str, provenance: dict, overwrite: bool) -> bool:
    """True if a compatible cached embedding file exists (else generate).

    Raises if a file exists but was produced by a different checkpoint/model.
    """
    if overwrite or not os.path.exists(path):
        return False
    try:
        _, _, attrs = read_embeddings_h5(path)
    except Exception as exc:  # noqa: BLE001 - corrupt cache -> regenerate
        logging.warning("Cached embeddings at %s unreadable (%s); regenerating.", path, exc)
        return False
    ok, mismatched = provenance_compatible(attrs, provenance)
    if not ok:
        raise ValueError(
            f"Cached embeddings at {path} are incompatible with this run "
            f"('{mismatched}' mismatch). Set inference.overwrite_embeddings=true "
            f"to regenerate, or point the *_embeddings_file at a new path."
        )
    return True


# ---------------------------------------------------------------------------
# Corpus extraction (multi-GPU / DDP)
# ---------------------------------------------------------------------------
def _extract_corpus_embeddings_ddp(
    experiment_config, model, trainer_kwargs, video_path: str, provenance: dict,
) -> Trainer:
    """Extract corpus video embeddings via Lightning (multi-GPU/DDP) and save.

    The corpus loader is built directly from ``dataset.inference`` and attached
    to the DataModule (no dataset.val sync). Lightning shards it with a
    DistributedSampler under DDP; the model gathers across ranks, de-dups the
    padding by sample_id, and writes once on global rank 0.
    """
    inf = experiment_config.inference
    corpus_vt = experiment_config.dataset.inference.video_text
    loader = get_video_text_dataloader(
        cfg=corpus_vt,
        batch_size=inf.batch_size,
        num_workers=inf.num_workers,
        transform=model.preprocess_val,
        tokenizer=None,
        shuffle=False,
        pin_memory=False,
        is_distributed=False,  # Lightning injects the DistributedSampler
        mode="val",
    )
    dm = VideoCLIPDataModule(
        experiment_config.dataset,
        model.tokenizer,
        resume_step=0,
        preprocess=(model.preprocess_train, model.preprocess_val),
        world_size=1,
    )
    dm._setup_done = True  # bypass dataset.val setup; use our corpus loader
    dm.val_dataset = loader

    model._inference_embed_cfg = {"path": video_path, "provenance": provenance}
    tkw = dict(trainer_kwargs)
    precision = getattr(getattr(experiment_config, "train", None), "precision", "fp32")
    tkw["precision"] = to_lightning_precision(precision)
    trainer = Trainer(**tkw)
    logging.info(
        "Extracting corpus video embeddings (devices=%s, precision=%s) -> %s",
        tkw.get("devices"), tkw["precision"], video_path,
    )
    trainer.test(model, datamodule=dm)
    return trainer


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------
def _run_text_queries(
    model, query_cfg, experiment_config, text_path: str, overwrite: bool,
    device: torch.device,
) -> Optional[Tuple[List[str], np.ndarray]]:
    """Encode inline text queries (+ optional text_file), cache-aware."""
    texts = list(getattr(query_cfg, "input_texts", []) or [])
    text_file = getattr(query_cfg, "text_file", None)
    if text_file:
        texts += load_text_file(text_file)
    texts = _dedup_preserve_order([t for t in texts if t])
    if not texts:
        return None

    provenance = build_provenance(experiment_config, normalized=False)
    cached = {}
    if os.path.exists(text_path) and not overwrite:
        try:
            ids, emb, attrs = read_embeddings_h5(text_path)
            ok, mismatched = provenance_compatible(attrs, provenance)
            if ok:
                cached = {t: emb[i] for i, t in enumerate(ids)}
                logging.info("Found %d reusable cached text embedding(s).", len(cached))
            else:
                logging.warning("Cached text embeddings incompatible ('%s'); re-encoding.", mismatched)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not read cached text embeddings (%s).", exc)

    missing = [t for t in texts if t not in cached]
    if missing:
        bs = max(1, experiment_config.inference.batch_size)
        new = []
        with torch.no_grad():
            for i in tqdm(range(0, len(missing), bs),
                          total=(len(missing) + bs - 1) // bs,
                          desc="Text query embeddings"):
                feats = extract_text_features(model, missing[i:i + bs], device)
                new.append(feats.detach().cpu().numpy())
        new = np.concatenate(new, axis=0)
        for t, v in zip(missing, new):
            cached[t] = v
        logging.info("Encoded %d new text query/queries.", len(missing))
    else:
        logging.info("All %d text query/queries served from cache.", len(texts))

    emb = np.stack([cached[t] for t in texts], axis=0).astype(np.float32)
    write_embeddings_h5(text_path, texts, emb, "text", provenance=provenance)
    logging.info("Saved %d text-query embeddings to %s", len(texts), text_path)
    return texts, emb


def _run_video_queries(
    model, query_cfg, device: torch.device, results_dir: str, experiment_config,
) -> Optional[Tuple[List[str], np.ndarray]]:
    """Embed inline ad-hoc video-file queries -> query_video_embeddings.h5."""
    paths = list(getattr(query_cfg, "input_videos", []) or [])
    if not paths:
        return None
    num_frames = int(getattr(query_cfg, "num_frames", 8))
    embs, ids = [], []
    with torch.no_grad():
        for path in tqdm(paths, desc="Video query embeddings"):
            frames = load_video_frames(path, num_frames, None, None, None, None)
            processed = [model.preprocess_val(f) for f in frames]
            video = _stack_processed_frames(processed).unsqueeze(0).to(device)
            feats = extract_visual_features(model, video)
            embs.append(feats.detach().cpu().numpy())
            ids.append(path)
    emb = np.concatenate(embs, axis=0).astype(np.float32)
    out = os.path.join(results_dir, "query_video_embeddings.h5")
    provenance = build_provenance(experiment_config, normalized=False)
    write_embeddings_h5(out, ids, emb, "video", provenance=provenance)
    logging.info("Saved %d video-query embeddings to %s", len(ids), out)
    return ids, emb


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _run_retrieval(
    experiment_config, results_dir: str, corpus_video_path: str,
    text_q, video_q,
) -> None:
    """Rank the corpus against the queries; write retrieval_results.json."""
    inf = experiment_config.inference
    search = getattr(inf, "search", None)
    metric = str(getattr(search, "search_metric", "cosine"))
    normalize = bool(getattr(search, "normalize", True))
    top_k = int(getattr(search, "top_k", 10))

    corpus_ids, corpus_emb, _ = read_embeddings_h5(corpus_video_path)
    labels, types, rows = [], [], []
    if text_q is not None:
        t_ids, t_emb = text_q
        for i, t in enumerate(t_ids):
            labels.append(t)
            types.append("text")
            rows.append(t_emb[i])
    if video_q is not None:
        v_ids, v_emb = video_q
        for i, v in enumerate(v_ids):
            labels.append(v)
            types.append("video")
            rows.append(v_emb[i])
    if not rows:
        logging.warning("No query embeddings available for retrieval.")
        return

    query_emb = np.stack(rows, axis=0).astype(np.float32)
    results, scores = text_to_video_search(
        corpus_ids, corpus_emb, labels, query_emb,
        metric=metric, normalize=normalize, top_k=top_k,
    )
    for record, qtype in zip(results, types):
        record["query_type"] = qtype

    payload = {
        "metric": metric, "normalize": normalize, "top_k": top_k,
        "num_corpus": len(corpus_ids), "num_queries": len(labels),
        "queries": results,
    }
    out = os.path.join(results_dir, "retrieval_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_similarity_stats(
        os.path.join(results_dir, "similarity_stats.json"),
        scores, corpus_emb, query_emb, metric,
    )
    logging.info(
        "Retrieval (%s, top_k=%d): %d queries x %d corpus -> %s",
        metric, top_k, len(labels), len(corpus_ids), out,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_experiment(experiment_config, key):
    """Run inference (mode=embeddings or mode=retrieval).

    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration.
    key : str
        Encryption key (unused, kept for TAO API compatibility).
    """
    del key  # Unused but required by TAO API

    model_path, trainer_kwargs = initialize_inference_experiment(
        experiment_config, experiment_config.encryption_key
    )

    inf = experiment_config.inference
    mode = str(getattr(inf, "mode", "embeddings"))
    results_dir = experiment_config.results_dir or inf.results_dir
    overwrite = bool(getattr(inf, "overwrite_embeddings", False))

    corpus_data = getattr(experiment_config.dataset, "inference", None)
    corpus_vt = getattr(corpus_data, "video_text", None) if corpus_data else None
    has_corpus = corpus_vt is not None and bool(getattr(corpus_vt, "metadata", None))

    query_cfg = getattr(inf, "query", None)
    q_texts = list(getattr(query_cfg, "input_texts", []) or []) if query_cfg else []
    q_text_file = getattr(query_cfg, "text_file", None) if query_cfg else None
    q_videos = list(getattr(query_cfg, "input_videos", []) or []) if query_cfg else []
    has_text_query = bool(q_texts) or bool(q_text_file)
    has_video_query = bool(q_videos)

    if mode == "retrieval":
        if not has_corpus:
            raise ValueError(
                "mode=retrieval requires a corpus: "
                "dataset.inference.video_text.metadata."
            )
        if not (has_text_query or has_video_query):
            raise ValueError(
                "mode=retrieval requires a query: inference.query.input_texts, "
                "input_videos, or text_file."
            )
    elif mode == "embeddings":
        if not (has_corpus or has_text_query or has_video_query):
            raise ValueError(
                "mode=embeddings requires at least one of: dataset.inference "
                "corpus, inference.query.input_texts/text_file, "
                "inference.query.input_videos."
            )
    else:
        raise ValueError(f"Unknown inference.mode: {mode}. Use 'embeddings' or 'retrieval'.")

    os.makedirs(results_dir, exist_ok=True)

    model = load_model_from_checkpoint(model_path, experiment_config, VideoCLIPPlModel)
    model.eval()

    # ---- Corpus video embeddings (multi-GPU, cache-aware) ----
    trainer = None
    video_path = None
    if has_corpus:
        video_path = resolve_embeddings_path(
            getattr(inf, "video_embeddings_file", None), results_dir, "video",
        )
        provenance = build_provenance(experiment_config, normalized=False)
        if _embeddings_cache_ok(video_path, provenance, overwrite):
            logging.info("Reusing cached corpus video embeddings: %s", video_path)
        else:
            trainer = _extract_corpus_embeddings_ddp(
                experiment_config, model, trainer_kwargs, video_path, provenance,
            )

    # Non-zero DDP ranks are done after the collective corpus extraction.
    if trainer is not None and not trainer.is_global_zero:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # ---- Query embeddings (rank 0) ----
    text_q = None
    if has_text_query:
        text_path = resolve_embeddings_path(
            getattr(inf, "text_embeddings_file", None), results_dir, "text",
        )
        text_q = _run_text_queries(
            model, query_cfg, experiment_config, text_path, overwrite, device,
        )
    video_q = None
    if has_video_query:
        video_q = _run_video_queries(
            model, query_cfg, device, results_dir, experiment_config,
        )

    # ---- Retrieval ----
    if mode == "retrieval":
        _run_retrieval(experiment_config, results_dir, video_path, text_q, video_q)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig
)
@monitor_status(name="VideoCLIP", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg, key=cfg.encryption_key)


if __name__ == "__main__":
    main()
