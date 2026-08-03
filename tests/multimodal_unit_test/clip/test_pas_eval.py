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

"""Unit tests for direct PAS evaluation helpers."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import nvidia_tao_pytorch.multimodal.clip.model.evaluation.pas as pas
from nvidia_tao_pytorch.config.clip.default_config import (
    CLIPEvaluateConfig,
    CLIPInferenceEvalConfig,
)
from nvidia_tao_pytorch.multimodal.clip.model.evaluation.pas import (
    PasPair,
    _extract_image_embeddings,
    _pair_image_key,
    _resolve_existing_image_items,
    _unique_image_candidates,
    aggregate_metric_rows,
    evaluate_text_to_image,
    evaluate_text_to_image_by_metadata,
    load_pas_pairs,
)
from nvidia_tao_pytorch.multimodal.clip.model.tokenizers import (
    CLIPCompatibleTokenizer,
    SigLIP2WrappedTokenizer,
)


def _write_attribute_vocab(path, width=2, not_visible=False):
    """Write a minimal ordered scalar vocabulary."""
    attributes = [f"attribute_{index}" for index in range(width)]
    value_to_id = {attribute: {} for attribute in attributes}
    if not_visible:
        attributes = ["top outer color", "top outer type"]
        value_to_id = {
            "top outer color": {
                "__missing__": 0,
                "black": 2,
                "not visible": 8,
            },
            "top outer type": {
                "__missing__": 0,
                "not visible": 5,
                "t shirt": 9,
            },
        }
    path.write_text(
        json.dumps(
            {
                "attributes": attributes,
                "value_to_id": value_to_id,
            }
        )
    )


def _write_accessory_vocab(path):
    """Write a minimal split-complete accessory vocabulary."""
    path.write_text(
        json.dumps(
            {
                "unknown_id": 0,
                "source_splits": ["train", "val", "test"],
                "id_to_value": ["__unknown__", "bag", "hat"],
                "value_to_id": {
                    "__unknown__": 0,
                    "bag": 1,
                    "hat": 2,
                },
                "vocab_sha256": "test-vocab",
            }
        )
    )


def _metric_row(
    dataset,
    query_type,
    num_queries,
    map_score,
    first_pos=1.0,
):
    return {
        "Dataset": dataset,
        "QueryType": query_type,
        "EasyAttribute": "",
        "num_queries": num_queries,
        "gallery_size": 10,
        "avg_gt_per_query": 1.0,
        "mAP": map_score,
        "Rank-1": map_score,
        "Rank-5": map_score,
        "Separability": map_score,
        "Match@5": map_score,
        "Zero@5": 1.0 - map_score,
        "First Pos": first_pos,
    }


@pytest.mark.multimodal_unit
def test_pas_ground_truth_mode_defaults_to_paired_caption():
    assert CLIPEvaluateConfig().pas_ground_truth_mode == "paired_caption"


@pytest.mark.multimodal_unit
def test_inference_config_excludes_pas_ground_truth_mode():
    assert not hasattr(
        CLIPInferenceEvalConfig(),
        "pas_ground_truth_mode",
    )


@pytest.mark.multimodal_unit
def test_query_metrics_assign_unique_ranks_to_tied_positives():
    metrics = pas._compute_query_metrics(
        np.array([0.8, 0.8, 0.2]),
        gt_indices=[0, 1],
    )

    assert metrics["ap"] == pytest.approx(1.0)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["first_pos"] == 1.0


@pytest.mark.multimodal_unit
def test_query_metrics_break_positive_negative_tie_by_gallery_order():
    metrics = pas._compute_query_metrics(
        np.array([0.8, 0.8, 0.2]),
        gt_indices=[1],
    )

    assert metrics["first_pos"] == 2.0
    assert metrics["rank1"] == 0.0
    assert metrics["ap"] == pytest.approx(0.5)
    assert metrics["auc"] == pytest.approx(0.5)


@pytest.mark.multimodal_unit
def test_metric_aggregate_macro_averages_datasets():
    rows = [
        _metric_row("A", "easy", 10, 0.0),
        _metric_row("B", "easy", 1, 1.0),
    ]

    aggregate = aggregate_metric_rows(rows, k=5, dataset_names=("A", "B"))

    assert aggregate[0]["Dataset"] == "AVG_2_DATASETS"
    assert aggregate[0]["num_queries"] == pytest.approx(5.5)
    assert aggregate[0]["mAP"] == pytest.approx(0.5)


@pytest.mark.multimodal_unit
def test_metric_weighted_aggregate_uses_query_count():
    rows = [
        _metric_row("A", "easy", 10, 0.0, first_pos=100.0),
        _metric_row("B", "easy", 1, 1.0, first_pos=1.0),
    ]

    aggregate = aggregate_metric_rows(
        rows,
        k=5,
        dataset_names=("A", "B"),
        weighted=True,
    )

    assert aggregate[0]["Dataset"] == "WAVG_2_DATASETS"
    assert aggregate[0]["num_queries"] == pytest.approx(11)
    assert aggregate[0]["mAP"] == pytest.approx(1 / 11)
    assert np.isnan(aggregate[0]["First Pos"])


@pytest.mark.multimodal_unit
def test_metadata_eval_filters_matches_by_required_accessories():
    pairs = [
        PasPair(
            dataset="D",
            query_type="hard",
            caption="person with bag",
            unique_name="img0.jpg",
            image_path="images/D/img0.jpg",
            prepared_image_path=Path("img0.jpg"),
            image_attr_values=(1, 2),
            text_attr_values=(1, 2),
            image_accessory_ids=(1,),
            text_accessory_ids=(1,),
        ),
        PasPair(
            dataset="D",
            query_type="hard",
            caption="person with hat",
            unique_name="img1.jpg",
            image_path="images/D/img1.jpg",
            prepared_image_path=Path("img1.jpg"),
            image_attr_values=(1, 2),
            text_attr_values=(1, 2),
            image_accessory_ids=(2,),
            text_accessory_ids=(2,),
        ),
        PasPair(
            dataset="D",
            query_type="hard",
            caption="person with bag and hat",
            unique_name="img2.jpg",
            image_path="images/D/img2.jpg",
            prepared_image_path=Path("img2.jpg"),
            image_attr_values=(1, 2),
            text_attr_values=(1, 2),
            image_accessory_ids=(1, 2),
            text_accessory_ids=(1, 2),
        ),
    ]
    image_embeddings = {
        f"D\timages/D/img{index}.jpg": np.eye(3, dtype=np.float32)[index]
        for index in range(3)
    }
    text_embeddings = {
        pair.caption: np.eye(3, dtype=np.float32)[index]
        for index, pair in enumerate(pairs)
    }

    scalar_rows = evaluate_text_to_image_by_metadata(
        pairs,
        image_embeddings,
        text_embeddings,
        ground_truth_mode="scalar_attributes",
    )
    accessory_rows = evaluate_text_to_image_by_metadata(
        pairs,
        image_embeddings,
        text_embeddings,
        ground_truth_mode="scalar_plus_accessories",
    )

    assert scalar_rows[0]["avg_gt_per_query"] == pytest.approx(3.0)
    assert accessory_rows[0]["avg_gt_per_query"] == pytest.approx(5 / 3)


@pytest.mark.multimodal_unit
def test_metadata_eval_deduplicates_caption_and_unions_signatures():
    pairs = [
        PasPair(
            dataset="D",
            query_type="hard",
            caption="person carrying an item",
            unique_name="img0.jpg",
            image_path="images/D/img0.jpg",
            prepared_image_path=Path("img0.jpg"),
            image_attr_values=(1,),
            text_attr_values=(1,),
            image_accessory_ids=(1,),
            text_accessory_ids=(1,),
        ),
        PasPair(
            dataset="D",
            query_type="hard",
            caption="person carrying an item",
            unique_name="img1.jpg",
            image_path="images/D/img1.jpg",
            prepared_image_path=Path("img1.jpg"),
            image_attr_values=(2,),
            text_attr_values=(2,),
            image_accessory_ids=(2,),
            text_accessory_ids=(2,),
        ),
        PasPair(
            dataset="D",
            query_type="original_captions",
            caption="gallery-only caption",
            unique_name="img2.jpg",
            image_path="images/D/img2.jpg",
            prepared_image_path=Path("img2.jpg"),
            image_attr_values=(1,),
            text_attr_values=(1,),
            image_accessory_ids=(1,),
            text_accessory_ids=(),
        ),
    ]
    image_embeddings = {
        f"D\timages/D/img{index}.jpg": np.eye(3, dtype=np.float32)[index]
        for index in range(3)
    }

    rows = evaluate_text_to_image_by_metadata(
        pairs,
        image_embeddings,
        {
            "person carrying an item": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        },
        ground_truth_mode="scalar_plus_accessories",
    )

    assert rows[0]["num_queries"] == 1
    assert rows[0]["avg_gt_per_query"] == pytest.approx(3.0)


@pytest.mark.multimodal_unit
def test_easy_query_ground_truth_modes_remain_separate():
    pairs = [
        PasPair(
            dataset="D",
            query_type="easy",
            caption=f"easy query {index}",
            unique_name=f"img{index}.jpg",
            image_path=f"images/D/img{index}.jpg",
            prepared_image_path=Path(f"img{index}.jpg"),
            image_attr_values=(1, 2),
            text_attr_values=(1, -1),
            image_accessory_ids=(index + 1,),
            text_accessory_ids=(),
        )
        for index in range(2)
    ]
    image_embeddings = {
        f"D\timages/D/img{index}.jpg": np.eye(2, dtype=np.float32)[index]
        for index in range(2)
    }
    text_embeddings = {
        f"easy query {index}": np.eye(2, dtype=np.float32)[index] for index in range(2)
    }

    paired = evaluate_text_to_image(pairs, image_embeddings, text_embeddings)
    scalar = evaluate_text_to_image_by_metadata(
        pairs,
        image_embeddings,
        text_embeddings,
        ground_truth_mode="scalar_attributes",
    )
    accessory = evaluate_text_to_image_by_metadata(
        pairs,
        image_embeddings,
        text_embeddings,
        ground_truth_mode="scalar_plus_accessories",
    )

    assert paired[0]["avg_gt_per_query"] == pytest.approx(1.0)
    assert scalar[0]["avg_gt_per_query"] == pytest.approx(2.0)
    assert accessory == scalar


@pytest.mark.multimodal_unit
def test_resolve_existing_image_items_uses_later_duplicate(tmp_path):
    missing_path = tmp_path / "missing.jpg"
    existing_path = tmp_path / "existing.jpg"
    existing_path.write_bytes(b"image")
    pairs = [
        PasPair(
            dataset="D",
            query_type="easy",
            caption="one",
            unique_name="missing.jpg",
            image_path="images/D/img0.jpg",
            prepared_image_path=missing_path,
        ),
        PasPair(
            dataset="D",
            query_type="easy",
            caption="two",
            unique_name="existing.jpg",
            image_path="images/D/img0.jpg",
            prepared_image_path=existing_path,
        ),
    ]
    key = _pair_image_key(pairs[0])

    candidates = _unique_image_candidates(pairs)
    resolved = _resolve_existing_image_items(candidates, pairs)

    assert candidates == [(key, missing_path)]
    assert resolved == [(key, existing_path)]


@pytest.mark.multimodal_unit
def test_image_embeddings_require_prepared_images(tmp_path):
    pair = PasPair(
        dataset="D",
        query_type="easy",
        caption="missing",
        unique_name="missing.jpg",
        image_path="images/D/img0.jpg",
        prepared_image_path=tmp_path / "missing.jpg",
    )

    with pytest.raises(ValueError, match="PAS prepared images are missing"):
        _extract_image_embeddings(
            model=object(),
            pairs=[pair],
            batch_size=1,
            num_workers=0,
            device=torch.device("cpu"),
        )


@pytest.mark.multimodal_unit
@pytest.mark.parametrize(
    ("canonicalize_text", "expected_tokenizer_text"),
    [
        (False, "A RED_Shirt!"),
        (True, "a red shirt"),
    ],
)
def test_text_embeddings_leave_canonicalization_to_tokenizer(
    canonicalize_text,
    expected_tokenizer_text,
):
    class RecordingProcessor:
        def __init__(self):
            self.text_batches = []

        def __call__(self, *, text, **kwargs):
            self.text_batches.append(list(text))
            batch_size = len(text)
            return {
                "input_ids": torch.ones((batch_size, 2), dtype=torch.long),
                "attention_mask": torch.ones(
                    (batch_size, 2),
                    dtype=torch.long,
                ),
            }

    class TextEncoder:
        def get_text_features(self, input_ids, attention_mask=None):
            return input_ids.float()

    processor = RecordingProcessor()
    tokenizer = CLIPCompatibleTokenizer(
        SigLIP2WrappedTokenizer(
            processor,
            canonicalize=canonicalize_text,
        )
    )
    model = SimpleNamespace(
        tokenizer=tokenizer,
        model=SimpleNamespace(
            backbone=SimpleNamespace(inner=TextEncoder()),
        ),
        eval=lambda: None,
    )

    embeddings = pas._extract_text_embeddings(
        model,
        ["A RED_Shirt!"],
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert processor.text_batches == [[expected_tokenizer_text]]
    assert list(embeddings) == ["A RED_Shirt!"]


@pytest.mark.multimodal_unit
def test_load_pas_pairs_treats_not_visible_as_missing(tmp_path):
    pairs_file = tmp_path / "val_pairs.json"
    pairs_file.write_text(
        json.dumps(
            [
                {
                    "dataset": "D",
                    "query_type": "easy",
                    "caption": "not visible query",
                    "unique_name": "img0.jpg",
                    "image_path": "images/D/img0.jpg",
                    "image_attr_values": [8, 5],
                    "text_attr_values": [8, 5],
                }
            ]
        )
    )
    _write_attribute_vocab(
        tmp_path / "attribute_vocab.json",
        not_visible=True,
    )

    pairs = load_pas_pairs(
        {"image_dir": str(tmp_path / "images")},
        pairs_file,
        ground_truth_mode="scalar_attributes",
    )

    assert pairs[0].image_attr_values == (-1, -1)
    assert pairs[0].text_attr_values == (-1, -1)


@pytest.mark.multimodal_unit
@pytest.mark.parametrize("query_type", pas.PAS_QUERY_TYPES)
def test_load_pas_pairs_accepts_supported_query_types(tmp_path, query_type):
    pairs_file = tmp_path / "test_pairs.json"
    pairs_file.write_text(
        json.dumps(
            [
                {
                    "dataset": "D",
                    "query_type": query_type,
                    "caption": "supported query",
                    "unique_name": "img0.jpg",
                    "image_path": "images/D/img0.jpg",
                }
            ]
        )
    )

    pairs = load_pas_pairs(
        {"image_dir": str(tmp_path / "images")},
        pairs_file,
    )

    assert pairs[0].query_type == query_type


@pytest.mark.multimodal_unit
def test_load_pas_pairs_rejects_unknown_query_type(tmp_path):
    pairs_file = tmp_path / "test_pairs.json"
    pairs_file.write_text(
        json.dumps(
            [
                {
                    "dataset": "D",
                    "query_type": "medum",
                    "caption": "query with typo",
                    "unique_name": "img0.jpg",
                    "image_path": "images/D/img0.jpg",
                }
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="query_type must be one of",
    ):
        load_pas_pairs(
            {"image_dir": str(tmp_path / "images")},
            pairs_file,
        )


@pytest.mark.multimodal_unit
def test_load_pas_pairs_reads_valid_accessories(tmp_path):
    pairs_file = tmp_path / "test_pairs.json"
    pairs_file.write_text(
        json.dumps(
            [
                {
                    "dataset": "D",
                    "query_type": "hard",
                    "caption": "person with bag",
                    "unique_name": "img0.jpg",
                    "image_path": "images/D/img0.jpg",
                    "image_attr_values": [1, 2],
                    "text_attr_values": [1, 2],
                    "image_accessory_ids": [1, 2],
                    "text_accessory_ids": [1],
                }
            ]
        )
    )
    _write_attribute_vocab(tmp_path / "attribute_vocab.json")
    _write_accessory_vocab(tmp_path / "accessory_vocab.json")

    pairs = load_pas_pairs(
        {"image_dir": str(tmp_path / "images")},
        pairs_file,
        ground_truth_mode="scalar_plus_accessories",
    )

    assert pairs[0].image_accessory_ids == (1, 2)
    assert pairs[0].text_accessory_ids == (1,)


@pytest.mark.multimodal_unit
def test_load_pas_pairs_requires_accessory_fields(tmp_path):
    pairs_file = tmp_path / "test_pairs.json"
    pairs_file.write_text(
        json.dumps(
            [
                {
                    "dataset": "D",
                    "query_type": "hard",
                    "caption": "legacy query",
                    "unique_name": "img0.jpg",
                    "image_path": "images/D/img0.jpg",
                    "image_attr_values": [1, 2],
                    "text_attr_values": [1, 2],
                }
            ]
        )
    )
    _write_attribute_vocab(tmp_path / "attribute_vocab.json")
    _write_accessory_vocab(tmp_path / "accessory_vocab.json")

    with pytest.raises(ValueError, match="fields are missing"):
        load_pas_pairs(
            {"image_dir": str(tmp_path / "images")},
            pairs_file,
            ground_truth_mode="scalar_plus_accessories",
        )


@pytest.mark.multimodal_unit
def test_load_pas_pairs_rejects_inconsistent_accessories(tmp_path):
    pairs_file = tmp_path / "test_pairs.json"
    base = {
        "dataset": "D",
        "query_type": "hard",
        "image_path": "images/D/img0.jpg",
        "image_attr_values": [1, 2],
        "text_attr_values": [1, 2],
    }
    pairs_file.write_text(
        json.dumps(
            [
                {
                    **base,
                    "caption": "person with bag",
                    "unique_name": "img0_a.jpg",
                    "image_accessory_ids": [1],
                    "text_accessory_ids": [1],
                },
                {
                    **base,
                    "caption": "person with hat",
                    "unique_name": "img0_b.jpg",
                    "image_accessory_ids": [2],
                    "text_accessory_ids": [2],
                },
            ]
        )
    )
    _write_attribute_vocab(tmp_path / "attribute_vocab.json")
    _write_accessory_vocab(tmp_path / "accessory_vocab.json")

    with pytest.raises(ValueError, match="inconsistent accessory"):
        load_pas_pairs(
            {"image_dir": str(tmp_path / "images")},
            pairs_file,
            ground_truth_mode="scalar_plus_accessories",
        )


@pytest.mark.multimodal_unit
@pytest.mark.parametrize("invalid_id", [1.9, "2", True])
def test_load_pas_pairs_rejects_non_integer_scalar_ids(tmp_path, invalid_id):
    pairs_file = tmp_path / "test_pairs.json"
    pairs_file.write_text(
        json.dumps(
            [
                {
                    "dataset": "D",
                    "query_type": "easy",
                    "caption": "query",
                    "unique_name": "img0.jpg",
                    "image_path": "images/D/img0.jpg",
                    "image_attr_values": [invalid_id, 2],
                    "text_attr_values": [1, 2],
                }
            ]
        )
    )
    _write_attribute_vocab(tmp_path / "attribute_vocab.json")

    with pytest.raises(ValueError, match="actual integer IDs"):
        load_pas_pairs(
            {"image_dir": str(tmp_path / "images")},
            pairs_file,
            ground_truth_mode="scalar_attributes",
        )


@pytest.mark.multimodal_unit
def test_load_pas_pairs_requires_scalar_fields(tmp_path):
    pairs_file = tmp_path / "test_pairs.json"
    pairs_file.write_text(
        json.dumps(
            [
                {
                    "dataset": "D",
                    "query_type": "easy",
                    "caption": "query",
                    "unique_name": "img0.jpg",
                    "image_path": "images/D/img0.jpg",
                    "image_attr_values": [1, 2],
                }
            ]
        )
    )
    _write_attribute_vocab(tmp_path / "attribute_vocab.json")

    with pytest.raises(ValueError, match="text_attr_values"):
        load_pas_pairs(
            {"image_dir": str(tmp_path / "images")},
            pairs_file,
            ground_truth_mode="scalar_attributes",
        )
