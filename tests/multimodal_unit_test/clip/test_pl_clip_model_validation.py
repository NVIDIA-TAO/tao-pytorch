# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for exact PAS validation during CLIP training."""

from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model as pl_clip_model
from nvidia_tao_pytorch.config.clip.default_config import CLIPValDataConfig
from nvidia_tao_pytorch.multimodal.clip.model.evaluation.pas import (
    PAS_METADATA_QUERY_TYPES,
    PasPair,
    build_pas_embedding_maps_from_rows,
    evaluate_pas_metadata_embeddings,
)
from nvidia_tao_pytorch.multimodal.clip.model.evaluation.retrieval import (
    RetrievalMetrics,
)
from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import (
    CLIPPlModel,
    _all_gather_variable_rows,
    _broadcast_pas_metrics,
)


class _FeatureModel:
    """Minimal image/text encoder used by validation-step tests."""

    def __call__(self, image=None, text=None):
        if image is not None:
            return {"image_features": image}
        return {"text_features": text}


class _CaptureEvaluator:
    """Capture standard paired-retrieval evaluator inputs."""

    def __init__(self):
        self.calls = []

    def evaluate_bidirectional(self, image_embeddings, text_embeddings, **kwargs):
        self.calls.append(
            {
                "image_embeddings": image_embeddings,
                "text_embeddings": text_embeddings,
                "kwargs": kwargs,
            }
        )
        metrics = RetrievalMetrics(
            recall_at_k={1: 1.0, 5: 1.0},
            map_score=1.0,
            median_rank=1.0,
            mean_rank=1.0,
            auc=1.0,
        )
        return {
            "image_to_text": metrics,
            "text_to_image": metrics,
        }


def _pas_pairs():
    """Build two one-image galleries with easy/medium/hard queries."""
    pairs = []
    for dataset_index, dataset in enumerate(("dataset_a", "dataset_b"), start=1):
        for row_offset, query_type in enumerate(PAS_METADATA_QUERY_TYPES):
            pairs.append(
                PasPair(
                    dataset=dataset,
                    query_type=query_type,
                    caption=f"{dataset} {query_type}",
                    unique_name=f"{dataset}_{row_offset}.jpg",
                    image_path=f"images/{dataset}/person.jpg",
                    prepared_image_path=Path(
                        f"/unused/{dataset}_{row_offset}.jpg"
                    ),
                    image_attr_values=(dataset_index,),
                    text_attr_values=(dataset_index,),
                    image_accessory_ids=(1,),
                    text_accessory_ids=(1,) if query_type == "hard" else (),
                )
            )
    return pairs


def _pas_row_features():
    """Return normalized row features aligned with ``_pas_pairs``."""
    image_rows = []
    text_rows = []
    for dataset_index in (0, 1):
        feature = torch.tensor(
            [1.0, 0.0] if dataset_index == 0 else [0.0, 1.0]
        )
        for _ in PAS_METADATA_QUERY_TYPES:
            image_rows.append(feature)
            text_rows.append(feature)
    return torch.stack(image_rows), torch.stack(text_rows)


def _validation_model(metadata_match_eval, evaluator):
    """Build a lightweight object for calling validation methods."""
    logged = []
    model = SimpleNamespace(
        metadata_match_eval=metadata_match_eval,
        metadata_match_mode="scalar_plus_accessories",
        retrieval_evaluator=evaluator,
        image_embeddings=[],
        text_embeddings=[],
        pas_row_indices=[],
        pas_validation_datasets=[],
        _pas_validation_pairs=_pas_pairs() if metadata_match_eval else None,
        model=_FeatureModel(),
        tokenizer=None,
        device=torch.device("cpu"),
        status_logging_dict={},
        log=lambda *args, **kwargs: logged.append((args, kwargs)),
        logged=logged,
        trainer=SimpleNamespace(sanity_checking=True),
    )
    for method_name in (
        "_evaluate_accumulated_retrieval",
        "_evaluate_accumulated_pas",
        "_load_pas_validation_pairs",
        "_log_pas_metrics",
        "_evaluate_and_log_paired_retrieval",
    ):
        setattr(
            model,
            method_name,
            MethodType(getattr(CLIPPlModel, method_name), model),
        )
    return model


def _run_two_rank_pas_validation(rank, world_size, init_method):
    """Exercise uneven row gathering and PAS metric broadcast with Gloo."""
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
    )
    try:
        pairs = _pas_pairs()
        image_rows, text_rows = _pas_row_features()
        indices = (
            torch.tensor([0, 2, 4, 5])
            if rank == 0
            else torch.tensor([1, 3])
        )
        gathered_images = _all_gather_variable_rows(
            image_rows[indices],
            torch.device("cpu"),
        )
        gathered_text = _all_gather_variable_rows(
            text_rows[indices],
            torch.device("cpu"),
        )
        gathered_indices = _all_gather_variable_rows(
            indices[:, None],
            torch.device("cpu"),
        )

        weighted_rows = None
        if rank == 0:
            image_embeddings, text_embeddings = (
                build_pas_embedding_maps_from_rows(
                    pairs,
                    gathered_indices.flatten(),
                    gathered_images,
                    gathered_text,
                )
            )
            output = evaluate_pas_metadata_embeddings(
                pairs,
                image_embeddings,
                text_embeddings,
                ground_truth_mode="scalar_plus_accessories",
            )
            weighted_rows = output["metadata_weighted_aggregate"]
        metrics = _broadcast_pas_metrics(
            weighted_rows,
            torch.device("cpu"),
        )
        assert set(metrics) == set(PAS_METADATA_QUERY_TYPES)
        assert all(value["mAP"] == 1.0 for value in metrics.values())
        assert all(value["num_queries"] == 2 for value in metrics.values())
    finally:
        dist.destroy_process_group()


@pytest.mark.multimodal_unit
class TestCLIPPASValidation:
    """Test opt-in exact PAS validation."""

    def test_metadata_match_eval_defaults_to_paired_validation(self):
        """Existing specs retain paired diagonal validation."""
        config = CLIPValDataConfig()

        assert config.metadata_match_eval is False
        assert config.metadata_match_mode == "scalar_attributes"

    def test_validation_step_collects_pas_row_indices(self):
        """Metadata validation collects embeddings and stable split rows."""
        model = _validation_model(
            metadata_match_eval=True,
            evaluator=_CaptureEvaluator(),
        )
        image_features = torch.eye(2)
        text_features = torch.eye(2)
        metadata = {
            "image_attr_values": torch.tensor([[1], [2]]),
            "text_attr_values": torch.tensor([[1], [2]]),
            "image_accessory_ids": torch.tensor([[1], [1]]),
            "text_accessory_ids": torch.tensor([[0], [1]]),
            "pas_row_index": torch.tensor([3, 4]),
        }

        CLIPPlModel.validation_step(
            model,
            (image_features, text_features, metadata),
            batch_idx=0,
        )

        assert torch.equal(model.image_embeddings[0], image_features)
        assert torch.equal(model.text_embeddings[0], text_features)
        assert torch.equal(
            model.pas_row_indices[0],
            metadata["pas_row_index"],
        )

    def test_validation_step_requires_pas_row_indices(self):
        """Missing stable row identity fails before metric computation."""
        model = _validation_model(
            metadata_match_eval=True,
            evaluator=_CaptureEvaluator(),
        )
        metadata = {
            "image_attr_values": torch.tensor([[1]]),
            "text_attr_values": torch.tensor([[1]]),
            "image_accessory_ids": torch.tensor([[1]]),
            "text_accessory_ids": torch.tensor([[1]]),
        }

        with pytest.raises(ValueError, match="pas_row_index"):
            CLIPPlModel.validation_step(
                model,
                (torch.eye(1), torch.eye(1), metadata),
                batch_idx=0,
            )

    def test_default_validation_preserves_paired_retrieval(self):
        """Default validation still invokes the existing evaluator."""
        evaluator = _CaptureEvaluator()
        model = _validation_model(
            metadata_match_eval=False,
            evaluator=evaluator,
        )
        model.image_embeddings.append(torch.eye(3))
        model.text_embeddings.append(torch.eye(3))

        CLIPPlModel.on_validation_epoch_end(model)

        assert evaluator.calls[0]["kwargs"] == {}

    def test_single_rank_pas_validation_matches_shared_evaluator(self):
        """Training validation uses the shared PAS weighted semantics."""
        model = _validation_model(
            metadata_match_eval=True,
            evaluator=_CaptureEvaluator(),
        )
        image_rows, text_rows = _pas_row_features()
        metrics = model._evaluate_accumulated_pas(
            image_rows,
            text_rows,
            torch.arange(len(_pas_pairs())),
        )

        assert set(metrics) == set(PAS_METADATA_QUERY_TYPES)
        assert all(value["mAP"] == 1.0 for value in metrics.values())
        assert all(value["num_queries"] == 2 for value in metrics.values())

        model._log_pas_metrics(metrics, "val")
        assert [call[0][0] for call in model.logged] == [
            "val/pas/easy_mAP",
            "val/pas/medium_mAP",
            "val/pas/hard_mAP",
        ]

    def test_multiple_pas_pair_files_load_in_dataloader_order(
        self, tmp_path, monkeypatch
    ):
        """Dataset-specific pairs concatenate in validation-loader order."""
        pair_files = [
            tmp_path / "first_pairs.json",
            tmp_path / "second_pairs.json",
        ]
        for pairs_file in pair_files:
            pairs_file.write_text("[]")
        datasets = [
            SimpleNamespace(attribute_pairs_file=str(pairs_file))
            for pairs_file in pair_files
        ]
        expected_by_path = {
            pair_files[0]: _pas_pairs()[:3],
            pair_files[1]: _pas_pairs()[3:],
        }
        calls = []

        def fake_load_pas_pairs(dataset, pairs_file, ground_truth_mode):
            calls.append((dataset, pairs_file, ground_truth_mode))
            return expected_by_path[pairs_file]

        monkeypatch.setattr(
            pl_clip_model,
            "load_pas_pairs",
            fake_load_pas_pairs,
        )
        model = _validation_model(
            metadata_match_eval=True,
            evaluator=_CaptureEvaluator(),
        )
        model.pas_validation_datasets = datasets
        model._pas_validation_pairs = None

        loaded = model._load_pas_validation_pairs()

        assert loaded == _pas_pairs()
        assert [call[1] for call in calls] == pair_files
        assert all(
            call[2] == "scalar_plus_accessories" for call in calls
        )
        assert model._load_pas_validation_pairs() is loaded
        assert len(calls) == 2

    def test_sanity_validation_skips_incomplete_pas_split(self):
        """Lightning's partial sanity pass does not require full coverage."""
        model = _validation_model(
            metadata_match_eval=True,
            evaluator=_CaptureEvaluator(),
        )
        model.image_embeddings.append(torch.eye(1))
        model.text_embeddings.append(torch.eye(1))
        model.pas_row_indices.append(torch.tensor([0]))
        model._evaluate_accumulated_pas = lambda *args: pytest.fail(
            "PAS evaluation must be skipped during sanity checking"
        )

        CLIPPlModel.on_validation_epoch_end(model)

        assert model.status_logging_dict == {}

    def test_two_rank_gloo_pas_validation(self, tmp_path):
        """Uneven rank shards produce identical exact PAS metrics."""
        init_method = f"file://{tmp_path / 'gloo_init'}"
        mp.spawn(
            _run_two_rank_pas_validation,
            args=(2, init_method),
            nprocs=2,
            join=True,
        )
