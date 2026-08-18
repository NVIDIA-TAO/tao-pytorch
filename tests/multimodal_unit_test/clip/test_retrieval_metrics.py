# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CLIP retrieval metrics and PLModel evaluation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nvidia_tao_pytorch.multimodal.clip.model.evaluation.metrics import (
    compute_ap,
    compute_auc,
    compute_ndcg,
)
from nvidia_tao_pytorch.multimodal.clip.model.evaluation.retrieval import (
    log_retrieval_metrics,
    RetrievalEvaluator,
    RetrievalMetrics,
)


@pytest.mark.multimodal_unit
class TestComputeAP:
    """Tests for Average Precision computation."""

    def test_perfect_ranking(self):
        """Test AP with perfect ranking (all positives first)."""
        labels = np.array([1, 1, 1, 0, 0, 0])
        ap = compute_ap(labels)
        assert pytest.approx(ap, rel=1e-5) == 1.0

    def test_worst_ranking(self):
        """Test AP with worst ranking (all negatives first)."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        ap = compute_ap(labels)
        # AP = (1/4 + 2/5 + 3/6) / 3 = (0.25 + 0.4 + 0.5) / 3 ≈ 0.383
        assert ap < 0.5

    def test_mixed_ranking(self):
        """Test AP with mixed ranking."""
        labels = np.array([1, 0, 1, 0, 1, 0])
        ap = compute_ap(labels)
        # AP = (1/1 + 2/3 + 3/5) / 3 = (1.0 + 0.667 + 0.6) / 3 ≈ 0.756
        assert 0.7 < ap < 0.8

    def test_no_positives(self):
        """Test AP with no positive samples."""
        labels = np.array([0, 0, 0])
        ap = compute_ap(labels)
        assert ap == 0.0


@pytest.mark.multimodal_unit
class TestComputeNDCG:
    """Tests for NDCG computation."""

    def test_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        labels = np.array([1, 1, 1, 0, 0])
        ndcg = compute_ndcg(labels, k=5)
        assert pytest.approx(ndcg, rel=1e-5) == 1.0

    def test_no_positives(self):
        """Test NDCG with no positive samples."""
        labels = np.array([0, 0, 0])
        ndcg = compute_ndcg(labels, k=3)
        assert ndcg == 0.0

    def test_k_larger_than_results(self):
        """Test NDCG when k is larger than result count."""
        labels = np.array([1, 0])
        ndcg = compute_ndcg(labels, k=10)
        assert ndcg > 0


@pytest.mark.multimodal_unit
class TestComputeAUC:
    """Tests for AUC computation."""

    def test_perfect_separation(self):
        """Test AUC with perfect separation."""
        sims = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        labels = np.array([1, 1, 1, 0, 0, 0])
        auc = compute_auc(sims, labels)
        assert pytest.approx(auc, rel=1e-5) == 1.0

    def test_random_separation(self):
        """Test AUC with mixed separation."""
        sims = np.array([0.9, 0.7, 0.5, 0.3])
        labels = np.array([1, 0, 1, 0])
        auc = compute_auc(sims, labels)
        # With mixed but somewhat ordered similarities
        assert 0.0 <= auc <= 1.0

    def test_no_negatives(self):
        """Test AUC with no negative samples."""
        sims = np.array([0.9, 0.8])
        labels = np.array([1, 1])
        auc = compute_auc(sims, labels)
        assert auc == 1.0


@pytest.mark.multimodal_unit
class TestRetrievalEvaluatorGPU:
    """Tests for GPU-accelerated RetrievalEvaluator."""

    def test_gpu_device_initialization(self):
        """Test that evaluator initializes with GPU device."""
        evaluator = RetrievalEvaluator(k_values=(1, 5, 10), device='cuda:0')
        assert evaluator.device == torch.device('cuda:0')

    def test_default_device_is_cuda(self):
        """Test that default device is CUDA."""
        evaluator = RetrievalEvaluator(k_values=(1, 5, 10))
        assert evaluator.device == torch.device('cuda')

    def test_batch_size_parameter(self):
        """Test that batch_size parameter is stored."""
        evaluator = RetrievalEvaluator(k_values=(1, 5, 10), batch_size=512)
        assert evaluator.batch_size == 512

    def test_evaluate_with_torch_tensors(self):
        """Test evaluation with torch tensor inputs."""
        torch.manual_seed(42)
        query_embs = torch.randn(10, 64)
        gallery_embs = query_embs.clone()

        gt = [[i] for i in range(10)]

        evaluator = RetrievalEvaluator(k_values=(1, 5, 10), device='cuda')
        metrics = evaluator.evaluate(query_embs, gallery_embs, gt)

        assert metrics.num_queries == 10
        assert pytest.approx(metrics.recall_at_k[1], rel=1e-3) == 1.0
        assert pytest.approx(metrics.map_score, rel=1e-3) == 1.0

    def test_evaluate_with_numpy_on_gpu(self):
        """Test that numpy arrays are converted to GPU tensors."""
        np.random.seed(42)
        query_embs = np.random.randn(10, 64).astype(np.float32)
        gallery_embs = query_embs.copy()

        gt = [[i] for i in range(10)]

        evaluator = RetrievalEvaluator(k_values=(1, 5, 10), device='cuda')
        metrics = evaluator.evaluate(query_embs, gallery_embs, gt)

        assert metrics.num_queries == 10
        assert pytest.approx(metrics.recall_at_k[1], rel=1e-3) == 1.0

    def test_gpu_similarity_matrix_correctness(self):
        """Test that GPU similarity computation matches expected values."""
        query_embs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
        gallery_embs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device='cuda')

        evaluator = RetrievalEvaluator(k_values=(1,), device='cuda')
        sim = evaluator._compute_similarity_matrix_gpu(query_embs, gallery_embs)

        sim_cpu = sim.cpu()
        assert sim_cpu.shape == (2, 3)
        assert pytest.approx(sim_cpu[0, 0].item(), rel=1e-5) == 1.0
        assert pytest.approx(sim_cpu[0, 1].item(), abs=1e-5) == 0.0
        assert pytest.approx(sim_cpu[1, 1].item(), rel=1e-5) == 1.0

    def test_batched_similarity_large_dataset(self):
        """Test batched computation with dataset larger than batch_size."""
        torch.manual_seed(42)
        n_queries = 2000
        n_gallery = 1000
        embed_dim = 128

        query_embs = torch.randn(n_queries, embed_dim)
        gallery_embs = torch.randn(n_gallery, embed_dim)

        gt = [[i % n_gallery] for i in range(n_queries)]

        evaluator = RetrievalEvaluator(k_values=(1, 5, 10), device='cuda', batch_size=256)
        metrics = evaluator.evaluate(query_embs, gallery_embs, gt)

        assert metrics.num_queries == n_queries
        assert metrics.gallery_size == n_gallery
        assert 0.0 <= metrics.recall_at_k[1] <= 1.0

    def test_auc_computed_by_default(self):
        """Test that AUC is computed by default."""
        np.random.seed(42)
        query_embs = np.random.randn(10, 64).astype(np.float32)
        gallery_embs = query_embs.copy()

        gt = [[i] for i in range(10)]

        evaluator = RetrievalEvaluator(k_values=(1, 5, 10), device='cuda')
        metrics = evaluator.evaluate(query_embs, gallery_embs, gt)

        assert metrics.auc is not None
        assert 0.0 <= metrics.auc <= 1.0


@pytest.mark.multimodal_unit
class TestRetrievalEvaluator:
    """Tests for RetrievalEvaluator."""

    def test_perfect_retrieval(self):
        """Test metrics with perfect retrieval."""
        np.random.seed(42)
        query_embs = np.random.randn(10, 64)
        gallery_embs = query_embs.copy()  # Identical
        
        gt = [[i] for i in range(10)]
        
        evaluator = RetrievalEvaluator(k_values=(1, 5, 10))
        metrics = evaluator.evaluate(query_embs, gallery_embs, gt)
        
        assert metrics.num_queries == 10
        assert metrics.gallery_size == 10
        assert pytest.approx(metrics.recall_at_k[1], rel=1e-3) == 1.0
        assert pytest.approx(metrics.map_score, rel=1e-3) == 1.0
        assert metrics.median_rank == 1.0

    def test_with_matrix_ground_truth(self):
        """Test metrics with binary matrix ground truth."""
        query_embs = np.eye(5)
        gallery_embs = np.eye(5)
        
        gt_matrix = np.eye(5)
        
        evaluator = RetrievalEvaluator(k_values=(1, 5, 10))
        metrics = evaluator.evaluate(query_embs, gallery_embs, gt_matrix)
        
        assert metrics.num_queries == 5
        assert pytest.approx(metrics.recall_at_k[1], rel=1e-3) == 1.0

    def test_multiple_relevant_items(self):
        """Test metrics with multiple relevant items per query."""
        np.random.seed(42)
        query_embs = np.random.randn(3, 16)
        gallery_embs = np.random.randn(10, 16)
        
        gt = [[0, 1, 2], [3, 4], [5]]
        
        evaluator = RetrievalEvaluator(k_values=(1, 3, 5))
        metrics = evaluator.evaluate(query_embs, gallery_embs, gt)
        
        assert metrics.num_queries == 3
        assert 1 in metrics.recall_at_k
        assert 3 in metrics.recall_at_k
        assert 5 in metrics.recall_at_k


@pytest.mark.multimodal_unit
class TestBidirectionalRetrieval:
    """Tests for bidirectional retrieval evaluation."""

    def test_auto_diagonal_ground_truth(self):
        """Test automatic diagonal ground truth for paired data."""
        np.random.seed(42)
        image_embs = np.random.randn(5, 32)
        text_embs = image_embs.copy()  # Identical for perfect retrieval
        
        evaluator = RetrievalEvaluator(k_values=(1, 5, 10))
        results = evaluator.evaluate_bidirectional(image_embs, text_embs)
        
        assert 'image_to_text' in results
        assert 'text_to_image' in results
        assert pytest.approx(results['image_to_text'].recall_at_k[1], rel=1e-3) == 1.0
        assert pytest.approx(results['text_to_image'].recall_at_k[1], rel=1e-3) == 1.0

    def test_with_explicit_ground_truth(self):
        """Test with explicit ground truth."""
        np.random.seed(42)
        image_embs = np.random.randn(3, 16)
        text_embs = np.random.randn(5, 16)
        
        image_to_text_gt = [[0, 1], [2], [3, 4]]
        text_to_image_gt = [[0], [0], [1], [2], [2]]
        
        evaluator = RetrievalEvaluator(k_values=(1, 5, 10))
        results = evaluator.evaluate_bidirectional(
            image_embs, text_embs,
            image_to_text_gt=image_to_text_gt,
            text_to_image_gt=text_to_image_gt
        )
        
        assert results['image_to_text'].num_queries == 3
        assert results['text_to_image'].num_queries == 5


@pytest.mark.multimodal_unit
class TestRetrievalMetrics:
    """Tests for RetrievalMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = RetrievalMetrics(
            recall_at_k={1: 0.8, 5: 0.95},
            map_score=0.75,
            median_rank=2.0,
            mean_rank=3.5,
            num_queries=100,
            gallery_size=1000,
        )
        
        d = metrics.to_dict()
        
        assert d['mAP'] == 0.75
        assert d['recall@1'] == 0.8
        assert d['recall@5'] == 0.95
        assert d['median_rank'] == 2.0
        assert d['num_queries'] == 100

    def test_str_representation(self):
        """Test string representation."""
        metrics = RetrievalMetrics(
            recall_at_k={1: 0.8, 5: 0.95},
            map_score=0.75,
            median_rank=2.0,
            mean_rank=3.5,
            num_queries=100,
            gallery_size=1000,
        )
        
        s = str(metrics)
        
        assert 'mAP: 0.75' in s
        assert 'R@1: 0.80' in s
        assert 'MedR: 2.0' in s


@pytest.mark.multimodal_unit
class TestLogRetrievalMetrics:
    """Tests for log_retrieval_metrics function."""

    def test_log_includes_auc_column(self):
        """Test that logged table includes AUC column."""
        metrics = {
            'image_to_text': RetrievalMetrics(
                recall_at_k={1: 0.8, 5: 0.95, 10: 0.98},
                map_score=0.75,
                median_rank=2.0,
                mean_rank=3.5,
                num_queries=100,
                gallery_size=1000,
                auc=0.92,
            ),
            'text_to_image': RetrievalMetrics(
                recall_at_k={1: 0.7, 5: 0.90, 10: 0.95},
                map_score=0.70,
                median_rank=3.0,
                mean_rank=4.5,
                num_queries=100,
                gallery_size=1000,
                auc=0.88,
            ),
        }

        with patch(
            'nvidia_tao_pytorch.multimodal.clip.model.evaluation.retrieval.logging'
        ) as mock_log:
            log_retrieval_metrics(metrics, prefix="test")

            mock_log.info.assert_called_once()
            log_output = str(mock_log.info.call_args)
            assert 'AUC' in log_output
            # Tabulate may use shorter format (0.92 vs 0.9200)
            assert '0.92' in log_output
            assert '0.88' in log_output
