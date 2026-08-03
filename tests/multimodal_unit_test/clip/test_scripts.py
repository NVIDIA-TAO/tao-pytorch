# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP scripts unit tests."""

import os
import tempfile
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

import nvidia_tao_pytorch.multimodal.clip.scripts.evaluate as evaluate_script
from nvidia_tao_pytorch.multimodal.clip.scripts.inference import (
    get_image_files,
    load_and_preprocess_batch,
    save_embeddings,
    load_text_file,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from nvidia_tao_pytorch.multimodal.clip.utils.utils import (
    load_model_from_checkpoint,
    SUPPORTED_CHECKPOINT_EXTENSIONS,
)
from nvidia_tao_pytorch.multimodal.clip.scripts.export import (
    CLIPVisionEncoder,
    CLIPTextEncoder,
    ExportFriendlyMHA,
    VALID_ENCODER_TYPES,
)
from nvidia_tao_pytorch.multimodal.clip.scripts.train import (
    _RankLocalDDPStrategy,
)


@pytest.mark.multimodal_unit
class TestRankLocalDDPSetup:
    """Test rank-local CUDA and NCCL initialization."""

    def test_ddp_process_group_omits_eager_device_id(self):
        """NCCL setup should stay lazy after the raw rank-local binding."""
        timeout = timedelta(seconds=30)
        cluster_environment = MagicMock()
        strategy = _RankLocalDDPStrategy(
            cluster_environment=cluster_environment,
            timeout=timeout,
        )

        with (
            patch.object(strategy, "set_world_ranks"),
            patch.object(
                strategy,
                "_get_process_group_backend",
                return_value="nccl",
            ),
            patch(
                "nvidia_tao_pytorch.multimodal.clip.scripts.train.reset_seed"
            ),
            patch(
                "nvidia_tao_pytorch.multimodal.clip.scripts.train."
                "_init_dist_connection"
            ) as init_dist,
        ):
            strategy.setup_distributed()

        init_dist.assert_called_once_with(
            cluster_environment,
            "nccl",
            timeout=timeout,
        )
        assert "device_id" not in init_dist.call_args.kwargs


@pytest.mark.multimodal_unit
class TestEvaluateRouting:
    """Test explicit routing between generic retrieval and direct PAS."""

    @staticmethod
    def _config(val_datasets, evaluate_datasets):
        return SimpleNamespace(
            encryption_key="",
            model=SimpleNamespace(type="test-model"),
            dataset=SimpleNamespace(
                val=SimpleNamespace(datasets=val_datasets),
            ),
            evaluate=SimpleNamespace(datasets=evaluate_datasets),
        )

    def test_validation_pairs_sidecar_stays_on_trainer_test(self, tmp_path):
        """A validation sidecar must not implicitly select direct PAS."""
        image_list = tmp_path / "val_list.txt"
        image_list.write_text("image.jpg\n")
        (tmp_path / "val_pairs.json").write_text("[]")
        config = self._config(
            [
                SimpleNamespace(
                    image_dir=str(tmp_path),
                    caption_dir=str(tmp_path),
                    image_list_file=str(image_list),
                )
            ],
            [],
        )
        model = MagicMock()
        datamodule = MagicMock()
        trainer = MagicMock()

        with (
            patch.object(
                evaluate_script,
                "initialize_evaluation_experiment",
                return_value=(None, {}),
            ),
            patch.object(
                evaluate_script,
                "CLIPPlModel",
                return_value=model,
            ),
            patch.object(
                evaluate_script,
                "CLIPDataModule",
                return_value=datamodule,
            ),
            patch.object(
                evaluate_script,
                "Trainer",
                return_value=trainer,
            ),
            patch.object(
                evaluate_script,
                "run_pas_evaluation",
            ) as run_pas,
        ):
            evaluate_script.run_experiment(config, key="")

        trainer.test.assert_called_once_with(model, datamodule=datamodule)
        run_pas.assert_not_called()

    def test_evaluate_datasets_explicitly_selects_direct_pas(self, tmp_path):
        """An evaluate dataset with a pairs sidecar selects direct PAS."""
        image_list = tmp_path / "test_list.txt"
        image_list.write_text("image.jpg\n")
        pairs_file = tmp_path / "test_pairs.json"
        pairs_file.write_text("[]")
        config = self._config(
            [],
            [SimpleNamespace(image_list_file=str(image_list))],
        )
        model = MagicMock()
        pairs = [MagicMock()]

        with (
            patch.object(
                evaluate_script,
                "initialize_evaluation_experiment",
                return_value=(None, {}),
            ),
            patch.object(
                evaluate_script,
                "CLIPPlModel",
                return_value=model,
            ),
            patch.object(
                evaluate_script,
                "resolve_pas_eval_data",
                return_value=(pairs, pairs_file),
            ) as resolve_pas,
            patch.object(
                evaluate_script,
                "run_pas_evaluation",
            ) as run_pas,
            patch.object(evaluate_script, "Trainer") as trainer_class,
        ):
            evaluate_script.run_experiment(config, key="")

        resolve_pas.assert_called_once_with(config, pairs_file)
        run_pas.assert_called_once_with(config, model, pairs)
        trainer_class.assert_not_called()

    def test_evaluate_datasets_requires_a_valid_pairs_file(self, tmp_path):
        """Explicit direct PAS configuration fails instead of falling back."""
        image_list = tmp_path / "test_list.txt"
        image_list.write_text("image.jpg\n")
        config = self._config(
            [SimpleNamespace(image_list_file=str(image_list))],
            [SimpleNamespace(image_list_file=str(image_list))],
        )

        with (
            patch.object(
                evaluate_script,
                "initialize_evaluation_experiment",
            ) as initialize,
            pytest.raises(
                ValueError,
                match="Direct PAS evaluation was requested",
            ),
        ):
            evaluate_script.run_experiment(config, key="")

        initialize.assert_not_called()


@pytest.mark.multimodal_unit
class TestGetImageFiles:
    """Test get_image_files function."""

    def test_finds_supported_extensions(self):
        """Test that all supported image extensions are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images with different extensions
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                img = Image.new('RGB', (100, 100), color='red')
                img.save(os.path.join(tmpdir, f'test{ext}'))

            found_files = get_image_files(tmpdir)
            assert len(found_files) == 6

    def test_ignores_unsupported_extensions(self):
        """Test that unsupported extensions are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(os.path.join(tmpdir, 'valid.jpg'))

            # Create files with unsupported extensions
            for ext in ['.txt', '.json', '.py', '.pdf']:
                with open(os.path.join(tmpdir, f'invalid{ext}'), 'w') as f:
                    f.write('test')

            found_files = get_image_files(tmpdir)
            assert len(found_files) == 1
            assert found_files[0].endswith('.jpg')

    def test_recursive_search(self):
        """Test that subdirectories are searched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectories
            subdir1 = os.path.join(tmpdir, 'subdir1')
            subdir2 = os.path.join(tmpdir, 'subdir1', 'subdir2')
            os.makedirs(subdir2)

            # Create images at different levels
            img = Image.new('RGB', (100, 100), color='red')
            img.save(os.path.join(tmpdir, 'root.jpg'))
            img.save(os.path.join(subdir1, 'level1.jpg'))
            img.save(os.path.join(subdir2, 'level2.jpg'))

            found_files = get_image_files(tmpdir)
            assert len(found_files) == 3

    def test_returns_sorted_list(self):
        """Test that returned list is sorted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new('RGB', (100, 100), color='red')
            for name in ['z.jpg', 'a.jpg', 'm.jpg']:
                img.save(os.path.join(tmpdir, name))

            found_files = get_image_files(tmpdir)
            assert found_files == sorted(found_files)

    def test_empty_directory(self):
        """Test behavior with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            found_files = get_image_files(tmpdir)
            assert not found_files


@pytest.mark.multimodal_unit
class TestSaveEmbeddings:
    """Test save_embeddings function."""

    def test_save_embeddings_creates_file(self):
        """Test that embeddings HDF5 file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = ["/path/to/img1.jpg", "/path/to/img2.jpg"]
            embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

            save_embeddings(image_paths, embeddings, tmpdir, embedding_type='image')

            output_path = os.path.join(tmpdir, "image_embeddings.h5")
            assert os.path.exists(output_path)

    def test_save_embeddings_correct_content(self):
        """Test that HDF5 file has correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = ["/path/to/img1.jpg"]
            embeddings = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

            save_embeddings(image_paths, embeddings, tmpdir, embedding_type='image')

            output_path = os.path.join(tmpdir, "image_embeddings.h5")
            with h5py.File(output_path, 'r') as f:
                loaded_embeddings = f['embeddings'][:]
                # Decode bytes to string if needed
                loaded_paths = [
                    p.decode('utf-8') if isinstance(p, bytes) else p
                    for p in f['image_paths'][:]
                ]

            assert len(loaded_paths) == 1
            assert loaded_paths[0] == "/path/to/img1.jpg"
            np.testing.assert_array_almost_equal(loaded_embeddings[0], [0.1, 0.2, 0.3])

    def test_save_embeddings_handles_unicode(self):
        """Test that embeddings with unicode paths are saved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = ["/path/to/图片.jpg"]
            embeddings = np.array([[0.1, 0.2]], dtype=np.float32)

            save_embeddings(image_paths, embeddings, tmpdir, embedding_type='image')

            output_path = os.path.join(tmpdir, "image_embeddings.h5")
            with h5py.File(output_path, 'r') as f:
                # Decode bytes to string if needed
                loaded_paths = [
                    p.decode('utf-8') if isinstance(p, bytes) else p
                    for p in f['image_paths'][:]
                ]

            assert loaded_paths[0] == "/path/to/图片.jpg"

    def test_save_embeddings_metadata(self):
        """Test that HDF5 file has correct metadata attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths = ["/path/to/img1.jpg", "/path/to/img2.jpg", "/path/to/img3.jpg"]
            embeddings = np.random.randn(3, 768).astype(np.float32)

            save_embeddings(image_paths, embeddings, tmpdir, embedding_type='image')

            output_path = os.path.join(tmpdir, "image_embeddings.h5")
            with h5py.File(output_path, 'r') as f:
                assert f.attrs['num_images'] == 3
                assert f.attrs['embedding_dim'] == 768

    def test_save_text_embeddings(self):
        """Test saving text embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            texts = ["a photo of a cat", "a photo of a dog"]
            embeddings = np.random.randn(2, 768).astype(np.float32)

            save_embeddings(texts, embeddings, tmpdir, embedding_type='text')

            output_path = os.path.join(tmpdir, "text_embeddings.h5")
            assert os.path.exists(output_path)

            with h5py.File(output_path, 'r') as f:
                assert f.attrs['num_texts'] == 2
                assert f.attrs['embedding_type'] == 'text'
                loaded_texts = [
                    t.decode('utf-8') if isinstance(t, bytes) else t
                    for t in f['texts'][:]
                ]
                assert loaded_texts[0] == "a photo of a cat"


@pytest.mark.multimodal_unit
class TestSupportedExtensions:
    """Test supported extension constants."""

    def test_image_extensions(self):
        """Test that common image extensions are supported."""
        expected = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        assert SUPPORTED_IMAGE_EXTENSIONS == expected

    def test_checkpoint_extensions(self):
        """Test that checkpoint extensions are consistent."""
        expected = {'.pth', '.ckpt'}
        assert SUPPORTED_CHECKPOINT_EXTENSIONS == expected


@pytest.mark.multimodal_unit
class TestLoadAndPreprocessBatch:
    """Test load_and_preprocess_batch function."""

    def test_returns_none_for_empty_batch(self):
        """Test that None is returned when no images can be loaded."""
        device = torch.device('cpu')

        # Non-existent files
        batch_files = ['/nonexistent/path/image1.jpg', '/nonexistent/path/image2.jpg']

        # Simple identity preprocess
        def preprocess(img):
            return torch.zeros(3, 224, 224)

        batch, valid_paths = load_and_preprocess_batch(batch_files, preprocess, device)
        assert batch is None
        assert not valid_paths

    def test_loads_valid_images(self):
        """Test that valid images are loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            img_paths = []
            for i in range(3):
                img = Image.new('RGB', (224, 224), color='red')
                path = os.path.join(tmpdir, f'img{i}.jpg')
                img.save(path)
                img_paths.append(path)

            device = torch.device('cpu')

            def preprocess(img):
                return torch.zeros(3, 224, 224)

            batch, valid_paths = load_and_preprocess_batch(img_paths, preprocess, device)

            assert batch is not None
            assert batch.shape == (3, 3, 224, 224)
            assert len(valid_paths) == 3

    def test_skips_invalid_images(self):
        """Test that invalid images are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create one valid image
            valid_path = os.path.join(tmpdir, 'valid.jpg')
            img = Image.new('RGB', (224, 224), color='red')
            img.save(valid_path)

            # Create one invalid file
            invalid_path = os.path.join(tmpdir, 'invalid.jpg')
            with open(invalid_path, 'w') as f:
                f.write('not an image')

            device = torch.device('cpu')

            def preprocess(img):
                return torch.zeros(3, 224, 224)

            batch, valid_paths = load_and_preprocess_batch(
                [valid_path, invalid_path], preprocess, device
            )

            assert batch is not None
            assert batch.shape == (1, 3, 224, 224)
            assert len(valid_paths) == 1
            assert valid_paths[0] == valid_path


@pytest.mark.multimodal_unit
class TestLoadModelFromCheckpoint:
    """Test load_model_from_checkpoint function."""

    def test_raises_for_unsupported_format(self):
        """Test that unsupported formats raise NotImplementedError."""
        # Mock model class (won't be used since format check happens first)
        mock_model_class = type('MockModel', (), {})
        with pytest.raises(NotImplementedError) as excinfo:
            load_model_from_checkpoint('/path/to/model.xyz', None, mock_model_class)
        assert "not supported" in str(excinfo.value)

    def test_raises_for_engine_format(self):
        """Test that engine format raises NotImplementedError with tao-deploy message."""
        # Mock model class (won't be used since format check happens first)
        mock_model_class = type('MockModel', (), {})
        with pytest.raises(NotImplementedError) as excinfo:
            load_model_from_checkpoint('/path/to/model.engine', None, mock_model_class)
        assert "tao-deploy" in str(excinfo.value)


@pytest.mark.multimodal_unit
class TestCLIPVisionEncoder:
    """Test CLIPVisionEncoder wrapper class."""

    @staticmethod
    def _make_mock_model(embed_dim=768):
        """Create a mock CLIP model with logit_scale and logit_bias."""
        class MockModel:
            def __init__(self):
                self.logit_scale = torch.nn.Parameter(torch.ones([]) * 2.3026)
                self.logit_bias = torch.nn.Parameter(torch.ones([]) * -10.0)

            def __call__(self, image=None, text=None):
                batch_size = image.shape[0]
                return {"image_features": torch.randn(batch_size, embed_dim)}

        return MockModel()

    def test_forward_with_dict_output(self):
        """Test forward pass when model returns dict output."""
        encoder = CLIPVisionEncoder(self._make_mock_model())
        dummy_input = torch.randn(2, 3, 224, 224)
        embedding, logit_scale, logit_bias = encoder(dummy_input)

        assert embedding.shape == (2, 768)
        assert logit_scale.shape == ()
        assert logit_bias.shape == ()

    def test_forward_with_tuple_output(self):
        """Test forward pass when model returns tuple output."""
        class MockModel:
            def __init__(self):
                self.logit_scale = torch.nn.Parameter(torch.ones([]) * 2.3026)
                self.logit_bias = torch.nn.Parameter(torch.ones([]) * -10.0)

            def __call__(self, image=None, text=None):
                batch_size = image.shape[0]
                return (torch.randn(batch_size, 768), torch.randn(batch_size, 768))

        encoder = CLIPVisionEncoder(MockModel())
        dummy_input = torch.randn(2, 3, 224, 224)
        embedding, _, _ = encoder(dummy_input)

        assert embedding.shape == (2, 768)

    def test_forward_preserves_batch_size(self):
        """Test that batch size is preserved through forward pass."""
        encoder = CLIPVisionEncoder(self._make_mock_model(embed_dim=512))

        for batch_size in [1, 4, 16]:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            embedding, _, _ = encoder(dummy_input)
            assert embedding.shape[0] == batch_size


@pytest.mark.multimodal_unit
class TestCLIPTextEncoder:
    """Test CLIPTextEncoder wrapper class."""

    def test_forward_with_dict_output(self):
        """Test forward pass when model returns dict output."""
        class MockModel:
            def __init__(self):
                self.logit_scale = torch.nn.Parameter(torch.ones([]) * 2.3026)
                self.logit_bias = torch.nn.Parameter(torch.ones([]) * -10.0)

            def __call__(self, image=None, text=None):
                batch_size = text['input_ids'].shape[0]
                return {"text_features": torch.randn(batch_size, 768)}

        encoder = CLIPTextEncoder(MockModel())
        dummy_input_ids = torch.zeros(2, 64, dtype=torch.long)
        dummy_attention_mask = torch.ones(2, 64, dtype=torch.long)
        embedding, logit_scale, logit_bias = encoder(dummy_input_ids, dummy_attention_mask)

        assert embedding.shape == (2, 768)
        assert logit_scale.shape == ()
        assert logit_bias.shape == ()

    def test_forward_with_tuple_output(self):
        """Test forward pass when model returns tuple output."""
        class MockModel:
            def __init__(self):
                self.logit_scale = torch.nn.Parameter(torch.ones([]) * 2.3026)
                self.logit_bias = torch.nn.Parameter(torch.ones([]) * -10.0)

            def __call__(self, image=None, text=None):
                batch_size = text['input_ids'].shape[0]
                return (torch.randn(batch_size, 768), torch.randn(batch_size, 768))

        encoder = CLIPTextEncoder(MockModel())
        dummy_input_ids = torch.zeros(2, 64, dtype=torch.long)
        dummy_attention_mask = torch.ones(2, 64, dtype=torch.long)
        embedding, _, _ = encoder(dummy_input_ids, dummy_attention_mask)

        assert embedding.shape == (2, 768)


@pytest.mark.multimodal_unit
class TestValidEncoderTypes:
    """Test encoder type validation."""

    def test_valid_encoder_types(self):
        """Test that valid encoder types are defined correctly."""
        assert VALID_ENCODER_TYPES == {'combined', 'separate'}


@pytest.mark.multimodal_unit
class TestLoadTextFile:
    """Test load_text_file function."""

    def test_loads_text_file(self):
        """Test that text file is loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = os.path.join(tmpdir, 'texts.txt')
            with open(text_path, 'w') as f:
                f.write("a photo of a cat\n")
                f.write("a photo of a dog\n")
                f.write("a sunset over the ocean\n")

            texts = load_text_file(text_path)

            assert len(texts) == 3
            assert texts[0] == "a photo of a cat"
            assert texts[2] == "a sunset over the ocean"

    def test_skips_empty_lines(self):
        """Test that empty lines are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = os.path.join(tmpdir, 'texts.txt')
            with open(text_path, 'w') as f:
                f.write("line one\n")
                f.write("\n")
                f.write("   \n")
                f.write("line two\n")

            texts = load_text_file(text_path)

            assert len(texts) == 2
            assert texts[0] == "line one"
            assert texts[1] == "line two"

    def test_handles_unicode(self):
        """Test that unicode text is handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = os.path.join(tmpdir, 'texts.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write("一只猫的照片\n")
                f.write("日落\n")

            texts = load_text_file(text_path)

            assert len(texts) == 2
            assert texts[0] == "一只猫的照片"


@pytest.mark.multimodal_unit
class TestExportFriendlyMHA:
    """Test ExportFriendlyMHA replacement for nn.MultiheadAttention."""

    def test_self_attention(self):
        """Test self-attention case where query, key, value have same seq_len."""
        embed_dim = 256
        num_heads = 8
        seq_len = 16
        batch_size = 2

        # Create a standard MHA module
        mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

        # Create export-friendly replacement
        export_mha = ExportFriendlyMHA(mha)

        # Create input tensors (seq_len, batch, embed_dim)
        query = torch.randn(seq_len, batch_size, embed_dim)
        key = query.clone()
        value = query.clone()

        # Run forward pass
        output, _ = export_mha(query, key, value)

        # Check output shape
        assert output.shape == (seq_len, batch_size, embed_dim)

    def test_self_attention_batch_first(self):
        """Test self-attention with batch_first=True."""
        embed_dim = 256
        num_heads = 8
        seq_len = 16
        batch_size = 2

        # Create MHA with batch_first=True
        mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        export_mha = ExportFriendlyMHA(mha)

        # Create input tensors (batch, seq_len, embed_dim)
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = query.clone()
        value = query.clone()

        output, _ = export_mha(query, key, value)

        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_cross_attention_different_seq_len(self):
        """Test cross/pooler attention where query has different seq_len than key/value."""
        embed_dim = 1152
        num_heads = 16
        q_seq_len = 1  # Single pooling token
        kv_seq_len = 256  # Image patches
        batch_size = 1

        # Create MHA module
        mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        export_mha = ExportFriendlyMHA(mha)

        # Create input tensors with different sequence lengths
        query = torch.randn(q_seq_len, batch_size, embed_dim)
        key = torch.randn(kv_seq_len, batch_size, embed_dim)
        value = torch.randn(kv_seq_len, batch_size, embed_dim)

        # Run forward pass - this should not raise an error
        output, _ = export_mha(query, key, value)

        # Output should have query's sequence length
        assert output.shape == (q_seq_len, batch_size, embed_dim)

    def test_cross_attention_batch_first(self):
        """Test cross/pooler attention with batch_first=True."""
        embed_dim = 768
        num_heads = 12
        q_seq_len = 1
        kv_seq_len = 196  # 14x14 patches
        batch_size = 4

        mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        export_mha = ExportFriendlyMHA(mha)

        # batch_first: (batch, seq_len, embed_dim)
        query = torch.randn(batch_size, q_seq_len, embed_dim)
        key = torch.randn(batch_size, kv_seq_len, embed_dim)
        value = torch.randn(batch_size, kv_seq_len, embed_dim)

        output, _ = export_mha(query, key, value)

        assert output.shape == (batch_size, q_seq_len, embed_dim)

    def test_output_matches_pytorch_mha_self_attention(self):
        """Test that self-attention output matches PyTorch's MHA."""
        embed_dim = 128
        num_heads = 4
        seq_len = 8
        batch_size = 2

        mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        export_mha = ExportFriendlyMHA(mha)

        query = torch.randn(seq_len, batch_size, embed_dim)

        # Compute outputs from both
        with torch.no_grad():
            mha.eval()
            export_mha.eval()
            expected_output, _ = mha(query, query, query)
            actual_output, _ = export_mha(query, query, query)

        # Outputs should be close (allowing for floating point differences)
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-4, atol=1e-4)
