# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for CLIP pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from PIL import Image

from nvidia_tao_pytorch.multimodal.clip.dataloader.custom_loader import (
    ImageTextDataset,
    get_custom_dataloader,
)


class MockCLIPAdapter(nn.Module):
    """Mock CLIP adapter for e2e testing."""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.vision_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, embed_dim),
        )
        self.text_encoder = nn.Linear(64, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        self.tokenizer = lambda x: [torch.randint(0, 100, (64,))]

    def encode_image(self, x, normalize=True):
        features = self.vision_encoder(x)
        if normalize:
            features = nn.functional.normalize(features, dim=-1)
        return features

    def encode_text(self, x, normalize=True):
        if isinstance(x, dict):
            x = x['input_ids']
        if x.ndim == 1:
            x = x.unsqueeze(0)
        features = self.text_encoder(x.float())
        if normalize:
            features = nn.functional.normalize(features, dim=-1)
        return features

    def forward(self, image, text):
        img_features = self.encode_image(image)
        txt_features = self.encode_text(text)
        return img_features, txt_features, self.logit_scale.exp(), self.logit_bias

    def vision_named_parameters(self):
        for name, param in self.vision_encoder.named_parameters():
            yield f'vision_encoder.{name}', param

    def text_named_parameters(self):
        for name, param in self.text_encoder.named_parameters():
            yield f'text_encoder.{name}', param

    def other_named_parameters(self):
        yield 'logit_scale', self.logit_scale
        yield 'logit_bias', self.logit_bias

    def set_grad_checkpointing(self, enable=True):
        pass


@pytest.fixture
def temp_image_dataset():
    """Create a temporary image-text dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        image_dir = tmpdir / "images"
        label_dir = tmpdir / "labels"
        image_dir.mkdir()
        label_dir.mkdir()

        # Create sample images and captions
        for i in range(20):
            img = Image.new('RGB', (64, 64), color=(i * 10 % 256, i * 20 % 256, i * 30 % 256))
            img.save(image_dir / f"img_{i}.jpg")
            (label_dir / f"img_{i}.txt").write_text(f"A sample image number {i}")

        image_list = tmpdir / "train.txt"
        image_list.write_text("\n".join([f"img_{i}.jpg" for i in range(20)]))

        yield {
            'image_dir': str(image_dir),
            'caption_dir': str(label_dir),
            'image_list_file': str(image_list),
        }


@pytest.mark.multimodal_integration
class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    def test_dataloader_to_model(self, temp_image_dataset):
        """Test data flows from dataloader through model."""
        # Create transforms and tokenizer
        def simple_transform(img):
            img = img.resize((64, 64))
            return torch.tensor([list(img.getdata())]).reshape(3, 64, 64).float() / 255.0

        def simple_tokenizer(text):
            return [torch.randint(0, 100, (64,))]

        # Create dataloader
        dataloader = get_custom_dataloader(
            datasets=[temp_image_dataset],
            batch_size=4,
            transform=simple_transform,
            tokenizer=simple_tokenizer,
            num_workers=0,
            mode='train',
        )

        # Create model
        adapter = MockCLIPAdapter()

        # Run one batch through model
        batch = next(iter(dataloader))
        images, texts = batch

        img_features, txt_features, scale, bias = adapter(images, texts)

        assert img_features.shape == (4, 128)
        assert txt_features.shape == (4, 128)

    def test_full_training_with_real_dataloader(self, temp_image_dataset):
        """Test full training loop with real dataloader."""
        def simple_transform(img):
            img = img.resize((64, 64))
            return torch.tensor([list(img.getdata())]).reshape(3, 64, 64).float() / 255.0

        def simple_tokenizer(text):
            return [torch.randint(0, 100, (64,))]

        train_loader = get_custom_dataloader(
            datasets=[temp_image_dataset],
            batch_size=4,
            transform=simple_transform,
            tokenizer=simple_tokenizer,
            num_workers=0,
            mode='train',
        )

        adapter = MockCLIPAdapter()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

        # Training loop
        losses = []
        for i, batch in enumerate(train_loader):
            if i >= 5:
                break
            images, texts = batch
            optimizer.zero_grad()

            img_features, txt_features, scale, bias = adapter(images, texts)
            logits = scale * (img_features @ txt_features.T) + bias
            targets = torch.eye(images.shape[0])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert len(losses) == 5
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)


@pytest.mark.multimodal_integration
class TestInferenceFlow:
    """Test inference flow."""

    def test_image_encoding(self):
        """Test image encoding for inference."""
        adapter = MockCLIPAdapter()

        adapter.eval()
        with torch.no_grad():
            images = torch.randn(4, 3, 64, 64)
            features = adapter.encode_image(images)

        assert features.shape == (4, 128)

    def test_text_encoding(self):
        """Test text encoding for inference."""
        adapter = MockCLIPAdapter()

        adapter.eval()
        with torch.no_grad():
            texts = torch.randint(0, 100, (4, 64))
            features = adapter.encode_text(texts)

        assert features.shape == (4, 128)

    def test_similarity_computation(self):
        """Test image-text similarity computation."""
        adapter = MockCLIPAdapter()

        adapter.eval()
        with torch.no_grad():
            images = torch.randn(4, 3, 64, 64)
            texts = torch.randint(0, 100, (4, 64))

            img_features, txt_features, scale, bias = adapter(images, texts)

            # Compute similarity
            similarity = scale * (img_features @ txt_features.T) + bias

        assert similarity.shape == (4, 4)
        assert torch.isfinite(similarity).all()


@pytest.mark.multimodal_integration
class TestMultipleDatasets:
    """Test with multiple datasets."""

    def test_combined_datasets(self):
        """Test training with multiple combined datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create two datasets
            datasets_configs = []
            for ds_idx in range(2):
                ds_dir = tmpdir / f"dataset_{ds_idx}"
                image_dir = ds_dir / "images"
                label_dir = ds_dir / "labels"
                image_dir.mkdir(parents=True)
                label_dir.mkdir()

                for i in range(10):
                    img = Image.new('RGB', (64, 64), color=(ds_idx * 100, i * 20, 100))
                    img.save(image_dir / f"img_{i}.jpg")
                    (label_dir / f"img_{i}.txt").write_text(f"Dataset {ds_idx} image {i}")

                image_list = ds_dir / "train.txt"
                image_list.write_text("\n".join([f"img_{i}.jpg" for i in range(10)]))

                datasets_configs.append({
                    'image_dir': str(image_dir),
                    'caption_dir': str(label_dir),
                    'image_list_file': str(image_list),
                })

            def simple_transform(img):
                img = img.resize((64, 64))
                return torch.tensor([list(img.getdata())]).reshape(3, 64, 64).float() / 255.0

            def simple_tokenizer(text):
                return [torch.randint(0, 100, (64,))]

            dataloader = get_custom_dataloader(
                datasets=datasets_configs,
                batch_size=4,
                transform=simple_transform,
                tokenizer=simple_tokenizer,
                num_workers=0,
                mode='train',
            )

            # Should have combined samples
            assert len(dataloader.dataset) == 20

            adapter = MockCLIPAdapter()

            batch = next(iter(dataloader))
            images, texts = batch
            img_features, txt_features, scale, bias = adapter(images, texts)

            logits = scale * (img_features @ txt_features.T) + bias
            targets = torch.eye(images.shape[0])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)

            assert torch.isfinite(loss)
