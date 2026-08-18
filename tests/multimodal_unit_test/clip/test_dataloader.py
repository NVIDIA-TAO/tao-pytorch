# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CLIP custom dataloader."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import nvidia_tao_pytorch.multimodal.clip.dataloader.custom_loader as custom_loader
import nvidia_tao_pytorch.multimodal.clip.dataloader.pl_clip_data_module as clip_data_module
from nvidia_tao_pytorch.multimodal.clip.dataloader.pl_clip_data_module import (
    CLIPDataModule,
)
from nvidia_tao_pytorch.multimodal.clip.dataloader.custom_loader import (
    ImageTextDataset,
    get_custom_dataloader,
)
from nvidia_tao_pytorch.multimodal.clip.dataloader.sampler import (
    BalancedByQueryTypeSampler,
    BalancedUniqueCaptionBatchSampler,
)


@pytest.fixture
def temp_dataset():
    """Create a temporary dataset with images and text files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        image_dir = tmpdir / "images"
        label_dir = tmpdir / "labels"
        image_dir.mkdir()
        label_dir.mkdir()

        # Create sample images and corresponding text files
        for i in range(5):
            img = Image.new('RGB', (64, 64), color=(i * 50, i * 50, i * 50))
            img_path = image_dir / f"image_{i}.jpg"
            img.save(img_path)

            text_path = label_dir / f"image_{i}.txt"
            text_path.write_text(f"Caption for image {i}")

        # Create image list file
        image_list_file = tmpdir / "image_list.txt"
        image_list_file.write_text("\n".join([f"image_{i}.jpg" for i in range(5)]))

        yield {
            'image_dir': str(image_dir),
            'caption_dir': str(label_dir),
            'image_list_file': str(image_list_file),
            'caption_file_suffix': '.txt',
            'tmpdir': tmpdir,
            'image_dir_path': image_dir,
            'caption_dir_path': label_dir,
        }


def _make_attribute_pair(idx, width=7, query_type="easy"):
    """Build one split-pairs metadata row with scalar attribute vectors."""
    return {
        "query_type": query_type,
        "caption": f"Caption for image {idx}",
        "unique_name": f"image_{idx}.jpg",
        "image_attr_values": list(range(idx, idx + width)),
        "text_attr_values": [idx] + [-1] + list(range(idx + 2, idx + width)),
    }


def _write_attribute_pairs(path, count, width=7):
    """Write count aligned metadata rows to a pairs JSON file."""
    rows = [
        _make_attribute_pair(
            idx=i,
            width=width,
            query_type="easy" if i % 2 == 0 else "medium",
        )
        for i in range(count)
    ]
    path.write_text(json.dumps(rows))
    return rows


def _write_accessory_vocab(path):
    """Write a small accessory vocabulary with zero reserved for padding."""
    path.write_text(json.dumps({
        "unknown_id": 0,
        "value_to_id": {
            "__unknown__": 0,
            "black backpack": 11,
            "red hat": 12,
        },
    }))


def _write_attribute_vocab(
    path,
    attributes,
    value_to_id,
    indent=None,
):
    """Write an ordered scalar-attribute vocabulary."""
    path.write_text(json.dumps(
        {
            "attributes": attributes,
            "value_to_id": value_to_id,
        },
        indent=indent,
    ))


def _attribute_dataset_config(temp_dataset, pairs_file):
    """Build one custom dataset config with scalar metadata."""
    return {
        'image_dir': temp_dataset['image_dir'],
        'caption_dir': temp_dataset['caption_dir'],
        'image_list_file': temp_dataset['image_list_file'],
        'caption_file_suffix': '.txt',
        'train_pairs_file': str(pairs_file),
    }


def _validation_attribute_dataset_config(temp_dataset, pairs_file):
    """Build one validation dataset config with scalar metadata."""
    config = _attribute_dataset_config(temp_dataset, pairs_file)
    config.pop("train_pairs_file")
    config["attribute_pairs_file"] = str(pairs_file)
    return config


def _write_attribute_source(
    tmp_path,
    name,
    attributes,
    value_to_id,
    write_vocab=True,
    indent=None,
):
    """Write one pairs/vocabulary source for multi-dataset tests."""
    source_dir = tmp_path / name
    source_dir.mkdir()
    pairs_file = source_dir / "train_pairs.json"
    _write_attribute_pairs(
        pairs_file,
        count=5,
        width=len(attributes),
    )
    if write_vocab:
        _write_attribute_vocab(
            source_dir / "attribute_vocab.json",
            attributes,
            value_to_id,
            indent=indent,
        )
    return pairs_file


@pytest.mark.multimodal_unit
class TestImageTextDataset:
    """Test ImageTextDataset class."""

    def test_initialization(self, temp_dataset):
        """Test dataset initialization with valid config."""
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
        }]

        dataset = ImageTextDataset(datasets=dataset_config, mode='train')

        assert len(dataset) == 5

    def test_getitem_returns_image_and_text(self, temp_dataset):
        """Test that __getitem__ returns image and text."""
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
        }]

        dataset = ImageTextDataset(datasets=dataset_config, mode='train')

        image, text = dataset[0]

        assert isinstance(image, Image.Image)
        assert isinstance(text, str)
        assert "Caption for image" in text

    def test_transform_applied(self, temp_dataset):
        """Test that transform is applied to images."""
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
        }]

        def dummy_transform(img):
            return torch.zeros(3, 32, 32)

        dataset = ImageTextDataset(
            datasets=dataset_config,
            transform=dummy_transform,
            mode='train'
        )

        image, text = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 32, 32)

    def test_tokenizer_applied(self, temp_dataset):
        """Test that tokenizer is applied to text."""
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
        }]

        def dummy_tokenizer(text):
            return [torch.tensor([1, 2, 3])]

        dataset = ImageTextDataset(
            datasets=dataset_config,
            tokenizer=dummy_tokenizer,
            mode='train'
        )

        image, text = dataset[0]

        assert isinstance(text, torch.Tensor)
        assert text.tolist() == [1, 2, 3]

    def test_multiple_datasets(self, temp_dataset):
        """Test initialization with multiple dataset configs."""
        # Create a second dataset
        image_dir2 = temp_dataset['tmpdir'] / "images2"
        caption_dir2 = temp_dataset['tmpdir'] / "labels2"
        image_dir2.mkdir()
        caption_dir2.mkdir()

        for i in range(3):
            img = Image.new('RGB', (64, 64), color=(100, 100, 100))
            img_path = image_dir2 / f"img2_{i}.jpg"
            img.save(img_path)

            text_path = caption_dir2 / f"img2_{i}.txt"
            text_path.write_text(f"Caption for second dataset {i}")

        image_list_file2 = temp_dataset['tmpdir'] / "image_list2.txt"
        image_list_file2.write_text("\n".join([f"img2_{i}.jpg" for i in range(3)]))

        dataset_configs = [
            {
                'image_dir': temp_dataset['image_dir'],
                'caption_dir': temp_dataset['caption_dir'],
                'image_list_file': temp_dataset['image_list_file'],
                'caption_file_suffix': '.txt',
            },
            {
                'image_dir': str(image_dir2),
                'caption_dir': str(caption_dir2),
                'image_list_file': str(image_list_file2),
                'caption_file_suffix': '.txt',
            },
        ]

        dataset = ImageTextDataset(datasets=dataset_configs, mode='train')

        assert len(dataset) == 8  # 5 + 3

    def test_zero_shot_eval_single_dataset(self, temp_dataset):
        """Test zero-shot eval mode with single dataset."""
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
        }]

        mapping = {"Caption for image 0": "mapped_text_0"}

        dataset = ImageTextDataset(
            datasets=dataset_config,
            zero_shot_eval=True,
            mapping=mapping,
            mode='val'
        )

        image, text = dataset[0]
        assert text == "mapped_text_0"

    def test_zero_shot_eval_multiple_datasets_raises(self, temp_dataset):
        """Test that zero-shot eval with multiple datasets raises error."""
        dataset_configs = [
            {
                'image_dir': temp_dataset['image_dir'],
                'caption_dir': temp_dataset['caption_dir'],
                'image_list_file': temp_dataset['image_list_file'],
                'caption_file_suffix': '.txt',
            },
            {
                'image_dir': temp_dataset['image_dir'],
                'caption_dir': temp_dataset['caption_dir'],
                'image_list_file': temp_dataset['image_list_file'],
                'caption_file_suffix': '.txt',
            },
        ]

        with pytest.raises(NotImplementedError):
            ImageTextDataset(
                datasets=dataset_configs,
                zero_shot_eval=True,
                mapping={},
                mode='val'
            )

    def test_empty_dataset_raises(self):
        """Test that empty dataset raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_dir = tmpdir / "images"
            image_dir.mkdir()

            image_list_file = tmpdir / "image_list.txt"
            image_list_file.write_text("")  # Empty list

            dataset_config = [{
                'image_dir': str(image_dir),
                'image_list_file': str(image_list_file),
            }]

            with pytest.raises(ValueError, match="No valid image-text pairs"):
                ImageTextDataset(datasets=dataset_config, mode='train')

    def test_glob_fallback_without_image_list(self, temp_dataset):
        """Test that dataset globs for images without image_list_file."""
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'caption_file_suffix': '.txt',
        }]

        dataset = ImageTextDataset(datasets=dataset_config, mode='train')

        assert len(dataset) == 5

    def test_attribute_metadata_returns_third_item(self, temp_dataset):
        """Test optional attribute metadata is returned with a sample."""
        pairs_file = temp_dataset['tmpdir'] / "train_pairs.json"
        rows = _write_attribute_pairs(pairs_file, count=5, width=7)
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
            'train_pairs_file': str(pairs_file),
        }]

        dataset = ImageTextDataset(
            datasets=dataset_config,
            mode='train',
            include_attribute_metadata=True,
        )

        image, text, metadata = dataset[0]

        assert isinstance(image, Image.Image)
        assert "Caption for image" in text
        assert set(metadata) == {"image_attr_values", "text_attr_values"}
        assert metadata["image_attr_values"].dtype == torch.long
        assert metadata["text_attr_values"].dtype == torch.long
        assert metadata["image_attr_values"].shape == (7,)
        assert metadata["text_attr_values"].shape == (7,)
        assert metadata["text_attr_values"].tolist() == rows[0]["text_attr_values"]
        assert metadata["text_attr_values"][1].item() == -1

    def test_attribute_metadata_warns_without_vocab(
        self, temp_dataset, monkeypatch
    ):
        """Test missing normalization vocabulary is surfaced to users."""
        warnings = []
        monkeypatch.setattr(
            custom_loader.logging,
            "warning",
            lambda *args: warnings.append(args),
        )
        pairs_file = temp_dataset['tmpdir'] / "train_pairs.json"
        _write_attribute_pairs(pairs_file, count=5, width=7)

        ImageTextDataset(
            datasets=[_attribute_dataset_config(temp_dataset, pairs_file)],
            mode='train',
            include_attribute_metadata=True,
        )

        assert len(warnings) == 1
        message, warned_pairs_file, warned_vocab_path = warnings[0]
        assert "not visible" in message
        assert warned_pairs_file == pairs_file
        assert warned_vocab_path == (
            temp_dataset['tmpdir'] / "attribute_vocab.json"
        )

    def test_attribute_metadata_includes_padded_accessory_ids(self, temp_dataset):
        """Test accessory lists are padded and returned with scalars."""
        pairs_file = temp_dataset['tmpdir'] / "train_pairs.json"
        rows = _write_attribute_pairs(pairs_file, count=5, width=7)
        image_accessories = [[11, 12], [11], [], [12], [11, 12]]
        text_accessories = [[11], [], [], [12], [11, 12]]
        for row, image_ids, text_ids in zip(
            rows, image_accessories, text_accessories
        ):
            row["image_accessory_ids"] = image_ids
            row["text_accessory_ids"] = text_ids
        pairs_file.write_text(json.dumps(rows))
        _write_accessory_vocab(temp_dataset['tmpdir'] / "accessory_vocab.json")
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
            'train_pairs_file': str(pairs_file),
        }]

        dataset = ImageTextDataset(
            datasets=dataset_config,
            mode='train',
            include_attribute_metadata=True,
        )

        _, _, first = dataset[0]
        _, _, second = dataset[1]
        assert set(first) == {
            "image_attr_values",
            "text_attr_values",
            "image_accessory_ids",
            "text_accessory_ids",
        }
        assert first["image_accessory_ids"].tolist() == [11, 12]
        assert first["text_accessory_ids"].tolist() == [11, 0]
        assert second["image_accessory_ids"].tolist() == [11, 0]

    def test_accessory_metadata_requires_paired_subset(self, temp_dataset):
        """Test a query cannot require an accessory absent from its image."""
        pairs_file = temp_dataset['tmpdir'] / "train_pairs.json"
        rows = _write_attribute_pairs(pairs_file, count=5, width=7)
        for row in rows:
            row["image_accessory_ids"] = [11]
            row["text_accessory_ids"] = [11]
        rows[0]["text_accessory_ids"] = [12]
        pairs_file.write_text(json.dumps(rows))
        _write_accessory_vocab(temp_dataset['tmpdir'] / "accessory_vocab.json")
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
            'train_pairs_file': str(pairs_file),
        }]

        with pytest.raises(ValueError, match="not contained"):
            ImageTextDataset(
                datasets=dataset_config,
                mode='train',
                include_attribute_metadata=True,
            )

    def test_attribute_metadata_treats_not_visible_as_missing(self, temp_dataset):
        """Test not-visible vocab IDs are normalized to wildcard values."""
        pairs_file = temp_dataset['tmpdir'] / "train_pairs.json"
        rows = _write_attribute_pairs(pairs_file, count=5, width=2)
        rows[0]["image_attr_values"] = [8, 5]
        rows[0]["text_attr_values"] = [8, 5]
        pairs_file.write_text(json.dumps(rows))
        vocab = {
            "attributes": ["top outer color", "top outer type"],
            "value_to_id": {
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
            },
        }
        (temp_dataset['tmpdir'] / "attribute_vocab.json").write_text(
            json.dumps(vocab)
        )
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
            'train_pairs_file': str(pairs_file),
        }]

        dataset = ImageTextDataset(
            datasets=dataset_config,
            mode='train',
            include_attribute_metadata=True,
        )

        _, _, metadata = dataset[0]
        assert metadata["image_attr_values"].tolist() == [-1, -1]
        assert metadata["text_attr_values"].tolist() == [-1, -1]

    def test_attribute_metadata_missing_field_raises(self, temp_dataset):
        """Test missing attribute metadata fields fail clearly."""
        pairs_file = temp_dataset['tmpdir'] / "train_pairs.json"
        rows = [_make_attribute_pair(i) for i in range(5)]
        rows[0].pop("text_attr_values")
        pairs_file.write_text(json.dumps(rows))
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
            'train_pairs_file': str(pairs_file),
        }]

        with pytest.raises(ValueError, match="missing 'text_attr_values'"):
            ImageTextDataset(
                datasets=dataset_config,
                mode='train',
                include_attribute_metadata=True,
            )

    def test_attribute_metadata_mismatched_length_raises(self, temp_dataset):
        """Test pairs/list length mismatch fails when metadata is requested."""
        pairs_file = temp_dataset['tmpdir'] / "train_pairs.json"
        _write_attribute_pairs(pairs_file, count=4, width=7)
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
            'train_pairs_file': str(pairs_file),
        }]

        with pytest.raises(ValueError, match="has 4 items but image list has 5"):
            ImageTextDataset(
                datasets=dataset_config,
                mode='train',
                include_attribute_metadata=True,
            )

    def test_attribute_metadata_infers_val_pairs_file(self, temp_dataset):
        """Test validation infers val_pairs.json from val_list.txt."""
        val_list_file = temp_dataset['tmpdir'] / "val_list.txt"
        val_list_file.write_text(
            "\n".join([f"image_{i}.jpg" for i in range(5)])
        )
        val_pairs_file = temp_dataset['tmpdir'] / "val_pairs.json"
        _write_attribute_pairs(val_pairs_file, count=5, width=7)
        _write_attribute_vocab(
            temp_dataset['tmpdir'] / "attribute_vocab.json",
            attributes=[f"attribute_{i}" for i in range(7)],
            value_to_id={},
        )
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': str(val_list_file),
            'caption_file_suffix': '.txt',
        }]

        dataset = ImageTextDataset(
            datasets=dataset_config,
            mode='val',
            include_attribute_metadata=True,
        )

        _, _, metadata = dataset[0]
        assert metadata["image_attr_values"].shape == (7,)
        assert metadata["text_attr_values"].shape == (7,)
        assert metadata["pas_row_index"].item() == 0

    def test_validation_attribute_metadata_rejects_reordered_pairs(
        self, temp_dataset
    ):
        """Test equal-length validation metadata cannot change row order."""
        pairs_file = temp_dataset['tmpdir'] / "val_pairs.json"
        rows = _write_attribute_pairs(pairs_file, count=5, width=7)
        rows[0], rows[1] = rows[1], rows[0]
        pairs_file.write_text(json.dumps(rows))

        with pytest.raises(
            ValueError,
            match="unique_name .* does not match image list entry",
        ):
            ImageTextDataset(
                datasets=[
                    _validation_attribute_dataset_config(
                        temp_dataset,
                        pairs_file,
                    )
                ],
                mode='val',
                include_attribute_metadata=True,
            )

    def test_validation_attribute_metadata_rejects_stale_caption(
        self, temp_dataset
    ):
        """Test validation JSON captions must match caption-file contents."""
        pairs_file = temp_dataset['tmpdir'] / "val_pairs.json"
        _write_attribute_pairs(pairs_file, count=5, width=7)
        (
            temp_dataset['caption_dir_path'] / "image_0.txt"
        ).write_text("Stale caption")

        with pytest.raises(
            ValueError,
            match="caption .* does not match caption file",
        ):
            ImageTextDataset(
                datasets=[
                    _validation_attribute_dataset_config(
                        temp_dataset,
                        pairs_file,
                    )
                ],
                mode='val',
                include_attribute_metadata=True,
            )

    def test_attribute_metadata_inconsistent_width_raises(self, temp_dataset):
        """Test inconsistent attribute vector widths fail clearly."""
        pairs_file = temp_dataset['tmpdir'] / "train_pairs.json"
        rows = [_make_attribute_pair(i, width=7) for i in range(5)]
        rows[1]["text_attr_values"].append(99)
        pairs_file.write_text(json.dumps(rows))
        dataset_config = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
            'train_pairs_file': str(pairs_file),
        }]

        with pytest.raises(ValueError, match="expected 7"):
            ImageTextDataset(
                datasets=dataset_config,
                mode='train',
                include_attribute_metadata=True,
            )

    @pytest.mark.parametrize("invalid_id", [1.9, "2", True])
    def test_attribute_metadata_rejects_non_integer_ids(
        self, temp_dataset, invalid_id
    ):
        """Test scalar attribute IDs are not silently coerced to integers."""
        pairs_file = temp_dataset['tmpdir'] / "train_pairs.json"
        rows = [_make_attribute_pair(i, width=2) for i in range(5)]
        rows[0]["image_attr_values"][0] = invalid_id
        pairs_file.write_text(json.dumps(rows))

        with pytest.raises(ValueError, match="must contain integers"):
            ImageTextDataset(
                datasets=[
                    _attribute_dataset_config(temp_dataset, pairs_file)
                ],
                mode='train',
                include_attribute_metadata=True,
            )

    def test_multi_dataset_attribute_metadata_accepts_matching_vocabularies(
        self, temp_dataset, tmp_path
    ):
        """Test equivalent ordered vocabularies can share one training set."""
        attributes = ["upper color", "lower color"]
        value_to_id = {
            "upper color": {"__missing__": 0, "black": 1},
            "lower color": {"__missing__": 0, "blue": 2},
        }
        first_pairs = _write_attribute_source(
            tmp_path,
            "first",
            attributes,
            value_to_id,
        )
        second_pairs = _write_attribute_source(
            tmp_path,
            "second",
            attributes,
            dict(reversed(list(value_to_id.items()))),
            indent=2,
        )

        dataset = ImageTextDataset(
            datasets=[
                _attribute_dataset_config(temp_dataset, first_pairs),
                _attribute_dataset_config(temp_dataset, second_pairs),
            ],
            mode='train',
            include_attribute_metadata=True,
        )

        assert len(dataset) == 10

    def test_multi_dataset_val_metadata_uses_global_row_indices(
        self, temp_dataset, tmp_path
    ):
        """Test matching vocabularies preserve concatenated validation order."""
        attributes = ["upper color", "lower color"]
        value_to_id = {
            "upper color": {"__missing__": 0, "black": 1},
            "lower color": {"__missing__": 0, "blue": 2},
        }
        first_pairs = _write_attribute_source(
            tmp_path,
            "first",
            attributes,
            value_to_id,
        )
        second_pairs = _write_attribute_source(
            tmp_path,
            "second",
            attributes,
            dict(reversed(list(value_to_id.items()))),
            indent=2,
        )

        dataset = ImageTextDataset(
            datasets=[
                _validation_attribute_dataset_config(
                    temp_dataset, first_pairs
                ),
                _validation_attribute_dataset_config(
                    temp_dataset, second_pairs
                ),
            ],
            mode='val',
            include_attribute_metadata=True,
        )

        assert len(dataset) == 10
        assert dataset[0][2]["pas_row_index"].item() == 0
        assert dataset[5][2]["pas_row_index"].item() == 5
        assert dataset[9][2]["pas_row_index"].item() == 9

    def test_multi_dataset_attribute_metadata_requires_each_vocab(
        self, temp_dataset, tmp_path
    ):
        """Test every scalar metadata source provides its vocabulary."""
        attributes = ["upper color", "lower color"]
        value_to_id = {
            "upper color": {"__missing__": 0, "black": 1},
            "lower color": {"__missing__": 0, "blue": 2},
        }
        first_pairs = _write_attribute_source(
            tmp_path,
            "first",
            attributes,
            value_to_id,
        )
        second_pairs = _write_attribute_source(
            tmp_path,
            "second",
            attributes,
            value_to_id,
            write_vocab=False,
        )

        with pytest.raises(
            ValueError,
            match="Multi-dataset attribute metadata requires",
        ):
            ImageTextDataset(
                datasets=[
                    _attribute_dataset_config(temp_dataset, first_pairs),
                    _attribute_dataset_config(temp_dataset, second_pairs),
                ],
                mode='train',
                include_attribute_metadata=True,
            )

    def test_multi_dataset_attribute_metadata_rejects_reordered_vocab(
        self, temp_dataset, tmp_path
    ):
        """Test equal-width sources cannot change attribute column order."""
        attributes = ["upper color", "lower color"]
        value_to_id = {
            "upper color": {"__missing__": 0, "black": 1},
            "lower color": {"__missing__": 0, "blue": 2},
        }
        first_pairs = _write_attribute_source(
            tmp_path,
            "first",
            attributes,
            value_to_id,
        )
        second_pairs = _write_attribute_source(
            tmp_path,
            "second",
            list(reversed(attributes)),
            value_to_id,
        )

        with pytest.raises(
            ValueError,
            match="same ordered attribute vocabulary",
        ):
            ImageTextDataset(
                datasets=[
                    _attribute_dataset_config(temp_dataset, first_pairs),
                    _attribute_dataset_config(temp_dataset, second_pairs),
                ],
                mode='train',
                include_attribute_metadata=True,
            )

    def test_multi_dataset_attribute_metadata_rejects_different_ids(
        self, temp_dataset, tmp_path
    ):
        """Test equal attribute order cannot use different value IDs."""
        attributes = ["upper color", "lower color"]
        first_pairs = _write_attribute_source(
            tmp_path,
            "first",
            attributes,
            {
                "upper color": {"__missing__": 0, "black": 1},
                "lower color": {"__missing__": 0, "blue": 2},
            },
        )
        second_pairs = _write_attribute_source(
            tmp_path,
            "second",
            attributes,
            {
                "upper color": {"__missing__": 0, "black": 3},
                "lower color": {"__missing__": 0, "blue": 2},
            },
        )

        with pytest.raises(
            ValueError,
            match="same ordered attribute vocabulary",
        ):
            ImageTextDataset(
                datasets=[
                    _attribute_dataset_config(temp_dataset, first_pairs),
                    _attribute_dataset_config(temp_dataset, second_pairs),
                ],
                mode='train',
                include_attribute_metadata=True,
            )

    def test_multi_dataset_val_metadata_rejects_different_accessory_vocab_json(
        self, temp_dataset, tmp_path
    ):
        """Test validation requires identical accessory vocabulary JSON."""
        attributes = ["upper color", "lower color"]
        value_to_id = {
            "upper color": {"__missing__": 0, "black": 1},
            "lower color": {"__missing__": 0, "blue": 2},
        }
        pair_files = [
            _write_attribute_source(
                tmp_path,
                name,
                attributes,
                value_to_id,
            )
            for name in ("first", "second")
        ]
        for source_index, pairs_file in enumerate(pair_files):
            rows = json.loads(pairs_file.read_text())
            for row in rows:
                row["image_accessory_ids"] = [11]
                row["text_accessory_ids"] = [11]
            pairs_file.write_text(json.dumps(rows))
            accessory_vocab = {
                "unknown_id": 0,
                "value_to_id": {
                    "__unknown__": 0,
                    "black backpack": 11,
                },
                "source": source_index,
            }
            (pairs_file.parent / "accessory_vocab.json").write_text(
                json.dumps(accessory_vocab)
            )

        with pytest.raises(
            ValueError,
            match="same accessory vocabulary",
        ):
            ImageTextDataset(
                datasets=[
                    _validation_attribute_dataset_config(
                        temp_dataset, pairs_file
                    )
                    for pairs_file in pair_files
                ],
                mode='val',
                include_attribute_metadata=True,
            )


@pytest.mark.multimodal_unit
class TestGetCustomDataloader:
    """Test get_custom_dataloader function."""

    def test_creates_dataloader(self, temp_dataset):
        """Test that dataloader is created successfully."""
        dataset_configs = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
        }]

        dataloader = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            num_workers=0,
            mode='train'
        )

        assert dataloader is not None
        assert len(dataloader) > 0

    def test_batch_size_respected(self, temp_dataset):
        """Test that batch size is respected."""
        dataset_configs = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
        }]

        def dummy_transform(img):
            return torch.zeros(3, 32, 32)

        dataloader = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            transform=dummy_transform,
            num_workers=0,
            mode='train'
        )

        batch = next(iter(dataloader))
        images, texts = batch
        assert len(images) == 2

    def test_val_mode_no_batch_sampler(self, temp_dataset):
        """Test that val mode doesn't use batch sampler."""
        dataset_configs = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
        }]

        def dummy_transform(img):
            return torch.zeros(3, 32, 32)

        dataloader = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            transform=dummy_transform,
            num_workers=0,
            shuffle=False,
            mode='val'
        )

        assert dataloader is not None

    def test_seed_reproducibility(self, temp_dataset):
        """Test that seed produces reproducible results."""
        dataset_configs = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
        }]

        def dummy_transform(img):
            return torch.zeros(3, 32, 32)

        dataloader1 = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            transform=dummy_transform,
            num_workers=0,
            seed=42,
            mode='train'
        )

        dataloader2 = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            transform=dummy_transform,
            num_workers=0,
            seed=42,
            mode='train'
        )

        # Both should have same length
        assert len(dataloader1) == len(dataloader2)

    def test_transform_and_tokenizer_passed(self, temp_dataset):
        """Test that transform and tokenizer are passed to dataset."""
        dataset_configs = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
        }]

        def dummy_transform(img):
            return torch.zeros(3, 32, 32)

        def dummy_tokenizer(text):
            return [torch.tensor([1, 2, 3])]

        dataloader = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            transform=dummy_transform,
            tokenizer=dummy_tokenizer,
            num_workers=0,
            mode='train'
        )

        images, texts = next(iter(dataloader))
        assert images.shape == (2, 3, 32, 32)
        assert texts.shape == (2, 3)

    def test_balanced_query_type_sampler_validates_metadata_length(self):
        """Test that balanced sampler raises a useful error for bad metadata."""
        with pytest.raises(ValueError, match="must match num_samples"):
            BalancedByQueryTypeSampler(
                num_samples=3,
                query_type_per_index=["easy", "hard"],
            )

    def test_balanced_query_type_sampler_uses_set_epoch(self):
        """Test that set_epoch changes rank partitioning deterministically."""
        sampler = BalancedByQueryTypeSampler(
            num_samples=10,
            query_type_per_index=["easy"] * 5 + ["hard"] * 5,
            num_replicas=2,
            rank=0,
            seed=7,
        )

        epoch0_indices = sampler._rank_indices()
        sampler.set_epoch(1)
        epoch1_indices = sampler._rank_indices()

        assert epoch0_indices != epoch1_indices
        assert sampler.epoch == 1

    def test_unique_caption_batch_sampler_keeps_epoch_external(self):
        """Test unique-caption batches and external epoch control."""
        sampler = BalancedUniqueCaptionBatchSampler(
            num_samples=4,
            query_type_per_index=["easy", "easy", "hard", "hard"],
            caption_per_index=["same", "easy-only", "same", "hard-only"],
            batch_size=2,
            seed=13,
            query_types_order=("easy", "hard"),
        )

        sampler.set_epoch(3)
        batches = list(sampler)

        assert sampler.epoch == 3
        assert batches
        for batch in batches:
            captions = [
                sampler.caption_per_index[idx]
                for idx in batch
            ]
            query_types = {
                sampler.query_type_per_index[idx]
                for idx in batch
            }
            assert len(batch) == 2
            assert len(captions) == len(set(captions))
            assert query_types == {"easy", "hard"}

    def test_balanced_metadata_reads_each_dataset_train_pairs_file(
        self, temp_dataset, tmp_path
    ):
        """Test per-dataset train_pairs_file handling for multi-dataset training."""
        second_image_dir = tmp_path / "images_2"
        second_caption_dir = tmp_path / "labels_2"
        second_image_dir.mkdir()
        second_caption_dir.mkdir()
        for i in range(2):
            image_name = f"second_{i}.jpg"
            Image.new("RGB", (32, 32)).save(second_image_dir / image_name)
            (second_caption_dir / f"second_{i}.txt").write_text(
                f"Second caption {i}"
            )

        second_image_list = tmp_path / "second_list.txt"
        second_image_list.write_text("second_0.jpg\nsecond_1.jpg\n")

        first_pairs_file = tmp_path / "first_pairs.json"
        first_pairs_file.write_text(json.dumps([
            {
                "query_type": "easy" if i % 2 == 0 else "hard",
                "caption": f"Caption for image {i}",
            }
            for i in range(5)
        ]))
        second_pairs_file = tmp_path / "second_pairs.json"
        second_pairs_file.write_text(json.dumps([
            {"query_type": "medium", "caption": "Second caption 0"},
            {"query_type": "hard", "caption": "Second caption 1"},
        ]))

        dataset_configs = [
            {
                "image_dir": temp_dataset["image_dir"],
                "caption_dir": temp_dataset["caption_dir"],
                "image_list_file": temp_dataset["image_list_file"],
                "caption_file_suffix": ".txt",
                "train_pairs_file": str(first_pairs_file),
            },
            {
                "image_dir": str(second_image_dir),
                "caption_dir": str(second_caption_dir),
                "image_list_file": str(second_image_list),
                "caption_file_suffix": ".txt",
                "train_pairs_file": str(second_pairs_file),
            },
        ]

        dataloader = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            num_workers=0,
            mode="train",
            balance_query_types=True,
            unique_caption_per_batch=False,
        )

        sampler = dataloader.batch_sampler.sampler
        assert isinstance(sampler, BalancedByQueryTypeSampler)
        assert sampler.num_samples == 7
        assert len(sampler.query_type_per_index) == 7

    def test_attribute_metadata_collates_to_batch_tensors(self, temp_dataset):
        """Test DataLoader collates optional attribute metadata to [B, A]."""
        pairs_file = temp_dataset['tmpdir'] / "train_pairs.json"
        _write_attribute_pairs(pairs_file, count=5, width=7)
        dataset_configs = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
            'train_pairs_file': str(pairs_file),
        }]

        def dummy_transform(img):
            return torch.zeros(3, 32, 32)

        dataloader = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            transform=dummy_transform,
            num_workers=0,
            mode='train',
            include_attribute_metadata=True,
        )

        images, texts, metadata = next(iter(dataloader))

        assert images.shape == (2, 3, 32, 32)
        assert len(texts) == 2
        assert metadata["image_attr_values"].shape == (2, 7)
        assert metadata["text_attr_values"].shape == (2, 7)
        assert metadata["image_attr_values"].dtype == torch.long
        assert metadata["text_attr_values"].dtype == torch.long
        assert (metadata["text_attr_values"][:, 1] == -1).all()

    def test_balanced_sampler_works_with_attribute_metadata(
        self, temp_dataset
    ):
        """Test query-type balancing still works when metadata batches are enabled."""
        pairs_file = temp_dataset['tmpdir'] / "train_pairs.json"
        _write_attribute_pairs(pairs_file, count=5, width=7)
        dataset_configs = [{
            'image_dir': temp_dataset['image_dir'],
            'caption_dir': temp_dataset['caption_dir'],
            'image_list_file': temp_dataset['image_list_file'],
            'caption_file_suffix': '.txt',
            'train_pairs_file': str(pairs_file),
        }]

        def dummy_transform(img):
            return torch.zeros(3, 32, 32)

        dataloader = get_custom_dataloader(
            datasets=dataset_configs,
            batch_size=2,
            transform=dummy_transform,
            num_workers=0,
            mode='train',
            balance_query_types=True,
            unique_caption_per_batch=False,
            include_attribute_metadata=True,
        )

        sampler = dataloader.batch_sampler.sampler
        images, _, metadata = next(iter(dataloader))

        assert isinstance(sampler, BalancedByQueryTypeSampler)
        assert images.shape == (2, 3, 32, 32)
        assert metadata["image_attr_values"].shape == (2, 7)


@pytest.mark.multimodal_unit
class TestCLIPDataModule:
    """Test CLIPDataModule custom dataloader wiring."""

    def test_train_forwards_include_attribute_metadata(self, monkeypatch):
        """Test train dataloader forwards attribute metadata config."""
        captured = {}

        def fake_get_custom_dataloader(**kwargs):
            captured.update(kwargs)
            return "train_loader"

        monkeypatch.setattr(
            clip_data_module, "get_custom_dataloader", fake_get_custom_dataloader
        )
        dataset_config = SimpleNamespace(
            seed=42,
            pin_memory=True,
            train=SimpleNamespace(
                type="custom",
                datasets=[{"image_dir": "/tmp/images"}],
                batch_size=4,
                num_workers=0,
                balance_query_types=False,
                unique_caption_per_batch=False,
                include_attribute_metadata=True,
            ),
            val=SimpleNamespace(datasets=[]),
        )
        dm = CLIPDataModule(
            dataset_config=dataset_config,
            tokenizer=lambda text: [text],
            resume_step=0,
            preprocess=(lambda image: image, lambda image: image),
            world_size=1,
        )

        dm._setup_train_dataloader(is_distributed=False)

        assert dm.train_dataset == "train_loader"
        assert captured["mode"] == "train"
        assert captured["include_attribute_metadata"] is True

    def test_val_enables_attribute_metadata_for_matching(
        self, monkeypatch
    ):
        """Test metadata-aware validation requests loader metadata."""
        captured = {}

        def fake_get_custom_dataloader(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(dataset=[0, 1])

        monkeypatch.setattr(
            clip_data_module,
            "get_custom_dataloader",
            fake_get_custom_dataloader,
        )
        dataset_config = SimpleNamespace(
            seed=42,
            pin_memory=True,
            train=SimpleNamespace(type="custom"),
            val=SimpleNamespace(
                datasets=[{"image_dir": "/tmp/images"}],
                batch_size=2,
                num_workers=0,
                metadata_match_eval=True,
            ),
        )
        data_module = CLIPDataModule(
            dataset_config=dataset_config,
            tokenizer=lambda text: [text],
            resume_step=0,
            preprocess=(
                lambda image: image,
                lambda image: image,
            ),
            world_size=1,
        )

        data_module._setup_val_dataloader()

        assert data_module.val_dataset is not None
        assert captured["mode"] == "val"
        assert captured["include_attribute_metadata"] is True

    def test_val_metadata_matching_allows_multiple_datasets(
        self, monkeypatch
    ):
        """Test metadata validation passes all datasets to the strict loader."""
        captured = {}

        def fake_get_custom_dataloader(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(dataset=[0, 1])

        monkeypatch.setattr(
            clip_data_module,
            "get_custom_dataloader",
            fake_get_custom_dataloader,
        )
        datasets = [
            {"image_dir": "/tmp/one"},
            {"image_dir": "/tmp/two"},
        ]
        dataset_config = SimpleNamespace(
            seed=42,
            pin_memory=True,
            train=SimpleNamespace(type="custom"),
            val=SimpleNamespace(
                datasets=datasets,
                batch_size=2,
                num_workers=0,
                metadata_match_eval=True,
            ),
        )
        data_module = CLIPDataModule(
            dataset_config=dataset_config,
            tokenizer=lambda text: [text],
            resume_step=0,
            preprocess=(
                lambda image: image,
                lambda image: image,
            ),
            world_size=1,
        )

        data_module._setup_val_dataloader()

        assert captured["datasets"] == datasets
        assert captured["include_attribute_metadata"] is True
