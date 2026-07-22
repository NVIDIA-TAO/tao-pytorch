# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tokenizer utilities."""

import os
import tempfile

import pytest
import torch

from nvidia_tao_pytorch.multimodal.clip.model.tokenizers import (
    canonicalize_text,
    SigLIP2WrappedTokenizer,
    OpenCLIPWrappedTokenizer,
    CLIPCompatibleTokenizer,
    save_tokenizer,
    load_tokenizer,
    get_tokenizer_dir,
)


@pytest.mark.multimodal_unit
class TestCanonicalizeText:
    """Test canonicalize_text function."""

    def test_lowercase(self):
        """Test that text is lowercased."""
        assert canonicalize_text("Hello World") == "hello world"
        assert canonicalize_text("UPPERCASE") == "uppercase"

    def test_punctuation_removal(self):
        """Test that punctuation is removed."""
        assert canonicalize_text("Hello, World!") == "hello world"
        assert canonicalize_text("What's up?") == "whats up"
        assert canonicalize_text("test...test") == "testtest"

    def test_underscore_to_space(self):
        """Test that underscores are converted to spaces."""
        assert canonicalize_text("hello_world") == "hello world"
        assert canonicalize_text("this_is_a_test") == "this is a test"

    def test_whitespace_normalization(self):
        """Test that multiple whitespaces are normalized to single space."""
        assert canonicalize_text("hello   world") == "hello world"
        assert canonicalize_text("  leading and trailing  ") == "leading and trailing"
        assert canonicalize_text("tabs\tand\nnewlines") == "tabs and newlines"

    def test_combined_transformations(self):
        """Test combined transformations."""
        assert canonicalize_text("Hello, World! This_is_a_test.") == "hello world this is a test"
        assert canonicalize_text("  Multiple   Spaces_AND_Punctuation!!!  ") == "multiple spaces and punctuation"

    def test_keep_punctuation_exact_string(self):
        """Test keeping specific punctuation strings."""
        result = canonicalize_text("Hello {} World", keep_punctuation_exact_string="{}")
        assert result == "hello {} world"

    def test_empty_string(self):
        """Test empty string input."""
        assert canonicalize_text("") == ""

    def test_only_punctuation(self):
        """Test string with only punctuation."""
        assert canonicalize_text("...!!!???") == ""

    def test_unicode_text(self):
        """Test unicode text is preserved (except punctuation)."""
        # Unicode letters should be preserved
        assert canonicalize_text("Café") == "café"


@pytest.mark.multimodal_unit
class TestSigLIP2WrappedTokenizer:
    """Test SigLIP2WrappedTokenizer class."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        class MockProcessor:
            def __call__(self, text, **kwargs):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = SigLIP2WrappedTokenizer(MockProcessor())
        assert tokenizer._max_length == 64

    def test_custom_max_length(self):
        """Test tokenizer with custom max length."""
        class MockProcessor:
            def __call__(self, text, **kwargs):
                max_len = kwargs.get('max_length', 64)
                return {
                    'input_ids': torch.zeros(len(text), max_len, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), max_len, dtype=torch.long),
                }

        tokenizer = SigLIP2WrappedTokenizer(MockProcessor(), max_length=128)
        assert tokenizer._max_length == 128

    def test_canonicalization_disabled_by_default(self):
        """Test that canonicalization is disabled by default."""
        received_text = []

        class MockProcessor:
            def __call__(self, text, **kwargs):
                received_text.extend(text)
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = SigLIP2WrappedTokenizer(MockProcessor())
        tokenizer(["Hello, World!", "Test_String"])

        # Text should NOT be canonicalized (default canonicalize=False)
        assert received_text == ["Hello, World!", "Test_String"]

    def test_canonicalization_enabled(self):
        """Test that canonicalization is applied when enabled."""
        received_text = []

        class MockProcessor:
            def __call__(self, text, **kwargs):
                received_text.extend(text)
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = SigLIP2WrappedTokenizer(MockProcessor(), canonicalize=True)
        tokenizer(["Hello, World!", "Test_String"])

        # Text should be canonicalized when enabled
        assert received_text == ["hello world", "test string"]

    def test_returns_dict(self):
        """Test that tokenizer returns dict with expected keys."""
        class MockProcessor:
            def __call__(self, text, **kwargs):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = SigLIP2WrappedTokenizer(MockProcessor())
        result = tokenizer(["test"])

        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert 'attention_mask' in result

    def test_unwraps_radio_tokenizer(self):
        """Test that passing RADIO's SigLIP2WrappedTokenizer (which stores
        the HF processor in ._proc) is automatically unwrapped."""
        class MockHFProcessor:
            def __call__(self, text, **kwargs):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        class MockRadioTokenizer:
            """Mimics RADIO's SigLIP2WrappedTokenizer."""
            def __init__(self, proc):
                self._proc = proc

            def __call__(self, text):
                return self._proc(text=text, return_tensors='pt',
                                  max_length=64, padding='max_length',
                                  truncation=True)

        hf_proc = MockHFProcessor()
        radio_tok = MockRadioTokenizer(hf_proc)
        tokenizer = SigLIP2WrappedTokenizer(radio_tok)

        # Should have unwrapped to the underlying HF processor
        assert tokenizer._processor is hf_proc
        result = tokenizer(["test"])
        assert 'input_ids' in result


@pytest.mark.multimodal_unit
class TestCLIPCompatibleTokenizer:
    """Test CLIPCompatibleTokenizer class."""

    def test_single_text_returns_list(self):
        """Test that single text input returns list format."""
        class MockSigLIP2Tokenizer:
            def __call__(self, text):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = CLIPCompatibleTokenizer(MockSigLIP2Tokenizer())
        result = tokenizer("single text")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_single_text_squeezed(self):
        """Test that single text has batch dimension squeezed."""
        class MockSigLIP2Tokenizer:
            def __call__(self, text):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = CLIPCompatibleTokenizer(MockSigLIP2Tokenizer())
        result = tokenizer("single text")

        # Should be squeezed from (1, 64) to (64,)
        assert result[0]['input_ids'].shape == (64,)
        assert result[0]['attention_mask'].shape == (64,)

    def test_list_text_keeps_batch(self):
        """Test that list of texts keeps batch dimension."""
        class MockSigLIP2Tokenizer:
            def __call__(self, text):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = CLIPCompatibleTokenizer(MockSigLIP2Tokenizer())
        result = tokenizer(["text1", "text2", "text3"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['input_ids'].shape == (3, 64)
        assert result[0]['attention_mask'].shape == (3, 64)

    def test_clip_dataloader_compatibility(self):
        """Test that result[0] gives the dict (CLIP dataloader pattern)."""
        class MockSigLIP2Tokenizer:
            def __call__(self, text):
                return {
                    'input_ids': torch.zeros(len(text), 64, dtype=torch.long),
                    'attention_mask': torch.ones(len(text), 64, dtype=torch.long),
                }

        tokenizer = CLIPCompatibleTokenizer(MockSigLIP2Tokenizer())

        # CLIP dataloader pattern: tokenizer(text)[0]
        result = tokenizer(["test"])[0]

        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert 'attention_mask' in result


@pytest.mark.multimodal_unit
class TestOpenCLIPWrappedTokenizer:
    """Test OpenCLIPWrappedTokenizer class."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        def mock_tokenizer(text):
            return torch.zeros(len(text), 77, dtype=torch.long)

        tokenizer = OpenCLIPWrappedTokenizer(mock_tokenizer)
        assert tokenizer._canonicalize is False

    def test_canonicalization_disabled_by_default(self):
        """Test that canonicalization is disabled by default."""
        received_text = []

        def mock_tokenizer(text):
            received_text.extend(text)
            return torch.zeros(len(text), 77, dtype=torch.long)

        tokenizer = OpenCLIPWrappedTokenizer(mock_tokenizer)
        tokenizer(["Hello, World!", "Test_String"])

        # Text should NOT be canonicalized (default canonicalize=False)
        assert received_text == ["Hello, World!", "Test_String"]

    def test_canonicalization_enabled(self):
        """Test that canonicalization is applied when enabled."""
        received_text = []

        def mock_tokenizer(text):
            received_text.extend(text)
            return torch.zeros(len(text), 77, dtype=torch.long)

        tokenizer = OpenCLIPWrappedTokenizer(mock_tokenizer, canonicalize=True)
        tokenizer(["Hello, World!", "Test_String"])

        # Text should be canonicalized when enabled
        assert received_text == ["hello world", "test string"]

    def test_returns_dict_with_input_ids(self):
        """Test that tokenizer returns dict with 'input_ids' key."""
        def mock_tokenizer(text):
            return torch.zeros(len(text), 77, dtype=torch.long)

        tokenizer = OpenCLIPWrappedTokenizer(mock_tokenizer)
        result = tokenizer(["test"])

        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert result['input_ids'].shape == (1, 77)


@pytest.mark.multimodal_unit
class TestTokenizerSaveLoadFlow:
    """Tests for tokenizer save/load flow used in training and deployment."""

    def test_get_tokenizer_dir_derives_path_from_checkpoint(self):
        """Test that get_tokenizer_dir returns sibling tokenizer directory."""
        assert get_tokenizer_dir("/path/to/train/model.pth") == "/path/to/train/tokenizer"
        assert get_tokenizer_dir("results/train/model.pth") == "results/train/tokenizer"

    def test_load_tokenizer_raises_on_missing_directory(self):
        """Test that load_tokenizer raises FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError, match="Tokenizer directory not found"):
            load_tokenizer("/nonexistent/path/tokenizer")

    def test_save_tokenizer_creates_directory(self):
        """Test that save_tokenizer creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "new_tokenizer_dir")

            class MockOpenCLIPTokenizer:
                pass

            mock_tokenizer = CLIPCompatibleTokenizer(MockOpenCLIPTokenizer())

            try:
                save_tokenizer(mock_tokenizer, output_dir, "openclip", None)
            except Exception:
                pass  # Expected to fail on HF download

            assert os.path.isdir(output_dir)

    def test_save_tokenizer_siglip2_detection(self):
        """Test that SigLIP2 tokenizers use save_pretrained directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "tokenizer")

            class MockHFTokenizer:
                def save_pretrained(self, path):
                    os.makedirs(path, exist_ok=True)
                    with open(os.path.join(path, "marker.txt"), "w") as f:
                        f.write("siglip2")

            class MockProcessor:
                def __init__(self):
                    self.tokenizer = MockHFTokenizer()

            mock_inner = SigLIP2WrappedTokenizer(MockProcessor())
            mock_tokenizer = CLIPCompatibleTokenizer(mock_inner)

            save_tokenizer(mock_tokenizer, output_dir, "siglip2-so400m-patch16-256", None)

            assert os.path.exists(os.path.join(output_dir, "marker.txt"))

    def test_export_tokenizer_copy_flow(self):
        """Test the tokenizer copy flow used during ONNX export."""
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            # Source: training saves tokenizer
            source_dir = os.path.join(tmpdir, "train", "tokenizer")
            os.makedirs(source_dir)
            with open(os.path.join(source_dir, "vocab.json"), "w") as f:
                f.write('{"test": 1}')

            # Export: copies tokenizer to ONNX directory
            export_dir = os.path.join(tmpdir, "export")
            os.makedirs(export_dir)
            output_dir = os.path.join(export_dir, "model_tokenizer")

            shutil.copytree(source_dir, output_dir)

            # Verify copy succeeded
            assert os.path.exists(os.path.join(output_dir, "vocab.json"))
            with open(os.path.join(output_dir, "vocab.json")) as f:
                assert f.read() == '{"test": 1}'
