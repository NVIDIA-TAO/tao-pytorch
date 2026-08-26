# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Video-CLIP tokenizer serialization."""

from types import SimpleNamespace

import open_clip
import torch

from nvidia_tao_pytorch.multimodal.video_clip.model import tokenizers


def test_save_openclip_tokenizer_offline_preserves_token_ids(
    tmp_path,
    monkeypatch,
):
    """Loaded OpenCLIP BPE assets save offline with exact token-ID parity."""
    source_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    output_dir = tmp_path / "tokenizer"
    texts = [
        "hello world",
        "Capitalization, punctuation!",
        "",
        "café — test",
    ]

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    with monkeypatch.context() as context:
        def _unexpected_download(*args, **kwargs):
            raise AssertionError("unexpected Hugging Face download")

        context.setattr(
            tokenizers.AutoTokenizer,
            "from_pretrained",
            _unexpected_download,
        )
        tokenizers.save_tokenizer(
            SimpleNamespace(_tokenizer=source_tokenizer),
            str(output_dir),
            model_type="internvideo2-clip-l14",
        )

    restored_tokenizer = tokenizers.load_tokenizer(str(output_dir))
    restored_ids = restored_tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=source_tokenizer.context_length,
        return_tensors="pt",
    ).input_ids

    torch.testing.assert_close(restored_ids, source_tokenizer(texts))
    assert restored_tokenizer.pad_token_id == 0
    assert restored_tokenizer.model_max_length == source_tokenizer.context_length
