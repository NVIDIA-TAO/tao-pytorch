# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for offline VideoCLIP tokenizer export."""

import json

from nvidia_tao_pytorch.multimodal.video_clip.model.tokenizers import save_tokenizer


def test_in_memory_openclip_tokenizer_exports_without_hub(monkeypatch, tmp_path):
    class RawTokenizer:
        encoder = {"<|startoftext|>": 0, "<|endoftext|>": 1, "hello</w>": 2}
        bpe_ranks = {("h", "e"): 0, ("he", "llo</w>"): 1}

    class ClipTokenizer:
        context_length = 77
        tokenizer = RawTokenizer()

    class Wrapper:
        _tokenizer = ClipTokenizer()

    monkeypatch.setattr(
        "nvidia_tao_pytorch.multimodal.video_clip.model.tokenizers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network lookup")),
    )
    save_tokenizer(Wrapper(), str(tmp_path), "internvideo2-clip-l14")
    assert json.loads((tmp_path / "vocab.json").read_text())["hello</w>"] == 2
    assert (tmp_path / "merges.txt").read_text().startswith("#version: 0.2")
    assert json.loads((tmp_path / "tokenizer_config.json").read_text())["model_max_length"] == 77
