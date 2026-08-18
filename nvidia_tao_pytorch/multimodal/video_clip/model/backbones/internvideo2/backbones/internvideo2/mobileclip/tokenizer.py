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
#
# Vendored and adapted from OpenGVLab InternVideo2 (multi_modality), Apache-2.0:
# https://github.com/OpenGVLab/InternVideo

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""CLIP byte-pair tokenizer used by the MobileCLIP text tower."""
from typing import Dict

import open_clip
from torch import Tensor, nn


class ClipTokenizer(nn.Module):
    """Wrap the open_clip BPE tokenizer as an nn.Module."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.context_length = cfg["text_cfg"]["context_length"]
        model_name = getattr(cfg["text_cfg"], "open_clip_tokenizer", "ViT-B-16")
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def get_vocab_size(self) -> int:
        """Return the size of the tokenizer vocabulary."""
        return len(self.tokenizer.encoder)

    def get_encodings(self) -> Dict[str, int]:
        """Return the token-to-id mapping."""
        return self.tokenizer.encoder

    def get_eot_token(self) -> int:
        # Tokenizing an empty string returns a list [sot_id, eot_id]
        """Return the end-of-text token id."""
        return self.tokenizer("")[1]

    def get_sot_token(self) -> int:
        # Tokenizing an empty string returns a list [sot_id, eot_id]
        """Return the start-of-text token id."""
        return self.tokenizer("")[0]

    def forward(self, input_sentence: str, *args, **kwargs) -> Tensor:
        # tokenizer returns indices as a string
        """Tokenize a sentence into a fixed-length id tensor."""
        tokenized_sentence = self.tokenizer(input_sentence, self.context_length)
        assert (
            tokenized_sentence.shape[-1] == self.context_length
        ), "Tokenized tensor should be exactly `context_length` long."
        return tokenized_sentence
