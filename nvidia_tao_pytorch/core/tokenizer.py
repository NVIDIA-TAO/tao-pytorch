# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities that could be used in create_tokenizer script."""

from typing import Optional
from omegaconf import MISSING
from dataclasses import dataclass


__all__ = ["TokenizerConfig"]


@dataclass
class TokenizerConfig:
    """Tokenizer config for use in create_tokenizer script."""

    # tokenizer type: "spe" or "wpe"
    tokenizer_type: str = MISSING
    # spe type if tokenizer_type == "spe"
    # choose from ['bpe', 'unigram', 'char', 'word']
    spe_type: str = MISSING
    # spe character coverage, defaults to 1.0
    spe_character_coverage: Optional[float] = 1.0
    # flag for lower case, defaults to True
    lower_case: Optional[bool] = True
