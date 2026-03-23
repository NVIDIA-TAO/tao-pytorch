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

"""Text processing utilities shared across backbone_v2 and multimodal modules."""

import string


def canonicalize_text(
    text: str,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Return canonicalized text (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/main/
    big_vision/evaluators/proj/image_text/prompt_engineering.py

    Args:
        text: String to be canonicalized.
        keep_punctuation_exact_string: If provided, this exact string is
            kept. For example providing '{}' will keep any occurrences of
            '{}' (but will still remove '{' and '}' that appear separately).
        trans_punctuation: Translation table for punctuation removal.

    Returns:
        Canonicalized text string.
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()
