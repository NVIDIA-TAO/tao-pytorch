# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Text tokenization utilities for OneFormer model using Hugging Face Transformers."""

from transformers import AutoTokenizer

# Load the standard CLIP tokenizer from Hugging Face.
# This single line replaces the entire SimpleTokenizer class, BPE file loading,
# and all helper functions. It will be downloaded and cached automatically.
# You can also use other CLIP versions like "openai/clip-vit-large-patch14".
_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


class Tokenize:
    """Text tokenizer class for OneFormer text processing.

    This class handles tokenization of text inputs for the OneFormer model,
    including special tokens and length constraints.
    """

    def __init__(self, max_seq_len=77, truncate=True):
        """Initialize the tokenizer.

        Args:
            max_seq_len (int): Maximum sequence length.
            truncate (bool): Whether to truncate sequences that are too long.
        """
        self.max_seq_len = max_seq_len
        self.truncate = truncate

    def __call__(self, texts):
        """Tokenize input text(s).

        The Hugging Face tokenizer handles batching, padding, truncation,
        and special tokens (<|startoftext|>, <|endoftext|>) automatically.

        Args:
            texts (str or list): Text or list of texts to tokenize.

        Returns:
            torch.Tensor: Tokenized text as a tensor of input IDs.
        """
        # The tokenizer expects a list of strings.
        is_string = isinstance(texts, str)
        if is_string:
            texts = [texts]

        # Tokenize the texts
        tokenized_output = _tokenizer(
            texts,
            padding="max_length",       # Pad to max_seq_len
            truncation=self.truncate,   # Truncate if longer than max_seq_len
            max_length=self.max_seq_len,
            return_tensors="pt"         # Return PyTorch tensors
        )

        # The tokenizer returns a dictionary; we only need the 'input_ids'.
        # The shape is already (batch_size, max_seq_len).
        result = tokenized_output["input_ids"]

        # If the original input was a single string, return a single tensor.
        if is_string:
            return result[0]

        return result
