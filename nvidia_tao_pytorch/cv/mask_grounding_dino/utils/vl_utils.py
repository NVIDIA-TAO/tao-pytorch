# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

""" VL utility functions. """

import torch


def create_positive_map(tokenized, tokens_positive, cat_list, caption, empty=False, max_text_len=256):
    """
    Construct a positive map where positive_map[i, j] = 1 iff box i corresponds to token j.
    Supports multi-word phrases with ordered matching (no max_gap).

    Args:
        tokenized: tokenizer output with char_to_token()
        tokens_positive: list[int], indices of cat_list referring to this image
        cat_list: list[str], candidate phrases or object names
        caption: str, full caption text
    """
    positive_map = torch.zeros((len(tokens_positive), max_text_len), dtype=torch.float)
    if empty:
        return positive_map

    # Handle Encoding (has .tokens as list property), list of wordpiece tokens like ['[CLS]', 'dolls', ',', 'book', ...]
    tokens = tokenized.tokens  # Encoding: direct property access
    lowered_tokens = [t.lower() for t in tokens]

    for j, label_idx in enumerate(tokens_positive):
        phrase = cat_list[label_idx].lower().split()
        # find token positions for each word in phrase
        positions = []
        for word in phrase:
            word_positions = [i for i, t in enumerate(lowered_tokens) if t == word]
            positions.append(word_positions)

        # find ordered combination of positions
        selected_positions = []
        current_pos = -1
        for word_pos_list in positions:
            # pick first position > current_pos
            valid_next = [p for p in word_pos_list if p > current_pos]
            if not valid_next:
                selected_positions = []  # no valid sequence
                break
            chosen = valid_next[0]
            selected_positions.append(chosen)
            current_pos = chosen

        # fill positive map for selected tokens
        for pos in selected_positions:
            if 0 <= pos < max_text_len:
                positive_map[j, pos] = 1.0
    return positive_map


def create_positive_map_from_span(tokenized, token_span, empty=False, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Args:
        tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
        empty (bool): if True, return all-zeros.
    """
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    if empty:
        return positive_map
    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except Exception:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except Exception:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)
