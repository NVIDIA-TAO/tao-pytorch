# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Model functions. """

import math
import torch
from torch import nn


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """generate sine position embedding from a position tensor.

    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is [x,y], the results will be [pos(y), pos(x)]. Defaults to True.

    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res


class ContrastiveEmbed(nn.Module):
    """ContrastiveEmbed Class."""

    def __init__(self,
                 max_text_len=256,
                 log_scale=None,
                 bias=False):
        """Initialize ContrastiveEmbed.

        Args:
            max_text_len (int, optional): Maximum length of text.
            log_scale (Optional[Union[str, float]]):  The initial value of a
            learnable parameter to multiply with the similarity
            matrix to normalize the output. Defaults to None.
            - If set to 'auto', the similarity matrix will be normalized by
                a fixed value ``sqrt(d_c)`` where ``d_c`` is the channel number.
            - If set to 'none' or ``None``, there is no normalization applied.
            - If set to a float number, the similarity matrix will be multiplied
                by ``exp(log_scale)``, where ``log_scale`` is learnable.
            bias (bool, optional): Whether to add bias to the output.
            If set to ``True``, a learnable bias that is initialized as -4.6
            will be added to the output. Useful when training from scratch.
            Defaults to False.
        """
        super().__init__()
        self.max_text_len = max_text_len
        self.log_scale = log_scale
        if isinstance(log_scale, float):
            self.log_scale = nn.Parameter(
                torch.Tensor([float(log_scale)]), requires_grad=True)
        elif log_scale not in ['auto', 'none', None]:
            raise ValueError(f'log_scale should be one of '
                             f'"auto", "none", None, but got {log_scale}')

        self.bias = None
        if bias:
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.bias = nn.Parameter(
                torch.Tensor([bias_value]), requires_grad=True)

    def forward(self, visual_feat, text_dict):
        """Forward function.

        Args:
            visual_feat (Tensor): Visual features.
            text_dict (Tensor): Text features.
                - encoded_text: encoded_text, # bs, 195, d_model
                - text_token_mask: text_token_mask, # bs, 195

        Returns:
            Tensor: Classification score.
        """
        text_feat = text_dict["encoded_text"]
        text_token_mask = text_dict["text_token_mask"]
        res = visual_feat @ text_feat.transpose(-1, -2)
        if isinstance(self.log_scale, nn.Parameter):
            res = res * self.log_scale.exp()
        elif self.log_scale == 'auto':
            # NOTE: similar to the normalizer in self-attention
            res = res / math.sqrt(visual_feat.shape[-1])
        if self.bias is not None:
            res = res + self.bias
        res.masked_fill_(~text_token_mask[:, None, :], float('-inf'))

        new_res = torch.full((*res.shape[:-1], self.max_text_len),
                             float('-inf'),
                             device=res.device)
        new_res[..., :res.shape[-1]] = res

        return new_res
