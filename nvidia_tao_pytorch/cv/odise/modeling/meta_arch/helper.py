# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import cutex


kernels = cutex.SourceModule(r"""
__global__ void max_in_chunks(Tensor<float, 2>  src,
                              Tensor<int, 1>    chunkSizes,
                              Tensor<int, 1>    startIdxs,
                              Tensor<float, 2>  out) {

    auto row = blockIdx.x * blockDim.x + threadIdx.x;
    auto chunkIdx = blockIdx.y * blockDim.y + threadIdx.y;

    // Return if our indices are out of bound.
    if (row >= src.size(0) || chunkIdx >= chunkSizes.size(0))
        return;

    auto i_start = startIdxs[chunkIdx];
    auto i_end = i_start + chunkSizes[chunkIdx];

    float max_val = -1000000.f;
    for (auto i = i_start; i < i_end; ++i)
    {
        if (src[row][i] > max_val)
            max_val = src[row][i];
    }

    out[row][chunkIdx] = max_val;

}
""", float_bits=32, boundscheck=False)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial NCHW tensors"""

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)


class FeatureExtractor(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def ignored_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self._modules.items():
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return super().state_dict(destination=destination, prefix=prefix)

    # don't save DDPM model
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return OrderedDict()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze()
        return self

    def _freeze(self):
        super().train(mode=False)
        for p in self.parameters():
            p.requires_grad = False

    @property
    @abstractmethod
    def feature_dims(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def feature_size(self) -> int:
        pass

    @property
    @abstractmethod
    def num_groups(self) -> int:
        pass

    @property
    @abstractmethod
    def grouped_indices(self, features):
        pass


def ensemble_logits_with_labels_legacy(
    logits: torch.Tensor, labels: List[List[str]], ensemble_method: str = "max"
):
    """Ensemble logits.
    Args:
        logits (torch.Tensor): logits of each model. The last dim is probability.
        labels (list[list[str]]): list of list of labels.
        ensemble_method (str): ensemble method. Options are 'mean' and 'max'.
    Returns:
        torch.Tensor: logits of ensemble model.
    """
    len_list = [len(l) for l in labels]
    assert logits.shape[-1] == sum(len_list), f"{logits.shape[-1]} != {sum(len_list)}"
    assert ensemble_method in ["mean", "max"]
    ensemble_logits = torch.zeros(
        *logits.shape[:-1], len(labels), dtype=logits.dtype, device=logits.device
    )
    if ensemble_method == "max":
        for i in range(len(labels)):
            ensemble_logits[..., i] = (
                logits[..., sum(len_list[:i]) : sum(len_list[: i + 1])].max(dim=-1).values
            )
    elif ensemble_method == "mean":
        for i in range(len(labels)):
            ensemble_logits[..., i] = logits[..., sum(len_list[:i]) : sum(len_list[: i + 1])].mean(
                dim=-1
            )
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")

    return ensemble_logits


def ensemble_logits_with_labels(
    logits: torch.Tensor, chunk_sizes_pyt, chunk_start_idx_pyt, ensemble_method: str = "max"
):
    assert ensemble_method in ["max"]

    x1 = logits.flatten(start_dim=0, end_dim=-2)  # flatten everything but the last dim
    res_out = torch.zeros((x1.shape[0], chunk_sizes_pyt.shape[0]), dtype=torch.float32, device=chunk_sizes_pyt.device)

    # Call the C++ kernel which computes max between the ranges.
    kernels.max_in_chunks(x1, chunk_sizes_pyt, chunk_start_idx_pyt,
                          res_out,
                          grid=(x1.shape[0] // 16 + 1, chunk_sizes_pyt.shape[0] // 16 + 1, 1), block=(16, 16, 1))
    torch.cuda.synchronize()
    res_out = res_out.unflatten(0, logits.shape[:-1])  # Expand all but the last dim.
    return res_out


# Ref:https://stackoverflow.com/questions/27049998/convert-a-mixed-nested-list-to-a-nested-tuple
def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


class MaskPooling(nn.Module):
    def __init__(
        self,
        hard_pooling=True,
        mask_threshold=0.5,
    ):
        super().__init__()
        # if the pooling is hard, it's not differentiable
        self.hard_pooling = hard_pooling
        self.mask_threshold = mask_threshold

    def extra_repr(self) -> str:
        return f"hard_pooling={self.hard_pooling}\n" f"mask_threshold={self.mask_threshold}\n"

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, Q, H, W]
        """

        assert x.shape[-2:] == mask.shape[-2:]

        mask = mask.detach()

        mask = mask.sigmoid()

        if self.hard_pooling:
            mask = (mask > self.mask_threshold).to(mask.dtype)

        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )
        # output = {"mask_pooled_features": mask_pooled_x}
        return mask_pooled_x
