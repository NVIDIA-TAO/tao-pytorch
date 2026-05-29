# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured pruning."""
import torch
import torch.nn as nn
from copy import deepcopy
from functools import reduce
from operator import mul
from abc import ABC, abstractstaticmethod
from typing import Sequence, Tuple


class BasePruningFunction(ABC):
    """Base pruning function
    """

    @classmethod
    def apply(cls, layer: nn.Module, idxs: Sequence[int], inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
        """Apply the pruning function."""
        idxs = list(set(idxs))
        cls.check(layer, idxs)
        nparams_to_prune = cls.calc_nparams_to_prune(layer, idxs)
        if dry_run:
            return layer, nparams_to_prune
        if not inplace:
            layer = deepcopy(layer)
        layer = cls.prune_params(layer, idxs)
        return layer, nparams_to_prune

    @staticmethod
    def check(layer: nn.Module, idxs: Sequence[int]) -> None:
        """check."""
        pass

    @abstractstaticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        pass

    @abstractstaticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        pass


class ConvPruning(BasePruningFunction):
    """Conv Pruning."""

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        layer.out_channels = layer.out_channels - len(idxs)
        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = len(idxs) * reduce(mul, layer.weight.shape[1:]) + (len(idxs) if layer.bias is not None else 0)
        return nparams_to_prune


class GroupConvPruning(ConvPruning):
    """Group Conv pruning."""

    @staticmethod
    def check(layer, idxs) -> nn.Module:
        """Check."""
        if layer.groups > 1:
            assert layer.groups == layer.in_channels and layer.groups == layer.out_channels, "only group conv with in_channel==groups==out_channels is supported"

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        layer.out_channels = layer.out_channels - len(idxs)
        layer.in_channels = layer.in_channels - len(idxs)
        layer.groups = layer.groups - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer


class RelatedConvPruning(BasePruningFunction):
    """Related Conv Pruning."""

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        layer.in_channels = layer.in_channels - len(idxs)
        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        # no bias pruning because it does not change the output size
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = len(idxs) * layer.weight.shape[0] * reduce(mul, layer.weight.shape[2:])
        return nparams_to_prune


class LinearPruning(BasePruningFunction):
    """Linear Pruning."""

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.out_features)) - set(idxs))
        layer.out_features = layer.out_features - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = len(idxs) * layer.weight.shape[1] + (len(idxs) if layer.bias is not None else 0)
        return nparams_to_prune


class RelatedLinearPruning(BasePruningFunction):
    """Related Linear Pruning."""

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.in_features)) - set(idxs))
        layer.in_features = layer.in_features - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = len(idxs) * layer.weight.shape[0]
        return nparams_to_prune


class BatchnormPruning(BasePruningFunction):
    """BatchNorm Pruning."""

    @staticmethod
    def prune_params(layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune parameters."""
        keep_idxs = list(set(range(layer.num_features)) - set(idxs))
        layer.num_features = layer.num_features - len(idxs)
        layer.running_mean = layer.running_mean.data.clone()[keep_idxs]
        layer.running_var = layer.running_var.data.clone()[keep_idxs]
        if layer.affine:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.Module, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = len(idxs) * (2 if layer.affine else 1)
        return nparams_to_prune


class PReLUPruning(BasePruningFunction):
    """PReLU pruning."""

    @staticmethod
    def prune_params(layer: nn.PReLU, idxs: list) -> nn.Module:
        """Prune parameters."""
        if layer.num_parameters == 1:
            return layer
        keep_idxs = list(set(range(layer.num_parameters)) - set(idxs))
        layer.num_parameters = layer.num_parameters - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        return layer

    @staticmethod
    def calc_nparams_to_prune(layer: nn.PReLU, idxs: Sequence[int]) -> int:
        """Compute number of parameters to prune."""
        nparams_to_prune = 0 if layer.num_parameters == 1 else len(idxs)
        return nparams_to_prune


# Funtional
def prune_conv(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune conv."""
    return ConvPruning.apply(layer, idxs, inplace, dry_run)


def prune_related_conv(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune related conv."""
    return RelatedConvPruning.apply(layer, idxs, inplace, dry_run)


def prune_group_conv(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune group conv."""
    return GroupConvPruning.apply(layer, idxs, inplace, dry_run)


def prune_batchnorm(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune Batch Norm."""
    return BatchnormPruning.apply(layer, idxs, inplace, dry_run)


def prune_linear(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune Linear."""
    return LinearPruning.apply(layer, idxs, inplace, dry_run)


def prune_related_linear(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune related linear."""
    return RelatedLinearPruning.apply(layer, idxs, inplace, dry_run)


def prune_prelu(layer: nn.modules.conv._ConvNd, idxs: list, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
    """Prune prelu."""
    return PReLUPruning.apply(layer, idxs, inplace, dry_run)
