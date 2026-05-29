# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head modules for neural networks."""

from timm.models.layers import trunc_normal_
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


class DinoHead(nn.Module):
    """DINO Head for self-supervised learning tasks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        norm: nn.Module = nn.Identity,
        act: nn.Module = nn.GELU,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        """Initialize the DINO head.

        Args:
            in_dim (int): The input dimension for the MLP.
            out_dim (int): The output dimension for the MLP.
            norm (nn.Module, optional): Normalization layer to apply after each linear transformation. Defaults to nn.Identity.
            act (nn.Module, optional): Activation function to apply after each linear transformation. Defaults to nn.GELU.
            num_layers (int, optional): Number of layers in the MLP. Defaults to 3.
            hidden_dim (int, optional): Hidden dimension of the MLP. Defaults to 2048.
            bottleneck_dim (int, optional): Bottleneck dimension for the final output. Defaults to 256.
        """
        super().__init__()

        assert num_layers >= 1

        if num_layers == 1:
            self.mlp = nn.Sequential(nn.Linear(in_dim, bottleneck_dim))
        else:
            layers = [nn.Linear(in_dim, hidden_dim), norm(hidden_dim), act()]

            for _ in range(num_layers - 2):
                layers.extend(
                    [nn.Linear(hidden_dim, hidden_dim), norm(hidden_dim), act()]
                )

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def forward(self, x):
        """Forward pass for DINO head.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Output tensor after applying the MLP and normalization.
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2, eps=1e-6)
        x = self.last_layer(x)

        return x

    def _init_weights(self, m):
        """Initialize the weights of the DINO head.

        Args:
            m (Module): The layer module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class MLPHead(nn.Module):
    """MLP Head for general feedforward tasks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: nn.Module = nn.Mish,
        mlp_factor: int = 4,
    ):
        """Initialize the MLP head.

        Args:
            in_dim (int): The input dimension for the MLP.
            out_dim (int): The output dimension for the MLP.
            act (nn.Module, optional): Activation function to apply after the first linear transformation. Defaults to nn.Mish.
            mlp_factor (int, optional): Factor by which to expand the input dimension for the hidden layer. Defaults to 4.
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * mlp_factor),
            act(),
            nn.Linear(in_dim * mlp_factor, out_dim),
        )

        self.apply(self._init_weights)

    def forward(self, x):
        """Forward pass for MLP head.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.mlp(x)

    def _init_weights(self, m):
        """Initialize the weights of the MLP head.

        Args:
            m (Module): The layer module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
