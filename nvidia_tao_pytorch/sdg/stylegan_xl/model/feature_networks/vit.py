# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Vision Transformer backbone for Discriminator's feature network. """

import torch
import torch.nn as nn


class Slice(nn.Module):
    """Custom slicing layer."""

    def __init__(self, start_index=1):
        """Initializes the Slice layer.

        Args:
            start_index (int, optional): Index to start slicing from. Default is 1.
        """
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Sliced tensor.
        """
        return x[:, self.start_index:]


class AddReadout(nn.Module):
    """Custom layer to add readout tokens."""

    def __init__(self, start_index=1):
        """Initializes the AddReadout layer.

        Args:
            start_index (int, optional): Index to start adding readout tokens from. Default is 1.
        """
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with added readout tokens.
        """
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    """Custom layer to project readout tokens."""

    def __init__(self, in_features, start_index=1):
        """Initializes the ProjectReadout layer.

        Args:
            in_features (int): Number of input features.
            start_index (int, optional): Index to start projecting readout tokens from. Default is 1.
        """
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with projected readout tokens.
        """
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    """Custom transpose layer."""

    def __init__(self, dim0, dim1):
        """Initializes the Transpose layer.

        Args:
            dim0 (int): First dimension to transpose.
            dim1 (int): Second dimension to transpose.
        """
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transposed tensor.
        """
        x = x.transpose(self.dim0, self.dim1)
        return x.contiguous()


def forward_vit(pretrained, x):
    # pylint: disable=unused-variable
    """Forward function for Vision Transformer.

    Args:
        pretrained (nn.Module): Pretrained Vision Transformer model.
        x (torch.Tensor): Input tensor.

    Returns:
        tuple: Output tensors from different layers.
    """
    b, c, h, w = x.shape

    _ = pretrained.model(x)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    layer_1 = pretrained.layer1[0:2](layer_1)
    layer_2 = pretrained.layer2[0:2](layer_2)
    layer_3 = pretrained.layer3[0:2](layer_3)
    layer_4 = pretrained.layer4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.layer1[3: len(pretrained.layer1)](layer_1)
    layer_2 = pretrained.layer2[3: len(pretrained.layer2)](layer_2)
    layer_3 = pretrained.layer3[3: len(pretrained.layer3)](layer_3)
    layer_4 = pretrained.layer4[3: len(pretrained.layer4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4


# Dictionary to store activations.
activations = {}


def get_activation(name):
    """Returns a hook function to store activations.

    Args:
        name (str): Name of the activation.

    Returns:
        function: Hook function to store activations.
    """
    def hook(model, input, output):
        # pylint: disable=W0622
        activations[name] = output

    return hook


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    """Returns the appropriate readout operation.

    Args:
        vit_features (int): Number of Vision Transformer features.
        features (list): List of feature sizes.
        use_readout (str): Type of readout operation ('ignore', 'add', 'project').
        start_index (int, optional): Index to start readout operation from. Default is 1.

    Returns:
        list: List of readout operations.
    """
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper


def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
):
    """Creates a Vision Transformer backbone with specified configurations.

    Args:
        model (nn.Module): Vision Transformer model.
        features (list, optional): List of feature sizes. Default is [96, 192, 384, 768].
        size (list, optional): Size of the input image. Default is [384, 384].
        hooks (list, optional): List of hook indices. Default is [2, 5, 8, 11].
        vit_features (int, optional): Number of Vision Transformer features. Default is 768.
        use_readout (str, optional): Type of readout operation ('ignore', 'add', 'project'). Default is 'ignore'.
        start_index (int, optional): Index to start readout operation from. Default is 1.

    Returns:
        nn.Module: Pretrained Vision Transformer backbone.
    """
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    # 32, 48, 136, 384
    pretrained.layer1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.layer2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.layer3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.layer4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    return pretrained
