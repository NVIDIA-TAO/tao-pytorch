# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model builder interface and joint model."""
import torch
import torch.nn as nn

from .resnet import resnet2d
from .resnet3d import resnet3d
from .i3d import InceptionI3d
import torch.nn.functional as F
from nvidia_tao_pytorch.core.utilities import patch_decrypt_checkpoint
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook


def load_pretrained_weights(pretrained_backbone_path):
    """Load pretrained weights for a PyTorch model.

    This function takes the path to the pretrained weights file as input. It loads the weights using PyTorch's
    `torch.load` function and applies a patch to decrypt the checkpoint state_dict if it is encrypted. It then
    extracts the state_dict from the loaded weights and converts the keys to match the format expected by the
    PyTorch model. Finally, it returns the state_dict.

    Args:
        pretrained_backbone_path (str): The path to the pretrained weights file.

    Returns:
        dict: The state_dict for the PyTorch model.
    """
    temp = torch.load(pretrained_backbone_path,
                      map_location="cpu",
                      weights_only=False)

    if temp.get("state_dict_encrypted", False):
        # Retrieve encryption key from TLTPyTorchCookbook.
        key = TLTPyTorchCookbook.get_passphrase()
        if key is None:
            raise PermissionError("Cannot access model state dict without the encryption key")
        temp = patch_decrypt_checkpoint(temp, key)

    # for loading pretrained I3D weights released on
    # https://github.com/piergiaj/pytorch-i3d
    if "state_dict" not in temp:
        return temp

    state_dict = {}
    for key, value in list(temp["state_dict"].items()):
        if "model" in key:
            new_key = ".".join(key.split(".")[1:])
            state_dict[new_key] = value
        else:
            state_dict[key] = value

    return state_dict


# @TODO(tylerz): imagenet_pretrained is an internal option for verification
def get_basemodel(backbone,
                  input_channel,
                  nb_classes,
                  imagenet_pretrained=False,
                  pretrained_backbone_path=None,
                  dropout_ratio=0.5,
                  pretrained_class_num=0):
    """Get backbone model for 2D input.

    This function takes the backbone architecture, input channel, number of classes, imagenet pretrained flag,
    pretrained backbone path, dropout ratio, and pretrained class number as input. It loads pretrained weights if
    specified, sets the number of classes to load based on the pretrained class number or number of classes, and
    constructs a 2D backbone model using the specified architecture, input channel, number of classes, imagenet
    pretrained flag, pretrained weights, and dropout ratio. If the pretrained class number is not zero, it replaces
    the logits layer of the model with a new one that outputs the specified number of classes. Finally, it returns
    the constructed model.

    Args:
        backbone (str): The backbone architecture.
        input_channel (int): The input channel.
        nb_classes (int): The number of classes.
        imagenet_pretrained (bool, optional): Whether to use imagenet pretrained weights. Defaults to False.
        pretrained_backbone_path (str, optional): The path to the pretrained backbone weights file. Defaults to None.
        dropout_ratio (float, optional): The dropout ratio. Defaults to 0.5.
        pretrained_class_num (int, optional): The number of classes in the pretrained backbone. Defaults to 0.

    Returns:
        torch.nn.Module: The constructed 2D backbone model.
    """
    if pretrained_backbone_path:
        print("loading trained weights from {}".format(
            pretrained_backbone_path))
        pretrained_weights = load_pretrained_weights(pretrained_backbone_path)
    else:
        pretrained_weights = None

    if pretrained_class_num != 0:
        load_n_classes = pretrained_class_num
    else:
        load_n_classes = nb_classes

    if "resnet" in backbone:

        model = resnet2d(backbone=backbone,
                         pretrained_weights=pretrained_weights,
                         channel=input_channel,
                         nb_classes=load_n_classes,
                         imagenet_pretrained=imagenet_pretrained,
                         dropout_ratio=dropout_ratio)

    if pretrained_class_num != 0:
        model.replace_logits(nb_classes)

    return model


def get_basemodel3d(backbone,
                    nb_classes,
                    modality="rgb",
                    pretrained_backbone_path=None,
                    imagenet_pretrained=False,
                    dropout_ratio=0.5,
                    pretrained_class_num=0):
    """Get backbone model for 3D input.

    This function takes the backbone architecture, number of classes, modality, pretrained backbone path, imagenet
    pretrained flag, dropout ratio, and pretrained class number as input. It loads pretrained weights if specified,
    sets the number of classes to load based on the pretrained class number or number of classes, and constructs a
    3D backbone model using the specified architecture, modality, number of classes, pretrained weights, and dropout
    ratio. If the backbone architecture is "i3d" and the modality is "rgb" or "of", it constructs an InceptionI3d
    model with the specified number of classes and input channels. If pretrained weights are specified, it loads
    them into the model. If the pretrained class number is not zero, it replaces the logits layer of the model with
    a new one that outputs the specified number of classes. Finally, it returns the constructed model.

    Args:
        backbone (str): The backbone architecture.
        nb_classes (int): The number of classes.
        modality (str, optional): The modality. Defaults to "rgb".
        pretrained_backbone_path (str, optional): The path to the pretrained backbone weights file. Defaults to None.
        imagenet_pretrained (bool, optional): Whether to use imagenet pretrained weights. Defaults to False.
        dropout_ratio (float, optional): The dropout ratio. Defaults to 0.5.
        pretrained_class_num (int, optional): The number of classes in the pretrained backbone. Defaults to 0.

    Returns:
        torch.nn.Module: The constructed 3D backbone model.
    """
    if pretrained_backbone_path:
        print("loading trained weights from {}".format(
            pretrained_backbone_path))
        pretrained_weights = load_pretrained_weights(pretrained_backbone_path)
    else:
        pretrained_weights = None

    if pretrained_class_num != 0:
        load_n_classes = pretrained_class_num
    else:
        load_n_classes = nb_classes

    if 'resnet' in backbone:
        model = resnet3d(backbone,
                         modality=modality,
                         nb_classes=load_n_classes,
                         pretrained_weights=pretrained_weights,
                         dropout_ratio=dropout_ratio,
                         pretrained2d=imagenet_pretrained)
    elif backbone == "i3d":
        if modality == "rgb":
            channels = 3
        elif modality == "of":
            channels = 2

        model = InceptionI3d(num_classes=load_n_classes, in_channels=channels,
                             dropout_keep_prob=dropout_ratio)

        if pretrained_weights is not None:
            model.load_state_dict(pretrained_weights)

    # Replace final FC layer to match dataset
    if pretrained_class_num != 0:
        model.replace_logits(nb_classes)

    return model


class JointModel(nn.Module):
    """Joint model module.

    This class defines a joint model module that takes two inputs, an RGB sequence and an optical flow sequence, and
    outputs a prediction for the action class
    """

    def __init__(self,
                 of_seq_length,
                 rgb_seq_length,
                 nb_classes,
                 num_fc=64,
                 backbone='resnet_18',
                 input_type="2d",
                 pretrain_of_model=None,
                 pretrain_rgb_model=None,
                 imagenet_pretrained=False,
                 dropout_ratio=0.5):
        """Initialize the JointModel

        Args:
            of_seq_length (int): The length of the optical flow sequence.
            rgb_seq_length (int): The length of the RGB sequence.
            nb_classes (int): The number of classes.
            num_fc (int, optional): The number of hidden units for the first fully connected layer. Defaults to 64.
            backbone (str, optional): The backbone architecture. Defaults to "resnet_18".
            input_type (str, optional): The input type, either "2d" or "3d". Defaults to "2d".
            pretrain_of_model (str, optional): The path to the pretrained optical flow backbone weights file. Defaults to None.
            pretrain_rgb_model (str, optional): The path to the pretrained RGB backbone weights file. Defaults to None.
            imagenet_pretrained (bool, optional): Whether to use imagenet pretrained weights. Defaults to False.
            dropout_ratio (float, optional): The dropout ratio. Defaults to 0.5.
        """
        super(__class__, self).__init__()  # pylint:disable=undefined-variable
        if input_type == "2d":
            self.model_rgb = get_basemodel(backbone=backbone,
                                           input_channel=rgb_seq_length * 3,
                                           nb_classes=nb_classes,
                                           imagenet_pretrained=imagenet_pretrained,
                                           pretrained_backbone_path=pretrain_rgb_model,
                                           dropout_ratio=dropout_ratio)
            self.model_of = get_basemodel(backbone=backbone,
                                          input_channel=of_seq_length * 2,
                                          nb_classes=nb_classes,
                                          imagenet_pretrained=imagenet_pretrained,
                                          pretrained_backbone_path=pretrain_of_model,
                                          dropout_ratio=dropout_ratio)
        elif input_type == "3d":
            self.model_rgb = get_basemodel3d(backbone=backbone,
                                             nb_classes=nb_classes,
                                             modality="rgb",
                                             pretrained_backbone_path=pretrain_rgb_model,
                                             imagenet_pretrained=imagenet_pretrained,
                                             dropout_ratio=dropout_ratio)
            self.model_of = get_basemodel3d(backbone=backbone,
                                            nb_classes=nb_classes,
                                            modality="of",
                                            pretrained_backbone_path=pretrain_of_model,
                                            imagenet_pretrained=imagenet_pretrained,
                                            dropout_ratio=dropout_ratio)

        self.fc1 = nn.Linear(2 * nb_classes, num_fc)
        self.fc2 = nn.Linear(num_fc, nb_classes)

    def forward(self, x):
        """Joint forward.

        This method takes two input sequences, an RGB sequence and an optical flow sequence, and passes them through
        the two backbone models to obtain their output features. It then concatenates the output features and passes
        them through two fully connected layers to output the final prediction.

        Args:
            x (tuple): A tuple containing the RGB sequence and optical flow sequence.

        Returns:
            torch.Tensor: The predicted action class probabilities.
        """
        x_rgb, x_of = x
        x_rgb = self.model_rgb(x_rgb)
        x_of = self.model_of(x_of)
        # x = (x_rgb + x_of)
        x = torch.cat((x_rgb, x_of), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class JointModel_ONNX(JointModel):
    """Joint model module for export.

    This class defines a joint model module that inherits from the `JointModel` class and adds support for exporting
    the model to ONNX format.
    """

    def __init__(self,
                 of_seq_length,
                 rgb_seq_length,
                 nb_classes,
                 num_fc=64,
                 backbone='resnet_18',
                 input_type="2d",
                 pretrain_of_model=None,
                 pretrain_rgb_model=None,
                 imagenet_pretrained=False,
                 dropout_ratio=0.5):
        """Initialize the JointModel for ONNX export

        Args:
            of_seq_length (int): The length of the optical flow sequence.
            rgb_seq_length (int): The length of the RGB sequence.
            nb_classes (int): The number of classes.
            num_fc (int, optional): The number of hidden units for the first fully connected layer. Defaults to 64.
            backbone (str, optional): The backbone architecture. Defaults to "resnet_18".
            input_type (str, optional): The input type, either "2d" or "3d". Defaults to "2d".
            pretrain_of_model (str, optional): The path to the pretrained optical flow backbone weights file. Defaults to None.
            pretrain_rgb_model (str, optional): The path to the pretrained RGB backbone weights file. Defaults to None.
            imagenet_pretrained (bool, optional): Whether to use imagenet pretrained weights. Defaults to False.
            dropout_ratio (float, optional): The dropout ratio. Defaults to 0.5.
        """
        super(__class__, self).__init__(of_seq_length=of_seq_length,  # pylint:disable=undefined-variable
                                        rgb_seq_length=rgb_seq_length,
                                        nb_classes=nb_classes,
                                        num_fc=num_fc,
                                        backbone=backbone,
                                        input_type=input_type,
                                        pretrain_of_model=pretrain_of_model,
                                        pretrain_rgb_model=pretrain_rgb_model,
                                        imagenet_pretrained=imagenet_pretrained,
                                        dropout_ratio=dropout_ratio)

    def forward(self, x_rgb, x_of):
        """Joint model forward."""
        x_rgb = self.model_rgb(x_rgb)
        x_of = self.model_of(x_of)
        x = torch.cat((x_rgb, x_of), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
