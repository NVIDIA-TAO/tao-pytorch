# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backbone modules for Metric Learning Recognition model."""

from functools import partial
from typing import Optional

import torch
from torch import nn

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.cv.backbone_v2.dino_v2 import vit_large_patch14_dinov2_swiglu_legacy
from nvidia_tao_pytorch.cv.backbone_v2.fan import (
    fan_tiny_8_p4_hybrid, fan_small_12_p4_hybrid, fan_base_16_p4_hybrid, fan_large_16_p4_hybrid
)
from nvidia_tao_pytorch.cv.backbone_v2.resnet import (
    resnet_50, resnet_101
)

mlrecog_backbone_dict = {
    "nvdinov2_vit_large_legacy": partial(vit_large_patch14_dinov2_swiglu_legacy, num_classes=0),
    "fan_tiny": fan_tiny_8_p4_hybrid,
    "fan_small": fan_small_12_p4_hybrid,
    "fan_base": fan_base_16_p4_hybrid,  # input size does not matter
    "fan_large": fan_large_16_p4_hybrid,
    "resnet_50": resnet_50,
    "resnet_101": resnet_101,
}


class MLPSeq(nn.Module):
    """This block implements a series of MLP layers given the input sizes."""

    def __init__(self, layer_sizes, final_relu=False):
        """Initiates the sequential module of MLP layers.

        Args:
            layer_sizes (List[List]): a nested list of MLP layer sizes
            final_relu (Boolean): if True, a ReLu activation layer is added after each MLP layer.
        """
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(torch.nn.ReLU(inplace=True))
            layer_list.append(torch.nn.Linear(input_size, curr_size))
        self.net = torch.nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        """Sequential MLP forward."""
        return self.net(x)


class RecognitionBase(nn.Module):
    """Base model for Metric Learning Recognition model. The model consists of a
    backbone (trunk) and a feature extractor (embedder). The backbone has a softmax
    layer and it would be replaced by an identity layer.
    """

    def __init__(self, trunk, embedder):
        """Initiates the joint modules of the backbone and feature extractors.

        Args:
            embedder (torch.Module): The MLP layers with embedding vector outputs
            trunk (torch.Module): the backbone with fc layer removed
        """
        super(RecognitionBase, self).__init__()
        self.embedder = embedder.module
        self.trunk = trunk.module
        self.embedder_class = embedder
        self.trunk_class = trunk
        if self.trunk_class.output_size != self.embedder_class.input_feature_size:
            raise ValueError("The output size of the trunk and the input size of the embedder must match. trunk output size: {self.trunk_class.output_size}, embedder input size: {self.embedder_class.input_feature_size}")

    def forward(self, x):
        """Joint forward function for the backbone and the feature extractor."""
        features_extracted = self.trunk(x)
        output_embeds = self.embedder(features_extracted)
        return output_embeds

    def load_checkpoint(self, model_path):
        """Load paramaters for the model from a .pth format pretrained weights.

        Args:
            model_path (str): Model path.
        """
        param_dict = torch.load(model_path)
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]

        if "resnet" in self.trunk_class.model_name:
            self.__load_v1_checkpoint_backward(param_dict)
        elif "fan" in self.trunk_class.model_name or "nvdinov2_vit_large_legacy" in self.trunk_class.model_name:
            self.__load_v2_checkpoint(param_dict)
        else:
            raise NotImplementedError(f"Trunk model {self.trunk_class.model_name} is not supported.")

    def __load_v1_checkpoint_backward(self, param_dict):
        for i in param_dict:
            key_split = i.split(".")
            if key_split[0] == '0':
                j = "trunk" + "." + ".".join(key_split[1:])
            elif key_split[0] == '1':
                j = "embedder" + "." + ".".join(key_split[1:])
            else:
                j = i
            if j in self.state_dict(destination=None).keys():
                # check size matches:
                if self.state_dict(destination=None)[j].shape != param_dict[i].shape:
                    message = f"param size mismatch: {j}, current size: {self.state_dict(destination=None)[j].shape}, checkpoint size: {param_dict[i].shape}"
                    status_logging.get_status_logger().write(
                        message=message,
                        status_level=status_logging.Status.SKIPPED)
                else:
                    self.state_dict(destination=None)[j].copy_(param_dict[i])
            else:
                status_logging.get_status_logger().write(
                    message=f"param not found in model: {j}",
                    status_level=status_logging.Status.SKIPPED)

    def __load_v2_checkpoint(self, param_dict):
        for i in param_dict:
            j = i.replace("model.", "", 1)
            if j in self.state_dict(destination=None).keys():
                # check size matches:
                if self.state_dict(destination=None)[j].shape != param_dict[i].shape:
                    message = f"param size mismatch: {j}, current size: {self.state_dict(destination=None)[j].shape}, checkpoint size: {param_dict[i].shape}"
                    status_logging.get_status_logger().write(
                        message=message,
                        status_level=status_logging.Status.SKIPPED)
                else:
                    self.state_dict(destination=None)[j].copy_(param_dict[i])
            else:
                status_logging.get_status_logger().write(
                    message=f"param not found in model: {j}",
                    status_level=status_logging.Status.SKIPPED)


class Dualhead_embeddings(torch.nn.Module):
    """
    Dual head mlp training. The classifier head is temporarily removed."""

    def __init__(self, input_feature=1024):
        """Initiate the dual head mlp module.

        Args:
            input_feature (int): the input feature size.
        """
        super(Dualhead_embeddings, self).__init__()

        self.classifier_feat = nn.Sequential(
            nn.Linear(input_feature, input_feature // 2),
            nn.GELU(),
            nn.Linear(input_feature // 2, input_feature))
        self.classifier_feat2 = nn.Sequential(
            nn.Linear(input_feature, input_feature // 2),
            nn.GELU(),
            nn.Linear(input_feature // 2, input_feature))
        self.classifier_feat3 = nn.Sequential(
            nn.Linear(input_feature, input_feature // 2),
            nn.GELU(),
            nn.Linear(input_feature // 2, input_feature))
        self.classifier_feat4 = nn.Sequential(
            nn.Linear(input_feature, input_feature // 2),
            nn.GELU(),
            nn.Linear(input_feature // 2, input_feature))

        self.embedding_fc = nn.Sequential(
            nn.LayerNorm(input_feature),
            nn.Linear(input_feature, input_feature))

    def forward(self, input_features):
        """Forward function for the dual head mlp module."""
        cls_feat = self.classifier_feat(input_features) + input_features
        cls_feat2 = self.classifier_feat2(cls_feat) + cls_feat
        cls_feat3 = self.classifier_feat3(cls_feat2) + cls_feat2
        cls_feat4 = self.classifier_feat4(cls_feat3) + cls_feat3

        embeddings = self.embedding_fc(cls_feat4)
        return embeddings


class Trunk:
    """Creates backbone network and makes it compatible with Embedder layers"""

    def __init__(self,
                 model_name,
                 train_backbone: bool = True,
                 **kwargs):
        """Initiates the backbone trunk module.

        Args:
            model_name (str): the name of the backbone model.
            train_backbone (bool): if True, the backbone would be trained. Default True.
            **kwargs: other arguments for the backbone model.
        """
        self.model_name = model_name
        self.module = mlrecog_backbone_dict[model_name](**kwargs)
        self.trunk_fc_layer()  # get output size
        self.train_backbone = train_backbone
        if not train_backbone:
            self.freeze()

    def freeze(self):
        """Freezes the backbone model."""
        for param in self.module.parameters():
            param.requires_grad = False

    def trunk_fc_layer(self):
        """Gets the output size of the backbone model."""
        if "resnet" in self.model_name:
            self.output_size = self.module.fc.in_features
            self.module.fc = nn.Identity()
        else:
            self.output_size = self.module.embed_dim

    def load_checkpoint(self, model_path):
        """Loads pretrained weights for the backbone.

        Args:
            model_path (str): the path to the pretrained weights.
        """
        param_dict = torch.load(model_path)
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        if "resnet" in self.model_name or "nvdinov2_vit_large_legacy" in self.model_name:
            for i in param_dict:
                if i in self.module.state_dict(destination=None).keys():
                    # check size matches:
                    if self.module.state_dict(destination=None)[i].shape != param_dict[i].shape:
                        message = f"param size mismatch: {i}, current size: {self.module.state_dict(destination=None)[i].shape}, checkpoint size: {param_dict[i].shape}"
                        status_logging.get_status_logger().write(
                            message=message,
                            status_level=status_logging.Status.SKIPPED)
                    else:
                        self.module.state_dict(destination=None)[i].copy_(param_dict[i])
                else:
                    status_logging.get_status_logger().write(
                        message=f"param not found in model: {i}",
                        status_level=status_logging.Status.SKIPPED)

        elif "fan" in self.model_name:
            for i in param_dict:
                if "head." in i:
                    continue  # skip fan backbone head
                j = i.replace("backbone.", "", 1)  # remove first backbone. only
                if j in self.module.state_dict(destination=None).keys():
                    # check size matches:
                    if self.module.state_dict(destination=None)[j].shape != param_dict[i].shape:
                        message = f"param size mismatch: {j}, current size: {self.module.state_dict(destination=None)[j].shape}, checkpoint size: {param_dict[i].shape}"
                        status_logging.get_status_logger().write(
                            message=message,
                            status_level=status_logging.Status.SKIPPED)
                    else:
                        self.module.state_dict(destination=None)[j].copy_(param_dict[i])
                else:
                    status_logging.get_status_logger().write(
                        message=f"param not found in model: {j}",
                        status_level=status_logging.Status.SKIPPED)
        else:
            raise NotImplementedError(f"Trunk model {self.model_name} is not supported.")


class Embedder:
    """Creates the embedding layer or dual head network. """

    def __init__(self,
                 embedder_type,
                 input_feature_size: int,
                 output_feature_size: int,
                 train_embedder: Optional[bool] = True):
        """Initiates the embedder module.

        Args:
            embedder_type (str): the type of the embedder. It can be either `linear_head` or `dual_head`.
            input_feature_size (int): the input feature size.
            output_feature_size (int): the output feature size.
            train_embedder (Optional bool): if True, the embedder would be trained. Default True.
        """
        self.embedder_type = embedder_type
        head_module = None
        if embedder_type == "linear_head":
            head_module = MLPSeq([input_feature_size, output_feature_size])
            self.output_feature_size = output_feature_size
        elif embedder_type == "dual_head":
            head_module = Dualhead_embeddings(input_feature_size)
            self.output_feature_size = input_feature_size  # output feature size is the same as input feature size
        self.module = head_module
        self.input_feature_size = input_feature_size
        self.train_embedder = train_embedder
        if not train_embedder:
            self.freeze()

    def freeze(self):
        """Freezes the embedder module"""
        for param in self.module.parameters():
            param.requires_grad = False

    def load_checkpoint(self, model_path):
        """Loads pretrained weights for the embedder.

        Args:
            model_path (str): the path to the pretrained weights.
        """
        param_dict = torch.load(model_path)
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]
        for i in param_dict:
            if i in self.module.state_dict(destination=None).keys():
                # check size matches:
                if self.module.state_dict(destination=None)[i].shape != param_dict[i].shape:
                    message = f"param size mismatch: {i}, current size: {self.module.state_dict(destination=None)[i].shape}, checkpoint size: {param_dict[i].shape}"
                    status_logging.get_status_logger().write(
                        message=message,
                        status_level=status_logging.Status.SKIPPED)
                else:
                    self.module.state_dict(destination=None)[i].copy_(param_dict[i])
            else:
                status_logging.get_status_logger().write(
                    message=f"param not found in model: {i}",
                    status_level=status_logging.Status.SKIPPED)
