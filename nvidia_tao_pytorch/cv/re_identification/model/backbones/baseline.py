# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline Module for Re-Identification."""

import copy
import torch
from torch import nn
from nvidia_tao_pytorch.cv.re_identification.model.backbones.resnet import Bottleneck, ResNet, BasicBlock
from nvidia_tao_pytorch.cv.re_identification.model.losses.metric_learning import Arcface, CircleLoss, Cosface, AMSoftmax


def shuffle_unit(features, shift, group, begin=1):
    """Shuffle and patch operation for feature manipulation.

    This function performs a two-step operation on the given tensor `features`:
    1. A cyclic shift by a specified number.
    2. A reshuffle operation on feature patches based on the specified group parameter.

    Args:
        features (torch.Tensor): The input tensor whose last dimension is treated as the feature dimension.
        shift (int): Number of positions by which the features are cyclically shifted.
        group (int): Number of groups for the patch shuffle operation.
        begin (int, optional): Starting position for the shift operation. Default is 1.

    Returns:
        torch.Tensor: The shuffled tensor after applying both shift and patch shuffle operations.

    """
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except Exception:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


def weights_init_xavier(m):
    """Initialize weights of the model using Xavier (Glorot) Initialization.

    Arguments:
        m (torch.nn.Module): The PyTorch module (layer) whose weights need to be initialized.

    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    """Initializes weights using Kaiming Normal initialization.

    Args:
        m (torch.nn.Module): PyTorch module whose weights are to be initialized.

    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """Initializes the weights of a classifier layer.

    Args:
        m (torch.nn.Module): PyTorch module whose weights are to be initialized.

    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    """Baseline model for re-identification tasks.

    This class generates a model based on the provided configuration. The model
    is primarily a ResNet variant, with additional features like bottleneck and classifier
    layers. The ResNet architecture can be one of the following variants: 18, 34, 50, 101, 152.

    Attributes:
        in_planes (int): Dimensionality of the input features.
        base (ResNet): Base ResNet model.
        gap (torch.nn.AdaptiveAvgPool2d): Global Average Pooling layer.
        num_classes (int): Number of output classes.
        neck (str): Specifies the neck architecture of the model.
        neck_feat (str): Specifies whether neck features are used.
        if_flip_feat (bool): Whether to flip the features or not.
        classifier (torch.nn.Linear): Classifier layer of the model.
        bottleneck (torch.nn.BatchNorm1d): Optional bottleneck layer of the model.
    """

    def __init__(self, cfg, num_classes):
        """Initializes the Baseline model with provided configuration and number of classes.

        Args:
            cfg (DictConfig): Configuration object containing model parameters.
            num_classes (int): Number of output classes.
        """
        super(Baseline, self).__init__()
        self.in_planes = cfg['model']['feat_dim']
        if "resnet" in cfg['model']['backbone']:

            arch_settings = {
                'resnet_18': (BasicBlock, [2, 2, 2, 2]),
                'resnet_34': (BasicBlock, [3, 4, 6, 3]),
                'resnet_50': (Bottleneck, [3, 4, 6, 3]),
                'resnet_101': (Bottleneck, [3, 4, 23, 3]),
                'resnet_152': (Bottleneck, [3, 8, 36, 3])
            }

            self.base = ResNet(feat_dim=cfg['model']['feat_dim'], last_stride=cfg['model']['last_stride'],
                               block=Bottleneck,
                               layers=arch_settings[cfg['model']['backbone']][1])

        if cfg['model']['pretrain_choice'] == 'imagenet':
            if cfg['model']['pretrained_model_path']:
                self.base.load_param(cfg['model']['pretrained_model_path'])
                print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = cfg['model']['neck']
        self.neck_feat = cfg['model']['neck_feat']
        self.if_flip_feat = cfg['model']['with_flip_feature']

        if not self.neck:
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        """Defines the forward pass of the Baseline model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after forward pass. This could be feature embeddings
            or the sum of feature embeddings in case of flipped features.
        """
        if self.training:
            return self.__forward(x)
        if self.if_flip_feat:
            y = torch.flip(x, [3])
            feat1 = self.__forward(y)
            feat2 = self.__forward(x)
            return feat2 + feat1
        return self.__forward(x)

    def __forward(self, x):
        """Internal method for processing the features through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing. This could be the class scores
            and global features during training or the feature embeddings during testing.
        """
        global_feat = self.gap(self.base(x))
        global_feat = global_feat.view(global_feat.shape[0], -1)

        if not self.neck:
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        if self.neck_feat == 'after':
            # cls_score = self.classifier(feat)
            return feat
            # return cls_score, global_feat  # global feature for triplet loss
        return global_feat


class Transformer(nn.Module):
    """A module that integrates Transformer-based architectures for re-identification tasks.
    The module supports various cosine-based loss functions including ArcFace, CosFace, AMSoftmax, and Circle loss.
    """

    def __init__(self, num_classes, camera_num, view_num, cfg, factory, semantic_weight):
        """Initialize the Transformer class for re-identification.

        Args:
            num_classes (int): The number of target classes.
            camera_num (int): Number of cameras.
            view_num (int): Number of views.
            cfg (Config): A configuration object containing model and training parameters.
            factory (dict): A factory dictionary mapping model names to their corresponding classes/functions.
            semantic_weight (float): Weight for the semantic component.
        """
        super(Transformer, self).__init__()
        model_path = cfg["model"]["pretrained_model_path"]
        pretrain_choice = cfg["model"]["pretrain_choice"]
        self.cos_layer = cfg["model"]["cos_layer"]
        self.neck = cfg["model"]["neck"]
        self.neck_feat = cfg["model"]["neck_feat"]
        self.reduce_feat_dim = cfg["model"]["reduce_feat_dim"]
        self.feat_dim = cfg["model"]["feat_dim"]
        self.dropout_rate = cfg["model"]["dropout_rate"]

        print('using {} as a backbone'.format(cfg["model"]["backbone"]))

        if cfg["model"]["sie_camera"]:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg["model"]["sie_view"]:
            view_num = view_num
        else:
            view_num = 0

        convert_weights = pretrain_choice == 'imagenet'
        self.base = factory[cfg["model"]["backbone"]](img_size=[cfg["model"]["input_height"], cfg["model"]["input_width"]],
                                                      drop_path_rate=cfg["model"]["drop_path"],
                                                      drop_rate=cfg["model"]["drop_out"],
                                                      attn_drop_rate=cfg["model"]["att_drop_rate"],
                                                      pretrained=model_path,
                                                      convert_weights=convert_weights,
                                                      semantic_weight=semantic_weight)
        if model_path != '':
            self.base.init_weights(model_path)
        self.in_planes = self.base.num_features[-1]

        self.num_classes = num_classes
        self.id_loss_type = cfg["model"]["id_loss_type"]
        if self.id_loss_type == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.id_loss_type, cfg["train"]["optim"]["cosine_scale"], cfg["train"]["optim"]["cosine_margin"]))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg["train"]["optim"]["cosine_scale"], m=cfg["train"]["optim"]["cosine_margin"])
        elif self.id_loss_type == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.id_loss_type, cfg["train"]["optim"]["cosine_scale"], cfg["train"]["optim"]["cosine_margin"]))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg["train"]["optim"]["cosine_scale"], m=cfg["train"]["optim"]["cosine_margin"])
        elif self.id_loss_type == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.id_loss_type, cfg["train"]["optim"]["cosine_scale"], cfg["train"]["optim"]["cosine_margin"]))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg["train"]["optim"]["cosine_scale"], m=cfg["train"]["optim"]["cosine_margin"])
        elif self.id_loss_type == 'circle':
            print('using {} with s:{}, m: {}'.format(self.id_loss_type, cfg["train"]["optim"]["cosine_scale"], cfg["train"]["optim"]["cosine_margin"]))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg["train"]["optim"]["cosine_scale"], m=cfg["train"]["optim"]["cosine_margin"])
        else:
            if self.reduce_feat_dim:
                self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
                self.fcneck.apply(weights_init_xavier)
                self.in_planes = cfg.model.feat_dim
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(self.dropout_rate)

        # if pretrain_choice == 'self':
        #     self.load_param(model_path)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        """
        Forward propagation through the network.

        Args:
            x (torch.Tensor): Input tensor.
            label (torch.Tensor, optional): Ground truth labels for the input samples. Required for training. Defaults to None.
            cam_label (torch.Tensor, optional): Camera labels. Defaults to None.
            view_label (torch.Tensor, optional): View labels. Defaults to None.

        Returns:
            During training:
                torch.Tensor: Classification scores.
                torch.Tensor: Global features which can be used for losses like triplet loss.
                torch.Tensor: Feature maps.
            During testing:
                torch.Tensor: Feature (before or after the bottleneck layer, based on configuration).
                torch.Tensor: Feature maps.
        """
        global_feat, featmaps = self.base(x)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)
        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)

        if self.training:
            if self.id_loss_type in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            return cls_score, global_feat, featmaps  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat, featmaps
            return global_feat, featmaps

    def load_param(self, trained_path):
        """
        Load trained parameters into the model.

        Args:
            trained_path (str): Path to the trained parameters.
        """
        param_dict = torch.load(trained_path, map_location='cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])  # noqa # pylint: disable=missing-kwoa
            except Exception:
                continue
        print('Loading pretrained model from {}'.format(trained_path))


class TransformerLocal(nn.Module):
    """A module that integrates a Transformer-based architecture with specialized local attention mechanisms for re-identification tasks.
    This model is enhanced with support for multiple branches to capture local features in addition to global ones.
    """

    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        """Initialize the TransformerLocal class.

        Args:
            num_classes (int): The number of target classes.
            camera_num (int): Number of cameras.
            view_num (int): Number of views.
            cfg (Config): A configuration object containing model and training parameters.
            factory (dict): A factory dictionary mapping model names to their corresponding classes/functions.
            rearrange (bool): Flag to decide if feature rearrangement should be applied.
        """
        super(TransformerLocal, self).__init__()
        model_path = cfg.model.pretrained_model_path
        pretrain_choice = cfg.model.pretrain_choice
        self.cos_layer = cfg.model.cos_layer
        self.neck = cfg.model.neck
        self.neck_feat = cfg.model.neck_feat

        print('using {} as a backbone'.format(cfg.model.backbone))

        if cfg.model.sie_camera:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.model.sie_view:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.model.backbone](img_size=[cfg.model.input_height, cfg.model.input_width], sie_xishu=cfg.model.sie_coe, local_feature=cfg.model.jpm, camera=camera_num, view=view_num, stride_size=cfg.model.stride_size, drop_path_rate=cfg.model.drop_path)
        self.in_planes = self.base.in_planes
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path, hw_ratio=cfg.model.pretrain_hw_ratio)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.id_loss_type = cfg.model.id_loss_type
        if self.id_loss_type == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.id_loss_type, cfg["train"]["optim"]["cosine_scale"], cfg["train"]["optim"]["cosine_margin"]))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg["train"]["optim"]["cosine_scale"], m=cfg["train"]["optim"]["cosine_margin"])
        elif self.id_loss_type == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.id_loss_type, cfg["train"]["optim"]["cosine_scale"], cfg["train"]["optim"]["cosine_margin"]))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg["train"]["optim"]["cosine_scale"], m=cfg["train"]["optim"]["cosine_margin"])
        elif self.id_loss_type == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.id_loss_type, cfg["train"]["optim"]["cosine_scale"], cfg["train"]["optim"]["cosine_margin"]))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg["train"]["optim"]["cosine_scale"], m=cfg["train"]["optim"]["cosine_margin"])
        elif self.id_loss_type == 'circle':
            print('using {} with s:{}, m: {}'.format(self.id_loss_type, cfg["train"]["optim"]["cosine_scale"], cfg["train"]["optim"]["cosine_margin"]))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg["train"]["optim"]["cosine_scale"], m=cfg["train"]["optim"]["cosine_margin"])
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.model.shuffle_group
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.model.shift_num
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.model.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'
        """Forward propagation through the network, featuring multiple branches for local feature extraction.

        Args:
            x (torch.Tensor): Input tensor.
            label (torch.Tensor, optional): Ground truth labels for the input samples. Required for training. Defaults to None.
            cam_label (torch.Tensor, optional): Camera labels. Defaults to None.
            view_label (torch.Tensor, optional): View labels. Defaults to None.

        Returns:
            During training:
                List[torch.Tensor]: A list containing classification scores from global and local branches.
                List[torch.Tensor]: A list containing global and local features.
            During testing:
                torch.Tensor: Concatenated features from global and local branches.
        """
        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features)  # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length * 2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length * 3:patch_length * 4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.id_loss_type in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4], \
                [global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            return torch.cat(
                [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4],
                dim=1
            )

    def load_param(self, trained_path):
        """Load trained parameters into the model.

        Args:
            trained_path (str): Path to the trained parameters.
        """
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])  # noqa # pylint: disable=missing-kwoa
        print('Loading pretrained model from {}'.format(trained_path))
