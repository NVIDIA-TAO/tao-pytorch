# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" CenterPose model. """

from torch import nn
import numpy as np

from nvidia_tao_pytorch.cv.centerpose.model.model_utils import ConvGRU, DLAUp, IDAUp, fill_fc_weights, get_dla_base, group_norm
from nvidia_tao_pytorch.cv.dino.model.fan import fan_model_dict
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import load_pretrained_weights
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank


class CenterPose_DLA34(nn.Module):
    """ This method mainly focuses on the Category-Level 6-DoF pose estimation from a single RGB image. """

    def __init__(self, model_config):
        """ Initializes the CenterPose model with DLA34 backbone.
        """
        super(CenterPose_DLA34, self).__init__()

        down_ratio = model_config.down_ratio
        final_kernel = model_config.final_kernel
        last_level = model_config.last_level
        head_conv = model_config.head_conv
        out_channel = model_config.out_channel
        use_convGRU = model_config.use_convGRU

        assert down_ratio in [2, 4, 8, 16], "only support downsample ratio in [2, 4, 8, 16]"
        self.heads = {'hm': 1, 'wh': 2, 'hps': 16, 'reg': 2, 'hm_hp': 8, 'hp_offset': 2, 'scale': 3}

        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = get_dla_base(pretrained=model_config.use_pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        self.use_convGRU = use_convGRU
        if self.use_convGRU is True:
            self.convGRU = ConvGRU(input_channels=channels[self.first_level],
                                   hidden_channels=[64, ],
                                   kernel_size=3, step=3,
                                   effective_step=[0, 1, 2])

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:

                modules = []
                modules.append(nn.Conv2d(channels[self.first_level], head_conv,
                                         kernel_size=3, padding=1, bias=True))

                if self.use_convGRU is True:
                    modules.append(group_norm(head_conv))

                modules.append(nn.ReLU(inplace=True))
                modules.append(nn.Conv2d(head_conv, classes,
                                         kernel_size=final_kernel, stride=1,
                                         padding=final_kernel // 2, bias=True))

                fc = nn.Sequential(*modules)

                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        """ Forward function of CenterPose Model

        Args:
            samples (torch.Tensor): batched images, of shape [batch_size x 3 x H x W]

        Returns:
            z (Dict): the heatmaps indicate the keypoints location.

        """
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}

        # Feature association
        if self.use_convGRU is True:
            gru_outputs, _ = self.convGRU(y[-1])

            for head in self.heads:
                if head in ['hm', 'wh', 'reg']:
                    z[head] = self.__getattr__(head)(gru_outputs[0])
                if head in ['hm_hp', 'hp_offset', 'hps']:
                    z[head] = self.__getattr__(head)(gru_outputs[1])
                if head == 'scale':
                    z[head] = self.__getattr__(head)(gru_outputs[2])

        else:
            for head in self.heads:
                z[head] = self.__getattr__(head)(y[-1])

        return [z]


class CenterPose_FAN(nn.Module):
    """ This method mainly focuses on the Category-Level 6-DoF pose estimation from a single RGB image. """

    def __init__(self, arch, model_config, out_channel=128, first_level=0, last_level=4):
        """ Initializes the CenterPose model with FAN backbone.
        """
        super(CenterPose_FAN, self).__init__()

        down_ratio = model_config.down_ratio
        final_kernel = model_config.final_kernel
        head_conv = model_config.head_conv
        use_convGRU = model_config.use_convGRU
        pretrained_backbone_path = model_config.backbone.pretrained_backbone_path
        self.first_level = first_level
        self.last_level = last_level

        assert down_ratio in [2, 4, 8, 16], "only support downsample ratio in [2, 4, 8, 16]"
        self.heads = {'hm': 1, 'wh': 2, 'hps': 16, 'reg': 2, 'hm_hp': 8, 'hp_offset': 2, 'scale': 3}

        self.base = fan_model_dict[arch](out_indices=[0, 1, 2, 3],
                                         activation_checkpoint=True)

        if model_config.use_pretrained and pretrained_backbone_path:
            checkpoint = load_pretrained_weights(pretrained_backbone_path)
            _tmp_st_output = self.base.load_state_dict(checkpoint, strict=False)
            if get_global_rank() == 0:
                print(f"Loaded pretrained weights from {pretrained_backbone_path}")
                print(f"{_tmp_st_output}")

        channels = np.array(self.base.out_channels)

        self.use_convGRU = use_convGRU
        if self.use_convGRU is True:
            self.convGRU = ConvGRU(input_channels=channels[self.first_level],
                                   hidden_channels=[128, ],
                                   kernel_size=3, step=3,
                                   effective_step=[0, 1, 2])
        self.ida_up = IDAUp(out_channel, channels,
                            [2 ** i for i in range(4)])

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:

                modules = []
                modules.append(nn.Conv2d(channels[self.first_level], head_conv,
                                         kernel_size=3, padding=1, bias=True))

                if self.use_convGRU is True:
                    modules.append(group_norm(head_conv))

                modules.append(nn.ReLU(inplace=True))
                modules.append(nn.Conv2d(head_conv, classes,
                                         kernel_size=final_kernel, stride=1,
                                         padding=final_kernel // 2, bias=True))

                fc = nn.Sequential(*modules)

                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        """ Forward function of CenterPose Model

        Args:
            samples (torch.Tensor): batched images, of shape [batch_size x 3 x H x W]

        Returns:
            z (Dict): the heatmaps indicate the keypoints location.

        """
        x = self.base.forward_feature_pyramid(x)
        out = list(x.values())

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(out[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}

        # Feature association
        if self.use_convGRU is True:
            gru_outputs, _ = self.convGRU(y[-1])

            for head in self.heads:
                if head in ['hm', 'wh', 'reg']:
                    z[head] = self.__getattr__(head)(gru_outputs[0])
                if head in ['hm_hp', 'hp_offset', 'hps']:
                    z[head] = self.__getattr__(head)(gru_outputs[1])
                if head == 'scale':
                    z[head] = self.__getattr__(head)(gru_outputs[2])

        else:
            for head in self.heads:
                z[head] = self.__getattr__(head)(y[-1])

        return [z]


def create_model(model_config):
    """Initialize the CenterPose model based on the backbone structures"""
    arch = model_config.backbone.model_type
    if arch in ['fan_small', 'fan_base', 'fan_large']:
        model = CenterPose_FAN(arch, model_config)
    elif arch in ['DLA34']:
        model = CenterPose_DLA34(model_config)
    else:
        raise ValueError(f"{arch} is not available backbone, please use one of the following backbone ['fan_tiny', 'fan_small', 'fan_base', 'fan_large', 'DLA34']")
    return model


class CenterPoseWrapped(nn.Module):
    """ This module mainly wrapped the CenterPose model and heatmap decoder into the ONNX model. """

    def __init__(self, model, hm_decoder):
        """ Wrapped the CenterPose model with the heatmap decoder.
        """
        super(CenterPoseWrapped, self).__init__()
        self.model = model
        self.hm_decoder = hm_decoder

    def forward(self, x):
        """ Forward function of Wrapped CenterPose Model

        Args:
            samples (torch.Tensor): batched images, of shape [batch_size x 3 x H x W]

        Returns:
            z (Dict): the decoded heatmaps indicate the keypoints location.

        """
        outputs = self.model(x)
        dets = self.hm_decoder(outputs)
        return dets
