# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections import OrderedDict
from typing import Tuple, Union
import torch
import torchvision.transforms as T
from torch.nn import functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList

logger = logging.getLogger(__name__)


class FrozenBackbone(Backbone):
    """Backbone implement following for any Detectron2 backbone"""

    def __init__(
        self,
        net: Backbone,
        backbone_in_size: Union[int, Tuple[int]] = (512, 512),
        pretrain_checkpoint=None,
        min_stride=None,
        max_stride=None,
        slide_training: bool = False,
    ):
        super().__init__()
        self.net = net

        if not os.environ.get("VISION_NO_PRETRAIN", ""):
            DetectionCheckpointer(self.net).load(pretrain_checkpoint)

        if backbone_in_size is None:
            self.image_preprocess = T.Compose([])
            self._slide_inference = False
        elif isinstance(backbone_in_size, int):
            self.image_preprocess = T.Resize(
                size=backbone_in_size, max_size=1280, interpolation=T.InterpolationMode.BICUBIC
            )
            self.backbone_in_size = (backbone_in_size, backbone_in_size)
            self._slide_inference = False
        else:
            self.image_preprocess = T.Resize(
                size=tuple(backbone_in_size), interpolation=T.InterpolationMode.BICUBIC
            )
            self.backbone_in_size = tuple(backbone_in_size)
            self._slide_inference = True

        self._slide_training = slide_training
        if self._slide_training:
            assert self._slide_inference, "slide training must be used with slide inference"

        self.min_stride = min_stride
        self.max_stride = max_stride

        logger.info(
            f"backbone_in_size: {backbone_in_size}, "
            f"slide_training: {self._slide_training}, \n"
            f"slide_inference: {self._slide_inference}, \n"
            f"out_feature_channels: {self._out_feature_channels}\n"
            f"out_feature_strides: {self._out_feature_strides}\n"
            f"min_stride: {self.min_stride}\n"
            f"max_stride: {self.max_stride}\n"
        )

        self._freeze()

    def train(self, mode: bool = True):
        super().train(mode=False)
        self._freeze()
        return self

    def _freeze(self):
        super().train(mode=False)
        for p in self.parameters():
            p.requires_grad = False

    @property
    def size_divisibility(self) -> int:
        return self.net.size_divisibility

    @property
    def _out_features(self):
        return self.net._out_features

    @property
    def _out_feature_channels(self):
        return self.net._out_feature_channels

    @property
    def _out_feature_strides(self):
        strides = {}
        for name in self._out_features:
            stride = self.net._out_feature_strides[name]

            if self.min_stride is not None:
                stride = max(stride, self.min_stride)
            if self.max_stride is not None:
                stride = min(stride, self.max_stride)

            strides[name] = stride

        return strides

    def ignored_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self._modules.items():
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return super().state_dict(destination=destination, prefix=prefix)

    def single_forward(self, img):

        # save memory
        input_image_size = img.shape[-2:]
        # print("input_image_size:", img.shape)
        img = self.image_preprocess(img)
        # print("processed_image_size:", img.shape)
        img = ImageList.from_tensors(list(img), self.size_divisibility).tensor
        # print("padded size:", img.shape)
        features = self.net(img)

        return self.forward_features(features, input_image_size)

    def forward_features(self, features, input_image_size):
        for name in self._out_features:
            stride = self._out_feature_strides[name]

            if self.min_stride is not None:
                stride = max(stride, self.min_stride)

            if self.max_stride is not None:
                stride = min(stride, self.max_stride)

            # print("before resize", name, features[name].shape)
            features[name] = F.interpolate(
                features.pop(name),
                size=(input_image_size[-2] // stride, input_image_size[-1] // stride),
            )
            # print("after resize", name, features[name].shape)

        return features

    def slide_forward_fast(self, img):

        batch_size, _, h_img, w_img = img.shape
        # output_features = {k: torch.zeros_like(v) for k, v in self.single_forward(img).items()}
        output_features = {}
        for k in self._out_features:
            stride = self._out_feature_strides[k]
            channel = self._out_feature_channels[k]
            output_features[k] = torch.zeros(
                (batch_size, channel, h_img // stride, w_img // stride),
                dtype=img.dtype,
                device=img.device,
            )
        count_mats = {k: torch.zeros_like(v) for k, v in output_features.items()}

        if self._slide_training:
            short_side = min(self.backbone_in_size)
        else:
            # if not slide training then use the shorter side to crop
            short_side = min(img.shape[-2:])

        # h_img, w_img = img.shape[-2:]

        h_crop = w_crop = short_side

        h_stride = w_stride = short_side

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        crop_imgs = []

        grid2batch_idx = {}
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                assert crop_img.shape[-2:] == (h_crop, w_crop)

                grid2batch_idx[(h_idx, w_idx)] = len(crop_imgs)

                crop_imgs.append(crop_img)

        crop_imgs = torch.cat(crop_imgs, dim=0)
        all_crop_features = self.single_forward(crop_imgs)

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_features = {
                    k: v[
                        grid2batch_idx[(h_idx, w_idx)] : grid2batch_idx[(h_idx, w_idx)] + batch_size
                    ]
                    for k, v in all_crop_features.items()
                }
                for k in crop_features:
                    k_x1 = x1 // self._out_feature_strides[k]
                    k_x2 = x2 // self._out_feature_strides[k]
                    k_y1 = y1 // self._out_feature_strides[k]
                    k_y2 = y2 // self._out_feature_strides[k]
                    output_features[k][:, :, k_y1:k_y2, k_x1:k_x2] += crop_features[k]
                    count_mats[k][..., k_y1:k_y2, k_x1:k_x2] += 1
        assert all((count_mats[k] == 0).sum() == 0 for k in count_mats)

        for k in output_features:
            output_features[k] /= count_mats[k]

        return output_features

    def slide_forward(self, img):

        batch_size, _, h_img, w_img = img.shape
        # output_features = {k: torch.zeros_like(v) for k, v in self.single_forward(img).items()}
        output_features = {}
        for k in self._out_features:
            stride = self._out_feature_strides[k]
            channel = self._out_feature_channels[k]
            output_features[k] = torch.zeros(
                (batch_size, channel, h_img // stride, w_img // stride),
                dtype=img.dtype,
                device=img.device,
            )
        count_mats = {k: torch.zeros_like(v) for k, v in output_features.items()}

        if self._slide_training:
            short_side = min(self.backbone_in_size)
        else:
            # if not slide training then use the shorter side to crop
            short_side = min(img.shape[-2:])

        # h_img, w_img = img.shape[-2:]

        h_crop = w_crop = short_side

        h_stride = w_stride = short_side

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        # print("img.shape:", img.shape)
        # for k in output_features:
        #     print(k, output_features[k].shape)
        # print("h_grids:", h_grids, "w_grids:", w_grids)
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                assert crop_img.shape[-2:] == (h_crop, w_crop)
                # print("crop_img.shape:", crop_img.shape)
                crop_features = self.single_forward(crop_img)
                for k in crop_features:
                    k_x1 = x1 // self._out_feature_strides[k]
                    k_x2 = x2 // self._out_feature_strides[k]
                    k_y1 = y1 // self._out_feature_strides[k]
                    k_y2 = y2 // self._out_feature_strides[k]
                    # output_features[k] += F.pad(
                    #     crop_features[k],
                    #     (
                    #         k_x1,
                    #         output_features[k].shape[-1] - k_x1 - crop_features[k].shape[-1],
                    #         k_y1,
                    #         output_features[k].shape[-2] - k_y1 - crop_features[k].shape[-2],
                    #     ),
                    # )
                    # this version should save some memory
                    output_features[k][:, :, k_y1:k_y2, k_x1:k_x2] += crop_features[k]
                    count_mats[k][..., k_y1:k_y2, k_x1:k_x2] += 1
        assert all((count_mats[k] == 0).sum() == 0 for k in count_mats)

        for k in output_features:
            output_features[k] /= count_mats[k]

        return output_features

    def forward(self, img):
        if (self.training and not self._slide_training) or not self._slide_inference:
            return self.single_forward(img)
        else:
            return self.slide_forward(img)
            # return self.slide_forward_fast(img)
