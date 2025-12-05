# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original implementation: https://github.com/lyuwenyu/RT-DETR
# Licensed under the Apache License 2.0
# Copyright (c) 2023 lyuwenyu. All Rights Reserved.

""" Transformation for RT-DETR."""

import torch

import torchvision
from torchvision import tv_tensors

import torchvision.transforms.v2 as T

from typing import Any, Dict, List, Optional

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.transforms import Compose, ResizeAndPad

torchvision.disable_beta_transforms_warning()


def build_transforms(augmentation_config, subtask_config=None, dataset_mode='train'):
    """Build Augmentations.

    Args:
        augmentation_config (dict): augmentation configuration.
        subtask_config (dict): subtask experiment configuration.
        dataset_mode (str): data mode (train, val, eval, infer).

    Returns:
        transforms (Compose): Final built transforms.

    Raises:
        If dataset_mode is set to other than given options (train, val, eval, infer), the code will raise the value error.
    """
    distortion_prob = augmentation_config.get("distortion_prob", 0.8)
    iou_crop_prob = augmentation_config.get("iou_crop_prob", 0.8)
    train_resize = augmentation_config.get("train_spatial_size", [640, 640])
    test_resize = augmentation_config.get("eval_spatial_size", [640, 640])
    preserve_aspect_ratio = augmentation_config["preserve_aspect_ratio"]

    if dataset_mode == 'train':
        transforms = T.Compose([
            T.RandomPhotometricDistort(p=distortion_prob),
            T.RandomZoomOut(fill=0),
            RandomIoUCrop(p=iou_crop_prob),
            T.SanitizeBoundingBoxes(min_size=1),
            T.RandomHorizontalFlip(),
            T.Resize(size=train_resize),
            T.ToImage(),
            T.ConvertImageDtype(),
            T.SanitizeBoundingBoxes(min_size=1),
            ConvertBox(out_fmt='cxcywh', normalize=True)
        ])
    elif dataset_mode in ('val', 'eval', 'infer'):
        if preserve_aspect_ratio:
            # Resize the longest edge to test_resize and zero pad the rest
            transforms = Compose([
                ResizeAndPad(max_size=max(test_resize)),
                T.ToImage(),
                T.ConvertImageDtype(),
                ConvertBox(out_fmt='cxcywh', normalize=True)
            ])
        else:
            transforms = Compose([
                T.Resize(size=test_resize),
                T.ToImage(),
                T.ConvertImageDtype(),
                ConvertBox(out_fmt='cxcywh', normalize=True)
            ])
    else:
        raise ValueError('There are only train, val, eval, and infer options in dataset_mode.')

    return transforms


class ConvertBox(T.Transform):
    """ConvertBox converts list to tv_tensors format."""

    _transformed_types = (
        tv_tensors.BoundingBoxes,
    )

    def __init__(self, out_fmt='', normalize=False) -> None:
        """Init function."""
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': tv_tensors.BoundingBoxFormat.XYXY,
            'cxcywh': tv_tensors.BoundingBoxFormat.CXCYWH
        }

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """transform function."""
        if self.out_fmt:
            spatial_size = inpt.canvas_size
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = tv_tensors.BoundingBoxes(inpt, format=self.data_fmt[self.out_fmt], canvas_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(inpt.canvas_size[::-1]).tile(2)[None]

        return inpt


class RandomIoUCrop(T.RandomIoUCrop):
    """RandomIoUCrop from torchvision.v2."""

    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5,
                 max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        """Init function."""
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        """__call__ function."""
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)
