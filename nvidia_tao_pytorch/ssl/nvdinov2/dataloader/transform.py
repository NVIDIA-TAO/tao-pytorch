# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data transformation for NVDINOv2."""

from PIL import Image
from torchvision import transforms as T


class DinoV2Transform:
    """Data transformation for NVDINOv2."""

    def __init__(
        self,
        *,
        global_crops_number=2,
        global_crops_scale=(0.32, 1.0),
        global_crops_size=224,
        global_crops_identical=False,
        local_crops_number=8,
        local_crops_scale=(0.05, 0.32),
        local_crops_size=98,
        local_crops_identical=False,
    ):
        """Initialize data transformation for NVDINOv2.

        Args:
            global_crops_number (int, optional): Number of global crops to generate. Defaults to 2.
            global_crops_scale (tuple, optional): Scale range for global crops. Defaults to (0.32, 1.0).
            global_crops_size (int, optional): Size of global crops. Defaults to 224.
            global_crops_identical (bool, optional): If True, use the same transformation for all global crops. Defaults to False.
            local_crops_number (int, optional): Number of local crops to generate. Defaults to 8.
            local_crops_scale (tuple, optional): Scale range for local crops. Defaults to (0.05, 0.32).
            local_crops_size (int, optional): Size of local crops. Defaults to 98.
            local_crops_identical (bool, optional): If True, use the same transformation for all local crops. Defaults to False.
        """
        flip_and_color_jitter = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                T.RandomGrayscale(p=0.2),
            ]
        )

        normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_crops_number = global_crops_number

        # transformation for the first global crop
        self.global_transfo1 = T.Compose(
            [
                T.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                T.GaussianBlur(9, (0.1, 2.0)),
                normalize,
            ]
        )
        # global_crops

        # transformation for the rest of global crops
        self.global_transfo2 = T.Compose(
            [
                T.RandomResizedCrop(
                    global_crops_size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                T.RandomApply([T.GaussianBlur(9, (0.1, 2.0))], p=0.1),
                T.RandomSolarize(threshold=128, p=0.2),
                normalize,
            ]
        )
        self.global_crops_identical = global_crops_identical
        self.local_crops_identical = local_crops_identical

        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = T.Compose(
            [
                T.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=Image.BICUBIC,
                ),
                flip_and_color_jitter,
                T.RandomApply([T.GaussianBlur(9, (0.1, 2.0))], p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        """Apply transformations to the input image.

        Args:
            image (torch.Tensor): Input image

        Returns:
            Dict: The dictionary with global and local cropped image
        """
        crops = []
        crops.append(self.global_transfo1(image))

        for _ in range(self.global_crops_number - 1):
            if self.global_crops_identical:
                crops.append(crops[0])
            else:
                crops.append(self.global_transfo2(image))

        crops.append(self.local_transfo(image))
        for _ in range(self.local_crops_number - 1):
            if self.local_crops_identical:
                crops.append(crops[-1])
            else:
                crops.append(self.local_transfo(image))

        return {
            "global_crops": crops[:self.global_crops_number],
            "local_crops": crops[self.global_crops_number:],
        }
