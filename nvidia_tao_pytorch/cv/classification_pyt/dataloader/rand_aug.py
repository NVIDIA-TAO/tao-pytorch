# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rand Augmentation Module."""
from typing import Any

import numpy as np
from PIL import Image
from timm.data.auto_augment import rand_augment_transform


class ImageAug:
    """Image aug base class."""

    def aug_image(self, image: Image.Image) -> Image.Image:
        """Augment an image."""
        raise NotImplementedError("This is an abstract method.")

    def __call__(self, feed_dict: dict | np.ndarray | Image.Image) -> dict | np.ndarray | Image.Image:
        """Call."""
        if isinstance(feed_dict, dict):
            output_dict = feed_dict
            image = feed_dict[self.key]
        else:
            output_dict = None
            image = feed_dict
        is_ndarray = isinstance(image, np.ndarray)
        if is_ndarray:
            image = Image.fromarray(image)

        image = self.aug_image(image)

        if is_ndarray:
            image = np.array(image)

        if output_dict is None:
            return image
        else:
            output_dict[self.key] = image
            return output_dict


class RandAug(ImageAug):
    """Random Aug."""

    def __init__(self, config: dict[str, Any], mean: tuple[float, float, float], key="img"):
        """
        Args:
            config (dict):  Configurations for RandAug
            mean (tuple): image mean
        """
        # TODO(@yuw): verify key
        n = config.get("n", 2)
        m = config.get("m", 5)
        mstd = config.get("mstd", 1.0)
        inc = config.get("inc", 1)
        tpct = config.get("tpct", 0.45)
        config_str = f"rand-n{n}-m{m}-mstd{mstd}-inc{inc}"

        aa_params = dict(
            translate_pct=tpct,
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
            interpolation=Image.BICUBIC,
        )
        self.aug_op = rand_augment_transform(config_str, aa_params)
        self.key = key

    def aug_image(self, image: Image.Image) -> Image.Image:
        """Augment an image."""
        return self.aug_op(image)

    def __repr__(self):
        """repr."""
        return self.aug_op.__repr__()
