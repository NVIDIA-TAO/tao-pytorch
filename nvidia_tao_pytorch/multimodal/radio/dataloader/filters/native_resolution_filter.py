# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filter decoded samples by native image resolution."""

import logging
from typing import Optional

import webdataset as wds

logger = logging.getLogger(__name__)


class NativeResolutionFilter(wds.PipelineStage):
    """Drop samples whose decoded image is too small or extreme.

    The image is expected at position ``image_tuple_idx`` in the sample tuple
    after ``wds.to_tuple``. Filtering here happens before resize/crop, so the
    remaining samples can support high-resolution crop training without
    upsampling low-resolution originals.
    """

    def __init__(
        self,
        image_tuple_idx: int = 0,
        min_short_side: Optional[int] = None,
        min_long_side: Optional[int] = None,
        min_area: Optional[int] = None,
        max_aspect_ratio: Optional[float] = None,
        verbose: bool = False,
    ):
        """Initialize the native-resolution filter.

        Args:
            image_tuple_idx (int): Position of the decoded image in each
                tuple sample.
            min_short_side (Optional[int]): Minimum allowed short-side length
                in pixels. ``None`` disables the check.
            min_long_side (Optional[int]): Minimum allowed long-side length in
                pixels. ``None`` disables the check.
            min_area (Optional[int]): Minimum allowed image area in pixels.
                ``None`` disables the check.
            max_aspect_ratio (Optional[float]): Maximum allowed long-side to
                short-side ratio. ``None`` disables the check.
            verbose (bool): Whether to log every filtered sample instead of
                periodic progress.

        Returns:
            None: The filter thresholds and counters are initialized in place.
        """
        super().__init__()
        self.image_tuple_idx = image_tuple_idx
        self.min_short_side = min_short_side
        self.min_long_side = min_long_side
        self.min_area = min_area
        self.max_aspect_ratio = max_aspect_ratio
        self.verbose = verbose
        self.num_seen = 0
        self.num_filtered = 0

    def _keep(self, width: int, height: int) -> bool:
        """Check whether an image satisfies the configured bounds.

        Args:
            width (int): Native image width in pixels.
            height (int): Native image height in pixels.

        Returns:
            bool: ``True`` when the image should remain in the stream.
        """
        short_side = min(width, height)
        long_side = max(width, height)

        if self.min_short_side is not None and short_side < self.min_short_side:
            return False
        if self.min_long_side is not None and long_side < self.min_long_side:
            return False
        if self.min_area is not None and width * height < self.min_area:
            return False
        if self.max_aspect_ratio is not None and short_side > 0:
            if long_side / short_side > self.max_aspect_ratio:
                return False

        return True

    def run(self, src):
        """Yield samples that satisfy the configured native-resolution bounds.

        Args:
            src (Iterable[tuple]): Source WebDataset sample stream after
                tuple conversion.

        Yields:
            tuple: Samples whose decoded image passes the native-resolution
                checks.
        """
        for sample in src:
            image = sample[self.image_tuple_idx]
            width, height = image.size
            self.num_seen += 1

            if self._keep(width, height):
                yield sample
                continue

            self.num_filtered += 1
            if self.verbose or (self.num_filtered % 100) == 0:
                pct_filtered = self.num_filtered / self.num_seen * 100
                logger.info(
                    "Filtered native-resolution image. size=%sx%s, "
                    "filtered=%s/%s (%.3f%%)",
                    width,
                    height,
                    self.num_filtered,
                    self.num_seen,
                    pct_filtered,
                )
