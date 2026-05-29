# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Init Stereo loss module."""
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.foundation_stereo import FoundationStereo
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.fast_foundation_stereo.fast_foundation_stereo import FastFoundationStereo
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.loss import SequenceLoss


class StereoDepthNet:
    """Placeholder class to return the model and loss class"""

    @staticmethod
    def get_model():
        """static function to return model class and loss"""
        # Why: registry keyed on lower-cased model_type (see pl_stereo_model.py).
        # FastFoundationStereo (FFS) reuses SequenceLoss; only the model class differs.
        return {
            'foundationstereo': (FoundationStereo, SequenceLoss),
            'fastfoundationstereo': (FastFoundationStereo, SequenceLoss),
        }
