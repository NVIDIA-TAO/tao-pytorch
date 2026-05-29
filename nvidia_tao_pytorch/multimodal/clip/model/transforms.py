# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image transforms for CLIP-compatible model training.

This module provides image preprocessing transforms for various model
architectures.

Classes:
    SigLIP2ImageTransform: Image transform using HuggingFace SigLIP2 processor
"""


class SigLIP2ImageTransform:
    """Image transform using HuggingFace SigLIP2 processor.

    This transform uses the native HF processor which handles:
    - Image resizing (fixed or NaFlex variable resolution)
    - Normalization
    - Creating pixel_attention_mask and spatial_shapes for NaFlex models

    Returns a dict with all processor outputs for proper handling of
    NaFlex models.

    Args:
        processor: HuggingFace processor instance
        is_train: Whether this transform is for training (default: True)
    """

    def __init__(self, processor, is_train=True):
        """Initialize the image transform.

        Args:
            processor: HuggingFace AutoProcessor instance for SigLIP2
            is_train: Whether this is for training data
                (may affect augmentations)
        """
        self._processor = processor
        self.is_train = is_train

    def __call__(self, image):
        """Transform a single PIL image.

        Args:
            image: PIL Image

        Returns:
            dict: Processor outputs with squeezed batch dimension containing:
                - pixel_values: Image tensor
                - pixel_attention_mask: Attention mask (for NaFlex models)
                - spatial_shapes: Spatial dimensions (for NaFlex models)
        """
        # Process single image
        result = self._processor(images=image, return_tensors='pt')
        # Squeeze batch dimension from all tensors
        return {k: v.squeeze(0) for k, v in result.items()}
