# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deferred spatial-transform image wrapper.

``DeferImage`` wraps a torch tensor and accumulates spatial transforms
(translate, scale, rotate, flip, arbitrary 3x3 STM) without materialising
intermediate results.  The transforms are fused into a single 3x3
homogeneous matrix and applied in one shot via ``spatial_transform_fn``
when ``__call__`` is invoked.

The C++ CUDA ``spatial_transform`` kernel is replaced here with a pure-
PyTorch fallback (``generate_homography_grid`` + ``F.grid_sample``).
The fallback is numerically equivalent for bilinear interpolation with
``pad_color=0`` and ``align_corners=False``.

NOTE: Many of the operations in this module may *seem* like they are the
inverse of what is being requested, such as translations moving in the
opposite direction, or scales being reciprocals.  This is because the
spatial-transform operates in *inverse-warp* mode: instead of pushing
source pixels forward, it maps each destination pixel back to a source
coordinate.  A useful mental model is a movable window sitting on top of
the source image — the transforms move the window, producing the inverse
visual effect on the output.
"""

from typing import Callable
import logging

import torch
from nvidia_tao_pytorch.multimodal.radio.dataloader.ops.spatial_transform import spatial_transform
from nvidia_tao_pytorch.multimodal.radio.dataloader.transforms.quad import _apply_single_stm

logger = logging.getLogger(__name__)


def _options(t: torch.Tensor):
    """Return dtype/device kwargs matching an existing tensor."""
    return dict(dtype=t.dtype, device=t.device)


class DeferImage(object):
    """Wrapper around a torch tensor that accumulates spatial transforms.

    Transforms are not applied until ``__call__`` is invoked.  This
    enables efficient batched execution on GPU.
    """

    def __init__(
        self,
        image: torch.Tensor,
        allow_eager=True,
        spatial_transform_fn: Callable = spatial_transform,
    ) -> None:
        """Initialise with a ``(C, H, W)`` or ``(B, C, H, W)`` image tensor."""
        super().__init__()
        self.image = image
        self.target_size = tuple(image.shape)
        self.stm = torch.eye(3, dtype=torch.float32)
        self.fresh = allow_eager
        self.original_shape = image.shape
        self.spatial_transform_fn = spatial_transform_fn

    @property
    def shape(self):
        """Returns the destination shape after pending transforms."""
        return self.target_size

    @property
    def dtype(self):
        """Return the dtype of the underlying image tensor."""
        return self.image.dtype

    def can_collate(self, other_defer):
        """Decide whether two ``DeferImage`` objects can be combined."""
        if not isinstance(other_defer, DeferImage):
            return False

        return (
            self.image.shape[-3:] == other_defer.image.shape[-3:] and
            self.target_size[-3:] == other_defer.target_size[-3:] and
            self.image.dtype == other_defer.image.dtype
        )

    def collate(self, other_defers, output_buffers=None):
        """Combine one or more other ``DeferImage`` objects into this one.

        If *output_buffers* is supplied, memory concatenation is performed
        inside pre-allocated storage.  The dict must have the format::

            {
                'image': (torch.Storage, [offset]),
                'stm':   (torch.Storage, [offset]),
            }
        """
        images = [self.image]
        stms = [self.stm]

        for df in other_defers:
            images.append(df.image)
            stms.append(df.stm)

        if output_buffers is None:
            self.image = torch.stack(images, dim=0)
            self.stm = torch.stack(stms, dim=0)
        else:
            for k, group in (("image", images), ("stm", stms)):
                output_buffer, output_offset = output_buffers[k]
                req_size = len(group) * group[0].nelement()
                if output_buffer.size() < output_offset[0] + req_size:
                    raise ValueError(
                        "The provided buffer for {} isn't large enough!".format(k)
                    )

                op_tensor = group[0].new()
                op_tensor.set_(output_buffer, output_offset[0], (len(group), *group[0].shape))

                torch.stack(group, dim=0, out=op_tensor)
                setattr(self, k, op_tensor)
                output_offset[0] += req_size

    def pin_memory(self):
        """Pin the memory for all owned tensors."""
        self.image = self.image.pin_memory()
        self.stm = self.stm.pin_memory()
        return self

    def cuda(self, **kwargs):
        """Move all tensors to the default CUDA device."""
        self.image = self.image.cuda(**kwargs)
        self.stm = self.stm.cuda(**kwargs)
        return self

    def __call__(self, pad_color: float = 0.0):
        """Apply all deferred operations and return the result.

        NOTE: The result is *not* cached.
        """
        if self.fresh:
            return self.image

        out_height, out_width = self.target_size[-2:]

        image = self.image
        stm = self.stm.T
        was_3d = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            stm = stm.unsqueeze(0)
            was_3d = True

        if image.dtype == torch.uint8:
            image = image.float().div_(255)

        image = image.contiguous()
        stm = stm.contiguous()

        image = self.spatial_transform_fn(
            image, stm, out_width, out_height, "bilinear", pad_color, False
        )

        if was_3d:
            image = image[0]

        return image

    def apply_stm(self, stm):
        """Apply a 3x3 homogeneous transformation matrix.

        The supplied *stm* is right-multiplied into the accumulated matrix.
        """
        self.stm = self.stm @ stm
        self.fresh = False

    def clip_translate(self, x, y, width, height):
        """Clip the image at the specified coordinates.

        Equivalent to a translation followed by a size change.
        """
        stm = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [x, y, 1],
        ], dtype=torch.float32)

        self.apply_stm(stm.T)
        self.target_size = self.target_size[:-2] + (height, width)

    def scale(self, scale_vector, retain_dims=False):
        """Scale the image by a 2D vector ``[x_scale, y_scale]``.

        If *retain_dims* is ``True``, the output size is left unchanged.
        """
        stm = torch.tensor([
            [1.0 / scale_vector[0], 0, 0],
            [0, 1.0 / scale_vector[1], 0],
            [0, 0, 1],
        ], dtype=torch.float32)

        self.apply_stm(stm)

        if not retain_dims:
            target_height = self.target_size[-2] * scale_vector[-1].item()
            target_width = self.target_size[-1] * scale_vector[-2].item()
            self.target_size = self.target_size[:-2] + (target_height, target_width)

    def translate(self, trans_vector):
        """Translate the image by ``[x_translation, y_translation]``."""
        stm = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [-trans_vector[0], -trans_vector[1], 1],
        ], dtype=torch.float32)

        self.apply_stm(stm.T)

    def flip(self, x):
        """Flip the image horizontally about the specified *x*-axis."""
        self.apply_stm(torch.tensor([
            [   -1, 0, 0],  # noqa
            [    0, 1, 0],  # noqa
            [2 * x, 0, 1]
        ], dtype=torch.float32).T)

    def rotate(self, rot_mat):
        """Rotate the image by a 2x2 rotation matrix.

        The output size is expanded to fully contain the rotated content.
        """
        stm = torch.eye(3, **_options(rot_mat))
        stm[:2, :2].copy_(rot_mat)

        self.apply_stm(stm.T)

        h2, w2 = self.target_size[-2] / 2, self.target_size[-1] / 2
        bds = torch.tensor([
            [-w2, -h2], [w2, -h2], [w2, h2], [-w2, h2]
        ], dtype=torch.float32)

        bds = _apply_single_stm(bds, stm)

        min_bds = bds.amin(dim=0)
        max_bds = bds.amax(dim=0)

        new_width = (max_bds[0] - min_bds[0]).item()
        new_height = (max_bds[1] - min_bds[1]).item()

        self.target_size = self.target_size[:-2] + (new_height, new_width)

    def get_bds(self):
        """Return the current bounds in source-image coordinates."""
        bds = torch.tensor([
            [0, 0],
            [self.original_shape[-1], 0],
            [self.original_shape[-1], self.original_shape[-2]],
            [0, self.original_shape[-2]]
        ], dtype=torch.float32)

        bds = _apply_single_stm(bds, torch.inverse(self.stm).T)

        return bds


def get_cuda_images(defer_image_groups, augmentation=None, additional=[]):
    """Convert image groups (on CPU) to a contiguous GPU batch.

    Operations performed:
        1. Move all image groups to GPU
        2. Apply pending spatial operations
        3. Concatenate all groups into a single batch on GPU
        4. Convert to float32 (uint8 is scaled by 1/255)
    """
    images = []
    for defer in defer_image_groups:
        defer.cuda(non_blocking=True)
        images.append(defer())

    cuda_images = torch.cat(images, dim=0)

    if cuda_images.dtype == torch.float16:
        cuda_images = cuda_images.to(torch.float32)
    elif cuda_images.dtype == torch.uint8:
        cuda_images = cuda_images.float().div_(255.0)

    invalid_color = torch.tensor(
        [0.485, 0.456, 0.406], **_options(cuda_images)
    ).reshape(1, -1, 1, 1).expand_as(cuda_images)
    invalid_mask = torch.any(cuda_images < 0, dim=1, keepdim=True).expand_as(cuda_images)

    if augmentation is not None:
        cuda_images: torch.Tensor = augmentation(cuda_images, *additional)

    cuda_images = torch.where(invalid_mask, invalid_color, cuda_images)

    return cuda_images
