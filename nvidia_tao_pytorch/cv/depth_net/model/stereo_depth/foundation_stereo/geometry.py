# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""FoundationStereo Geometry Module."""

import torch
import torch.nn.functional as F
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.utils import bilinear_sampler


def _halve_last_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Averages adjacent pairs along the last dimension, halving its size.

    Equivalent to ``F.avg_pool2d(x, [1, 2], stride=[1, 2])`` on a 4-D tensor, including
    the floor that drops a trailing odd element: a last dim of size D becomes D // 2.
    fp32 results are bit-identical: both round the pair sum once and then halve it.

    Why not avg_pool2d: the pyramids built here fold every pixel into the batch axis,
    so the pool sees a batch of N = B * H/4 * W/4. The AveragePool kernels used by
    TensorRT and by ONNX Runtime's CUDA EP (cuDNN) fail once N exceeds a ceiling that
    depends on the pooled width D. It was measured at D = 48, 80 and 184 and is
    consistent with 65535 * floor(128 / (D / 2)) there; that is a fit to three widths,
    not a documented limit. At 480x736 input, N = 22,080 * B and the initial correlation
    volume has D = 184, where the measured ceiling is exactly 65,535, so an exported
    engine fails from batch 3 up (the geometry volume, D = 48, would fail from batch 15).
    Strided slices, an add and a multiply have no such ceiling.

    The slice bounds are relative (``:-1`` and open-ended), not computed from the
    trace-time width, so an ONNX graph exported with dynamic H/W stays correct for any
    width, even or odd.

    Args:
        x (torch.Tensor): Input tensor whose last dimension is pooled. Shape `(..., D)`.

    Returns:
        torch.Tensor: The pooled tensor. Shape `(..., D // 2)`.

    Raises:
        RuntimeError: If D < 2, which avg_pool2d also rejects. Checked in eager mode
            only, so tracing records no data-dependent branch.
    """
    if not torch.jit.is_tracing() and x.shape[-1] < 2:
        raise RuntimeError(
            f"Cannot halve a last dimension of size {x.shape[-1]}: need at least 2 elements.")
    # x[..., 0:-1:2] takes the even indices but stops before the last element, so for odd
    # D the unpaired trailing element is dropped and both slices have D // 2 elements.
    return 0.5 * (x[..., 0:-1:2] + x[..., 1::2])


class CombinedGeoEncodingVolume:
    """
    Samples features from multiple volumes to create a multi-level encoding pyramid.

    The class initializes a pyramid of both geometric and initial correlation volumes.
    In the forward pass (`__call__`), it samples from these pyramids using a given
    disparity map and coordinates, effectively creating a feature vector at each
    pixel that encodes both geometric and local correlation information across scales.

    Attributes:
        num_levels (int): The number of pyramid levels to build and sample from.
        geo_volume_pyramid (list): A list of tensors representing the downsampled
                                   geometric encoding volumes.
        init_corr_pyramid (list): A list of tensors representing the downsampled
                                  initial correlation volumes.
        dx (torch.Tensor): A tensor representing a disparity offset.
    """

    def __init__(self, init_fmap1: torch.Tensor, init_fmap2: torch.Tensor,
                 geo_volume: torch.Tensor, num_levels: int = 2, dx: torch.Tensor = None):
        """
        Initializes the CombinedGeoEncodingVolume by building the feature pyramids.

        Args:
            init_fmap1 (torch.Tensor): The initial feature map from the left image.
                                       Shape `(B, C(128), H/4, W/4)`.
            init_fmap2 (torch.Tensor): The initial feature map from the right image.
                                       Shape `(B, C(128), H/4, W/4)`.
            geo_volume (torch.Tensor): A geometry encoding volume. This volume
                                       typically contains information about the scene's
                                       3D structure or geometric priors. Shape `(B, C, D, H, W)`.
            num_levels (int, optional): The number of pyramid levels to create. Defaults to 2.
            dx (torch.Tensor, optional): A disparity offset tensor used for coordinate
                                         transformations. Defaults to None.
        """
        super().__init__()
        self.num_levels = num_levels
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []
        self.dx = dx

        # Compute the initial correlation volume from the two feature maps.
        init_corr = CombinedGeoEncodingVolume.corr(init_fmap1, init_fmap2)

        batch, height, width, _, w2 = init_corr.shape
        _, channel, depth, _, _ = geo_volume.shape

        # Reshape the geometric volume for pyramid creation.
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, channel, 1, depth).contiguous()

        # Reshape the initial correlation volume.
        init_corr = init_corr.reshape(batch * height * width, 1, 1, w2)

        # Add the initial volumes to their respective pyramids.
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)

        # Build the pyramids by halving the last dim (a [1, 2] average pool). Not
        # F.avg_pool2d: see _halve_last_dim for the batch ceiling it hits in TensorRT and
        # in cuDNN (ONNX Runtime CUDA EP).
        for _ in range(self.num_levels - 1):
            geo_volume = _halve_last_dim(geo_volume)
            self.geo_volume_pyramid.append(geo_volume)

        for _ in range(self.num_levels - 1):
            init_corr = _halve_last_dim(init_corr)
            self.init_corr_pyramid.append(init_corr)

    def make_ignore_mask(self, width: int):
        """
        Creates an ignore mask for self-attention.

        This method generates a lower triangular matrix where the lower triangle
        is set to 0 and the upper triangle is 1. This can be used to mask out
        future disparities in a disparity-attention mechanism, enforcing a causal
        relationship.

        Args:
            width (int): The width of the square mask.
        """
        # Create an identity matrix and set the lower triangle to zero.
        self.ignore_mask = torch.ones((width, width), dtype=torch.bool,
                                      device='cuda', requires_grad=False)
        for w in range(width):
            self.ignore_mask[w][:w + 1] = False

    def __call__(self, disp: torch.Tensor, coords: torch.Tensor, low_memory: bool = False) -> torch.Tensor:
        """
        Performs the forward pass of the volume sampling.

        This method samples from the pre-built geometric and correlation pyramids
        at the locations specified by the input disparity and coordinates. It
        then concatenates the sampled features to form the final combined encoding.

        Args:
            disp (torch.Tensor): A disparity tensor. Shape `(B, 1, H/4, W/4)`.
            coords (torch.Tensor): A coordinate tensor. Shape `(B, H/4, W/4, 2)`.
            low_memory (bool, optional): If True, uses a low-memory sampling mode.
                                         Defaults to False.

        Returns:
            torch.Tensor: The combined feature encoding volume. Shape `(B, C, H/4, W/4)`.
        """
        batch, _, height, width = disp.shape
        self.dx = self.dx.to(disp.device)
        out_pyramid = []

        for i in range(self.num_levels):
            # Sample from the geometric volume pyramid.
            geo_volume = self.geo_volume_pyramid[i]

            # Calculate coordinates for sampling based on disparity and pyramid level.
            x0 = self.dx + disp.reshape(batch * height * width, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)
            disp_lvl = torch.cat([x0, y0], dim=-1)

            # Use bilinear sampling to get features from the geo volume.
            geo_volume = bilinear_sampler(geo_volume, disp_lvl, low_memory=low_memory)
            geo_volume = geo_volume.reshape(batch, height, width, -1)

            # Sample from the initial correlation volume pyramid.
            init_corr = self.init_corr_pyramid[i]

            # Calculate coordinates for sampling based on coords, disp, and pyramid level.
            init_x0 = coords.reshape(batch * height * width, 1, 1, 1) / 2**i - \
                disp.reshape(batch * height * width, 1, 1, 1) / 2**i + self.dx
            init_coords_lvl = torch.cat([init_x0, y0], dim=-1)

            # Use bilinear sampling to get features from the correlation volume.
            init_corr = bilinear_sampler(init_corr, init_coords_lvl, low_memory=low_memory)
            init_corr = init_corr.reshape(batch, height, width, -1)

            # Append the sampled features to the output list.
            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)

        # Concatenate all sampled features and permute to the desired output shape.
        out_pyramid = torch.cat(out_pyramid, dim=-1)
        return out_pyramid.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        """
        Computes the correlation volume between two feature maps.

        This static method calculates the per-pixel similarity between two feature
        maps. The similarity is computed as the dot product of normalized feature
        vectors, which is equivalent to the cosine similarity. The output is a
        4D tensor representing the correlation at each spatial location.

        Args:
            fmap1 (torch.Tensor): The first feature map. Shape `(B, C, H, W1)`.
            fmap2 (torch.Tensor): The second feature map. Shape `(B, C, H, W2)`.

        Returns:
            torch.Tensor: The correlation volume. Shape `(B, H, W1, 1, W2)`.
        """
        batch, _, height, width1 = fmap1.shape
        _, _, _, width2 = fmap2.shape

        # Ensure the calculation is performed in float32 for numerical stability.
        with torch.amp.autocast('cuda', enabled=False):
            # Normalize the feature maps along the channel dimension.
            norm_fmap1 = F.normalize(fmap1.float(), dim=1)
            norm_fmap2 = F.normalize(fmap2.float(), dim=1)

            # Compute the dot product using `torch.einsum`. This is a highly
            # efficient way to compute the correlation volume.
            # 'aijk' -> B x C x H x W1
            # 'aijh' -> B x C x H x W2
            # 'ajkh' -> B x H x W1 x W2
            corr = torch.einsum('aijk,aijh->ajkh', norm_fmap1, norm_fmap2)

            # Reshape the output for the desired volume format.
            corr = corr.reshape(batch, height, width1, 1, width2)

        return corr
