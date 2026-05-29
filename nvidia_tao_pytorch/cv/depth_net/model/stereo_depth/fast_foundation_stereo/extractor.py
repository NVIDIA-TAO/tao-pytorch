# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FFS-only context-net backbone."""

from torch import nn

from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.convolution_helper import Conv


class ContextNetSharedBackbone(nn.Module):
    """Single 1/4-resolution path with two parallel BN-wrapped Conv2d heads.

    The two ``conv04`` heads carry different output widths (passed via
    ``output_dim`` as a list of two single-element lists). Outputs feed
    GRU's net_init (head 0 -> tanh) and inp_init (head 1 -> relu).
    """

    def __init__(self, cfg, c04, c08, c16, output_dim):
        super().__init__()
        self.cfg = cfg
        self.conv04 = nn.ModuleList([
            Conv(c04, output_dim[0][0],
                 relu=True, norm_type='batch2d', conv_type='conv2d',
                 kernel_size=3, padding=1),
            Conv(c04, output_dim[1][0],
                 relu=True, norm_type='batch2d', conv_type='conv2d',
                 kernel_size=3, padding=1),
        ])

    def forward(self, x4, x8=None, x16=None):
        """Apply the two ``conv04`` heads to the 1/4-resolution feature tensor.

        ``x8`` and ``x16`` are accepted for signature parity with the upstream
        ContextNetwork but are unused by this single-resolution variant.
        """
        return ([conv(x4) for conv in self.conv04],)
