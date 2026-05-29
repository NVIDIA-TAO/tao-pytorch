# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FoundationStereo Module"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo import utils
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.iterative_refinement import \
    BasicSelectiveMultiUpdateBlock
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.extractor import (
    ContextNetwork, Feature, FeatureAtt)
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.geometry import \
    CombinedGeoEncodingVolume
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.submodule import (
    HourGlass, SpatialAttentionExtractor, ChannelAttentionEnhancement)
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.convolution_helper import (
    Conv, Conv2xDownScale, ResnetBasicBlock3D)


AUTOCAST = torch.amp.autocast


class FoundationStereo(nn.Module):
    """Foundation Stereo model for disparity estimation."""

    def __init__(self, args, export=False):
        """Initializes the FoundationStereo model.

        Args:
            args: Configuration arguments.
        """
        super().__init__()
        self.args = args
        self.export = export

        context_dims = args.hidden_dims
        self.cv_group = 8
        volume_dim = 28

        self.cnet = ContextNetwork(self.args, output_dim=[args.hidden_dims, context_dims],
                                   downsample=args.n_downsample)

        self.update_block = BasicSelectiveMultiUpdateBlock(
            self.args, self.args.hidden_dims[0], volume_dim=volume_dim)
        self.sam = SpatialAttentionExtractor()
        self.cam = ChannelAttentionEnhancement(self.args.hidden_dims[0])

        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, kernel_size=3, padding=3 // 2)
            for i in range(self.args.n_gru_layers)
        ])

        self.feature = Feature(args, export=self.export)

        self.proj_cmb = nn.Conv2d(self.feature.d_out[0], 12, kernel_size=1, padding=0)

        self.stem_2 = nn.Sequential(Conv(3, 32, conv_type='conv2d',
                                         norm_type='instance2d',
                                         relu=True,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1),
                                    nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                                    nn.InstanceNorm2d(32),
                                    nn.ReLU()
                                    )
        self.stem_4 = nn.Sequential(Conv(32, 48, conv_type='conv2d',
                                         norm_type='instance2d',
                                         relu=True, kernel_size=3, stride=2, padding=1),
                                    nn.Conv2d(48, 48, 3, 1, 1, bias=False),
                                    nn.InstanceNorm2d(48),
                                    nn.ReLU()
                                    )
        self.spx_2_gru = Conv2xDownScale(32, 32, norm_type=None, conv_type='deconv2d', relu=True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1))

        self.corr_stem = nn.Sequential(nn.Conv3d(32, volume_dim, kernel_size=1),
                                       Conv(volume_dim, volume_dim,
                                       norm_type='instance3d',
                                       conv_type='conv3d',
                                       relu=True,
                                       kernel_size=3,
                                       padding=1),
                                       ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
                                       ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
                                       )
        self.corr_feature_att = FeatureAtt(volume_dim, self.feature.d_out[0])
        self.cost_agg = HourGlass(cfg=self.args,
                                  in_channels=volume_dim,
                                  feat_dims=self.feature.d_out
                                  )
        self.classifier = nn.Sequential(Conv(volume_dim, volume_dim // 2,
                                             norm_type='instance3d',
                                             conv_type='conv3d',
                                             relu=True,
                                             kernel_size=3,
                                             padding=1),
                                        ResnetBasicBlock3D(volume_dim // 2, volume_dim // 2, kernel_size=3, stride=1, padding=1),
                                        nn.Conv3d(volume_dim // 2, 1, kernel_size=7, padding=3),
                                        )
        r = self.args.corr_radius
        dx = torch.linspace(-r, r, 2 * r + 1, requires_grad=False).reshape(1, 1, 2 * r + 1, 1)
        self.dx = dx

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        """
        Upsamples the disparity map using learned spatial propagation and feature fusion.

        This method takes a low-resolution disparity map and uses a GRU-based network to
        generate a high-resolution version. It leverages features from a mask and a
        stem to guide the upsampling process, ensuring that the upsampled disparity
        is spatially consistent and accurate. Mixed precision is used for
        performance optimization.

        Args:
            disp (torch.Tensor): A 1/4 resolution disparity map. Its shape is typically
                                (batch_size, 1, H/4, W/4), where H and W are the
                                height and width of the original image.
            mask_feat_4 (torch.Tensor): Features from a mask at 1/4 resolution. These
                                        features likely encode information about
                                        discontinuity boundaries and object details.
                                        The shape is (batch_size, C1, H/4, W/4).
            stem_2x (torch.Tensor): Features from a stem (an early layer of the network)
                                    at 1/2 resolution. These provide high-level context
                                    and structural information. The shape is
                                    (batch_size, C2, H/2, W/2).

        Returns:
            torch.Tensor: The upsampled disparity map at the original input resolution,
                        with a shape of (batch_size, 1, H, W). The values are scaled
                        to be four times the original disparity values.
        """
        with AUTOCAST('cuda', enabled=self.args.mixed_precision):
            # Use a GRU to process mask and stem features at 1/2 resolution.
            # This step learns the spatial propagation cues.
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)

            # The GRU's output is fed into another GRU to predict the spatial
            # propagation weights.
            spx_pred = self.spx_gru(xspx)

            # Apply softmax to the predictions to get a probability distribution,
            # which acts as the spatial propagation weights.
            spx_pred = F.softmax(spx_pred, 1)

            # Upsample the disparity map using the predicted spatial weights.
            # The disparity is multiplied by 4 before upsampling to match the
            # output resolution scale. unsqueeze(1) is used to add a channel
            # dimension for consistency.
            up_disp = utils.context_upsample(disp * 4., spx_pred).unsqueeze(1)

        return up_disp.float()

    def forward(self, left_image, right_image, iters=22, flow_init=None, test_mode=False, low_memory=False, init_disp=None):
        """Estimates disparity between a pair of frames.

        Args:
            left_image: Left image tensor.
            right_image: Right image tensor.
            iters: Number of gru iterations to refine the disparity.
            flow_init: Initial flow (unused).
            test_mode: Whether in test mode.
            low_memory: Whether to use low memory mode.
            init_disp: Initial disparity.

        Returns:
            Initial disparity and list of predicted disparities.
        """
        batch_size = len(left_image)
        low_memory = low_memory or (self.args.get('low_memory', False))
        with AUTOCAST('cuda', enabled=self.args.mixed_precision):
            out, vit_feat = self.feature(torch.cat([left_image, right_image], dim=0))
            vit_feat = vit_feat[:batch_size]
            features_left = [tensor[:batch_size] for tensor in out]
            features_right = [tensor[batch_size:] for tensor in out]
            stem_2x = self.stem_2(left_image)
            gwc_volume = utils.build_gwc_volume(features_left[0],
                                                features_right[0],
                                                self.args.max_disparity // 4,
                                                self.cv_group)

            # Group-wise correlation volume (B, N_group, max_disparity, H, W)
            left_tmp = self.proj_cmb(features_left[0])
            right_tmp = self.proj_cmb(features_right[0])
            concat_volume = utils.build_concat_volume(left_tmp,
                                                      right_tmp,
                                                      maxdisp=self.args.max_disparity // 4
                                                      )

            del left_tmp, right_tmp
            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            del concat_volume, gwc_volume
            comb_volume = self.corr_stem(comb_volume)

            comb_volume = self.corr_feature_att(comb_volume, features_left[0])
            comb_volume = self.cost_agg(comb_volume, features_left)

            prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)
            if init_disp is None:
                init_disp = utils.disparity_regression(prob, self.args.max_disparity // 4)

            cnet_list = self.cnet(left_image, vit_feat=vit_feat, num_layers=self.args.n_gru_layers)
            cnet_list = list(cnet_list)

            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [self.cam(x) * x for x in inp_list]
            att = [self.sam(x) for x in inp_list]

        geo_fn = CombinedGeoEncodingVolume(
            features_left[0].float(), features_right[0].float(), comb_volume.float(),
            num_levels=self.args.corr_levels, dx=self.dx
        )
        b, _, h, w = features_left[0].shape
        coords = torch.arange(
            w,
            dtype=torch.float,
            device=init_disp.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp.float()

        disp_preds = []
        del comb_volume, features_left, features_right, cnet_list
        # GRUs iterations to update disparity (1/4 resolution)

        if test_mode:
            for _ in range(iters):
                disp = disp.detach()
                geo_feat = geo_fn(disp, coords, low_memory=low_memory)
                with AUTOCAST('cuda', enabled=self.args.mixed_precision):
                    net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
                disp = disp + delta_disp.float()
            disp_up = self.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float())
            return disp_up
        else:
            for _ in range(iters):
                disp = disp.detach()
                geo_feat = geo_fn(disp, coords, low_memory=low_memory)
                with AUTOCAST('cuda', enabled=self.args.mixed_precision):
                    net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
                disp = disp + delta_disp.float()
                disp_up = self.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float())
                disp_preds.append(disp_up)
            return init_disp, disp_preds
