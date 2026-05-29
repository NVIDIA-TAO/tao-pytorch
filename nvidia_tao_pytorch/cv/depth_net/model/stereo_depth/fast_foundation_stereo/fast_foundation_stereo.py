# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastFoundationStereo composes TAO foundation_stereo classes via
configurable knobs on each submodule.

Remaining ckpt-key shifts (e.g. ``cost_agg.atts.4.*`` ->
``cost_agg.atts.cost_vol_disp.*``) are handled by ckpt_utils.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo import utils
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.geometry import (
    CombinedGeoEncodingVolume,
)
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.extractor import (
    DepthAnythingFeature,
    Feature,
)
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.submodule import (
    SpatialAttentionExtractor,
    ChannelAttentionEnhancement,
    CostVolumeDisparityAttention,
    FeatureAtt,
    HourGlass,
)
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.convolution_helper import (
    Conv,
    Conv2xDownScale,
    Conv3dNormActReduced,
    ResnetBasicBlock3D,
)
from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.foundation_stereo.iterative_refinement import (
    BasicSelectiveMultiUpdateBlock,
)

from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.fast_foundation_stereo.extractor import (
    ContextNetSharedBackbone,
)


AUTOCAST = torch.amp.autocast


def _phantom_vit_feat_dim(cfg):
    """Phantom-ViT channel band that ``Feature.conv4`` allocates without a ViT branch."""
    encoder = getattr(cfg, 'encoder', None)
    if encoder not in DepthAnythingFeature.model_configs:
        raise KeyError(
            f"cfg.encoder must be one of "
            f"{sorted(DepthAnythingFeature.model_configs)}; got {encoder!r}")
    return DepthAnythingFeature.model_configs[encoder]['features'] // 2


def _list_or_none(field):
    """OmegaConf list-or-None to plain Python list-or-None."""
    if field is None:
        return None
    return list(field)


class _FFSForwardHelper(nn.Module):
    """Sequential whose ``FeatureAtt`` element receives a side feature."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, left_feat=None):
        for layer in self.layers:
            if isinstance(layer, FeatureAtt):
                x = layer(x, left_feat)
            else:
                x = layer(x)
        return x


class _FFSPostForwardHelper(nn.Module):
    """Upsample-merge-out chain. ``layers`` is a flat list with a 'sum'
    or 'concat' sentinel splitting upsample (Sequential) from out (ModuleList).
    """

    def __init__(self, layers):
        super().__init__()
        sentinel_pos = None
        for pos, item in enumerate(layers):
            if item in ('sum', 'concat'):
                sentinel_pos = pos
                self.op = item
                break
        if sentinel_pos is None:
            raise ValueError("layers must contain 'sum' or 'concat' sentinel")
        self.upsample = nn.Sequential(*layers[:sentinel_pos])
        self.out = nn.ModuleList(layers[sentinel_pos + 1:])

    def forward(self, conv_high, conv_low, left_feat=None):
        upsampled = self.upsample(conv_low)
        if self.op == 'sum':
            x = upsampled + conv_high
        else:
            x = torch.cat((upsampled, conv_high), dim=1)
        for layer in self.out:
            if isinstance(layer, FeatureAtt):
                x = layer(x, left_feat)
            else:
                x = layer(x)
        return x


class _FFSHourGlass(HourGlass):
    """FFS commercial-ckpt-specific HourGlass variant.

    Widths and kernels are hardcoded to match
    ``model_best_bp2_serialize.pth`` (distilled bp2). For other distill
    variants, introduce a new helper class or promote values to
    configurable model-config knobs.

    Distillation diff vs TAO HourGlass:
      - parent's ``conv1`` and ``conv2`` (downsample stages) are folded into
        ``feature_att_8.layers`` / ``feature_att_16.layers`` and replaced with
        ``nn.Identity()`` so they vanish from state_dict.
      - ``feature_att_8`` becomes a 3-elem ForwardHelper:
        BasicConv → Conv3dNormActReduced(dk=17) → FeatureAtt(56, 192).
      - ``feature_att_16`` becomes a 3-elem ForwardHelper (no end FA):
        BasicConv → Conv3dNormActReduced(dk=13) → Conv3dNormActReduced(dk=3).
        feat[2] gating is moved to ``post32_to_16.out[0]``.
      - new ``post32_to_16`` / ``post16_to_8`` / ``post8_to_4`` PostForward
        helpers replace the legacy agg_0/agg_1/conv_out path.
      - ``feature_att_32``, ``feature_att_up_8``, ``feature_att_up_16``
        keep the parent's single-FeatureAtt structure (mid = feat//2 already
        matches the dump: 152, 96, 160).
    """

    def __init__(self, cfg, in_channels, feat_dims):
        super().__init__(
            cfg=cfg, in_channels=in_channels, feat_dims=feat_dims,
            agg_1_conv_type='conv',
            atts_max_len_divisor=16,
            feat_att_norm_type='batch2d',
            conv_patch_padding=[0, 0, 0],
            # FFS commercial bp2 was trained with the non-standard
            # heads-as-sequence Q/K/V reshape. Propagates through
            # parent's atts['cost_vol_disp'] and the FFS-specific
            # post8_to_4.upsample[1] CostVolumeDisparityAttention below.
            qkv_layout='upstream_compat',
        )

        c = in_channels
        self.conv1 = nn.Identity()
        self.conv2 = nn.Identity()

        self.feature_att_8 = _FFSForwardHelper([
            Conv(c, c, conv_type='conv3d', norm_type='batch3d',
                 relu=True, kernel_size=3, padding=1, stride=2),
            Conv3dNormActReduced(c, c * 2, kernel_size=3, kernel_disp=17),
            FeatureAtt(c * 2, feat_dims[1], norm_type='batch2d'),
        ])

        self.feature_att_16 = _FFSForwardHelper([
            Conv(c * 2, c * 4, conv_type='conv3d', norm_type='batch3d',
                 relu=True, kernel_size=3, padding=1, stride=2),
            Conv3dNormActReduced(c * 4, c * 2, kernel_size=3, kernel_disp=13),
            Conv3dNormActReduced(c * 2, c * 4, kernel_size=3, kernel_disp=3),
        ])

        self.post32_to_16 = _FFSPostForwardHelper([
            Conv(c * 6, c * 4, conv_type='deconv3d', norm_type='batch3d',
                 relu=True, kernel_size=3, padding=1, stride=2,
                 output_padding=1),
            'sum',
            FeatureAtt(c * 4, feat_dims[2], norm_type='batch2d'),
            Conv(c * 4, c * 4, conv_type='conv3d', norm_type='batch3d',
                 relu=True, kernel_size=3, padding=1),
            Conv3dNormActReduced(c * 4, c * 4, kernel_size=3, kernel_disp=9),
            Conv(c * 4, c * 4, conv_type='conv3d', norm_type='batch3d',
                 relu=True, kernel_size=3, padding=1),
        ])

        self.post16_to_8 = _FFSPostForwardHelper([
            Conv(c * 4, c * 2, conv_type='deconv3d', norm_type='batch3d',
                 relu=True, kernel_size=4, padding=1, stride=2),
            'sum',
            Conv(c * 2, c * 2, conv_type='conv3d', norm_type='batch3d',
                 relu=True, kernel_size=3, padding=1),
            FeatureAtt(c * 2, feat_dims[1], norm_type='batch2d'),
            Conv3dNormActReduced(c * 2, c * 2, kernel_size=3, kernel_disp=9),
        ])

        self.post8_to_4 = _FFSPostForwardHelper([
            # Pickled bp2 inspection: upsample is actually a "downsample 4x →
            # disparity attention at 1/16 → trilinear-upsample 4x" chain
            # (mirrors parent's conv_patch + atts['cost_vol_disp'] +
            # F.interpolate path). Naming is upstream's; the helper is shared
            # with post32_to_16 / post16_to_8 which DO upsample.
            # upstream's BasicConv at this position uses relu=False (no
            # LeakyReLU after the BN) — pickled bp2 inspection confirms.
            Conv(c, c, conv_type='conv3d', norm_type='batch3d',
                 relu=False, kernel_size=4, padding=0, stride=4),
            CostVolumeDisparityAttention(
                d_model=c, nhead=4, dim_feedforward=c * 4,
                norm_first=False, num_transformer=1,
                # After the 4x downsample above, disparity dim = max_disparity//16,
                # matching the parent atts['cost_vol_disp']'s atts_max_len_divisor=16.
                max_len=cfg['max_disparity'] // 16,
                qkv_layout='upstream_compat'),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False),
            'sum',
            Conv(c, c, conv_type='conv3d', norm_type='batch3d',
                 relu=True, kernel_size=3, padding=1),
            ResnetBasicBlock3D(c, c, kernel_size=3, padding=1),
        ])

    def forward(self, x, features, atts_before_upsample: bool = True):
        """Run FFS-distilled hourglass cost-aggregation.

        Args:
            x (torch.Tensor): cost volume at 1/4 resolution, shape (B, C, D, H, W).
            features (list[torch.Tensor]): left features at 1/4, 1/8, 1/16, 1/32.
            atts_before_upsample (bool): kept for parity with parent HourGlass; unused.

        Returns:
            torch.Tensor: aggregated cost volume at 1/4 resolution.
        """
        conv1 = self.feature_att_8(x, features[1])
        conv2 = self.feature_att_16(conv1, features[2])
        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])
        conv2 = self.post32_to_16(conv2, conv3, features[2])
        conv1 = self.post16_to_8(conv1, conv2, features[1])
        conv = self.conv1_up(conv1)
        conv = self.post8_to_4(x, conv)
        return conv


class FastFoundationStereo(nn.Module):
    """FFS model. Composes TAO foundation_stereo classes via configurable knobs.

    Args:
        args: model_config OmegaConf node. Required: ``encoder``,
            ``hidden_dims``, ``n_gru_layers``, ``corr_radius``, ``corr_levels``,
            ``max_disparity``, ``mixed_precision``, plus the bp2-distilled
            width override fields (motion_encoder_widths, gru_*, mask_widths,
            spx_2_gru_widths, classifier_mid, cam_mid_channels, etc.).
        export: kept for API parity with regular FS.
    """

    def __init__(self, args, export=False):
        super().__init__()
        self.args = args
        self.export = export

        context_dims = args.hidden_dims
        # bp2 ckpt invariants (defaults 8 / 24 / 28 in DepthNetModelConfig).
        self.cv_group = args.cv_group
        self.concat_channel = args.concat_channel
        volume_dim = args.volume_dim
        self.volume_dim = volume_dim

        # gru_hidden controls the GRU hidden state width and the disp_head /
        # mask input dim. Falls back to hidden_dims[0] for non-distilled callers.
        gru_hidden = args.gru_hidden if args.gru_hidden is not None else args.hidden_dims[0]

        # cnet output widths default to hidden_dims[0] for both heads.
        cnet_widths = _list_or_none(args.cnet_conv04_widths)
        if cnet_widths is None:
            cnet_widths = [args.hidden_dims[0], args.hidden_dims[0]]

        # gru04 input width = motion_features (= motion_encoder_final + 1 disp)
        # cat with inp[0] (= cnet head 1). For TAO defaults this equals
        # ``hidden_dims[0]*2`` -- the legacy formula.
        me_final = (args.motion_encoder_final
                    if args.motion_encoder_final is not None
                    else args.hidden_dims[0] - 1)
        gru04_in = me_final + 1 + cnet_widths[1]

        self.update_block = BasicSelectiveMultiUpdateBlock(
            args, hidden_dim=gru_hidden, volume_dim=volume_dim,
            motion_encoder_widths=_list_or_none(args.motion_encoder_widths),
            motion_encoder_final=args.motion_encoder_final,
            gru_gating_conv_widths=_list_or_none(args.gru_gating_conv_widths),
            disp_head_intermediate=args.disp_head_intermediate,
            disp_head_pwconv1_widths=_list_or_none(args.disp_head_pwconv1_widths),
            mask_widths=_list_or_none(args.mask_widths),
            gru04_input_dim=gru04_in)

        self.sam = SpatialAttentionExtractor()
        # cam input width = cnet head 1 (inp_init). mid_channels via
        # cam_mid_channels spec field.
        self.cam = ChannelAttentionEnhancement(
            in_planes=cnet_widths[1],
            mid_channels=args.cam_mid_channels)
        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3,
                      kernel_size=3, padding=3 // 2)
            for i in range(args.n_gru_layers)
        ])

        self.feature = Feature(
            args, export=export,
            use_vit_backbone=False,
            phantom_vit_feat_dim=_phantom_vit_feat_dim(args))

        self.proj_cmb = nn.Conv2d(
            self.feature.d_out[0], self.concat_channel // 2,
            kernel_size=1, padding=0)

        self.cnet = ContextNetSharedBackbone(
            args,
            c04=self.feature.d_out[0],
            c08=self.feature.d_out[1],
            c16=self.feature.d_out[2],
            output_dim=[[cnet_widths[0]], [cnet_widths[1]]])

        # stem_2: per-yaml widths (default [32, 32] reproduces TAO).
        stem_widths = _list_or_none(args.stem_2_widths) or [32, 32]
        self.stem_2 = nn.Sequential(
            Conv(3, stem_widths[0],
                 relu=True, norm_type='instance2d', conv_type='conv2d',
                 kernel_size=3, stride=2, padding=1),
            nn.Conv2d(stem_widths[0], stem_widths[1], 3, 1, 1, bias=False),
            nn.InstanceNorm2d(stem_widths[1]),
            nn.ReLU(),
        )

        # spx_2_gru: TAO Conv2xDownScale with explicit per-conv widths via
        # the rem_channels and conv2_out_channels knobs.
        # spx_2_gru_widths is a 4-elem [in, mid, rem, out] list.
        spx_widths = _list_or_none(args.spx_2_gru_widths) or [32, 32, 32, 32]
        self.spx_2_gru = Conv2xDownScale(
            in_channels=spx_widths[0],
            out_channels=spx_widths[1],
            rem_channels=spx_widths[2],
            conv2_out_channels=spx_widths[3],
            norm_type=None, conv_type='deconv2d',
            concat=True, conv2_type='conv',
            conv2_relu=True)  # upstream Conv2x default behaviour for FFS bp2
        # spx_gru ConvTranspose; input = spx_2_gru output (= spx_widths[3]),
        # output = spx_gru_out yaml field.
        self.spx_gru = nn.Sequential(
            nn.ConvTranspose2d(spx_widths[3], args.spx_gru_out,
                               kernel_size=4, stride=2, padding=1),
        )

        # corr_feature_att: inline Sequential matching the FFS commercial ckpt
        # ``corr_feature_att.layers.{0,1,2}`` slot — two BN3d-Conv3d layers
        # then a 2D FeatureAtt. forward unrolls the Sequential because layer
        # 2 (FeatureAtt) takes (cv, feat) while 0/1 take (cv).
        self.corr_feature_att = nn.Sequential(
            Conv(self.proj_cmb.out_channels * 2 + self.cv_group, volume_dim,
                 relu=True, norm_type='batch3d', conv_type='conv3d',
                 kernel_size=3, padding=1),
            Conv(volume_dim, volume_dim,
                 relu=True, norm_type='batch3d', conv_type='conv3d',
                 kernel_size=3, padding=1),
            FeatureAtt(volume_dim, self.feature.d_out[0],
                       norm_type='batch2d'),
        )

        self.cost_agg = _FFSHourGlass(
            cfg=args, in_channels=volume_dim, feat_dims=self.feature.d_out)

        # classifier: inline Sequential of one Conv3d (k=3) — matches ckpt
        # ``classifier.layers.0`` slot. Single-element Sequential preserves
        # the ckpt key-prefix shape (``classifier.0.*``). bias=True because
        # ckpt has ``classifier.layers.0.bias``.
        self.classifier = nn.Sequential(
            nn.Conv3d(volume_dim, 1, kernel_size=3, padding=1),
        )

        # dx int8 buffer; geometry.py:126 moves it to disp.device, then PyTorch
        # type promotion (int8 + float -> float) handles dtype at sample time.
        r = args.corr_radius
        dx = torch.arange(-r, r + 1, requires_grad=False, dtype=torch.int8).reshape(
            1, 1, 2 * r + 1, 1)
        self.register_buffer("dx", dx)

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        """Learned 4x upsample of the 1/4-resolution disparity.

        Args:
            disp (torch.Tensor): disparity at 1/4 resolution, shape (B, 1, H, W).
            mask_feat_4 (torch.Tensor): mask feature at 1/4 resolution.
            stem_2x (torch.Tensor): stem feature at 1/2 resolution.

        Returns:
            torch.Tensor: full-resolution disparity, shape (B, 1, H*4, W*4), fp32.
        """
        with AUTOCAST('cuda', enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = utils.context_upsample(disp * 4., spx_pred).unsqueeze(1)
        return up_disp.float()

    def _run_corr_feature_att(self, cv, feat):
        """Unroll corr_feature_att Sequential (layer 2 needs (cv, feat))."""
        for layer in self.corr_feature_att:
            if isinstance(layer, FeatureAtt):
                cv = layer(cv, feat)
            else:
                cv = layer(cv)
        return cv

    def forward(self, left_image, right_image, iters=12, flow_init=None,
                test_mode=False, low_memory=False, init_disp=None):
        """Estimate disparity for a stereo pair.

        Args:
            left_image (torch.Tensor): left input, shape (B, 3, H, W).
            right_image (torch.Tensor): right input, shape (B, 3, H, W).
            iters (int): number of GRU refinement iterations.
            flow_init: unused, kept for FoundationStereo signature parity.
            test_mode (bool): True returns final upsampled disparity only.
            low_memory (bool): True streams the geometry encoding in chunks.
            init_disp (torch.Tensor or None): initial disparity at 1/4
                resolution; computed from softmax-regression when None.

        Returns:
            test_mode=True: torch.Tensor disparity at full resolution
            (B, 1, H, W). test_mode=False: tuple (init_disp, disp_preds)
            where disp_preds is a list of per-iter full-resolution disparities.
        """
        batch_size = len(left_image)
        low_memory = low_memory or (self.args.get('low_memory', False))
        with AUTOCAST('cuda', enabled=self.args.mixed_precision):
            out, _ = self.feature(torch.cat([left_image, right_image], dim=0))
            features_left = [tensor[:batch_size] for tensor in out]
            features_right = [tensor[batch_size:] for tensor in out]
            stem_2x = self.stem_2(left_image)

            gwc_volume = utils.build_gwc_volume(
                features_left[0], features_right[0],
                self.args.max_disparity // 4, self.cv_group,
                normalize=self.args.gwc_feature_normalize)

            left_tmp = self.proj_cmb(features_left[0])
            right_tmp = self.proj_cmb(features_right[0])
            concat_volume = utils.build_concat_volume(
                left_tmp, right_tmp,
                maxdisp=self.args.max_disparity // 4)
            del left_tmp, right_tmp
            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            del concat_volume, gwc_volume

            comb_volume = self._run_corr_feature_att(comb_volume, features_left[0])
            comb_volume = self.cost_agg(
                comb_volume, features_left, atts_before_upsample=True)

            prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)
            if init_disp is None:
                init_disp = utils.disparity_regression(
                    prob, self.args.max_disparity // 4)

            cnet_list = self.cnet(features_left[0],
                                  features_left[1],
                                  features_left[2])
            cnet_list = list(cnet_list)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [self.cam(x) * x for x in inp_list]
            att = [self.sam(x) for x in inp_list]

        geo_fn = CombinedGeoEncodingVolume(
            features_left[0].float(), features_right[0].float(),
            comb_volume.float(),
            num_levels=self.args.corr_levels, dx=self.dx)
        b, _, h, w = features_left[0].shape
        coords = torch.arange(
            w, dtype=torch.float, device=init_disp.device
        ).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp.float()

        disp_preds = []
        del comb_volume, features_left, features_right, cnet_list

        if test_mode:
            for _ in range(iters):
                disp = disp.detach()
                geo_feat = geo_fn(disp, coords, low_memory=low_memory)
                with AUTOCAST('cuda', enabled=self.args.mixed_precision):
                    net_list, mask_feat_4, delta_disp = self.update_block(
                        net_list, inp_list, geo_feat, disp, att)
                disp = disp + delta_disp.float()
            disp_up = self.upsample_disp(
                disp.float(), mask_feat_4.float(), stem_2x.float())
            return disp_up

        for _ in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords, low_memory=low_memory)
            with AUTOCAST('cuda', enabled=self.args.mixed_precision):
                net_list, mask_feat_4, delta_disp = self.update_block(
                    net_list, inp_list, geo_feat, disp, att)
            disp = disp + delta_disp.float()
            disp_up = self.upsample_disp(
                disp.float(), mask_feat_4.float(), stem_2x.float())
            disp_preds.append(disp_up)
        return init_disp, disp_preds
