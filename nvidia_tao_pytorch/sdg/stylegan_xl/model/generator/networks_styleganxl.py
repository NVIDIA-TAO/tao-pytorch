# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generator architecture from the paper
"StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets".
"""

import re
import numpy as np
import scipy.signal
import scipy.optimize
import torch

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.sdg.stylegan_xl.utils import misc
from nvidia_tao_pytorch.sdg.stylegan_xl.utils.ops import conv2d_gradfix
from nvidia_tao_pytorch.sdg.stylegan_xl.utils.ops import filtered_lrelu
from nvidia_tao_pytorch.sdg.stylegan_xl.utils.ops import bias_act
from nvidia_tao_pytorch.sdg.stylegan_xl.utils import dnnlib


def modulated_conv2d(
    x,
    w,
    s,
    demodulate=True,
    padding=0,
    input_gain=None,
):
    """Performs a modulated convolution operation.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, in_channels, in_height, in_width].
        w (torch.Tensor): Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
        s (torch.Tensor): Style tensor of shape [batch_size, in_channels].
        demodulate (bool, optional): Apply weight demodulation. Defaults to True.
        padding (int or list, optional): Padding for the convolution. Defaults to 0.
        input_gain (Tensor, optional): Scale factors for the input channels. Defaults to None.

    Returns:
        torch.Tensor: Output tensor after modulated convolution.
    """
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(s, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1, 2, 3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0)  # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels)  # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)

    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


def modulated_conv2d_export(
    x,
    w,
    s,
    demodulate=True,
    padding=0,
    input_gain=None,
):
    """Performs a modulated convolution operation.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, in_channels, in_height, in_width].
        w (torch.Tensor): Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
        s (torch.Tensor): Style tensor of shape [batch_size, in_channels].
        demodulate (bool, optional): Apply weight demodulation. Defaults to True.
        padding (int or list, optional): Padding for the convolution. Defaults to 0.
        input_gain (Tensor, optional): Scale factors for the input channels. Defaults to None.

    Returns:
        torch.Tensor: Output tensor after modulated convolution.
    """
    dtype = x.dtype
    x = x.to(w.dtype)

    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(s, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1, 2, 3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    x = x * s.unsqueeze(2).unsqueeze(3)  # [NIHW]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels)  # [NI]
        x = x * input_gain.unsqueeze(2).unsqueeze(3)  # [NIHW]

    # Demodulate weights.
    if demodulate:
        x = conv2d_gradfix.conv2d(input=x.to(dtype), weight=w.to(dtype), padding=padding)

        w = w.unsqueeze(0)  # [NOIkk]
        w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
        x = x * dcoefs.view(batch_size, out_channels, 1, 1).to(dtype)
    else:
        x = conv2d_gradfix.conv2d(input=x.to(dtype), weight=w.to(dtype), padding=padding)

    x = x.reshape(batch_size, -1, *x.shape[2:]).to(dtype)
    return x


def find_all_embed_layers(model, target_name=".embed"):
    """Function to find all layers that contain ".embed" but NOT "embed_proj"""
    embed_layers = {}
    for name, module in model.named_modules():
        if re.search(r"\.embed$", name):  # Match only names ending with ".embed"
            embed_layers[name] = module
    return embed_layers


def load_pretrained_embedding_for_embed_layers(model, embedding_checkpoint):
    """Function to load pretrained embed for all layers of discriminator/generator that contain ".embed" but NOT "embed_proj"""
    embed_layers = find_all_embed_layers(model)
    if embed_layers:
        pretrained_embed = torch.load(embedding_checkpoint, map_location=torch.device('cpu'))
        logging.info(f'Found and load the pretrained embedding for following {len(embed_layers)} embed layers:')
        for name in embed_layers:
            embed_layers[name].load_state_dict(pretrained_embed)
            logging.info(f" - {name}")
    else:
        raise AssertionError("No '.embed' layers found.")


class FullyConnectedLayer(torch.nn.Module):
    """Fully Connected Layer."""

    def __init__(
        self,
        in_features,
        out_features,
        activation='linear',
        bias=True,
        lr_multiplier=1,
        weight_init=1,
        bias_init=0,
    ):
        """Initializes the FullyConnectedLayer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            activation (str, optional): Activation function. Defaults to 'linear'.
            bias (bool, optional): Apply additive bias. Defaults to True.
            lr_multiplier (float, optional): Learning rate multiplier. Defaults to 1.
            weight_init (float, optional): Initial standard deviation of the weight tensor. Defaults to 1.
            bias_init (float, optional): Initial value of the additive bias. Defaults to 0.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the fully connected layer.
        """
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            if torch.onnx.is_in_onnx_export():
                x = bias_act.bias_act(x, b, act=self.activation,
                                      export=True)
            else:
                x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        """Extra representation for printing."""
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class MappingNetwork(torch.nn.Module):
    """Mapping Network."""

    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        num_ws,
        num_layers=2,
        lr_multiplier=0.01,
        w_avg_beta=0.998,
        embed_path=None,
        rand_embedding=False,
    ):
        """Initializes the MappingNetwork.

        Args:
            z_dim (int): Input latent (Z) dimensionality.
            c_dim (int): Conditioning label (C) dimensionality, 0 = no labels.
            w_dim (int): Intermediate latent (W) dimensionality.
            num_ws (int): Number of intermediate latents to output.
            num_layers (int, optional): Number of mapping layers. Defaults to 2.
            lr_multiplier (float, optional): Learning rate multiplier. Defaults to 0.01.
            w_avg_beta (float, optional): Decay for tracking the moving average of W. Defaults to 0.998.
            rand_embedding (bool, optional): Use random weights for class embedding. Defaults to False.
        """
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        # Additions
        self.embed = torch.nn.Embedding(num_embeddings=1000, embedding_dim=320)  # This embed layer will be loaded with a pretrained embed if needed. Find "load_pretrained_embedding".

        # Construct layers.
        self.embed_proj = FullyConnectedLayer(self.embed.embedding_dim, self.z_dim, activation='lrelu') if self.c_dim > 0 else None

        features = [self.z_dim + (self.z_dim if self.c_dim > 0 else 0)] + [self.w_dim] * self.num_layers

        for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
            layer = FullyConnectedLayer(in_features, out_features, activation='lrelu', lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if self.c_dim > 0:
            self.register_buffer('w_avg', torch.zeros([self.c_dim, w_dim]))
        else:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1.0, truncation_cutoff=None, update_emas=False):
        """Forward pass through the MappingNetwork.

        Args:
            z (torch.Tensor): Input latent tensor of shape [batch_size, z_dim].
            c (torch.Tensor): Conditioning label tensor of shape [batch_size, c_dim].
            truncation_psi (float, optional): Truncation psi value for controlling the truncation trick. Defaults to 1.0.
            truncation_cutoff (int, optional): Number of layers to apply truncation. Defaults to num_ws.
            update_emas (bool, optional): Whether to update the moving averages of W. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_ws, w_dim].
        """
        misc.assert_shape(z, [None, self.z_dim])
        if truncation_cutoff is None:
            truncation_cutoff = self.num_ws

        # Embed, normalize, and concatenate inputs.
        x = z.to(torch.float32)
        x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        if self.c_dim > 0:
            misc.assert_shape(c, [None, self.c_dim])
            y = self.embed_proj(self.embed(c.argmax(1)))
            y = y * (y.square().mean(1, keepdim=True) + 1e-8).rsqrt()
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Execute layers.
        for idx in range(self.num_layers):
            x = getattr(self, f'fc{idx}')(x)

        # Update moving average of W.
        if update_emas:
            # Track class-wise center
            if self.c_dim > 0:
                for i, label in enumerate(c.argmax(1)):
                    self.w_avg[label].copy_(x[i].detach().lerp(self.w_avg[label], self.w_avg_beta))
            else:
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast and apply truncation.
        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        if truncation_psi != 1:
            if self.c_dim > 0:
                for i, label in enumerate(c.argmax(1)):
                    x[i, :truncation_cutoff] = self.w_avg[label].lerp(x[i, :truncation_cutoff], truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)

        return x

    def extra_repr(self):
        """Provides a string representation of the MappingNetwork.

        Returns:
            str: A string representation of the MappingNetwork instance, including
                the dimensions of z, c, w, and the number of intermediate latents.
        """
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'


class SynthesisInput(torch.nn.Module):
    """Synthesis Input Network."""

    def __init__(
        self,
        w_dim,          # Intermediate latent (W) dimensionality.
        channels,       # Number of output channels.
        size,           # Output spatial size: int or [width, height].
        sampling_rate,  # Output sampling rate.
        bandwidth,      # Output bandwidth.
    ):
        """Initializes the SynthesisInput.

        Args:
            w_dim (int): Intermediate latent (W) dimensionality.
            channels (int): Number of output channels.
            size (int or list): Output spatial size, either an integer or a list of [width, height].
            sampling_rate (float): Output sampling rate.
            bandwidth (float): Output bandwidth.
        """
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1, 0, 0, 0])
        self.register_buffer('transform', torch.eye(3, 3))  # User-specified inverse transform wrt. resulting image.
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w):
        """Forward pass through the SynthesisInput.

        Args:
            w (torch.Tensor): Intermediate latent tensor of shape [batch_size, w_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, channels, height, width].
        """
        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0)  # [batch, row, col]
        freqs = self.freqs.unsqueeze(0)  # [batch, channel, xy]
        phases = self.phases.unsqueeze(0)  # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w)  # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True)  # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])  # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]   # r'_c
        m_r[:, 0, 1] = -t[:, 1]  # r'_s
        m_r[:, 1, 0] = t[:, 1]   # r'_s
        m_r[:, 1, 1] = t[:, 0]   # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1])  # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2]  # t'_x
        m_t[:, 1, 2] = -t[:, 3]  # t'_y
        transforms = m_r @ m_t @ transforms  # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3)  # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # x = F.interpolate(x.permute(0,3,1,2), 24, mode='bilinear', align_corners=True).permute(0,2,3,1)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2)  # [batch, channel, height, width]
        # misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
        return x.contiguous()

    def extra_repr(self):
        """Provides a string representation of the SynthesisInput.

        Returns:
            str: A string representation of the SynthesisInput instance, including
                the dimensions of w, channels, size, sampling rate, and bandwidth.
        """
        return '\n'.join([
            f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
            f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])


class SynthesisLayer(torch.nn.Module):
    """Synthesis Layer."""

    def __init__(
        self,
        w_dim,
        is_torgb,
        is_critically_sampled,
        use_fp16,

        # Input & output specifications.
        in_channels,
        out_channels,
        in_size,
        out_size,
        in_sampling_rate,
        out_sampling_rate,
        in_cutoff,
        out_cutoff,
        in_half_width,
        out_half_width,

        # Hyperparameters.
        conv_kernel=3,
        filter_size=6,
        lrelu_upsampling=2,
        use_radial_filters=False,
        conv_clamp=256,
        magnitude_ema_beta=0.999,
    ):
        """Initializes the SynthesisLayer.

        Args:
            w_dim (int): Intermediate latent (W) dimensionality.
            is_torgb (bool): Indicates if this is the final ToRGB layer.
            is_critically_sampled (bool): Indicates if this layer uses critical sampling.
            use_fp16 (bool): Indicates if this layer uses FP16.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            in_size (int or list): Input spatial size, either an integer or a list of [width, height].
            out_size (int or list): Output spatial size, either an integer or a list of [width, height].
            in_sampling_rate (float): Input sampling rate.
            out_sampling_rate (float): Output sampling rate.
            in_cutoff (float): Input cutoff frequency.
            out_cutoff (float): Output cutoff frequency.
            in_half_width (float): Input transition band half-width.
            out_half_width (float): Output transition band half-width.
            conv_kernel (int, optional): Convolution kernel size. Defaults to 3.
            filter_size (int, optional): Low-pass filter size relative to the lower resolution when up/downsampling. Defaults to 6.
            lrelu_upsampling (int, optional): Relative sampling rate for leaky ReLU. Defaults to 2.
            use_radial_filters (bool, optional): Use radially symmetric downsampling filter. Defaults to False.
            conv_clamp (int, optional): Clamp the output to [-X, +X]. Defaults to 256.
            magnitude_ema_beta (float, optional): Decay rate for the moving average of input magnitudes. Defaults to 0.999.
        """
        super().__init__()
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else conv_kernel
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta
        self.filter_size = filter_size
        self.use_radial_filters = use_radial_filters
        self.lrelu_upsampling = lrelu_upsampling

        # Setup parameters and buffers.
        self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.register_buffer('magnitude_ema', torch.ones([]))

        self.init_upfirdn(in_sampling_rate, out_sampling_rate, in_cutoff, out_cutoff,
                          in_half_width, out_half_width, in_size, out_size)

    def init_upfirdn(self, in_sampling_rate, out_sampling_rate, in_cutoff, out_cutoff,
                     in_half_width, out_half_width, in_size, out_size, filt_suff=''):
        """Initializes the upsampling and downsampling filters.

        Args:
            in_sampling_rate (float): Input sampling rate.
            out_sampling_rate (float): Output sampling rate.
            in_cutoff (float): Input cutoff frequency.
            out_cutoff (float): Output cutoff frequency.
            in_half_width (float): Input transition band half-width.
            out_half_width (float): Output transition band half-width.
            in_size (int or list): Input spatial size.
            out_size (int or list): Output spatial size.
            filt_suff (str, optional): Suffix for filter names. Defaults to ''.
        """
        # reset values
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(self.in_sampling_rate, out_sampling_rate) * (1 if self.is_torgb else self.lrelu_upsampling)
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate

        self.up_taps = self.filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.register_buffer('up_filter' + filt_suff, self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width * 2, fs=self.tmp_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = self.filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = self.use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter' + filt_suff, self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width * 2, fs=self.tmp_sampling_rate, radial=self.down_radial))

        # Compute padding.
        pad_total = (self.out_size - 1) * self.down_factor + 1  # Desired output size before downsampling.
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor  # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2  # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2  # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def _interp_lrelufilters(self, filt_interp_weight):
        """Interpolates between leaky ReLU filters.

        Args:
            filt_interp_weight (float): Interpolation weight.
        """
        if self.up_filter is not None:
            self.up_filter = self.up_filter4.lerp(self.up_filter16, filt_interp_weight)
        if self.down_filter is not None:
            self.down_filter = self.down_filter4.lerp(self.down_filter16, filt_interp_weight)

    def forward(self, x, w, noise_mode='random', force_fp32=False, update_emas=False):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].
            w (torch.Tensor): Weight tensor of shape [batch_size, w_dim].
            noise_mode (str, optional): Mode for adding noise. Defaults to 'random'.
            force_fp32 (bool, optional): Force the use of float32 precision. Defaults to False.
            update_emas (bool, optional): Whether to update the moving averages. Defaults to False.

        Returns:
            torch.Tensor: Output tensor after applying the forward pass.
        """
        assert noise_mode in ['random', 'const', 'none']  # unused
        # misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
        misc.assert_shape(w, [x.shape[0], self.w_dim])

        # Track input magnitude.
        if update_emas:
            with torch.autograd.profiler.record_function('update_magnitude_ema'):
                magnitude_cur = x.detach().to(torch.float32).square().mean()
                self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = self.magnitude_ema.rsqrt()

        # Execute affine layer.
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain

        # Execute modulated conv2d.
        # dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        if torch.onnx.is_in_onnx_export():
            dtype = torch.float16 if (self.use_fp16 and not force_fp32) else torch.float32
            x = modulated_conv2d_export(x=x.to(dtype), w=self.weight, s=styles,
                                        padding=self.conv_kernel - 1, demodulate=(not self.is_torgb), input_gain=input_gain)
        else:
            dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
            x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
                                 padding=self.conv_kernel - 1, demodulate=(not self.is_torgb), input_gain=input_gain)
        # Execute bias, filtered leaky ReLU, and clamping.
        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        if torch.onnx.is_in_onnx_export():
            x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
                                              up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp,
                                              export=True)
        else:
            x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
                                              up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

        # Ensure correct shape and dtype.
        # misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])

        assert x.dtype == dtype, (x.dtype, dtype)
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        """Designs a low-pass filter.

        Args:
            numtaps (int): Number of filter taps.
            cutoff (float): Cutoff frequency.
            width (float): Transition width.
            fs (float): Sampling frequency.
            radial (bool, optional): Whether to use a radially symmetric filter. Defaults to False.

        Returns:
            torch.Tensor: The designed low-pass filter.
        """
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        """Provides a string representation of the class.

        Returns:
            str: A string representation of the instance, including
                various attributes such as dimensions and configurations.
        """
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])


class SynthesisNetwork(torch.nn.Module):
    """Synthesis Network."""

    def __init__(
        self,
        w_dim,
        img_resolution,
        img_channels,
        channel_base=32768,
        channel_max=512,
        num_layers=14,
        num_critical=2,
        first_cutoff=2,
        first_stopband=2 ** 2.1,
        last_stopband_rel=2 ** 0.3,
        margin_size=10,
        output_scale=0.25,
        num_fp16_res=4,
        **layer_kwargs,
    ):
        """Initializes the SynthesisNetwork.

        Args:
            w_dim (int): Intermediate latent (W) dimensionality.
            img_resolution (int): Output image resolution.
            img_channels (int): Number of color channels.
            channel_base (int, optional): Overall multiplier for the number of channels. Defaults to 32768.
            channel_max (int, optional): Maximum number of channels in any layer. Defaults to 512.
            num_layers (int, optional): Total number of layers, excluding Fourier features and ToRGB. Defaults to 14.
            num_critical (int, optional): Number of critically sampled layers at the end. Defaults to 2.
            first_cutoff (float, optional): Cutoff frequency of the first layer. Defaults to 2.
            first_stopband (float, optional): Minimum stopband of the first layer. Defaults to 2**2.1.
            last_stopband_rel (float, optional): Minimum stopband of the last layer, expressed relative to the cutoff. Defaults to 2**0.3.
            margin_size (int, optional): Number of additional pixels outside the image. Defaults to 10.
            output_scale (float, optional): Scale factor for the output image. Defaults to 0.25.
            num_fp16_res (int, optional): Use FP16 for the N highest resolutions. Defaults to 4.
            **layer_kwargs: Additional arguments for SynthesisLayer.
        """
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = num_layers + 2
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.margin_size = margin_size
        self.output_scale = output_scale
        self.num_fp16_res = num_fp16_res
        self.channel_base = channel_base
        self.last_stopband_rel = last_stopband_rel
        self.first_stopband = first_stopband
        self.first_cutoff = first_cutoff
        self.channel_max = channel_max
        self.num_layers = num_layers
        self.conv_kernel = layer_kwargs['conv_kernel']
        self.use_radial_filters = layer_kwargs['use_radial_filters']

        cutoffs, stopbands, sampling_rates, half_widths, sizes, channels = self.get_layer_specs()  # pylint: disable=unused-variable
        self.sampling_rates = sampling_rates

        # Construct layers.
        self.input = SynthesisInput(
            w_dim=self.w_dim, channels=int(channels[0]), size=int(sizes[0]),
            sampling_rate=sampling_rates[0], bandwidth=cutoffs[0])
        self.layer_names = []
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == self.num_layers)
            is_critically_sampled = (idx >= self.num_layers - self.num_critical)  # idx >= 10 - 2, 8, 9, 10
            use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
            layer = SynthesisLayer(
                w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
                in_channels=int(channels[prev]), out_channels=int(channels[idx]),
                in_size=int(sizes[prev]), out_size=int(sizes[idx]),
                in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
                in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev], out_half_width=half_widths[idx],
                **layer_kwargs)
            name = f'L{idx}_{layer.out_size[0]}_{layer.out_channels}'
            setattr(self, name, layer)
            self.layer_names.append(name)

    def get_layer_specs(self):
        """Computes the specifications for each layer.

        Returns:
            tuple: A tuple containing cutoffs, stopbands, sampling rates, half widths, sizes, and channels for each layer.
        """
        # Geometric progression of layer cutoffs and min. stopbands.
        last_cutoff = self.img_resolution / 2  # f_{c,N}
        last_stopband = last_cutoff * self.last_stopband_rel  # f_{t,N}
        exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
        cutoffs = self.first_cutoff * (last_cutoff / self.first_cutoff) ** exponents  # f_c[i]
        stopbands = self.first_stopband * (last_stopband / self.first_stopband) ** exponents  # f_t[i]

        # Compute remaining layer parameters.
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution))))  # s[i]
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs  # f_h[i]
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = self.img_resolution
        channels = np.rint(np.minimum((self.channel_base / 2) / cutoffs, self.channel_max))
        channels[-1] = self.img_channels
        return cutoffs, stopbands, sampling_rates, half_widths, sizes, channels

    def forward(self, ws, **layer_kwargs):
        """Forward pass through the SynthesisNetwork.

        Args:
            ws (torch.Tensor): Input tensor of shape [batch_size, num_ws, w_dim].
            **layer_kwargs: Additional arguments for the layers.

        Returns:
            torch.Tensor: Output image tensor.
        """
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)

        # Execute layers.
        x = self.input(ws[0])

        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, **layer_kwargs)
        if self.output_scale != 1:
            x = x * self.output_scale

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
        x = x.to(torch.float32)
        return x

    def extra_repr(self):
        """Provides a string representation of the SynthesisNetwork.

        Returns:
            str: A string representation of the SynthesisNetwork instance, including
                various attributes such as dimensions and configurations.
        """
        return '\n'.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_layers={self.num_layers:d}, num_critical={self.num_critical:d},',
            f'margin_size={self.margin_size:d}, num_fp16_res={self.num_fp16_res:d}'])


class Generator(torch.nn.Module):
    """Generator Network."""

    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        img_resolution,
        img_channels,
        embed_path=None,
        mapping_kwargs={},
        **synthesis_kwargs,
    ):
        """Initializes the Generator.

        Args:
            z_dim (int): Input latent (Z) dimensionality.
            c_dim (int): Conditioning label (C) dimensionality.
            w_dim (int): Intermediate latent (W) dimensionality.
            img_resolution (int): Output resolution.
            img_channels (int): Number of output color channels.
            mapping_kwargs (dict, optional): Arguments for MappingNetwork. Defaults to {}.
            **synthesis_kwargs: Additional arguments for SynthesisNetwork.
        """
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, embed_path=embed_path, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        """Forward pass through the Generator.

        Args:
            z (torch.Tensor): Input latent tensor of shape [batch_size, z_dim].
            c (torch.Tensor): Conditioning label tensor of shape [batch_size, c_dim].
            truncation_psi (float, optional): Truncation psi value for controlling the truncation trick. Defaults to 1.
            truncation_cutoff (int, optional): Number of layers to apply truncation. Defaults to None.
            update_emas (bool, optional): Whether to update the moving averages. Defaults to False.
            **synthesis_kwargs: Additional arguments for the synthesis network.

        Returns:
            torch.Tensor: Output image tensor.
        """
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img

    def onnx_forward(self, z, labels):
        """Get the dlatent from a list of random seeds, using the truncation trick (this could be optional)."""
        truncation_psi = 1.0  # fixed truncation_psi TODO parameterize?
        if self.c_dim != 0:
            # sample random labels if no class idx is given
            class_indices = torch.argmax(labels, dim=-1)
            w_avg = self.mapping.w_avg.index_select(0, class_indices)
        else:
            w_avg = self.mapping.w_avg.unsqueeze(0)

        w = self.mapping(z, labels)

        w_avg = w_avg.unsqueeze(1).repeat(1, self.mapping.num_ws, 1)
        w = w_avg + (w - w_avg) * truncation_psi

        images = self.synthesis(w)

        return images

    def load_pretrained_embedding(self, embedding_checkpoint):
        """Find embedding layers and load the pretrained embedding checkpoint for each layers"""
        # Example: Find all ".embed" layers in the discriminator
        load_pretrained_embedding_for_embed_layers(self, embedding_checkpoint)


class SuperresGenerator(torch.nn.Module):
    """Super-resolution Generator Network."""

    def __init__(
        self,
        G_stem,
        # img_resolution,
        head_layers,
        up_factor,
        **synthesis_kwargs,
    ):
        assert up_factor in [2, 4, 8, 16], "Supported up_factors: [2, 4, 8, 16]"

        super().__init__()
        self.up_factor = up_factor

        self.mapping = G_stem.mapping
        self.synthesis = G_stem.synthesis

        # update G params
        self.z_dim = G_stem.z_dim
        self.c_dim = G_stem.c_dim
        self.w_dim = G_stem.w_dim
        self.img_channels = G_stem.img_channels
        self.channel_base = G_stem.synthesis.channel_base
        self.channel_max = G_stem.synthesis.channel_max
        self.margin_size = G_stem.synthesis.margin_size
        self.last_stopband_rel = G_stem.synthesis.last_stopband_rel
        self.num_critical = G_stem.synthesis.num_critical
        self.num_fp16_res = G_stem.synthesis.num_fp16_res
        self.conv_kernel = G_stem.synthesis.conv_kernel
        self.use_radial_filters = G_stem.synthesis.use_radial_filters

        # cut off critically sampled layers
        for name in reversed(self.synthesis.layer_names):
            if getattr(self.synthesis, name).is_critically_sampled:
                delattr(self.synthesis, name)
                self.synthesis.layer_names.pop()
        stem_len = len(self.synthesis.layer_names) + 1

        # update G and G.synthesis params
        self.img_resolution = G_stem.img_resolution * up_factor
        self.synthesis.img_resolution = self.img_resolution
        # assert img_resolution == self.img_resolution, f"Resolution mismatch. Dataset: {img_resolution}, G output: {self.img_resolution}"

        self.num_layers = stem_len + head_layers
        self.synthesis.num_layers = self.num_layers

        self.num_ws = stem_len + head_layers + 1
        self.mapping.num_ws = self.num_ws
        self.synthesis.num_ws = self.num_ws

        # initialize new_layers
        last_stem_layer = getattr(self.synthesis, self.synthesis.layer_names[-1])
        fparams = self.compute_superres_filterparams(up_factor, self.img_resolution, last_stem_layer, head_layers)

        self.head_layer_names = []
        for idx in range(head_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == head_layers)
            is_critically_sampled = (idx >= head_layers - self.num_critical)
            use_fp16 = (fparams.sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
            layer = SynthesisLayer(
                w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
                in_channels=int(fparams.channels[prev]), out_channels=int(fparams.channels[idx]),
                in_size=int(fparams.sizes[prev]), out_size=int(fparams.sizes[idx]),
                in_sampling_rate=int(fparams.sampling_rates[prev]), out_sampling_rate=int(fparams.sampling_rates[idx]),
                in_cutoff=fparams.cutoffs[prev], out_cutoff=fparams.cutoffs[idx],
                in_half_width=fparams.half_widths[prev], out_half_width=fparams.half_widths[idx],
                conv_kernel=self.conv_kernel, use_radial_filters=self.use_radial_filters,
            )
            name = f'L{idx + stem_len}_{layer.out_size[0]}_{layer.out_channels}'
            setattr(self.synthesis, name, layer)
            self.synthesis.layer_names.append(name)
            self.head_layer_names.append(name)

    def reinit_stem(self, G_stem):
        """Reinitialize the stem layers from a new G_stem.

        Args:
            G_stem (torch.nn.Module): The new base generator network.
        """
        logging.info("Reinitialize stem")

        # cut off critically sampled layers
        for name in reversed(G_stem.synthesis.layer_names):
            if getattr(G_stem.synthesis, name).is_critically_sampled:
                delattr(G_stem.synthesis, name)
                G_stem.synthesis.layer_names.pop()

        # synthesis reinit
        for name in G_stem.synthesis.layer_names:
            layer_src = getattr(G_stem.synthesis, name)
            layer_dst = getattr(self.synthesis, name)
            misc.copy_params_and_buffers(layer_src, layer_dst)

        # mapping reinit
        misc.copy_params_and_buffers(G_stem.mapping, self.mapping)

    def reinit_mapping(self):
        """Reinitialize the mapping network."""
        logging.info("Reinitialize mapping")
        self.mapping = self.new_mapping

    def compute_superres_filterparams(self, up_factor, img_resolution, last_stem_layer, head_layers, num_critical=2):
        """Compute filter parameters for the super-resolution layers.

        Args:
            up_factor (int): Upscaling factor.
            img_resolution (int): Output resolution.
            last_stem_layer (torch.nn.Module): The last layer of the stem network.
            head_layers (int): Number of additional layers for the super-resolution head.
            num_critical (int, optional): Number of critically sampled layers. Defaults to 2.

        Returns:
            dnnlib.EasyDict: Filter parameters for the super-resolution layers.
        """
        # begin with output of last stem layer
        first_cutoff = last_stem_layer.out_cutoff
        first_stopband = last_stem_layer.out_half_width + first_cutoff

        # Geometric progression of layer cutoffs and min. stopbands.
        last_cutoff = img_resolution / 2  # f_{c,N}
        last_stopband = last_cutoff * self.last_stopband_rel  # f_{t,N}
        exponents = np.minimum(np.arange(head_layers + 1) / (head_layers - num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents  # f_c[i]
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents  # f_t[i]

        # set sampling rates
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, img_resolution))))
        sampling_rates[0] = last_stem_layer.out_sampling_rate

        # Compute remaining layer parameters.
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs  # f_h[i]
        sizes = sampling_rates + self.margin_size * 2
        sizes[-2:] = img_resolution
        channels = np.rint(np.minimum((self.channel_base / 2) / cutoffs, self.channel_max))
        channels[0] = last_stem_layer.out_channels
        channels[-1] = self.img_channels

        # save in dict
        fparams = dnnlib.EasyDict()
        fparams.cutoffs = cutoffs
        fparams.stopbands = stopbands
        fparams.sampling_rates = sampling_rates
        fparams.half_widths = half_widths
        fparams.sizes = sizes
        fparams.channels = channels
        return fparams

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        """Forward pass through the SuperresGenerator.

        Args:
            z (torch.Tensor): Input latent tensor of shape [batch_size, z_dim].
            c (torch.Tensor): Conditioning label tensor of shape [batch_size, c_dim].
            truncation_psi (float, optional): Truncation psi value for controlling the truncation trick. Defaults to 1.
            truncation_cutoff (int, optional): Number of layers to apply truncation. Defaults to None.
            update_emas (bool, optional): Whether to update the moving averages. Defaults to False.
            **synthesis_kwargs: Additional arguments for the synthesis network.

        Returns:
            torch.Tensor: Output image tensor.
        """
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img

    def onnx_forward(self, z, labels):
        """Get the dlatent from a list of random seeds, using the truncation trick (this could be optional)."""
        truncation_psi = 1.0  # fixed truncation_psi TODO parameterize?
        if self.c_dim != 0:
            # sample random labels if no class idx is given
            class_indices = torch.argmax(labels, dim=-1)
            w_avg = self.mapping.w_avg.index_select(0, class_indices)
        else:
            w_avg = self.mapping.w_avg.unsqueeze(0)

        w = self.mapping(z, labels)

        w_avg = w_avg.unsqueeze(1).repeat(1, self.mapping.num_ws, 1)
        w = w_avg + (w - w_avg) * truncation_psi

        images = self.synthesis(w)

        return images

    def load_pretrained_embedding(self, embedding_checkpoint):
        """Find embedding layers and load the pretrained embedding checkpoint for each layers"""
        # Example: Find all ".embed" layers in the discriminator
        load_pretrained_embedding_for_embed_layers(self, embedding_checkpoint)
