# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""linear quantiztion warp module."""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import sys
import math
from modelopt.torch.quantization.nn.modules.quant_linear import QuantLinear
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer


def gelu(x):
    """GELU"""
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


def bias_gelu(bias, y):
    """
    GELU with bias
    used only for triton inference
    """
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


def bias_gelu_training(bias, y):
    """
    GELU with bias
    used specifically for training since torch.nn.functional.gelu breaks ONNX export
    """
    x = bias + y
    return torch.nn.functional.gelu(x)  # Breaks ONNX export


def bias_tanh(bias, y):
    """Tanh bias"""
    x = bias + y
    return torch.tanh(x)


def swish(x):
    """swish"""
    return x * torch.sigmoid(x)


def bias_noact(bias, y):
    """bias without activation"""
    return bias + y


# torch.nn.functional.gelu(x) # Breaks ONNX export
ACT2FN = {"gelu": gelu, "bias_gelu": bias_gelu, "bias_tanh": bias_tanh, "relu": torch.nn.functional.relu, "swish": swish,
          "bias_noact": bias_noact}


class QuantizedConv2d(nn.Module):
    """Conv2d with INT8 quantization"""

    __constants__ = ['bias']

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=True):
        """Initialize"""
        super(QuantizedConv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = Parameter(torch.Tensor(out_channel, in_channel, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channel))
        else:
            self.register_parameter('bias', None)

        self._input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
        self._weight_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_weight)
        self._aftergemm_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)

        self.reset_parameters()

    def reset_parameters(self):
        """reset parameters"""
        init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """forward function"""
        x = self._input_quantizer(x)
        weight = self._weight_quantizer(self.weight)
        output = self._aftergemm_quantizer(F.conv2d(x, weight, bias=None, stride=self.stride))

        if self.bias is not None:
            output += self.bias.reshape(self.out_channel, 1, 1)

        return output


class LinearActivation(nn.Module):
    """Fused Linear and Activation Module."""

    __constants__ = ['bias']

    def __init__(self, in_features, out_features, act='noact', bias=True, do_quant=True):
        """Initialize"""
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_fn = nn.Identity()
        self.biased_act_fn = None

        if isinstance(act, str) or (sys.version_info[0] == 2 and isinstance(act, str)):
            if bias and 'bias' not in act:
                act = 'bias_' + act
                self.biased_act_fn = ACT2FN[act]
            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.do_quant = do_quant
        if do_quant:
            self._input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
            self._weight_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_weight)
            self._aftergemm_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)

        self.reset_parameters()

    def reset_parameters(self):
        """reset parameters"""
        init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """forward function"""
        if self.do_quant:
            x = self._input_quantizer(x)
            weight = self._weight_quantizer(self.weight)
        else:
            weight = self.weight

        if self.bias is not None:
            if self.do_quant:
                out = self.biased_act_fn(self.bias, self._aftergemm_quantizer(F.linear(x, weight, None)))
            else:
                out = self.biased_act_fn(self.bias, F.linear(x, weight, None))
        else:
            if self.do_quant:
                out = self.act_fn(self._aftergemm_quantizer(F.linear(x, weight, None)))
            else:
                out = self.act_fn(F.linear(x, weight, None))
        return out

    def extra_repr(self) -> str:
        """extra representation of the module"""
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
