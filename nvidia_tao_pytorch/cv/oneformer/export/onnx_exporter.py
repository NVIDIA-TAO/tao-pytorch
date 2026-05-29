# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Generates TRT compatible OneFormer onnx model. """

import torch
import onnx
from torch.onnx import register_custom_op_symbolic


# register plugin
def nvidia_msda(g, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
    """Returns nvidia_msda."""
    return g.op("nvidia::MultiscaleDeformableAttnPlugin_TRT", value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)


class ONNXExporter(object):
    """Onnx Exporter"""

    @classmethod
    def setUpClass(cls):
        """SetUpclass to set the manual seed for reproduceability"""
        torch.manual_seed(152)

    def export_model(self, model, batch_size, onnx_file, dummy_input, do_constant_folding=False, opset_version=17,
                     output_names=None, input_names=None, verbose=False):
        """ Export_model.

        The do_constant_folding = False avoids MultiscaleDeformableAttnPlugin_TRT error (tensors on 2 devices) when torch > 1.9.0.
        However, it would cause tensorrt 8.0.3.4 (nvcr.io/nvidia/pytorch:21.11-py3 env) reports clip node error.
        This error is fixed in tensorrt >= 8.2.1.8 (nvcr.io/nvidia/tensorrt:22.01-py3).

        Args:
            model (nn.Module): torch model to export.
            batch_size (int): batch size of the ONNX model. -1 means dynamic batch size.
            onnx_file (str): output path of the onnx file.
            dummy_input (torch.Tensor): input tensor.
            do_constant_folding (bool): flag to indicate whether to fold constants in the ONNX model.
            opset_version (int): opset_version of the ONNX file.
            output_names (str): output names of the ONNX file.
            input_names (str): input names of the ONNX file.
            verbose (bool): verbosity level.
        """
        if batch_size is None or batch_size == -1:
            dynamic_axes = {"inputs": {0: "batch"},
                            "pred_logits": {0: "batch"},
                            "pred_masks": {0: "batch"}}
        else:
            dynamic_axes = None

        # CPU version requires opset_version > 16
        if not next(model.parameters()).is_cuda and opset_version < 16:
            print(f"CPU version of Deformable MHA requires opset version larger than 16. Overriding provided opset {opset_version} to 17.")
            opset_version = 17
        register_custom_op_symbolic('nvidia::MultiscaleDeformableAttnPlugin_TRT', nvidia_msda, opset_version)
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_file,
                input_names=input_names,
                output_names=output_names,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                custom_opsets={"nvidia": opset_version},
                verbose=verbose,
                dynamic_axes=dynamic_axes,
                dynamo=False)

    @staticmethod
    def check_onnx(onnx_file):
        """Check onnx file.

        Args:
            onnx_file (str): path to ONNX file.
        """
        model = onnx.load(onnx_file)
        onnx.checker.check_model(model)
