# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Generates TRT compatible Mask Grounding DINO onnx model. """

import torch
from torch.onnx import register_custom_op_symbolic

from nvidia_tao_pytorch.cv.grounding_dino.utils.onnx_export import ONNXExporter
from nvidia_tao_pytorch.cv.grounding_dino.utils.onnx_export import nvidia_msda


class MaskGDINOExporter(ONNXExporter):
    """MaskGDINO ONNX Exporter."""

    def export_model(self, model, batch_size, onnx_file, args, do_constant_folding=False, opset_version=17,
                     output_names=None, input_names=None, verbose=False):
        """ Export_model.

        The do_constant_folding = False avoids MultiscaleDeformableAttnPlugin_TRT error (tensors on 2 devices) when torch > 1.9.0.
        However, it would cause tensorrt 8.0.3.4 (nvcr.io/nvidia/pytorch:21.11-py3 env) reports clip node error.
        This error is fixed in tensorrt >= 8.2.1.8 (nvcr.io/nvidia/tensorrt:22.01-py3).

        Args:
            model (nn.Module): torch model to export.
            batch_size (int): batch size of the ONNX model. -1 means dynamic batch size.
            onnx_file (str): output path of the onnx file.
            args (Tuple[torch.Tensor]): Tuple of input tensors.
            do_constant_folding (bool): flag to indicate whether to fold constants in the ONNX model.
            opset_version (int): opset_version of the ONNX file.
            output_names (str): output names of the ONNX file.
            input_names (str): input names of the ONNX file.
            verbose (bool): verbosity level.
        """
        if batch_size is None or batch_size == -1:
            dynamic_axes = {
                "inputs": {0: "batch_size"},
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "position_ids": {0: "batch_size"},
                "token_type_ids": {0: "batch_size"},
                "text_token_mask": {0: "batch_size"},
                "pred_logits": {0: "batch_size"},
                "pred_boxes": {0: "batch_size"},
                "pred_masks": {0: "batch_size"},
                "no_targets": {0: "batch_size"},
                "union_mask_logits": {0: "batch_size"},
            }
        else:
            dynamic_axes = None

        # CPU version requires opset_version > 16
        if not next(model.parameters()).is_cuda and opset_version < 16:
            print(f"CPU version of Deformable MHA requires opset version larger than 16. Overriding provided opset {opset_version} to 17.")
            opset_version = 17

        register_custom_op_symbolic('nvidia::MultiscaleDeformableAttnPlugin_TRT', nvidia_msda, opset_version)

        torch.onnx.export(model, args, onnx_file,
                          input_names=input_names, output_names=output_names, export_params=True,
                          training=torch.onnx.TrainingMode.EVAL, opset_version=opset_version, do_constant_folding=do_constant_folding,
                          custom_opsets={"nvidia": opset_version}, verbose=verbose, dynamic_axes=dynamic_axes)
