# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Generates TRT compatible SegFormer onnx model. """

import os.path as osp
from typing import Any, Optional, Union

import onnx
import torch
import mmengine
from torch.onnx import register_custom_op_symbolic


# register plugin
def nvidia_msda(g, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
    """Returns nvidia_msda."""
    return g.op(
        "nvidia::MultiscaleDeformableAttnPlugin_TRT",
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights
    )


class ONNXExporter(object):
    """Onnx Exporter"""

    @classmethod
    def setUpClass(cls):
        """SetUpclass to set the manual seed for reproduceability"""
        torch.manual_seed(123)

    def export_model(self, img: Any, work_dir: str, save_file: str, deploy_cfg: Union[str, mmengine.Config],
                     model_cfg: Union[str, mmengine.Config], model_checkpoint: Optional[str] = None, device: str = 'cuda:0'):
        """ Convert PyTorch model to ONNX model.
        This is a workaround for exporting onnx for model with custom operator with mmengine backend.

        The do_constant_folding = False avoids MultiscaleDeformableAttnPlugin_TRT error (tensors on 2 devices) when torch > 1.9.0.
        However, it would cause tensorrt 8.0.3.4 (nvcr.io/nvidia/pytorch:21.11-py3 env) reports clip node error.
        This error is fixed in tensorrt >= 8.2.1.8 (nvcr.io/nvidia/tensorrt:22.01-py3).

        Args:
            img (str | np.ndarray | torch.Tensor): Input image used to assist
                converting model.
            work_dir (str): A working directory to save files.
            save_file (str): Filename to save onnx model.
            deploy_cfg (str | mmengine.Config): Deployment config file or
                Config object.
            model_cfg (str | mmengine.Config): Model config file or Config object.
            model_checkpoint (str): A checkpoint path of PyTorch model,
                defaults to `None`.
            device (str): A string specifying device type, defaults to 'cuda:0'.
        """
        # FIXME: This is a workaround. We'll need to make it more aligned with other pipeline such as DINO and VisualChangeNet
        from mmdeploy.utils import (get_dynamic_axes, get_input_shape, get_onnx_config, load_config)

        # load deploy_cfg if necessary
        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
        mmengine.mkdir_or_exist(osp.abspath(work_dir))

        input_shape = get_input_shape(deploy_cfg)

        # create model and inputs
        from mmdeploy.apis import build_task_processor
        task_processor = build_task_processor(model_cfg, deploy_cfg, device)

        torch_model = task_processor.build_pytorch_model(model_checkpoint)
        # Skipping the data dictionary since data is not being used in the tuple that's returned.
        _, model_inputs = task_processor.create_input(
            img,
            input_shape,
            data_preprocessor=getattr(torch_model, 'data_preprocessor', None))

        if isinstance(model_inputs, list) and len(model_inputs) == 1:
            model_inputs = model_inputs[0]

        # export to onnx
        context_info = dict()
        context_info['deploy_cfg'] = deploy_cfg
        output_prefix = osp.join(
            work_dir,
            osp.splitext(osp.basename(save_file))[0]
        )

        onnx_cfg = get_onnx_config(deploy_cfg)
        opset_version = onnx_cfg.get('opset_version', 11)

        input_names = onnx_cfg['input_names']
        output_names = onnx_cfg['output_names']
        axis_names = input_names + output_names
        dynamic_axes = get_dynamic_axes(deploy_cfg, axis_names)
        verbose = not onnx_cfg.get('strip_doc_string', True) or onnx_cfg.get('verbose', False)
        register_custom_op_symbolic('nvidia::MultiscaleDeformableAttnPlugin_TRT', nvidia_msda, opset_version)
        with torch.no_grad():
            torch.onnx.export(
                torch_model, model_inputs, output_prefix + '.onnx',
                input_names=input_names, output_names=output_names, export_params=True,
                training=torch.onnx.TrainingMode.EVAL, opset_version=opset_version, do_constant_folding=True,
                custom_opsets={"nvidia": opset_version}, verbose=verbose, dynamic_axes=dynamic_axes,
                dynamo=False,
            )

    @staticmethod
    def check_onnx(onnx_file):
        """Check onnx file.

        Args:
            onnx_file (str): path to ONNX file.
        """
        model = onnx.load(onnx_file)
        onnx.checker.check_model(model)
