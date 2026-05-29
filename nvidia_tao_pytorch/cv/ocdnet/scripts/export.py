#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export module."""
import os
import torch
from torch.onnx import register_custom_op_symbolic
import copy
import onnx
import onnx_graphsurgeon as onnx_gs
from torchvision.ops import DeformConv2d
import tempfile

from nvidia_tao_pytorch.config.ocdnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.cv.ocdnet.model.model import Model
from nvidia_tao_pytorch.cv.ocdnet.utils.util import load_checkpoint
from omegaconf import OmegaConf


def symbolic_dcnv2_forward(g, *inputs):
    """symbolic_dcnv2_forward"""
    # weights as last input to align with TRT plugin
    return g.op("ModulatedDeformConv2d", inputs[0], inputs[2], inputs[3], inputs[1])


class Export():
    """Export OCDNet model."""

    def __init__(self, config_file):
        """Initialize."""
        self.model_path = config_file.export.checkpoint
        self.config_file = config_file
        self.opset_version = config_file.export.opset_version
        self.gpu_id = config_file.export.gpu_id
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        checkpoint = load_checkpoint(self.model_path, to_cpu=True)

        config = OmegaConf.to_container(config_file)
        config['model']['pretrained'] = False
        config["dataset"]["train_dataset"] = config["dataset"]["validate_dataset"]
        load_pruned_graph = config['model']['load_pruned_graph']
        if load_pruned_graph:
            self.model = load_checkpoint(config['model']['pruned_graph_path'], only_state_dict=False)
        else:
            self.model = Model(config['model'])
        layers = checkpoint.keys()
        ckpt = dict()
        # Support loading official pretrained weights for eval
        for layer in layers:
            new_layer = layer
            if new_layer.startswith("model.module."):
                new_layer = new_layer[13:]
            if new_layer == "decoder.in5.weight":
                new_layer = "neck.in5.weight"
            elif new_layer == "decoder.in4.weight":
                new_layer = "neck.in4.weight"
            elif new_layer == "decoder.in3.weight":
                new_layer = "neck.in3.weight"
            elif new_layer == "decoder.in2.weight":
                new_layer = "neck.in2.weight"
            elif new_layer == "decoder.out5.0.weight":
                new_layer = "neck.out5.0.weight"
            elif new_layer == "decoder.out4.0.weight":
                new_layer = "neck.out4.0.weight"
            elif new_layer == "decoder.out3.0.weight":
                new_layer = "neck.out3.0.weight"
            elif new_layer == "decoder.out2.weight":
                new_layer = "neck.out2.weight"
            elif new_layer == "decoder.binarize.0.weight":
                new_layer = "head.binarize.0.weight"
            elif new_layer == "decoder.binarize.1.weight":
                new_layer = "head.binarize.1.weight"
            elif new_layer == "decoder.binarize.1.bias":
                new_layer = "head.binarize.1.bias"
            elif new_layer == "decoder.binarize.1.running_mean":
                new_layer = "head.binarize.1.running_mean"
            elif new_layer == "decoder.binarize.1.running_var":
                new_layer = "head.binarize.1.running_var"
            elif new_layer == "decoder.binarize.3.weight":
                new_layer = "head.binarize.3.weight"
            elif new_layer == "decoder.binarize.3.bias":
                new_layer = "head.binarize.3.bias"
            elif new_layer == "decoder.binarize.4.weight":
                new_layer = "head.binarize.4.weight"
            elif new_layer == "decoder.binarize.4.bias":
                new_layer = "head.binarize.4.bias"
            elif new_layer == "decoder.binarize.4.running_mean":
                new_layer = "head.binarize.4.running_mean"
            elif new_layer == "decoder.binarize.4.running_var":
                new_layer = "head.binarize.4.running_var"
            elif new_layer == "decoder.binarize.6.weight":
                new_layer = "head.binarize.6.weight"
            elif new_layer == "decoder.binarize.6.bias":
                new_layer = "head.binarize.6.bias"
            elif new_layer == "decoder.thresh.0.weight":
                new_layer = "head.thresh.0.weight"
            elif new_layer == "decoder.thresh.1.weight":
                new_layer = "head.thresh.1.weight"
            elif new_layer == "decoder.thresh.1.bias":
                new_layer = "head.thresh.1.bias"
            elif new_layer == "decoder.thresh.1.running_mean":
                new_layer = "head.thresh.1.running_mean"
            elif new_layer == "decoder.thresh.1.running_var":
                new_layer = "head.thresh.1.running_var"
            elif new_layer == "decoder.thresh.3.weight":
                new_layer = "head.thresh.3.weight"
            elif new_layer == "decoder.thresh.3.bias":
                new_layer = "head.thresh.3.bias"
            elif new_layer == "decoder.thresh.4.weight":
                new_layer = "head.thresh.4.weight"
            elif new_layer == "decoder.thresh.4.bias":
                new_layer = "head.thresh.4.bias"
            elif new_layer == "decoder.thresh.4.running_mean":
                new_layer = "head.thresh.4.running_mean"
            elif new_layer == "decoder.thresh.4.running_var":
                new_layer = "head.thresh.4.running_var"
            elif new_layer == "decoder.thresh.6.weight":
                new_layer = "head.thresh.6.weight"
            elif new_layer == "decoder.thresh.6.bias":
                new_layer = "head.thresh.6.bias"
            elif "num_batches_tracked" in new_layer:
                continue
            elif "backbone.fc" in new_layer:
                continue
            elif "backbone.smooth" in new_layer:
                continue
            ckpt[new_layer] = checkpoint[layer]

        self.model.load_state_dict(ckpt)
        self.model.to(self.device)

    def export(self):
        """Export."""
        input_w = self.config_file.export.width
        input_h = self.config_file.export.height
        dummy_image = torch.rand(
            (1, 3, input_h, input_w),
            dtype=torch.float32,
            device=self.device
        )

        self.model.eval()

        # Set default output filename if the filename isn't provided over the command line.
        if self.config_file.export.onnx_file is None:
            split_name = os.path.splitext(self.model_path)[0]
            self.config_file.export.onnx_file = "{}.onnx".format(split_name)

        # Warn the user if an exported file already exists.
        if os.path.exists(self.config_file.export.onnx_file):
            raise FileExistsError(f"Output file already exists at {self.config_file.export.onnx_file}")

        self.output_model = self.config_file.export.onnx_file

        handle, temp_onnx = tempfile.mkstemp()
        os.close(handle)

        # Register custom symbolic function
        register_custom_op_symbolic("torchvision::deform_conv2d", symbolic_dcnv2_forward, self.opset_version)

        with torch.no_grad():
            torch.onnx.export(
                self.model,
                (dummy_image,),
                temp_onnx,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                keep_initializers_as_inputs=True,
                input_names=['input'],
                output_names=['pred'],
                verbose=self.config_file.export.verbose,
                dynamic_axes={
                    "input": {0: "batch"},
                },
                dynamo=False,
            )
        # Import and add DCNv2 attributes
        onnx_model = onnx.load(temp_onnx)
        gs_graph = onnx_gs.import_onnx(onnx_model)
        layer_dict = {}
        attrs_dict = {}
        for name, layer in self.model.named_modules():
            if isinstance(layer, DeformConv2d):
                attrs_dict["stride"] = list(layer.stride)
                attrs_dict["padding"] = list(layer.padding)
                attrs_dict["dilation"] = list(layer.dilation)
                attrs_dict["group"] = 1
                attrs_dict["deformable_group"] = 1
                name = name.replace("backbone.", "") + ".ModulatedDeformConv2d"
                layer_dict[name] = copy.deepcopy(attrs_dict)
        for node in gs_graph.nodes:
            if node.op == "ModulatedDeformConv2d":
                key = (".".join(node.name.split("/")[-3:]))
                node.attrs = layer_dict[key]

        gs_graph.fold_constants(size_threshold=1024 * 1024 * 1024)
        gs_graph.cleanup().toposort()
        new_onnx_model = onnx_gs.export_onnx(gs_graph)

        onnx.save(new_onnx_model, self.output_model)

        print("Model exported to {}".format(self.output_model))

        os.remove(temp_onnx)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="export", schema=ExperimentConfig
)
@monitor_status(name="OCDNet", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """Run the export process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)

    exporter = Export(config_file=cfg)
    exporter.export()


if __name__ == "__main__":
    main()
