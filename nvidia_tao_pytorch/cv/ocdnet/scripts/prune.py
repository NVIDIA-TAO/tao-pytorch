# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prune module."""
import os
import torch
from torch import nn
import torch_pruning as tp
from typing import Sequence
from omegaconf import OmegaConf
from torchvision.ops import DeformConv2d

from nvidia_tao_pytorch.config.ocdnet.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.cv.ocdnet.model.model import Model
from nvidia_tao_pytorch.cv.ocdnet.data_loader.build_dataloader import get_dataloader
from nvidia_tao_pytorch.cv.ocdnet.utils.util import mkdir, load_checkpoint
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs
from nvidia_tao_pytorch.cv.ocdnet.model.backbone.fan import TokenMixing, ChannelProcessing
from nvidia_tao_pytorch.cv.backbone.fan import ClassAttn
from nvidia_tao_pytorch.cv.ocdnet.pruning.dependency import TAO_DependencyGraph

# force pycuda on primary context before using TensorRT
import pycuda
import pycuda.autoinit
pyc_dev = pycuda.autoinit.device
pyc_ctx = pyc_dev.retain_primary_context()
tp.dependency.DependencyGraph.update_index_mapping = TAO_DependencyGraph.update_index_mapping


class DeformConv2dPruner(tp.function.ConvPruner):
    """DCNv2 Pruning."""

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        """Prune out channels parameters."""
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        keep_idxs.sort()
        layer.out_channels = layer.out_channels - len(idxs)
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 0)
        if layer.bias is not None:
            layer.bias = self._prune_parameter_and_grad(layer.bias, keep_idxs, 0)
        return layer

    def prune_in_channels(self, layer: nn.Linear, idxs: Sequence[int]) -> nn.Module:
        """Prune in channels parameters."""
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        keep_idxs.sort()
        layer.in_channels = layer.in_channels - len(idxs)
        if layer.groups > 1:
            keep_idxs = keep_idxs[:len(keep_idxs) // layer.groups]
        layer.weight = self._prune_parameter_and_grad(layer.weight, keep_idxs, 1)
        # no bias pruning because it does not change the output channels
        return layer


class Prune():
    """Prune."""

    def __init__(self, config):
        """Initialize."""
        config['model']['pretrained'] = False
        self.activate_checkpoint = config['model']['activation_checkpoint']
        config['model']['activation_checkpoint'] = False
        self.validate_loader = get_dataloader(config['dataset']['validate_dataset'], False)
        self.model = None
        self.ch_sparsity = config['prune']['ch_sparsity']
        self.p = config['prune']['p']
        self.round_to = config['prune']['round_to']
        self.output_dir = config['prune']['results_dir']
        self.gpu_id = config['prune']['gpu_id']
        self.verbose = config['prune']['verbose']
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        checkpoint = load_checkpoint(config['prune']['checkpoint'], to_cpu=True)

        self.model = Model(config['model'])
        self.model_backbone = config['model']['backbone']
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
            elif "kv.weight" in new_layer:
                dim = checkpoint[layer].shape[-1]
                ckpt[new_layer.replace('kv', 'k')] = checkpoint[layer][:dim, :]
                ckpt[new_layer.replace('kv', 'v')] = checkpoint[layer][dim:, :]
                continue
            elif "kv.bias" in new_layer:
                dim = checkpoint[layer].shape[-1] // 2
                ckpt[new_layer.replace('kv', 'k')] = checkpoint[layer][:dim]
                ckpt[new_layer.replace('kv', 'v')] = checkpoint[layer][dim:]
                continue
            ckpt[new_layer] = checkpoint[layer]
        self.model.load_state_dict(ckpt)
        self.model.to(self.device)

    def prune(self):
        """Prune function."""
        input_dict = next(iter(self.validate_loader))
        if self.verbose:
            print('---------------- original model ----------------')
            print(self.model)
        with torch.no_grad():
            if self.model is not None:
                for key, value in input_dict.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            input_dict[key] = value.to(self.device)

        example_inputs = input_dict["img"]
        imp = tp.importance.MagnitudeImportance(p=self.p, group_reduction="mean")
        base_flops, base_params = tp.utils.count_ops_and_params(self.model, example_inputs)
        channel_groups = {}

        # All heads should be pruned simultaneously, so we group channels by head.
        for m in self.model.modules():
            if isinstance(m, TokenMixing):
                channel_groups[m.q] = m.num_heads
                channel_groups[m.k] = m.num_heads
                channel_groups[m.v] = m.num_heads

            if isinstance(m, ChannelProcessing):
                channel_groups[m.q] = m.num_heads

            if isinstance(m, ClassAttn):
                channel_groups[m.q] = m.num_heads
                channel_groups[m.k] = m.num_heads
                channel_groups[m.v] = m.num_heads

        ignored_layers = []
        for name, module in self.model.named_modules():
            if 'head' in name:
                ignored_layers.append(module)
            if 'fan' in self.model_backbone and 'linear_fuse' in name:
                ignored_layers.append(module)
            if 'resnet' in self.model_backbone:
                if 'conv2_offset' in name:
                    ignored_layers.append(module)
                if 'neck.out' in name:
                    ignored_layers.append(module)

        pruner = tp.pruner.MagnitudePruner(self.model,
                                           example_inputs,
                                           importance=imp,
                                           ch_sparsity=self.ch_sparsity,
                                           channel_groups=channel_groups,
                                           ignored_layers=ignored_layers,
                                           round_to=self.round_to,
                                           root_module_types=[nn.Conv2d, nn.Linear, DeformConv2d],
                                           customized_pruners={DeformConv2d: DeformConv2dPruner()})

        for g in pruner.step(interactive=True):
            if self.verbose:
                print(g)
            g.prune()

        for m in self.model.modules():
            if isinstance(m, TokenMixing):
                m.head_dim = m.q.out_features // m.num_heads
                m.dim = m.q.out_features

            if isinstance(m, ClassAttn):
                m.head_dim = m.q.out_features // m.num_heads
                m.dim = m.q.out_features

            if isinstance(m, ChannelProcessing):
                m.head_dim = m.q.out_features // m.num_heads
                m.dim = m.q.out_features

        # Do inference to sanity check the pruned model
        with torch.no_grad():
            self.model(input_dict["img"])
        if 'fan' in self.model_backbone and self.activate_checkpoint:
            self.model.backbone.use_checkpoint = True
        if self.verbose:
            print(self.model)
        # Save pruned model
        pruned_flops, pruned_params = tp.utils.count_ops_and_params(self.model, example_inputs)
        print("Base FLOPs: %d G, Pruned FLOPs: %d G" % (base_flops / 1e9, pruned_flops / 1e9))
        print("Base Params: %d M, Pruned Params: %d M" % (base_params / 1e6, pruned_params / 1e6))
        print(f"Pruning ratio: {pruned_params / base_params}")
        if not os.path.exists(self.output_dir):
            mkdir(self.output_dir)
        assert os.path.exists(self.output_dir) and os.path.isdir(self.output_dir), "The output_folder should exist."
        save_path = os.path.join(self.output_dir, f"pruned_{self.ch_sparsity}.pth")
        torch.save(self.model, save_path)
        print(f'Pruned model save to {save_path}')


def run_experiment(experiment_config):
    """Run experiment."""
    gpu_id = experiment_config.prune.gpu_id
    torch.cuda.set_device(gpu_id)

    experiment_config = OmegaConf.to_container(experiment_config)

    pruner = Prune(experiment_config)
    pruner.prune()

    pyc_ctx.pop()


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="prune", schema=ExperimentConfig
)
@monitor_status(name="OCDNet", mode="prune")
def main(cfg: ExperimentConfig) -> None:
    """Run the pruning process."""
    # Obfuscate logs.
    obfuscate_logs(cfg)

    pyc_ctx.push()

    run_experiment(experiment_config=cfg)


if __name__ == "__main__":
    main()
