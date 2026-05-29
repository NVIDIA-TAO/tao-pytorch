# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prune OCRNet script.
"""
import os
import argparse
import torch


from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.config.ocrnet.default_config import ExperimentConfig


def legacy_pruner(opt, model, device):
    """Legacy pruner based on deprecated torch pruning 0.2.7 API

    Args:
        opt (argparse): options
        model (nn.module): The model to be pruned
        device (str): The device parameter
    """
    import nvidia_tao_pytorch.pruning.torch_pruning_v0 as tp

    num_params_before_pruning = tp.utils.count_params(model)
    # 1. build dependency graph
    dep_graph = tp.DependencyGraph()
    dep_graph.build_dependency(model, example_inputs=(torch.randn([1, opt.input_channel, opt.imgH, opt.imgW]).to(device),
                                                      torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)))
    # 1.1 excluded layer @TODO(tylerz): only support for CTC now.
    excluded_layers = list(model.modules())[-11:]

    pruned_module = []
    prunable_list = [torch.nn.Conv2d]

    # 2. loop through the graph to execute the pruning:
    if opt.prune_mode in ["amount", "threshold"]:
        strategy = tp.strategy.LNStrategy(p=opt.p, mode=opt.prune_mode)
        if opt.prune_mode == "amount":
            th = opt.amount
        else:
            th = opt.threshold
        for _, m in model.named_modules():

            if isinstance(m, tuple(prunable_list)) and m not in excluded_layers and m not in pruned_module:
                pruned_idxs = strategy(m.weight, amount=th, round_to=opt.granularity)

                prune_func = tp.prune.prune_conv
                pruning_plan = dep_graph.get_pruning_plan(m, prune_func, idxs=pruned_idxs)
                if pruning_plan is not None:
                    pruning_plan.exec()
                else:
                    continue
    else:  # experimental hybrid path
        strategy = tp.strategy.CustomScoreStrategy()
        global_thresh, module2scores = tp.utils.get_global_thresh(model, prune_ratio=opt.amount)
        merged_sets = {}
        # 2.1 find the merged set:
        for _, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                prune_func = tp.prune.prune_conv
                merged_set = tp.dependency.find_merged_set(dep_graph.module_to_node[m], prune_func)
                merged_sets[m] = merged_set

        tp.utils.execute_custom_score_prune(model,
                                            global_thresh=global_thresh,
                                            module2scores=module2scores,
                                            dep_graph=dep_graph,
                                            granularity=opt.granularity,
                                            excluded_layers=excluded_layers,
                                            merged_sets=merged_sets)

    num_params_after_pruning = tp.utils.count_params(model)
    print("  Params: %s => %s" % (num_params_before_pruning, num_params_after_pruning))
    encoded_output_file = opt.output_file
    print(f"Pruned model is saved to {encoded_output_file}")
    torch.save(model, encoded_output_file)


def torch_pruner(opt, model, device):
    """Torch pruner based on pip installed torch_pruning package

    Args:
        opt (argparse): options
        model (nn.module): The model to be pruned
        device (str): The device parameter
    """
    import torch_pruning as tp

    from nvidia_tao_pytorch.pruning.dependency import TAO_DependencyGraph
    from nvidia_tao_pytorch.cv.ocrnet.model.fan import TokenMixing, ChannelProcessing
    from nvidia_tao_pytorch.cv.backbone.fan import ClassAttn
    tp.dependency.DependencyGraph.update_index_mapping = TAO_DependencyGraph.update_index_mapping

    imp = tp.importance.MagnitudeImportance(p=opt.p, group_reduction="mean")
    example_inputs = (torch.randn([1, opt.input_channel, opt.imgH, opt.imgW]).to(device),
                      torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device))
    base_flops, base_params = tp.utils.count_ops_and_params(model, example_inputs)

    ignored_layers = []
    for name, module in model.named_modules():
        if 'head' in name:
            ignored_layers.append(module)
        if 'linear_fuse' in name:
            ignored_layers.append(module)
        if "Prediction" in name:
            ignored_layers.append(module)

        if isinstance(module, (TokenMixing, ChannelProcessing, ClassAttn)):
            ignored_layers.append(module)

    pruner = tp.pruner.MagnitudePruner(model,
                                       example_inputs,
                                       importance=imp,
                                       ch_sparsity=opt.amount,
                                       channel_groups={},
                                       ignored_layers=ignored_layers,
                                       round_to=opt.granularity,
                                       root_module_types=[torch.nn.Conv2d, torch.nn.Linear])

    for g in pruner.step(interactive=True):
        g.prune()

    with torch.no_grad():
        model(example_inputs[0], example_inputs[1])

    # Save pruned model
    pruned_flops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    print("Base FLOPs: %d G, Pruned FLOPs: %d G" % (base_flops / 1e9, pruned_flops / 1e9))
    print("Base Params: %d M, Pruned Params: %d M" % (base_params / 1e6, pruned_params / 1e6))
    print(f"Pruning ratio: {pruned_params / base_params}")
    torch.save(model, opt.output_file)
    print(f'Pruned model save to {opt.output_file}')


def prune(opt):
    """Prune the the OCRNet according to option"""
    from nvidia_tao_pytorch.cv.ocrnet.utils.utils import (CTCLabelConverter,
                                                          AttnLabelConverter,
                                                          load_checkpoint)
    from nvidia_tao_pytorch.cv.ocrnet.model.model import Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    ckpt = load_checkpoint(opt.saved_model, key=opt.encryption_key, to_cpu=True)
    if not isinstance(ckpt, Model):
        model = Model(opt)
        state_dict = ckpt
        model.load_state_dict(state_dict)
    else:
        model = ckpt

    model = model.to(device)

    model.eval()
    if opt.FeatureExtraction in ["ResNet", "ResNet2X"]:
        legacy_pruner(opt, model, device)
    elif opt.FeatureExtraction == "FAN_tiny_2X":
        torch_pruner(opt, model, device)
    else:
        raise ValueError("Only supports pruning [ResNet, ResNet2X, FAN_tiny_2X] backbone")


def init_configs(experiment_spec: ExperimentConfig):
    """Pass the yaml config to argparse.Namespace"""
    if "train_gt_file" in experiment_spec["dataset"]:
        if experiment_spec["dataset"]["train_gt_file"] == "":
            experiment_spec["dataset"]["train_gt_file"] = None
    if "val_gt_file" in experiment_spec["dataset"]:
        if experiment_spec["dataset"]["val_gt_file"] == "":
            experiment_spec["dataset"]["val_gt_file"] = None
    parser = argparse.ArgumentParser()

    opt, _ = parser.parse_known_args()
    opt.encryption_key = experiment_spec.encryption_key
    opt.output_file = experiment_spec.prune.pruned_file

    # 1. Init dataset params
    dataset_config = experiment_spec.dataset
    model_config = experiment_spec.model
    opt.batch_max_length = dataset_config.max_label_length
    opt.imgH = model_config.input_height
    opt.imgW = model_config.input_width
    opt.input_channel = model_config.input_channel
    if model_config.input_channel == 3:
        opt.rgb = True
    else:
        opt.rgb = False

    # load character list:
    # Don't convert the characters to lower case
    with open(dataset_config.character_list_file, "r") as f:
        characters = "".join([ch.strip() for ch in f.readlines()])
    opt.character = characters

    # 2. Init Model params
    opt.saved_model = experiment_spec.prune.checkpoint
    if model_config.TPS:
        opt.Transformation = "TPS"
    else:
        opt.Transformation = "None"

    opt.FeatureExtraction = model_config.backbone
    opt.SequenceModeling = model_config.sequence
    opt.Prediction = model_config.prediction
    opt.num_fiducial = model_config.num_fiducial
    opt.output_channel = model_config.feature_channel
    opt.hidden_size = model_config.hidden_size

    opt.baiduCTC = False

    # 3. Init pruning params:
    prune_config = experiment_spec.prune.prune_setting
    opt.prune_mode = prune_config.mode
    if opt.prune_mode in ["amount", "experimental_hybrid"]:
        opt.amount = prune_config.amount
    elif opt.prune_mode in ["threshold"]:
        opt.threshold = prune_config.threshold
        opt.amount = opt.threshold  # For FAN backbone
    else:
        raise ValueError("Only supports prune mode in [amount, threshold, \
            experimental_hybrid]")

    opt.granularity = prune_config.granularity
    if prune_config.raw_prune_score == "L2":
        opt.p = 2
    else:
        opt.p = 1

    return opt


def run_experiment(experiment_spec):
    """run experiment."""
    opt = init_configs(experiment_spec)
    # Set default output filename if the filename
    # isn't provided over the command line.
    if opt.output_file is None:
        split_name = os.path.splitext(opt.saved_model)[0]
        opt.output_file = "pruned_{}.etlt".format(split_name)

    # Warn the user if an exported file already exists.
    assert not os.path.exists(opt.output_file), "Default output file {} already "\
        "exists".format(opt.output_file)

    prune(opt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
@monitor_status(name="OCRNet", mode="prune")
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    run_experiment(experiment_spec=cfg)


if __name__ == '__main__':
    main()
