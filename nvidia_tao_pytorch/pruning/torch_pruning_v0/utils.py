# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utils for pruning."""
from .dependency import TORCH_CONV, TORCH_BATCHNORM, TORCH_PRELU, TORCH_LINEAR
from . import prune
import torch


def count_prunable_params(module):
    """Count prunable parameters."""
    if isinstance(module, (TORCH_CONV, TORCH_LINEAR)):
        num_params = module.weight.numel()
        if module.bias is not None:
            num_params += module.bias.numel()
        return num_params
    if isinstance(module, TORCH_BATCHNORM):
        num_params = module.running_mean.numel() + module.running_var.numel()
        if module.affine:
            num_params += module.weight.numel() + module.bias.numel()
        return num_params
    if isinstance(module, TORCH_PRELU):
        if len(module.weight) == 1:
            return 0
        return module.weight.numel
    return 0


def count_prunable_channels(module):
    """Count prunable channels."""
    if isinstance(module, TORCH_CONV):
        return module.weight.shape[0]
    if isinstance(module, TORCH_LINEAR):
        return module.out_features
    if isinstance(module, TORCH_BATCHNORM):
        return module.num_features
    if isinstance(module, TORCH_PRELU):
        if len(module.weight) == 1:
            return 0
        return len(module.weight)
    return 0


def count_params(module):
    """Count parameters."""
    return sum([p.numel() for p in module.parameters()])


def ln_normalize_scores(weight, p=2):
    """Compute ln cumsum-normalized socres

    Args:
        weight (dict): weights of a torch module
        p (int, optional): 1 for l1 norm, 2 for l2 norm. Defaults to 2

    Returns:
        scores: normalized ln scores
    """
    # compute l2 norm of each output filter:
    scores = torch.norm(weight.view(len(weight), -1), p=p, dim=1)
    # sort scores in an ascending order
    sorted_scores, sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp) - 1]
    # normalize by cumulative sum
    sorted_scores /= (scores.sum() - scores_cumsum)
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape, device=scores.device)
    new_scores[sorted_idx] = sorted_scores

    return new_scores.view(scores.shape)


def get_global_thresh(model, prune_ratio, prunable_list=[torch.nn.Conv2d], p=2):
    """Get global thresh and importance socres of modules to be pruned

    Args:
        model (torch.module): Model to be pruned
        prunable_list (list of torch.module): basic module to be pruned

    Returns:
        global_thresh (float): threshold for pruning
        module2scores (dict): dict mapping module to the corresponding scores.
    """
    total_scores = []
    module2scores = {}
    total_filters = 0
    for _, m in model.named_modules():
        if isinstance(m, tuple(prunable_list)):
            scores = ln_normalize_scores(m.weight, p=p)
            total_scores.append(scores)
            module2scores[m] = scores
            total_filters += len(m.weight)
    concat_scores = torch.cat(total_scores, dim=0)
    topks, _ = torch.topk(concat_scores, int(total_filters * (1 - prune_ratio)))
    global_thresh = topks[-1]

    return global_thresh, module2scores


def execute_custom_score_prune(model,
                               global_thresh,
                               module2scores,
                               dep_graph,
                               granularity=8,
                               prunable_list=[torch.nn.Conv2d],
                               excluded_layers=[],
                               merged_sets=None):
    """Execute pruning algorithm

    Args:
        model (nn.Module): The model to be pruned
        global_thresh (float): the threshold to prune the model
        module2scores (Dict[string, list[float]]): the dict mapping module to its pruning scores
        dep_graph : DependenecyGraph of the model
        granularity (int, optional): the pruning granularity. The number of pruned channels should be divisible by the granularity. Defautlts to 8
        prunable_list (list, optional): the list of module that will be pruned. Defaults to [torch.nn.Conv2d]
        excluded_layers (list, optional): the layers will not be pruned. Defaults to []
        merged_sets (list, optional): . Defaults to None.
    """
    pruned_module = set()
    strategy = prune.strategy.CustomScoreStrategy()
    for _, m in model.named_modules():
        if isinstance(m, tuple(prunable_list)) and m not in excluded_layers and m not in pruned_module:
            if m in merged_sets:
                pruned_module.add(m)
                score_list = []
                score_list.append(module2scores[m])
                merged_set = merged_sets[m]
                for dep_m in merged_set:
                    score_list.append(module2scores[dep_m])
                    pruned_module.add(dep_m)
                scores = torch.max(torch.stack(score_list), dim=0).values
                merged_idxs = strategy(scores=scores, thresh=global_thresh, round_to=granularity)
            else:
                merged_idxs = strategy(scores=module2scores[m], thresh=global_thresh, round_to=granularity)

            if isinstance(m, TORCH_CONV):
                prune_func = prune.prune_conv
            elif isinstance(m, TORCH_LINEAR):
                prune_func = prune.prune_linear
            pruning_plan = dep_graph.get_pruning_plan(m, prune_func, idxs=merged_idxs)

            pruning_plan.exec()
