# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eval for Re-Identification Module."""

import torch
from nvidia_tao_pytorch.cv.re_identification.utils.common_utils import plot_evaluation_results


def eval_func(cfg, distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths, prepare_for_training):
    """Evaluates person re-identification (ReID) performance using Market1501 metric.

    For each query identity, it discards gallery images from the same camera view.
    After that, it calculates the cumulative matching characteristics (CMC) curve and mean Average Precision (mAP).
    If the program is not in training mode and if plotting is enabled, it also plots the evaluation results.

    Args:
        cfg (DictConfig): Configuration file.
        distmat (numpy.ndarray): Pairwise distance matrix between query and gallery features.
        q_pids (numpy.ndarray): Array containing query person IDs.
        g_pids (numpy.ndarray): Array containing gallery person IDs.
        q_camids (numpy.ndarray): Array containing query camera IDs.
        g_camids (numpy.ndarray): Array containing gallery camera IDs.
        q_img_paths (list of str): List containing query image paths.
        g_img_paths (list of str): List containing gallery image paths.
        prepare_for_training (bool): Flag indicating whether the system is in training mode.

    Returns:
        list: The Cumulative Matching Characteristics (CMC) rank list.
        float: The mean Average Precision (mAP) score.
    """
    num_q, num_g = distmat.shape
    max_rank = cfg["re_ranking"]["max_rank"]
    if num_g < max_rank:
        max_rank = num_g
    _, indices = torch.sort(distmat, dim=1)

    g_pids = torch.tensor(g_pids, device=indices.device)
    q_pids = torch.tensor(q_pids, device=indices.device)
    g_camids = torch.tensor(g_camids, device=indices.device)
    q_camids = torch.tensor(q_camids, device=indices.device)
    matches = (g_pids[indices] == q_pids.unsqueeze(1)).int()

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    query_maps = []

    for q_idx in range(num_q):

        query_map = []

        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_img_path = q_img_paths[q_idx]

        # build the first column of the sampled matches image output
        query_map.append([q_img_path, False])

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]

        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove

        # build the rest of the columns of the sampled matches image output
        if not prepare_for_training and cfg["evaluate"]["output_sampled_matches_plot"]:
            res_list = list(map(g_img_paths.__getitem__, order))
            for g_img_path, value in zip(res_list[:max_rank], matches[q_idx][:max_rank]):
                query_map.append([g_img_path, value])
            query_maps.append(query_map)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not torch.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum(0)
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum().item()
        tmp_cmc = orig_cmc.cumsum(0)
        y = torch.arange(1, tmp_cmc.shape[0] + 1, device=tmp_cmc.device).float()
        tmp_cmc = tmp_cmc / y
        tmp_cmc = tmp_cmc * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if not prepare_for_training and cfg["evaluate"]["output_sampled_matches_plot"]:
        plot_evaluation_results(cfg["re_ranking"]["num_query"], query_maps, max_rank, cfg["evaluate"]["output_sampled_matches_plot"])

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery."

    all_cmc = torch.stack(all_cmc).float()
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = torch.tensor(all_AP).mean().item()

    return all_cmc, mAP
