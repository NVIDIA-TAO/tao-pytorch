# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-Identification Metrics."""

import numpy as np
import torch

from nvidia_tao_pytorch.cv.re_identification.utils.eval_reid import eval_func
from nvidia_tao_pytorch.cv.re_identification.utils.re_ranking import rerank_gpu


def euclidean_distance(qf, gf):
    """Compute the euclidean distance between two given matrices.

    Args:
        qf (torch.Tensor): Matrix A of size (m x n)
        gf (torch.Tensor): Matrix B of size (p x q)

    Returns:
        numpy.ndarray: A numpy array of euclidean distance, of size (m x p).
    """
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    """Compute the cosine similarity between two given matrices.

    Args:
        qf (torch.Tensor): Matrix A of size (m x n)
        gf (torch.Tensor): Matrix B of size (p x q)

    Returns:
        numpy.ndarray: A numpy array of cosine similarity, of size (m x p).
    """
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


class R1_mAP():
    """Class to compute the rank-1 mean Average Precision (mAP) for re-identification.

    This class provides the functions to compute the rank-1 mean Average Precision,
    a common evaluation metric in person re-identification tasks.
    """

    def __init__(self, num_query, cfg, prepare_for_training, feat_norm=True):
        """Initialize the R1_mAP class with the given configuration.

        Args:
            num_query (int): The number of query images.
            cfg (dict): Configuration dictionary containing re_ranking parameters.
            prepare_for_training (bool): Specify whether the data is prepared for training.
            feat_norm (bool, optional): Whether to normalize the feature vectors. Defaults to True.
        """
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = cfg["re_ranking"]["max_rank"]
        self.feat_norm = feat_norm
        self.feats = []
        self.pids = []
        self.camids = []
        self.img_paths = []
        self.cfg = cfg
        self.prepare_for_training = prepare_for_training

    def reset(self):
        """Reset the stored feature vectors, person IDs, camera IDs, and image paths."""
        self.feats = []
        self.pids = []
        self.camids = []
        self.img_paths = []

    def update(self, feat, pid, camid, img_path):
        """Update the stored feature vectors, person IDs, camera IDs, and image paths with new data.

        Args:
            feat (torch.Tensor): The feature vectors.
            pid (list): The person IDs.
            camid (list): The camera IDs.
            img_path (list): The image paths.
        """
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.img_paths.extend(img_path)

    def compute(self):
        """Compute the rank-1 mean Average Precision (mAP) and CMC rank list.

        Returns:
            list: The Cumulative Matching Characteristics (CMC) rank list.
            float: The mean Average Precision (mAP) score.
        """
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test features are normalized.")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_img_paths = self.img_paths[:self.num_query]

        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_img_paths = self.img_paths[self.num_query:]

        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
        cmc, mAP = eval_func(self.cfg, distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths, self.prepare_for_training)

        return cmc, mAP


class R1_mAP_reranking():
    """Class to compute the rank-1 mean Average Precision (mAP) with re-ranking for re-identification.

    This class provides the functions to compute the rank-1 mean Average Precision with re-ranking,
    a common evaluation metric in person re-identification tasks.
    """

    def __init__(self, num_query, cfg, prepare_for_training, feat_norm=True):
        """Initialize the R1_mAP_reranking class with the given configuration.

        Args:
            num_query (int): The number of query images.
            cfg (dict): Configuration dictionary containing re_ranking parameters.
            prepare_for_training (bool): Specify whether the data is prepared for training.
            feat_norm (bool, optional): Whether to normalize the feature vectors. Defaults to True.
        """
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = cfg["re_ranking"]["max_rank"]
        self.feat_norm = feat_norm
        self.feats = []
        self.pids = []
        self.camids = []
        self.img_paths = []
        self.cfg = cfg
        self.prepare_for_training = prepare_for_training

    def reset(self):
        """Reset the stored feature vectors, person IDs, camera IDs, and image paths."""
        self.feats = []
        self.pids = []
        self.camids = []
        self.img_paths = []

    def update(self, feat, pid, camid, img_path):
        """Update the stored feature vectors, person IDs, camera IDs, and image paths with new data.

        Args:
            feat (torch.Tensor): The feature vectors.
            pid (list): The person IDs.
            camid (list): The camera IDs.
            img_path (list): The image paths.
        """
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.img_paths.extend(img_path)

    def compute(self):
        """Compute the rank-1 mean Average Precision (mAP) and CMC rank list using re-ranking.

        This method first applies re-ranking on the feature vectors, then computes the mAP and CMC rank list.

        Returns:
            list: The Cumulative Matching Characteristics (CMC) rank list.
            float: The mean Average Precision (mAP) score.
        """
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test features are normalized.")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_img_paths = self.img_paths[:self.num_query]
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_img_paths = self.img_paths[self.num_query:]

        print("The distance matrix is computed using euclidean distance. It is then processed by re-ranking.")
        distmat = rerank_gpu(qf, gf, k1=self.cfg["re_ranking"]["k1"], k2=self.cfg["re_ranking"]["k2"], lambda_value=self.cfg["re_ranking"]["lambda_value"])
        cmc, mAP = eval_func(self.cfg, distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths, self.prepare_for_training)

        return cmc, mAP
