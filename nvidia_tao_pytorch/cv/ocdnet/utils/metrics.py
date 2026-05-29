#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics module."""
import numpy as np


class RunningScore(object):
    """Calcuate accuracy score."""

    def __init__(self, n_classes):
        """Initialize."""
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)

        if np.sum((label_pred[mask] < 0)) > 0:
            print(label_pred[label_pred < 0])
        hist = np.bincount(n_class * label_true[mask].astype(int) +
                           label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        """Print label_trues.dtype, label_preds.dtype."""
        for lt, lp in zip(label_trues, label_preds):
            try:
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            except Exception:
                pass

    def get_scores(self):
        """Returns accuracy score evaluation result."""
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / (hist.sum() + 0.0001)
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 0.0001)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 0.0001)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / (hist.sum() + 0.0001)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc': acc,
                'Mean Acc': acc_cls,
                'FreqW Acc': fwavacc,
                'Mean IoU': mean_iu, }, cls_iu

    def reset(self):
        """Reset confusion matrix."""
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
