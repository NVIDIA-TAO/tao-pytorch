# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""SegFormer metric utils."""

import numpy as np
import torch


class MeanIoUMeter:
    """Computes and stores the average and current value."""

    def __init__(self, n_class: int):
        """init."""
        self.initialized = False
        self.area_intersect = None
        self.area_union = None
        self.area_pred_label = None
        self.area_label = None
        self.count = None
        self.n_class = n_class

    def initialize(
        self, area_intersect: np.ndarray, area_union: np.ndarray, area_pred_label: np.ndarray, area_label: np.ndarray
    ):
        """Initialize counter variables for metric calculation."""
        self.area_intersect = area_intersect
        self.area_union = area_union
        self.area_pred_label = area_pred_label
        self.area_label = area_label
        self.count = 1
        self.initialized = True

    def add(
        self, area_intersect: np.ndarray, area_union: np.ndarray, area_pred_label: np.ndarray, area_label: np.ndarray
    ):
        """Update MeanIoUMeter with new value and weight."""
        self.area_intersect += area_intersect
        self.area_union += area_union
        self.area_pred_label += area_pred_label
        self.area_label += area_label

    def update(
        self, area_intersect: np.ndarray, area_union: np.ndarray, area_pred_label: np.ndarray, area_label: np.ndarray
    ):
        """Update MeanIoUMeter with new value and weight."""
        if not self.initialized:
            self.initialize(area_intersect, area_union, area_pred_label, area_label)
        else:
            self.add(area_intersect, area_union, area_pred_label, area_label)

    def clear(self):
        """Clear the initialized status."""
        self.initialized = False

    @staticmethod
    def intersect_and_union(pred_label: torch.Tensor, label: torch.Tensor, num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_pred_label = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_label = torch.histc(label.float(), bins=(num_classes), min=0, max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return (
            area_intersect.numpy(),
            area_union.numpy(),
            area_pred_label.numpy(),
            area_label.numpy(),
        )

    @staticmethod
    def total_area_to_metrics(
        total_area_intersect: np.ndarray,
        total_area_union: np.ndarray,
        total_area_pred_label: np.ndarray,
        total_area_label: np.ndarray,
        n_class: int,
    ):
        """Calculate evaluation metrics.

        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
        """
        all_acc = total_area_intersect.sum() / total_area_label.sum()
        iou = total_area_intersect / total_area_union
        precision = total_area_intersect / total_area_pred_label
        recall = total_area_intersect / total_area_label
        f1 = 2 * recall * precision / (recall + precision)

        all_acc = all_acc.astype(float)
        iou = iou.astype(float)
        precision = precision.astype(float)
        recall = recall.astype(float)
        f1 = f1.astype(float)

        cls_iou = {}
        cls_precision = {}
        cls_recall = {}
        cls_f1 = {}
        for i in range(n_class):
            cls_iou["iou_" + str(i)] = np.nan_to_num(iou[i])
            cls_precision["precision_" + str(i)] = np.nan_to_num(precision[i])
            cls_recall["recall_" + str(i)] = np.nan_to_num(recall[i])
            cls_f1["f1_" + str(i)] = np.nan_to_num(f1[i])

        score_dict = {"acc": all_acc, "miou": np.nanmean(iou), "mf1": np.nanmean(f1)}
        score_dict.update(cls_iou)
        score_dict.update(cls_f1)
        score_dict.update(cls_precision)
        score_dict.update(cls_recall)

        mean_score_dict = {"mprecision": np.nanmean(precision), "mrecall": np.nanmean(recall)}
        return score_dict, mean_score_dict

    def update_cm(self, pr: torch.Tensor, gt: torch.Tensor):
        """Get the current confusion matrix, calculate the current F1 score, and update the confusion matrix"""
        gt = torch.squeeze(gt, dim=1) if gt.dim() == 4 else gt
        for lt, lp in zip(gt, pr):
            area_intersect, area_union, area_pred_label, area_label = self.intersect_and_union(
                pred_label=lp, label=lt, num_classes=self.n_class, ignore_index=255
            )
            self.update(area_intersect, area_union, area_pred_label, area_label)

    def get_scores(self):
        """get scores from confusion matrix"""
        return self.total_area_to_metrics(
            self.area_intersect, self.area_union, self.area_pred_label, self.area_label, n_class=self.n_class
        )
