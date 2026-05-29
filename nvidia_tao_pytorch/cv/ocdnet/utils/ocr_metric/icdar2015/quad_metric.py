#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""quad_metric module."""
import numpy as np

from .detection.iou import DetectionIoUEvaluator


def get_metric(config):
    """get metric."""
    try:
        if 'args' not in config:
            args = {}
        else:
            args = config['args']
        if isinstance(args, dict):
            cls = globals()[config['type']](**args)
        else:
            cls = globals()[config['type']](args)
        return cls
    except Exception:
        return None


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        """Initialize."""
        self.reset()

    def reset(self):
        """reset."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self


class QuadMetric():
    """QuadMetric class."""

    def __init__(self, is_output_polygon=False):
        """Initialize."""
        self.is_output_polygon = is_output_polygon
        self.evaluator = DetectionIoUEvaluator(is_output_polygon=is_output_polygon)

    def measure(self, batch, output, box_thresh=0.6):
        """Measure the quad metric

        Args:
            batch (dict): Produced by dataloaders. It is a dict of image, polygons and ignore_tags.
                          The image is a tensor with shape (N, C, H, W). The polygons is a tensor of
                          shape (N, K, 4, 2). The ignore_tags is a tensor of shape (N, K), indicates
                          whether a region is ignorable or not. The shape is the original shape of images.
                          The filename is the original filenames of images.
            output: The prediction polygons and scores.
        """
        results = []
        gt_polyons_batch = batch['text_polys']
        ignore_tags_batch = batch['ignore_tags']
        pred_polygons_batch = np.array(output[0], dtype=object)
        pred_scores_batch = np.array(output[1], dtype=object)
        for polygons, pred_polygons, pred_scores, ignore_tags in zip(gt_polyons_batch, pred_polygons_batch, pred_scores_batch, ignore_tags_batch):
            gt = [dict(points=np.int64(polygons[i]), ignore=ignore_tags[i]) for i in range(len(polygons))]
            if self.is_output_polygon:
                pred = [dict(points=pred_polygons[i]) for i in range(len(pred_polygons))]
            else:
                pred = []
                for i in range(pred_polygons.shape[0]):
                    if pred_scores[i] >= box_thresh:
                        pred.append(dict(points=pred_polygons[i, :, :].astype(int)))
                res = self.evaluator.evaluate_image(gt, pred)
            results.append(res)
        return results

    def validate_measure(self, batch, output, box_thresh=0.6):
        """validate measure."""
        return self.measure(batch, output, box_thresh)

    def evaluate_measure(self, batch, output):
        """evaluate measure."""
        return self.measure(batch, output), np.linspace(0, batch['image'].shape[0]).tolist()

    def gather_measure(self, raw_metrics):
        """gather measure."""
        raw_metrics = [image_metrics
                       for batch_metrics in raw_metrics
                       for image_metrics in batch_metrics]

        result = self.evaluator.combine_results(raw_metrics)

        precision = AverageMeter()
        recall = AverageMeter()
        hmean = AverageMeter()

        precision.update(result['precision'], n=len(raw_metrics))
        recall.update(result['recall'], n=len(raw_metrics))
        hmean_score = 2 * precision.val * recall.val / (precision.val + recall.val + 1e-8)
        hmean.update(hmean_score)

        return {
            'precision': precision,
            'recall': recall,
            'hmean': hmean
        }
