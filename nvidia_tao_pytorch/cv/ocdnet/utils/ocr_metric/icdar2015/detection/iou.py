#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection IOU module."""
# pylint: disable=W0612,R1705
import numpy as np
from shapely.geometry import Polygon
import cv2


def iou_rotate(box_a, box_b, method='union'):
    """iou rotate."""
    rect_a = cv2.minAreaRect(box_a)
    rect_b = cv2.minAreaRect(box_b)
    r1 = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if r1[0] == 0:
        return 0
    else:
        inter_area = cv2.contourArea(r1[1])
        area_a = cv2.contourArea(box_a)
        area_b = cv2.contourArea(box_b)
        union_area = area_a + area_b - inter_area
        if union_area == 0 or inter_area == 0:
            return 0
        if method == 'union':
            iou = inter_area / union_area
        elif method == 'intersection':
            iou = inter_area / min(area_a, area_b)
        else:
            raise NotImplementedError
        return iou


class DetectionIoUEvaluator(object):
    """Define the evaluator.

    The evaluator will
        - Iterate through the ground truth, save the ones which are valid and not_care.
        - Iterate through the predicted polygon, save the ones which are valid and not_care.
        - Calculate the number of valid ground truth and predicted polygons
        - Calculate the number when ground truth and predicted polygons are matched
        - Calculate recall, precision and hmean
    """

    def __init__(self, is_output_polygon=False, iou_constraint=0.5, area_precision_constraint=0.5):
        """Initialize."""
        self.is_output_polygon = is_output_polygon
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):
        """evaluate image."""
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def compute_ap(conf_list, match_list, num_gt_care):
            correct = 0
            AP = 0
            if len(conf_list) > 0:
                conf_list = np.array(conf_list)
                match_list = np.array(match_list)
                sorted_ind = np.argsort(-conf_list)
                conf_list = conf_list[sorted_ind]
                match_list = match_list[sorted_ind]
                for n in range(len(conf_list)):
                    match = match_list[n]
                    if match:
                        correct += 1
                        AP += float(correct) / (n + 1)

                if num_gt_care > 0:
                    AP /= num_gt_care

            return AP

        per_sample_metrics = {}

        matched_sum = 0

        num_globalcare_gt = 0
        num_globalcare_det = 0

        recall = 0
        precision = 0
        hmean = 0

        det_matched = 0

        iou_mat = np.empty([1, 1])

        gt_pols = []
        det_pols = []

        gt_pol_points = []
        det_pol_points = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gt_dontcare_pols_mum = []
        # Array of Detected Polygons' matched with a don't Care GT
        det_dontcare_pols_num = []

        pairs = []
        det_matched_nums = []

        evaluation_log = ""

        # Iterate through the ground truth
        for n in range(len(gt)):
            points = gt[n]['points']
            dont_care = gt[n]['ignore']

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            # Save the valid polygon
            gt_pol = points
            gt_pols.append(gt_pol)
            gt_pol_points.append(points)
            # Save the dont_care polygon
            if dont_care:
                gt_dontcare_pols_mum.append(len(gt_pols) - 1)

        evaluation_log += "GT polygons: " + str(len(gt_pols)) + (" (" + str(len(
            gt_dontcare_pols_mum)) + " don't care)\n" if len(gt_dontcare_pols_mum) > 0 else "\n")

        # Iterate through the predicted polygons
        for n in range(len(pred)):
            points = pred[n]['points']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            # Save the valid polygon
            det_pol = points
            det_pols.append(det_pol)
            det_pol_points.append(points)
            if len(gt_dontcare_pols_mum) > 0:
                # Iterate through the dont_care polygons, calculate the intersection against predicted polygon
                for dontcare_pol in gt_dontcare_pols_mum:
                    # Find the dont_care polygon
                    dontcare_pol = gt_pols[dontcare_pol]
                    # Calculate the intersection between dont_care polygon and predicted polygon
                    intersected_area = get_intersection(dontcare_pol, det_pol)
                    # Calculate the area of predicted polygon
                    pd_dimensions = Polygon(det_pol).area
                    # Calcuate precision
                    precision = 0 if pd_dimensions == 0 else intersected_area / pd_dimensions
                    # Save the polygon number if precision is higher than the constraint
                    if (precision > self.area_precision_constraint):
                        det_dontcare_pols_num.append(len(det_pols) - 1)
                        break

        evaluation_log += "DET polygons: " + str(len(det_pols)) + (" (" + str(len(
            det_dontcare_pols_num)) + " don't care)\n" if len(det_dontcare_pols_num) > 0 else "\n")

        # If both groud truth polygon and predicted polygon are valid and available
        if len(gt_pols) > 0 and len(det_pols) > 0:
            # Calculate IoU and precision matrixs
            output_shape = [len(gt_pols), len(det_pols)]
            iou_mat = np.empty(output_shape)
            gt_rect_mat = np.zeros(len(gt_pols), np.int8)
            det_rect_mat = np.zeros(len(det_pols), np.int8)
            # Iterate through the ground truth and the predicted polygons, then calculate the IOU
            for gt_num in range(len(gt_pols)):
                for det_num in range(len(det_pols)):
                    pG = gt_pols[gt_num]
                    pD = det_pols[det_num]
                    iou_mat[gt_num, det_num] = get_intersection_over_union(pD, pG)

            for gt_num in range(len(gt_pols)):
                for det_num in range(len(det_pols)):
                    if gt_rect_mat[gt_num] == 0 and det_rect_mat[det_num] == 0 and gt_num not in gt_dontcare_pols_mum and det_num not in det_dontcare_pols_num:
                        # Check if ground truth and predicted polygons are matched, and save the number
                        if iou_mat[gt_num, det_num] > self.iou_constraint:
                            gt_rect_mat[gt_num] = 1
                            det_rect_mat[det_num] = 1
                            det_matched += 1
                            pairs.append({'gt': gt_num, 'det': det_num})
                            det_matched_nums.append(det_num)
                            evaluation_log += "Match GT #" + \
                                str(gt_num) + " with Det #" + str(det_num) + "\n"
        # Calcuate number of valid ground truth and predicted polygons
        num_gt_care = (len(gt_pols) - len(gt_dontcare_pols_mum))
        num_det_care = (len(det_pols) - len(det_dontcare_pols_num))
        # Calcuate recall, precision and hmean
        if num_gt_care == 0:
            recall = float(1)
            precision = float(0) if num_det_care > 0 else float(1)
        else:
            recall = float(det_matched) / num_gt_care
            precision = 0 if num_det_care == 0 else float(
                det_matched) / num_det_care

        hmean = 0 if (precision + recall) == 0 else 2.0 * \
            precision * recall / (precision + recall)

        matched_sum += det_matched
        num_globalcare_gt += num_gt_care
        num_globalcare_det += num_det_care

        per_sample_metrics = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iou_mat': [] if len(det_pols) > 100 else iou_mat.tolist(),
            'gt_pol_points': gt_pol_points,
            'det_pol_points': det_pol_points,
            'gt_care': num_gt_care,
            'det_care': num_det_care,
            'gt_dontcare': gt_dontcare_pols_mum,
            'det_dontcare': det_dontcare_pols_num,
            'det_matched': det_matched,
            'evaluation_log': evaluation_log
        }

        return per_sample_metrics

    def combine_results(self, results):
        """combine results."""
        num_globalcare_gt = 0
        num_globalcare_det = 0
        matched_sum = 0
        for result in results:
            num_globalcare_gt += result['gt_care']
            num_globalcare_det += result['det_care']
            matched_sum += result['det_matched']

        method_recall = 0 if num_globalcare_gt == 0 else float(
            matched_sum) / num_globalcare_gt
        method_precision = 0 if num_globalcare_det == 0 else float(
            matched_sum) / num_globalcare_det
        method_hmean = 0 if method_recall + method_precision == 0 else 2 * \
            method_recall * method_precision / (method_recall + method_precision)

        method_metrics = {'precision': method_precision,
                          'recall': method_recall, 'hmean': method_hmean}

        return method_metrics
