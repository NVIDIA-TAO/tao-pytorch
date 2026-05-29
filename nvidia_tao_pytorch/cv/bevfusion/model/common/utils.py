# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BEVFusion common utility functions"""

import torch
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from mmdet.models.task_modules import AssignResult, BaseAssigner, BaseBBoxCoder
from mmengine.structures import InstanceData
from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class TAO3DBBoxCoder(BaseBBoxCoder):
    """ TAO3DBBoxCoder Class"""

    def __init__(
        self,
        pc_range,
        out_size_factor,
        voxel_size,
        post_center_range=None,
        score_threshold=None,
        code_size=12,
    ):
        """
        TAO3D BBox Coder init
        Args:
            pc_range (List[float]): The point cloud range list
            out_size_factor (int): Out size factor
            voxel_size (List[float]): The voxel size list
            post_center_range (List[float]): Post center range for point cloud
            score_threshold (float): Score Threshold for bboxcoder
            code_size (int): code size for bboxcoder. Defaults to 12.
        """
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.code_size = code_size

    def encode(self, dst_boxes):
        """ encoding function """
        targets = torch.zeros([dst_boxes.shape[0],
                               self.code_size]).to(dst_boxes.device)
        targets[:, 0] = (dst_boxes[:, 0] - self.pc_range[0]) / (
            self.out_size_factor * self.voxel_size[0])
        targets[:, 1] = (dst_boxes[:, 1] - self.pc_range[1]) / (
            self.out_size_factor * self.voxel_size[1])
        # bottom center to gravity center
        targets[:, 2] = dst_boxes[:, 2] + dst_boxes[:, 5] * 0.5

        targets[:, 3] = dst_boxes[:, 3].log()
        targets[:, 4] = dst_boxes[:, 4].log()
        targets[:, 5] = dst_boxes[:, 5].log()
        assert dst_boxes.shape[1] == 9
        assert self.code_size == 12

        targets[:, 6] = torch.sin(dst_boxes[:, 6])  # rx-s
        targets[:, 7] = torch.cos(dst_boxes[:, 6])  # rx-c
        targets[:, 8] = torch.sin(dst_boxes[:, 7])  # ry-s
        targets[:, 9] = torch.cos(dst_boxes[:, 7])  # ry-c
        targets[:, 10] = torch.sin(dst_boxes[:, 8])  # rz-s
        targets[:, 11] = torch.cos(dst_boxes[:, 8])  # rz-c

        return targets

    def decode(self, heatmap, rot, dim, center, height, vel, filter=False):
        """Decode bboxes.
        Args:
            heat (torch.Tensor): Heatmap with the shape of
                [B, num_cls, num_proposals].
            rot (torch.Tensor): Rotation with the shape of
                [B, 2, num_proposals] for public, [B, 6, num_proposals] for tao
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 3, num_proposals].
            center (torch.Tensor): bev center of the boxes with the shape of
                [B, 2, num_proposals]. (in feature map metric)
            height (torch.Tensor): height of the boxes with the shape of
                [B, 2, num_proposals]. (in real world metric)
            vel (torch.Tensor): Velocity with the shape of
                [B, 2, num_proposals].
            filter: if False, return all box without checking score and
                center_range
        Returns:
            list[dict]: Decoded boxes.
        """
        # noqa pylint: disable=W0622
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        # change size to real world metric
        center[:,
               0, :] = center[:,
                              0, :] * self.out_size_factor * self.voxel_size[
                                  0] + self.pc_range[0]
        center[:,
               1, :] = center[:,
                              1, :] * self.out_size_factor * self.voxel_size[
                                  1] + self.pc_range[1]
        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        height = height - dim[:,
                              2:3, :] * 0.5  # gravity center to bottom center (z-coordinate)

        rots, rotc = rot[:, 0:1, :].squeeze(1), rot[:, 1:2, :].squeeze(1)
        rot_x = torch.atan2(rots, rotc)
        rots, rotc = rot[:, 2:3, :].squeeze(1), rot[:, 3:4, :].squeeze(1)
        rot_y = torch.atan2(rots, rotc)
        rots, rotc = rot[:, 4:5, :].squeeze(1), rot[:, 5:6, :].squeeze(1)
        rot_z = torch.atan2(rots, rotc)
        rot_xyz = torch.stack([rot_x, rot_y, rot_z], dim=1)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot_xyz],
                                        dim=1).permute(0, 2, 1)
        else:
            raise NotImplementedError(
                'vel is not supported to TAO3D dataset')

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heatmap.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(heatmap.shape[0]):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts


@TASK_UTILS.register_module()
class BBoxBEVL1Cost(object):
    """ BBoxBEVL1Cost Class """

    def __init__(self, weight):
        """ BBoxBEVL1Cost Init """
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes, train_cfg):
        """ BBoxBEVL1Cost Call Funciton """
        pc_start = bboxes.new(train_cfg['point_cloud_range'][0:2])
        pc_range = bboxes.new(
            train_cfg['point_cloud_range'][3:5]) - bboxes.new(
                train_cfg['point_cloud_range'][0:2])
        # normalize the box center to [0, 1]
        normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
        normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
        reg_cost = torch.cdist(
            normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
        return reg_cost * self.weight


@TASK_UTILS.register_module()
class IoU3DCost(object):
    """ IoU3DCost Class """

    def __init__(self, weight):
        """ IoU3DCost Init """
        self.weight = weight

    def __call__(self, iou):
        """ IoU3DCost Call Funciton """
        iou_cost = -iou
        return iou_cost * self.weight


@TASK_UTILS.register_module()
class HeuristicAssigner3D(BaseAssigner):
    """ HeuristicAssigner3D Class """

    def __init__(self,
                 dist_thre=100,
                 iou_calculator={'type': 'BboxOverlaps3D'}):
        """ HeuristicAssigner3D Init """
        self.dist_thre = dist_thre  # distance in meter
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self,
               bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               query_labels=None):
        """ HeuristicAssigner3D assign function """
        dist_thre = self.dist_thre
        num_gts, num_bboxes = len(gt_bboxes), len(bboxes)

        bev_dist = torch.norm(
            bboxes[:, 0:2][None, :, :] - gt_bboxes[:, 0:2][:, None, :],
            dim=-1)  # [num_gts, num_bboxes]
        if query_labels is not None:
            # only match the gt box and query with same category
            not_same_class = (query_labels[None] != gt_labels[:, None])
            bev_dist += not_same_class * dist_thre

        # for each gt box, assign it to the nearest pred box
        _, nearest_indices = bev_dist.min(1)  # [num_gts]
        assigned_gt_inds = torch.ones([
            num_bboxes,
        ]).to(bboxes) * 0
        assigned_gt_vals = torch.ones([
            num_bboxes,
        ]).to(bboxes) * 10000
        assigned_gt_labels = torch.ones([
            num_bboxes,
        ]).to(bboxes) * -1
        for idx_gts in range(num_gts):
            # for idx_pred in torch.where(bev_dist[idx_gts] < dist_thre)[0]:
            # # each gt match to all the pred box within some radius
            idx_pred = nearest_indices[
                idx_gts]  # each gt only match to the nearest pred box
            if bev_dist[idx_gts, idx_pred] <= dist_thre:
                # if this pred box is assigned, then compare
                if bev_dist[idx_gts, idx_pred] < assigned_gt_vals[idx_pred]:
                    assigned_gt_vals[idx_pred] = bev_dist[idx_gts, idx_pred]
                    # for AssignResult, 0 is negative, -1 is ignore, 1-based
                    # indices are positive
                    assigned_gt_inds[idx_pred] = idx_gts + 1
                    assigned_gt_labels[idx_pred] = gt_labels[idx_gts]

        max_overlaps = torch.zeros([
            num_bboxes,
        ]).to(bboxes)
        matched_indices = torch.where(assigned_gt_inds > 0)
        matched_iou = self.iou_calculator(
            gt_bboxes[assigned_gt_inds[matched_indices].long() - 1],
            bboxes[matched_indices]).diag()
        max_overlaps[matched_indices] = matched_iou

        return AssignResult(
            num_gts,
            assigned_gt_inds.long(),
            max_overlaps,
            labels=assigned_gt_labels)


@TASK_UTILS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """ HungarianAssigner3D Class """

    def __init__(self,
                 cls_cost={'type': 'ClassificationCost', 'weight': 1.0},
                 reg_cost={'type': 'BBoxBEVL1Cost', 'weight': 1.0},
                 iou_cost={'type': 'IoU3DCost', 'weight': 1.0},
                 iou_calculator={'type': 'BboxOverlaps3D'}):
        """ HungarianAssigner3D init function """
        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.reg_cost = TASK_UTILS.build(reg_cost)
        self.iou_cost = TASK_UTILS.build(iou_cost)
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, train_cfg):
        """ HungarianAssigner3D assign function """
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        # 1. assign -1 by default
        assigned_gt_inds = bboxes.new_full((num_bboxes, ),
                                           -1,
                                           dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes, ), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # Hard code here to be compatible with the interface of
        # `ClassificationCost` in mmdet.
        gt_instances, pred_instances = InstanceData(
            labels=gt_labels), InstanceData(scores=cls_pred[0].T)
        cls_cost = self.cls_cost(pred_instances, gt_instances)
        reg_cost = self.reg_cost(bboxes, gt_bboxes, train_cfg)

        iou = self.iou_calculator(bboxes, gt_bboxes)
        iou_cost = self.iou_cost(iou)

        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        max_overlaps = torch.zeros_like(iou.max(1).values)
        max_overlaps[matched_row_inds] = iou[matched_row_inds,
                                             matched_col_inds]
        # max_overlaps = iou.max(1).values
        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
