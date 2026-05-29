# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" CenterPose Post processing for inference. """

import torch
from torch import nn
import numpy as np
from nvidia_tao_pytorch.cv.centerpose.model.post_processing_utils import _nms, _topk, _transpose_and_gather_feat, _topk_channel, transform_preds, soft_nms, pnp_shell


class HeatmapDecoder(nn.Module):
    """This module converts the model's output into the format expected by post-processing."""

    def __init__(self, num_select=100):
        """PostProcess constructor.

        Args:
            num_select (int): top K predictions to select from
        """
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs):
        """Decode the heatmap output into the expected formats"""
        out = outputs[-1]
        hm, wh, kps, reg, hm_hp, hp_offset, scale = out['hm'], out['wh'], out['hps'], out['reg'], out['hm_hp'], out['hp_offset'], out['scale']

        heat = hm.sigmoid_()
        hm_hp = hm_hp.sigmoid_()
        obj_scale = scale

        K = self.num_select
        batch, _, _, _ = hm.size()
        num_joints = kps.shape[1] // 2

        # Perform NMS on heatmaps
        heat = _nms(heat)
        scores, inds, clses, ys, xs = _topk(heat, K=K)

        kps = _transpose_and_gather_feat(kps, inds)  # 100*34
        kps = kps.view(batch, K, num_joints * 2)  # joint offset from the centroid loc
        kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)  # + centroid loc
        kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
        if reg is not None:
            reg = _transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5

        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

        # Decode the Height, Width heatmap and bbox.
        if wh is not None:
            wh = _transpose_and_gather_feat(wh, inds)
            # The number of "2" means height and width.
            wh = wh.view(batch, K, 2)

            bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                                ys - wh[..., 1:2] / 2,
                                xs + wh[..., 0:1] / 2,
                                ys + wh[..., 1:2] / 2], dim=2)
            if hm_hp is not None:
                hm_hp = _nms(hm_hp)
                thresh = 0.1
                kps = kps.view(batch, K, num_joints, 2).permute(
                    0, 2, 1, 3).contiguous()  # b x J x K x 2

                mask_temp = torch.ones((batch, num_joints, K, 1)).to(kps.device)
                mask_temp = (mask_temp > 0).float().expand(batch, num_joints, K, 2)
                kps_displacement_mean = mask_temp * kps
                kps_displacement_mean = kps_displacement_mean.permute(0, 2, 1, 3).contiguous().view(
                    batch, K, num_joints * 2)

                # Continue normal processing
                reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)  # b x J x K x K x 2
                hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
                if hp_offset is not None:
                    hp_offset = _transpose_and_gather_feat(
                        hp_offset, hm_inds.view(batch, -1))
                    hp_offset = hp_offset.view(batch, num_joints, K, 2)
                    hm_xs = hm_xs + hp_offset[:, :, :, 0]
                    hm_ys = hm_ys + hp_offset[:, :, :, 1]
                else:
                    hm_xs = hm_xs + 0.5
                    hm_ys = hm_ys + 0.5

                # Filter by thresh
                mask = (hm_score > thresh).float()
                hm_score = (1 - mask) * -1 + mask * hm_score  # -1 or hm_score
                hm_ys = (1 - mask) * (-10000) + mask * hm_ys  # -10000 or hm_ys
                hm_xs = (1 - mask) * (-10000) + mask * hm_xs

                # Find the nearest keypoint in the corresponding heatmap for each displacement representation
                hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
                    2).expand(batch, num_joints, K, K, 2)
                dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)  # b x J x K x K
                min_dist, min_ind = dist.min(dim=3)  # b x J x K
                hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
                min_dist = min_dist.unsqueeze(-1)
                min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
                    batch, num_joints, K, 1, 2)
                hm_kps = hm_kps.gather(3, min_ind)
                hm_kps = hm_kps.view(batch, num_joints, K, 2)

                left = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
                top = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
                right = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
                bottom = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
                mask = (hm_kps[..., 0:1] < left).long() + (hm_kps[..., 0:1] > right).long() + \
                    (hm_kps[..., 1:2] < top).long() + (hm_kps[..., 1:2] > bottom).long() + \
                    (hm_score < thresh).long() + (min_dist > (torch.max(bottom - top, right - left) * 0.3)).long()
                mask = (mask > 0).float().expand(batch, num_joints, K, 2)

                kps = (1 - mask) * hm_kps + mask * kps

                kps = kps.permute(0, 2, 1, 3).contiguous().view(
                    batch, K, num_joints * 2)

                # Have to satisfy all the requirements: within an enlarged 2D bbox/
                # hm_score high enough/center_score high enough/not far away from the corresponding representation
                scores_copy = scores.unsqueeze(1).expand(batch, num_joints, K, 2)

                mask_2 = (hm_kps[..., 0:1] > 0.8 * left) + (hm_kps[..., 0:1] < 1.2 * right) + \
                    (hm_kps[..., 1:2] > 0.8 * top) + (hm_kps[..., 1:2] < 1.2 * bottom) + \
                    (hm_score > thresh) + (min_dist < (torch.max(bottom - top, right - left) * 0.5)) + \
                    (scores_copy > thresh)

                mask_2 = (mask_2 == 7).float().expand(batch, num_joints, K, 2)
                hm_kps_filtered = mask_2 * hm_kps + (1 - mask_2) * -10000

                hm_xs_filtered = hm_kps_filtered[:, :, :, 0]
                hm_ys_filtered = hm_kps_filtered[:, :, :, 1]

                # Fit a 2D gaussian distribution on the heatmap
                # Save a copy for further processing
                kps_heatmap_mean = torch.ones([batch, K, num_joints * 2], dtype=torch.float32) * -10000

                # Need optimization
                for idx_batch in range(batch):
                    for idx_joint in range(num_joints):
                        for idx_K in range(K):
                            if torch.equal(hm_xs_filtered[idx_batch][idx_joint][idx_K], torch.tensor(-10000).to(hm_xs_filtered.device)) or \
                                    torch.equal(hm_ys_filtered[idx_batch][idx_joint][idx_K], torch.tensor(-10000).to(hm_xs_filtered.device)):
                                continue
                            else:
                                win = 11
                                ran = win // 2

                                # For fair comparison, do not use fitted gaussian for correction
                                mu_x = ran
                                mu_y = ran

                                kps_heatmap_mean[idx_batch][idx_K][idx_joint * 2:idx_joint * 2 + 2] = \
                                    torch.FloatTensor([hm_xs_filtered[idx_batch][idx_joint][idx_K] + mu_x - ran,
                                                       hm_ys_filtered[idx_batch][idx_joint][idx_K] + mu_y - ran])

                kps_heatmap_mean = kps_heatmap_mean.to(kps_displacement_mean.device)

        if obj_scale is not None:
            obj_scale = _transpose_and_gather_feat(obj_scale, inds)
            obj_scale = obj_scale.view(batch, K, 3)
        else:
            obj_scale = torch.zeros([batch, K, 3], dtype=torch.float32)
            obj_scale = obj_scale.to(scores.device)

        detections = {'bboxes': bboxes,
                      'scores': scores,
                      'kps': kps,
                      'clses': clses,
                      'obj_scale': obj_scale,
                      'kps_displacement_mean': kps_displacement_mean,
                      'kps_heatmap_mean': kps_heatmap_mean}

        return detections


class TransformOutputs(nn.Module):
    """This module transform the outputs to the format expected by pnp solver."""

    def __init__(self, output_res_x=128, output_res_y=128):
        """PostProcess constructor.

        Args:
            output_res (int): the output resolution of the networks
        """
        super().__init__()
        self.output_res_x = output_res_x
        self.output_res_y = output_res_y

    @torch.no_grad()
    def forward(self, dets, c, s):
        """Transform the network outputs to the format expected by pnp solver

        Args:
            dets: detection results
            c: principle points
            s: the maximum axis of the orginal images
        """
        w = self.output_res_x
        h = self.output_res_y

        # Convert the results to numpy
        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()

        c = c.detach().cpu().numpy()
        s = s.detach().cpu().numpy()

        # Scale bbox & pts and Regroup
        if not ('scores' in dets):
            return [[{}]]

        ret = []

        for i in range(dets['scores'].shape[0]):

            preds = []

            for j in range(len(dets['scores'][i])):
                item = {}
                item['score'] = float(dets['scores'][i][j])
                item['cls'] = int(dets['clses'][i][j])
                item['obj_scale'] = dets['obj_scale'][i][j]

                # from w,h to c[i], s[i]
                bbox = transform_preds(dets['bboxes'][i, j].reshape(-1, 2), c[i], s[i], (w, h))
                item['bbox'] = bbox.reshape(-1, 4).flatten()

                item['ct'] = [(item['bbox'][0] + item['bbox'][2]) / 2, (item['bbox'][1] + item['bbox'][3]) / 2]

                kps = transform_preds(dets['kps'][i, j].reshape(-1, 2), c[i], s[i], (w, h))
                item['kps'] = kps.reshape(-1, 16).flatten()

                kps_displacement_mean = transform_preds(dets['kps_displacement_mean'][i, j].reshape(-1, 2), c[i], s[i],
                                                        (w, h))
                item['kps_displacement_mean'] = kps_displacement_mean.reshape(-1, 16).flatten()

                kps_heatmap_mean = transform_preds(dets['kps_heatmap_mean'][i, j].reshape(-1, 2), c[i], s[i], (w, h))
                item['kps_heatmap_mean'] = kps_heatmap_mean.reshape(-1, 16).flatten()

                preds.append(item)

            ret.append(preds)

        return ret


class MergeOutput(nn.Module):
    """This module group all the detection result from different scales on a single image."""

    def __init__(self, vis_threshold=0.3):
        """PostProcess constructor.

        Args:
            visualization_threshold (int): top K predictions to select from
        """
        super().__init__()
        self.thresh = vis_threshold

    @torch.no_grad()
    def forward(self, detections, nms=True):
        """Merge the detection results according to the score and nms.

        Args:
            detections: detection results of the model
            nms: Non-maximum Suppression for removing the redundunt bbox
        """
        ret = []
        for k in range(len(detections)):
            results = []
            for det in detections[k]:
                if det['score'] > self.thresh:
                    results.append(det)
            results = np.array(results)
            if nms:
                keep = soft_nms(results, Nt=0.5, method=2, threshold=self.thresh)
                results = results[keep]
            ret.append(results)
        return ret


class PnPProcess(nn.Module):
    """This module is to get 2d projection of keypoints & 6-DoF & 3d keypoint in camera frame."""

    def __init__(self, experiment_spec):
        """PostProcess constructor.

        Args:
            camera_intrinsic : camera intrinsic matrix used for the pnp solver
        """
        super().__init__()
        infer_config = experiment_spec.inference
        cx = infer_config['principle_point_x']
        cy = infer_config['principle_point_y']
        fx = infer_config['focal_length_x']
        fy = infer_config['focal_length_y']
        skew = infer_config['skew']

        self.camera_intrinsic = np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]])
        self.opencv = infer_config.opencv

    def set_intrinsic_matrix(self, intrinsic):
        """Set the intrinsic matrix manually"""
        self.camera_intrinsic = intrinsic.cpu().numpy()

    def set_3d_keypoints_format(self, eval_config):
        """Set the 3D keypoints format for the evaluation"""
        self.opencv = eval_config.opencv

    @torch.no_grad()
    def forward(self, det):
        """PnP solver for getting 2d projection of keypoints & 6-DoF & 3d keypoint"""
        # cv2 pnp solver can not batch processing.
        ret = []
        for idx in range(len(det)):
            results = {'keypoints_2d': [], 'projected_points': [], 'point_3d_cam': [], 'bbox': [], 'obj_scale': [], 'location': [], 'quaternion': [], 'score': []}
            outputs = det[idx]
            camera_intrinsic = self.camera_intrinsic[idx] if len(self.camera_intrinsic.shape) == 3 else self.camera_intrinsic

            for bbox in outputs:
                # 16 representation
                points_1 = np.array(bbox['kps_displacement_mean']).reshape(-1, 2)
                points_1 = [(x[0], x[1]) for x in points_1]
                points_2 = np.array(bbox['kps_heatmap_mean']).reshape(-1, 2)
                points_2 = [(x[0], x[1]) for x in points_2]
                points = np.hstack((points_1, points_2)).reshape(-1, 2)
                points_filtered = np.array(points)

                pnp_out = pnp_shell(points_filtered, np.array(bbox['obj_scale']), camera_intrinsic, opencv_return=self.opencv)

                if pnp_out is not None:
                    projected_points, point_3d_cam, location, quaternion = pnp_out
                    results['projected_points'].append(projected_points)
                    results['point_3d_cam'].append(point_3d_cam)
                    results['location'].append(location)
                    results['quaternion'].append(quaternion)
                    results['bbox'].append(bbox['bbox'])
                    results['obj_scale'].append(bbox['obj_scale'])
                    results['keypoints_2d'].append(np.array(points_1).reshape(-1, 2))
                    results['score'].append(bbox['score'])

            ret.append(results)
        return ret
