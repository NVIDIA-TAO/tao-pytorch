# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Criterion Loss functions. """

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import math


def _gather_feat(feat, ind, mask=None):
    """Processed the feature maps format"""
    if len(ind.size()) > 2:

        num_symmetry = ind.size(1)
        dim = feat.size(2)
        ind = ind.unsqueeze(3).expand(ind.size(0), ind.size(1), ind.size(2),
                                      dim)
        ind = ind.view(ind.size(0), -1, ind.size(3))
        feat = feat.gather(1, ind)
        feat = feat.view(ind.size(0), num_symmetry, -1,
                         ind.size(2))
        if mask is not None:
            mask = mask.unsqueeze(3).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
    else:
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)

        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """Transposed and processed the feature maps"""
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _neg_loss(pred, gt):
    """ Modified focal loss. Exactly the same as CornerNet"""
    num_symmetry = gt.size(1)
    pred = pred.unsqueeze(1).repeat(1, num_symmetry, 1, 1, 1)

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = torch.zeros((gt.size(0), num_symmetry), dtype=torch.float32)
    loss = loss.to(pred.device)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum(dim=(2, 3, 4))
    pos_loss = pos_loss.sum(dim=(2, 3, 4))
    neg_loss = neg_loss.sum(dim=(2, 3, 4))

    loss = loss - neg_loss * num_pos.eq(0).float()
    # if num_pos ==0, plus 1 to avoid nan issue
    loss = loss - (pos_loss + neg_loss) / (num_pos + num_pos.eq(0).float()) * (~num_pos.eq(0)).float()
    return loss


def _reg_loss(regr, gt_regr, mask):
    """L1 regression loss"""
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class FocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        """initialize the focal loss"""
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        """Forward function for focal loss"""
        return self.neg_loss(out, target)


class RegLoss(nn.Module):
    """Regression loss for an output tensor"""

    def forward(self, output, mask, ind, target):
        """Forward function for regression loss"""
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):
    """Regression l1 loss for an output tensor
    Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    """

    def forward(self, output, mask, ind, target, relative_loss=False, dimension_ref=None):
        """Forward function for regression l1 loss"""
        pred = _transpose_and_gather_feat(output, ind)

        mask = mask.unsqueeze(3).expand_as(pred).float()

        if not relative_loss:
            if dimension_ref is None:
                loss = torch.abs(target * mask - pred * mask).sum(dim=(2, 3))
            else:
                # use_residual
                dimension_ref = torch.FloatTensor(dimension_ref).to(pred.device)
                pred = pred.exp() * dimension_ref
                loss = torch.abs(target * mask - pred * mask).sum(dim=(2, 3))
        else:
            target_rmzero = target.clone()
            target_rmzero[target_rmzero == 0] = 1e-06
            loss = torch.abs((1 * mask - pred * mask) / target_rmzero).sum(dim=(2, 3))
        loss = loss / (mask.sum(dim=(2, 3)) + 1e-4)

        return loss


class RegKLDScaleLoss(nn.Module):
    """Regression KLD Scale loss for an output tensor"""

    def forward(self, output, uncertainty, mask, ind, target, loss_config):
        """Forward function of the regression KLD scale loss for object scale"""
        pred = _transpose_and_gather_feat(output, ind)
        pred_uncertainty = _transpose_and_gather_feat(uncertainty, ind)

        mask = mask.unsqueeze(3).expand_as(pred).float()

        mse_loss = nn.MSELoss(reduction='none')

        a = mse_loss(target, pred) * mask

        # KLD
        b = torch.ones_like(a) * loss_config.KL_scale_uncertainty
        var = torch.exp(pred_uncertainty)
        loss = (pred_uncertainty - torch.log(b) + (b * torch.exp(-a / b) + a) / var - 1 + 0.5 * torch.abs(var)) * mask
        loss = loss.sum(dim=(2, 3))
        loss = loss / (mask.sum(dim=(2, 3)) + 1e-6)

        return loss


class RegKLDKeyLoss(nn.Module):
    """Regression KLD Key loss for an output tensor"""

    def forward(self, output, uncertainty, mask, ind, target, loss_config):
        """Forward function of the regression KLD key loss for object keypoint"""
        pred = _transpose_and_gather_feat(output, ind)
        pred_uncertainty = _transpose_and_gather_feat(uncertainty, ind)
        mask = mask.float()

        mse_loss = nn.MSELoss(reduction='none')
        a = mse_loss(target * mask, pred * mask)

        # KLD
        b = torch.ones_like(a) * loss_config.KL_kps_uncertainty
        var = torch.exp(pred_uncertainty)
        loss = (pred_uncertainty - torch.log(b) + (b * torch.exp(-a / b) + a) / var - 1 + 0.5 * torch.abs(var)) * mask
        loss = loss.sum(dim=(2, 3))
        loss = loss / (mask.sum(dim=(2, 3)) + 1e-6)

        return loss


class RegWeightedL1Loss(nn.Module):
    """Regression weighted l1 loss for an output tensor"""

    def forward(self, output, mask, ind, target):
        """Forward function of the regression l1 loss"""
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.float()

        loss = torch.abs(target * mask - pred * mask).sum(dim=(2, 3))
        loss = loss / (mask.sum(dim=(2, 3)) + 1e-4)
        return loss


class L1Loss(nn.Module):
    """L1 loss for an output tensor"""

    def forward(self, output, mask, ind, target):
        """forward function for the l1 loss"""
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


class ObjectPoseLoss(torch.nn.Module):
    """ This class computes the loss for CenterPose."""

    def __init__(self, loss_config):
        """ Create the criterion."""
        super(ObjectPoseLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_hm_hp = FocalLoss()
        self.crit_kp = RegWeightedL1Loss()
        self.crit_kp_uncertainty = RegKLDKeyLoss()
        self.crit_reg = RegL1Loss()
        self.crit_reg_uncertainty = RegKLDScaleLoss()
        self.loss_config = loss_config

    def forward(self, outputs, batch, phase):
        """ Forward function for loss function.

        hm: object center heatmap
        wh: 2D bounding box size
        hps/hp: keypoint displacements
        reg/off: sub-pixel offset for object center
        hm_hp: keypoint heatmaps
        hp_offset: sub-pixel offsets for keypoints
        scale/obj_scale: relative cuboid dimensions
        """
        loss_config = self.loss_config
        hm_loss, wh_loss, hp_loss = 0, 0, 0
        off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0
        obj_scale_loss = 0

        for s in range(loss_config.num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])
            if loss_config.hm_hp and not loss_config.mse_loss:
                output['hm_hp'] = _sigmoid(output['hm_hp'])

            hm_loss += self.crit(output['hm'], batch['hm']) / loss_config.num_stacks
            if loss_config.dense_hp:
                mask_weight = batch['dense_hps_mask'].sum() + 1e-4
                hp_loss += (self.crit_kp(output['hps'] * batch['dense_hps_mask'],
                                         batch['dense_hps'] * batch['dense_hps_mask']) /
                            mask_weight) / loss_config.num_stacks
            else:

                if not loss_config.hps_uncertainty or phase == 'val':

                    hp_loss += self.crit_kp(output['hps'], batch['hps_mask'],
                                            batch['ind'], batch['hps']) / loss_config.num_stacks
                else:
                    # KLD loss
                    hp_loss += self.crit_kp_uncertainty(output['hps'], output['hps_uncertainty'], batch['hps_mask'],
                                                        batch['ind'], batch['hps'], self.loss_config) / loss_config.num_stacks

            if loss_config.reg_bbox and loss_config.wh_weight > 0:
                wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                                         batch['ind'], batch['wh']) / loss_config.num_stacks

            # Add obj_scale
            if loss_config.obj_scale and loss_config.obj_scale_weight > 0:

                if phase == 'train':
                    if not loss_config.obj_scale_uncertainty:
                        if loss_config.use_residual:
                            obj_scale_loss += self.crit_reg(output['scale'], batch['reg_mask'],
                                                            batch['ind'], batch['scale'],
                                                            dimension_ref=loss_config.dimension_ref) / loss_config.num_stacks
                        else:
                            obj_scale_loss += self.crit_reg(output['scale'], batch['reg_mask'],
                                                            batch['ind'], batch['scale']) / loss_config.num_stacks

                    else:
                        # KLD loss
                        obj_scale_loss += self.crit_reg_uncertainty(output['scale'], output['scale_uncertainty'],
                                                                    batch['reg_mask'],
                                                                    batch['ind'], batch['scale'],
                                                                    self.loss_config) / loss_config.num_stacks
                else:
                    # Calculate relative loss only on validation phase
                    obj_scale_loss += self.crit_reg(output['scale'], batch['reg_mask'],
                                                    batch['ind'], batch['scale'], relative_loss=True) / loss_config.num_stacks

            if loss_config.reg_offset and loss_config.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / loss_config.num_stacks
            if loss_config.reg_hp_offset and loss_config.off_weight > 0:
                hp_offset_loss += self.crit_reg(
                    output['hp_offset'], batch['hp_mask'],
                    batch['hp_ind'], batch['hp_offset']) / loss_config.num_stacks
            if loss_config.hm_hp and loss_config.hm_hp_weight > 0:
                hm_hp_loss += self.crit_hm_hp(
                    output['hm_hp'], batch['hm_hp']) / loss_config.num_stacks

        loss = loss_config.hm_weight * hm_loss + loss_config.wh_weight * wh_loss + \
            loss_config.off_weight * off_loss + loss_config.hp_weight * hp_loss + \
            loss_config.hm_hp_weight * hm_hp_loss + loss_config.off_weight * hp_offset_loss + \
            loss_config.obj_scale_weight * obj_scale_loss

        # Calculate the valid_mask where samples are valid
        valid_mask = torch.gt(batch['ind'].sum(dim=2), 0)
        pos_inf = torch.zeros_like(loss)
        pos_inf[~valid_mask] = math.inf

        # Argmin to choose the best matched gt
        choice_list = torch.argmin(loss * valid_mask.float() + pos_inf, dim=1)

        # Update all the losses according to the choice (7+2 in total for now)
        hm_loss = torch.stack([hm_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()
        hp_loss = torch.stack([hp_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()

        if loss_config.reg_bbox and loss_config.wh_weight > 0:
            wh_loss = torch.stack([wh_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()
        if loss_config.obj_scale and loss_config.obj_scale_weight > 0:
            obj_scale_loss = torch.stack([obj_scale_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()
        if loss_config.reg_offset and loss_config.off_weight > 0:
            off_loss = torch.stack([off_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()
        if loss_config.reg_hp_offset and loss_config.off_weight > 0:
            hp_offset_loss = torch.stack([hp_offset_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()
        if loss_config.hm_hp and loss_config.hm_hp_weight > 0:
            hm_hp_loss = torch.stack([hm_hp_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()

        loss = loss_config.hm_weight * hm_loss + loss_config.wh_weight * wh_loss + \
            loss_config.off_weight * off_loss + loss_config.hp_weight * hp_loss + \
            loss_config.hm_hp_weight * hm_hp_loss + loss_config.off_weight * hp_offset_loss + \
            loss_config.obj_scale_weight * obj_scale_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'hp_loss': hp_loss,
                      'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'obj_scale_loss': obj_scale_loss
                      }

        # Fix the bug in multi gpus
        for key in loss_stats:
            if isinstance(loss_stats[key], int):
                loss_stats[key] = torch.from_numpy(np.array(loss_stats[key])).type(torch.FloatTensor).to(
                    'cuda' if loss_config.gpus[0] >= 0 else 'cpu')
        return loss
