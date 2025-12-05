# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mask Grounding DINO loss functions."""

import torch
import torch.nn.functional as F
from torch import nn
import math
from typing import Tuple

from nvidia_tao_pytorch.core.distributed.comm import get_world_size, is_dist_avail_and_initialized
from nvidia_tao_pytorch.cv.deformable_detr.utils import box_ops
from nvidia_tao_pytorch.cv.mask_grounding_dino.utils.vl_utils import create_positive_map, create_positive_map_from_span
from nvidia_tao_pytorch.cv.mask_grounding_dino.model.model_utils import (
    dice_loss,
    nested_tensor_from_tensor_list,
    sigmoid_focal_loss,
)


def rela_mask_loss(
    pred_logits: torch.Tensor,
    ground_truth: torch.Tensor,
    mask_valid: torch.Tensor
) -> torch.Tensor:
    """
    Compute BCE loss for background and foreground logits,
    applied only on valid pixels as indicated by mask_valid.

    Args:
        pred_logits (torch.Tensor): Tensor of shape [B, 2, H, W] containing raw logits.
        ground_truth (torch.Tensor): Tensor of shape [B, 2, H, W] with:
            - channel 0: binary background (0 or 1)
            - channel 1: soft foreground probability [0, 1]
        mask_valid (torch.Tensor): Tensor of shape [B, 1, H, W]; 1 for valid pixels, 0 for invalid.

    Returns:
        torch.Tensor: Scalar tensor representing the weighted BCE loss:
            total_loss = 0.9 * bg_loss + 1.1 * fg_loss
    """
    C = pred_logits.shape[1]
    if C != 2:
        raise ValueError("Only 2-class foreground/background supported")

    # === Background BCE ===
    bg_logits = pred_logits[:, 0:1]
    bg_target = ground_truth[:, 0:1]
    bg_loss_map = F.binary_cross_entropy_with_logits(bg_logits, bg_target, reduction='none')
    bg_loss_val = (bg_loss_map * mask_valid).sum() / (mask_valid.sum() + 1e-6)

    # === Foreground Soft BCE ===
    fg_logits = pred_logits[:, 1:2]
    fg_target = ground_truth[:, 1:2]
    fg_loss_map = F.binary_cross_entropy_with_logits(fg_logits, fg_target, reduction='none')
    fg_loss_val = (fg_loss_map * mask_valid).sum() / (mask_valid.sum() + 1e-6)

    # Weighted sum
    total_loss = 0.9 * bg_loss_val + 1.1 * fg_loss_val
    return total_loss


# JIT compilation for speed
rela_mask_loss_jit = torch.jit.script(rela_mask_loss)


def cpp_pooling(
    label_map: torch.Tensor,
    output_size: Tuple[int, int],
    valid_mask: bool = False,
    skip_background: bool = False
) -> torch.Tensor:
    """
    Batched CPP pooling for background (class 0) vs. foreground (all other classes).

    Args:
        label_map (torch.Tensor): Input tensor of shape [N, 1, H, W] with class indices.
        output_size (Tuple[int, int]): Desired output size (H_out, W_out).
        valid_mask (bool, optional): If True, return only background mask. Defaults to False.
        skip_background (bool, optional): If True, return only foreground probability. Defaults to False.

    Returns:
        torch.Tensor: CPP output of shape [N, 2, H_out, W_out] (or [N, 1, H_out, W_out] if
                      valid_mask or skip_background is True), where:
                      - channel 0: background probability
                      - channel 1: foreground probability
    """
    if label_map.ndim != 4 or label_map.shape[1] != 1:
        raise ValueError("Input `label_map` must have shape [N, 1, H, W]")

    N, _, H, W = label_map.shape
    H_out, W_out = output_size

    # Handle non-divisible sizes by interpolation
    if (H % H_out != 0) or (W % W_out != 0):
        # Nearest interpolation keeps discrete class values (no mixing)
        label_map = F.interpolate(
            label_map.float(),
            size=(H_out * (H // H_out), W_out * (W // W_out)),
            mode="nearest",
            align_corners=None
        )
        H, W = label_map.shape[-2:]

    scale_h = H // H_out
    scale_w = W // W_out

    # Unfold to extract non-overlapping regions
    patches = F.unfold(
        label_map.float(), kernel_size=(scale_h, scale_w), stride=(scale_h, scale_w)
    )  # shape: [N, scale_h*scale_w, H_out*W_out]

    total_pixels = scale_h * scale_w

    # Count foreground: where label != 0
    fg_mask = (patches != 0).float()
    fg_count = fg_mask.sum(dim=1)  # [N, H_out*W_out]
    fg_prob = fg_count / total_pixels  # [N, H_out*W_out]
    bg_prob = 1.0 - fg_prob  # background probability

    if valid_mask:
        return bg_prob.view(N, 1, H_out, W_out)
    if skip_background:
        return fg_prob.view(N, 1, H_out, W_out)

    # Stack probabilities: [N, 2, H_out*W_out] -> [N, 2, H_out, W_out]
    probs = torch.stack([bg_prob, fg_prob], dim=1)
    return probs.view(N, 2, H_out, W_out)


def find_grid(N: int, H: int, W: int) -> Tuple[int, int]:
    """
    Find grid dimensions (n, m) for N queries to match aspect ratio H:W.

    Args:
        N (int): Number of queries.
        H (int): Height of the image.
        W (int): Width of the image.

    Returns:
        Tuple[int, int]:
            - n: number of rows in grid
            - m: number of columns in grid
    """
    best_n, best_m = 1, N
    best_diff = float("inf")

    # Step 1: Search factor pairs of N
    for n in range(1, int(math.isqrt(N)) + 1):
        if N % n == 0:
            m = N // n
            # Compare aspect ratios using cross product to avoid divisions
            diff = abs(H * m - W * n)
            if diff < best_diff:
                best_diff = diff
                best_n, best_m = n, m

    return best_n, best_m


class SetCriterion(nn.Module):
    """ This class computes the loss for Grounding DINO.

    The process happens in two steps:
        1) Compute hungarian assignment between ground truth boxes and the outputs of the model
        2) Supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, focal_alpha, focal_gamma, losses):
        """ Create the criterion.

        Args:
            matcher (nn.Module): module able to compute a matching between targets and proposals
            focal_alpha (float): alpha in token_sigmoid_binary_focal_loss
            focal_gamma (float): gamma in token_sigmoid_binary_focal_loss
            losses (list[str]): list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        if "rela" in self.losses:
            self.register_buffer("rela_weights", torch.tensor([0.9, 1.1], dtype=torch.float32), persistent=False)

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Args:
            outputs (dict[torch.Tensor]): computed outputs
            targets (List[dict]): target annotations
                targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
                target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
            indices (list): matching indices
            num_boxes (int): number of bounding boxes

        Returns:
            bbox loss and giou loss
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def token_sigmoid_binary_focal_loss(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the contrastive embedding.

        Args:
            outputs (dict[torch.Tensor]): computed outputs
            targets (List[dict]): target annotations
            indices (list): matching indices
            num_boxes (int): number of bounding boxes

        Returns:
            contrastive embedding loss.
        """
        pred_logits = outputs['pred_logits']
        new_targets = outputs['one_hot'].to(pred_logits.device)
        text_mask = outputs['text_mask']

        assert (new_targets.dim() == 3)
        assert (pred_logits.dim() == 3)

        if text_mask is not None:
            # each sample has different mask
            text_mask = text_mask.repeat(1, pred_logits.size(1)).view(outputs['text_mask'].shape[0], -1, outputs['text_mask'].shape[1])

            pred_logits = torch.masked_select(pred_logits, text_mask)
            new_targets = torch.masked_select(new_targets, text_mask)

        new_targets = new_targets.float()
        p = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, new_targets, reduction="none")
        p_t = p * new_targets + (1 - p) * (1 - new_targets)
        loss = ce_loss * ((1 - p_t) ** self.focal_gamma)

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * new_targets + (1 - self.focal_alpha) * (1 - new_targets)
            loss = alpha_t * loss

        total_num_pos = 0
        for batch_indices in indices:
            total_num_pos += len(batch_indices[0])
        num_pos_avg_per_gpu = max(total_num_pos, 1.0)
        loss = loss.sum() / num_pos_avg_per_gpu

        losses = {'loss_ce': loss}
        return losses

    def _get_src_permutation_idx(self, indices):
        """Permute predictions following indices.

        Args:
            indices (list): matching indices.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """Permute targets following indices.

        Args:
            indices (list): matching indices.
        """
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"]  # list[tensor]: bs x [1, num_inst, num_frames, H/4, W/4]

        for i in range(len(src_masks)):
            if len(src_masks[i].shape) == 4:
                return {'loss_mask': 0, 'loss_dice': 0}

        bs = len(targets)
        # src_masks: bs x [1, num_inst, num_frames, H/4, W/4] or [bs, num_inst, num_frames, H/4, W/4]
        if isinstance(src_masks, list):
            src_masks = torch.cat(src_masks, dim=1)[0]  # [num_all_inst, num_frames, H/4, W/4]
        if src_masks.ndim == 0:
            # no mask label (only box label)
            losses = {}
            losses['loss_mask'] = src_masks * 0.0
            losses['loss_dice'] = src_masks * 0.0
            return losses

        # src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks = self.get_target_masks(targets, src_masks)

        num_frames = src_masks.shape[1]
        # # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
        #                         mode="bilinear", align_corners=False)
        target_masks = target_masks.reshape(bs, -1, num_frames, target_masks.shape[-2], target_masks.shape[-1])
        target_masks = target_masks[tgt_idx]  # [num_all_inst, num_frames, H/4, W/4]

        # num_boxes = src_masks.shape[0] if self.ota else num_boxes

        if len(target_masks) == 0:  # no gt
            losses = {}
            losses['loss_mask'] = src_masks.sum() * 0.0
            losses['loss_dice'] = src_masks.sum() * 0.0
            return losses

        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        # src_masks/target_masks: [n_targets, num_frames* H * W]

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_rela(self, outputs, targets, indices, num_boxes):
        """
        Compute the ReLA (Region-Language Attention) related losses including:
        - no-target classification loss
        - minimap prediction loss
        - union mask prediction loss

        Args:
            outputs (dict): Model outputs containing at least:
                - "minimaps": tensor of shape [bs, 2, n_query]
                - "union_mask_logits": tensor of shape [bs, 2, H, W]
                - "no_targets": tensor of shape [bs, 2]
                - "valid_masks" (optional): tensor of shape [bs, 1, H, W]
            targets (list[dict]): List of target annotations for each sample, each containing:
                - "masks": tensor of shape [num_target_masks, H, W]
                - "empty" (optional): bool indicating no-target
            indices: Unused, kept for interface consistency.
            num_boxes: Unused, kept for interface consistency.

        Returns:
            dict: Dictionary containing the computed losses:
                - "loss_rela_nt": no-target classification loss
                - "loss_rela_minimap": minimap prediction loss
                - "loss_rela_union_mask": union mask prediction loss
        """
        # Prepare target masks
        masks = []
        for t in targets:
            m = t["masks"]
            if m.shape[0] == 0:
                m = torch.zeros((1, *m.shape[1:]), device=m.device, dtype=m.dtype)
            masks.append(m.any(dim=0, keepdim=True))

        src_minimap = outputs.get("minimaps", None)
        if src_minimap is not None:
            src_minimap = src_minimap.permute(0, 2, 1)  # [bs, n_query, 2]
        src_union_mask_logits = outputs["union_mask_logits"]
        src_nts = outputs["no_targets"]

        masks, _ = nested_tensor_from_tensor_list(
            masks,
            size_divisibility=8,
            split=False
        ).decompose()
        b, _, h, w = src_union_mask_logits.shape
        with torch.amp.autocast("cuda", enabled=True):
            masks = masks.to(src_union_mask_logits, non_blocking=True)
            # Pool masks to match union mask size
            target_masks = cpp_pooling(masks, (h, w)).to(dtype=src_union_mask_logits.dtype)
            valid_masks = cpp_pooling(outputs.get("valid_masks", masks), (h, w), valid_mask=True)
            loss_rela_union_mask = rela_mask_loss_jit(src_union_mask_logits, target_masks, valid_masks)

            if src_minimap is not None:
                n_query = src_minimap.size(2)
                H, W = masks.shape[-2:]
                n, m = find_grid(n_query, H, W)
                target_minimaps = cpp_pooling(masks, (n, m))
                src_minimap = src_minimap.view(b, -1, n, m)
                dummy_mask = torch.ones_like(src_minimap[:, :1])
                loss_rela_minimap = rela_mask_loss_jit(src_minimap, target_minimaps, dummy_mask)
            else:
                loss_rela_minimap = 0.0   # Validation does not require minimap loss.

            # Convert target no-target signal to tensor
            target_nts = torch.stack(
                [torch.tensor(t.get("empty", False), dtype=torch.long) for t in targets]
            ).to(src_nts.device)

            # Compute losses
            loss_rela_nt = F.cross_entropy(src_nts, target_nts, weight=self.rela_weights)

        losses = {
            "loss_rela_nt": loss_rela_nt,
            "loss_rela_minimap": loss_rela_minimap,
            "loss_rela_union_mask": loss_rela_union_mask,
        }

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Args:
            loss (str): name of the loss to get
            outputs (dict[torch.Tensor]): computed outputs
            targets (List[dict]): target annotations
            indices (list): matching indices
            num_boxes (int): number of bounding boxes

        Returns:
            the loss value given the loss name
        """
        loss_map = {
            'labels': self.token_sigmoid_binary_focal_loss,  # Now replaced CE w/ binary focal loss
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'rela': self.loss_rela,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, cat_list, caption, one_hot_token, return_indices=False):
        """ Performs the loss computation.

        Args:
            outputs (dict[torch.Tensor]): dict of tensors, see the output specification of the model for the format
            targets (List[dict]): list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses applied, see each loss' doc

        Returns:
            losses (dict): Dictionary of computed losses
        """
        device = next(iter(outputs.values())).device
        one_hot = torch.zeros(outputs['pred_logits'].size(), dtype=torch.int64)  # torch.Size([bs, 900, 256])
        token = one_hot_token

        label_map_list = []
        indices = []

        for j in range(len(cat_list)):  # bs
            if len(targets[j].get("positive_tokens", [])) > 0:
                tokens_positive = targets[j]["positive_tokens"]  # List[List[(char_start, char_end)]]
                label_map = create_positive_map_from_span(one_hot_token[j], [tokens_positive], empty=targets[j].get('empty', False))
            else:
                label_map = []
                for i in range(len(cat_list[j])):
                    label_id = torch.tensor([i])
                    per_label = create_positive_map(token[j], label_id, cat_list[j], caption[j], empty=targets[j].get('empty', False))
                    label_map.append(per_label)
                label_map = torch.stack(label_map, dim=0).squeeze(1)
            label_map_list.append(label_map)

        for j in range(len(cat_list)):  # bs
            if targets[j].get('empty', False):
                inds = [(torch.zeros((0,), dtype=torch.int64), torch.zeros((0,), dtype=torch.int64))]
                indices.extend(inds)
                continue
            for_match = {
                "pred_logits": outputs['pred_logits'][j].unsqueeze(0),
                "pred_boxes": outputs['pred_boxes'][j].unsqueeze(0)
            }
            inds = self.matcher(for_match, [targets[j]], label_map_list[j])
            indices.extend(inds)

        # indices : A list of size batch_size, containing tuples of (index_i, index_j) where:
        # - index_i is the indices of the selected predictions (in order)
        # - index_j is the indices of the corresponding selected targets (in order)

        tgt_ids = [v["labels"].cpu() for v in targets]

        for i in range(len(indices)):
            # Skip empty samples
            if targets[i].get('empty', False):
                continue

            # Index into target labels using matched indices
            tgt_ids[i] = tgt_ids[i][indices[i][1]]
            one_hot[i, indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)

        outputs['one_hot'] = one_hot
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_list = [len(t["labels"]) for t in targets]
        num_boxes = sum(num_boxes_list)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = []
                for j in range(len(cat_list)):  # bs
                    if targets[j].get('empty', False):
                        inds = [(torch.zeros((0,), dtype=torch.int64), torch.zeros((0,), dtype=torch.int64))]
                    else:
                        aux_output_single = {
                            'pred_logits': aux_outputs['pred_logits'][j].unsqueeze(0),
                            'pred_boxes': aux_outputs['pred_boxes'][j].unsqueeze(0)
                        }
                        inds = self.matcher(aux_output_single, [targets[j]], label_map_list[j])
                    indices.extend(inds)

                one_hot_aux = torch.zeros(outputs['pred_logits'].size(), dtype=torch.int64)
                for i in range(len(indices)):
                    tgt_ids[i] = tgt_ids[i][indices[i][1]]
                    one_hot_aux[i, indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)

                aux_outputs['one_hot'] = one_hot_aux
                aux_outputs['text_mask'] = outputs['text_mask']
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    if loss in ["rela"]:
                        # Rela loss is not computed for auxiliary outputs.
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = []
            for j in range(len(cat_list)):  # bs
                if targets[j].get('empty', False):
                    inds = [(torch.zeros((0,), dtype=torch.int64), torch.zeros((0,), dtype=torch.int64))]
                else:
                    interm_output_single = {
                        'pred_logits': interm_outputs['pred_logits'][j].unsqueeze(0),
                        'pred_boxes': interm_outputs['pred_boxes'][j].unsqueeze(0)
                    }
                    inds = self.matcher(interm_output_single, [targets[j]], label_map_list[j])
                indices.extend(inds)

            one_hot_aux = torch.zeros(outputs['pred_logits'].size(), dtype=torch.int64)
            for i in range(len(indices)):
                tgt_ids[i] = tgt_ids[i][indices[i][1]]
                one_hot_aux[i, indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)

            interm_outputs['one_hot'] = one_hot_aux
            interm_outputs['text_mask'] = outputs['text_mask']
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                kwargs = {}
                if loss in ['masks', "masks_boxinst", "rela"]:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    # Rela loss is not computed for intermediate outputs.
                    continue
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + '_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def get_target_masks(self, targets, src_masks):
        """Generate target masks from input."""
        self.mask_out_stride = 4
        target_masks, _ = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets],
            size_divisibility=8,
            split=False).decompose()
        target_masks = target_masks.to(src_masks)
        # downsample ground truth masks with ratio mask_out_stride
        if self.mask_out_stride != 1:
            start = int(self.mask_out_stride // 2)
            im_h, im_w = target_masks.shape[-2:]
            target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
            assert target_masks.size(2) * self.mask_out_stride == im_h
            assert target_masks.size(3) * self.mask_out_stride == im_w
        return target_masks
