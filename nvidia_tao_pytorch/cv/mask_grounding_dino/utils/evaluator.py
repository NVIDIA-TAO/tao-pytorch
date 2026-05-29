# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" (OD/VG) Evaluator for Mask Grounding DINO. """

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as ops
import torch.distributed as dist
from collections import defaultdict
from tabulate import tabulate

from nvidia_tao_pytorch.core.tlt_logging import logger
from nvidia_tao_pytorch.core.distributed.comm import synchronize, get_world_size, get_global_rank

# --- Metrics Utility ---


def ap_per_mask(pstats: torch.Tensor) -> torch.Tensor:
    """Compute AP@[0.5:0.95] treating all masks as a single class."""
    device = pstats.device
    niou = pstats.shape[1] - 2
    conf = pstats[:, -2]

    # Sort by descending confidence
    pstats = pstats[torch.argsort(-conf)]

    # Ground truth count estimate (max TP across IoUs)
    n_gt = pstats[:, :niou].sum(dim=0).max().item()
    if n_gt == 0:
        return torch.zeros(niou, device=device)

    tps = pstats[:, :niou].float()
    fps = 1.0 - tps

    tps_cum = torch.cumsum(tps, dim=0)
    fps_cum = torch.cumsum(fps, dim=0)

    recall_curve = tps_cum / (n_gt + 1e-16)
    precision_curve = tps_cum / (tps_cum + fps_cum + 1e-16)

    ap = torch.zeros(niou, device=device)
    for j in range(niou):
        r = recall_curve[:, j]
        p = precision_curve[:, j]
        # Precision envelope (monotonically decreasing)
        r = torch.cat([torch.tensor([0.0], device=device), r, torch.tensor([1.0], device=device)])
        p = torch.cat([torch.tensor([0.0], device=device), p, torch.tensor([0.0], device=device)])
        p = torch.flip(torch.cummax(torch.flip(p, dims=(0,)), dim=0)[0], dims=(0,))
        idx = (r[1:] != r[:-1]).nonzero().flatten()
        ap[j] = torch.sum((r[idx + 1] - r[idx]) * p[idx + 1])

    return ap

# --- Base Evaluator ---


class BaseEvaluator:
    """Base class for distributed evaluation providing common utilities."""

    def __init__(self, dataset_name, device, output_dir):
        """Initialize common evaluator attributes."""
        self.dataset_name = dataset_name
        self.device = device
        self.output_dir = output_dir
        self.iouv = torch.linspace(0.5, 0.95, 10).to(self.device)

    def reset(self):
        """Reset internal evaluation state."""
        raise NotImplementedError

    def all_tensor_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors of varying sizes from all ranks."""
        world_size = get_world_size()
        if world_size == 1:
            return tensor

        local_size = torch.tensor([tensor.shape[0]], device=self.device)
        size_list = [torch.tensor([0], device=self.device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(sz.item()) for sz in size_list]
        max_size = max(size_list)
        if max_size == 0:
            return torch.empty((0, *tensor.shape[1:]), dtype=tensor.dtype, device=self.device)
        # Pad local tensor to match max_size for all_gather
        if tensor.shape[0] < max_size:
            padding_shape = (max_size - tensor.shape[0], *tensor.shape[1:])
            padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=self.device)
            tensor = torch.cat([tensor, padding], dim=0)

        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)

        # Crop back to original sizes and concatenate
        return torch.cat([t[:sz] for t, sz in zip(tensor_list, size_list)], dim=0)

# --- OD Evaluator ---


class OD_Evaluator(BaseEvaluator):
    """Object Detection and Segmentation Evaluator for DDP."""

    def __init__(self, class_names=None, iou_types=['bbox'], **kwargs):
        """Initialize OD evaluator."""
        super().__init__(kwargs.get('dataset_name', 'coco'),
                         kwargs.get('device'),
                         kwargs.get('output_dir'))
        self.class_names = class_names
        self.iou_types = sorted(iou_types)
        self.reset()

    def reset(self):
        """Reset accumulated OD statistics."""
        self.stats = {task: [] for task in self.iou_types}
        self.gt_ious = {task: defaultdict(list) for task in self.iou_types}
        self.seen_images = 0

    def update(self, predictions, targets):
        """Update evaluator with predictions and targets from one batch."""
        targets_dict = {t['image_id'].item(): t for t in targets} if isinstance(targets, list) else targets
        for img_id, pred in predictions.items():
            self.seen_images += 1
            if img_id in targets_dict:
                target = targets_dict[img_id]
                for task in self.iou_types:
                    self.process_batch(pred, target, task, img_id)

    def process_batch(self, pred, target, task, img_id):
        """Process predictions and targets for a single image and task."""
        p_scores = pred['scores'].detach().cpu()
        p_labels = pred['labels'].detach().cpu()
        t_labels = target['labels'].detach().cpu()
        img_id_val = img_id.item() if torch.is_tensor(img_id) else img_id

        p_img_ids = torch.full((len(p_scores),), img_id_val, dtype=torch.long)
        t_img_ids = torch.full((len(t_labels),), img_id_val, dtype=torch.long)

        if len(p_scores) == 0:
            if len(t_labels) > 0:
                for cls in t_labels:
                    self.gt_ious[task][cls.item()].append((0.0, img_id_val))
            self.stats[task].append((torch.zeros(0, 10, dtype=torch.bool), p_scores, p_labels, t_labels, p_img_ids, t_img_ids))
            return

        if task == 'bbox':
            p_boxes, t_boxes = pred['boxes'].detach().cpu(), target['boxes'].detach().cpu()
            if len(t_boxes) == 0 or len(p_boxes) == 0:
                iou_matrix = torch.zeros(len(t_boxes), len(p_boxes))
            else:
                img_h, img_w = target.get("orig_size", (1.0, 1.0))
                if t_boxes.numel() > 0 and t_boxes.max() <= 1.0:
                    scale = torch.tensor([img_w, img_h, img_w, img_h])
                    t_boxes = ops.box_convert(t_boxes * scale, 'cxcywh', 'xyxy')
                iou_matrix = ops.box_iou(t_boxes, p_boxes)
        elif task == 'segm':
            iou_matrix = self._process_segm(pred, target).cpu()
        else:
            return

        if len(t_labels) > 0 and iou_matrix.numel() > 0:
            class_match = (t_labels.unsqueeze(1) == p_labels.unsqueeze(0)).float()
            best_iou_per_gt, _ = (iou_matrix * class_match).max(dim=1)
            for i, cls in enumerate(t_labels):
                self.gt_ious[task][cls.item()].append((best_iou_per_gt[i].item(), img_id_val))

        correct = torch.zeros(len(p_scores), 10, dtype=torch.bool)
        if iou_matrix.numel() > 0:
            for i, thresh in enumerate(self.iouv.cpu()):
                matches = self._match_predictions(p_labels, t_labels, iou_matrix, thresh)
                if matches.shape[0] > 0:
                    correct[matches[:, 0].long(), i] = True
        self.stats[task].append((correct, p_scores, p_labels, t_labels, p_img_ids, t_img_ids))

    def _process_segm(self, pred, target):
        """Process segmentation predictions and targets."""
        p_masks, t_masks = pred['masks'].detach(), target['masks'].detach()
        # Safeguard: Prevent empty tensor interpolation
        if len(p_masks) == 0 or len(t_masks) == 0:
            return torch.zeros((len(t_masks), len(p_masks)), device=self.device)
        org_h, org_w = target.get("orig_size", t_masks.shape[-2:])
        aug_h, aug_w = target.get("size", t_masks.shape[-2:])

        t_masks = F.interpolate(t_masks[:, :int(aug_h), :int(aug_w)].unsqueeze(0).float(),
                                size=(int(org_h), int(org_w)), mode='nearest').squeeze(0) > 0.5
        if p_masks.shape[-2:] != (int(org_h), int(org_w)):
            p_masks = F.interpolate(p_masks.unsqueeze(0).float(),
                                    size=(int(org_h), int(org_w)), mode='bilinear').squeeze(0) > 0.5

        t_flat, p_flat = t_masks.flatten(1).float(), p_masks.flatten(1).float()
        intersection = torch.mm(t_flat, p_flat.t())
        union = t_flat.sum(1).view(-1, 1) + p_flat.sum(1).view(1, -1) - intersection
        return intersection / (union + 1e-6)

    def _match_predictions(self, p_labels, t_labels, iou_matrix, iou_thresh):
        """Match predictions to ground truth."""
        candidates = torch.where(iou_matrix > iou_thresh)
        if len(candidates[0]) == 0:
            return torch.tensor([])
        matches, gt_seen, pred_seen = [], set(), set()
        sort_idx = torch.argsort(iou_matrix[candidates], descending=True)
        for i in sort_idx:
            gt_idx, pred_idx = candidates[0][i].item(), candidates[1][i].item()
            if gt_idx not in gt_seen and pred_idx not in pred_seen and p_labels[pred_idx] == t_labels[gt_idx]:
                matches.append([pred_idx, gt_idx])
                gt_seen.add(gt_idx)
                pred_seen.add(pred_idx)
        return torch.tensor(matches)

    @synchronize
    def synchronize_between_processes(self):
        """Synchronize statistics between processes."""
        if get_world_size() == 1:
            return
        merged_stats = {task: [] for task in self.iou_types}
        for task in self.iou_types:
            if self.stats[task]:
                cor, conf, p_cls, t_cls, p_img, t_img = zip(*self.stats[task])
                local_p = torch.cat([torch.cat(cor, 0).float(), torch.cat(conf, 0).unsqueeze(1),
                                     torch.cat(p_cls, 0).unsqueeze(1).float(), torch.cat(p_img, 0).unsqueeze(1).float()], 1).to(self.device)
                local_t = torch.stack([torch.cat(t_cls, 0).float(), torch.cat(t_img, 0).float()], 1).to(self.device)
            else:
                local_p, local_t = torch.zeros((0, 13), device=self.device), torch.zeros((0, 2), device=self.device)

            gp, gt = self.all_tensor_gather(local_p).cpu(), self.all_tensor_gather(local_t).cpu()
            merged_stats[task] = [(gp[:, :10].bool(), gp[:, 10], gp[:, 11].long(), gt[:, 0].long(), gp[:, 12].long(), gt[:, 1].long())]

        self.stats = merged_stats
        merged_gt_ious = {task: defaultdict(list) for task in self.iou_types}
        for task in self.iou_types:
            flat = [[v, c, i] for c, items in self.gt_ious[task].items() for v, i in items]
            gp = self.all_tensor_gather(torch.tensor(flat, device=self.device) if flat else torch.zeros((0, 3), device=self.device)).cpu().numpy()
            for v, c, i in gp:
                merged_gt_ious[task][int(c)].append((float(v), int(i)))
        self.gt_ious = merged_gt_ious

    def summarize(self):
        """Summarize evaluation results."""
        self.synchronize_between_processes()
        if get_global_rank() != 0:
            return {}
        final_results = {task: self._summarize_task(task) for task in self.iou_types}
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f'{self.dataset_name}_results.json'), 'w') as f:
                json.dump(final_results, f, indent=4, default=lambda x: x.item() if isinstance(x, np.generic) else x)
        return final_results

    def _summarize_task(self, task):
        """Summarize evaluation results for a specific task."""
        if not self.stats[task]:
            return {}
        correct, conf, pred_cls, target_cls, p_img_ids, t_img_ids = zip(*self.stats[task])
        tp, conf, pred_cls = torch.cat(correct, 0), torch.cat(conf, 0), torch.cat(pred_cls, 0)
        target_cls, p_img_ids, t_img_ids = torch.cat(target_cls, 0), torch.cat(p_img_ids, 0), torch.cat(t_img_ids, 0)

        # Standard indexing
        v_t, v_p = np.arange(len(t_img_ids)), np.arange(len(p_img_ids))
        target_cls, tp, conf, pred_cls = target_cls[v_t], tp[v_p], conf[v_p], pred_cls[v_p]

        idx = torch.argsort(conf, descending=True)
        tp, pred_cls = tp[idx], pred_cls[idx]

        results, stats_table = {}, []
        all_metrics = []

        unique_classes = target_cls.unique().int().tolist()
        for c in unique_classes:
            i = pred_cls == c
            n_gt = (target_cls == c).sum().item()
            if i.sum() == 0:
                metrics = [0.0] * 5
            else:
                tpc = tp[i].float().cumsum(0)
                fpc = (1 - tp[i].float()).cumsum(0)
                recall, precision = tpc / (n_gt + 1e-16), tpc / (tpc + fpc + 1e-16)
                ap_per_iou = [self._compute_ap(recall[:, j].numpy(), precision[:, j].numpy()) for j in range(10)]
                miou_list = [v for v, img in self.gt_ious[task][c]]
                miou = np.mean(miou_list) if miou_list else 0.0
                metrics = [precision[:, 0][-1].item(), recall[:, 0][-1].item(), ap_per_iou[0], np.mean(ap_per_iou), miou]

            name = self.class_names[c] if self.class_names and c < len(self.class_names) else str(c)
            results[name] = dict(zip(['P', 'R', 'mAP@50', 'mAP@50-95', 'mIoU'], metrics))
            stats_table.append([name, n_gt] + metrics)
            all_metrics.append([n_gt] + metrics)

        # Add "all" row (mean of all classes)
        if all_metrics:
            all_metrics_np = np.array(all_metrics)
            # Sum for instances, Mean for metrics
            avg_metrics = all_metrics_np.mean(axis=0)
            avg_metrics[0] = all_metrics_np[:, 0].sum()  # Total instances
            stats_table.append(["all", int(avg_metrics[0])] + avg_metrics[1:].tolist())
            results["all"] = dict(zip(['P', 'R', 'mAP@50', 'mAP@50-95', 'mIoU'], avg_metrics[1:].tolist()))

        logger.info(f"\n{task.upper()} OD Evaluation Results:\n" + tabulate(stats_table, headers=["Class", "Inst", "P", "R", "mAP@50", "mAP@50-95", "mIoU"], floatfmt=".4f"))
        return results

    def _compute_ap(self, recall, precision):
        """Compute Average Precision (AP) from recall and precision curves."""
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        return np.trapz(np.interp(np.linspace(0, 1, 101), mrec, mpre), np.linspace(0, 1, 101))

# --- VG Evaluator ---


class VG_Evaluator(BaseEvaluator):
    """Optimized Evaluator for Visual Grounding with No-Target support."""

    def __init__(self, **kwargs):
        """Initialize VG evaluator."""
        super().__init__(kwargs.get('dataset_name', 'vg'),
                         kwargs.get('device'),
                         kwargs.get('output_dir'))
        self.pr_thresholds = [0.5, 0.7, 0.9]
        self.predictions = []
        self.pstats = []

    def reset(self):
        """Reset accumulated VG predictions and statistics."""
        self.predictions.clear()
        self.pstats.clear()

    def update(self, results, targets, no_targets=None):
        """Update evaluator with predictions and targets from one batch."""
        for i, (pred, target) in enumerate(zip(results, targets)):
            img_id = target.get("image_id", 0)
            gt_empty = target.get("empty", False)
            org_h, org_w = target["orig_size"]

            # Ground Truth Mask Processing
            gt_mask = None
            raw_gt = target["masks"]
            if not gt_empty and len(raw_gt) > 0:
                raw_gt = raw_gt[:, : target["size"][0], : target["size"][1]]
                gt_mask = F.interpolate(raw_gt.unsqueeze(0).float(),
                                        size=(org_h, org_w), mode='nearest')[0].any(dim=0) > 0.5

            # Prediction Processing
            pred_nt = (pred["scores"].numel() == 0) or (no_targets is not None and no_targets[i].bool())
            pred_mask, conf = None, torch.tensor(0.0, device=self.device)
            raw_pred = pred["masks"]
            if not pred_nt and len(raw_pred) > 0:
                p_masks = F.interpolate(raw_pred.unsqueeze(0),
                                        size=(org_h, org_w), mode='bilinear', align_corners=False)[0] > 0.5
                pred_mask = p_masks.any(dim=0)
                conf = pred["scores"].mean()

            # IoU and Binary Logic
            if gt_mask is None and pred_mask is None:
                inter, union, iou = 0.0, 0.0, 1.0
            elif gt_mask is None or pred_mask is None:
                inter, union, iou = 0.0, (gt_mask.sum().item() if gt_mask is not None else pred_mask.sum().item()), 0.0
            else:
                inter = (gt_mask & pred_mask).sum().item()
                union = (gt_mask | pred_mask).sum().item()
                iou = inter / max(union, 1e-6)

            # [correct_flags, confidence, dummy_class]
            pstat = torch.cat([(iou >= self.iouv)[None].to(torch.bool),
                               conf.view(1, 1),
                               torch.zeros((1, 1), device=self.device)], dim=1)
            self.pstats.append(pstat)

            # [img_id, gt_empty, pred_nt, inter, union, iou]
            self.predictions.append(torch.tensor([int(img_id), int(gt_empty), int(pred_nt), inter, union, iou], device=self.device))

    @synchronize
    def evaluate(self) -> dict | None:
        """Evaluate VG predictions and targets."""
        if not self.predictions:
            return None
        all_preds = self.all_tensor_gather(torch.stack(self.predictions))
        all_pstats = self.all_tensor_gather(torch.cat(self.pstats))

        if get_global_rank() != 0:
            return None

        # Accuracy Metrics
        gt_empty_mask = all_preds[:, 1] == 1
        gt_content_mask = all_preds[:, 1] == 0

        # N_acc: Correctly predicted "no target" when target was empty
        n_acc = all_preds[gt_empty_mask, 2].mean().item() if gt_empty_mask.any() else 0.0
        # T_acc: Correctly predicted "content exists" when target had content
        t_acc = (1 - all_preds[gt_content_mask, 2]).mean().item() if gt_content_mask.any() else 0.0

        ap = ap_per_mask(all_pstats.to(self.device))

        final_res = {
            "dataset": self.dataset_name,
            "mIoU": 100.0 * all_preds[:, 5].mean().item(),
            "overall_IoU": 100.0 * all_preds[:, 3].sum().item() / max(all_preds[:, 4].sum().item(), 1e-6),
            "mAP50": 100.0 * ap[0].item(),
            "mAP": 100.0 * ap.mean().item(),
            "T_acc": 100.0 * t_acc,
            "N_acc": 100.0 * n_acc
        }

        # Recall at thresholds (Pr@X)
        valid_ious = all_preds[gt_content_mask, 5]
        for th in self.pr_thresholds:
            final_res[f"Pr@{th}"] = 100.0 * (valid_ious >= th).float().mean().item() if valid_ious.numel() > 0 else 0.0

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f"vg_{self.dataset_name}_metrics.json"), "w") as f:
                json.dump(final_res, f, indent=4)

        logger.info(f"--- VG Evaluation Results ---\n{json.dumps(final_res, indent=2)}")
        return final_res
