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

"""Evaluator for Referring Expression Segmentation."""

# Built-ins
import os
import json
from collections import defaultdict

# External
import torch
import torch.nn.functional as F
from torch import distributed as dist
import torchvision
from tqdm import tqdm

# Project-specific
from nvidia_tao_pytorch.core.tlt_logging import logger
from nvidia_tao_pytorch.core.distributed.comm import synchronize, get_world_size, get_global_rank
from nvidia_tao_pytorch.cv.mask_grounding_dino.utils.metrics import ap_per_mask


def all_tensor_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor with variable first dimension across distributed ranks.

    Args:
        tensor (torch.Tensor): Shape [N, D], where N may vary across ranks.

    Returns:
        torch.Tensor: Concatenated tensor from all ranks: [sum(N_i), D].
    """
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    device = tensor.device
    local_size = torch.tensor([tensor.shape[0]], device=device)
    size_list = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(sz.item()) for sz in size_list]
    max_size = max(size_list)

    # Pad tensor to max size
    if max_size > tensor.shape[0]:
        pad = torch.zeros((max_size - tensor.shape[0], tensor.shape[1]),
                          dtype=tensor.dtype, device=device)
        tensor = torch.cat([tensor, pad], dim=0)

    # Gather tensors from all ranks
    tensor_list = [torch.empty((max_size, tensor.shape[1]),
                               dtype=tensor.dtype, device=device)
                   for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    # Remove padding
    out = [t[:sz] for t, sz in zip(tensor_list, size_list)]
    return torch.cat(out, dim=0)


class ReferStyleCocoEvaluator:
    """Evaluator for referring expression datasets in COCO format."""

    def __init__(self, dataset_name: str, device: torch.device,
                 output_dir: str = None):
        """
        Args:
            dataset_name (str): Name of the dataset.
            device (torch.device): Device for tensor operations.
            output_dir (str, optional): Directory to save results.
            save_mask (bool, optional): Whether to save mask visualizations.
        """
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.predictions = []
        self.pstats = []
        self.pr_thresholds = [0.7, 0.8, 0.9]
        self.device = device
        self.iouv = torch.linspace(0.5, 0.95, 10).to(self.device)
        self.niou = self.iouv.numel()

    def reset(self):
        """Reset stored predictions and statistics."""
        self.predictions.clear()
        self.pstats.clear()

    def update(self, results, targets, no_targets):
        """Update evaluator with a batch of predictions and targets."""
        batch_preds, batch_pstats = self.convert_to_batch_preds_unified(results, targets, no_targets)
        self.predictions.extend(batch_preds)
        self.pstats.extend(batch_pstats)

    @synchronize
    def evaluate(self) -> dict | None:
        """Compute evaluation metrics across all distributed ranks.

        Returns:
            dict | None: Evaluation results if rank 0, else None.
        """
        if not self.predictions or not self.pstats:
            return None

        local_preds = torch.stack(self.predictions, dim=0)
        local_pstats = torch.cat(self.pstats, dim=0)

        all_preds_tensor = all_tensor_gather(local_preds)
        all_pstats_tensor = all_tensor_gather(local_pstats)

        if get_global_rank() != 0:
            return None

        # Convert gathered tensor to list of dicts
        all_preds = [
            {
                "img_id": int(row[0]),
                "expr_id": int(row[1]),
                "sent_id": int(row[2]),
                "gt_nt": bool(row[3]),
                "pred_nt": bool(row[4]),
                "I": int(row[5]),
                "U": int(row[6]),
            }
            for row in all_preds_tensor.cpu().tolist()
        ]

        # Initialize accumulators
        accum_I = accum_U = accum_IoU = 0.0
        pr_count = defaultdict(int)
        total_count = empty_count = not_empty_count = 0
        nt_stats = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        results_detailed = []

        for pred in tqdm(all_preds, desc="Evaluating"):
            gt_nt = pred["gt_nt"]
            pred_nt = pred["pred_nt"]
            intersection, union = pred["I"], pred["U"]

            total_count += 1
            result = pred.copy()

            if gt_nt:
                empty_count += 1
                if pred_nt:
                    nt_stats["TP"] += 1
                    accum_IoU += 1.0
                    result.update({"I": 0, "U": 0, "cIoU": 1.0})
                else:
                    nt_stats["FN"] += 1
                    accum_U += union
                    result.update({"I": 0, "U": union, "cIoU": 0.0})
            else:
                not_empty_count += 1
                if pred_nt:
                    nt_stats["FP"] += 1
                    intersection = 0
                else:
                    nt_stats["TN"] += 1
                iou = 0.0 if union == 0 else intersection / union
                accum_I += intersection
                accum_U += union
                accum_IoU += iou
                for th in self.pr_thresholds:
                    if iou >= th:
                        pr_count[th] += 1
                result.update({"I": intersection, "U": union, "cIoU": iou})

            results_detailed.append(result)

        # Compute AP per mask
        ap = ap_per_mask(all_pstats_tensor.to(self.device))
        map50, map_val = ap[0], ap.mean()

        # Aggregate final results
        final_results = {
            "dataset": self.dataset_name,
            "gIoU": 100.0 * accum_IoU / max(total_count, 1),
            "cIoU": 100.0 * accum_I / max(accum_U, 1),
            "mAP50": 100.0 * map50.cpu().item(),
            "mAP": 100.0 * map_val.cpu().item()
        }

        if empty_count > 0:
            total_t = nt_stats["TN"] + nt_stats["FP"]
            total_n = nt_stats["TP"] + nt_stats["FN"]
            final_results["T_acc"] = 100 * nt_stats["TN"] / max(total_t, 1)
            final_results["N_acc"] = 100 * nt_stats["TP"] / max(total_n, 1)
        else:
            final_results["T_acc"] = final_results["N_acc"] = 0.0

        for th in self.pr_thresholds:
            final_results[f"Pr@{th:.1f}"] = 100.0 * pr_count[th] / max(not_empty_count, 1)

        # Save results
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, f"{self.dataset_name}_results.json"), "w") as f:
                json.dump(final_results, f, indent=4)
            with open(os.path.join(self.output_dir, f"{self.dataset_name}_detailed_results.json"), "w") as f:
                json.dump(results_detailed, f, indent=4)
            logger.info(f"Saved evaluation results to {self.output_dir}")

        logger.info(final_results)
        return final_results

    def convert_to_batch_preds_unified(self, results, targets, no_targets):
        """Convert predictions and targets into unified mask-level AP format.

        Args:
            results (list[dict]): Model outputs.
            targets (list[dict]): Ground truth annotations.
            no_targets (list[bool]): Optional indicators for no-target cases.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: batch_preds, pstats
        """
        batch_preds = []
        pstats = []

        for i, (pred, target) in enumerate(zip(results, targets)):
            img_id = target["image_id"]
            expr_id = target["caption_id"]
            sent_id = target["sent_id"]
            gt_empty = target["empty"]
            org_h, org_w = target["orig_size"]

            gt_merged_mask = pred_merged_mask = None
            best_conf = torch.tensor(0.0, device=self.device)

            # Process ground truth masks
            if not gt_empty:
                gt_masks = target["masks"]
                aug_h, aug_w = target["size"]
                gt_masks = gt_masks[:, :aug_h, :aug_w]
                gt_masks = torchvision.ops.misc.interpolate(
                    gt_masks[:, None].float(),
                    size=(org_h, org_w),
                    mode='nearest',
                    align_corners=None
                )[:, 0] > 0.5
                gt_merged_mask = gt_masks.any(dim=0)

            pred_nt = pred["scores"].numel() == 0
            if no_targets is not None:
                pred_nt = no_targets[i].bool() or pred_nt

            if not pred_nt:
                pred_masks = F.interpolate(
                    pred["masks"].unsqueeze(0),
                    size=(org_h, org_w),
                    mode='bilinear',
                    align_corners=False
                )[0].gt(0.5)
                pred_merged_mask = pred_masks.any(dim=0)
                best_conf = pred["scores"].mean()

            # Compute intersection and union
            if gt_merged_mask is None or pred_merged_mask is None:
                intersection, union = 0, 0
            else:
                intersection = (gt_merged_mask & pred_merged_mask).sum().item()
                union = (gt_merged_mask | pred_merged_mask).sum().item()

            # Compute per-IoU correctness
            correct = torch.ones(self.niou, dtype=torch.bool, device=self.device) \
                if gt_merged_mask is None and pred_merged_mask is None \
                else (torch.zeros(self.niou, dtype=torch.bool, device=self.device) if union == 0
                      else (intersection / (union + 1e-6)) >= self.iouv)

            # pstats format: [correct flags, confidence, dummy class]
            pstats.append(torch.cat([
                correct[None].to(torch.bool),
                best_conf.view(1, 1),
                torch.zeros((1, 1), device=self.device)  # dummy class
            ], dim=1))

            # batch_preds format
            batch_preds.append(torch.tensor([
                int(img_id), int(expr_id), int(sent_id), int(gt_empty), int(pred_nt), int(intersection), int(union)
            ], dtype=torch.int32, device=self.device))

        return batch_preds, pstats
