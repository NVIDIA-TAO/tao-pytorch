# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Evaluation 2D utils."""
import random
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from collections import OrderedDict
from typing import Dict, Optional, List
from torchmetrics import Metric, MetricCollection
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


class PanopticQuality(Metric):
    """TorchMetrics implementation of Panoptic Quality (PQ) metrics."""

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        metadata,
        matching_threshold: float = 0.5,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        """Initialize Panoptic Quality / Depth-aware Panoptic Quality accumulator.

        This metric accumulates true positives / false positives / false negatives and
        IoU sums per semantic class for panoptic predictions encoded as integer IDs.
        A prediction and a ground-truth segment are considered a match when:

        - They belong to the same semantic class, and
        - Their IoU exceeds ``matching_threshold``.

        The internal state is stored as per-class tensors and can be reduced across
        distributed workers using TorchMetrics' state synchronization (when enabled).

        Args:
            metadata: Dataset metadata object containing ``class_info``. Each item in
                ``class_info`` is expected to include:
                - ``trainId`` (int-like): contiguous class id used for training/eval
                - ``name`` (str): human readable class name
                - ``isthing`` (optional, bool/int-like): thing/stuff indicator
            matching_threshold: IoU threshold used to match predicted and GT segments.
            device: Device where metric state should be stored (CPU by default).
        """
        super().__init__(dist_sync_on_step=False)

        self.matching_threshold = matching_threshold
        self.metadata = metadata

        # Build class mappings
        self.class_id_to_name: OrderedDict[int, str] = OrderedDict()
        self.class_id_to_name[0] = "void"
        self._class_is_thing: Dict[int, bool] = {0: False}

        for item in metadata.class_info:
            class_id = int(item["trainId"])
            self.class_id_to_name[class_id] = item["name"]
            self._class_is_thing[class_id] = bool(item.get("isthing", 1))

        self._class_ids: List[int] = sorted(self.class_id_to_name.keys())
        self._class_lookup: Dict[int, int] = {cid: idx for idx, cid in enumerate(self._class_ids)}

        # Initialize metric states for epoch-wise computation
        self.zeros = torch.zeros(len(self._class_ids), dtype=torch.double).to(device)
        self.add_state("iou_sum", default=self.zeros.clone(), dist_reduce_fx="sum")
        self.add_state("tp", default=self.zeros.clone(), dist_reduce_fx="sum")
        self.add_state("fp", default=self.zeros.clone(), dist_reduce_fx="sum")
        self.add_state("fn", default=self.zeros.clone(), dist_reduce_fx="sum")

    def update(self, preds, gts, ignore_label=13, max_ins=1000, offset=256 * 256):
        """Accumulate PQ statistics for a batch of panoptic predictions.

        Each element in ``preds`` and ``gts`` is a 2D array (or tensor) of encoded
        panoptic IDs.

        Args:
            preds: List of predicted panoptic id maps. Each item can be a NumPy array
                or a torch.Tensor; tensors are detached and moved to CPU internally.
            gts: List of ground-truth panoptic id maps. Same type/shape constraints
                as ``preds``. Must have the same length as ``preds``.
            ignore_label: Semantic class id to ignore (e.g., void).
            max_ins: Maximum number of instances per semantic class used in the ID encoding.
            offset: Large constant used to compute a unique intersection ID:
                ``int_id = gt_id * offset + pred_id``. Must exceed the maximum possible
                encoded panoptic id value.
        """
        assert isinstance(preds, list)
        assert isinstance(gts, list)
        assert len(preds) == len(gts)

        for i in range(len(preds)):
            pred_ids = preds[i]
            gt_ids = gts[i]

            iou_per_class = self.zeros.clone().cpu().numpy().astype(np.float64)
            tp_per_class = iou_per_class.copy()
            fn_per_class = iou_per_class.copy()
            fp_per_class = iou_per_class.copy()

            def _ids_to_counts(id_array: Tensor):
                # Ensure tensor is on CPU and converted to numpy to avoid threading issues
                if isinstance(id_array, Tensor):
                    id_array = id_array.detach().cpu().numpy()
                unique_ids, counts = np.unique(id_array, return_counts=True)
                # Convert to plain Python ints to avoid tensor keys
                return {int(uid): int(cnt) for uid, cnt in zip(unique_ids, counts)}

            # Convert tensors to CPU numpy arrays upfront to avoid threading issues
            if isinstance(pred_ids, Tensor):
                pred_ids = pred_ids.detach().cpu().numpy()
            if isinstance(gt_ids, Tensor):
                gt_ids = gt_ids.detach().cpu().numpy()

            pred_areas = _ids_to_counts(pred_ids)
            gt_areas = _ids_to_counts(gt_ids)

            void_id = ignore_label * max_ins
            ign_ids = {gt_id for gt_id in gt_areas.keys() if (gt_id // max_ins) == ignore_label}

            int_ids = gt_ids * offset + pred_ids
            int_areas = _ids_to_counts(int_ids)

            def pred_void_overlap(pred_id, int_areas=int_areas, void_id=void_id, offset=offset):
                """Calculate overlap between prediction and void region."""
                return int_areas.get(void_id * offset + pred_id, 0)

            def pred_ignored_overlap(pred_id, int_areas=int_areas, ign_ids=ign_ids, offset=offset):
                """Calculate overlap between prediction and ignored regions."""
                return sum(
                    int_areas.get(ign_id * offset + pred_id, 0) for ign_id in ign_ids
                )

            gt_matched = set()
            pred_matched = set()

            for int_id, int_area in int_areas.items():
                pred_id = int_id % offset
                pred_cat = pred_id // max_ins
                if pred_cat == ignore_label:
                    continue
                gt_id = int_id // offset
                gt_cat = gt_id // max_ins
                if gt_cat != pred_cat:
                    continue
                union = (
                    gt_areas[gt_id] + pred_areas[pred_id] - int_area - pred_void_overlap(pred_id)
                )
                iou = int_area / union
                if iou > self.matching_threshold:
                    tp_per_class[gt_cat] += 1
                    iou_per_class[gt_cat] += iou
                    gt_matched.add(gt_id)
                    pred_matched.add(pred_id)

            for gt_id in gt_areas.keys():
                if gt_id in gt_matched:
                    continue
                gt_cat = gt_id // max_ins
                if gt_cat == ignore_label:
                    continue
                fn_per_class[gt_cat] += 1

            for pred_id in pred_areas.keys():
                if pred_id in pred_matched:
                    continue
                if (pred_ignored_overlap(pred_id) / pred_areas[pred_id]) > 0.5:
                    continue
                pred_cat = pred_id // max_ins
                if pred_cat == ignore_label:
                    continue
                fp_per_class[pred_cat] += 1

            self.iou_sum += torch.tensor(iou_per_class, device=self.iou_sum.device)
            self.tp += torch.tensor(tp_per_class, device=self.tp.device)
            self.fn += torch.tensor(fn_per_class, device=self.fn.device)
            self.fp += torch.tensor(fp_per_class, device=self.fp.device)

    def compute(self):
        """Compute PQ/SQ/RQ from the currently accumulated state.

        Returns:
            A tuple ``(pq, sq, rq)`` where each element is a tensor of shape
            ``(num_classes,)`` in percentage units (0-100).
        """
        out = self.compute_local()
        return out

    def compute_local(self):
        """Compute PQ/SQ/RQ without any explicit distributed synchronization.

        Returns:
            A tuple ``(pq, sq, rq)`` where:
            - ``sq`` (Segmentation Quality) is ``IoU_sum / tp`` (with safe denom)
            - ``rq`` (Recognition Quality) is ``tp / (tp + 0.5*fp + 0.5*fn)``
            - ``pq`` (Panoptic Quality) is ``sq * rq``
            Each tensor has shape ``(num_classes,)`` and is expressed in percentage units.
        """
        # Compute PQ from current tp/fp/fn state (synced if in DDP mode)
        sq = self.iou_sum / torch.maximum(self.tp, torch.ones_like(self.tp))
        rq = self.tp / torch.maximum(self.tp + 0.5 * self.fn + 0.5 * self.fp, torch.ones_like(self.tp))
        pq = sq * rq
        return pq * 100, sq * 100, rq * 100


class MetricsManager(nn.Module):
    """Unified manager for DPQ metrics across different phases.

    DPQ with threshold -1.0 is equivalent to PQ, so we only need DPQ metrics.
    Supports both batch-wise and epoch-wise computation for train/val/test phases.
    """

    NUM_CLASSES = 13
    MAX_INS = 1000
    IS_MATTERPORT = False
    THING_CLASS_INDICES = slice(1, 10)
    STUFF_CLASS_INDICES = slice(10, None)

    def __init__(
        self, metadata,
        dpq_thresholds: Optional[List[float]] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize a unified manager for PQ/DPQ metrics across phases.

        Args:
            metadata: Dataset metadata passed to each :class:`PanopticQuality`.
            dpq_thresholds: List of depth thresholds used by the depth-filtering logic.
                A value of ``-1.0`` is treated as the "plain PQ" variant (no depth filter),
                while positive thresholds correspond to DPQ variants.
            device: Optional device to place the metric collections on.
        """
        super().__init__()
        self.metadata = metadata
        self.device = device
        if dpq_thresholds is None:
            # Include -1.0 for PQ-equivalent computation
            dpq_thresholds = [-1.0, 0.5, 0.25, 0.1]
        self.dpq_thresholds = dpq_thresholds

        # Create PQ metric for each threshold and register as child modules for proper DDP sync
        # Sanitize keys for Module naming: no '.' or '-' allowed
        self._thr_to_key: Dict[float, str] = {
            float(t): ("thr_" + str(float(t)).replace("-", "m").replace(".", "_"))
            for t in self.dpq_thresholds
        }
        self.dpq_train = self.build_metric_collection()
        self.dpq_test = self.build_metric_collection()
        self.dpq_val = self.build_metric_collection()

        # Move metric collections to the desired device if provided
        if self.device is not None:
            self.dpq_train = self.dpq_train.to(self.device)
            self.dpq_test = self.dpq_test.to(self.device)
            self.dpq_val = self.dpq_val.to(self.device)

    def build_metric_collection(self):
        """Build a MetricCollection containing one PQ metric per threshold.

        Returns:
            A :class:`~torchmetrics.MetricCollection` keyed by sanitized threshold
            names (e.g. ``thr_0_5``), with each value being a :class:`PanopticQuality`.
        """
        return MetricCollection({
            self._thr_to_key[float(t)]: PanopticQuality(self.metadata, device=self.device)
            for t in self.dpq_thresholds
        })

    def get_metric(self, mode="train", threshold=-1):
        """Retrieve the metric instance for a given split and threshold.

        Args:
            mode: Data split name: ``"train"``, ``"val"``, or ``"test"``.
            threshold: Depth threshold value as configured in ``dpq_thresholds``.

        Returns:
            The corresponding :class:`PanopticQuality` instance from the appropriate
            metric collection.
        """
        if mode == "train":
            return self.dpq_train[self._thr_to_key[float(threshold)]]
        elif mode == "test":
            return self.dpq_test[self._thr_to_key[float(threshold)]]
        elif mode == "val":
            return self.dpq_val[self._thr_to_key[float(threshold)]]
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _save_debug_images(
        self, batch, preds, gts, depth_preds, depth_gts, pred_modified,
        depth_masks, ignored_masks, threshold, mode, batch_idx=0,
        debug_dir=None
    ):
        """Save debug visualization grids for qualitative inspection.

        This helper writes one PNG per image in the batch. Each PNG contains a 3x3
        grid showing:
        - Panoptic instance IDs and semantic predictions
        - Ground-truth semantic/instance labels
        - Depth prediction / ground-truth depth
        - Depth validity mask and ignored-pixel mask (based on the depth threshold)
        - Modified semantic prediction after applying the depth-based filtering

        Args:
            batch: Input batch dict (used for titles/metadata only).
            preds: List of predicted panoptic id maps (H, W).
            gts: List of GT panoptic id maps (H, W).
            depth_preds: List of predicted depth maps (H, W) (typically scaled to match GT units).
            depth_gts: List of GT depth maps (H, W).
            pred_modified: List of panoptic predictions after applying depth filtering.
            depth_masks: List of boolean masks indicating valid depth pixels.
            ignored_masks: List of boolean masks indicating pixels ignored by depth filtering.
            threshold: Depth threshold used to build the ignored mask (for filenames/titles).
            mode: Split name, used to create ``debug_images_{mode}`` directory.
            batch_idx: Batch index included in the saved filename prefix.
            debug_dir: Optional output directory path. If None, a repo-relative
                directory under ``nvidia_tao_pytorch/cv/nvpanoptix3d/test/`` is used.
        """
        if debug_dir is None:
            debug_dir = Path(f"nvidia_tao_pytorch/cv/nvpanoptix3d/test/debug_images_{mode}")
        else:
            debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(preds)):
            prefix = f"{debug_dir}/batch{batch_idx:04d}_img{i:02d}_thr{threshold:.2f}"

            # Create a figure with multiple subplots
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            fig.suptitle(f"Debug Visualization - Batch {batch_idx}, Image {i}, Threshold {threshold}",
                         fontsize=16)

            # 1. Original panoptic prediction
            pan_seg_vis = preds[i] % self.MAX_INS
            axes[0, 0].imshow(pan_seg_vis, cmap="tab20")
            axes[0, 0].set_title("Panoptic Seg (Instance IDs)")
            axes[0, 0].axis("off")

            # 2. Semantic segmentation prediction
            sem_seg_vis = preds[i] // self.MAX_INS
            axes[0, 1].imshow(sem_seg_vis, cmap="tab20", vmin=0, vmax=self.NUM_CLASSES)
            axes[0, 1].set_title("Semantic Seg Prediction")
            axes[0, 1].axis("off")

            # 3. Ground truth semantic segmentation
            gt_sem_seg_vis = gts[i] // self.MAX_INS
            axes[0, 2].imshow(gt_sem_seg_vis, cmap="tab20", vmin=0, vmax=self.NUM_CLASSES)
            axes[0, 2].set_title("Ground Truth Semantic Seg")
            axes[0, 2].axis("off")

            # 4. Ground truth instance segmentation
            gt_inst_seg_vis = gts[i] % self.MAX_INS
            axes[1, 0].imshow(gt_inst_seg_vis, cmap="tab20")
            axes[1, 0].set_title("Ground Truth Instance Seg")
            axes[1, 0].axis("off")

            # 5. Depth prediction
            depth_pred_normalized = depth_preds[i].squeeze()
            im = axes[1, 1].imshow(depth_pred_normalized, cmap="viridis")
            axes[1, 1].set_title(f"Depth Prediction (max: {depth_pred_normalized.max():.1f})")
            axes[1, 1].axis("off")
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

            # 6. Depth ground truth
            depth_gt_normalized = depth_gts[i].squeeze()
            im = axes[1, 2].imshow(depth_gt_normalized, cmap="viridis")
            axes[1, 2].set_title(f"Depth Ground Truth (max: {depth_gt_normalized.max():.1f})")
            axes[1, 2].axis("off")
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

            # 7. Depth mask (valid depth pixels)
            if i < len(depth_masks):
                axes[2, 0].imshow(depth_masks[i], cmap="gray")
                axes[2, 0].set_title(f"Depth Valid Mask ({depth_masks[i].sum()} pixels)")
            else:
                axes[2, 0].text(0.5, 0.5, "No depth mask", ha="center", va="center")
            axes[2, 0].axis("off")

            # 8. Ignored prediction mask (large depth errors)
            if i < len(ignored_masks):
                axes[2, 1].imshow(ignored_masks[i], cmap="Reds")
                axes[2, 1].set_title(f"Ignored Mask ({ignored_masks[i].sum()} pixels)")
            else:
                axes[2, 1].text(0.5, 0.5, "No ignored mask", ha="center", va="center")
            axes[2, 1].axis("off")

            # 9. Modified prediction (after applying depth masking)
            if i < len(pred_modified):
                pred_mod_sem = pred_modified[i] // self.MAX_INS
                axes[2, 2].imshow(pred_mod_sem, cmap="tab20", vmin=0, vmax=self.NUM_CLASSES)
                axes[2, 2].set_title("Modified Prediction (after depth filter)")
            else:
                axes[2, 2].imshow(sem_seg_vis, cmap="tab20", vmin=0, vmax=self.NUM_CLASSES)
                axes[2, 2].set_title("Prediction (no depth filter)")
            axes[2, 2].axis("off")

            plt.tight_layout()
            save_path = f"{prefix}.png"
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

    def update(self, batch, outputs, mode="train", return_stats=False, debug_save=False, batch_idx=0):
        """Update metrics for all configured depth thresholds for a batch.

        Args:
            batch: Input batch containing raw segmentation and depth data
            outputs: Model predictions containing panoptic_seg and depth
            mode: Data split mode ("train", "val", or "test")
            return_stats: If True, return per-batch metric statistics
            debug_save: If True, save debug visualization images to debug_images_{mode}/ directory
            batch_idx: Batch index used for debug image filenames (only used if debug_save=True)

        Returns:
            If ``return_stats`` is True, returns a nested dict with per-threshold DPQ
            means (overall/thing/stuff) computed *locally* for the current batch.
            Otherwise returns an empty dict.
        """
        target_device = outputs[0]["panoptic_seg"][0].device
        preds, gts, depth_preds, depth_gts = [], [], [], []

        for i in range(len(outputs)):
            panoptic_seg = outputs[i]["panoptic_seg"]
            panoptic_seg_map, seg_info = panoptic_seg
            panoptic_seg_map = panoptic_seg_map.detach()
            semantic_seg = torch.ones_like(panoptic_seg_map) * self.NUM_CLASSES

            for segm in seg_info:
                semantic_seg[panoptic_seg_map == segm["id"]] = segm["category_id"]

            gt_semantic_seg = batch["raw_sem_seg"][i].astype(np.uint32)
            gt_instance_seg = batch["raw_inst_seg"][i].cpu().numpy().astype(np.uint32)
            gt = gt_semantic_seg * self.MAX_INS + gt_instance_seg
            gts.append(gt)

            panoptic_seg_map = panoptic_seg_map.cpu().numpy().astype(np.uint32)
            semantic_seg = semantic_seg.cpu().numpy().astype(np.uint32)
            pred = semantic_seg * self.MAX_INS + panoptic_seg_map

            if pred.shape != gt.shape:
                pred = Image.fromarray(pred).resize(gt.shape[::-1], Image.NEAREST)
                pred = np.array(pred)
            preds.append(pred)

            depth_gt_i = batch["raw_depth"][i]
            depth_gts.append(depth_gt_i)

            depth_pred_i = outputs[i]["depth"].detach().cpu().numpy() * 256
            depth_pred_i = depth_pred_i.astype(np.int32)
            depth_preds.append(depth_pred_i)

        # loop for depth thresholds
        for threshold in self.dpq_thresholds:

            # ensure preds/gts and metric are on the same device before metric update
            metric_inst = self.get_metric(mode, threshold)
            if hasattr(metric_inst, "iou_sum") and metric_inst.iou_sum.device != target_device:
                metric_inst.to(target_device)

            # For debug visualization
            depth_masks_list = []
            ignored_masks_list = []
            pred_modified_list = []

            if threshold > 0:
                # Concatenate across images by flattening to 1D
                # depth_mask = np.concatenate(depth_gts, axis=1) > 0

                for depth_pred, depth_gt, pred in zip(depth_preds, depth_gts, preds):
                    # Resize depth_gt to match pred shape if needed (for multi-scale training)
                    if depth_gt.shape != pred.shape:
                        depth_gt = np.array(Image.fromarray(depth_gt).resize(pred.shape[::-1], Image.NEAREST))
                    if depth_pred.shape != pred.shape:
                        depth_pred = np.array(Image.fromarray(depth_pred).resize(pred.shape[::-1], Image.NEAREST))
                    depth_mask = depth_gt > 0
                    pred_in_depth_mask = pred[depth_mask]
                    ignored_pred_mask = (
                        np.abs(depth_pred[depth_mask] - depth_gt[depth_mask]) / depth_gt[depth_mask]
                    ) > threshold
                    pred_in_depth_mask[ignored_pred_mask] = self.NUM_CLASSES * self.MAX_INS
                    pred[depth_mask] = pred_in_depth_mask

                    # Store debug information
                    if debug_save:
                        depth_masks_list.append(depth_mask)
                        ignored_mask_full = np.zeros_like(depth_mask, dtype=bool)
                        ignored_mask_full[depth_mask] = ignored_pred_mask
                        ignored_masks_list.append(ignored_mask_full)
                        pred_modified_list.append(pred.copy())

            def _gt_process(img):
                """Normalize GT panoptic IDs into the expected uint32 encoding.

                Args:
                    img: Panoptic ID map where semantic and instance information is
                        encoded using ``MAX_INS``.

                Returns:
                    A ``np.uint32`` array with the normalized encoding.
                """
                return ((img // self.MAX_INS) * self.MAX_INS + (img % self.MAX_INS)).astype(np.uint32)

            gts = list(map(_gt_process, gts))

            # Save debug images only if debug_save, on GPU rank 0, and with 0.5 random probability
            # TODO: Apply correct visualization for cropping case
            if debug_save:
                global_rank = 0
                try:
                    # torch.distributed may not be initialized
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        global_rank = torch.distributed.get_rank()
                except Exception:
                    # fallback if any error with distributed
                    global_rank = 0
                if global_rank == 0 and random.random() < 0.05:  # Low probability of saving debug images
                    self._save_debug_images(
                        batch, preds, gts, depth_preds, depth_gts,
                        pred_modified_list, depth_masks_list, ignored_masks_list,
                        threshold, mode, batch_idx
                    )

            # update metric for this threshold
            self.get_metric(mode, threshold).update(
                preds, gts, ignore_label=self.NUM_CLASSES, max_ins=self.MAX_INS
            )

        metric_outputs = {}
        if return_stats:
            for threshold in self.dpq_thresholds:
                # Place temp metric on the same device as predictions to avoid CPU DDP sync with NCCL
                temp_metric = PanopticQuality(self.metadata, device=target_device).to(target_device)
                temp_metric.update(preds, gts, ignore_label=self.NUM_CLASSES, max_ins=self.MAX_INS)
                # Use local compute to avoid DDP all_gather during per-step logging
                results = temp_metric.compute_local()
                pq = results[0][:self.NUM_CLASSES]
                pq_th, pq_st = pq[self.THING_CLASS_INDICES], pq[self.STUFF_CLASS_INDICES]
                pq_th_mean, pq_st_mean = pq_th.mean().item(), pq_st.mean().item()
                # Use weighted average (concatenate then mean) to match original evaluation
                pq_mean = torch.cat([pq_th, pq_st]).mean().item()
                metric_outputs[str(threshold)] = {
                    "dpq": pq_mean, "dpq_th": pq_th_mean, "dpq_st": pq_st_mean
                }

            metric_outputs["averages"] = {
                "dpq": np.mean([metric_outputs[str(threshold)]["dpq"]
                                for threshold in self.dpq_thresholds if threshold > 0]),
                "dpq_th": np.mean([metric_outputs[str(threshold)]["dpq_th"]
                                   for threshold in self.dpq_thresholds if threshold > 0]),
                "dpq_st": np.mean([metric_outputs[str(threshold)]["dpq_st"]
                                   for threshold in self.dpq_thresholds if threshold > 0]),
            }

        return metric_outputs

    def compute(self, mode="train"):
        """Compute final epoch-level metrics for a given split.

        Args:
            mode: Data split name: ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            A dict keyed by threshold string (e.g. ``"0.5"``) containing:
            - ``dpq``: mean PQ over thing+stuff classes
            - ``dpq_th``: mean PQ over thing classes
            - ``dpq_st``: mean PQ over stuff classes

            Additionally includes an ``"averages"`` key with the mean over only the
            positive thresholds (threshold > 0).
        """
        outputs = {}
        positive_th_results = {key: [] for key in ["dpq", "dpq_th", "dpq_st"]}

        for threshold in self.dpq_thresholds:
            pq = self.get_metric(mode, threshold).compute()[0][:self.NUM_CLASSES]
            pq_th, pq_st = pq[self.THING_CLASS_INDICES], pq[self.STUFF_CLASS_INDICES]

            # Use weighted average (concatenate then mean) to match original evaluation
            metrics = {
                "dpq": torch.cat([pq_th, pq_st]).mean(),
                "dpq_th": pq_th.mean(),
                "dpq_st": pq_st.mean()
            }
            outputs[str(threshold)] = metrics

            if threshold > 0:
                for key in positive_th_results:
                    positive_th_results[key].append(metrics[key])

        outputs["averages"] = {
            key: torch.stack(vals).mean()
            for key, vals in positive_th_results.items()
        }
        return outputs

    def reset_epoch_metrics(self, mode="train"):
        """Reset accumulated metric state for a given split.

        Args:
            mode: Data split name: ``"train"``, ``"val"``, or ``"test"``.
        """
        for threshold in self.dpq_thresholds:
            self.get_metric(mode, threshold).reset()
