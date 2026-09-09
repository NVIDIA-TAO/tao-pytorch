# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVPanoptix3Dv2-Reasoning: Qwen ``[SEG]`` embedding as a frozen SAM3 prompt.

Multi-view reasoning segmentation (single-view is the ``S=1`` special case):

    views + query -> Qwen3-VL -> hidden([SEG] per positive view) -> projector -> SAM3 prompt(s)
    frozen SAM3 detector/segmentation head -> per-view instance masks
    frozen VGGT -> dense world points -> per-view masks fused into one segmented point cloud

Only Qwen LoRA/new-token rows and the ``[SEG]``->SAM prompt projector are
trainable. SAM3 and VGGT are frozen. Each ``[SEG]`` is injected in SAM3's
language-prompt tensor slot (replacing SAM's own text-encoder output) for its own
view; VGGT then lifts the per-view masks into one fused segmented point cloud.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.metric_scale_head import (
    MetricScaleHead,
    apply_metric_scale,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.reasoning.qwen_vlm import (
    Qwen3VLReasoner,
    bind_seg_to_views,
)


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Set ``requires_grad`` for every parameter in ``module``."""
    for param in module.parameters():
        param.requires_grad = flag


class SegToSAMPromptProjector(nn.Module):
    """Project Qwen ``[SEG]`` hidden states to SAM3 prompt-space tokens.

    SAM3 image uses a 256-d prompt/model dimension. Qwen3-VL-4B exposes a 2560-d
    LLM hidden state, so the default projector is a small MLP ``4096 -> 4096 -> 256``.
    Variable ``[SEG]`` counts are reduced to the first ``[SEG]`` per sample for
    this single-target prototype.
    """

    def __init__(self, d_llm: int, d_sam: int = 256, hidden: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.d_sam = int(d_sam)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_llm),
            nn.Linear(d_llm, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_sam),
            nn.LayerNorm(d_sam),
        )
        self.null_prompt = nn.Parameter(torch.zeros(1, d_sam))
        nn.init.normal_(self.null_prompt, std=0.02)

    def forward(self, h_seg: Tensor, batch_index: Tensor, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Return ``(prompt, valid)``.

        Args:
            h_seg: ``[N, d_llm]`` hidden states at all ``[SEG]`` positions.
            batch_index: ``[N]`` sample id for each row of ``h_seg``.
            batch_size: batch size ``B``.

        Returns:
            prompt: ``[1, B, 256]`` SAM language prompt tokens.
            valid:  ``[B]`` bool, true when that sample had a real ``[SEG]``.
        """
        device = h_seg.device if h_seg.numel() else self.null_prompt.device
        prompt = self.null_prompt.to(device).expand(batch_size, self.d_sam).clone()
        valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if h_seg.numel():
            projected = self.proj(h_seg)
            seen = set()
            for row, b_raw in enumerate(batch_index.tolist()):
                b = int(b_raw)
                if b in seen or b < 0 or b >= batch_size:
                    continue
                prompt[b] = projected[row]
                valid[b] = True
                seen.add(b)
        return prompt.unsqueeze(0), valid

    def project_all(self, h_seg: Tensor) -> Tensor:
        """Project every ``[SEG]`` hidden state to a SAM prompt token.

        Unlike :meth:`forward` (which reduces to one prompt per sample), this returns one
        prompt per ``[SEG]`` row, used by the multi-view path where each ``[SEG]`` prompts its
        own view.

        Args:
            h_seg: ``[K, d_llm]`` hidden states (``K`` may be 0).

        Returns:
            ``[K, d_sam]`` prompt tokens.
        """
        if h_seg.numel() == 0:
            return h_seg.new_zeros((0, self.d_sam))
        return self.proj(h_seg)


class NVPanoptix3Dv2Reasoning(nn.Module):
    """Qwen reasoner, frozen SAM3 masks, and optional frozen VGGT point clouds."""

    def __init__(
        self,
        qwen: Qwen3VLReasoner,
        sam3_image_model: nn.Module,
        projector: SegToSAMPromptProjector,
        vggt_geometry_model: Optional[nn.Module] = None,
        metric_depth_head: Optional[MetricScaleHead] = None,
        sam_resolution: int = 1008,
        point_mask_threshold: float = 0.5,
        point_conf_threshold: Optional[float] = None,
        freeze_sam: bool = True,
        freeze_vggt: bool = True,
    ):
        super().__init__()
        self.qwen = qwen
        self.sam = sam3_image_model
        self.vggt = vggt_geometry_model
        self.metric_depth_head = metric_depth_head
        self.projector = projector
        self.sam_resolution = int(sam_resolution)
        self.point_mask_threshold = float(point_mask_threshold)
        self.point_conf_threshold = point_conf_threshold
        if freeze_sam:
            set_requires_grad(self.sam, False)
        if self.vggt is not None and freeze_vggt:
            set_requires_grad(self.vggt, False)
        # SAM3 has mixed checkpoint dtypes in some environments; fp32 is the
        # most stable baseline for the projector experiment.
        self.sam.float()
        self.sam.eval()
        if self.vggt is not None:
            self.vggt.float()
            self.vggt.eval()

    @staticmethod
    def lift_bindings_to_point_clouds(
        pred_masks: Tensor,
        pred_logits: Tensor,
        world_points: Tensor,
        sample_idx: Tensor,
        view_idx: Tensor,
        batch_size: int,
        world_points_conf: Optional[Tensor] = None,
        mask_threshold: float = 0.5,
        conf_threshold: Optional[float] = None,
    ) -> Tuple[List[Tensor], Tensor, Tensor, Tensor]:
        """Lift each ``(binding)`` mask into its own view's VGGT points and fuse per sample.

        Args:
            pred_masks: ``[N,Q,h,w]`` SAM mask logits (one row per binding).
            pred_logits: ``[N,Q]`` SAM query logits.
            world_points: ``[B,S,H,W,3]`` dense 3D points (shared coordinate frame).
            sample_idx: ``[N]`` sample id per binding.
            view_idx: ``[N]`` view id per binding.
            batch_size: ``B``.
            world_points_conf: optional ``[B,S,H,W]`` (or ``[B,S,H,W,1]``) confidence map.
            mask_threshold: foreground threshold applied after sigmoid.
            conf_threshold: optional VGGT confidence gate.

        Returns:
            ``(point_clouds, dense_mask, selected_query, prob)``: per-sample fused clouds
            (list length ``B``), dense boolean selection ``[N,H,W]``, selected query ids ``[N]``,
            and resized mask probabilities ``[N,H,W]``.
        """
        if pred_masks.dim() != 4:
            raise ValueError(f"pred_masks must be [N,Q,h,w], got {tuple(pred_masks.shape)}")
        if world_points.dim() != 5 or world_points.shape[-1] != 3:
            raise ValueError(
                f"world_points must be [B,S,H,W,3], got {tuple(world_points.shape)}"
            )
        N = pred_masks.shape[0]
        _, _, H, W, _ = world_points.shape
        if N == 0:
            empty = pred_masks.new_zeros((0, H, W))
            return (
                [world_points.new_zeros((0, 3)) for _ in range(batch_size)],
                empty.bool(),
                pred_masks.new_zeros((0,), dtype=torch.long),
                empty,
            )
        selected_query = pred_logits.float().argmax(dim=1)  # [N]
        selected = pred_masks[torch.arange(N, device=pred_masks.device), selected_query]  # [N,h,w]
        prob = F.interpolate(
            selected.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
        ).sigmoid()[:, 0].to(world_points.device)  # [N,H,W]
        wp = world_points[sample_idx, view_idx]  # [N,H,W,3]
        valid_xyz = torch.isfinite(wp).all(dim=-1)  # [N,H,W]
        dense = (prob > float(mask_threshold)) & valid_xyz
        if world_points_conf is not None and conf_threshold is not None:
            conf = world_points_conf
            if conf.dim() == 5 and conf.shape[-1] == 1:
                conf = conf[..., 0]
            conf_sel = conf[sample_idx, view_idx].to(world_points.device)  # [N,H,W]
            dense = dense & (conf_sel >= float(conf_threshold))
        point_clouds: List[Tensor] = []
        for b in range(batch_size):
            rows = (sample_idx == b).nonzero(as_tuple=False).flatten().tolist()
            if rows:
                pts = torch.cat([wp[r][dense[r]] for r in rows], dim=0).detach()
            else:
                pts = world_points.new_zeros((0, 3))
            point_clouds.append(pts)
        return point_clouds, dense, selected_query, prob

    def sam_forward_with_prompt(self, images: Tensor, prompt: Tensor) -> Dict[str, Tensor]:
        """Run frozen SAM3 using ``prompt`` as its language-feature tensor.

        Args:
            images: ``[B,3,H,W]`` in [0, 1].
            prompt: ``[1,B,256]`` projected Qwen prompt.

        Returns:
            dict containing SAM3 ``pred_masks`` ``[B,Q,h,w]``, ``pred_logits``
            ``[B,Q]``, and optional ``presence_logits`` ``[B]``.
        """
        from sam3.model.data_misc import FindStage

        B = int(images.shape[0])
        device = next(self.sam.parameters()).device
        if device.type == "cuda":
            torch.cuda.set_device(device)
        images = images.to(device)
        prompt = prompt.to(device)

        x = F.interpolate(
            images,
            size=(self.sam_resolution, self.sam_resolution),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - 0.5) / 0.5

        self.sam.eval()
        autocast_dev = "cuda" if x.is_cuda else "cpu"
        with torch.no_grad(), torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
            backbone_out = self.sam.backbone.forward_image(x)

        # Override SAM's text encoder output with the projected Qwen [SEG] token.
        backbone_out["language_features"] = prompt
        backbone_out["language_mask"] = torch.zeros(B, 1, dtype=torch.bool, device=device)

        find_input = FindStage(
            img_ids=torch.arange(B, device=device, dtype=torch.long),
            text_ids=torch.arange(B, device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        geometric_prompt = self.sam._get_dummy_prompt(num_prompts=B)

        # Keep autograd enabled here: SAM parameters are frozen, but gradients
        # must flow from SAM mask loss back to the projected Qwen prompt.
        with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
            out = self.sam.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geometric_prompt,
            )

        pred_logits = out["pred_logits"].squeeze(-1).float()
        presence = out.get("presence_logit_dec")
        if presence is not None:
            presence = presence.squeeze(-1).float()
        return {
            "pred_masks": out["pred_masks"].float(),
            "pred_logits": pred_logits,
            "presence_logits": presence,
        }

    def vggt_forward(self, images: Tensor) -> Optional[Dict[str, object]]:
        """Run frozen VGGT and lift SAM masks into dense point clouds later."""
        if self.vggt is None:
            return None
        device = next(self.vggt.parameters()).device
        x = images.to(device)
        self.vggt.eval()
        autocast_dev = "cuda" if x.is_cuda else "cpu"
        with torch.no_grad(), torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16):
            out = self.vggt(x)
        return {
            "vggt_feats": out.get("vggt_feats"),
            "world_points": out.get("world_points"),
            "world_points_conf": out.get("world_points_conf"),
            "depth": out.get("depth"),
            "depth_conf": out.get("depth_conf"),
            "pose_enc": out.get("pose_enc"),
        }

    def apply_metric_depth_head(self, vggt_out: Dict[str, object]) -> Dict[str, object]:
        """Expose metric outputs using the shared metric-scale-head convention."""
        metric_head = self.metric_depth_head
        if metric_head is None:
            return {}
        depth = vggt_out.get("depth")
        pose_enc = vggt_out.get("pose_enc")
        vggt_feats = vggt_out.get("vggt_feats")
        world_points = vggt_out.get("world_points")
        if depth is None or pose_enc is None or vggt_feats is None:
            return {}

        H_d, W_d = depth.shape[2], depth.shape[3]
        params = metric_head(
            vggt_feats=vggt_feats,
            rel_depth=depth,
            pose_enc=pose_enc,
            image_size_hw=(H_d, W_d),
        )
        metric_out = {
            "scale_shift_params": params,
            "intrinsics": params["intrinsics"],
            "metric_depth": apply_metric_scale(depth, params),
        }
        if world_points is not None:
            metric_out["metric_points"] = (
                world_points * params["scale"][:, :, None, None, None]
            )
        return metric_out

    def forward(
        self,
        images: Tensor,
        qwen_inputs: Dict[str, Tensor],
        seg_view_indices: Optional[List[List[int]]] = None,
    ) -> Dict[str, object]:
        """Forward for multi-view reasoning segmentation (single-view is the ``S=1`` case).

        Each ``[SEG]`` token prompts SAM3 on its own (positive) view; all ``(sample, view)``
        bindings are flattened into one SAM3 pseudo-batch of size ``N``. VGGT runs once over all
        ``S`` views and the per-view masks are fused into one point cloud per sample.

        Args:
            images: ``[B,S,3,H,W]`` or ``[B,3,H,W]`` in [0,1].
            qwen_inputs: output of :meth:`Qwen3VLReasoner.build_inputs` (all ``S`` views).
            seg_view_indices: per-sample target views (ascending), one inner list per sample;
                the j-th ``[SEG]`` of a sample binds to ``seg_view_indices[b][j]``. Defaults to
                ``[[0]] * B`` (single-view / one-``[SEG]`` behavior).
        """
        if images.dim() == 4:
            images = images.unsqueeze(1)                  # [B,1,3,H,W]
        elif images.dim() != 5:
            raise ValueError(f"images must be [B,S,3,H,W] or [B,3,H,W], got {tuple(images.shape)}")
        B, S = images.shape[:2]
        device = images.device
        if seg_view_indices is None:
            seg_view_indices = [[0] for _ in range(B)]

        qout = self.qwen(qwen_inputs)
        h_seg, batch_index = self.qwen.extract_seg_hidden(qout)          # [M,d], [M]
        sample_idx, view_idx, keep = bind_seg_to_views(batch_index, seg_view_indices, S)
        N = int(keep.numel())

        dec_dtype = next(self.projector.parameters()).dtype
        if N > 0:
            prompts = self.projector.project_all(h_seg[keep].to(dec_dtype))  # [N,256]
            prompt = prompts.unsqueeze(0)                                    # [1,N,256]
            img_sel = images[sample_idx, view_idx]                           # [N,3,H,W]
            sam_out = self.sam_forward_with_prompt(img_sel, prompt)
        else:
            prompt = None
            sam_out = {
                "pred_masks": images.new_zeros((0, 1, 1, 1)),
                "pred_logits": images.new_zeros((0, 1)),
                "presence_logits": None,
            }

        sample_has_seg = torch.zeros(B, dtype=torch.bool, device=device)
        if N > 0:
            sample_has_seg[sample_idx] = True

        out = {
            "lm_logits": qout.lm_logits,
            "labels": qout.labels,
            "h_seg": h_seg,
            "batch_index": batch_index,
            "sam_prompt": prompt,
            "seg_valid": torch.ones(N, dtype=torch.bool, device=device),
            "seg_sample_idx": sample_idx,
            "seg_view_idx": view_idx,
            "sample_has_seg": sample_has_seg,
            **sam_out,
        }
        vggt_out = self.vggt_forward(images)
        if vggt_out is not None:
            out.update(vggt_out)
            out.update(self.apply_metric_depth_head(vggt_out))

        point_source = out.get("metric_points", out.get("world_points"))
        if point_source is not None:
            point_clouds, point_mask, selected_query, point_mask_prob = self.lift_bindings_to_point_clouds(
                pred_masks=out["pred_masks"],
                pred_logits=out["pred_logits"],
                world_points=point_source,
                sample_idx=sample_idx,
                view_idx=view_idx,
                batch_size=B,
                world_points_conf=out.get("world_points_conf"),
                mask_threshold=self.point_mask_threshold,
                conf_threshold=self.point_conf_threshold,
            )
            out.update({
                "segmented_point_clouds": point_clouds,
                "segmented_point_mask": point_mask,       # [N,H,W]
                "selected_query": selected_query,         # [N]
                "point_mask_prob": point_mask_prob,       # [N,H,W]
            })
            if "metric_points" in out:
                out["metric_segmented_point_clouds"] = point_clouds
        return out
