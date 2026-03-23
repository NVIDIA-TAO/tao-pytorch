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

"""2D stage for NVPanoptix3D model."""

import torch
from torch import nn
from torch.nn import functional as F

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.mask2former.model.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder

from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt import VGGT
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.blocks import DepthProjector
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.helper import retry_if_cuda_oom
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.augmentations import ModelInputResize
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.mp_occ.back_projection import BackProjection
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.mp_occ.multiplane_occupancy import MultiPlaneOccupancyHead
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.transformer_decoder.joint_depth_transformer_decoder import \
    DepthAwareMultiScaleMaskedTransformerDecoder


class Postprocessor(nn.Module):
    """Postprocess 2D MaskFormer outputs into panoptic predictions.

    This module converts raw logits and mask predictions into:
    - a panoptic segmentation map with instance ids,
    - per-instance metadata (thing/stuff and class id),
    - a semantic probability tensor, and
    - a depth map aligned to the output resolution.

    It performs confidence filtering, overlap handling, and stuff-region merging
    to match typical panoptic inference behavior.
    """

    def __init__(self, cfg):
        """
        Initialize the postprocessor.

        Args:
            cfg: Configuration object containing model and dataset parameters.
                Expected attributes:
                - model.test_topk_per_image: Top-k predictions per image
                - model.sem_seg_head.num_classes: Number of segmentation classes
                - model.mask_former.num_object_queries: Number of query objects
                - model.object_mask_threshold: Threshold for mask confidence
                - model.overlap_threshold: Threshold for mask overlap ratio
                - dataset.depth_scale: Maximum depth value for clamping
                - dataset.num_thing_classes: Number of thing classes
        """
        super().__init__()
        self.cfg = cfg
        self.test_topk_per_image = cfg.model.test_topk_per_image
        self.num_classes = cfg.model.sem_seg_head.num_classes
        self.num_queries = cfg.model.mask_former.num_object_queries
        self.object_mask_threshold = cfg.model.object_mask_threshold
        self.overlap_threshold = cfg.model.overlap_threshold
        self.depth_scale = cfg.dataset.depth_scale
        self.num_thing_classes = cfg.dataset.num_thing_classes

    def panoptic_inference(self, mask_cls, mask_pred, depth_pred):
        """
        Perform panoptic segmentation inference.

        Combines classification scores and mask predictions to generate panoptic
        segmentation map. Handles merging of stuff regions and filtering of low-confidence
        predictions.

        Args:
            mask_cls (Tensor): Classification logits of shape (N_queries, N_classes + 1).
            mask_pred (Tensor): Mask predictions of shape (N_queries, H, W).
            depth_pred (Tensor): Depth predictions of shape (1, H, W).

        Returns:
            tuple: A tuple containing:
                - Tensor: Panoptic segmentation map of shape (H, W) with segment IDs.
                - Tensor: Clamped depth prediction of shape (H, W).
                - list: List of segment info dictionaries with keys 'id', 'isthing', 'category_id'.
                - Tensor: Semantic probability masks of shape (N_classes, H, W).
        """
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        semantic_prob_masks = torch.zeros((
            self.num_classes, h, w
        ), dtype=torch.float32, device=cur_masks.device)

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return panoptic_seg, depth_pred[0, :, :], segments_info, semantic_prob_masks
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            stuff_mask_ids = []
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in list(range(1, self.num_thing_classes + 1))
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        stuff_mask_ids.append(k)
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            semantic_prob_masks[int(pred_class)][mask] = cur_prob_masks[k][mask]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    semantic_prob_masks[int(pred_class)][mask] = cur_prob_masks[k][mask]

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            if stuff_mask_ids:
                # recover void pixels
                stuff_mask_ids = torch.tensor(stuff_mask_ids, dtype=torch.long, device=cur_prob_masks.device)
                cur_stuff_ids = stuff_mask_ids[cur_prob_masks[stuff_mask_ids].argmax(0)]
                empty_pixel_mask = panoptic_seg == 0
                for k in stuff_mask_ids:
                    k = k.item()
                    pred_class = cur_classes[k].item()
                    mask = empty_pixel_mask & (cur_stuff_ids == k)
                    panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                    semantic_prob_masks[int(pred_class)][mask] = cur_prob_masks[k][mask]

            # clamp depth_pred
            depth_pred = depth_pred[0, ...].clamp(min=0, max=self.depth_scale)
            return panoptic_seg, depth_pred, segments_info, semantic_prob_masks

    def sem_seg_postprocess(self, result, img_size, crop=True):
        """
        Return semantic segmentation predictions in the original resolution.

        Crops and interpolates predictions to match the original image size.

        Args:
            result (Tensor): Predictions of shape (C, H_pad, W_pad).
            img_size (tuple): Original image size as (H, W).
            crop (bool): Whether to crop the predictions to the original image size.
        Returns:
            Tensor: Resized predictions of shape (C, H, W).
        """
        if crop:
            # Crop each image in the batch to the original img_size
            result = result[:, :img_size[0], :img_size[1]]

        # result is (C, H, W)
        if tuple(result.shape[-2:]) != tuple(img_size):
            # Interpolate to the desired output size
            result = F.interpolate(
                result.expand(1, -1, -1, -1),
                size=(img_size[0], img_size[1]),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
        return result

    def forward(
        self, outputs, orig_shape, orig_pad_shape, output_scale=1.0, per_image_metadata=None
    ):
        """
        Forward pass through postprocessor.

        Interpolates predictions to original padded shape (scaled) and performs panoptic inference.

        Args:
            outputs (dict): Model outputs containing:
                - pred_logits: Classification logits of shape (B, N_queries, N_classes + 1)
                - pred_masks: Mask predictions of shape (B, N_queries, H, W)
                - pred_depths: Depth predictions of shape (B, 1, H, W)
            orig_shape (tuple): Original image shape as (H, W).
            orig_pad_shape (tuple): Padded image shape as (..., H_pad, W_pad).
            output_scale (float): Scale factor for output resolution. Default 1.0 (full res).
            per_image_metadata (list[dict], optional): Per-image metadata dicts to merge into results.

        Returns:
            list: List of processed results, each containing:
                - panoptic_seg: Tuple of (segmentation map, segment info list)
                - depth: Depth map of shape (H*scale, W*scale)
                - semantic_seg: Semantic probability masks of shape
                  (N_classes, H*scale, W*scale; stored under "sem_seg")
                - Additional keys from per_image_metadata if provided

        Raises:
            ValueError: If mode is not 'panoptic'.
        """
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        depth_pred_results = outputs.get("pred_depths_pad", None)
        if depth_pred_results is None:
            depth_pred_results = outputs.get("pred_depths", None)
        if depth_pred_results is None:
            raise KeyError("Expected pred_depths_pad or pred_depths in outputs for depth postprocessing.")

        # Compute scaled output size (padding size is batch-shared)
        scaled_pad_h = int(orig_pad_shape[-2] * output_scale)
        scaled_pad_w = int(orig_pad_shape[-1] * output_scale)
        scaled_orig_shape = (int(orig_shape[0] * output_scale), int(orig_shape[1] * output_scale))

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(scaled_pad_h, scaled_pad_w),
            mode="bilinear",
            align_corners=False,
        )

        depth_pred_results = F.interpolate(
            depth_pred_results,
            size=(scaled_pad_h, scaled_pad_w),
            mode="bilinear",
            align_corners=False,
        )

        outputs["pred_depths_pad"] = depth_pred_results

        if self.cfg.model.mode == "panoptic":
            processed_results = []
            for idx, (mask_cls_result, mask_pred_result, depth_pred_result) in enumerate(zip(
                mask_cls_results, mask_pred_results, depth_pred_results
            )):
                result = {}

                mask_pred_result = retry_if_cuda_oom(self.sem_seg_postprocess)(
                    mask_pred_result, scaled_orig_shape, crop=True
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

                depth_pred_result = retry_if_cuda_oom(self.sem_seg_postprocess)(
                    depth_pred_result, scaled_orig_shape, crop=True
                )

                panoptic_seg, depth_r, segments_info, semantic_prob_mask = retry_if_cuda_oom(
                    self.panoptic_inference
                )(mask_cls_result, mask_pred_result, depth_pred_result)

                result["panoptic_seg"] = (panoptic_seg, segments_info)
                result["depth"] = depth_r
                result["sem_seg"] = semantic_prob_mask

                # Merge per-image metadata if provided
                if per_image_metadata is not None and idx < len(per_image_metadata):
                    result.update(per_image_metadata[idx])

                processed_results.append(result)

            return processed_results

        else:
            raise ValueError("Only panoptic mode is supported for 2D model.")


class MaskFormerHead(nn.Module):
    """MaskFormer head for depth-aware panoptic segmentation.

    The head consists of:
    - a multi-scale pixel decoder (MSDeformAttn),
    - a depth-aware transformer decoder for class/mask prediction,
    - optional multi-plane occupancy prediction, and
    - a depth feature projector to align depth features to mask scales.

    It returns a predictions dict compatible with MaskFormer-style losses.
    """

    def __init__(self, cfg, input_shape, export):
        """
        Initialize MaskFormerHead.

        Args:
            cfg: Configuration object with model parameters.
            input_shape (dict): Dictionary mapping feature names to their shapes.
        """
        super().__init__()
        self.pixel_decoder = self.pixel_decoder_init(cfg, input_shape, export)
        self.predictor = self.predictor_init(cfg, export)
        self.occupancy_module = MultiPlaneOccupancyHead()

        depth_dim = 256
        if hasattr(cfg.model, "projection") and hasattr(cfg.model.projection, "depth_feature_dim"):
            depth_dim = cfg.model.projection.depth_feature_dim
        self.depth_projector = DepthProjector(in_channels=depth_dim, out_channels=depth_dim)
        self.device = torch.device("cuda")

    def pixel_decoder_init(self, cfg, input_shape, export):
        """
        Initialize pixel decoder module.

        Creates a multi-scale deformable attention pixel decoder for feature extraction.

        Args:
            cfg: Configuration object containing pixel decoder parameters.
            input_shape (dict): Dictionary mapping feature names to their shapes.

        Returns:
            MSDeformAttnPixelDecoder: Initialized pixel decoder module.
        """
        transformer_dropout = cfg.model.mask_former.dropout
        transformer_nheads = cfg.model.mask_former.nheads
        common_stride = cfg.model.sem_seg_head.common_stride
        transformer_dim_feedforward = cfg.model.mask_former.transformer_dim_feedforward
        transformer_enc_layers = cfg.model.sem_seg_head.transformer_enc_layers
        conv_dim = cfg.model.sem_seg_head.convs_dim
        mask_dim = cfg.model.sem_seg_head.mask_dim
        transformer_in_features = cfg.model.sem_seg_head.deformable_transformer_encoder_in_features
        norm = cfg.model.sem_seg_head.norm

        pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape, transformer_dropout, transformer_nheads, transformer_dim_feedforward,
            transformer_enc_layers, conv_dim, mask_dim, transformer_in_features, common_stride,
            norm=norm, export=export
        )
        return pixel_decoder

    def predictor_init(self, cfg, export):
        """
        Initialize predictor module.

        Creates a depth-aware transformer decoder for mask and class predictions.

        Args:
            cfg: Configuration object containing predictor parameters.

        Returns:
            DepthAwareMultiScaleMaskedTransformerDecoder: Initialized predictor module.
        """
        in_channels = cfg.model.sem_seg_head.convs_dim
        num_classes = cfg.model.sem_seg_head.num_classes
        mask_dim = cfg.model.sem_seg_head.mask_dim
        hidden_dim = cfg.model.mask_former.hidden_dim
        num_queries = cfg.model.mask_former.num_object_queries
        nheads = cfg.model.mask_former.nheads
        dim_feedforward = cfg.model.mask_former.dim_feedforward
        dec_layers = cfg.model.mask_former.dec_layers - 1
        pre_norm = cfg.model.mask_former.pre_norm
        enforce_input_project = False
        mask_classification = True
        predictor = DepthAwareMultiScaleMaskedTransformerDecoder(
            in_channels, num_classes, mask_classification, hidden_dim, num_queries, nheads,
            dim_feedforward, dec_layers, pre_norm, mask_dim, enforce_input_project, export
        )
        return predictor

    def forward(
        self,
        multi_scale_features,
        multi_scale_depth_features,
        use_occ_head=False,
        mask=None,
    ):
        """
        Forward pass.

        Processes multi-scale features through pixel decoder and predictor to generate
        panoptic segmentation predictions with optional occupancy predictions.

        Args:
            multi_scale_features (list of Tensor): Multi-scale image features from backbone.
                Each tensor has shape (B, C_i, H_i, W_i).
            multi_scale_depth_features (list of Tensor): Multi-scale depth features.
                Each tensor has shape (B, C_i, H_i, W_i).
            use_occ_head (bool): Whether to use occupancy head for predictions.
            mask (Tensor, optional): Optional mask for masked attention.

        Returns:
            tuple: A tuple containing:
                - dict: Predictions with keys 'pred_logits' and 'pred_masks'.
                - Tensor or None: Occupancy predictions if use_occ_head is True, else None.
        """
        # get multi-plane occupancy prediction:
        if self.occupancy_module is not None and use_occ_head:
            occupancy_pred = self.occupancy_module(multi_scale_features)
        else:
            occupancy_pred = None

        # get mask features
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(
            multi_scale_features
        )

        # get depth feature shape
        depth_feature_shape = mask_features.shape[-2:]
        size_list = []
        for i in range(len(multi_scale_features)):
            size_list.append(multi_scale_features[i].shape[-2:])

        depth_features, multi_scale_depth_features = self.depth_projector(
            multi_scale_depth_features, depth_feature_shape, size_list
        )

        predictions = self.predictor(
            multi_scale_features, multi_scale_depth_features, mask_features, depth_features, mask
        )

        return predictions, occupancy_pred


class MaskFormerModel(nn.Module):
    """End-to-end 2D stage model for NVPanoptix3D.

    This model wires together:
    - a backbone for RGB + depth feature extraction,
    - the MaskFormer head for class/mask prediction,
    - optional multi-plane occupancy output, and
    - a postprocessor to generate user-facing panoptic + depth outputs.

    The forward path can return raw outputs for training and optionally
    postprocessed predictions for inference/visualization.
    """

    def __init__(self, cfg, export=False):
        """
        Initialize MaskFormerModel.

        Args:
            cfg: Configuration object containing model parameters.
                Expected attributes include backbone type, pretrained weights,
                and size divisibility settings.
            export: Whether to export the model.
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = self.build_backbone(cfg)
        self.sem_seg_head = MaskFormerHead(cfg, self.backbone_feature_shape, export)
        self.post_processor = Postprocessor(cfg)
        self.back_projection = BackProjection(cfg)
        self.size_divisibility = cfg.model.mask_former.size_divisibility
        self.model_input_resize = ModelInputResize(self.size_divisibility)
        self.device = torch.device("cuda")

    def build_backbone(self, cfg):
        """Build and initialize the 2D backbone.

        Currently this implementation supports the VGGT backbone, optionally
        loading pretrained weights and freezing backbone parameters according to
        the model design (VGGT frozen, DPT head trainable).

        Args:
            cfg: Configuration object. Expected fields:
                - cfg.model.backbone.backbone_type: backbone identifier (e.g. "vggt")
                - cfg.model.backbone.pretrained_weights: optional checkpoint path

        Returns:
            An initialized backbone module that provides:
            - output_shape() for feature-shape metadata
            - freeze_modules() to freeze selected submodules

        Raises:
            NotImplementedError: If an unsupported backbone type is requested.
        """
        model_type = cfg.model.backbone.backbone_type
        if model_type == "vggt":
            backbone = VGGT()
            if cfg.model.backbone.pretrained_model_path != "":
                backbone.load_state_dict(
                    torch.load(
                        cfg.model.backbone.pretrained_model_path,
                        map_location="cpu"),
                    strict=False
                )
                logging.info("VGGT pretrained weights loaded successfully!")
        else:
            raise NotImplementedError(f"Unsupported backbone type '{model_type}'. Only 'vggt' is available.")

        self.backbone_feature_shape = backbone.output_shape()
        # Freeze VGGT, trainable DPT head
        backbone.freeze_modules()
        return backbone

    def forward(
        self,
        batch_inputs,
        post_process: bool = True,
        nopad_image_shape=None,
        room_mask: torch.Tensor | None = None,
    ):
        """
        Forward pass through the complete model.

        Processes input images through backbone, segmentation head, and optional
        postprocessing to generate panoptic segmentation and depth predictions.

        Args:
            batch_inputs (Tensor): Input images of shape (B, C, H, W).
            post_process (bool): Whether to apply postprocessing to outputs.
            nopad_image_shape (Tensor): Original image shape before size divisibility padding as (B, H, W).

        Returns:
            dict: Dictionary containing:
                - outputs (dict): Raw model outputs with keys:
                    - pred_logits: Classification logits of shape (B, N_queries, N_classes + 1)
                    - pred_masks: Mask predictions of shape (B, N_queries, H, W)
                    - pred_depths: Depth predictions of shape (B, 1, H, W)
                    - pose_enc: Pose encoding features
                    - occupancy_preds: Multi-plane occupancy predictions
                - processed_outputs (list or None): Postprocessed outputs if post_process=True,
                    containing panoptic segmentation maps, depth maps, and semantic masks.
        """
        processed_images, orig_pad_shape, resized_shape = self.resize_img_backbone(
            batch_inputs, return_shape=True
        )
        backbone_outputs = self.backbone(processed_images)

        multi_scale_features = backbone_outputs["multi_scale_features"]
        multi_scale_depth_features = backbone_outputs["multi_scale_depth_features"]

        outputs, occupancy_preds = self.sem_seg_head(
            multi_scale_features,
            multi_scale_depth_features,
            use_occ_head=True
        )
        outputs["pred_depths"] = backbone_outputs["depth"]
        outputs["pose_enc"] = backbone_outputs["pose_enc"]
        outputs["occupancy_preds"] = occupancy_preds

        processed_outputs = None
        if self.post_processor is not None and post_process:
            if nopad_image_shape is None:
                b = int(batch_inputs.shape[0])
                h = int(batch_inputs.shape[-2])
                w = int(batch_inputs.shape[-1])
                nopad_image_shape = [(h, w)] * b
            processed_outputs = self.post_processor(
                outputs, orig_shape=nopad_image_shape[0], orig_pad_shape=orig_pad_shape
            )
            if room_mask is not None:
                # room_mask is (B, H_pad, W_pad) after dataset padding/collate.
                for i, out in enumerate(processed_outputs):
                    h, w = nopad_image_shape[i]
                    # Crop to the unpadded (pre-divisibility) size.
                    rm = room_mask[i, :h, :w].to(dtype=torch.bool, device=out["depth"].device)
                    out["depth"] = out["depth"].masked_fill(~rm, 0)

        outputs["orig_pad_shape"] = orig_pad_shape
        outputs["resized_shape"] = resized_shape

        return {"outputs": outputs, "processed_outputs": processed_outputs}

    def resize_img_backbone(self, batch_imgs, target_size=448, size_divisibility=14, return_shape=False):
        """
        Resize and preprocess images for VGGT backbone input.

        Applies size divisibility padding, normalization, and resizing to prepare
        images for the backbone. Height is adjusted to be divisible by 14.

        Args:
            batch_imgs (Tensor): Batch of input images of shape (B, C, H, W).
            target_size (int): Target width for resizing.
            return_shape (bool): Whether to return the new & padded shape before resizing.

        Returns:
            Tensor or tuple: Returns images formatted for VGGT.
                VGGT expects either [S, 3, H, W] or [B, S, 3, H, W].
                We always provide [B, 1, 3, H, W] (single-frame sequence).

                If return_shape is False, returns processed images of shape (B, 1, 3, H', W').
                If True, returns tuple of (processed_imgs, (H_pad, W_pad), (H_resized, W_resized)).
        """
        batch_imgs = self.model_input_resize.apply_image(batch_imgs)
        batch_imgs = batch_imgs / 255.0
        pad_h, pad_w = batch_imgs.shape[-2:]
        pad_h, pad_w = int(pad_h), int(pad_w)
        new_height, new_width = pad_h, pad_w
        if target_size is not None:
            new_width = target_size
            # Use padded dimensions for consistent scaling (matches model/resize_img_vggt behavior)
            new_height = round(pad_h * (new_width / pad_w) / size_divisibility) * size_divisibility
            processed_imgs = F.interpolate(
                batch_imgs, size=(new_height, new_width), mode="bilinear", align_corners=False,
            )
            batch_imgs = processed_imgs.unsqueeze(1)
        if return_shape:
            return batch_imgs, (pad_h, pad_w), (new_height, new_width)
        return batch_imgs
