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

"""3D stage model for NVPanoptix3D using WarpConvNet."""

import torch
from torch.nn import functional as F

from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.coords.ops.batch_index import offsets_from_batch_index
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.activations import Sigmoid

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.blocks import ProjectionBlock
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.helper import retry_if_cuda_oom
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.model_2d import MaskFormerModel
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.reconstruction import SparseProjection, FrustumDecoder
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.mp_occ import OccupancyAwareLifting

# Import WarpConvNet utilities
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.sparse_utils import (
    get_voxel_coordinates_at_batch,
    get_voxel_features_at_batch,
    sparse_collate,
    prepare_instance_masks_thicken,
    to_dense
)

# Import WarpConvNet coordinate transforms
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.coords_transform import (
    fuse_sparse_tensors, generate_multiscale_feat3d
)


class NVPanoptix3DModel(MaskFormerModel):
    """3D stage model that lifts 2D predictions into a sparse 3D frustum.

    The model reuses the frozen 2D MaskFormer stage to predict panoptic masks
    and depth, then performs occupancy-aware lifting and sparse frustum
    completion using WarpConvNet. The output includes dense 3D geometry and
    panoptic volumes (instance + semantic) along with auxiliary metadata needed
    for evaluation.
    """

    def __init__(self, cfg, export=False):
        """Initialize the 3D model.

        Args:
            cfg: Hydra/OmegaConfig configuration object. Expected to contain
                model projection parameters, frustum settings, and dataset
                settings (e.g., downsample factor).
        """
        super().__init__(cfg, export)
        self.cfg = cfg

        # disable gradients for the 2D model
        for _, param in self.named_parameters():
            param.requires_grad_(False)

        # 3D modules - use WarpConvNet versions
        self.reprojection = SparseProjection(self.cfg)
        self.completion = FrustumDecoder(self.cfg)
        self.projector = ProjectionBlock(
            self.cfg.model.projection.depth_feature_dim,
            self.cfg.model.projection.depth_feature_dim
        )
        self.ol = OccupancyAwareLifting(self.cfg)

        # 3D model parameters
        self.downsample_factor = cfg.dataset.downsample_factor
        self.frustum_dims = [cfg.model.frustum3d.frustum_dims] * 3
        self.iso_recon_value = cfg.model.frustum3d.iso_recon_value
        self.truncation = cfg.model.frustum3d.truncation
        self.num_classes = cfg.model.sem_seg_head.num_classes
        self.object_mask_threshold = cfg.model.object_mask_threshold
        self.overlap_threshold = cfg.model.overlap_threshold

    def infer_2d(self, batched_inputs, room_mask=None):
        """Run the frozen 2D model and postprocess outputs for 3D lifting.

        This method:
        - Runs the 2D backbone + MaskFormer head (no gradients),
        - Resizes/crops mask and depth predictions to remove padding,
        - Runs panoptic inference to obtain per-image segment info and semantic
          probability masks.

        Args:
            batched_inputs: Batch dict expected to contain:
                - image: input images tensor (B, C, H, W)
                - intrinsic: per-image intrinsics (B, 3, 3) or similar
                - height / width: original image sizes per sample
            room_mask: Optional room mask used to zero out depth outside the room.

        Returns:
            Tuple (outputs_2d, occupancy_preds, processed_results) where:
            - outputs_2d: dict produced by the 2D model head/backbone
            - occupancy_preds: multi-plane occupancy predictions
            - processed_results: list of per-image dicts containing:
                panoptic_seg, depth, image_size, padded_size,
                intrinsic, and semantic_seg (stored under sem_seg).
        """
        images = batched_inputs["image"]
        processed_images, orig_pad_shape, _ = self.resize_img_backbone(images, return_shape=True)
        with torch.no_grad():
            vggt_outputs = self.backbone(processed_images)
            multi_scale_features = vggt_outputs["multi_scale_features"]
            multi_scale_depth_features = vggt_outputs["multi_scale_depth_features"]
            outputs, occupancy_preds = self.sem_seg_head(
                multi_scale_features,
                multi_scale_depth_features,
                use_occ_head=True
            )
            outputs["pose_enc"] = vggt_outputs["pose_enc"]
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            depth_pred_results = vggt_outputs["depth"]

            padded_out_h, padded_out_w = orig_pad_shape[0] // 2, orig_pad_shape[1] // 2
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(padded_out_h, padded_out_w),
                mode="bilinear",
                align_corners=False,
            )
            depth_pred_results = F.interpolate(
                depth_pred_results,
                size=(padded_out_h, padded_out_w),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []
            # When multi-scale training is enabled, the unpadded input image size can vary.
            # We therefore crop to the true unpadded region (via nopad_image_shape) and
            # then resize the 2D predictions back to the reduced plane size.
            target_width = int(self.cfg.dataset.reduced_target_size[0])
            target_height = int(self.cfg.dataset.reduced_target_size[1])
            nopad_shapes = batched_inputs.get("nopad_image_shape", None)

            for idx, (
                mask_cls_result, mask_pred_result, depth_pred_result,
                per_image_intrinsic, height, width
            ) in enumerate(zip(
                mask_cls_results, mask_pred_results,
                depth_pred_results, batched_inputs["intrinsic"],
                batched_inputs["height"], batched_inputs["width"]
            )):
                if nopad_shapes is not None:
                    cur_height, cur_width = nopad_shapes[idx]
                    out_height, out_width = int(cur_height) // 2, int(cur_width) // 2
                else:
                    out_height, out_width = int(height) // 2, int(width) // 2
                processed_results.append({})

                # remove padding due to size divisibility
                mask_pred_result = mask_pred_result[:, :out_height, :out_width]
                depth_pred_result = depth_pred_result[:, :out_height, :out_width]

                # Bring predictions to the fixed reduced plane for 3D projection/lifting.
                if (out_height, out_width) != (target_height, target_width):
                    mask_pred_result = F.interpolate(
                        mask_pred_result.unsqueeze(0),
                        size=(target_height, target_width),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    depth_pred_result = F.interpolate(
                        depth_pred_result.unsqueeze(0),
                        size=(target_height, target_width),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                panoptic_seg, depth_r, segments_info, semantic_prob_masks = retry_if_cuda_oom(
                    self.post_processor.panoptic_inference
                )(
                    mask_cls_result,
                    mask_pred_result,
                    depth_pred_result
                )
                depth_r = depth_r[None]

                if room_mask is not None:
                    idx = len(processed_results) - 1
                    mask = F.interpolate(
                        room_mask[[idx], None].float(),
                        size=depth_r.shape,
                        mode="nearest",
                    )[0, 0].bool()
                    depth_r[~mask] = 0

                processed_results[-1]["panoptic_seg"] = (panoptic_seg, segments_info)
                processed_results[-1]["depth"] = depth_r[0]
                processed_results[-1]["image_size"] = (width, height)
                processed_results[-1]["padded_size"] = (orig_pad_shape[-1], orig_pad_shape[-2])
                processed_results[-1]["intrinsic"] = per_image_intrinsic
                processed_results[-1]["sem_seg"] = semantic_prob_masks

            return outputs, occupancy_preds, processed_results

    def forward(self, batched_inputs, kept, mapping, postprocess=True, is_matterport=False):
        """Forward pass for the 3D stage.

        Args:
            batched_inputs: Batch dict consumed by the 2D model and 3D lifting.
            kept: Boolean tensor indicating voxels within the frustum (used by
                occupancy-aware lifting).
            mapping: Voxel mapping tensor providing back-projection metadata for
                lifting.
            postprocess: If True, returns postprocessed per-sample 3D outputs
                (panoptic/semantic/instance masks).
            is_matterport: If True, uses Matterport-specific room mask key.

        Returns:
            If postprocess is True:
                List of per-sample dict outputs (see postprocess).
            Else:
                Raw 3D decoder outputs dict.
        """
        outputs_2d, occupancy_preds, processed_results = self.infer_2d(batched_inputs)

        # occupancy-aware lifting
        if is_matterport:
            room_mask = batched_inputs["room_mask_buol"]
        else:
            room_mask = None

        feat3d, mask3d = self.ol(processed_results, kept, mapping, occupancy_preds, room_mask)
        del occupancy_preds, mask3d

        multi_scale_features = list(reversed(outputs_2d["enc_features"]))
        depth_features = self.projector(
            outputs_2d["depth_features"],
            outputs_2d["mask_features"].shape[-2:]
        )
        encoder_features = torch.cat([outputs_2d["mask_features"], depth_features], dim=1)
        # sparse_multi_scale_features: list Voxels with native coordinates, without scaling by stride.
        sparse_multi_scale_features, sparse_encoder_features = self.reprojection(
            multi_scale_features, encoder_features, processed_results
        )

        del multi_scale_features
        del encoder_features

        # 3D frustum completion
        segm_queries = outputs_2d["segm_decoder_out"]
        frustum_mask = batched_inputs["frustum_mask"]
        frustum_mask_64 = F.max_pool3d(
            frustum_mask[:, None].float(),
            kernel_size=2,
            stride=4
        ).bool()

        # fuse features: occupancy-aware + visual features with zero padding
        # OccupancyAwareLifting now outputs coordinates aligned with SparseProjection
        if not is_matterport:
            multi_scale_feat3d = generate_multiscale_feat3d(feat3d)

            fused_multi_scale_features = []
            for i in range(len(multi_scale_feat3d)):
                fused_multi_scale_features.append(
                    fuse_sparse_tensors(
                        sparse_multi_scale_features[i], multi_scale_feat3d[i]
                    )
                )

            del sparse_multi_scale_features
            del multi_scale_feat3d

        try:
            fused_encoder_features = fuse_sparse_tensors(
                sparse_encoder_features, feat3d
            )
        except Exception:
            # feature fusion failed, using encoder features only
            logging.warning("Feature fusion failed, using encoder features only")
            fused_encoder_features = sparse_encoder_features

        del sparse_encoder_features
        del feat3d

        outputs_3d = self.completion(
            fused_multi_scale_features if not is_matterport else sparse_multi_scale_features,
            fused_encoder_features,
            segm_queries,
            frustum_mask_64
        )

        # copy over 2D results for matching
        outputs_3d["pred_logits"] = outputs_2d["pred_logits"]
        outputs_3d["pred_masks"] = outputs_2d["pred_masks"]

        if postprocess:
            return self.postprocess(outputs_3d, outputs_2d, processed_results, frustum_mask)

        return outputs_3d

    def panoptic_3d_inference(
        self,
        geometry,
        mask_cls,
        sparse_mask_tuple,
        min_coordinates,
        dense_dimensions
    ):
        """Convert sparse 3D mask predictions into dense panoptic/semantic volumes.

        Args:
            geometry: Dense geometry volume (e.g., SDF) used for surface masking.
            mask_cls: Per-query class logits of shape (Q, C+1).
            sparse_mask_tuple: Tuple (coords, sparse_masks, stride) where
                coords are sparse coordinates, sparse_masks are per-query
                mask logits/features at those coordinates, and stride is the
                voxel stride for coordinates.
            min_coordinates: Minimum coordinates used for densification.
            dense_dimensions: Desired dense output shape metadata.

        Returns:
            Tuple (panoptic_seg, panoptic_semantic_mapping, semantic_seg):
            - panoptic_seg: dense int32 volume of instance ids
            - panoptic_semantic_mapping: dict mapping instance id -> semantic class
            - semantic_seg: dense int32 volume of semantic labels
        """
        panoptic_seg = torch.zeros(geometry.shape, dtype=torch.int32, device=mask_cls.device)
        semantic_seg = torch.zeros_like(panoptic_seg)
        panoptic_semantic_mapping = {}

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        keep = labels.ne(self.num_classes) & \
            labels.ne(0) & \
            (scores > self.object_mask_threshold)

        coords, sparse_masks, stride = sparse_mask_tuple
        cur_scores = scores[keep]
        cur_classes = labels[keep]

        # Extract batch indices and spatial coordinates
        batch_indices = coords[:, 0]
        spatial_coords = coords[:, 1:]

        # Handle empty coordinates case
        if spatial_coords.shape[0] == 0:
            return panoptic_seg, panoptic_semantic_mapping, semantic_seg

        # IMPORTANT: Ensure int32 dtype for batch_indices
        batch_indices = batch_indices.int()
        spatial_coords = spatial_coords.int()

        # Handle empty case for offsets
        if batch_indices.numel() == 0:
            offsets = torch.tensor([0, 0], dtype=torch.int64, device=batch_indices.device)
        else:
            offsets = offsets_from_batch_index(batch_indices)

        mask_voxels = Voxels(
            batched_coordinates=IntCoords(spatial_coords, offsets=offsets, tensor_stride=stride),
            batched_features=CatFeatures(sparse_masks[:, keep], offsets=offsets),
        )

        # Apply sigmoid
        mask_voxels = Sigmoid()(mask_voxels)

        # Use the built-in to_dense method
        # Calculate max_coords in the strided coordinate space
        max_coords = tuple(
            (dense_dim - 1) for dense_dim in dense_dimensions[2:]  # dense_dimensions = [1, 1, 256, 256, 256]
        )

        cur_masks = mask_voxels.to_dense(
            channel_dim=1,
            min_coords=tuple(min_coordinates.tolist()),
            max_coords=max_coords
        )

        cur_masks = cur_masks.squeeze(0)

        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1, 1) * cur_masks

        current_segment_id = 0
        # Check cur_masks has valid predictions
        if cur_masks.shape[0] > 0:
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            query_to_segment_id = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in list(range(1, self.post_processor.num_thing_classes + 1))
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask.sum().item() > 0:
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            query_to_segment_id[k] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    query_to_segment_id[k] = current_segment_id
                    panoptic_semantic_mapping[current_segment_id] = int(pred_class)

            surface_mask = geometry.abs() <= 1.5

            # fill unassigned surface voxels
            unassigned_mask = surface_mask & (panoptic_seg == 0)
            for k in range(cur_classes.shape[0]):
                mask = (cur_mask_ids == k) & unassigned_mask
                if mask.sum().item() > 0 and k in query_to_segment_id.keys():
                    panoptic_seg[mask] = query_to_segment_id[k]

            for segm_id, semantic_label in panoptic_semantic_mapping.items():
                instance_mask = panoptic_seg == segm_id
                semantic_seg[instance_mask] = semantic_label

        return panoptic_seg, panoptic_semantic_mapping, semantic_seg

    def postprocess(self, outputs_3d, outputs_2d, processed_results, frustum_mask):
        """Postprocess raw 3D outputs into user-facing dense predictions.

        This step densifies sparse geometry, decodes per-sample 3D panoptic
        volumes, and prepares thickened instance masks for evaluation.

        Args:
            outputs_3d: Raw outputs from the 3D completion decoder.
            outputs_2d: Raw outputs from the 2D stage (used for class logits).
            processed_results: Per-image 2D postprocessed results from infer_2d.
            frustum_mask: Boolean frustum mask volume per batch element.

        Returns:
            List of per-sample dicts containing:
            - 2D fields: intrinsic, image_size, depth, panoptic_seg_2d
            - 3D fields: geometry, panoptic_seg, semantic_seg,
              panoptic_semantic_mapping, and instance_info_pred.
        """
        dense_dimensions = torch.Size([1, 1] + self.frustum_dims)
        min_coordinates = torch.IntTensor([0, 0, 0])

        geometry_results = to_dense(
            outputs_3d["pred_geometry"],
            dense_dimensions, min_coordinates,
            default_value=self.truncation
        )[0]

        mask_3d_results = outputs_3d["pred_segms"][-1]
        mask_cls_results = outputs_2d["pred_logits"]  # [batch_size, num_queries, num_classes]

        processed_results_3d = []

        for idx, (geometry_result, mask_cls_result) in enumerate(zip(
            geometry_results,
            mask_cls_results
        )):
            coords = get_voxel_coordinates_at_batch(mask_3d_results, idx)
            mask_3d = get_voxel_features_at_batch(mask_3d_results, idx)
            coords, mask_3d = sparse_collate([coords], [mask_3d])
            geometry_result = geometry_result.squeeze(0)
            panoptic_seg, panoptic_semantic_mapping, semantic_seg = self.panoptic_3d_inference(
                geometry_result,
                mask_cls_result,
                (coords, mask_3d, mask_3d_results.tensor_stride),
                min_coordinates,
                dense_dimensions,
            )

            processed_results_3d.append({
                "intrinsic": processed_results[idx]["intrinsic"],
                "image_size": processed_results[idx]["image_size"],
                "depth": processed_results[idx]["depth"],
                "panoptic_seg_2d": processed_results[idx]["panoptic_seg"],
                "geometry": geometry_result,
                "panoptic_seg": panoptic_seg,
                "semantic_seg": semantic_seg,
                "panoptic_semantic_mapping": panoptic_semantic_mapping,
                "instance_info_pred": prepare_instance_masks_thicken(
                    panoptic_seg,
                    panoptic_semantic_mapping,
                    geometry_result,
                    frustum_mask[idx],
                    iso_value=self.iso_recon_value,
                    downsample_factor=self.downsample_factor
                ),
            })

        return processed_results_3d
