# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder for NVPanoptix3D using WarpConvNet."""

import torch
from torch import nn
from typing import List, Optional

from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.coords.ops.batch_index import offsets_from_batch_index
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.transforms import cat
from warpconvnet.nn.modules.activations import ReLU, Sigmoid
from warpconvnet.nn.modules.normalizations import InstanceNorm
from warpconvnet.nn.modules.mlp import Linear
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.nn.modules.prune import SparsePrune

from nvidia_tao_pytorch.cv.nvpanoptix3d.model.reconstruction.resnet import (
    BasicBlock3D,
    SparseBasicBlock3D,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.sparse_utils import (
    _is_empty_sparse,
    sparse_cat_union,
    add_voxels,
)


class SparseToDense(nn.Module):
    """Convert Voxels to a dense tensor."""

    def __init__(self, input_size: List[int]) -> None:
        """Initialize SparseToDense."""
        super().__init__()
        assert len(input_size) == 3
        self.input_size = input_size

    def forward(self, feature: Voxels) -> torch.Tensor:
        """Forward pass.
        Args:
            feature (Voxels): Voxels to convert to a dense tensor.
        Returns:
            torch.Tensor: Dense tensor.
        """
        stride = feature.tensor_stride

        out_size = (
            torch.div(
                torch.tensor(self.input_size),
                torch.tensor(stride),
                rounding_mode="floor"
            )
        ).tolist()

        coords = feature.batch_indexed_coordinates
        mask = (
            (coords[:, 1] >= 0) &
            (coords[:, 1] < out_size[0]) &
            (coords[:, 2] >= 0) &
            (coords[:, 2] < out_size[1]) &
            (coords[:, 3] >= 0) &
            (coords[:, 3] < out_size[2])
        )
        feature = SparsePrune()(feature, mask)

        dense = feature.to_dense(
            channel_dim=1,
            spatial_shape=out_size,
            min_coords=(0, 0, 0),
        )
        return dense


class FrustumDecoder(nn.Module):
    """Frustum decoder using WarpConvNet."""

    def __init__(self, cfg) -> None:
        """Initialize FrustumDecoder."""
        super().__init__()
        num_output_features = cfg.model.frustum3d.unet_output_channels
        num_features = cfg.model.frustum3d.unet_features
        sign_channel = cfg.model.projection.sign_channel
        mask_dim = cfg.model.sem_seg_head.mask_dim
        depth_dim = cfg.model.sem_seg_head.depth_dim
        num_classes = cfg.model.sem_seg_head.num_classes
        frustum_dims = cfg.model.frustum3d.grid_dimensions
        frustum_dims = [frustum_dims] * 3
        self.use_ms_features = cfg.model.frustum3d.use_multi_scale
        self.truncation = cfg.model.frustum3d.truncation

        buol_dim = 16
        if cfg.dataset.name == "matterport":
            ms_feature_channels = cfg.model.sem_seg_head.convs_dim
        else:
            ms_feature_channels = cfg.model.sem_seg_head.convs_dim + buol_dim

        # Init self.input_dims and self.input_encoders
        self.input_dims = [
            2 if sign_channel else 1,
            mask_dim + depth_dim,
            num_classes,
            buol_dim
        ]

        # encode inputs:
        self.input_encoders = nn.ModuleList()
        for input_dim in self.input_dims:
            downsample = nn.Sequential(
                SparseConv3d(
                    input_dim,
                    num_features,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                ),
                InstanceNorm(num_features),
            )
            self.input_encoders.append(
                SparseBasicBlock3D(
                    input_dim,
                    num_features,
                    downsample=downsample,
                )
            )

        # init level_encoders:
        self.level_encoders = nn.ModuleList(
            [
                self.make_encoder(
                    len(self.input_encoders) * num_features, num_features
                ),  # 16 * 3 = 48 -> 16
                self.make_encoder(num_features, num_features * 2),                         # 16 -> 32
                self.make_encoder(num_features * 2, num_features * 4, is_sparse=False),    # 32 -> 64
                self.make_encoder(num_features * 4, num_features * 8, is_sparse=False),    # 64 -> 128
                self.make_encoder(num_features * 8, num_features * 8, is_sparse=False),    # 128 -> 128
            ]
        )

        sparse_to_dense = SparseToDense(frustum_dims)

        if self.use_ms_features:
            self.feature_adapters = nn.ModuleList(
                [
                    self.make_adapter(ms_feature_channels, num_features),
                    self.make_adapter(ms_feature_channels, num_features * 2),
                    self.make_adapter(
                        ms_feature_channels, num_features * 4, [sparse_to_dense]
                    ),
                ]
            )
        else:
            self.feature_adapters = None

        self.enc_level_conversion = nn.ModuleList(
            [
                nn.Identity(),
                sparse_to_dense,
                nn.Identity(),
                nn.Identity(),
            ]
        )

        self.level_decoders = nn.ModuleList(
            [
                self.make_decoder(num_features * 3, num_output_features, last_layer=True),
                self.make_decoder(
                    num_features * 6, num_features * 2,
                    extra_layers=[SparseBasicBlock3D(num_features * 2, num_features * 2)],
                ),
                self.make_decoder(num_features * 8, num_features * 2, is_sparse=False),
                self.make_decoder(num_features * 16, num_features * 4, is_sparse=False),
                self.make_decoder(num_features * 8, num_features * 8, is_sparse=False),
            ]
        )

        # occupancy heads
        self.level_occupancy_heads = nn.ModuleList(
            [
                nn.Sequential(
                    InstanceNorm(num_output_features),
                    ReLU(inplace=True),
                    SparseBasicBlock3D(num_output_features, num_output_features),
                    SparseConv3d(num_output_features, 1, kernel_size=3, bias=True),
                ),
                Linear(num_features * 2, 1),
                nn.Linear(num_features * 4, 1),
            ]
        )

        # panoptic heads
        self.level_segm_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    InstanceNorm(num_output_features), ReLU(inplace=True),
                    SparseBasicBlock3D(num_output_features, num_output_features),
                ),
                SparseBasicBlock3D(num_features * 3, num_features * 3),
                nn.Sequential(
                    BasicBlock3D(num_features * 4, num_features * 4),
                    BasicBlock3D(num_features * 4, num_features * 4),
                ),
            ]
        )
        self.level_segm_query_projection = nn.ModuleList(
            [
                nn.Linear(mask_dim, num_output_features),  # 256 --> 16
                nn.Linear(mask_dim, num_features * 3),     # 256 --> 48
                nn.Linear(mask_dim, num_features * 4),     # 256 --> 64
            ]
        )

        # geometry head
        self.geometry_head = nn.Sequential(
            InstanceNorm(num_output_features),
            ReLU(inplace=True),
            SparseBasicBlock3D(num_output_features, num_output_features),
            SparseConv3d(num_output_features, 1, kernel_size=3, bias=True),
        )

        self.register_buffer(
            "frustum_dimensions", torch.tensor(frustum_dims), persistent=False
        )

    @staticmethod
    def forward_sparse_segm(segm_features: Voxels, queries: List[torch.Tensor]):
        """Forward pass for sparse segmentation.
        Args:
            segm_features (Voxels): Voxels to perform segmentation on.
            queries (List[torch.Tensor]): Queries to perform segmentation with.
        Returns:
            Voxels: Segmented voxels.
        """
        # Use batched_features indexing for cleaner decomposition
        features_list = [segm_features.batched_features[idx] for idx in range(segm_features.batch_size)]
        # Perform matrix multiplication for each batch
        outputs = []
        for idx in range(len(features_list)):
            feats_idx = features_list[idx]
            outputs.append(feats_idx @ queries[idx].T)

        stacked = torch.cat(outputs, dim=0)
        return segm_features.replace(batched_features=stacked)

    @staticmethod
    def make_encoder(input_dim, output_dim, is_sparse=True):
        """Make encoder.
        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            is_sparse (bool): Whether to use sparse convolution.
        Returns:
            nn.Module: Encoder.
        """
        if is_sparse:
            downsample = nn.Sequential(
                SparseConv3d(input_dim, output_dim, kernel_size=4, stride=2, bias=True),
                InstanceNorm(output_dim),
            )
            module = nn.Sequential(
                SparseBasicBlock3D(input_dim, output_dim, stride=2, downsample=downsample),
                SparseBasicBlock3D(output_dim, output_dim),
            )
        else:
            downsample = nn.Conv3d(
                input_dim, output_dim,
                kernel_size=4, stride=2,
                padding=1, bias=False
            )
            module = nn.Sequential(
                BasicBlock3D(input_dim, output_dim, stride=2, downsample=downsample),
                BasicBlock3D(output_dim, output_dim),
            )
        return module

    @staticmethod
    def make_decoder(input_dim, output_dim, is_sparse=True, extra_layers: Optional[List] = None, last_layer=False):
        """Make decoder.
        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            is_sparse (bool): Whether to use sparse convolution.
            extra_layers (Optional[List]): Extra layers to add to the decoder.
            last_layer (bool): Whether to add a last layer to the decoder.
        Returns:
            nn.Module: Decoder.
        """
        if extra_layers is None:
            extra_layers = []
        if is_sparse:
            return nn.Sequential(
                SparseConv3d(
                    input_dim, output_dim, kernel_size=4,
                    stride=2, bias=False,
                    transposed=True,
                    generative=True,
                ),
                InstanceNorm(output_dim),
                ReLU(inplace=True),
                *extra_layers,
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose3d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm3d(output_dim),
                nn.ReLU(inplace=True),
                *extra_layers,
            )

    @staticmethod
    def make_adapter(
        input_dim: int, output_dim: int, extra_layers: Optional[List] = None
    ):
        """Make adapter.
        Args:
            input_dim (int): Input dimension.
            output_dim (int): Output dimension.
            extra_layers (Optional[List]): Extra layers to add to the adapter.
        Returns:
            nn.Module: Adapter.
        """
        if extra_layers is None:
            extra_layers = []
        downsample = nn.Sequential(
            SparseConv3d(input_dim, output_dim, kernel_size=1, stride=1, bias=True),
            InstanceNorm(output_dim),
        )
        return nn.Sequential(
            SparseBasicBlock3D(input_dim, output_dim, downsample=downsample),
            *extra_layers,
        )

    def forward(
        self,
        ms_features: List[Voxels],
        features: Voxels,
        segm_queries,
        frustum_mask: torch.Tensor,
    ):
        """Forward pass.
        Args:
            ms_features (List[Voxels]): Multi-scale features.
            features (Voxels): Features.
            segm_queries: Segmentation queries.
            frustum_mask (torch.Tensor): Frustum mask.
        Returns:
            Dict: Predictions.
        """
        start_dim = 0
        encoded_inputs = []
        for dim, encoder in zip(self.input_dims, self.input_encoders):
            feature_slice = features.replace(
                batched_features=features.feature_tensor[:, start_dim:start_dim + dim]
            )
            encoded_inputs.append(encoder(feature_slice))
            start_dim += dim

        # concatenate encoded inputs:
        encoded_inputs = cat(*encoded_inputs)  # -> Voxels
        lvls = len(self.level_encoders)

        encoder_outputs = []
        encoder_inputs = [encoded_inputs]

        # ENCODER:
        for idx, encoder in enumerate(self.level_encoders):
            encoded = encoder(encoder_inputs[idx])   # idx = 0, encoder_inputs[idx] (1,1,1) -> (2,2,2)
            if self.use_ms_features and idx < len(self.feature_adapters):  # only for 0, 1, 2 levels
                feat = self.feature_adapters[idx](ms_features[idx])
                if isinstance(encoded, torch.Tensor):
                    encoded = encoded + feat
                else:
                    encoded = add_voxels(encoded, feat)
            encoder_outputs.append(encoded)
            if idx < lvls - 1:   # 0, 1, 2, 3
                encoder_inputs.append(self.enc_level_conversion[idx](encoded))

        # low to high resolution
        decoder_outputs = []
        decoder_inputs = [encoder_outputs[-1]]
        pred_occupancies = []
        pred_segms = []
        pred_geometry = None

        # U-Net:
        for idx in reversed(range(lvls)):
            decoded = self.level_decoders[idx](decoder_inputs[lvls - 1 - idx])
            decoder_outputs.append(decoded)

            if idx <= 1:
                # level 128, 256
                occupancy = self.level_occupancy_heads[idx](decoded)
                # mask invalid voxels outside of frustum
                coords = occupancy.batch_indexed_coordinates[:, 1:]
                stride = torch.tensor(occupancy.tensor_stride).flatten()
                dims = (self.frustum_dimensions.flatten() // stride.to(coords.device))

                valid_mask = ((coords >= 0) & (coords < dims)).all(-1)
                pred_occupancies.append(SparsePrune()(occupancy, valid_mask))

                occupancy_scores = (
                    Sigmoid()(occupancy).feature_tensor.squeeze(-1)
                )

                pruning_mask = (occupancy_scores > 0.5) & valid_mask
                valid_support = SparsePrune()(decoded, valid_mask)
                sparse_out = SparsePrune()(decoded, pruning_mask)
                if _is_empty_sparse(sparse_out):
                    # Preserve a valid sparse support set so downstream sparse
                    # heads still receive coordinates and produce a penalized
                    # prediction instead of failing on an empty tensor.
                    sparse_out = valid_support if not _is_empty_sparse(valid_support) else decoded

                if idx > 0:
                    sparse_out = sparse_cat_union(encoder_outputs[idx - 1], sparse_out)
                    coords = sparse_out.batch_indexed_coordinates[:, 1:]

                    stride = torch.tensor(occupancy.tensor_stride).flatten()
                    dims = (self.frustum_dimensions.flatten() // stride.to(coords.device))
                    valid_mask = ((coords >= 0) & (coords < dims)).all(-1)
                    decoder_inputs.append(SparsePrune()(sparse_out, valid_mask))
                else:
                    pred_geometry = self.geometry_head(sparse_out)
                    predicted_values = torch.clamp(
                        pred_geometry.feature_tensor, 0.0, self.truncation
                    )
                    pred_geometry = pred_geometry.replace(
                        batched_features=predicted_values
                    )

                    coords = pred_geometry.batch_indexed_coordinates[:, 1:]
                    valid_mask = ((coords >= 0) & (coords < self.frustum_dimensions)).all(-1)
                    pred_geometry = SparsePrune()(pred_geometry, valid_mask)

                queries = self.level_segm_query_projection[idx](segm_queries)
                segm_features = self.level_segm_embeddings[idx](sparse_out)
                pred_segm = self.forward_sparse_segm(segm_features, queries)

                # Validate coordinates accounting for tensor stride
                coords = pred_segm.batch_indexed_coordinates[:, 1:]
                valid_mask = ((coords >= 0) & (coords < dims)).all(-1)
                pred_segms.append(SparsePrune()(pred_segm, valid_mask))

            elif idx == 2:
                # level 64
                decoded = torch.cat([encoder_inputs[idx], decoded], dim=1)  # dense
                occupancy = self.level_occupancy_heads[idx](
                    decoded.permute(0, 2, 3, 4, 1)
                ).squeeze(-1)

                pred_occupancies.append(
                    occupancy.masked_fill(
                        ~frustum_mask.squeeze(1), -torch.inf
                    )
                )

                queries = self.level_segm_query_projection[idx](segm_queries)
                segm_features = self.level_segm_embeddings[idx](decoded)
                pred_segm = torch.einsum(
                    "bqc,bchwd->bqhwd", queries, segm_features
                )

                pred_segms.append(
                    pred_segm.masked_fill(~frustum_mask, -torch.inf)  # 1, 100, 64, 64, 64
                )

                pruning_mask = (occupancy.sigmoid() > 0.5) & frustum_mask.squeeze(1)
                coords = pruning_mask.nonzero()
                sparse_feats = decoded[
                    coords[:, 0], :, coords[:, 1], coords[:, 2], coords[:, 3]
                ]

                encoded = encoder_outputs[idx - 1]
                stride = encoded.tensor_stride
                batch_size = encoded.batch_size
                coords = coords.clone()
                batch_indices = coords[:, 0].int()
                spatial_coords = coords[:, 1:].int()
                # Ensure batch_size matches the encoded tensor
                offsets = offsets_from_batch_index(batch_indices)
                # If the computed offsets imply a smaller batch_size, pad it
                if offsets.numel() - 1 < batch_size:
                    padded_offsets = torch.zeros(
                        batch_size + 1, dtype=torch.int64, device=offsets.device
                    )
                    padded_offsets[:offsets.numel()] = offsets
                    offsets = padded_offsets

                sparse_out = Voxels(
                    batched_coordinates=IntCoords(
                        spatial_coords,
                        offsets=offsets,
                        tensor_stride=stride,
                    ),
                    batched_features=CatFeatures(sparse_feats, offsets=offsets),
                )

                combined = sparse_cat_union(encoded, sparse_out)
                decoder_inputs.append(combined)
            else:
                decoder_inputs.append(torch.cat([encoder_inputs[idx], decoded], dim=1))

        return {
            "pred_geometry": pred_geometry,
            "pred_occupancies": pred_occupancies,
            "pred_segms": pred_segms,
        }
