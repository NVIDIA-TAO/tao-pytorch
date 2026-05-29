# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse4D Head for Sparse4D."""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict, Any

from nvidia_tao_pytorch.cv.sparse4d.model.blocks import DeformableFeatureAggregation
from nvidia_tao_pytorch.cv.sparse4d.model.instance_bank import InstanceBank
from nvidia_tao_pytorch.cv.sparse4d.model.detection3d.detection3d_blocks import SparseBox3DEncoder
from nvidia_tao_pytorch.cv.sparse4d.model.detection3d.target import SparseBox3DTarget
from nvidia_tao_pytorch.cv.sparse4d.model.detection3d.detection3d_blocks import SparseBox3DKeyPointsGenerator
from nvidia_tao_pytorch.cv.sparse4d.model.detection3d.detection3d_blocks import SparseBox3DRefinementModule
from nvidia_tao_pytorch.cv.sparse4d.model.detection3d.decoder import SparseBox3DDecoder

from nvidia_tao_pytorch.cv.sparse4d.model.blocks import VisibilityNet, BNNeck
from nvidia_tao_pytorch.cv.sparse4d.model.blocks import AsymmetricFFN
from nvidia_tao_pytorch.cv.sparse4d.model.blocks import MultiheadAttention


def reduce_mean(tensor):
    """Obtain the mean of tensor on different GPUs.

    Args:
        tensor (Tensor): Tensor to be reduced.

    Returns:
        Tensor: Reduced tensor.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    tensor = tensor.clone()
    torch.distributed.all_reduce(tensor.div_(torch.distributed.get_world_size()), op=torch.distributed.ReduceOp.SUM)
    return tensor


class Sparse4DHead(nn.Module):
    """Sparse4D head."""

    def __init__(
        self,
        config: Dict[str, Any],
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        loss_cls=None,
        loss_reg=None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        return_feature: bool = False,
        num_fc_feat: int = 1,
        use_reid_sampling: bool = False,
        loss_id=None,
        bnneck=None,
        visibility_net=None,
        use_temporal_align=False,
        **kwargs,
    ):
        """Initialize Sparse4DHead.

        Args:
            instance_bank (InstanceBank): Instance bank.
            anchor_encoder (AnchorEncoder): Anchor encoder.
            graph_model (nn.Module): Graph model.
            norm_layer (nn.Module): Normalization layer.
            ffn (nn.Module): Feed-forward network.
            deformable_model (nn.Module): Deformable model.
            refine_layer (nn.Module): Refinement layer.
            num_decoder (int): Number of decoder layers.
            num_single_frame_decoder (int): Number of single frame decoder layers.
            temp_graph_model (nn.Module): Temporal graph model.
            loss_cls (nn.Module): Classification loss.
            loss_reg (nn.Module): Regression loss.
            decoder (nn.Module): Decoder.
            sampler (nn.Module): Sampler.
            gt_cls_key (str): Key for ground truth classification.
            gt_reg_key (str): Key for ground truth regression.
            reg_weights (List): List of regression weights.
            operation_order (List): Order of operations.
            cls_threshold_to_reg (float): Classification threshold to regression.
            dn_loss_weight (float): Denoising loss weight.
            decouple_attn (bool): Whether to decouple attention.
            return_feature (bool): Whether to return feature.
            num_fc_feat (int): Number of FC features.
            use_reid_sampling (bool): Whether to use reid sampling.
            loss_id (nn.Module): ID loss.
            bnneck (nn.Module): BN neck.
            visibility_net (nn.Module): Visibility net.
            use_temporal_align (bool): Whether to use temporal alignment.
            **kwargs: Additional keyword arguments.
        """
        super(Sparse4DHead, self).__init__()

        self.config = config
        self.head_cfg = config["model"]["head"]
        self.instance_bank_cfg = self.head_cfg["instance_bank"]
        self.anchor_encoder_cfg = self.head_cfg["anchor_encoder"]
        self.graph_model_cfg = self.head_cfg["graph_model"]
        self.norm_layer_cfg = self.head_cfg["norm_layer"]
        self.ffn_cfg = self.head_cfg["ffn"]
        self.deformable_cfg = self.head_cfg["deformable_model"]
        self.refine_cfg = self.head_cfg["refine_layer"]
        self.temp_graph_cfg = self.head_cfg["temp_graph_model"]
        self.decoder_cfg = self.head_cfg["decoder"]
        self.sampler_cfg = self.head_cfg["sampler"]
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn
        self.return_feature = return_feature
        self.num_fc_feat = num_fc_feat
        self.use_reid_sampling = use_reid_sampling
        self.use_temporal_align = use_temporal_align
        self.visibility_cfg = self.head_cfg["visibility_net"]
        self.bnneck_cfg = self.head_cfg["bnneck"]

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        # =========== build modules ===========
        self.instance_bank = InstanceBank(
            num_anchor=self.instance_bank_cfg["num_anchor"],
            embed_dims=self.instance_bank_cfg["embed_dims"],
            anchor=self.instance_bank_cfg["anchor"],
            anchor_handler=SparseBox3DKeyPointsGenerator(),
            num_temp_instances=self.instance_bank_cfg["num_temp_instances"],
            default_time_interval=self.instance_bank_cfg["default_time_interval"],
            confidence_decay=self.instance_bank_cfg["confidence_decay"],
            feat_grad=self.instance_bank_cfg["feat_grad"],
            use_temporal_align=self.instance_bank_cfg["use_temporal_align"],
        )
        self.anchor_encoder = SparseBox3DEncoder(
            embed_dims=self.anchor_encoder_cfg["embed_dims"],
            vel_dims=self.anchor_encoder_cfg["vel_dims"],
            mode=self.anchor_encoder_cfg["mode"],
            output_fc=self.anchor_encoder_cfg["output_fc"],
            in_loops=self.anchor_encoder_cfg["in_loops"],
            out_loops=self.anchor_encoder_cfg["out_loops"],
        )
        self.sampler = SparseBox3DTarget(
            num_dn_groups=self.sampler_cfg["num_dn_groups"],
            num_temp_dn_groups=self.sampler_cfg["num_temp_dn_groups"],
            dn_noise_scale=self.sampler_cfg["dn_noise_scale"],
            max_dn_gt=self.sampler_cfg["max_dn_gt"],
            add_neg_dn=self.sampler_cfg["add_neg_dn"],
            cls_weight=self.sampler_cfg["cls_weight"],
            box_weight=self.sampler_cfg["box_weight"],
            reg_weights=self.sampler_cfg["reg_weights"],
            use_temporal_align=self.sampler_cfg["use_temporal_align"],
            gt_assign_threshold=self.sampler_cfg["gt_assign_threshold"],
        )
        self.decoder = SparseBox3DDecoder(
            num_output=self.head_cfg["num_output"],
            score_threshold=self.decoder_cfg["score_threshold"],
        )

        if self.use_reid_sampling:
            self.visibility_net = VisibilityNet(
                embedding_dim=self.visibility_cfg["embedding_dim"],
                hidden_channels=self.visibility_cfg["hidden_channels"],
            )
            self.bnneck = BNNeck(
                feat_dim=self.bnneck_cfg["feat_dim"],
                num_ids=self.bnneck_cfg["num_ids"],
            )

        # Create layers based on operation order
        self.layers = nn.ModuleList()
        for op in self.operation_order:
            if op == "temp_gnn":
                self.layers.append(
                    MultiheadAttention(
                        embed_dims=self.temp_graph_cfg["embed_dims"],
                        num_heads=self.temp_graph_cfg["num_heads"],
                        dropout=self.temp_graph_cfg["dropout"],
                        batch_first=self.temp_graph_cfg["batch_first"],
                    )
                )
            elif op == "gnn":
                self.layers.append(
                    MultiheadAttention(
                        embed_dims=self.graph_model_cfg["embed_dims"],
                        num_heads=self.graph_model_cfg["num_heads"],
                        dropout=self.graph_model_cfg["dropout"],
                        batch_first=self.graph_model_cfg["batch_first"],
                    )
                )
            elif op == "norm":
                self.layers.append(
                    nn.LayerNorm(
                        normalized_shape=256,
                        eps=1e-5,
                    )
                )
            elif op == "ffn":
                self.layers.append(
                    AsymmetricFFN(
                        in_channels=self.ffn_cfg["in_channels"],
                        pre_norm=self.ffn_cfg["pre_norm"],
                        embed_dims=self.ffn_cfg["embed_dims"],
                        feedforward_channels=self.ffn_cfg["feedforward_channels"],
                        num_fcs=self.ffn_cfg["num_fcs"],
                        act_cfg=self.ffn_cfg["act_cfg"],
                        ffn_drop=self.ffn_cfg["ffn_drop"],
                    )
                )
            elif op == "deformable":
                self.layers.append(
                    DeformableFeatureAggregation(
                        embed_dims=self.deformable_cfg["embed_dims"],
                        num_groups=self.deformable_cfg["num_groups"],
                        num_levels=self.deformable_cfg["num_levels"],
                        num_cams=self.deformable_cfg["num_cams"],
                        max_num_cams=self.deformable_cfg["max_num_cams"],
                        proj_drop=self.deformable_cfg["proj_drop"],
                        attn_drop=self.deformable_cfg["attn_drop"],
                        kps_generator=SparseBox3DKeyPointsGenerator(
                            embed_dims=self.deformable_cfg["kps_generator"]["embed_dims"],
                            num_learnable_pts=self.deformable_cfg["kps_generator"]["num_learnable_pts"],
                            fix_scale=self.deformable_cfg["kps_generator"]["fix_scale"]
                        ),
                        use_deformable_func=self.deformable_cfg["use_deformable_func"],
                        use_camera_embed=self.deformable_cfg["use_camera_embed"],
                        residual_mode=self.deformable_cfg["residual_mode"],
                        reid_dims=self.config["model"]["head"]["reid_dims"],
                        use_reid_sampling=self.config["model"]["head"]["use_reid_sampling"],
                    )
                )
            elif op == "refine":
                self.layers.append(
                    SparseBox3DRefinementModule(
                        embed_dims=self.refine_cfg["embed_dims"],
                        num_cls=len(self.config["dataset"]["classes"]),
                        refine_yaw=self.refine_cfg["refine_yaw"],
                        with_quality_estimation=self.refine_cfg["with_quality_estimation"],
                    )
                )
            else:
                self.layers.append(None)

        self.embed_dims = self.instance_bank.embed_dims

        # overall feature dims
        self.feat_dims = self.embed_dims

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.fc_feat = nn.Identity()
        self.fc_reid = nn.Identity()

        self.init_weights()

    def init_weights(self):
        """Initialize the weights of the model."""
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        """Forward function."""
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        """Forward function."""
        if self.use_temporal_align:
            # reset gt_index_mapping for different groups
            group_indices_curr = [metas["img_metas"][i]["group_idx"] for i in range(len(metas["img_metas"]))]
            scene_indices_curr = [metas["img_metas"][i]["scene_idx"] for i in range(len(metas["img_metas"]))]
            group_indices_prev, scene_indices_prev = self.instance_bank.get_data_indices()

            # check if we need to reset gt_index_mapping for different groups
            flags_new_group = []  # need to be reset if flag is True
            if group_indices_prev is not None:
                for group_idx_prev, group_idx_curr in zip(group_indices_prev, group_indices_curr):
                    if group_idx_prev != -1 and group_idx_curr != -1:
                        flags_new_group.append(group_idx_prev != group_idx_curr)
                    else:
                        flags_new_group.append(False)
            else:
                flags_new_group = [False] * len(group_indices_curr)

            if scene_indices_prev is not None:
                for i, (scene_idx_prev, scene_idx_curr) in enumerate(zip(scene_indices_prev, scene_indices_curr)):
                    if scene_idx_prev != -1 and scene_idx_curr != -1:
                        # group_idx is changed OR scene_idx is changed
                        flags_new_group[i] = flags_new_group[i] or (scene_idx_prev != scene_idx_curr)

            self.instance_bank.reset_gt_index_mapping_by_data_indices(flags_new_group)
            self.instance_bank.set_data_indices(group_indices_curr, scene_indices_curr)

        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # ========= get instance info ============
        if (
            self.sampler.dn_metas is not None and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
            _,
        ) = self.instance_bank.get(
            batch_size, metas, dn_metas=self.sampler.dn_metas
        )

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            if "instance_id" in metas:
                gt_instance_id = metas["instance_id"]
            else:
                gt_instance_id = None

            dn_metas = self.sampler.get_dn_anchors(
                metas["gt_labels_3d"],
                metas["gt_bboxes_3d"],
                gt_instance_id=gt_instance_id,
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        anchor_embed = self.anchor_encoder(anchor)  # (bs, num_instance, 256)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # =================== forward the layers ====================
        prediction = []
        classification = []
        quality = []
        if self.use_reid_sampling:
            backbone_features = []
            bnn_features = []
            predicted_ids = []
            visibility_scores = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask
                    if temp_instance_feature is None
                    else None,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                output_deform = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
                instance_feature = output_deform["instance_feature"]

                if self.use_reid_sampling:
                    backbone_feature = output_deform["backbone_feature"]
                    visibility_score = self.visibility_net(backbone_feature)
                    softmax_weights = torch.softmax(visibility_score, dim=2).unsqueeze(-1)
                    weighted_backbone_feature = (backbone_feature * softmax_weights).sum(dim=2)
                    predicted_id, bnn_feature = self.bnneck(weighted_backbone_feature)

            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training or len(prediction) == self.num_single_frame_decoder - 1 or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if self.use_reid_sampling:
                    visibility_scores.append(visibility_score)
                    backbone_features.append(weighted_backbone_feature)
                    bnn_features.append(bnn_feature)
                    predicted_ids.append(predicted_id)
                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    if (
                        dn_metas is not None and self.sampler.num_temp_dn_groups > 0 and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]
            if self.use_reid_sampling:
                bnn_features = [x[:, :num_free_instance] for x in bnn_features]
                backbone_features = [x[:, :num_free_instance] for x in backbone_features]
                visibility_scores = [x[:, :num_free_instance] for x in visibility_scores]
                predicted_ids = [x[:, :num_free_instance] for x in predicted_ids]

            # cache dn_metas for temporal denoising
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )

        if self.use_reid_sampling:
            if self.training:
                output["reid_feature"] = backbone_features
                output["predicted_id"] = predicted_ids
                output["visibility_scores"] = visibility_scores

            else:
                output["reid_feature"] = bnn_features[-1]
                output["visibility_scores"] = visibility_scores[-1]

        # cache current instances for temporal modeling
        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        if not self.training:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
            if self.return_feature:
                output["instance_feature"] = instance_feature
        return output

    @torch.no_grad()
    def post_process(self, model_outs, output_idx=-1):
        """Post-process the model outputs."""
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("instance_feature"),
            model_outs.get("quality"),
            model_outs.get("reid_feature", None),
            model_outs.get("visibility_scores", None),
            output_idx=output_idx,
        )


def build_head(config: Dict[str, Any]) -> nn.Module:
    """Build a detection head according to the configuration.

    Args:
        config: Configuration dictionary for the head

    Returns:
        nn.Module: Detection head
    """
    head_config = config["model"]["head"]
    head_type = head_config["type"]

    if head_type == 'sparse4d':

        # Create head with the constructed modules
        head = Sparse4DHead(
            config=config,
            num_decoder=head_config["num_decoder"],
            num_single_frame_decoder=head_config["num_single_frame_decoder"],
            reg_weights=head_config["reg_weights"],
            operation_order=head_config["operation_order"],
            cls_threshold_to_reg=head_config["cls_threshold_to_reg"],
            decouple_attn=head_config["decouple_attn"],
            return_feature=head_config["return_feature"],
            use_reid_sampling=head_config["use_reid_sampling"],
            bnneck=head_config["bnneck"],
            visibility_net=head_config["visibility_net"],
        )

        return head
    else:
        raise ValueError(f"Unsupported head type: {head_type}")
