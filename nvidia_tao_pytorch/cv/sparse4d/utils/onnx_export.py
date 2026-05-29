# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Generates TRT compatible Sparse4D onnx model. """

import onnx
import torch
from torch import nn

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.sparse4d.model.blocks import DeformableFeatureAggregation
from nvidia_tao_pytorch.cv.sparse4d.model.instance_bank import topk
from nvidia_tao_pytorch.cv.sparse4d.model.detection3d.detection3d_blocks import SparseBox3DRefinementModule, SparseBox3DKeyPointsGenerator
from nvidia_tao_pytorch.cv.sparse4d.model.box3d import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX
from nvidia_tao_pytorch.cv.sparse4d.model.ops.deformable_aggregation import DeformableAggregationFunction, feature_maps_format


def deformable_feature_aggregation_project_points(key_points, projection_mat, image_wh=None):
    """Project 3D points to 2D points using a projection matrix."""
    pts_extend = torch.cat(
        [key_points, torch.ones_like(key_points[..., :1])], dim=-1
    )
    points_2d = torch.matmul(
        projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
    )
    points_2d = points_2d.view(*points_2d.shape[:-1])
    points_2d = points_2d[..., :2] / torch.clamp(
        points_2d[..., 2:3], min=1e-5
    )
    if image_wh is not None:
        points_2d = points_2d / image_wh[:, :, None, None]
    return points_2d


def symbolic_for_deformable_aggregation_function(
    g,
    mc_ms_feat,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights
):
    """Symbolic function for deformable aggregation."""
    return g.op("nv::MSDA", mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights)


class NoJitTrace(torch.no_grad):
    """No JIT trace."""

    def __enter__(self):
        """Enter the context."""
        super().__enter__()
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        """Exit the context."""
        super().__exit__(*args)
        torch._C._set_tracing_state(self.state)
        self.state = None


class Sparse4DExporter(nn.Module):
    """Sparse4D exporter."""

    def __init__(self, model, return_dict=False):
        """Initialize the Sparse4D exporter."""
        super().__init__()
        self.model = model.model
        self.return_dict = return_dict
        self.input_names = [
            'img',
            'timestamp',
            'projection_mat',
            'image_wh',
            'input_cached_feature',
            'input_cached_anchor',
            'prev_exists',
            'interval_mask'
        ]
        self.output_names = [
            "classification1",
            "classification2",
            "prediction1",
            "prediction2",
            "prediction3",
            "prediction4",
            "prediction5",
            "prediction6",
            "quality1",
            "quality2",
            "output_cached_feature",
            "output_cached_anchor"
        ]
        self.dynamic_axes = {
            'img': {0: "batch_size", 1: "num_cams"},
            "timestamp": {0: "batch_size"},
            "projection_mat": {0: "batch_size", 1: "num_cams"},
            "image_wh": {0: "batch_size", 1: "num_cams"},
            "input_cached_feature": {0: "batch_size"},
            "input_cached_anchor": {0: "batch_size"},
            "prev_exists": {0: "batch_size"},
            "interval_mask": {0: "batch_size"}
        }

    def memorybank_update(self, bank_obj, prev_exists, cached_feature, cached_anchor, instance_feature_old, anchor_old, confidence, interval_mask):
        """Update the memory bank."""
        N = bank_obj.num_anchor - bank_obj.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor), _ = topk(
            confidence, N, instance_feature_old, anchor_old
        )
        selected_feature = torch.cat(
            [cached_feature, selected_feature], dim=1
        )
        selected_anchor = torch.cat(
            [cached_anchor, selected_anchor], dim=1
        )
        instance_feature = torch.where(
            interval_mask, selected_feature, instance_feature_old
        )
        anchor = torch.where(interval_mask, selected_anchor, anchor_old)
        return instance_feature * prev_exists + instance_feature_old * (1 - prev_exists), anchor * prev_exists + anchor_old * (1 - prev_exists)

    def graph_model(
        self,
        head,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        """Graph model."""
        if head.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = head.fc_before(value)
        return head.fc_after(
            head.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def sparse_head_forward(
        self,
        head,
        feature_maps,
        timestamp,
        projection_mat,
        image_wh,
        cached_feature,
        cached_anchor,
        prev_exists,
        interval_mask
    ):
        """Sparse head forward."""
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]

        with NoJitTrace():
            batch_size = feature_maps[0].shape[0]
            anchor = torch.tile(head.instance_bank.anchor[None], (batch_size, 1, 1))
            instance_feature = torch.tile(head.instance_bank.instance_feature[None], (batch_size, 1, 1))
            anchor_embed = head.anchor_encoder(anchor)
            time_interval = head.instance_bank.time_interval

        temp_instance_feature = cached_feature
        temp_anchor_embed = head.anchor_encoder(cached_anchor) if cached_anchor is not None else None
        attn_mask = None
        prediction = []
        classification = []
        quality = []
        metas_for_deformable = {
            "projection_mat": projection_mat,
            "image_wh": image_wh,
            "timestamp": timestamp,
        }
        for i, op in enumerate(head.operation_order):
            if head.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    head,
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
                    head,
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = head.layers[i](instance_feature)
            elif op == "deformable":
                # Call the deformable layer and get the output dictionary
                output_deform = head.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas_for_deformable
                )
                # Extract the feature tensor from the dictionary
                instance_feature = output_deform["instance_feature"]
            elif op == "refine":
                anchor, cls, qt = head.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        head.training or len(prediction) == head.num_single_frame_decoder - 1 or i == len(head.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if len(prediction) == head.num_single_frame_decoder:
                    instance_feature, anchor = self.memorybank_update(
                        head.instance_bank,
                        prev_exists,
                        cached_feature,
                        cached_anchor,
                        instance_feature,
                        anchor,
                        cls,
                        interval_mask
                    )
                if i != len(head.operation_order) - 1:
                    anchor_embed = head.anchor_encoder(anchor)
                if (len(prediction) > head.num_single_frame_decoder and temp_anchor_embed is not None):
                    temp_anchor_embed = anchor_embed[:, : head.instance_bank.num_temp_instances] * prev_exists
            else:
                raise NotImplementedError(f"{op} is not supported.")

        classification = [item for item in classification if item is not None]
        prediction = [item for item in prediction if item is not None]
        quality = [item for item in quality if item is not None]
        return classification, prediction, quality, instance_feature, anchor

    def simple_test(self, feature_maps, timestamp, projection_mat, image_wh, cached_feature, cached_anchor, prev_exists, interval_mask):
        """Simple test."""
        model_outs = self.sparse_head_forward(self.model.head, feature_maps, timestamp, projection_mat, image_wh, cached_feature, cached_anchor, prev_exists, interval_mask)
        return model_outs

    @staticmethod
    def sparse_box_3d_refinement_module_forward(
        module,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        """Sparse box 3D refinement module forward."""
        feature = instance_feature + anchor_embed
        output = module.layers(feature)
        if module.refine_yaw:
            output = torch.cat([output[..., :8] + anchor[..., :8], output[..., 8:]], dim=-1)
        else:
            output = torch.cat([output[..., :6] + anchor[..., :6], output[..., 6:]], dim=-1)

        if module.normalize_yaw:
            output = torch.cat([
                output[..., :SIN_YAW],
                torch.nn.functional.normalize(output[..., [SIN_YAW, COS_YAW]], dim=-1),
                output[..., COS_YAW + 1:]
            ], dim=-1)
        if module.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            translation = torch.transpose(output[..., VX:], 0, -1)
            velocity = torch.transpose(translation / time_interval, 0, -1)

            output = torch.cat([
                output[..., :VX],
                velocity
            ], dim=-1)

        if return_cls:
            assert module.with_cls_branch, "Without classification layers !!!"
            cls = module.cls_layers(instance_feature)
        else:
            cls = None
        if return_cls and module.with_quality_estimation:
            quality = module.quality_layers(feature)
        else:
            quality = None
        return output, cls, quality

    @staticmethod
    def sparse_box_3d_key_points_generator_forward(
        generator,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        """Sparse box 3D key points generator forward."""
        bs, num_anchor = anchor.shape[:2]
        size = anchor[..., None, [W, L, H]].exp()
        generator.fix_scale = torch.tensor(generator.fix_scale, device=size.device, dtype=size.dtype)
        key_points = generator.fix_scale * size
        if generator.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                generator.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, generator.num_learnable_pts, 3)
                .sigmoid() - 0.5
            )
            key_points = torch.cat([key_points, learnable_scale * size], dim=-2)

        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1
        rotation_mat = rotation_mat[:, :, None]

        key_points = torch.matmul(
            rotation_mat, key_points[..., None]
        )
        key_points = key_points.view(*key_points.shape[:-1])
        key_points = key_points + anchor[..., None, [X, Y, Z]]
        return key_points

    @staticmethod
    def deformable_feature_aggregation_project_points(_unused_self, key_points, projection_mat, image_wh=None):
        """Project 3D points to 2D points using a projection matrix."""
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        )
        points_2d = points_2d.view(*points_2d.shape[:-1])
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def symbolic_for_deformable_aggregation_function(
        g,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights
    ):
        """Symbolic function for deformable aggregation."""
        return g.op("nv::MSDA", mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights)

    def forward(self, img, timestamp, projection_mat, image_wh, input_cached_feature, input_cached_anchor,  prev_exists,  interval_mask):
        """Forward pass."""
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        img = self.model.grid_mask(img)

        feature_maps = self.model.img_backbone.forward_feature_pyramid(img)
        if self.model.img_neck is not None:
            feature_maps = list(self.model.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if self.model.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)

        outputs = self.simple_test(feature_maps, timestamp, projection_mat, image_wh, input_cached_feature, input_cached_anchor, prev_exists, interval_mask)
        if self.return_dict:
            classification, prediction, quality, instance_feature, anchor = outputs
            outputs = dict(
                classification1=classification[0],
                classification2=classification[-1],
                prediction1=prediction[0],
                prediction2=prediction[1],
                prediction3=prediction[2],
                prediction4=prediction[3],
                prediction5=prediction[4],
                prediction6=prediction[5],
                quality1=quality[0],
                quality2=quality[1],
                output_cached_feature=instance_feature,
                output_cached_anchor=anchor,
            )
        return outputs

    def export_model(self, experiment_config, model, output_file):
        """Export the model."""
        # Update the forward methods with the static methods from Sparse4DExporter
        SparseBox3DKeyPointsGenerator.forward = Sparse4DExporter.sparse_box_3d_key_points_generator_forward
        SparseBox3DRefinementModule.forward = Sparse4DExporter.sparse_box_3d_refinement_module_forward
        DeformableAggregationFunction.symbolic = Sparse4DExporter.symbolic_for_deformable_aggregation_function
        DeformableFeatureAggregation.project_points = Sparse4DExporter.deformable_feature_aggregation_project_points

        B = 1  # Only support batch size 1 indicating single BEV frame.
        NUM_CAMS = 20  # Dummy number of cameras.
        device = "cuda"
        dtype = torch.float32
        width = experiment_config["model"]["input_shape"][0]
        height = experiment_config["model"]["input_shape"][1]
        logging.info(f"Using width: {width} & height: {height}")

        img = torch.randn(B, NUM_CAMS, 3, height, width, dtype=dtype).to(device)
        cached_anchor = torch.zeros(B, 600, 11, dtype=dtype).to(device)
        cached_feature = torch.zeros(B, 600, 256, dtype=dtype).to(device)
        timestamp = torch.full((B,), 0.5, dtype=dtype).to(device)
        prev_exists = torch.full((B,), 0, dtype=dtype).to(device)
        projection_mat = torch.randn(B, NUM_CAMS, 4, 4, dtype=dtype, device=device)
        interval_mask = torch.zeros(B, 1, 1, dtype=torch.bool, device=device)
        image_wh = torch.tensor([[height, width]], dtype=torch.float32, device='cuda:0').repeat(1, NUM_CAMS, 1)

        inputs = (img, timestamp, projection_mat, image_wh, cached_feature, cached_anchor, prev_exists, interval_mask)

        sparse4d = Sparse4DExporter(model, return_dict=True).cuda().eval()

        with torch.no_grad():
            torch.onnx.export(
                sparse4d,
                inputs,
                output_file,
                input_names=self.input_names,
                output_names=self.output_names,
                dynamic_axes=self.dynamic_axes,
                opset_version=17,
                verbose=True,
                dynamo=False,
            )

    def check_onnx(self, onnx_file):
        """Check onnx file.

        Args:
            onnx_file (str): path to ONNX file.
        """
        model = onnx.load(onnx_file)
        onnx.checker.check_model(model)
