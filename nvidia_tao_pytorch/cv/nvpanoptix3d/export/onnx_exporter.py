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

"""ONNX Exporter for NVPanoptix3D Model."""

import os
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx import register_custom_op_symbolic
from typing import List, Optional
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.model_2d import MaskFormerModel
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.blocks import ProjectionBlock
from nvidia_tao_pytorch.cv.nvpanoptix3d.export.symbolic_funcs import (
    nvidia_msda,
    meshgrid_onnx,
    layer_norm_onnx,
    cartesian_prod_onnx,
    upsample_bicubic2d_aa,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d.export.utils import load_2d_model
from nvidia_tao_pytorch.cv.nvpanoptix3d.model.vggt.layers.attention import Attention, MemEffAttention


class MaskFormerModelWrapper(nn.Module):
    """Wrapper for MaskFormerModel to export to ONNX."""

    def __init__(self, model, projector):
        """Initialize wrapper.

        Args:
            model: MaskFormerModel instance
            projector: ProjectionBlock instance
        """
        super().__init__()
        self.model = model
        self.projector = projector

    def forward(self, images):
        """Forward pass for ONNX export.

        Args:
            images: Input images [B, 3, H, W] in uint8-range (0..255) or float

        Returns:
            tuple: Output tensors
        """
        processed_images, orig_pad_shape, _ = self.model.resize_img_backbone(images, return_shape=True)
        vggt_outputs = self.model.backbone(processed_images)
        multi_scale_features = vggt_outputs["multi_scale_features"]
        multi_scale_depth_features = vggt_outputs["multi_scale_depth_features"]
        preds, occupancy_preds = self.model.sem_seg_head(
            multi_scale_features,
            multi_scale_depth_features,
            use_occ_head=True
        )

        padded_out_h, padded_out_w = orig_pad_shape[0] // 2, orig_pad_shape[1] // 2
        pred_depths = vggt_outputs["depth"]
        pred_depths = F.interpolate(
            pred_depths,
            size=(padded_out_h, padded_out_w),
            mode="bilinear",
            align_corners=False,
        )
        mask_pred_results = preds["pred_masks"]
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(padded_out_h, padded_out_w),
            mode="bilinear",
            align_corners=False,
        )

        depth_features = preds["depth_features"]
        depth_features = self.projector(depth_features, preds["mask_features"].shape[-2:])
        encoder_features = torch.cat([preds["mask_features"], depth_features], dim=1)

        return (
            preds["pred_logits"],
            mask_pred_results,
            pred_depths,
            encoder_features,
            preds["segm_decoder_out"],
            occupancy_preds,
            preds["enc_features"][0],
            preds["enc_features"][1],
            preds["enc_features"][2],
            preds["pred_masks"],
            torch.tensor([orig_pad_shape[0], orig_pad_shape[1]], dtype=torch.int64),
        )


class ONNXExporter:
    """ONNX Exporter for 2D stage"""

    def __init__(self, opset_version: int = 17):
        """Initialize exporter.

        Args:
            opset_version: ONNX opset version (default 17 for CPU compatibility)
        """
        self.opset_version = opset_version
        self._custom_ops_registered = False

    def _register_custom_ops(self):
        """Register custom ONNX symbolic functions."""
        register_custom_op_symbolic(
            "nvidia::MultiscaleDeformableAttnPlugin_TRT",
            nvidia_msda,
            self.opset_version
        )
        register_custom_op_symbolic(
            "aten::meshgrid",
            meshgrid_onnx,
            self.opset_version
        )
        register_custom_op_symbolic(
            "aten::layer_norm",
            layer_norm_onnx,
            self.opset_version
        )
        register_custom_op_symbolic(
            "aten::cartesian_prod",
            cartesian_prod_onnx,
            self.opset_version
        )
        register_custom_op_symbolic(
            "aten::_upsample_bicubic2d_aa",
            upsample_bicubic2d_aa,
            self.opset_version
        )
        self._custom_ops_registered = True

    @staticmethod
    def _patch_xformers_attention_for_export(model: nn.Module) -> None:
        """Patch xformers attention modules for ONNX export tracing.

        xformers memory_efficient_attention relies on CUTLASS ops with
        c10::SymInt arguments that the legacy JIT tracer cannot export.
        For each MemEffAttention module in the model, this method replaces
        ``forward`` with ``Attention.forward`` so export uses the
        scaled-dot-product-attention path that is traceable.

        Args:
            model: Model tree whose MemEffAttention modules are patched in-place.
        """
        for module in model.modules():
            if isinstance(module, MemEffAttention):
                module.forward = Attention.forward.__get__(module, Attention)

    def export_model(
        self,
        model: nn.Module,
        onnx_file: str,
        dummy_input: torch.Tensor,
        input_names: List[str],
        output_names: List[str],
        batch_size: Optional[int] = None,
        do_constant_folding: bool = False,
        external_data: bool = True,
        verbose: bool = False,
    ) -> str:
        """Export model to ONNX.

        Args:
            model: PyTorch model to export
            onnx_file: Output ONNX file path
            dummy_input: Dummy input tensor for tracing
            batch_size: Batch size (None or -1 for dynamic)
            input_names: Names for input tensors
            output_names: Names for output tensors
            do_constant_folding: Whether to fold constants
            external_data: Whether to use external data due to large model weights
            verbose: Verbose output

        Returns:
            Path to exported ONNX file
        """
        # Patch attention implementation to avoid non-traceable xformers ops.
        self._patch_xformers_attention_for_export(model)

        # Register custom ops
        self._register_custom_ops()
        # Setup dynamic axes.
        dynamic_axes = None
        if batch_size is None or batch_size == -1:
            dynamic_axes = {}
            if input_names:
                for name in input_names:
                    dynamic_axes[name] = {0: "batch"}
            if output_names:
                for name in output_names:
                    dynamic_axes[name] = {0: "batch"}

        # Export
        with torch.no_grad():
            export_kwargs = dict(
                verbose=verbose,
                opset_version=self.opset_version,
                do_constant_folding=do_constant_folding,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                external_data=bool(external_data),
                custom_opsets={"nvidia": self.opset_version}
            )
            torch.onnx.export(model, dummy_input, onnx_file, **export_kwargs)

        return onnx_file

    @staticmethod
    def check_onnx(onnx_file):
        """Check onnx file.

        Args:
            onnx_file (str): path to ONNX file.
        """
        model = onnx.load(onnx_file)
        onnx.checker.check_model(model)


def export_2d_model(
    cfg,
    output_path: str,
    batch_size: int = 1,
    input_height: int = 256,
    input_width: int = 320,
    device: str = "cpu",
    opset_version: int = 17,
    verbose: bool = False,
    checkpoint_path: str = None,
) -> str:
    """Export 2D stage model (MaskFormerModel) to ONNX.

    Args:
        cfg: Model configuration
        output_path: Output ONNX file path
        batch_size: Batch size (use None for dynamic)
        input_height: Input image height
        input_width: Input image width
        device: Device ("cpu" or "cuda")
        opset_version: ONNX opset version
        verbose: Verbose output
        checkpoint_path: Path to checkpoint file
    Returns:
        Path to exported ONNX file
    """
    # Create model
    model = MaskFormerModelWrapper(MaskFormerModel(cfg).eval(), ProjectionBlock(256, 256)).to(device)
    model = load_2d_model(model, checkpoint_path, "cuda")

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, input_height, input_width, device=device)

    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Export 2D model
    exporter = ONNXExporter(opset_version=opset_version)
    onnx_file = exporter.export_model(
        model=model,
        onnx_file=output_path,
        dummy_input=dummy_input,
        batch_size=batch_size,
        input_names=["batch_inputs"],
        output_names=[
            "pred_logits", "pred_masks", "pred_depths",
            "encoder_features", "segm_decoder_out", "occupancy_preds",
            "enc_feat_0", "enc_feat_1", "enc_feat_2",
            "ori_pred_masks", "orig_pad_shape",
        ],
        verbose=verbose,
    )

    return onnx_file
