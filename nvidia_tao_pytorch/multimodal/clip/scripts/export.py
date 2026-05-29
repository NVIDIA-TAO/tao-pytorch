# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export CLIP model to ONNX."""

import math
import os
from typing import Optional, Tuple

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.onnx import symbolic_helper

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import (
    TLTPyTorchCookbook,
)
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.utilities import encrypt_onnx
from nvidia_tao_pytorch.core.tlt_logging import logging

from nvidia_tao_pytorch.config.clip.default_config import (
    CLIPExperimentConfig as ExperimentConfig,
)
from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import CLIPPlModel
from nvidia_tao_pytorch.multimodal.clip.model.tokenizers import save_tokenizer
from nvidia_tao_pytorch.multimodal.clip.utils.utils import (
    register_checkpoint_safe_globals,
)


# Register custom ONNX symbolic for anti-aliased bilinear upsample, which
# PyTorch uses internally (e.g. in HuggingFace SigLIP2 NaFlex) but has no
# built-in ONNX exporter mapping.  We map it to the standard ONNX Resize
# op with mode=linear, dropping the anti-alias flag (negligible at inference).
def _onnx_upsample_bilinear2d_aa(g, inp, output_size, align_corners, *args):
    """ONNX symbolic for aten::_upsample_bilinear2d_aa.

    Maps anti-aliased bilinear upsample to standard ONNX Resize op.
    The anti-alias flag is implicit in the op name and is dropped for ONNX
    (negligible impact at inference).

    Accepts *args for remaining scale parameters since the JIT graph may
    pass them in varying forms depending on the PyTorch version.
    """
    align_corners_i = symbolic_helper._maybe_get_const(align_corners, "b")
    coord_mode = "align_corners" if align_corners_i else "asymmetric"
    empty_tensor = g.op(
        "Constant", value_t=torch.tensor([], dtype=torch.float32)
    )
    return g.op(
        "Resize", inp, empty_tensor, empty_tensor, output_size,
        mode_s="linear",
        coordinate_transformation_mode_s=coord_mode,
    )


torch.onnx.register_custom_op_symbolic(
    "aten::_upsample_bilinear2d_aa", _onnx_upsample_bilinear2d_aa, 17
)


class ExportFriendlyMHA(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention with ONNX-friendly ops.

    PyTorch's nn.MultiheadAttention uses a fused _native_multi_head_attention
    kernel that has no ONNX symbolic.  This module performs the identical
    computation using standard linear, reshape, softmax, and matmul ops that
    the ONNX exporter fully supports.

    Use ``_replace_mha_for_export`` to swap all nn.MultiheadAttention modules
    in a model before calling ``torch.onnx.export``.
    """

    def __init__(self, mha: nn.MultiheadAttention):
        """Initialize from existing nn.MultiheadAttention, sharing weights."""
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = mha.embed_dim // mha.num_heads
        self.batch_first = mha.batch_first

        # Share weight tensors (no copy)
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using decomposed multi-head attention ops.

        Handles both self-attention (query == key == value) and cross/pooler
        attention (query differs from key/value) by detecting sequence length
        mismatch and using separate projections when needed.
        """
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        seq_len, batch_size, _ = query.shape
        k_len = key.shape[0]

        # Detect if this is self-attention or cross/pooler attention
        is_self_attention = (seq_len == k_len)

        if is_self_attention:
            # Self-attention: single projection for Q, K, V from query
            qkv = F.linear(  # pylint: disable=not-callable
                query, self.in_proj_weight, self.in_proj_bias
            )
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # Cross/pooler attention: separate projections for Q vs K/V
            # Split the combined in_proj_weight into Q, K, V components
            q_proj_weight, k_proj_weight, v_proj_weight = self.in_proj_weight.chunk(3, dim=0)
            if self.in_proj_bias is not None:
                q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)
            else:
                q_bias, k_bias, v_bias = None, None, None

            q = F.linear(query, q_proj_weight, q_bias)  # pylint: disable=not-callable
            k = F.linear(key, k_proj_weight, k_bias)  # pylint: disable=not-callable
            v = F.linear(value, v_proj_weight, v_bias)  # pylint: disable=not-callable

        # Reshape for multi-head: (S, B, D) -> (S, B, H, Dh) -> (B, H, S, Dh)
        q = q.reshape(seq_len, batch_size, self.num_heads, self.head_dim)
        q = q.permute(1, 2, 0, 3)
        k = k.reshape(k_len, batch_size, self.num_heads, self.head_dim)
        k = k.permute(1, 2, 0, 3)
        v = v.reshape(k_len, batch_size, self.num_heads, self.head_dim)
        v = v.permute(1, 2, 0, 3)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)

        # (B, H, S, Dh) -> (S, B, D)
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(
            seq_len, batch_size, self.embed_dim
        )

        # Output projection
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, None


def _replace_mha_for_export(model: nn.Module) -> None:
    """Replace all nn.MultiheadAttention modules with ExportFriendlyMHA.

    Walks the module tree and swaps each nn.MultiheadAttention with an
    ExportFriendlyMHA that shares the same weights but uses only
    ONNX-exportable ops.

    Parameters
    ----------
    model : nn.Module
        Model to prepare for ONNX export (modified in-place).
    """
    for _, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                setattr(module, attr_name, ExportFriendlyMHA(child))


# Valid encoder types for export (aligned with CLIPExportConfig.encoder_type)
VALID_ENCODER_TYPES = {'combined', 'separate'}


class CLIPVisionEncoder(nn.Module):
    """Wrapper to export only the vision encoder of CLIP.

    This module wraps the CLIP model to export only the vision encoder
    component, which produces image embeddings from input images.

    Parameters
    ----------
    clip_model : nn.Module
        The full CLIP model containing vision and text encoders.
    """

    def __init__(self, clip_model: nn.Module):
        """Initialize CLIPVisionEncoder."""
        super().__init__()
        self.model = clip_model

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through vision encoder.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (image_embedding, logit_scale, logit_bias) where embedding has
            shape (B, D) and logit_scale/logit_bias are scalar tensors.
        """
        output = self.model(image=image)
        if isinstance(output, dict):
            image_features = output["image_features"]
        else:
            image_features = output[0]
        return image_features, self.model.logit_scale.exp(), self.model.logit_bias


class CLIPTextEncoder(nn.Module):
    """Wrapper to export only the text encoder of CLIP.

    This module wraps the CLIP model to export only the text encoder
    component, which produces text embeddings from tokenized text.

    Parameters
    ----------
    clip_model : nn.Module
        The full CLIP model containing vision and text encoders.
    """

    def __init__(self, clip_model: nn.Module):
        """Initialize CLIPTextEncoder."""
        super().__init__()
        self.model = clip_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through text encoder.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenized text input IDs of shape (B, seq_len).
        attention_mask : torch.Tensor
            Attention mask of shape (B, seq_len). Accepted for backward
            compatibility but ignored — all-ones is used internally.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (text_embedding, logit_scale, logit_bias) where embedding has
            shape (B, D) and logit_scale/logit_bias are scalar tensors.
        """
        # Ignore user-provided attention_mask: SigLIP2 requires all-ones,
        # and CLIP/OpenCLIP adapters discard it anyway.
        # Tie attention_mask into input_ids via `+ mask * 0` so the ONNX tracer
        # keeps it as a graph input for backward compatibility. This works even
        # when the adapter only consumes input_ids (OpenCLIP/CLIP path).
        input_ids = input_ids + (attention_mask * 0).to(input_ids.dtype)
        text_input = {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids)}
        output = self.model(text=text_input)
        if isinstance(output, dict):
            text_features = output["text_features"]
        else:
            text_features = output[1]
        return text_features, self.model.logit_scale.exp(), self.model.logit_bias


class CLIPCombinedEncoder(nn.Module):
    """Wrapper to export both vision and text encoders as a single ONNX model.

    Takes flat tensor inputs (image, input_ids, attention_mask) so that ONNX
    tracing works without dict construction during trace. Internally calls
    the model's combined forward path which returns both embeddings and the
    learned logit scale.

    Parameters
    ----------
    clip_model : nn.Module
        The full CLIP model containing vision and text encoders.
    """

    def __init__(self, clip_model: nn.Module):
        """Initialize CLIPCombinedEncoder."""
        super().__init__()
        self.model = clip_model

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through both encoders.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape (B, C, H, W).
        input_ids : torch.Tensor
            Tokenized text input IDs of shape (B, seq_len).
        attention_mask : torch.Tensor
            Attention mask of shape (B, seq_len). Accepted for backward
            compatibility but ignored — all-ones is used internally.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (image_embedding, text_embedding, logit_scale, logit_bias) where
            embeddings have shape (B, D) and logit_scale/logit_bias are
            scalar tensors.
        """
        # Ignore user-provided attention_mask (see CLIPTextEncoder for rationale)
        input_ids = input_ids + (attention_mask * 0).to(input_ids.dtype)
        text_input = {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids)}
        image_features, text_features, logit_scale, logit_bias = self.model(
            image=image, text=text_input
        )
        return image_features, text_features, logit_scale, logit_bias


def export_single_encoder(
    pl_model: CLIPPlModel,
    encoder_type: str,
    export_config,
    experiment_config: ExperimentConfig,
    output_file: str,
    device: str
) -> str:
    """Export a single encoder (vision or text) to ONNX.

    Parameters
    ----------
    pl_model : CLIPPlModel
        Loaded PyTorch Lightning model.
    encoder_type : str
        Type of encoder to export: 'vision' or 'text'.
    export_config : ExportExpConfig
        Export configuration.
    experiment_config : ExperimentConfig
        Full experiment configuration.
    output_file : str
        Output ONNX file path.
    device : str
        Device to use ('cpu' or 'cuda').

    Returns
    -------
    str
        Path to the exported ONNX file.
    """
    opset_version = export_config.opset_version
    batch_size = export_config.batch_size
    key = experiment_config.encryption_key

    if batch_size is None or batch_size == -1:
        input_batch_size = 1
        is_dynamic = True
    else:
        input_batch_size = batch_size
        is_dynamic = False

    if encoder_type == 'vision':
        encoder = CLIPVisionEncoder(pl_model.model)
        input_channel = export_config.input_channel
        input_width = export_config.input_width
        input_height = export_config.input_height
        input_shape = [input_channel, input_height, input_width]

        dummy_input = torch.randn(
            input_batch_size, *input_shape, device=device)
        input_names = ['image']
        output_names = ['image_embedding', 'logit_scale', 'logit_bias']

        if is_dynamic:
            dynamic_axes = {
                'image': {0: 'batch_size'},
                'image_embedding': {0: 'batch_size'}
            }
        else:
            dynamic_axes = None

        logging.info(
            f"Exporting vision encoder with input shape: "
            f"{[input_batch_size] + input_shape}"
        )

    else:  # text encoder
        encoder = CLIPTextEncoder(pl_model.model)
        # Infer sequence length from the model's tokenizer rather than
        # hardcoding, since different text encoders have different context
        # lengths (e.g., SigLIP2: 64, OpenCLIP/DFN CLIP: 77)
        dummy_tokens = pl_model.tokenizer(["test"])[0]
        seq_length = list(dummy_tokens.values())[0].shape[-1]

        dummy_input_ids = torch.zeros(
            input_batch_size, seq_length, dtype=torch.long, device=device
        )
        dummy_attention_mask = torch.ones(
            input_batch_size, seq_length, dtype=torch.long, device=device
        )
        dummy_input = (dummy_input_ids, dummy_attention_mask)
        input_names = ['input_ids', 'attention_mask']
        output_names = ['text_embedding', 'logit_scale', 'logit_bias']

        if is_dynamic:
            dynamic_axes = {
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'text_embedding': {0: 'batch_size'}
            }
        else:
            dynamic_axes = None

        logging.info(
            f"Exporting text encoder with sequence length: {seq_length}")

    encoder.eval()
    if device == 'cuda':
        encoder.cuda()

    # Determine actual output file (handle encryption)
    if output_file.endswith('.etlt'):
        tmp_onnx_file = output_file.replace('.etlt', '.onnx')
    else:
        tmp_onnx_file = output_file

    logging.info(
        f"Exporting {encoder_type} encoder to ONNX with opset "
        f"version {opset_version}"
    )

    # Estimate model size to determine if external data will be needed
    param_size_bytes = sum(
        p.numel() * p.element_size() for p in encoder.parameters()
    )
    size_gb = param_size_bytes / (1024 ** 3)
    use_external_data = size_gb > 1.9

    if use_external_data:
        external_data_path = (
            os.path.splitext(tmp_onnx_file)[0] + "_weights.bin"
        )
        external_data_name = os.path.basename(external_data_path)
        logging.warning(
            f"Model size (~{size_gb:.2f} GB) exceeds 2GB ONNX protobuf "
            f"limit. Weights will be stored in external file: "
            f"{external_data_name}. Both the .onnx file and "
            f"{external_data_name} are required for inference."
        )
    else:
        logging.info(
            f"Model size (~{size_gb:.2f} GB) fits in single ONNX file."
        )

    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            dummy_input,
            tmp_onnx_file,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=export_config.verbose,
            dynamo=False,
        )

    # If model is large, consolidate external data into a single file
    if use_external_data:
        logging.info(f"Consolidating external data into: {external_data_name}")
        onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)
        onnx.save_model(
            onnx_model,
            tmp_onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_name,
            size_threshold=0,
        )
        logging.info(
            f"ONNX export completed: {tmp_onnx_file} + {external_data_name}"
        )
    else:
        logging.info(f"ONNX export completed: {tmp_onnx_file}")

    # Verify ONNX model
    try:
        if use_external_data:
            onnx.checker.check_model(tmp_onnx_file)
        else:
            onnx_model = onnx.load(tmp_onnx_file)
            onnx.checker.check_model(onnx_model)
        logging.info(
            f"ONNX model validation passed for {encoder_type} encoder"
        )
    except Exception as e:
        logging.warning(f"ONNX model validation failed: {e}")

    # Handle encryption if needed
    if output_file.endswith('.etlt') and key:
        encrypt_onnx(
            tmp_file_name=tmp_onnx_file,
            output_file_name=output_file,
            key=key
        )
        os.remove(tmp_onnx_file)
        logging.info(f"Encrypted ONNX file stored at {output_file}")
        return output_file
    else:
        logging.info(f"ONNX file stored at {tmp_onnx_file}")
        return tmp_onnx_file


def export_combined_encoder(
    pl_model: CLIPPlModel,
    export_config,
    experiment_config: ExperimentConfig,
    output_file: str,
    device: str
) -> str:
    """Export both vision and text encoders as a single combined ONNX model.

    The combined model takes (image, input_ids, attention_mask) and produces
    (image_embedding, text_embedding, logit_scale, logit_bias) in one graph.

    Parameters
    ----------
    pl_model : CLIPPlModel
        Loaded PyTorch Lightning model.
    export_config : CLIPExportConfig
        Export configuration.
    experiment_config : ExperimentConfig
        Full experiment configuration.
    output_file : str
        Output ONNX file path.
    device : str
        Device to use ('cpu' or 'cuda').

    Returns
    -------
    str
        Path to the exported ONNX file.
    """
    opset_version = export_config.opset_version
    batch_size = export_config.batch_size
    key = experiment_config.encryption_key

    if batch_size is None or batch_size == -1:
        input_batch_size = 1
        is_dynamic = True
    else:
        input_batch_size = batch_size
        is_dynamic = False

    # Build combined encoder wrapper
    encoder = CLIPCombinedEncoder(pl_model.model)
    encoder.eval()
    if device == 'cuda':
        encoder.cuda()

    # Vision dummy input
    input_channel = export_config.input_channel
    input_width = export_config.input_width
    input_height = export_config.input_height
    input_shape = [input_channel, input_height, input_width]
    dummy_image = torch.randn(input_batch_size, *input_shape, device=device)

    # Text dummy input -- infer sequence length from the model's tokenizer
    dummy_tokens = pl_model.tokenizer(["test"])[0]
    seq_length = list(dummy_tokens.values())[0].shape[-1]
    dummy_input_ids = torch.zeros(
        input_batch_size, seq_length, dtype=torch.long, device=device
    )
    dummy_attention_mask = torch.ones(
        input_batch_size, seq_length, dtype=torch.long, device=device
    )

    dummy_input = (dummy_image, dummy_input_ids, dummy_attention_mask)

    input_names = ['image', 'input_ids', 'attention_mask']
    output_names = ['image_embedding', 'text_embedding', 'logit_scale', 'logit_bias']

    if is_dynamic:
        dynamic_axes = {
            'image': {0: 'batch_size'},
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'image_embedding': {0: 'batch_size'},
            'text_embedding': {0: 'batch_size'},
        }
    else:
        dynamic_axes = None

    logging.info(
        f"Exporting combined encoder with image shape "
        f"{[input_batch_size] + input_shape}, sequence length {seq_length}"
    )

    # Determine actual output file (handle encryption)
    if output_file.endswith('.etlt'):
        tmp_onnx_file = output_file.replace('.etlt', '.onnx')
    else:
        tmp_onnx_file = output_file

    logging.info(
        f"Exporting combined encoder to ONNX with opset version "
        f"{opset_version}"
    )

    # Estimate model size to determine if external data will be needed
    # Protobuf has a 2GB limit - PyTorch will automatically use external data
    param_size_bytes = sum(
        p.numel() * p.element_size() for p in encoder.parameters()
    )
    size_gb = param_size_bytes / (1024 ** 3)
    use_external_data = size_gb > 1.9  # Use 1.9GB threshold for safety margin

    if use_external_data:
        external_data_path = os.path.splitext(tmp_onnx_file)[0] + "_weights.bin"
        external_data_name = os.path.basename(external_data_path)
        logging.warning(
            f"Model size (~{size_gb:.2f} GB) exceeds 2GB ONNX protobuf limit. "
            f"Weights will be stored in external file: {external_data_name}. "
            f"Both the .onnx file and {external_data_name} are required for inference."
        )
    else:
        logging.info(f"Model size (~{size_gb:.2f} GB) fits in single ONNX file.")

    with torch.no_grad():
        torch.onnx.export(
            encoder,
            dummy_input,
            tmp_onnx_file,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=export_config.verbose,
            dynamo=False,
        )

    # If model is large, consolidate external data into a single file
    if use_external_data:
        logging.info(f"Consolidating external data into: {external_data_name}")

        # Load model with external data from scattered files
        onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)

        # Save with all tensors in a single external file
        onnx.save_model(
            onnx_model,
            tmp_onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_name,
            size_threshold=0,  # Save all tensors externally
        )

        logging.info(
            f"ONNX export completed: {tmp_onnx_file} + {external_data_name}"
        )
    else:
        logging.info(f"ONNX export completed: {tmp_onnx_file}")

    # Verify ONNX model
    try:
        if use_external_data:
            # For large models, check using file path to avoid loading into memory
            onnx.checker.check_model(tmp_onnx_file)
        else:
            onnx_model = onnx.load(tmp_onnx_file)
            onnx.checker.check_model(onnx_model)
        logging.info("ONNX model validation passed for combined encoder")
    except Exception as e:
        logging.warning(f"ONNX model validation failed: {e}")

    # Handle encryption if needed
    if output_file.endswith('.etlt') and key:
        encrypt_onnx(
            tmp_file_name=tmp_onnx_file,
            output_file_name=output_file,
            key=key
        )
        os.remove(tmp_onnx_file)
        logging.info(f"Encrypted ONNX file stored at {output_file}")
        return output_file
    else:
        logging.info(f"ONNX file stored at {tmp_onnx_file}")
        return tmp_onnx_file


def run_export(experiment_config: ExperimentConfig) -> None:
    """Run ONNX export for CLIP model.

    The encoder_type config option controls the export mode:
    - 'combined': Single ONNX with both vision and text encoders
    - 'separate': Two ONNX files (vision + text)

    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration containing export settings.
    """
    register_checkpoint_safe_globals()
    export_config = experiment_config.export
    gpu_id = export_config.gpu_id
    on_cpu = export_config.on_cpu

    if not on_cpu:
        torch.cuda.set_device(gpu_id)

    # Get export parameters
    model_path = export_config.checkpoint
    key = experiment_config.encryption_key
    TLTPyTorchCookbook.set_passphrase(key)

    output_file = export_config.onnx_file
    encoder_type = getattr(export_config, 'encoder_type', 'combined')

    # Validate encoder type
    if encoder_type not in VALID_ENCODER_TYPES:
        raise ValueError(
            f"Invalid encoder_type '{encoder_type}'. "
            f"Must be one of: {VALID_ENCODER_TYPES}"
        )

    # Set default output filename if checkpoint provided
    if output_file is None:
        if model_path:
            split_name = os.path.splitext(model_path)[0]
            output_file = f"{split_name}.onnx"
        else:
            raise ValueError(
                "onnx_file must be specified when exporting without checkpoint"
            )

    # Create output directory
    output_root = os.path.dirname(os.path.realpath(output_file))
    if output_root and not os.path.exists(output_root):
        os.makedirs(output_root)

    device = 'cpu' if on_cpu else 'cuda'

    # Load model from checkpoint or build from HuggingFace pretrained weights
    if model_path:
        logging.info(f"Loading model from checkpoint: {model_path}")
        # pylint: disable=no-value-for-parameter
        pl_model = CLIPPlModel.load_from_checkpoint(
            model_path,
            map_location=device,
            experiment_spec=experiment_config
        )
    else:
        logging.info(
            f"No checkpoint provided. Building model from HuggingFace "
            f"pretrained weights: {experiment_config.model.type}"
        )
        pl_model = CLIPPlModel(experiment_config)
        pl_model = pl_model.to(device)

    # Replace nn.MultiheadAttention with export-friendly decomposed version.
    # PyTorch's fused _native_multi_head_attention has no ONNX symbolic, so
    # we swap in an equivalent module that uses standard ops the exporter
    # understands (linear, reshape, softmax, matmul).
    _replace_mha_for_export(pl_model)

    # Export based on encoder_type
    if encoder_type == 'combined':
        if os.path.exists(output_file):
            raise FileExistsError(
                f"Output ONNX file already exists: {output_file}"
            )
        logging.info("Exporting combined (vision + text) encoder")
        export_combined_encoder(
            pl_model,
            export_config,
            experiment_config,
            output_file,
            device,
        )

    else:  # separate
        base_name = os.path.splitext(output_file)[0]
        ext = os.path.splitext(output_file)[1]

        vision_file = f"{base_name}_vision{ext}"
        text_file = f"{base_name}_text{ext}"

        if os.path.exists(vision_file):
            raise FileExistsError(
                f"Output ONNX file already exists: {vision_file}"
            )
        if os.path.exists(text_file):
            raise FileExistsError(
                f"Output ONNX file already exists: {text_file}"
            )

        logging.info(
            "Exporting vision and text encoders as separate ONNX files"
        )
        export_single_encoder(
            pl_model,
            'vision',
            export_config,
            experiment_config,
            vision_file,
            device,
        )
        export_single_encoder(
            pl_model,
            'text',
            export_config,
            experiment_config,
            text_file,
            device,
        )

        logging.info(f"Both encoders exported: {vision_file}, {text_file}")

    # Inject learned logit parameters into the config so the saved YAML
    # contains trained values, not just initial ones.
    experiment_config.model.init_logit_scale = (
        pl_model.model.logit_scale.item()
    )
    experiment_config.model.init_logit_bias = (
        pl_model.model.logit_bias.item()
    )

    # Save experiment config alongside ONNX for deployment inference
    # This allows tao-deploy to auto-load settings like canonicalize_text
    base_name = os.path.splitext(output_file)[0]
    config_path = f"{base_name}_config.yaml"
    OmegaConf.save(experiment_config, config_path)
    logging.info(f"Experiment config saved to {config_path}")

    # Save tokenizer alongside ONNX for deployment
    tokenizer_dir = f"{base_name}_tokenizer"
    save_tokenizer(
        pl_model.tokenizer,
        tokenizer_dir,
        model_type=experiment_config.model.type,
        adaptor_name=getattr(experiment_config.model, 'adaptor_name', None),
    )
    logging.info(f"Tokenizer saved to {tokenizer_dir}")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig
)
@monitor_status(name="CLIP", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """Run the ONNX export process.

    Parameters
    ----------
    cfg : ExperimentConfig
        Hydra configuration object populated from experiment spec.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_export(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
