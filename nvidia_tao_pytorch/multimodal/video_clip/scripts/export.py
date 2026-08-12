# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Export CLIP model to ONNX."""

import copy
import os
from typing import Optional, Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.export import Dim

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import (
    TLTPyTorchCookbook,
)
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.utilities import encrypt_onnx
from nvidia_tao_pytorch.core.tlt_logging import logging

from nvidia_tao_pytorch.config.video_clip.default_config import (
    VideoCLIPExperimentConfig as ExperimentConfig,
)
from nvidia_tao_pytorch.multimodal.video_clip.model.pl_video_clip_model import VideoCLIPPlModel
from nvidia_tao_pytorch.multimodal.clip.model.lora import LoRALinear, merge_lora
from nvidia_tao_pytorch.multimodal.video_clip.model.tokenizers import save_tokenizer
from nvidia_tao_pytorch.multimodal.video_clip.utils.utils import (
    register_checkpoint_safe_globals,
)


# Valid encoder types for export (aligned with CLIPExportConfig.encoder_type)
VALID_ENCODER_TYPES = {'combined', 'separate'}


def _get_video_export_num_frames(model: nn.Module) -> Optional[int]:
    """Return video frame count for InternVideo2-style CLIP models."""
    if not getattr(model, "is_internvideo2", False):
        return None
    num_frames = getattr(model, "num_frames", None)
    if num_frames is None:
        return None
    try:
        num_frames = int(num_frames)
    except (TypeError, ValueError):
        return None
    return num_frames if num_frames > 0 else None


def _uses_tensor_text_input(model: nn.Module) -> bool:
    """Return whether the CLIP adapter expects token tensors directly."""
    return bool(getattr(model, "is_internvideo2", False))


def _infer_token_sequence_length(tokenized_text) -> int:
    """Infer tokenizer sequence length from tensor or dict tokenizer output."""
    if isinstance(tokenized_text, (list, tuple)):
        tokenized_text = tokenized_text[0]
    if isinstance(tokenized_text, torch.Tensor):
        return tokenized_text.shape[-1]
    if hasattr(tokenized_text, "values"):
        return next(iter(tokenized_text.values())).shape[-1]
    raise TypeError(
        f"Unsupported tokenizer output type: {type(tokenized_text)!r}"
    )


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
            Input image tensor of shape (B, C, H, W), or video tensor of
            shape (B, T, C, H, W) for video CLIP models.

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
        if _uses_tensor_text_input(self.model):
            text_input = input_ids
        else:
            text_input = {
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids)
            }
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
            Input tensor of shape (B, T, C, H, W) for video models or
            (B, C, H, W) for image models.
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
        # InternVideo2 expects a raw token tensor; CLIP/OpenCLIP expect a dict
        if _uses_tensor_text_input(self.model):
            text_input = input_ids
        else:
            text_input = {
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids)
            }
        image_features, text_features, logit_scale, logit_bias = self.model(
            image=image, text=text_input
        )
        return image_features, text_features, logit_scale, logit_bias


def _fuse_rms_normalization(onnx_path: str) -> int:
    """Fuse torch's decomposed RMSNorm into the ONNX opset-23 RMSNormalization op.

    The exporter lowers RMSNorm (custom or ``torch.nn.RMSNorm``) to a
    ``Pow -> ReduceMean -> Add -> Sqrt -> Reciprocal -> Mul -> Mul(scale)``
    subgraph. This rewrites each such subgraph, in place, into a single
    ``RMSNormalization`` node (ONNX opset 23) with ``stash_type`` set to fp32, so
    the normalization runs in fp32 on the deploy engine without pinning the
    individual ``Pow``/``ReduceMean`` layers.

    Reuses onnxscript's RMSNormalization fusion pattern, generalized to accept
    the concrete last-axis index the torch exporter emits (e.g. ``[2]``) rather
    than only the literal ``[-1]`` the stock rule expects. onnxscript is imported
    lazily so it is only required for opset >= 23 exports.

    Returns the number of subgraphs fused.
    """
    import onnxscript.ir as ir
    from onnxscript.rewriter import pattern, _ir_utils, _fusion_utils

    class _RmsNormFlexAxes(pattern.RewriteRuleClassBase):
        def __init__(self, name, mul_order):
            super().__init__(name)
            self._mul_order = mul_order

        def pattern(self, op, x, scale, epsilon, axes, compute_dtype, target_dtype):
            x = pattern.OrValue([op.Cast(x, to=compute_dtype), x])
            mean_square = op.ReduceMean(
                op.Pow(x, 2.0), axes, keepdims=1, noop_with_empty_axes=0
            )
            normalized = op.Mul(x, op.Reciprocal(op.Sqrt(op.Add(mean_square, epsilon))))
            normalized = pattern.OrValue([op.Cast(normalized, to=target_dtype), normalized])
            if self._mul_order:
                return op.Mul(normalized, scale)
            return op.Mul(scale, normalized)

        def check(self, op, x, scale, epsilon, axes, compute_dtype, target_dtype, **_):
            result = pattern.MatchResult()
            if not isinstance(_ir_utils.get_singleton_value(epsilon), float):
                return result.fail("epsilon is not a scalar float", epsilon)
            axes_val = _ir_utils.get_numpy_value(axes)
            if axes_val is None or axes_val.size != 1:
                return result.fail("reduction axes is not a single value", axes)
            rank = len(x.shape) if x.shape is not None else None
            axis = int(axes_val.reshape(-1)[0])
            if not (axis == -1 or (rank is not None and axis == rank - 1)):
                return result.fail("reduction is not over the last axis", axes)
            self._stash_dtype = (
                compute_dtype.as_int() if compute_dtype is not None else x.dtype
            )
            return result

        def rewrite(self, op, x, scale, epsilon, **_):
            return op.RMSNormalization(
                x, scale, axis=-1,
                epsilon=_ir_utils.get_singleton_value(epsilon),
                stash_type=self._stash_dtype,
            )

    fuse = _fusion_utils.apply_fusion_rules(pattern.RewriteRuleSet([
        _RmsNormFlexAxes.rule("RmsNormFlexAxes_mul", mul_order=True),
        _RmsNormFlexAxes.rule("RmsNormFlexAxes_scale", mul_order=False),
    ]))
    model = ir.load(onnx_path)
    count = fuse(model)
    ir.save(model, onnx_path)
    return count


# Default tolerances for the ONNX-vs-torch parity check (fp32). fp16 exports
# relax these automatically inside _verify_onnx_parity.
_PARITY_RTOL = 1e-3
_PARITY_ATOL = 1e-3


def _verify_onnx_parity(
    encoder: nn.Module,
    dummy_input,
    onnx_path: str,
    input_names,
    output_names,
    encoder_type: str,
):
    """Check the exported ONNX reproduces the torch encoder's outputs.

    Runs the eval-mode torch encoder and onnxruntime on identical inputs and
    compares each named output, catching the per-export failure modes that vary
    by checkpoint: tracing, constant-folding, the >2GB external-data round-trip,
    dtype, and onnxruntime op semantics. Both sides run on CPU so the fp32
    comparison is apples-to-apples (no spurious GPU-vs-CPU drift).

    Mode is controlled by the ``VIDEO_CLIP_EXPORT_PARITY`` environment variable:
    ``strict`` (default) raises on mismatch, ``warn`` logs and continues,
    ``off`` skips the check entirely.
    """
    mode = os.environ.get("VIDEO_CLIP_EXPORT_PARITY", "strict").lower()
    if mode == "off":
        logging.warning(
            "Skipping ONNX parity check for %s encoder "
            "(VIDEO_CLIP_EXPORT_PARITY=off).", encoder_type
        )
        return
    strict = mode != "warn"

    inputs = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)

    # Reference on CPU to match the onnxruntime CPU session. Move the encoder to
    # CPU for the one-shot forward and restore its device so the caller sees no
    # side effect.
    orig_device = next(encoder.parameters()).device
    encoder.eval()
    encoder.cpu()
    cpu_inputs = tuple(t.cpu() for t in inputs)
    with torch.no_grad():
        torch_out = encoder(*cpu_inputs)
    encoder.to(orig_device)
    if isinstance(torch_out, torch.Tensor):
        torch_out = (torch_out,)
    torch_np = [t.detach().cpu().numpy() for t in torch_out]

    rtol, atol = _PARITY_RTOL, _PARITY_ATOL
    out_dtype = next(
        (t.dtype for t in torch_out if t.is_floating_point()), None
    )
    if out_dtype == torch.float16:
        rtol, atol = max(rtol, 2e-2), max(atol, 2e-2)

    try:
        import onnxruntime as ort
        # Pin threads: this is a tiny one-shot inference, and leaving threads
        # implicit triggers noisy pthread_setaffinity warnings on multi-NUMA
        # hosts.
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001 - inability to load is itself a failure
        msg = (
            f"Could not run ONNX parity check for {encoder_type} encoder: {exc}"
        )
        # onnxruntime only loads official released opsets (<= 22 as of ORT
        # 1.22). A newer opset (e.g. 23) is a tooling gap, not a model-
        # correctness failure, so warn and skip rather than fail an
        # otherwise-valid (onnx.checker-passing) export.
        exc_l = str(exc).lower()
        opset_unsupported = "opset" in exc_l and (
            "support" in exc_l or "under development" in exc_l
        )
        if strict and not opset_unsupported:
            raise RuntimeError(msg) from exc
        logging.warning(msg)
        return

    feed = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(input_names, cpu_inputs)
    }
    onnx_out = sess.run(None, feed)

    mismatches = []
    for name, ref, got in zip(output_names, torch_np, onnx_out):
        got = np.asarray(got)
        max_abs = float(np.max(np.abs(ref - got))) if ref.size else 0.0
        ok = np.allclose(ref, got, rtol=rtol, atol=atol)
        logging.info(
            "ONNX parity [%s] %s: max_abs_diff=%.3e (rtol=%g, atol=%g) -> %s",
            encoder_type, name, max_abs, rtol, atol,
            "OK" if ok else "MISMATCH",
        )
        if not ok:
            mismatches.append(f"{name} (max_abs_diff={max_abs:.3e})")

    if mismatches:
        msg = (
            f"ONNX export for the {encoder_type} encoder does not match the "
            f"torch model within tolerance: {', '.join(mismatches)}. The "
            f"exported graph would produce different embeddings."
        )
        if strict:
            raise RuntimeError(msg)
        logging.warning(msg)
    else:
        logging.info(
            "ONNX parity check passed for %s encoder.", encoder_type
        )


def _resolve_export_batch_size(batch_size, use_dynamo):
    """Return the tracing batch size and whether the ONNX batch is dynamic."""
    if batch_size is not None and (batch_size == 0 or batch_size < -1):
        raise ValueError("export.batch_size must be -1 or a positive integer")
    is_dynamic = batch_size is None or batch_size == -1
    if not is_dynamic:
        return batch_size, False
    # torch.export specializes example dimensions of size 0 or 1. Batch 2 is
    # only a tracing sample; it does not constrain the exported ONNX maximum.
    return (2 if use_dynamo else 1), True


def _dynamic_export_spec(dummy_input, is_dynamic, use_dynamo, dynamic_axes):
    """Select the ONNX dynamism spec by exporter.

    The dynamo exporter (opset > 20) ignores ``dynamic_axes`` and consumes
    ``dynamic_shapes`` instead; the legacy TorchScript exporter uses
    ``dynamic_axes``. Returns ``(dynamic_axes, dynamic_shapes)`` with at most one
    populated so batch stays dynamic on either exporter path.
    """
    if not (is_dynamic and use_dynamo):
        return dynamic_axes, None
    inputs = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)
    batch = Dim("batch", min=1)
    return None, tuple({0: batch} for _ in inputs)


def export_single_encoder(
    pl_model: VideoCLIPPlModel,
    encoder_type: str,
    export_config,
    experiment_config: ExperimentConfig,
    output_file: str,
    device: str
) -> str:
    """Export a single encoder (vision or text) to ONNX.

    Parameters
    ----------
    pl_model : VideoCLIPPlModel
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

    use_dynamo = opset_version > 20
    input_batch_size, is_dynamic = _resolve_export_batch_size(
        batch_size, use_dynamo
    )

    if encoder_type == 'vision':
        encoder = CLIPVisionEncoder(pl_model.model)
        input_channel = export_config.input_channel
        input_width = export_config.input_width
        input_height = export_config.input_height
        num_frames = _get_video_export_num_frames(pl_model.model)
        if num_frames is not None:
            input_shape = [
                num_frames, input_channel, input_height, input_width
            ]
            input_kind = "video"
        else:
            input_shape = [input_channel, input_height, input_width]
            input_kind = "image"

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
            f"{[input_batch_size] + input_shape} ({input_kind})"
        )

    else:  # text encoder
        encoder = CLIPTextEncoder(pl_model.model)
        # Infer sequence length from the model's tokenizer rather than
        # hardcoding, since different text encoders have different context
        # lengths (e.g., SigLIP2: 64, OpenCLIP/DFN CLIP: 77)
        dummy_tokens = pl_model.tokenizer(["test"])
        seq_length = _infer_token_sequence_length(dummy_tokens)

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
    dynamic_axes, dynamic_shapes = _dynamic_export_spec(
        dummy_input, is_dynamic, use_dynamo, dynamic_axes)
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            dummy_input,
            tmp_onnx_file,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamic_shapes=dynamic_shapes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=export_config.verbose,
            # The legacy TorchScript exporter caps at opset 20; opset 21+
            # (e.g. the default 23) is only supported by the dynamo exporter.
            dynamo=use_dynamo,
        )

    # opset 23 introduces the fused RMSNormalization op; fold the decomposed
    # RMSNorm subgraphs into it so the deploy engine normalizes in fp32 without
    # per-layer precision pinning.
    if opset_version >= 23:
        n_rms = _fuse_rms_normalization(tmp_onnx_file)
        logging.info(
            "Fused %d decomposed RMSNorm subgraph(s) into the ONNX opset-23 "
            "RMSNormalization op (fp32 stash_type).", n_rms,
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

    # Numerically verify the exported graph against the torch model before
    # any encryption (onnxruntime cannot read the encrypted .etlt).
    _verify_onnx_parity(
        encoder, dummy_input, tmp_onnx_file,
        input_names, output_names, encoder_type,
    )

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
    pl_model: VideoCLIPPlModel,
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
    pl_model : VideoCLIPPlModel
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

    use_dynamo = opset_version > 20
    input_batch_size, is_dynamic = _resolve_export_batch_size(
        batch_size, use_dynamo
    )

    # Build combined encoder wrapper
    encoder = CLIPCombinedEncoder(pl_model.model)
    encoder.eval()
    if device == 'cuda':
        encoder.cuda()

    # Vision dummy input -- prepend the temporal (num_frames) dim for video models
    input_channel = export_config.input_channel
    input_width = export_config.input_width
    input_height = export_config.input_height
    num_frames = _get_video_export_num_frames(pl_model.model)
    if num_frames is not None:
        input_shape = [num_frames, input_channel, input_height, input_width]
    else:
        input_shape = [input_channel, input_height, input_width]
    dummy_image = torch.randn(input_batch_size, *input_shape, device=device)

    # Text dummy input -- infer sequence length from the model's tokenizer
    # (handles both tensor (InternVideo2) and dict (CLIP/OpenCLIP) outputs)
    seq_length = _infer_token_sequence_length(pl_model.tokenizer(["test"]))
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

    dynamic_axes, dynamic_shapes = _dynamic_export_spec(
        dummy_input, is_dynamic, use_dynamo, dynamic_axes)
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            dummy_input,
            tmp_onnx_file,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamic_shapes=dynamic_shapes,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=export_config.verbose,
            # The legacy TorchScript exporter caps at opset 20; opset 21+
            # (e.g. the default 23) is only supported by the dynamo exporter.
            dynamo=use_dynamo,
        )

    # opset 23 introduces the fused RMSNormalization op; fold the decomposed
    # RMSNorm subgraphs into it so the deploy engine normalizes in fp32 without
    # per-layer precision pinning.
    if opset_version >= 23:
        n_rms = _fuse_rms_normalization(tmp_onnx_file)
        logging.info(
            "Fused %d decomposed RMSNorm subgraph(s) into the ONNX opset-23 "
            "RMSNormalization op (fp32 stash_type).", n_rms,
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

    # Numerically verify the exported graph against the torch model before
    # any encryption (onnxruntime cannot read the encrypted .etlt).
    _verify_onnx_parity(
        encoder, dummy_input, tmp_onnx_file,
        input_names, output_names, 'combined',
    )

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
        # Preservation regularization is training-only; skip the frozen-teacher
        # deep copy so export does not carry a second copy of the backbone.
        restore_config = copy.deepcopy(experiment_config)
        restore_reg = getattr(restore_config, "regularization", None)
        if restore_reg is not None and getattr(restore_reg, "enabled", False):
            restore_reg.enabled = False
        # pylint: disable=no-value-for-parameter
        pl_model = VideoCLIPPlModel.load_from_checkpoint(
            model_path,
            map_location=device,
            experiment_spec=restore_config
        )
    else:
        logging.info(
            f"No checkpoint provided. Building model from HuggingFace "
            f"pretrained weights: {experiment_config.model.type}"
        )
        pl_model = VideoCLIPPlModel(experiment_config)
        pl_model = pl_model.to(device)

    # Match the vision dummy-input resolution to the model when the user left
    # the export defaults untouched. IV2-CLIP uses image_size=224 while the
    # export config defaults to 256; a mismatch breaks the vision tower's
    # positional embeddings during tracing.
    model_image_size = getattr(pl_model.model, "image_size", None)
    try:
        model_image_size = int(model_image_size) if model_image_size else None
    except (TypeError, ValueError):
        model_image_size = None
    if (
        model_image_size and
        model_image_size != 256 and
        export_config.input_height == 256 and
        export_config.input_width == 256
    ):
        logging.info(
            f"Overriding default export resolution 256x256 with model "
            f"image_size {model_image_size}x{model_image_size}"
        )
        export_config.input_height = model_image_size
        export_config.input_width = model_image_size

    # Merge LoRA weights into the base model if present (PEFT mode). Runs only
    # when the loaded checkpoint contains LoRALinear modules (i.e. the spec had
    # peft.enabled); for standard SFT checkpoints this is a no-op. Must happen
    # before the MHA swap so the exported graph has zero LoRA overhead.
    if any(isinstance(m, LoRALinear) for m in pl_model.modules()):
        merged = merge_lora(pl_model)
        logging.info(f"Merged {merged} LoRA modules into base weights for export.")

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
@monitor_status(name="VideoCLIP", mode="export")
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
