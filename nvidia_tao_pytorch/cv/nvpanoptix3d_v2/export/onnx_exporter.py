# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ONNX exporter for the NVPanoptix3Dv2 panoptic variant.

The graph takes one multi-view image tensor and returns the panoptic head's
logits and masks alongside VGGT's geometry, covering everything
``predict_step`` consumes. ``panoptic_inference`` stays outside the graph: it is
threshold-driven, emits variable-length segment tables, and is cheap next to the
backbone.

Four things do not survive tracing, and each is handled here rather than in the
model:

1. The vocabulary is text, which ONNX cannot take as an input. It is encoded
   once up front and baked in as a constant ``[C, D]`` matrix, making the
   exported model closed-vocabulary against the class list it was exported with.
   That list is written beside the ONNX as a sidecar JSON.
2. ``true_shape`` is only shape-checked by the model, so the wrapper synthesizes
   it rather than exposing a second, information-free input.
3. ``torch.quantile`` has no lowering, so
   :func:`export_safe_metric_scale_head` swaps ``MetricScaleHead``'s depth
   statistics for a sort-based equivalent during the export.
4. xFormers memory-efficient attention uses operators the legacy ONNX tracer
   cannot lower, so the wrapper switches those modules to their equivalent
   scaled-dot-product attention implementation.

Batch is the only axis that can be dynamic. The view count ``S`` is frozen at
the traced value, since ``torch.triu_indices(S, S)`` and the mask transformer's
positional tiling both need it as a Python int.
"""

import json
import math
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import onnx
import torch
from torch import nn

from nvidia_tao_pytorch.cv.nvpanoptix3d.export.utils import (
    patch_xformers_attention_for_export,
)
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.export.symbolic_funcs import register_symbolic_functions
from nvidia_tao_pytorch.cv.nvpanoptix3d_v2.model.metric_scale_head import MetricScaleHead


# Canonical ONNX output ordering. Entries absent from a given build (no
# objectness head, no metric-scale head) are dropped, never reordered.
OUTPUT_ORDER = (
    "pred_logits",
    "pred_masks",
    "pred_objectness",
    "depth",
    "depth_conf",
    "world_points",
    "world_points_conf",
    "pose_enc",
    "metric_depth",
    "metric_points",
    "intrinsics",
)

INPUT_NAME = "images"


class FrozenTextEmbeddings(nn.Module):
    """Stand-in for :class:`TextEncoder` that returns pre-encoded embeddings.

    The real encoder maps a list of Python strings to L2-normalised SigLIP
    embeddings. Once the vocabulary is fixed there is nothing left to compute,
    so this returns the cached matrix and ignores its argument, which keeps the
    decoder's call site (``self.text_encoder(classes)``) untouched.

    Args:
        embeddings: ``[C, D]`` L2-normalised class embeddings.
    """

    def __init__(self, embeddings: torch.Tensor):
        super().__init__()
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(
                "Frozen text embeddings must have shape [classes, channels]"
            )
        self.register_buffer("embeddings", embeddings)

    def forward(self, classes=None) -> torch.Tensor:
        """Return the frozen ``[C, D]`` class embeddings."""
        del classes
        return self.embeddings


def depth_stats_export_safe(self, rel_depth: torch.Tensor) -> torch.Tensor:
    """Export-safe replacement for :meth:`MetricScaleHead.depth_stats`.

    Numerically identical to the original: ``torch.quantile`` defaults to
    linear interpolation, which is what the sort/lerp below reproduces. The
    quantile positions are constants at trace time, so this lowers to a single
    sort plus gathers with no data-dependent shapes.

    Args:
        self: The :class:`MetricScaleHead` instance (bound at patch time).
        rel_depth: ``[B, S, H, W, 1]`` or ``[B, S, H, W]`` relative depth.

    Returns:
        ``[B, 5]`` — (mean, std, median, p10, p90) averaged over views.
    """
    if rel_depth.dim() == 5:
        rel_depth = rel_depth.squeeze(-1)

    # flatten(2) rather than reshape(B, S, H * W) so the batch axis stays
    # symbolic instead of being frozen into the graph.
    num_pixels = int(rel_depth.shape[-2]) * int(rel_depth.shape[-1])
    d = rel_depth.flatten(2).clamp(min=self.depth_min, max=self.depth_max)

    mean = d.mean(dim=-1)
    std = d.std(dim=-1, unbiased=False)

    ordered, _ = torch.sort(d, dim=-1)

    def quantile(q: float) -> torch.Tensor:
        position = q * (num_pixels - 1)
        low_index = int(math.floor(position))
        high_index = int(math.ceil(position))
        low = ordered[..., low_index]
        if high_index == low_index:
            return low
        return low + (ordered[..., high_index] - low) * (position - low_index)

    per_view = torch.stack(
        [mean, std, quantile(0.5), quantile(0.1), quantile(0.9)], dim=-1,
    )
    return per_view.mean(dim=1)


@contextmanager
def export_safe_metric_scale_head():
    """Temporarily swap ``MetricScaleHead.depth_stats`` for its traceable form.

    Patched on the class rather than the instance so it also covers the head
    reached through a wrapper, and restored unconditionally so an export
    failure cannot leave the training path altered.
    """
    original = MetricScaleHead.depth_stats
    MetricScaleHead.depth_stats = depth_stats_export_safe
    try:
        yield
    finally:
        MetricScaleHead.depth_stats = original


def flatten_outputs(
    panoptic_output: Dict,
    geometry_output: Dict,
) -> Dict[str, torch.Tensor]:
    """Merge the model's two output dicts into one flat, ordered tensor map.

    Drops everything ONNX cannot carry: the deep-supervision ``aux_outputs``
    list, the detached ``out_queries``, and the nested ``metric_scale_params``
    dict (its ``intrinsics`` is already promoted to a top-level key).

    Args:
        panoptic_output: First return value of ``NVPanoptix3Dv2Panoptic.forward``.
        geometry_output: Second return value of the same call.

    Returns:
        Ordered mapping from ONNX output name to tensor, in :data:`OUTPUT_ORDER`.
    """
    merged = {}
    for name in OUTPUT_ORDER:
        value = panoptic_output.get(name)
        if value is None:
            value = geometry_output.get(name)
        if isinstance(value, torch.Tensor):
            merged[name] = value
    return merged


def resolve_output_names(model: nn.Module, dummy_input: torch.Tensor) -> List[str]:
    """Determine which ONNX outputs this build produces, by running it once.

    Which keys exist depends on the enabled heads (objectness, metric scale,
    VGGT's camera/depth/point heads). Probing is more reliable than mirroring
    that logic here, and it surfaces a broken checkpoint before the much slower
    tracer runs.

    Args:
        model: A :class:`NVPanoptix3Dv2Panoptic` in eval mode, with its text
            encoder already frozen.
        dummy_input: ``[B, S, 3, H, W]`` probe tensor on the model's device.

    Returns:
        Output names in :data:`OUTPUT_ORDER`.
    """
    true_shape = dummy_input.new_zeros(dummy_input.shape[0], dummy_input.shape[1], 2)
    with torch.no_grad():
        panoptic_output, geometry_output = model(dummy_input, true_shape, None)
    return list(flatten_outputs(panoptic_output, geometry_output).keys())


class NVPanoptix3Dv2PanopticExportWrapper(nn.Module):
    """Traceable single-input / flat-tuple-output view of the panoptic model.

    Args:
        model: A :class:`NVPanoptix3Dv2Panoptic` in eval mode.
        class_embeddings: ``[C, D]`` L2-normalised embeddings for the export
            vocabulary. Replaces the SigLIP text encoder in-place, so ``model``
            must not be reused for open-vocabulary inference afterwards.
        probe_input: ``[B, S, 3, H, W]`` tensor used to discover which outputs
            this build produces. Required unless ``output_names`` is given.
        output_names: Explicit output ordering, skipping the probe.

    Raises:
        ValueError: If neither ``probe_input`` nor ``output_names`` is given.
    """

    def __init__(
        self,
        model: nn.Module,
        class_embeddings: torch.Tensor,
        probe_input: Optional[torch.Tensor] = None,
        output_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.model = model
        patch_xformers_attention_for_export(model)

        decoder = model.panoptic_decoder
        decoder.text_encoder = FrozenTextEmbeddings(class_embeddings)
        # Aux predictions are a training-time signal only; suppressing them
        # keeps the traced graph from retaining six extra head evaluations.
        decoder.deep_supervision = False

        # Probed only after the swap above: the real text encoder rejects the
        # ``None`` vocabulary this wrapper passes.
        if output_names is None:
            if probe_input is None:
                raise ValueError("Pass either probe_input or output_names")
            output_names = resolve_output_names(model, probe_input)
        self.output_names = list(output_names)
        if not self.output_names:
            raise ValueError("The model produced no exportable ONNX outputs")
        if len(self.output_names) != len(set(self.output_names)):
            raise ValueError("ONNX output names must be unique")
        unknown_names = set(self.output_names).difference(OUTPUT_ORDER)
        if unknown_names:
            raise ValueError(
                f"Unsupported ONNX output names: {sorted(unknown_names)}"
            )

    def forward(self, images: torch.Tensor):
        """Run the panoptic model and return its outputs as a flat tuple.

        Args:
            images: ``[B, S, 3, H, W]`` multi-view images in ``[0, 1]``.

        Returns:
            Tuple of tensors ordered exactly as ``self.output_names``.
        """
        # ``true_shape`` is only shape-validated by the model, so it is
        # synthesized here instead of becoming a second, information-free input.
        true_shape = images.new_zeros(images.shape[0], images.shape[1], 2)

        panoptic_output, geometry_output = self.model(images, true_shape, None)
        merged = flatten_outputs(panoptic_output, geometry_output)
        return tuple(merged[name] for name in self.output_names)


def encode_vocabulary(pl_model, classes: List[str], categories: Optional[List[Dict]] = None) -> torch.Tensor:
    """Pre-encode the export vocabulary and return its embedding matrix.

    Args:
        pl_model: The panoptic Lightning module.
        classes: Class names the exported model will predict against.
        categories: Optional category dicts aligned with ``classes``.

    Returns:
        ``[C, D]`` L2-normalised class embeddings, detached and on CPU.
    """
    pl_model.set_classes(
        classes=classes,
        categories=categories,
    )
    text_encoder = pl_model.model.panoptic_decoder.text_encoder
    with torch.no_grad():
        embeddings = text_encoder(classes)
    return embeddings.detach().float().cpu()


class ONNXExporter:
    """Traces the wrapped panoptic model to ONNX.

    Args:
        opset_version: ONNX opset to export at.
    """

    def __init__(self, opset_version: int = 17):
        self.opset_version = opset_version

    def export_model(
        self,
        model: nn.Module,
        onnx_file: str,
        dummy_input: torch.Tensor,
        input_names: List[str],
        output_names: List[str],
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """Export ``model`` to ``onnx_file``.

        Args:
            model: Wrapped, eval-mode model to trace.
            onnx_file: Destination path for the ONNX file.
            dummy_input: Input tensor used for tracing.
            input_names: Names for the graph inputs.
            output_names: Names for the graph outputs.
            batch_size: Traced batch size. ``None`` or ``-1`` marks axis 0 of
                every tensor dynamic.
            verbose: Verbose exporter logging.

        Returns:
            The path written.
        """
        register_symbolic_functions(self.opset_version)

        dynamic_axes = None
        if batch_size in (None, -1):
            dynamic_axes = {name: {0: "batch"} for name in list(input_names) + list(output_names)}

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_file,
                input_names=input_names,
                output_names=output_names,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,
                opset_version=self.opset_version,
                do_constant_folding=False,
                dynamic_axes=dynamic_axes,
                # VGGT-1B exceeds protobuf's 2 GB embedded-weight limit.
                external_data=True,
                verbose=verbose,
                dynamo=False,
            )

        return onnx_file

    @staticmethod
    def check_onnx(onnx_file: str) -> None:
        """Validate an exported ONNX file.

        Checked by path, not by loaded proto: this model exceeds protobuf's
        2 GB message limit, which the in-memory form of the checker cannot take.

        Args:
            onnx_file: Path to the ONNX file.
        """
        onnx.checker.check_model(onnx_file)


def write_vocabulary_sidecar(onnx_file: str, classes: List[str], categories: Optional[List[Dict]]) -> str:
    """Record the baked vocabulary next to the ONNX file.

    ``pred_logits`` channel ``i`` means ``classes[i]``, and that mapping is not
    recoverable from the graph, so it is persisted for the deployment side.

    Args:
        onnx_file: Path to the exported ONNX file.
        classes: Class names baked into the graph, in channel order.
        categories: Optional category dicts aligned with ``classes``.

    Returns:
        Path to the JSON sidecar.
    """
    sidecar = f"{os.path.splitext(onnx_file)[0]}.classes.json"
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump({"classes": classes, "categories": categories}, handle, indent=2)
        handle.write("\n")
    return sidecar
