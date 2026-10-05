# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantize a DepthNet model using the configured backend.

This script loads a trained DepthNet checkpoint (or an exported ONNX graph), prepares the
calibration data from the dataset specified in ``quant_calibration_dataset``, runs
quantization via ``ModelQuantizer``, and saves the quantized model.

Two model families are supported:

* **Mono** (``MetricDepthAnything`` / ``RelativeDepthAnything``): calibration images come
  from ``quant_calibration_dataset.images_dir``; the model takes a single image tensor.
* **Stereo** (``FoundationStereo`` / ``FastFoundationStereo``): calibration pairs come from
  ``quant_calibration_dataset.data_sources`` (falling back to ``test_dataset``); the model
  takes a left and a right image. With the ``modelopt.onnx`` backend the pairs are mapped
  onto the ONNX graph inputs (``left_image`` / ``right_image``) and resized to the graph's
  static H/W when needed.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import obfuscate_logs, logging

from nvidia_tao_pytorch.config.depth_net.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.quantization import ModelQuantizer
from nvidia_tao_pytorch.core.quantization.quantizer_base import FileBasedQuantizerBase
from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import build_pl_model, get_pl_module
from nvidia_tao_pytorch.cv.depth_net.dataloader.pl_mono_data_module import MonoDepthNetDataModule
from nvidia_tao_pytorch.cv.depth_net.dataloader.pl_stereo_data_module import StereoDepthNetDataModule


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STEREO_MODEL_TYPES = ("FoundationStereo", "FastFoundationStereo")
# Dataloader batch key -> preferred ONNX input name. Order matters: it is the positional
# fallback when the ONNX graph uses different input names.
STEREO_BATCH_TO_ONNX_INPUT = (("image", "left_image"), ("right_image", "right_image"))


def is_stereo_model(cfg: ExperimentConfig) -> bool:
    """Return True when the configured model consumes a stereo pair."""
    return cfg.model.model_type in STEREO_MODEL_TYPES


class StereoCalibrationWrapper(nn.Module):
    """Adapt a stereo depth model to the single-tensor calibration forward loop.

    The PyTorch quantization backends call ``model(x)`` with one tensor per batch. This
    wrapper accepts the left and right images concatenated along the channel axis
    (``[B, 6, H, W]``), splits them and runs the stereo model in test mode.
    """

    def __init__(self, model: nn.Module, iters: int):
        """Initialize the wrapper.

        Args:
            model (nn.Module): stereo model with ``forward(left, right, iters=..., test_mode=...)``.
            iters (int): number of GRU refinement iterations to run during calibration.
        """
        super().__init__()
        self.model = model
        self.iters = iters

    def forward(self, x: torch.Tensor):
        """Split the channel-concatenated pair and run the stereo model."""
        channels = x.shape[1] // 2
        left, right = x[:, :channels], x[:, channels:]
        return self.model(left, right, iters=self.iters, test_mode=True)


class StereoPairLoader:
    """Iterate a stereo dict dataloader as channel-concatenated tensors.

    Produces ``torch.cat([batch['image'], batch['right_image']], dim=1)`` so that the
    generic single-tensor calibration loop can drive :class:`StereoCalibrationWrapper`.
    """

    def __init__(self, loader):
        """Initialize with the underlying dict dataloader."""
        self.loader = loader

    def __len__(self):
        """Number of batches."""
        return len(self.loader)

    def __iter__(self):
        """Yield channel-concatenated stereo pairs."""
        for batch in self.loader:
            yield torch.cat([batch["image"], batch["right_image"]], dim=1)


def load_stereo_model(cfg: ExperimentConfig, model_path: str) -> nn.Module:
    """Load a stereo DepthNet ``nn.Module`` from a checkpoint.

    ``FastFoundationStereo`` commercial checkpoints are research-pickled modules, so they go
    through ``load_ffs_pretrained`` (mirrors ``scripts/export.py``); everything else is a
    Lightning checkpoint.

    Args:
        cfg (ExperimentConfig): experiment configuration.
        model_path (str): checkpoint path.

    Returns:
        nn.Module: the stereo model.
    """
    if cfg.model.model_type == "FastFoundationStereo":
        from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.fast_foundation_stereo.ckpt_utils import (
            load_ffs_pretrained,
        )
        pl_model = build_pl_model(cfg, export=True)
        result = load_ffs_pretrained(pl_model.model, model_path)
        if result["missing"] or result["unexpected"]:
            raise RuntimeError(
                f"FFS checkpoint does not match the configured model. "
                f"missing={result['missing']} unexpected={result['unexpected']}"
            )
        return pl_model.model
    pl_model = get_pl_module(cfg).load_from_checkpoint(
        model_path,
        map_location="cpu",
        experiment_spec=cfg,
        export=True,
    )
    return pl_model.model


def get_onnx_input_dims(onnx_path: str) -> Dict[str, List[Optional[int]]]:
    """Return the graph inputs of an ONNX file as ``{name: [dim or None, ...]}``.

    Initializers that are also listed as graph inputs (older exporters) are skipped.
    Symbolic / unknown dimensions are reported as ``None``.

    Args:
        onnx_path (str): path to the ONNX file.

    Returns:
        dict: input name -> list of dimensions (``None`` for dynamic axes).
    """
    model = onnx.load(onnx_path, load_external_data=False)
    initializer_names = {init.name for init in model.graph.initializer}
    inputs = {}
    for graph_input in model.graph.input:
        if graph_input.name in initializer_names:
            continue
        dims = []
        for dim in graph_input.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.HasField("dim_value") and dim.dim_value > 0 else None)
        inputs[graph_input.name] = dims
    return inputs


def resolve_stereo_onnx_inputs(onnx_inputs: Dict[str, List[Optional[int]]]) -> Dict[str, str]:
    """Map the stereo dataloader batch keys onto the ONNX image inputs.

    Prefers the names the DepthNet exporter emits (``left_image`` / ``right_image``) and
    falls back to the first two 4-D inputs in graph order. Any additional graph input is an
    error because the calibration loop has no value to feed it.

    Args:
        onnx_inputs (dict): output of :func:`get_onnx_input_dims`.

    Returns:
        dict: batch key (``image`` / ``right_image``) -> ONNX input name.
    """
    image_inputs = [name for name, dims in onnx_inputs.items() if len(dims) == 4]
    extra_inputs = [name for name in onnx_inputs if name not in image_inputs]
    if extra_inputs:
        raise ValueError(
            f"ONNX graph has non-image inputs {extra_inputs} that the stereo calibration loop "
            f"cannot populate. Re-export with the scalar arguments constant-folded."
        )
    if len(image_inputs) != 2:
        raise ValueError(
            f"Expected exactly two 4-D image inputs for a stereo ONNX graph, found {image_inputs}."
        )
    preferred = [onnx_name for _, onnx_name in STEREO_BATCH_TO_ONNX_INPUT]
    if all(name in onnx_inputs for name in preferred):
        return dict(STEREO_BATCH_TO_ONNX_INPUT)
    logging.warning(
        f"ONNX inputs {image_inputs} do not match {preferred}; mapping left/right positionally."
    )
    return {batch_key: onnx_name for (batch_key, _), onnx_name in zip(STEREO_BATCH_TO_ONNX_INPUT, image_inputs)}


def _match_static_hw(tensor: torch.Tensor, dims: List[Optional[int]], name: str) -> torch.Tensor:
    """Resize ``tensor`` (``[B, C, H, W]``) to the static H/W of an ONNX input when they differ.

    Mirrors the tao-deploy DepthNet dataloader, which resizes images to the engine input
    shape before normalization.
    """
    target_h, target_w = dims[2], dims[3]
    if target_h is None or target_w is None:
        return tensor
    if tensor.shape[-2:] == (target_h, target_w):
        return tensor
    logging.debug(f"Resizing calibration tensor {tuple(tensor.shape)} -> {name} static HxW ({target_h}x{target_w})")
    return F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)


def collect_stereo_calibration_data(loader, onnx_inputs: Dict[str, List[Optional[int]]]) -> Dict[str, np.ndarray]:
    """Collect stereo calibration pairs into a dict of ndarrays keyed by ONNX input name.

    Args:
        loader: dataloader yielding dict batches with ``image`` and ``right_image``.
        onnx_inputs (dict): output of :func:`get_onnx_input_dims`.

    Returns:
        dict: ``{onnx_input_name: float32 ndarray [N, C, H, W]}``.
    """
    key_map = resolve_stereo_onnx_inputs(onnx_inputs)
    collected = {onnx_name: [] for onnx_name in key_map.values()}
    num_samples = 0
    for batch in loader:
        for batch_key, onnx_name in key_map.items():
            tensor = batch[batch_key].detach().cpu().float()
            tensor = _match_static_hw(tensor, onnx_inputs[onnx_name], onnx_name)
            collected[onnx_name].append(tensor.numpy())
        num_samples += batch["image"].shape[0]
    if num_samples == 0:
        raise ValueError("Stereo calibration dataset is empty.")
    calibration_data = {}
    for name, arrays in collected.items():
        shapes = {array.shape[1:] for array in arrays}
        if len(shapes) > 1:
            raise ValueError(
                f"Calibration pairs for ONNX input '{name}' have mixed shapes {sorted(shapes)} and the "
                f"graph has dynamic H/W, so they cannot be batched. Quantize the static-H/W ONNX "
                f"(the Q/DQ placement carries over) or use same-size calibration images."
            )
        calibration_data[name] = np.concatenate(arrays, axis=0)
    shapes = {name: array.shape for name, array in calibration_data.items()}
    logging.info(f"Collected {num_samples} stereo calibration pairs: {shapes}")
    return calibration_data


def build_stereo_calibration_loader(cfg: ExperimentConfig):
    """Build the stereo calibration dataloader from ``quant_calibration_dataset``."""
    dm = StereoDepthNetDataModule(cfg.dataset)
    dm.setup(stage="calibration")
    return dm.calib_dataloader()


def quantize_stereo(cfg: ExperimentConfig) -> None:
    """Run quantization for a stereo DepthNet model.

    ONNX backend: calibration pairs are gathered into a multi-input dict and handed to the
    backend directly (``quantize_model`` would collapse them into a single array).
    PyTorch backends: the model is wrapped so the generic single-tensor forward loop can
    drive it with channel-concatenated pairs.
    """
    needs_calibration = cfg.quantize.mode != "weight_only_ptq"
    quantizer = ModelQuantizer(cfg.quantize)

    if isinstance(quantizer.quantizer, FileBasedQuantizerBase):
        onnx_inputs = get_onnx_input_dims(cfg.quantize.model_path)
        logging.info(f"ONNX graph inputs: {onnx_inputs}")
        quantizer.prepare(None)
        if needs_calibration:
            calibration_data = collect_stereo_calibration_data(
                build_stereo_calibration_loader(cfg), onnx_inputs
            )
            if not hasattr(quantizer.quantizer, "set_calibration_data"):
                raise NotImplementedError(
                    f"Backend '{cfg.quantize.backend}' cannot take multi-input calibration data."
                )
            quantizer.quantizer.set_calibration_data(calibration_data)
        quantized_model = quantizer.quantize(None)
        logging.info("Quantization finished; saving model")
        quantizer.save_model(quantized_model, cfg.quantize.results_dir)
        return

    if cfg.quantize.model_path.endswith(".onnx"):
        raise ValueError(
            f"Backend '{cfg.quantize.backend}' needs a PyTorch checkpoint, got ONNX: {cfg.quantize.model_path}"
        )
    model = load_stereo_model(cfg, cfg.quantize.model_path)
    wrapped = StereoCalibrationWrapper(model, iters=cfg.model.valid_iters)
    calibration_loader = StereoPairLoader(build_stereo_calibration_loader(cfg)) if needs_calibration else None
    quantized_wrapper = quantizer.quantize_model(wrapped, calibration_loader)
    logging.info("Quantization finished; saving model")
    quantizer.save_model(quantized_wrapper.model, cfg.quantize.results_dir)


def quantize_mono(cfg: ExperimentConfig) -> None:
    """Run quantization for a monocular DepthNet model."""
    # Build the Lightning model and extract the underlying nn.Module
    logging.debug("Loading DepthNet checkpoint")
    if not cfg.quantize.model_path.endswith(".onnx"):
        pl_model = get_pl_module(cfg).load_from_checkpoint(
            cfg.quantize.model_path,
            map_location="cpu",
            experiment_spec=cfg,
        )
        orig_model = pl_model.model
    else:
        orig_model = None  # ModelOpt ONNX backend loads the model from the file.

    # Prepare calibration dataloader via DataModule
    calib_cfg = cfg.dataset.quant_calibration_dataset
    calib_images_dir = getattr(calib_cfg, "images_dir", "")
    if cfg.quantize.mode != "weight_only_ptq" and calib_images_dir:
        dm = MonoDepthNetDataModule(cfg.dataset)
        dm.setup(stage="calibration")
        calibration_loader = dm.calib_dataloader()
    else:
        calibration_loader = None

    # Create quantizer and quantize the model
    quantizer = ModelQuantizer(cfg.quantize)
    quantized_model = quantizer.quantize_model(orig_model, calibration_loader)
    logging.info("Quantization finished; saving model")
    quantizer.save_model(quantized_model, cfg.quantize.results_dir)


# Load experiment specification, additionally using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="quantize",
    schema=ExperimentConfig,
)
@monitor_status(name="DepthNet", mode="quantize")
def main(cfg: ExperimentConfig) -> None:
    """Run the quantization process.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration including the ``quantize`` section.
    """
    # Obfuscate logs.
    obfuscate_logs(cfg)

    logging.info(f"Starting DepthNet quantization ({cfg.model.model_type})")
    if is_stereo_model(cfg):
        quantize_stereo(cfg)
    else:
        quantize_mono(cfg)
    logging.info("DepthNet quantization completed successfully")


if __name__ == "__main__":
    main()
