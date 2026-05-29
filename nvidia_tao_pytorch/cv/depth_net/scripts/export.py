# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script to export a DepthNet model to ONNX format."""

import os
import torch
import onnx

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import get_pl_module
from nvidia_tao_pytorch.config.depth_net.default_config import ExperimentConfig

spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTOCAST = torch.amp.autocast


def _create_mono_onnx_model(
    model,
    input_shape,
    input_batch_size,
    output_path,
    input_names,
    output_names,
    opset_version=17,
    on_cpu=False,
    dynamic_axis=False,
    dynamic_hw=False,
):
    """Export mono depth model to ONNX with optional dynamic batch and H/W axes.

    Local fork of `nvidia_tao_pytorch.ssl.mae.scripts.export.create_onnx_model`
    so other models keep batch-only dynamic axes.

    Args:
        model (nn.Module): mono depth model to trace.
        input_shape (list[int]): [C, H, W] of single sample.
        input_batch_size (int): batch dim of dummy input.
        output_path (str): destination ONNX path.
        input_names (list[str]): ONNX input tensor names.
        output_names (list[str]): ONNX output tensor names.
        opset_version (int): ONNX opset target.
        on_cpu (bool): True traces on CPU; False on CUDA.
        dynamic_axis (bool): True marks batch axis dynamic.
        dynamic_hw (bool): True additionally marks H/W axes dynamic.

    Returns:
        None. Writes ONNX file at output_path.
    """
    model.eval()
    if not on_cpu:
        model.cuda()
    model.float()

    if input_shape[0] not in [1, 3]:
        raise ValueError(
            f"Invalid input channel: {input_shape[0]}. Only 1 or 3 are supported."
        )

    if os.path.exists(output_path):
        raise ValueError(
            f"Default onnx file {output_path} already exists"
        )

    if on_cpu:
        dummy_input = torch.ones(input_batch_size, *input_shape, device='cpu').float()
    else:
        dummy_input = torch.ones(input_batch_size, *input_shape, device='cuda').float()

    dynamic_axes = None
    if dynamic_axis or dynamic_hw:
        dynamic_axes = {}
        for input_name in input_names:
            axes = {}
            if dynamic_axis:
                axes[0] = 'batch_size'
            if dynamic_hw:
                axes[2] = 'height'
                axes[3] = 'width'
            dynamic_axes[input_name] = axes
        # Mono depth output is 3D (B, H, W) — see RelativeDepthAnythingV2.forward
        # `depth.squeeze(1)` at dpt.py:380. So height is axis 1, width axis 2.
        for output_name in output_names:
            out_axes = {}
            if dynamic_axis:
                out_axes[0] = 'batch_size'
            if dynamic_hw:
                out_axes[1] = 'height'
                out_axes[2] = 'width'
            dynamic_axes[output_name] = out_axes

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    logging.info("Verifying ONNX model")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment_spec", schema=ExperimentConfig
)
@monitor_status(name="DepthNet", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """Entry point for DepthNet model export process.

    This function serves as the main entry point for exporting a trained MAE (Masked Autoencoder)
    model. It configures the PyTorch backend settings and delegates the actual export process
    to the run_export function.

    Args:
        cfg (ExperimentConfig): Hydra configuration object containing all export parameters.
            This object is automatically populated by Hydra based on the experiment spec
            and command line arguments.

    Note:
        This function is decorated with @hydra_runner and @monitor_status to handle
        configuration loading and workflow monitoring respectively. The actual export
        logic is implemented in the run_export function.

    Example:
        >>> # The function is typically called via command line:
        >>> # python export.py export.onnx_file=model.onnx export.input_channel=3
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_export(cfg)


def mono_onnx_export(model,
                     input_shape,
                     input_batch_size,
                     output_file,
                     on_cpu,
                     opset_version,
                     valid_iters=22,
                     dynamic_axis=True,
                     dynamic_hw=False):
    """
    Exports a monocular depth estimation model to the ONNX format.

    This function serves as a wrapper for the `create_onnx_model` utility, handling
    the specifics of exporting a monocular depth estimation model. It defines the
    input and output tensor names and determines whether to enable dynamic axes
    based on the provided `batch_size`. If `batch_size` is -1, the exported ONNX
    model will have a dynamic batch dimension, allowing for flexible input
    sizes at inference time. The export process is delegated to
    `create_onnx_model`, which handles the core ONNX conversion logic.

    Args:
        model (torch.nn.Module): The PyTorch monocular depth estimation model to
                                  be exported.
        input_shape (list or tuple): The shape of a single input image tensor,
                                     excluding the batch dimension. Example:
                                     `[3, 256, 256]` for a 256x256 color image.
        input_batch_size (int): The batch size to be used for the dummy input
                                during the export process. This is the concrete
                                batch size used to create the dummy tensor,
                                even if `batch_size` is set to -1 for a
                                dynamic axis.
        output_file (str): The path to the output ONNX file (e.g., 'model.onnx').
        on_cpu (bool): If True, the dummy input tensors will be created on the CPU.
                       If False, they will be created on the GPU (CUDA).
        opset_version (int): The ONNX operator set version to use for the export.

    Raises:
        Exception: If the underlying `create_onnx_model` function or any part
                   of the export process fails, an exception is raised with
                   a detailed error message.
    """
    input_names = ['images']
    output_names = ['outputs']
    # Mono export covers RelativeDepthAnything and MetricDepthAnything, both
    # of which use a DINOv2 ViT-L backbone whose positional embedding
    # interpolation captures the trace-time patch count as a Python int.
    # torch.onnx.export bakes that int as a constant, so a dynamic-H/W engine
    # would silently produce a wrong-shape pos-embed at runtime. Ignore the
    # request with a warning and fall back to static H/W.
    if dynamic_hw:
        import warnings
        warnings.warn(
            "dynamic_hw=True is unsafe for mono export: the DINOv2 backbone "
            "constant-folds the trace patch count into the positional "
            "embedding, so a dynamic-H/W engine would silently produce a "
            "wrong-shape pos-embed at runtime. Falling back to static H/W. "
            "Use dynamic_hw=True only with FastFoundationStereo.",
            stacklevel=2,
        )
        dynamic_hw = False
    try:
        _create_mono_onnx_model(
            model,
            input_shape,
            input_batch_size,
            output_file,
            input_names,
            output_names,
            on_cpu=on_cpu,
            dynamic_axis=dynamic_axis,
            dynamic_hw=dynamic_hw,
            opset_version=opset_version,
        )
        logging.info("ONNX model saved to {output_file}".format(output_file=output_file))
    except Exception as e:
        logging.error(f"Error exporting model: {e}")
        raise e


def stereo_onnx_export(model,
                       input_shape,
                       input_batch_size,
                       output_file,
                       on_cpu,
                       opset_version,
                       valid_iters=22,
                       dynamic_axis=False,
                       dynamic_hw=False):
    """
    Exports a stereo depth estimation model to the ONNX format.

    This function prepares dummy input tensors and then uses `torch.onnx.export` to
    convert the PyTorch model into an ONNX representation. After export, it
    verifies the integrity of the generated ONNX file using `onnx.checker.check_model`.

    Two independent flags control which axes are dynamic:

    * `dynamic_axis=True` marks the batch axis dynamic. Set via
      `export.batch_size: -1` in the spec.
    * `dynamic_hw=True` additionally marks the height and width axes
      dynamic. Set via `export.dynamic_hw: true`.

    `dynamic_hw=True` is safe only for FastFoundationStereo (EdgeNeXt-only
    backbone, no positional embeddings). For FoundationStereo and the mono
    path the DINOv2 backbone constant-folds the trace patch count into
    pos-embed shape arithmetic, and a dynamic-H/W engine built from such
    an ONNX silently produces a wrong-shape pos-embed at runtime; this
    helper warns and falls back to static H/W in that case.

    Args:
        model (torch.nn.Module): The PyTorch stereo depth estimation model to be exported.
                                  The model should accept two images (left and right)
                                  and a set of additional arguments as input.
        input_shape (list or tuple): The shape of a single input image,
                                     excluding the batch dimension.
                                     Example: `[3, 256, 256]` for a color image
                                     of size 256x256.
        input_batch_size (int): The batch size to be used for the dummy input.
                                With `dynamic_axis=True` the exported ONNX accepts
                                any batch size at runtime.
        output_file (str): The path to the output ONNX file (e.g., 'model.onnx').
        on_cpu (bool): If True, the dummy input tensors will be created on the CPU.
                       If False, they will be created on the GPU (CUDA).
        opset_version (int): The ONNX operator set version to use for the export.
                             A higher version may support more recent operations.
        valid_iters (int): Number of GRU refinement iterations baked into the
                           exported graph.
        dynamic_axis (bool): If True, mark the batch axis as dynamic.
        dynamic_hw (bool): If True, additionally mark the height and width
                           axes as dynamic. Honored only for
                           FastFoundationStereo; emits a warning and falls
                           back to static H/W otherwise.

    Raises:
        Exception: If the ONNX export or the subsequent model check fails,
                   an exception is raised, providing details about the error.
    """
    import warnings

    input_names = ['left_image', 'right_image', 'iters',
                   'flow_init', 'test_mode', 'low_mem', 'init_disp']
    output_names = ["disparity"]

    input_shape.insert(0, input_batch_size)
    with AUTOCAST('cuda', enabled=True):
        if on_cpu:
            dummy_input1 = torch.rand(input_shape, device='cpu')
            dummy_input2 = torch.rand(input_shape, device='cpu')
        else:
            dummy_input1 = torch.rand(input_shape, device='cuda')
            dummy_input2 = torch.rand(input_shape, device='cuda')
    # Honor dynamic_hw only for FastFoundationStereo. The mono path and
    # FoundationStereo share a DINOv2 backbone whose positional embedding
    # uses the trace-time patch count as a Python int — torch.onnx.export
    # bakes that int as a constant, so a dynamic-H/W engine then mismatches
    # the runtime patch tokens. FastFoundationStereo uses EdgeNeXt only,
    # which is fully convolutional and has no pos-embed.
    model_class = type(model).__name__
    if dynamic_hw and model_class != 'FastFoundationStereo':
        warnings.warn(
            f"dynamic_hw=True is unsafe for model class {model_class!r}: the "
            "DINOv2 backbone constant-folds the trace patch count into the "
            "positional embedding, so a dynamic-H/W engine would silently "
            "produce a wrong-shape pos-embed at runtime. Falling back to "
            "static H/W. Use dynamic_hw=True only with FastFoundationStereo.",
            stacklevel=2,
        )
        dynamic_hw = False

    try:
        axes_config = {}
        if dynamic_axis:
            for t in input_names[:2] + output_names:
                axes_config.setdefault(t, {})[0] = 'batch_size'
        if dynamic_hw:
            for t in input_names[:2] + output_names:
                axes_config.setdefault(t, {})[2] = 'height'
                axes_config.setdefault(t, {})[3] = 'width'
        if not axes_config:
            axes_config = None

        torch.onnx.export(model,
                          args=(dummy_input1, dummy_input2, valid_iters, None, True, False, None),
                          f=output_file,
                          input_names=input_names,
                          opset_version=opset_version,
                          output_names=output_names,
                          do_constant_folding=True,
                          verbose=True,
                          dynamic_axes=axes_config,
                          dynamo=False)

        # Verify ONNX exported correctly.
        loaded_model = onnx.load(output_file)
        onnx.checker.check_model(loaded_model)

    except Exception as e:
        logging.error(f"Error exporting model: {e}")
        raise e


def onnx_model_export(model_type):
    """ Factory function to export ONNX for mono and stereo models.
    Args:
        model_type (str): the model type to be exported.

    Returns:
        an export function for either modes.

    """
    onnx_export_method = {
        'metricdepthanything': mono_onnx_export,
        'relativedepthanything': mono_onnx_export,
        'foundationstereo': stereo_onnx_export,
        'fastfoundationstereo': stereo_onnx_export}

    if model_type.lower() not in onnx_export_method:
        raise (NotImplementedError(f'{model_type} does not have onnx export implemented!'))
    return onnx_export_method[model_type.lower()]


def run_export(experiment_config: ExperimentConfig) -> None:
    """Execute the DepthNet model export process to ONNX format.

    This function handles the core export process for a trained MAE (Masked Autoencoder) model.
    It processes the experiment configuration, sets up the model with proper encryption,
    and exports it to ONNX format with the specified parameters.

    Args:
        experiment_config (ExperimentConfig): Configuration object containing export parameters:
            - export.gpu_id: GPU device ID to use for export
            - export.checkpoint: Path to the model checkpoint file
            - encryption_key: Key for model encryption/decryption
            - export.onnx_file: Output path for the ONNX model
            - export.input_channel: Number of input channels (e.g., 3 for RGB)
            - export.input_width: Input image width
            - export.input_height: Input image height
            - export.opset_version: ONNX opset version for export
            - export.batch_size: Batch size for export (defaults to 1 if None or -1)
            - export.on_cpu: Whether to perform export on CPU instead of GPU
            - export.valid_iters: Number of GPU valid iterations to refine disparity

    Raises:
        AssertionError: If the output ONNX file already exists at the specified path.
        RuntimeError: If model loading fails or if there are issues during ONNX export.
        ValueError: If required configuration parameters are missing or invalid.

    Note:
        The function handles both CPU and GPU exports based on the configuration.
        For GPU exports, it sets the appropriate CUDA device before proceeding.

    Example:
        >>> config = ExperimentConfig()
        >>> config.export.onnx_file = "model.onnx"
        >>> config.export.input_channel = 3
        >>> config.export.input_width = 924
        >>> config.export.input_height = 518
        >>> run_export(config)
    """
    # Convert DictConfig to ExperimentConfig
    gpu_id = experiment_config.export.gpu_id
    torch.cuda.set_device(gpu_id)

    # Parsing command line arguments.
    model_path = experiment_config.export.checkpoint
    key = experiment_config.encryption_key

    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    output_file = experiment_config.export.onnx_file
    input_channel = experiment_config.export.input_channel
    input_width = experiment_config.export.input_width
    input_height = experiment_config.export.input_height
    input_shape = [input_channel, input_height, input_width]
    opset_version = experiment_config.export.opset_version
    batch_size = experiment_config.export.batch_size
    on_cpu = experiment_config.export.on_cpu
    valid_iters = experiment_config.export.valid_iters

    if batch_size is None or batch_size == -1:
        input_batch_size = 1
        dynamic_axis = True
    else:
        input_batch_size = batch_size
        dynamic_axis = False

    # Optional dynamic H/W on the mono ONNX. Read via OmegaConf .get with a
    # False default so existing specs (and the schema) remain unchanged;
    # activate via the spec field `export.dynamic_hw: True` (or via Hydra
    # override `+export.dynamic_hw=True`).
    try:
        dynamic_hw = bool(experiment_config.export.get("dynamic_hw", False))
    except Exception:
        dynamic_hw = False

    logging.info(f"Input batch size: {input_batch_size}")
    logging.info(f"Dynamic axis: {dynamic_axis}")
    logging.info(f"Dynamic H/W: {dynamic_hw}")

    device = 'cpu'
    if not on_cpu:
        device = 'cuda'

    # Set default output filename if the filename.
    if output_file is None:
        split_name = os.path.splitext(model_path)[0]
        output_file = "{}.onnx".format(split_name)

    # Create output directory
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Load model
    # FFS commercial ckpt is a research-pickled nn.Module (not a PL ckpt).
    # Route it through load_ffs_pretrained so we can export directly from
    # the bp2 raw weights without first detouring through a PL-format
    # wrapper. Mirrors scripts/{inference,evaluate,train}.py.
    if experiment_config.model.model_type == 'FastFoundationStereo':
        from nvidia_tao_pytorch.cv.depth_net.model.stereo_depth.fast_foundation_stereo.ckpt_utils import (
            load_ffs_pretrained,
        )
        from nvidia_tao_pytorch.cv.depth_net.model.build_pl_model import build_pl_model
        pl_model = build_pl_model(experiment_config, export=True)
        result = load_ffs_pretrained(pl_model.model, model_path)
        assert not result['missing'], (
            f"FFS ckpt missing keys: {result['missing']}")
        assert not result['unexpected'], (
            f"FFS ckpt unexpected keys: {result['unexpected']}")
        pl_model = pl_model.to(device)
    else:
        pl_model = get_pl_module(experiment_config).load_from_checkpoint(
            model_path,
            experiment_spec=experiment_config,
            export=True,  # to use regular Attention instead of Memory-efficient Attention for export
            map_location=device
        )

    model = pl_model.model
    export_fn = onnx_model_export(experiment_config.model.model_type)
    export_kwargs = dict(
        valid_iters=valid_iters,
        dynamic_axis=dynamic_axis,
        dynamic_hw=dynamic_hw,
    )
    export_fn(model,
              input_shape,
              input_batch_size,
              output_file,
              on_cpu,
              opset_version,
              **export_kwargs)


if __name__ == "__main__":
    main()
