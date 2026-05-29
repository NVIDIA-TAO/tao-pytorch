# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper utils."""

import os
import copy
import json
import random
import itertools
import numpy as np
import matplotlib.cm as cm
from functools import wraps
from contextlib import contextmanager
from typing import Tuple, Union, Optional, Iterator

import torch
from torch.optim.lr_scheduler import MultiStepLR
from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.mask2former.model.pl_model import rgetattr
from nvidia_tao_pytorch.cv.mask2former.utils.lr_scheduler import WarmupPolyLR
from nvidia_tao_pytorch.cv.mask2former.utils.solver import maybe_add_gradient_clipping
from nvidia_tao_pytorch.cv.nvpanoptix3d.dataloader.preprocessor import DatasetConstants
from nvidia_tao_pytorch.cv.nvpanoptix3d.utils.io import write_image


@contextmanager
def set_seed(seed: int) -> Iterator[None]:
    """Temporarily set Python + NumPy RNG seed and restore original state.

    Used to make random multi-scale resize choice *batch-consistent* in DDP when
    a batch is assembled from multiple DataLoader workers.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(int(seed))
        # numpy expects uint32 seed range
        np.random.seed(int(seed) % (2**32 - 1))
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    Ref: (https://github.com/facebookresearch/detectron2/blob/
    fd27788985af0f4ca800bca563acdb700bb890e2/detectron2/utils/memory.py)
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.
    Ref: (https://github.com/facebookresearch/detectron2/blob/
    fd27788985af0f4ca800bca563acdb700bb890e2/detectron2/utils/memory.py)

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu")
        return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)))
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped


def prepare_kept_mapping(
    model,
    cfg,
    dataset,
    frustum_mask=None,
    intrinsic=None,
):
    """
    Prepare kept and mapping tensors using back projection.

    Args:
        model: The model instance with back_projection method
        cfg: Configuration object
        dataset: Dataset name ('front3d' or others)
        frustum_mask: Optional frustum mask tensor
        intrinsic: Intrinsic matrix tensor

    Returns:
        tuple: (kept, mapping) tensors from back projection
    """
    intrinsic = adjust_intrinsic(
        intrinsic,
        tuple(cfg.dataset.target_size),
        tuple(cfg.dataset.reduced_target_size)
    )
    kept, mapping = model.back_projection(
        tuple(cfg.dataset.reduced_target_size[::-1]) + (256,),
        intrinsic,
        frustum_mask
    )
    return kept, mapping


def get_kept_mapping(model, cfg, batch, device):
    """
    Get kept and mapping for a batch of data (used for non-front3d datasets).

    Args:
        model: The model instance with back_projection method
        cfg: Configuration object
        batch: Batch data containing frustum_mask and intrinsic
        device: Device to place tensors on

    Returns:
        tuple: (kept, mapping) tensors
    """
    frustum_mask = batch["frustum_mask"].to(device)
    intrinsic = batch["intrinsic"].float().to(device)
    dataset = cfg.dataset.name

    kept, mapping = prepare_kept_mapping(
        model,
        cfg,
        dataset,
        frustum_mask=frustum_mask,
        intrinsic=intrinsic,
    )

    return kept, mapping


def get_metadata(cfg):
    """Build dataset metadata from a label-map JSON and config settings.

    Args:
        cfg: Experiment/config object. Expected to provide:
            - ``cfg.dataset.label_map``: path to a JSON label-map file
            - ``cfg.dataset.contiguous_id``: whether label-map IDs are already contiguous
            - ``cfg.model.sem_seg_head.num_classes``: number of semantic classes

    Returns:
        A dict with (at minimum) the following keys:
        - ``thing_classes`` / ``thing_colors``
        - ``stuff_classes`` / ``stuff_colors``
        - ``thing_dataset_id_to_contiguous_id`` / ``stuff_dataset_id_to_contiguous_id``
        - ``class_info``: the full category list loaded from JSON
    """
    label_map = cfg.dataset.label_map
    num_classes = cfg.model.sem_seg_head.num_classes
    with open(label_map, "r", encoding="utf-8") as f:
        categories = json.load(f)

    if not cfg.dataset.contiguous_id:
        categories_full = [
            {"name": "nan", "color": [0, 0, 0], "isthing": 1, "id": i + 1}
            for i in range(num_classes)
        ]
        for cat in categories:
            categories_full[cat["id"] - 1] = cat
        categories = categories_full

    meta = {}
    thing_classes = [k["name"] for k in categories]
    thing_colors = [k["color"] for k in categories]
    stuff_classes = [k["name"] for k in categories]
    stuff_colors = [k["color"] for k in categories]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in categories:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    # Create class_info combining all categories
    meta["class_info"] = categories
    return meta


def freeze_modules(model, freeze, status_logging):
    """Freeze (disable gradients for) a set of submodules by attribute path.

    Args:
        model: PyTorch module to modify in-place.
        freeze: Iterable of attribute paths (strings) to freeze (e.g. ``"backbone"`` or
            ``"model.encoder"``).
        status_logging: TAO status logging object used to emit RUNNING/SKIPPED messages.

    Returns:
        The same ``model`` instance (mutated).
    """
    freezed_modules = []
    skipped_modules = []
    for module in freeze:
        try:
            module_to_freeze = rgetattr(model, module)
            for p in module_to_freeze.parameters():
                p.requires_grad = False
            freezed_modules.append(module)
        except AttributeError:
            skipped_modules.append(module)
    if freezed_modules:
        status_logging.get_status_logger().write(
            message=f"Freezed module {freezed_modules}",
            status_level=status_logging.Status.RUNNING,
            verbosity_level=status_logging.Verbosity.INFO)
    if skipped_modules:
        status_logging.get_status_logger().write(
            message=f"module {skipped_modules} not found. Skipped freezing",
            status_level=status_logging.Status.SKIPPED,
            verbosity_level=status_logging.Verbosity.WARNING)
    return model


def configure_optimizers(cfg, model):
    """Create optimizer and LR scheduler configuration for training.

    Args:
        cfg: Experiment/config object. Expected to provide optimizer and scheduler
            configuration under ``cfg.train.optim`` as well as gradient clipping
            configuration under ``cfg.train``.
        model: Model whose parameters are to be optimized.

    Returns:
        A dict compatible with PyTorch Lightning-style ``configure_optimizers``
        return values, containing:
        - ``optimizer``: the constructed optimizer
        - ``lr_scheduler``: scheduler configuration dict (interval/frequency/etc.)
        - ``monitor``: metric name to monitor (if applicable)

    Raises:
        NotImplementedError: If the optimizer or scheduler type is not supported.
    """
    defaults = {}
    defaults["lr"] = cfg.train.optim.lr
    defaults["weight_decay"] = cfg.train.optim.weight_decay

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params = []
    memo = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if "backbone" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * cfg.train.optim.backbone_multiplier
            if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = 0.0
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = 0.0
            params.append({"params": [value], **hyperparams})

    def maybe_add_full_model_gradient_clipping(optim):
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_value = cfg.train.clip_grad_norm
        enable = (
            cfg.train.clip_grad_type == "full" and clip_norm_value > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_value)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    # CONFIG OPTIMIZER:
    optim_type = cfg.train.optim.type.lower()
    if optim_type == "sgd":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.train.optim.lr, momentum=cfg.train.optim.momentum)
    elif optim_type == "adamw":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.train.optim.lr)
    else:
        raise NotImplementedError(
            f"Optimizer type ({cfg.train.optim.type}) not supported.")

    if cfg.train.clip_grad_type != "full":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)

    # CONFIG LR SCHEDULER:
    total_iters = cfg.train.optim.max_steps
    if cfg.train.optim.lr_scheduler.lower() == "warmuppoly":  # Step based
        interval = "step"
        lr_scheduler = WarmupPolyLR(optimizer, total_iters,
                                    warmup_factor=cfg.train.optim.warmup_factor,
                                    warmup_iters=cfg.train.optim.warmup_iters,
                                    warmup_method="linear",
                                    last_epoch=-1,
                                    power=0.9,
                                    constant_ending=0.0)
    elif cfg.train.optim.lr_scheduler.lower() == "multistep":  # Epoch based
        interval = "epoch"
        lr_scheduler = MultiStepLR(
            optimizer, cfg.train.optim.milestones,
            gamma=cfg.train.optim.gamma
        )
    else:
        raise NotImplementedError(f"{cfg.train.optim.lr_scheduler} is not supported.")
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": interval,
            "frequency": 1,
            "step_after_optimizer": True
        },
        "monitor": cfg.train.optim.monitor_name}


def clear_cuda_cache():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def adjust_intrinsic(
    intrinsic: Union[np.array, torch.Tensor],
    intrinsic_image_dim: Tuple,
    image_dim: Tuple
) -> Union[np.array, torch.Tensor]:
    """
    Adjust intrinsic camera parameters for image dimension changes.

    Args:
        intrinsic: Camera intrinsic matrix (numpy array or torch tensor)
        intrinsic_image_dim: Original image dimensions (width, height)
        image_dim: Target image dimensions (width, height)

    Returns:
        Adjusted intrinsic matrix (same type as input)
    """
    if intrinsic_image_dim == image_dim:
        return intrinsic

    # Calculate scaling factors
    height_after = image_dim[1]
    height_before = intrinsic_image_dim[1]
    width_after = image_dim[0]
    width_before = intrinsic_image_dim[0]

    width_scale = float(width_after) / float(width_before)
    height_scale = float(height_after) / float(height_before)
    width_offset_scale = float(width_after - 1) / float(width_before - 1)
    height_offset_scale = float(height_after - 1) / float(height_before - 1)

    # handle numpy array case
    if isinstance(intrinsic, np.ndarray):
        intrinsic_return = np.copy(intrinsic)

        intrinsic_return[0, 0] *= width_scale
        intrinsic_return[1, 1] *= height_scale
        # account for cropping/padding here
        intrinsic_return[0, 2] *= width_offset_scale
        intrinsic_return[1, 2] *= height_offset_scale

        return intrinsic_return

    # handle torch tensor case
    elif isinstance(intrinsic, torch.Tensor):
        intrinsic_return = intrinsic.clone()

        intrinsic_return[:, 0, 0] *= width_scale
        intrinsic_return[:, 1, 1] *= height_scale

        intrinsic_return[:, 0, 2] *= width_offset_scale
        intrinsic_return[:, 1, 2] *= height_offset_scale

        return intrinsic_return

    else:
        raise TypeError(f"Unsupported input type: {type(intrinsic)}.")


def create_frustum_mask(
    intrinsics: np.ndarray,
    volume_shape: Tuple[int, int, int],
    depth_range: Tuple[float, float] = (0.1, 10.0),
    image_shape: Optional[Tuple[int, int]] = None,
    voxel_size: float = 0.01,
    padding_pixels: float = 0.0,
    volume_origin: Optional[np.ndarray] = None,
    z_axis_reversed: bool = False,
) -> np.ndarray:
    """
    Create a frustum mask for a voxel volume based on camera intrinsics.

    This function determines which voxels in a 3D volume are visible from a camera
    by checking if they project within the image bounds and depth range.

    Args:
        intrinsics: Camera intrinsic matrix (3x3 or 4x4)
        volume_shape: Shape of the voxel volume (nx, ny, nz)
        depth_range: Min and max depth in meters (z_min, z_max)
        image_shape: Image dimensions (height, width). If None, inferred from principal point
        voxel_size: Size of each voxel in meters (uniform for all axes)
        padding_pixels: Expand frustum bounds by this many pixels (can be negative to shrink)
        volume_origin: Origin of the volume in camera space. If None, centered at camera
        z_axis_reversed: If True, z-index 0 is farthest, index nz-1 is nearest (default: False)

    Returns:
        frustum_mask: Boolean mask of shape volume_shape indicating voxels inside frustum
    """
    # Input validation
    if not isinstance(intrinsics, np.ndarray):
        intrinsics = np.array(intrinsics)

    assert intrinsics.shape in [(3, 3), (4, 4)], f"Intrinsics must be 3x3 or 4x4, got shape {intrinsics.shape}"
    assert voxel_size > 0, f"voxel_size must be positive, got {voxel_size}"
    assert depth_range[0] < depth_range[1], f"depth_range must be (min, max) with min < max, got {depth_range}"
    assert depth_range[0] > 0, f"depth_range min must be positive, got {depth_range[0]}"

    # Extract camera parameters from intrinsics
    if intrinsics.shape == (4, 4):
        K = intrinsics[:3, :3]
    else:
        K = intrinsics

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Determine image shape if not provided
    if image_shape is None:
        # Assume principal point is at image center
        image_height = int(2 * cy)
        image_width = int(2 * cx)
    else:
        image_height, image_width = image_shape

    # Apply padding to image bounds
    u_min = -padding_pixels
    u_max = image_width + padding_pixels
    v_min = -padding_pixels
    v_max = image_height + padding_pixels

    # Set volume origin if not provided
    if volume_origin is None:
        # Center the volume in camera space
        # X and Y: center laterally
        # Z: center around the middle of the depth range
        volume_origin = np.array([
            -(volume_shape[0] * voxel_size) / 2,
            -(volume_shape[1] * voxel_size) / 2,
            (depth_range[0] + depth_range[1]) / 2 - (volume_shape[2] * voxel_size) / 2
        ])

    # Create voxel grid coordinates with uniform spacing
    x_coords = np.arange(volume_shape[0]) * voxel_size + volume_origin[0]
    y_coords = np.arange(volume_shape[1]) * voxel_size + volume_origin[1]
    z_coords = np.arange(volume_shape[2]) * voxel_size + volume_origin[2]

    # Optionally reverse Z-axis so index 0 is farthest
    if z_axis_reversed:
        z_coords = z_coords[::-1]

    # Create meshgrid for all voxel centers
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    # Stack into voxel centers array (N, 3)
    voxel_centers = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    # Check depth constraint (using camera-space Z coordinate)
    depth_mask = (voxel_centers[:, 2] >= depth_range[0]) & (voxel_centers[:, 2] <= depth_range[1])

    # Project voxel centers to image plane using pinhole camera model
    # Avoid division by zero
    valid_depth = voxel_centers[:, 2] > 1e-6
    u = np.full(len(voxel_centers), -1.0)
    v = np.full(len(voxel_centers), -1.0)

    u[valid_depth] = (fx * voxel_centers[valid_depth, 0] / voxel_centers[valid_depth, 2]) + cx
    v[valid_depth] = (fy * voxel_centers[valid_depth, 1] / voxel_centers[valid_depth, 2]) + cy

    # Check if projected points are within image bounds (with optional padding)
    image_mask = (u >= u_min) & (u < u_max) & (v >= v_min) & (v < v_max)

    # Combine masks: must satisfy depth range, image bounds, and valid depth
    frustum_mask_1d = depth_mask & image_mask & valid_depth

    # Reshape to volume shape
    frustum_mask = frustum_mask_1d.reshape(volume_shape)

    return frustum_mask


def create_color_palette():
    """Return a fixed RGB palette used for mesh / semantic visualization.

    Returns:
        A list of ``(R, G, B)`` tuples in 0-255 integer space.

    Notes:
        The palette is intentionally stable across runs to make qualitative
        comparisons easier.
    """
    return [
        (0, 0, 0),
        (174, 199, 232),		# wall
        (152, 223, 138),		# floor
        (31, 119, 180), 		# cabinet
        (255, 187, 120),		# bed
        (188, 189, 34), 		# chair
        (140, 86, 75),  		# sofa
        (255, 152, 150),		# table
        (214, 39, 40),  		# door
        (197, 176, 213),		# window
        (148, 103, 189),		# bookshelf
        (196, 156, 148),		# picture
        (23, 190, 207), 		# counter
        (178, 76, 76),
        (247, 182, 210),		# desk
        (66, 188, 102),
        (219, 219, 141),		# curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14), 		# refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),		# shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  		# toilet
        (112, 128, 144),		# sink
        (96, 207, 209),
        (227, 119, 194),		# bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  		# otherfurn
        (100, 85, 144),
        (172, 172, 172),
    ]


def visualize_2d_predictions(
    output_dir: str,
    frame_name: str,
    processed_output: dict,
    image: Optional[torch.Tensor] = None,
    depth_cmap: str = "viridis",
) -> None:
    """
    Save 2D predictions as a single combined image with 4 columns.

    Output layout: [Input Image | Instance | Semantic | Depth]

    Args:
        output_dir: Directory to save output image.
        frame_name: Base name for output file.
        processed_output: Dict from model postprocessor containing:
            - panoptic_seg: Tuple of (panoptic_map (H, W), segments_info list)
            - depth: Depth map tensor (H, W)
            - semantic_seg: Semantic probability masks (N_classes, H, W; stored under "sem_seg")
        image: Input RGB image tensor (C, H, W) for visualization.
        depth_cmap: Matplotlib colormap name for depth visualization.
    """
    os.makedirs(output_dir, exist_ok=True)

    panoptic_seg, segments_info = processed_output["panoptic_seg"]
    depth_map = processed_output["depth"]

    # Convert tensors to numpy
    if isinstance(panoptic_seg, torch.Tensor):
        panoptic_seg = panoptic_seg.detach().cpu().numpy()
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()

    h, w = panoptic_seg.shape

    # --- Input image ---
    if image is not None:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.max() > 1.0:
            image = image / 255.0
        image_rgb = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # --- Semantic segmentation ---
    seg_to_cat = {seg["id"]: seg["category_id"] for seg in segments_info}
    semantic_map = np.zeros((h, w), dtype=np.int32)
    for seg_id, cat_id in seg_to_cat.items():
        semantic_map[panoptic_seg == seg_id] = cat_id

    semantic_colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in DatasetConstants.SEMANTIC_CLASS_COLORS.items():
        semantic_colored[semantic_map == class_id] = color

    # --- Instance segmentation ---
    instance_colored = np.zeros((h, w, 3), dtype=np.uint8)
    instance_palette = np.array(create_color_palette(), dtype=np.uint8)
    for i, seg_info in enumerate(segments_info):
        seg_id = seg_info["id"]
        cat_id = seg_info["category_id"]
        if cat_id in DatasetConstants.SEMANTIC_CLASS_COLORS:
            base_color = np.array(DatasetConstants.SEMANTIC_CLASS_COLORS[cat_id], dtype=np.uint8)
        else:
            base_color = instance_palette[(cat_id + 1) % len(instance_palette)]
        variation = ((i * 37) % 60) - 30
        color = np.clip(base_color.astype(np.int32) + variation, 0, 255).astype(np.uint8)
        instance_colored[panoptic_seg == seg_id] = color

    # --- Depth visualization ---
    valid_mask = depth_map > 0
    if valid_mask.any():
        depth_min_val = depth_map[valid_mask].min()
        depth_max_val = depth_map[valid_mask].max()
        depth_normalized = np.clip(
            (depth_map - depth_min_val) / (depth_max_val - depth_min_val + 1e-8),
            0, 1
        )
    else:
        depth_normalized = np.zeros_like(depth_map)

    # Apply colormap to depth
    cmap = cm.get_cmap(depth_cmap)
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)

    # --- Combine into single image: [Input | Instance | Semantic | Depth] ---
    combined = np.concatenate([image_rgb, instance_colored, semantic_colored, depth_colored], axis=1)

    output_path = os.path.join(output_dir, f"{frame_name}_viz.png")
    write_image(combined, output_path)
