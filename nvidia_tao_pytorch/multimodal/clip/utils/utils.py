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

"""CLIP utils."""

import os

import numpy as np
import torch
from apex.optimizers import FusedLAMB
from tabulate import tabulate
from torch.optim import AdamW

from nvidia_tao_pytorch.core.tlt_logging import logging


SUPPORTED_CHECKPOINT_EXTENSIONS = {'.pth', '.ckpt'}


def register_checkpoint_safe_globals():
    """Register numpy types as safe globals for PyTorch 2.6+ checkpoint loading.

    PyTorch 2.6+ uses weights_only=True by default when loading checkpoints,
    which requires explicit allowlisting of non-tensor types. This function
    registers numpy types commonly found in checkpoints (e.g., from HuggingFace
    or older training runs) to allow safe loading.

    Should be called early in scripts that load checkpoints via PyTorch Lightning
    or torch.load with weights_only=True.
    """
    try:
        from numpy._core.multiarray import scalar as np_scalar
    except ImportError:
        from numpy.core.multiarray import scalar as np_scalar

    torch.serialization.add_safe_globals([
        np_scalar,
        np.dtype,
        np.ndarray,
    ])


VALID_OPTIMIZER_TYPES = {'adamw', 'lamb'}
VALID_SCHEDULERS = {'cosine', 'constant', 'linear'}


def _is_bias_or_norm(name, param):
    """Return True if param should be excluded from weight decay."""
    return (param.ndim < 2 or "bn" in name or "ln" in name or
            "bias" in name or "logit_scale" in name)


def _create_optimizer(optimizer_type, param_groups, lr, betas, eps):
    """Instantiate an optimizer by type name.

    Parameters
    ----------
    optimizer_type : str
        'adamw' or 'lamb'.
    param_groups : list[dict]
        Parameter groups with 'params' and 'weight_decay' keys.
    lr : float
        Base learning rate.
    betas : list[float]
        Beta parameters.
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    torch.optim.Optimizer
    """
    if optimizer_type == 'adamw':
        return AdamW(param_groups, lr=lr, betas=tuple(betas), eps=eps)
    elif optimizer_type == 'lamb':
        return FusedLAMB(param_groups, lr=lr, betas=tuple(betas), eps=eps)
    else:
        raise ValueError(
            f"Unknown optimizer_type '{optimizer_type}'. "
            f"Supported: {VALID_OPTIMIZER_TYPES}"
        )


def _make_tower_groups(named_params, cfg, tower_label):
    """Split named parameters into bias/norm and rest groups.

    Parameters
    ----------
    named_params : iterable of (str, Parameter)
        Named parameters for a single tower.
    cfg : config object
        Must have lr, weight_decay, betas, eps attributes.
    tower_label : str
        Label for logging (e.g., 'vision', 'text', 'logit').

    Returns
    -------
    list[dict]
        Two param groups: [bias_norm_group, rest_group].
        Each group has metadata keys '_tower' and '_is_bias_norm'.
    """
    bias_norm = []
    rest = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if _is_bias_or_norm(name, param):
            bias_norm.append(param)
        else:
            rest.append(param)

    groups = []
    if bias_norm:
        groups.append({
            'params': bias_norm,
            'lr': cfg.lr,
            'betas': tuple(cfg.betas),
            'eps': cfg.eps,
            'weight_decay': 0.0,
            '_tower': tower_label,
            '_is_bias_norm': True,
        })
    if rest:
        groups.append({
            'params': rest,
            'lr': cfg.lr,
            'betas': tuple(cfg.betas),
            'eps': cfg.eps,
            'weight_decay': cfg.weight_decay,
            '_tower': tower_label,
            '_is_bias_norm': False,
        })
    return groups


class _TowerCfg:
    """Lightweight config holder for a single tower's optimizer settings."""

    __slots__ = ('lr', 'weight_decay', 'betas', 'eps')

    def __init__(self, lr, weight_decay, betas, eps):
        """Initialize tower config."""
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps


def build_optimizer(model, train_cfg):
    """Build optimizer with per-tower parameter groups.

    Parameters
    ----------
    model : BaseCLIPAdapter
        Model with vision_named_parameters() and text_named_parameters().
    train_cfg : CLIPTrainConfig
        Training config with optim containing vision_lr/text_lr.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer with per-tower parameter groups.
    """
    cfg = train_cfg.optim

    v_cfg = _TowerCfg(
        lr=cfg.vision_lr, weight_decay=cfg.weight_decay,
        betas=cfg.betas, eps=cfg.eps,
    )
    t_cfg = _TowerCfg(
        lr=cfg.text_lr, weight_decay=cfg.weight_decay,
        betas=cfg.betas, eps=cfg.eps,
    )

    param_groups = []
    param_groups.extend(
        _make_tower_groups(model.vision_named_parameters(), v_cfg, 'vision')
    )
    param_groups.extend(
        _make_tower_groups(model.text_named_parameters(), t_cfg, 'text')
    )
    param_groups.extend(
        _make_tower_groups(model.other_named_parameters(), t_cfg, 'logit')
    )

    if not param_groups:
        logging.warning("No trainable parameters found.")
        return AdamW([{'params': []}], lr=cfg.vision_lr)

    optimizer = _create_optimizer(
        cfg.optimizer_type, param_groups,
        lr=cfg.vision_lr, betas=cfg.betas, eps=cfg.eps
    )

    _log_optimizer_summary(cfg, param_groups)
    return optimizer


def _log_optimizer_summary(cfg, param_groups):
    """Log a summary of the optimizer configuration."""
    vision_params = sum(
        p.numel() for g in param_groups if g['_tower'] == 'vision'
        for p in g['params']
    )
    text_params = sum(
        p.numel() for g in param_groups if g['_tower'] == 'text'
        for p in g['params']
    )
    logit_params = sum(
        p.numel() for g in param_groups if g['_tower'] == 'logit'
        for p in g['params']
    )

    table_data = [
        ["vision", f"{vision_params:,}", f"{cfg.vision_lr:.2e}",
         f"{cfg.weight_decay:.2e}", cfg.warmup_steps, cfg.scheduler],
        ["text", f"{text_params:,}", f"{cfg.text_lr:.2e}",
         f"{cfg.weight_decay:.2e}", cfg.warmup_steps, cfg.scheduler],
        ["logit", f"{logit_params:,}", "", "", "", ""],
    ]
    headers = ["Tower", "Params", "LR", "WD", "Warmup", "Schedule"]
    table = tabulate(table_data, headers=headers, tablefmt="simple")

    logging.info(f"Optimizer: {cfg.optimizer_type.upper()}\n{table}")


# ---- LR Schedulers ----

def compute_lr(step, base_lr, warmup_steps, max_steps, scheduler='cosine'):
    """Compute learning rate for a given step.

    Parameters
    ----------
    step : int
        Current training step.
    base_lr : float
        Base learning rate (peak after warmup).
    warmup_steps : int
        Steps for linear warmup.
    max_steps : int
        Total training steps.
    scheduler : str
        'cosine', 'constant', or 'linear'.

    Returns
    -------
    float
        Learning rate for the current step.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)

    if scheduler == 'constant':
        return base_lr

    progress = step - warmup_steps
    total = max(max_steps - warmup_steps, 1)

    if scheduler == 'linear':
        return base_lr * max(1.0 - progress / total, 0.0)

    # cosine (default)
    return 0.5 * (1 + np.cos(np.pi * progress / total)) * base_lr


def load_model_from_checkpoint(model_path, experiment_config, model_class):
    """Load CLIP model from checkpoint.

    Parameters
    ----------
    model_path : str
        Path to the model checkpoint file.
    experiment_config : ExperimentConfig
        Experiment configuration object.
    model_class : class
        The PyTorch Lightning model class to load (e.g., CLIPPlModel).

    Returns
    -------
    model
        Loaded PyTorch Lightning model.

    Raises
    ------
    NotImplementedError
        If the model format is not supported.
    """
    register_checkpoint_safe_globals()
    ext = os.path.splitext(model_path)[1].lower()

    if ext in SUPPORTED_CHECKPOINT_EXTENSIONS:
        model = model_class.load_from_checkpoint(
            model_path,
            map_location="cpu",
            experiment_spec=experiment_config
        )
        logging.info(f"Model loaded from {model_path}")
        return model

    if ext == '.engine':
        raise NotImplementedError(
            "TensorRT inference is supported through "
            "tao-deploy, not tao-pytorch."
        )

    raise NotImplementedError(
        f"Model format '{ext}' is not supported. "
        f"Supported formats: {SUPPORTED_CHECKPOINT_EXTENSIONS}"
    )
