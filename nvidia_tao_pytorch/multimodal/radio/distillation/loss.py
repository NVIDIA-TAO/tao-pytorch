# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distillation Loss module for knowledge distillation."""
import os
import inspect
import math
from typing import Union, List, Tuple, Dict, Optional
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.distillation.losses import LPCriterion, KLDivCriterion
from nvidia_tao_pytorch.cv.backbone_v2.radio import RADIO
from nvidia_tao_pytorch.multimodal.radio.dataloader.transforms.generate_homography_grid import (
    generate_homography_grid,
)
from nvidia_tao_pytorch.multimodal.radio.distillation.hadamard import get_hadamard_matrix
from nvidia_tao_pytorch.multimodal.radio.dataloader.dataset import NOCLASS_IDX
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank, get_world_size


class Cross_Entropy(nn.Module):
    """Cross Entropy Loss with label smoothing.

    Args:
        weight (Tensor): A manual rescaling weight given to each class.
        label_smoothing (float): The label smoothing value.
        soft (bool): If True, allow soft label from a teacher model.
    """

    def __init__(self, weight=None, label_smoothing=0.1, soft=False):
        super().__init__()
        self.soft = soft
        if soft:
            self.loss = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            self.loss = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                reduction="mean",
                ignore_index=NOCLASS_IDX,
            )

    def forward(self, pred, target):
        """Forward pass."""
        return self.loss(pred, target)


def _mse_element_wise(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-element squared error (EVFM-style reduction='none')."""
    return (pred - target) ** 2


def dampened_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dampened MSE (EVFM-style): for |diff| < 1 use 0.5*diff^2, else 2*sqrt(|diff|+eps)-1.5.

    Reduces sensitivity to large residuals.
    """
    diff = pred - target
    abs_diff = torch.abs(diff)
    dampened = 2 * torch.sqrt(abs_diff + 1e-8) - 1.5
    mse = 0.5 * (diff ** 2)
    return torch.where(abs_diff.detach() < 1, mse, dampened)


def masked_sum(t: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
    """Compute a masked sum and masked count.

    Args:
        t: Input tensor to be reduced.
        mask: Boolean mask indicating which elements of `t` to include. Must be
            broadcastable to `t`.
        **kwargs: Extra keyword arguments forwarded to `Tensor.sum` (e.g., `dim`).

    Returns:
        Tuple of `(sum, count)` where `sum` is the masked sum over `t` and `count`
        is the number of valid (True) elements aggregated with a matching `dtype`.
    """
    s = torch.where(mask, t, 0).sum(**kwargs)
    a2 = dict(kwargs)
    if 'dtype' not in a2:
        a2['dtype'] = s.dtype
    ct = mask.sum(**a2)

    return s, ct


def masked_mean(t: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
    """Compute a masked mean.

    Args:
        t: Input tensor to be averaged.
        mask: Boolean mask indicating valid elements. Must be broadcastable to `t`.
        **kwargs: Extra keyword arguments forwarded to `masked_sum`/`Tensor.sum`.

    Returns:
        The masked mean over `t`.
    """
    s, ct = masked_sum(t, mask, **kwargs)
    return s / ct


class LossFnStateBase(nn.Module):
    """Base class for maintaining running state for feature normalization losses.

    Tracks masked running statistics over teacher features (sample count and sum)
    and exposes helper transformations for targets, student features, and loss.
    Also supports distributed synchronization of its internal state.
    """

    def __init__(self, name: str, feature_dim: int, ohem: bool):
        """Initialize base state.

        Args:
            name: Identifier for logging and cache naming.
            feature_dim: Feature channel dimension being tracked.
            ohem: Whether Online Hard Example Mining is enabled (reserved flag).
        """
        super().__init__()
        self.name = name
        self.feature_dim = feature_dim
        self.ohem = ohem
        self.dist_group: dist.ProcessGroup = None

        self.register_buffer('fwd_count', torch.tensor(0, dtype=torch.float64), persistent=True)
        self.register_buffer('num_samples', torch.tensor(0.0, dtype=torch.float64), persistent=True)
        self.register_buffer('sample_sum', torch.zeros(feature_dim, dtype=torch.float64), persistent=True)

    def masked_mean(self, t: torch.Tensor, mask: torch.Tensor, **kwargs):
        """Masked mean helper that expands a spatial mask over channel dimension."""
        return masked_mean(t, mask.unsqueeze(1).expand(-1, self.feature_dim, -1, -1), **kwargs)

    def masked_sum(self, t: torch.Tensor, mask: torch.Tensor, **kwargs):
        """Masked sum helper that expands a spatial mask to match `t` shape."""
        s, ct = masked_sum(t, mask.unsqueeze(1).expand_as(t), **kwargs)
        return s, ct[0]

    @property
    def expected_mean(self):
        """Current estimate of the per-channel mean from accumulated statistics."""
        return torch.where(self.num_samples > 0, self.sample_sum / self.num_samples, 0)

    @torch.no_grad()
    def update(self, loss_fn_base, teacher_features: torch.Tensor, loss_mask: torch.Tensor):
        """Accumulate masked running statistics from teacher features.

        Args:
            loss_fn_base: Unused placeholder for compatibility with derived classes.
            teacher_features: Teacher feature map of shape [B, C, H, W].
            loss_mask: Boolean mask of shape [B, H, W] selecting valid positions.

        Returns:
            Updated expected mean tensor with shape [C].
        """
        self.fwd_count += 1

        sample_sum, num_samples = self.masked_sum(teacher_features, loss_mask, dim=(0, 2, 3), dtype=torch.float64)

        if dist.is_initialized():
            dist.all_reduce(sample_sum, op=dist.ReduceOp.SUM, group=self.dist_group)
            dist.all_reduce(num_samples, op=dist.ReduceOp.SUM, group=self.dist_group)

        self.sample_sum += sample_sum
        self.num_samples += num_samples

        return self.expected_mean

    def transform_targets(self, teacher_features: torch.Tensor) -> torch.Tensor:
        """Transform teacher features into the target space (identity by default)."""
        return teacher_features

    def transform_student(self, student_features: torch.Tensor) -> torch.Tensor:
        """Transform student features into the same space as the targets (identity)."""
        return student_features

    def transform_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Optionally transform the computed loss (identity by default)."""
        return loss

    def modify_linear(self, final: nn.Linear):
        """Optionally modify a final linear layer to account for normalization."""
        pass

    def get_state_components(self):
        """Return a flat dict of scalar state components for logging/monitoring."""
        ret = dict()
        self.add_state_components(ret)
        return ret

    def add_state_components(self, components: dict):
        """Populate external dict with scalar state components (override in subclasses)."""
        pass

    @torch.no_grad()
    def synchronize(self):
        """Synchronize internal buffers across processes in the distributed group."""
        if not dist.is_initialized():
            return

        src_rank = self._global_rank_for_group_rank()

        if src_rank >= 0:
            self._broadcast(src_rank)

    def _global_rank_for_group_rank(self, target_rank: int = 0, reduction_group: dist.ProcessGroup = None):
        """Resolve the global rank corresponding to a rank within `self.dist_group`.

        Args:
            target_rank: Rank within the group to act as the source.
            reduction_group: Group over which to reduce for selection. Defaults to `self.dist_group`.

        Returns:
            Global rank integer of the selected source, or -1 if none.
        """
        if not dist.is_initialized():
            return target_rank

        group_rank = dist.get_rank(self.dist_group)
        global_rank = dist.get_rank()

        # Figure out which rank runs the broadcast
        src_rank = torch.tensor(global_rank if group_rank == target_rank else -1, dtype=torch.int32, device='cuda')
        dist.all_reduce(src_rank, op=dist.ReduceOp.MAX, group=reduction_group)
        src_rank = src_rank.item()
        return src_rank

    def _broadcast(self, src_rank: int, group: dist.ProcessGroup = None):
        """Broadcast internal buffers from `src_rank` to all processes in `group`."""
        dist.broadcast(self.fwd_count, src_rank, group=group)
        dist.broadcast(self.num_samples, src_rank, group=group)
        dist.broadcast(self.sample_sum, src_rank, group=group)


class WhitenNormState(LossFnStateBase):
    """Maintain whitening/denormalization projections estimated from teacher features.

    Periodically updates a whitening projection and its inverse based on running
    covariance estimates computed under a spatial mask, with optional caching and
    distributed synchronization.
    """

    def __init__(self, name: str, feature_dim: int, ohem: bool, update_period: int = 100):
        """Initialize whitening state and running statistics.

        Args:
            name: Identifier for logging/caching.
            feature_dim: Channel dimension of features.
            ohem: OHEM flag (reserved).
            update_period: Steps between projection updates.
        """
        super().__init__(name, feature_dim, ohem)
        self.update_period = update_period
        self.register_buffer('eye', torch.eye(feature_dim, dtype=torch.float64), persistent=False)
        self.register_buffer('inv_whiten', self.eye.clone(), persistent=True)
        self.register_buffer('whiten', self.eye.clone(), persistent=True)
        self.register_buffer('cov_sum', torch.zeros(feature_dim, feature_dim, dtype=torch.float64), persistent=True)

    @property
    def covariance(self):
        """Sample covariance matrix estimated from accumulated sums."""
        return self.cov_sum / (self.num_samples - 1)

    @property
    def max_samples(self) -> int:
        """Maximum number of samples to use for estimating the projections."""
        return 30 * self.update_period

    @torch.no_grad()
    @torch.autocast('cuda', enabled=False)
    def update(self, loss_fn_base, teacher_features: torch.Tensor, loss_mask: torch.Tensor):
        """Update running statistics and periodically refresh whitening projections."""
        fwd_count = int(self.fwd_count.item())

        if fwd_count == 0 and self._load_from_cache(teacher_features):
            return

        # Annoyingly, `eigh`, `svd`, and `eig` aren't stable for producing the eigenvectors,
        # which means that this method will consistently produce different rotations.
        # The good news is that once we get enough samples, we're pretty close to the expectation, and we can
        # stop re-estimating this.
        if fwd_count > self.max_samples:
            self.fwd_count += 1
            return

        self._update_samples(loss_fn_base, teacher_features, loss_mask)

        if self.num_samples.item() < 2:
            self.fwd_count.zero_()
            return

        if fwd_count % self.update_period == 0:
            self._wrap_update_projections(fwd_count)
            self._calc_projection_error()

        if fwd_count == self.max_samples:
            self._save_cache(teacher_features)

    def _get_cache_path(self, teacher_features: torch.Tensor):
        """Compute a cache file path for storing/restoring projection state."""
        resolution = teacher_features.shape[-2:]
        if dist.is_initialized():
            resolutions = [None for _ in range(get_world_size(self.dist_group))]
            dist.all_gather_object(resolutions, resolution, group=self.dist_group)
            resolution = '-'.join(f'{y}x{x}' for y, x in sorted(set(resolutions)))
        else:
            resolution = f'{resolution[0]}x{resolution[1]}'

        safe_name = self.name.replace('(', '_').replace(')', '_').replace(' ', '_').replace(',', '-')
        fname = f'{safe_name}_res-{resolution}.pth'
        cache_dir = os.path.join(torch.hub.get_dir(), 'evfm', 'fd_loss_states', 'whiten')
        # cache_dir = os.path.join(torch.hub.get_dir(), 'evfm', 'fd_loss_states', 'whiten-4part')
        cache_path = os.path.join(cache_dir, fname)
        return cache_path

    def _load_from_cache(self, teacher_features: torch.Tensor) -> bool:
        """Load projections from cache if available.

        Returns:
            True if state was loaded successfully, False otherwise.
        """
        return False

    def _save_cache(self, teacher_features: torch.Tensor):
        """Persist current projection state to cache (no-op by default)."""
        pass

    def _update_samples(self, loss_fn_base, teacher_features: torch.Tensor, loss_mask: torch.Tensor):
        """Accumulate masked sums and covariance from a chunk of teacher features.

        Returns:
            Tuple of (expected_mean, flattened_features) for downstream processing.
        """
        flat_feat = rearrange(teacher_features, 'b c h w -> (b h w) c')
        flat_mask = loss_mask.flatten()

        batch_sum, batch_num_samples = self.masked_sum(flat_feat, flat_mask, dim=0, dtype=torch.float64)

        if dist.is_initialized():
            dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM, group=self.dist_group)
            dist.all_reduce(batch_num_samples, op=dist.ReduceOp.SUM, group=self.dist_group)

        if batch_num_samples.item() == 0:
            return self.expected_mean, flat_feat

        self.fwd_count += 1

        batch_mean = batch_sum / batch_num_samples.clamp_min(1)
        mean_delta = batch_mean - self.expected_mean

        self.num_samples += batch_num_samples
        self.sample_sum += batch_sum

        chunk_centered = flat_feat - batch_mean
        chunk_centered = torch.where(flat_mask.unsqueeze(1), chunk_centered, 0)
        cov_chunk = chunk_centered.T @ chunk_centered

        if dist.is_initialized():
            dist.all_reduce(cov_chunk, op=dist.ReduceOp.SUM, group=self.dist_group)

        correction = mean_delta[:, None] * mean_delta[None, :] * batch_num_samples * (self.num_samples - batch_num_samples) / self.num_samples

        self.cov_sum += cov_chunk + correction

        return self.expected_mean, flat_feat

    def _wrap_update_projections(self, fwd_count: int):
        """Update projections and log change energy; then broadcast in distributed runs."""
        inv_whiten = self.inv_whiten.clone()
        whiten = self.whiten.clone()

        self._update_projections(fwd_count)

        if get_global_rank(self.dist_group) == 0:
            # This allows us to measure how much the projections are changing
            # by measuring how close the new estimate is to reconstructing the
            # identity matrix given the old estimate.
            p2 = self.inv_whiten @ whiten - self.eye
            p3 = inv_whiten @ self.whiten - self.eye
            energy = (p2 + p3) / 2
            logging.info(f'Rotation Change Energy: {energy.norm().item():.6f}')

        if dist.is_initialized():
            group_rank_0_global_rank = self._global_rank_for_group_rank(reduction_group=self.dist_group)
            self._broadcast(group_rank_0_global_rank, self.dist_group)
        pass

    def _update_projections(self, fwd_count: int):
        """Compute `whiten` and `inv_whiten` from the current covariance estimate.

        Implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this!")

    @torch.autocast('cuda', enabled=False)
    def transform_targets(self, teacher_features: torch.Tensor) -> torch.Tensor:
        """Apply whitening transform to teacher targets for normalized training."""
        b, c, h, w = teacher_features.shape

        flat_feat = rearrange(teacher_features, 'b c h w -> (b h w) c')

        flat_feat = flat_feat - self.expected_mean.unsqueeze(0)

        flat_white = flat_feat @ self.whiten.T

        teacher_features = rearrange(flat_white, '(b h w) c -> b c h w',
                                     b=b, c=c, h=h, w=w).to(teacher_features.dtype)

        if get_global_rank(self.dist_group) == 0 and int(self.fwd_count.item()) % 50 == 0:
            whiten_error = (torch.cov(flat_white.T) - self.eye).abs().mean()
            logging.info(f'Whiten Error ({self.name}): {whiten_error.item()}')

        return teacher_features

    @torch.no_grad()
    def transform_student(self, student_features: torch.Tensor) -> torch.Tensor:
        """Invert whitening to return student features to the original space."""
        mean = self.expected_mean.to(student_features.dtype)
        inv_whiten = self.inv_whiten.to(student_features.dtype)

        b, c, h, w = student_features.shape

        flat_feat = rearrange(student_features, 'b c h w -> (b h w) c')

        flat_feat = flat_feat @ inv_whiten.T
        flat_feat = flat_feat + mean

        student_features = rearrange(flat_feat, '(b h w) c -> b c h w', b=b, c=c, h=h, w=w)

        return student_features

    def modify_linear(self, final: nn.Linear):
        """De-normalize a final linear layer to match the unwhitened feature space."""
        logging.info(f'De-normalizing linear layer! Method: {type(self).__name__}')
        m = self.expected_mean.to(final.weight.dtype)
        w = self.inv_whiten.to(final.weight.dtype)

        replicas = final.weight.shape[0] // w.shape[1]
        bw = w[None].expand(replicas, -1, -1)

        bfinal_weight = rearrange(final.weight, '(r h) c -> r h c', r=replicas, h=bw.shape[-1])

        bw2 = torch.bmm(bw, bfinal_weight)

        w2 = rearrange(bw2, 'r h c -> (r h) c')
        final.weight.data.copy_(w2)

        if final.bias is not None:
            bfinal_bias = rearrange(final.bias, '(r h c) -> r h c', r=replicas, h=bw.shape[-1], c=1)

            bb2 = torch.bmm(bw, bfinal_bias)

            b2 = bb2.flatten()
            final.bias.data.copy_(b2)

            final.bias.data += m.repeat(replicas)

    def _calc_projection_error(self):
        """Log magnitude statistics of the inverse whitening columns for monitoring."""
        if get_global_rank(self.dist_group) != 0:
            return

        # Measure the magnitude error for each input
        norm = self.inv_whiten.norm(dim=0)

        minVal = norm.amin().item()
        maxVal = norm.amax().item()
        valRange = maxVal - minVal

        logging.info(f'Projection Error Mag - Mean: {norm.mean().item():.4f}, Min: {minVal:.4f}, Max: {maxVal:.4f}, Std: {norm.std().item():.4f}, Range: {valRange:.4f}')
        pass

    def _eig_decomp(self, cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Robust eigen-decomposition with scaling and small-value handling.

        Args:
            cov: Covariance matrix.

        Returns:
            Tuple `(eigenvalues, eigenvectors, mask)` where `mask` indicates
            retained eigenvalues after thresholding.
        """
        # To deal with dead neurons
        cov = torch.where(cov != 0, cov, 1e-10 * self.eye)

        factor = 1 / cov.diag().median()
        cov = cov * factor

        # # L is the eigenvalue vector
        # # V is the eigenvector matrix, in column format
        # # L, V = torch.linalg.eigh(cov)
        # V, L, _ = torch.linalg.svd(cov)

        L, V = torch.linalg.eigh(cov)

        # threshold = L.amax() * L.shape[0] * torch.finfo(L.dtype).eps
        threshold = 0
        mask = L > threshold

        L /= factor

        return L, V, mask

    def _broadcast(self, src_rank: int, group: dist.ProcessGroup = None):
        """Broadcast whitening projections and covariance buffers across processes."""
        super()._broadcast(src_rank, group)
        dist.broadcast(self.inv_whiten, src_rank, group=group)
        dist.broadcast(self.whiten, src_rank, group=group)
        dist.broadcast(self.cov_sum, src_rank, group=group)


class PHIStandardization(WhitenNormState):
    """PHI standardization that whitens by average spectrum and optional rotation.

    Uses an orthogonal Hadamard rotation for stable whitening direction, combined with
    a scalar alpha derived from the mean eigenvalue to scale features.
    """

    def __init__(self, name: str, feature_dim: int, ohem: bool, update_period: int = 100, rotate: bool = True):
        """Initialize PHI standardization module.

        Args:
            name: Identifier for logging/caching.
            feature_dim: Channel dimension of features.
            ohem: OHEM flag (reserved).
            update_period: Steps between projection updates.
            rotate: Whether to apply Hadamard-based rotation before scaling.
        """
        super().__init__(name, feature_dim, ohem, update_period)

        self.rotate = rotate

        H = get_hadamard_matrix(feature_dim)
        if dist.is_initialized():
            dist.broadcast(H, src=0)
        self.register_buffer('rotation', H, persistent=True)
        self.register_buffer('alpha', torch.tensor(0, dtype=torch.float32, device=H.device))

    def _update_projections(self, fwd_count: int):
        """Compute PHI whitening using mean eigenvalue scaling and optional rotation."""
        cov = self.covariance

        L, V = torch.linalg.eigh(cov)
        mask = L >= 0
        L = torch.where(mask, L, 0)

        alpha = L.mean().rsqrt()
        inv_alpha = 1 / alpha

        self.alpha.copy_(alpha)

        if self.rotate:
            rotation: torch.Tensor = self.rotation
            w_rot = rotation @ V.T
            inv_rot = V @ rotation.T
        else:
            w_rot = inv_rot = torch.eye(self.feature_dim, dtype=alpha.dtype, device=alpha.device)

        whiten = alpha * w_rot
        inv_whiten = inv_alpha * inv_rot

        self.inv_whiten.copy_(inv_whiten)
        self.whiten.copy_(whiten)

        return L, V, mask

    def _broadcast(self, src_rank: int, group: dist.ProcessGroup = None):
        """Broadcast PHI-specific buffers (rotation and alpha) across processes."""
        super()._broadcast(src_rank, group)
        dist.broadcast(self.rotation, src_rank, group=group)
        dist.broadcast(self.alpha, src_rank, group=group)

    def add_state_components(self, components):
        """Add PHI-specific scalar components to the external state dictionary."""
        super().add_state_components(components)
        components['phi-s_alpha'] = self.alpha.item()


class ProjectionMLP(nn.Module):
    """Multi-layer perceptron for feature projection and dimension alignment in distillation.

    This MLP is designed to project features from one dimension to another, commonly used
    in knowledge distillation to align student and teacher feature dimensions. It supports
    optional pre-normalization, configurable depth with residual connections, and spatial
    upsampling for feature map distillation.

    The architecture consists of:
    1. Optional pre-normalization (LayerNorm + GELU)
    2. Input projection layer
    3. Configurable number of inner residual blocks
    4. Final projection layer with LayerNorm + GELU
    5. Optional spatial upsampling for feature maps

    Args:
        input_size (int): Input feature dimension.
        hidden_size (int): Hidden layer dimension (before upsampling adjustment).
        output_size (int): Output feature dimension (before upsampling adjustment).
        num_inner (int, optional): Number of inner residual blocks. Default: 0.
        pre_norm (bool, optional): Whether to apply pre-normalization. Default: False.
        device (torch.device, optional): Device to place the module on. Default: None.
        upsample_factor (int, optional): Factor for spatial upsampling. Default: 1.
        upsample_rank (int, optional): Maximum rank constraint for upsampled hidden size. Default: 0.
        **kwargs: Additional arguments (unused).

    Attributes:
        pre_norm (nn.Module): Pre-normalization layer or identity.
        upsample_factor (int): Upsampling factor for spatial dimensions.
        fc1 (nn.Linear): Input projection layer.
        blocks (nn.ModuleList): List of inner residual blocks.
        final (nn.Sequential): Final projection with normalization and activation.

    Example:
        >>> # Basic projection MLP
        >>> proj = ProjectionMLP(input_size=768, hidden_size=1024, output_size=512)
        >>> x = torch.randn(32, 196, 768)  # [batch, tokens, features]
        >>> output = proj(x)  # Shape: [32, 196, 512]

        >>> # MLP with upsampling for spatial feature maps
        >>> proj = ProjectionMLP(
        ...     input_size=256, hidden_size=512, output_size=512,
        ...     upsample_factor=2, num_inner=2
        ... )
        >>> x = torch.randn(32, 49, 256)  # [batch, 7*7 tokens, features]
        >>> output = proj(x)  # Shape: [32, 196, 512] (14*14 tokens after upsampling)

    Note:
        When upsample_factor > 1, the input is assumed to represent spatial tokens
        arranged in a square grid (h = w = sqrt(num_tokens)). The output will have
        (upsample_factor^2) times more spatial tokens.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_inner: int = 0,
                 pre_norm: bool = False,
                 device: torch.device = None,
                 upsample_factor: int = 1,
                 upsample_rank: int = 0,
                 **kwargs) -> None:
        super().__init__()
        self.pre_norm = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.GELU(),
        ) if pre_norm else nn.Identity()

        self.upsample_factor = upsample_factor
        self._real_output_dim = output_size

        hidden_size = hidden_size * upsample_factor
        if upsample_rank:
            hidden_size = min(hidden_size, upsample_rank)
        output_size *= (upsample_factor ** 2)

        self.fc1 = nn.Linear(input_size, hidden_size, device=device)

        blocks = []
        for _ in range(num_inner):
            blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_size, device=device),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size, device=device),
            ))
        self.blocks = nn.ModuleList(blocks)

        flin = nn.Linear(hidden_size, output_size, device=device)
        self.final = nn.Sequential(
            nn.LayerNorm(hidden_size, device=device),
            nn.GELU(),
            flin,
        )
        flin.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the ProjectionMLP."""
        x = self.pre_norm(x)
        x = self.fc1(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.final(x)

        if self.upsample_factor > 1:
            h = w = int(math.sqrt(x.shape[1]))
            x = rearrange(x, 'b (h w) (u1 u2 c) -> b (h u1 w u2) c',
                          h=h, w=w, u1=self.upsample_factor, u2=self.upsample_factor,
                          c=self._real_output_dim)

        return x


class AttnFDHead(nn.Module):
    """Attention-based feature-distillation head used by C-RADIO v4.

    The head applies ViT attention blocks before the projection MLP used to
    align student spatial features to the teacher feature dimension.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_inner: int = 0,
                 num_blocks: int = 2,
                 num_heads: int = 16,
                 pre_norm: bool = False,
                 device: torch.device = None,
                 upsample_factor: int = 1,
                 upsample_rank: int = 0,
                 **kwargs) -> None:
        """Initialize the attention feature-distillation head.

        Args:
            input_size (int): Channel dimension of student spatial features.
            hidden_size (int): Hidden dimension used by the projection MLP.
            output_size (int): Channel dimension expected by the teacher.
            num_inner (int): Number of residual inner MLP blocks.
            num_blocks (int): Number of attention blocks to apply before the
                projection MLP.
            num_heads (int): Number of attention heads in each block.
            pre_norm (bool): Whether to normalize inputs before the first MLP
                projection.
            device (torch.device): Optional device for MLP parameters.
            upsample_factor (int): Spatial upsampling factor applied by the
                projection MLP.
            upsample_rank (int): Optional cap on the upsampled hidden size.
            **kwargs: Additional keyword arguments accepted for compatibility
                with projection-head construction.

        Returns:
            None: The attention blocks and projection MLP are initialized in
                place.
        """
        super().__init__()
        from timm.models.vision_transformer import Block
        self.blocks = nn.Sequential(*[
            Block(input_size, num_heads=num_heads, init_values=1e-5)
            for _ in range(num_blocks)
        ])
        self.mlp = ProjectionMLP(
            input_size, hidden_size, output_size,
            num_inner=num_inner,
            pre_norm=pre_norm,
            device=device,
            upsample_factor=upsample_factor,
            upsample_rank=upsample_rank,
        )
        self.upsample_factor = upsample_factor

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through attention blocks and the projection MLP.

        Args:
            x (torch.Tensor): Student spatial features with shape ``B x N x C``.
            **kwargs: Extra arguments forwarded to the projection MLP.

        Returns:
            torch.Tensor: Projected features aligned to the teacher feature
                dimension.
        """
        x = self.blocks(x)
        x = self.mlp(x, **kwargs)
        return x


class CosineSimilarityLoss():
    """Cosine similarity loss for feature distillation."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def __call__(self, normalized_student_features: torch.Tensor, normalized_teacher_features: torch.Tensor):
        """Compute cosine similarity loss."""
        cs = nn.CosineSimilarity(dim=-1, eps=self.eps)(normalized_student_features, normalized_teacher_features)
        return 1.0 - cs.mean()


class BalancedFeatureLoss:
    """Balanced feature loss for feature distillation."""

    def __init__(self, weight: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def __call__(self, normalized_student_features: torch.Tensor, normalized_teacher_features: torch.Tensor):
        """Compute balanced feature loss."""
        loss_l1 = nn.SmoothL1Loss(beta=2.0)(normalized_student_features, normalized_teacher_features)
        loss_cos = CosineSimilarityLoss(eps=self.eps)(normalized_student_features, normalized_teacher_features)
        loss = (1 - self.weight) * loss_cos + self.weight * loss_l1
        return loss


class SummaryCosineLoss(nn.Module):
    """Cosine similarity loss for summary/embedding distillation (EVFM-style).
    loss = 1 - cos_sim(student, teacher), reduced over batch.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return 1 minus the mean cosine similarity between ``pred`` and ``target``."""
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)
        x_norm = pred_flat.norm(p=2, dim=-1).clamp_min(self.eps)
        y_norm = target_flat.norm(p=2, dim=-1).clamp_min(self.eps)
        cos_sim = (pred_flat * target_flat).sum(dim=-1) / (x_norm * y_norm)
        loss = (1 - cos_sim).mean()
        return loss


class SummaryAngleLoss(nn.Module):
    """Angle loss for summary distillation (EVFM-style).
    loss = angle_sq / angle_variance, with running stats for teacher direction variance.
    """

    def __init__(self, feature_dim: int, max_samples: float = 1e7):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_samples = max_samples
        self.register_buffer('num_samples', torch.tensor(0, dtype=torch.float64))
        self.register_buffer('sum_direction', torch.zeros(feature_dim, dtype=torch.float64))
        self.register_buffer('sum_angle_variance', torch.tensor(0.0, dtype=torch.float64))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the variance-normalized angular loss between ``pred`` and ``target``."""
        flat_pred = pred.flatten(1)
        flat_target = target.flatten(1)
        with torch.no_grad():
            if self.num_samples < self.max_samples:
                curr_num = torch.tensor(flat_target.shape[0], dtype=torch.float64, device=pred.device)
                curr_dir_sum = flat_target.detach().sum(dim=0, dtype=torch.float64)
                if dist.is_initialized():
                    dist.all_reduce(curr_num, op=dist.ReduceOp.SUM)
                    dist.all_reduce(curr_dir_sum, op=dist.ReduceOp.SUM)
                self.num_samples.add_(curr_num)
                self.sum_direction.add_(curr_dir_sum)
                mean_direction = self.sum_direction / self.num_samples
                target_cos_to_mean = F.cosine_similarity(
                    flat_target.detach(),
                    mean_direction.unsqueeze(0).to(flat_target.dtype),
                    dim=-1,
                )
                target_angle_to_mean = torch.acos(target_cos_to_mean.clamp(-1 + 1e-6, 1 - 1e-6))
                curr_angle_var = target_angle_to_mean.pow(2).sum(dtype=torch.float64)
                if dist.is_initialized():
                    dist.all_reduce(curr_angle_var, op=dist.ReduceOp.SUM)
                self.sum_angle_variance.add_(curr_angle_var)
        angle_variance = (self.sum_angle_variance / self.num_samples).to(pred.dtype).clamp_min(1e-8)
        cos_theta = F.cosine_similarity(flat_pred, flat_target, dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        angle_sq = torch.acos(cos_theta).pow(2)
        loss = (angle_sq / angle_variance).mean()
        return loss


class SummaryTangentSphereLoss(nn.Module):
    """Tangent-space sphere loss for summary distillation (EVFM-style).
    Normalize to unit sphere, map to tangent space at running mean direction, then MSE.
    """

    def __init__(self, feature_dim: int, max_samples: int = 16384):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_samples = max_samples
        self.register_buffer('fwd_ct', torch.tensor(0, dtype=torch.int64))
        self.register_buffer('num_samples', torch.tensor(0, dtype=torch.float64))
        self.register_buffer('phis_alpha', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('pole', torch.zeros(1, feature_dim, dtype=torch.float64))
        self.register_buffer('tan_mean', torch.zeros(1, feature_dim, dtype=torch.float64))
        self.register_buffer('whiten', torch.eye(feature_dim, dtype=torch.float64))
        self.register_buffer('hadamard', get_hadamard_matrix(feature_dim).to(torch.float64), persistent=False)
        self._samples: List[torch.Tensor] = []

    def _update_phis(self, target: torch.Tensor) -> None:
        if self.fwd_ct > 0:
            return
        with torch.no_grad():
            target_n = F.normalize(target.flatten(1), dim=-1).to(torch.float64)
            batch_n = target_n.shape[0]
            if dist.is_initialized():
                n_t = torch.tensor(batch_n, dtype=torch.float64, device=target.device)
                dist.all_reduce(n_t, op=dist.ReduceOp.SUM)
                self.num_samples.add_(n_t)
            else:
                self.num_samples.add_(float(batch_n))
            self._samples.append(target_n.cpu())
            if self.num_samples < self.max_samples:
                return
            self.fwd_ct.fill_(1)
        with torch.no_grad():
            samples = torch.cat(self._samples, dim=0).to(device=target.device, dtype=torch.float64)
            self._samples = []
            n_global = self.num_samples.item()
            mean_dir = samples.sum(dim=0, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(mean_dir, op=dist.ReduceOp.SUM)
            mean_dir = mean_dir / self.num_samples.to(samples.device)
            mean_dir = F.normalize(mean_dir, dim=-1)
            self.pole.copy_(mean_dir)
            cos_theta = (samples @ mean_dir.T).clamp(-1 + 1e-6, 1 - 1e-6)
            theta = torch.acos(cos_theta)
            sin_theta = torch.sin(theta).clamp_min(1e-6)
            null_space = samples - mean_dir * cos_theta
            log_map = (theta / sin_theta) * null_space
            log_map_mean = log_map.sum(dim=0, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(log_map_mean, op=dist.ReduceOp.SUM)
            log_map_mean = log_map_mean / self.num_samples.to(samples.device)
            self.tan_mean.copy_(log_map_mean)
            centered = log_map - log_map_mean
            cov = centered.T @ centered
            if dist.is_initialized():
                dist.all_reduce(cov, op=dist.ReduceOp.SUM)
            cov = cov / (n_global - 1) if n_global > 1 else torch.eye(
                self.feature_dim, device=centered.device, dtype=torch.float64
            )
            L, V = torch.linalg.eigh(cov)
            L = torch.where(L >= 0, L, torch.zeros_like(L))
            alpha = L.mean().rsqrt().clamp_min(1e-8)
            self.phis_alpha.copy_(alpha.to(torch.float32))
            hadamard = self.hadamard.to(samples.device)
            w_rot = hadamard @ V.T
            self.whiten.copy_(alpha * w_rot)

    def _apply_phis(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = F.normalize(x.flatten(1), dim=-1).to(torch.float64)
        if self.fwd_ct == 0:
            return x.to(orig_dtype)
        device = x.device
        pole = self.pole.to(device)
        tan_mean = self.tan_mean.to(device)
        whiten = self.whiten.to(device)
        cos_theta = (x @ pole.T).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sin(theta).clamp_min(1e-6)
        null_space = x - pole * cos_theta
        log_map = (theta / sin_theta) * null_space
        centered = log_map - tan_mean
        transformed = centered @ whiten.T
        return transformed.to(orig_dtype)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the tangent-sphere loss between ``pred`` and ``target``."""
        l_cos = 1 - F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=-1)
        self._update_phis(target)
        if self.fwd_ct == 0:
            return l_cos.mean()
        pred_tan = self._apply_phis(pred)
        target_tan = self._apply_phis(target)
        loss = F.mse_loss(pred_tan, target_tan, reduction='none').mean(dim=-1)
        return loss.mean()


class DistillationLoss(nn.Module):
    """A modular distillation loss module that supports various loss types for knowledge distillation.

    This module can handle both logit distillation and feature map distillation, automatically
    handling dimension mismatches between teacher and student models through projection layers.

    Supported loss types:
    - "CE": Cross Entropy loss for logit distillation
    - "KL": KL Divergence loss for logit distillation
    - "L1": L1 loss for feature distillation
    - "L2": L2 loss for feature distillation
    - "FD": Feature Distillation using Smooth L1 loss
    - "CS": Cosine Similarity loss for feature distillation
    - "BALANCED": Balanced feature loss for feature distillation
    """

    def __init__(
        self,
        loss_type: str,
        student_model: nn.Module,
        teacher_model: nn.Module,
        num_classes: int,
        distillation_mode: str = "auto",
        temperature: float = 1.0,
        use_mlp: bool = True,
        mlp_hidden_size: int = 1024,
        mlp_num_inner: int = 2,
        spatial_mlp_version: str = "v2",
        spatial_num_inner: Optional[int] = None,
        summary_loss_weight: float = 1.0,
        fd_loss_weight: float = 1.0,
        summary_loss_type: str = "CE",
        spatial_loss_type: str = "mse",
        summary_token_idx: Optional[int] = None,
    ):
        """
        Initialize the DistillationLoss module.

        Args:
            loss_type (str): Type of distillation loss. One of ["CE", "KL", "L1", "L2", "FD", "CS"]
            student_model (nn.Module): Student model for distillation
            teacher_model (nn.Module): Teacher model for distillation
            num_classes (int, optional): Number of classes. Used for validation in feature distillation modes.
            distillation_mode (str): Mode for distillation. Options:
                - "logits": Use model.forward() for logit distillation
                - "summary": Use model.forward_pre_logits() for summary/cls token distillation
                - "auto": Automatically determine based on loss_type (CE/KL -> logits, others -> features)
            temperature (float): Temperature for knowledge distillation. Default: 1.0
            use_mlp (bool): Whether to use MLP for projection. Default: False
            mlp_hidden_size (int): Hidden size for MLP. Default: 1024
            mlp_num_inner (int): Number of inner layers for MLP. Default: 2
            spatial_mlp_version (str): Spatial projection head version. Use
                "v2" for ``ProjectionMLP`` or "attn" for ``AttnFDHead``.
            spatial_num_inner (Optional[int]): Number of inner layers for the
                spatial projection head. Defaults to ``mlp_num_inner`` for
                "v2" and 0 for "attn".
            summary_loss_weight (float): Weight for summary/CLS loss in combo mode. Default: 1.0
            fd_loss_weight (float): Weight for spatial/fd loss in combo mode. Default: 1.0
            summary_loss_type (str): Summary loss in combo mode. One of ["CE", "angle", "cosine", "tangent_sphere"].
                Default: "CE" (soft cross-entropy with temperature). EVFM-style options: angle, cosine, tangent_sphere.
            spatial_loss_type (str): Spatial (feature map) loss in combo/spatial mode. One of ["mse", "dampened_mse"].
                EVFM-style: "mse" = per-element squared error; "dampened_mse" = dampened for large residuals.
                Default: "mse".
            summary_token_idx (int): Optional RADIO summary-token slot for per-teacher summary distillation.
        """
        super().__init__()

        self.loss_type = loss_type.upper()
        self.summary_loss_type = (summary_loss_type or "CE").lower()
        self.spatial_loss_type = (spatial_loss_type or "mse").lower()
        self.summary_loss_weight = float(summary_loss_weight)
        self.fd_loss_weight = float(fd_loss_weight)
        self.summary_token_idx = summary_token_idx
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.num_classes = num_classes
        self.temperature = temperature

        # Validate loss type
        valid_loss_types = ["CE", "KL", "L1", "L2", "FD", "CS", "BALANCED", "MSE"]
        if self.loss_type not in valid_loss_types:
            raise ValueError(f"Unsupported loss type: {loss_type}. Must be one of {valid_loss_types}")

        # Determine distillation mode
        if distillation_mode.lower() == "auto":
            # Auto-detect based on loss type
            if self.loss_type in ["CE", "KL"]:
                self.distillation_mode = "logits"
            elif self.loss_type in ["BALANCED", "MSE"]:
                self.distillation_mode = "spatial"
                # in spatial mode, we only distill the last feature map
            else:
                self.distillation_mode = "summary"
        else:
            valid_modes = ["logits", "summary", "spatial", "combo"]
            if distillation_mode.lower() not in valid_modes:
                raise ValueError(f"Invalid distillation_mode: {distillation_mode}. Must be one of {valid_modes} or 'auto'")
            self.distillation_mode = distillation_mode.lower()

        # Validate configuration for feature distillation
        if self.loss_type in ["FD", "CS", "BALANCED", "MSE"] and self.distillation_mode == "logits":
            raise ValueError(f"Use L1, L2, KL or CE loss for logits distillation, but {self.loss_type} was specified.")

        if self.loss_type in ["FD", "CS", "BALANCED", "MSE"] and num_classes > 0:
            raise ValueError(f"Number of classes must be 0 when using '{self.loss_type}' for distillation")

        # Get model dimensions by checking available methods
        self.student_dim, self.teacher_dim = self._get_model_dimensions()
        logging.info(f"student_dim: {self.student_dim}, teacher_dim: {self.teacher_dim}")

        # Create projection layer if dimensions differ and we're doing feature distillation
        self.projection_layer = None
        self.projection_layer_summary = None
        spatial_mlp_version = (spatial_mlp_version or "v2").lower()
        if spatial_mlp_version not in ("v2", "attn"):
            raise ValueError(f"Unsupported spatial_mlp_version: {spatial_mlp_version}. Must be 'v2' or 'attn'.")
        if spatial_num_inner is None:
            spatial_num_inner = 0 if spatial_mlp_version == "attn" else mlp_num_inner
        self.spatial_mlp_version = spatial_mlp_version
        if self.student_dim != self.teacher_dim or isinstance(self.student_model, RADIO) or isinstance(self.teacher_model, RADIO):
            if use_mlp:
                if spatial_mlp_version == "attn":
                    self.projection_layer = AttnFDHead(
                        self.student_dim, mlp_hidden_size, self.teacher_dim,
                        num_inner=spatial_num_inner,
                    )
                else:
                    self.projection_layer = ProjectionMLP(
                        self.student_dim, mlp_hidden_size, self.teacher_dim,
                        num_inner=spatial_num_inner,
                    )
                if self.distillation_mode == "combo":
                    if isinstance(self.student_model, RADIO):
                        student_dim_summary = self._summary_feature_dim(self.student_model, self.student_dim)
                    else:
                        student_dim_summary = self.student_dim
                    if isinstance(self.teacher_model, RADIO):
                        teacher_dim_summary = self._summary_feature_dim(self.teacher_model, self.teacher_dim)
                    else:
                        teacher_dim_summary = self.teacher_dim
                    self.projection_layer_summary = ProjectionMLP(student_dim_summary, mlp_hidden_size, teacher_dim_summary, num_inner=mlp_num_inner)
            else:
                self.projection_layer = nn.Linear(self.student_dim, self.teacher_dim, bias=True)
                if self.distillation_mode == "combo":
                    if isinstance(self.student_model, RADIO):
                        student_dim_summary = self._summary_feature_dim(self.student_model, self.student_dim)
                    else:
                        student_dim_summary = self.student_dim
                    if isinstance(self.teacher_model, RADIO):
                        teacher_dim_summary = self._summary_feature_dim(self.teacher_model, self.teacher_dim)
                    else:
                        teacher_dim_summary = self.teacher_dim
                    self.projection_layer_summary = nn.Linear(student_dim_summary, teacher_dim_summary, bias=True)
        # always use linear even if dimensions are the same
        # Initialize loss functions
        self.criterions = {
            "L1": LPCriterion(p=1),
            "L2": LPCriterion(p=2),
            "KL": KLDivCriterion(),
            "CE": Cross_Entropy(soft=True, label_smoothing=False),
            "FD": nn.SmoothL1Loss(beta=2.0),
            "CS": CosineSimilarityLoss(eps=1e-8),
            "BALANCED": BalancedFeatureLoss(eps=1e-8),
            "MSE": nn.MSELoss(),
        }

        # Create layer normalization for feature distillation if specified
        if self.distillation_mode == "summary":
            self.teacher_norm = nn.LayerNorm(self.teacher_dim, elementwise_affine=False)
        else:
            self.teacher_norm = None

        if self.distillation_mode == "spatial" or self.distillation_mode == "combo":
            self.phi_norm = PHIStandardization(
                name='phi_norm',
                feature_dim=self.teacher_dim,
                ohem=False,
                update_period=100,  # Update projections every 100 batches
                rotate=True  # Use Hadamard rotation (default)
            )
            valid_spatial_losses = ("mse", "dampened_mse")
            if self.spatial_loss_type not in valid_spatial_losses:
                raise ValueError(
                    f"spatial_loss_type must be one of {valid_spatial_losses}, got {self.spatial_loss_type!r}"
                )
            logging.info(f"Using spatial_loss_type={self.spatial_loss_type} for feature map distillation")

        # Summary loss criterion for combo mode (EVFM-style: angle, cosine, tangent_sphere)
        self.summary_criterion = None
        if self.distillation_mode == "combo":
            valid_summary_losses = ("ce", "angle", "cosine", "tangent_sphere")
            if self.summary_loss_type not in valid_summary_losses:
                raise ValueError(
                    f"summary_loss_type must be one of {valid_summary_losses}, got {summary_loss_type!r}"
                )
            else:
                logging.info(f"Using {self.summary_loss_type} loss for summary in combo mode")
            if self.summary_loss_type != "ce":
                if isinstance(self.teacher_model, RADIO):
                    teacher_dim_summary = self._summary_feature_dim(self.teacher_model, self.teacher_dim)
                else:
                    teacher_dim_summary = self.teacher_dim
                if self.summary_loss_type == "angle":
                    self.summary_criterion = SummaryAngleLoss(feature_dim=teacher_dim_summary)
                elif self.summary_loss_type == "cosine":
                    self.summary_criterion = SummaryCosineLoss()
                else:
                    self.summary_criterion = SummaryTangentSphereLoss(feature_dim=teacher_dim_summary)

    def _summary_feature_dim(self, model: nn.Module, fallback_dim: int) -> int:
        """Return the summary dimension used by this loss for a model.

        Args:
            model (nn.Module): Student or teacher model that may be a RADIO
                model with multiple summary tokens.
            fallback_dim (int): Dimension to use for non-RADIO models.

        Returns:
            int: Per-token summary feature dimension used by the loss.
        """
        if not isinstance(model, RADIO):
            return fallback_dim
        summary_idxs = getattr(model, "summary_idxs", None)
        if self.summary_token_idx is None or summary_idxs is None:
            return int(model.num_features)
        token_count = len(summary_idxs)
        if token_count <= 0:
            return int(model.num_features)
        if model.num_features % token_count != 0:
            raise ValueError(
                f"RADIO num_features={model.num_features} is not divisible by "
                f"len(summary_idxs)={token_count}"
            )
        return int(model.num_features // token_count)

    def _summary_token_position(self, model: nn.Module) -> Optional[int]:
        """Map a RADIO summary-token slot to its position in ``summary_idxs``.

        Args:
            model (nn.Module): Student or teacher model whose summary token
                layout should be inspected.

        Returns:
            Optional[int]: Position of ``summary_token_idx`` in
                ``model.summary_idxs``, or ``None`` when no selection is
                needed.
        """
        if not isinstance(model, RADIO) or self.summary_token_idx is None:
            return None
        summary_idxs = getattr(model, "summary_idxs", None)
        if summary_idxs is None:
            return None

        token_idx = int(self.summary_token_idx)
        summary_idx_list = [int(idx) for idx in summary_idxs]
        if token_idx in summary_idx_list:
            return summary_idx_list.index(token_idx)
        if summary_idx_list == list(range(len(summary_idx_list))) and 0 <= token_idx < len(summary_idx_list):
            return token_idx
        raise ValueError(
            f"summary_token_idx={token_idx} is not present in RADIO summary_idxs={summary_idx_list}"
        )

    def _select_summary_token(self, summary: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Select the per-teacher RADIO summary token.

        Args:
            summary (torch.Tensor): Summary tensor from the student or teacher
                model. RADIO outputs may be flattened or tokenized.
            model (nn.Module): Model that produced ``summary``.

        Returns:
            torch.Tensor: The selected summary-token features, or the original
                summary when no per-teacher token selection is configured.
        """
        position = self._summary_token_position(model)
        if position is None:
            return summary

        summary_idxs = getattr(model, "summary_idxs", None)
        token_count = len(summary_idxs)
        token_dim = self._summary_feature_dim(model, summary.shape[-1])

        if summary.ndim == 3:
            if summary.shape[1] <= position:
                raise ValueError(
                    f"Cannot select summary token position {position} from summary shape {list(summary.shape)}"
                )
            return summary[:, position].contiguous()

        if summary.ndim != 2:
            raise ValueError(f"Expected RADIO summary with 2 or 3 dims, got shape {list(summary.shape)}")
        if summary.shape[-1] == token_dim:
            return summary
        if summary.shape[-1] % token_count != 0:
            raise ValueError(
                f"Cannot split flattened RADIO summary shape {list(summary.shape)} into "
                f"{token_count} tokens for summary_token_idx={self.summary_token_idx}"
            )
        return summary.reshape(summary.shape[0], token_count, summary.shape[-1] // token_count)[:, position].contiguous()

    def _get_model_dimensions(self):
        """Get the output dimensions for student and teacher models."""
        if self.distillation_mode == "logits":
            # For logits, try to get num_classes or use a test forward pass
            student_dim = teacher_dim = self.num_classes
        elif self.distillation_mode == "summary":
            # For features, try to get num_features
            student_dim = self.student_model.num_features
            teacher_dim = self.teacher_model.num_features
        else:
            if isinstance(self.student_model, RADIO):
                student_dim = self.student_model.num_features // len(self.student_model.summary_idxs)
            else:
                student_dim = self.student_model.num_features
            if isinstance(self.teacher_model, RADIO):
                teacher_dim = self.teacher_model.num_features // len(self.teacher_model.summary_idxs)
            else:
                teacher_dim = self.teacher_model.num_features
        return student_dim, teacher_dim

    def _interpolate_to_size(self, features: Union[torch.Tensor, List[torch.Tensor]], shape: Tuple[int, int]):
        """Interpolate feature map(s) to a target spatial size if needed.

        Args:
            features: Tensor or list of tensors shaped [B, C, H, W].
            shape: Target spatial size `(H, W)`.

        Returns:
            Interpolated tensor or list matching the input type.
        """
        if isinstance(features, (list, tuple)):
            return [self._interpolate_to_size(ft, shape) for ft in features]

        if features.shape[2:] != shape:
            features = F.interpolate(
                features,
                size=shape,
                mode='bilinear',
                align_corners=True,
            )
        return features

    @torch.autocast('cuda', enabled=False)
    def _apply_phi_s(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``phi_norm`` standardization to arbitrary feature tensors.

        Args:
            x (torch.Tensor): Student feature tensor whose last dimension
                matches ``phi_norm`` statistics.

        Returns:
            torch.Tensor: Standardized features with the original dtype.
        """
        mean = self.phi_norm.expected_mean.to(torch.float32)
        whiten = self.phi_norm.whiten.to(torch.float32)
        out_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        return ((x_fp32 - mean) @ whiten.T).to(out_dtype)

    @staticmethod
    def _get_last_feature_map(features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]):
        """Extract the last feature map from a list/tuple/dict or return the tensor itself."""
        if isinstance(features, (list, tuple)):
            return features[-1]
        elif isinstance(features, dict):
            return list(features.values())[-1]
        return features

    @staticmethod
    def _build_mask(valid_mask, target_shape, device):
        """Resize a valid-mask to *target_shape* ``(H, W)`` and return a bool tensor."""
        H, W = target_shape
        if valid_mask.shape[-2:] != (H, W):
            mask = F.adaptive_avg_pool2d(
                valid_mask.unsqueeze(1).float(), (H, W)
            ).squeeze(1)
        else:
            mask = valid_mask.float()
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        return mask.to(device)

    @staticmethod
    def _align_features(student_feat, teacher_feat,
                        student_valid_mask, teacher_valid_mask,
                        spatial_transform):
        """Align student/teacher feature maps using spatial_transform when available.

        When *spatial_transform* is provided the student features are warped
        into the teacher feature space (or vice-versa, whichever is smaller)
        using ``generate_homography_grid`` + ``F.grid_sample``, matching the
        EVFM ``feature_distillation_loss`` alignment logic.

        Returns ``(student_feat, teacher_feat)`` in ``[B, H*W, C]`` layout.
        """
        if spatial_transform is not None:
            s_valid = student_valid_mask.float()
            t_valid = teacher_valid_mask.float()

            if teacher_feat.shape[-1] <= student_feat.shape[-1]:
                grid = generate_homography_grid(spatial_transform, teacher_feat.shape)
                student_feat = F.grid_sample(
                    student_feat, grid, mode='bilinear', align_corners=True,
                )
                s_valid = F.grid_sample(
                    s_valid.unsqueeze(1), grid, mode='bilinear', align_corners=True,
                ).squeeze(1)
            else:
                inv_transform = torch.linalg.inv(spatial_transform)
                grid = generate_homography_grid(inv_transform, student_feat.shape)
                teacher_feat = F.grid_sample(
                    teacher_feat, grid, mode='bilinear', align_corners=True,
                )
                t_valid = F.grid_sample(
                    t_valid.unsqueeze(1), grid, mode='bilinear', align_corners=True,
                ).squeeze(1)
                s_valid = F.adaptive_avg_pool2d(
                    s_valid.unsqueeze(1), student_feat.shape[-2:]
                ).squeeze(1)

            valid_mask = s_valid * t_valid
            eps = 1e-8
            weighted_valid = valid_mask * (1 - eps) + eps
            eq_valid = torch.all(grid.abs() <= 1, dim=-1)
            valid_mask = torch.where(eq_valid, weighted_valid, torch.zeros_like(weighted_valid))

            student_feat = rearrange(student_feat, 'b c h w -> b (h w) c')
            teacher_feat = rearrange(teacher_feat, 'b c h w -> b (h w) c')
            valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
            return student_feat, teacher_feat, valid_mask
        else:
            if student_feat.shape[2:] != teacher_feat.shape[2:]:
                target = tuple(
                    min(s, t)
                    for s, t in zip(student_feat.shape[2:], teacher_feat.shape[2:])
                )
                student_feat = F.interpolate(student_feat, size=target, mode='bilinear', align_corners=False)
                teacher_feat = F.interpolate(teacher_feat, size=target, mode='bilinear', align_corners=False)
            student_feat = rearrange(student_feat, 'b c h w -> b (h w) c')
            teacher_feat = rearrange(teacher_feat, 'b c h w -> b (h w) c')
            return student_feat, teacher_feat, None

    def _spatial_feature_loss(
        self,
        student_spatial: torch.Tensor,
        teacher_spatial: torch.Tensor,
        eq_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """EVFM-style spatial (feature map) loss: element-wise loss, mean over channels, then masked mean over positions.

        Supports "mse" (per-element squared error) and "dampened_mse" (dampened for large residuals).
        Same reduction as EVFM BasicLossFn with reduction='none' then mean(dim=1) then reduce_loss(mask).
        """
        if self.spatial_loss_type == "dampened_mse":
            element_wise = dampened_mse_loss(student_spatial, teacher_spatial)
        else:
            element_wise = _mse_element_wise(student_spatial, teacher_spatial)
        # Per-position loss: mean over channels [B, N, C] -> [B, N] (EVFM: loss.mean(dim=1))
        loss_per_pos = element_wise.mean(dim=-1)
        if eq_mask is not None:
            num_valid = eq_mask.sum().clamp(min=1)
            loss_spatial = (loss_per_pos * eq_mask).sum() / num_valid
        else:
            loss_spatial = loss_per_pos.mean()
        return loss_spatial

    def forward(
        self,
        batch_input: torch.Tensor,
        teacher_batch_input: Optional[torch.Tensor] = None,
        student_valid_mask: Optional[torch.Tensor] = None,
        teacher_valid_mask: Optional[torch.Tensor] = None,
        spatial_transform: Optional[torch.Tensor] = None,
        student_summary: Optional[torch.Tensor] = None,
        student_spatial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute distillation loss between student and teacher outputs.

        Args:
            batch_input: Input batch for the student model (and teacher when multi-view not used).
            teacher_batch_input: Optional separate input for the teacher (multi-view).
            student_valid_mask: Optional [B, H, W] mask for student spatial loss.
            teacher_valid_mask: Optional [B, H, W] mask for teacher spatial loss.
            spatial_transform: Optional [B, 3, 3] homography for aligning
                teacher and student feature grids.
            student_summary: Optional pre-computed student summary features (cls token /
                pre-logits). When provided the student forward pass is skipped, avoiding
                a redundant computation when the caller already ran the student model.
            student_spatial: Optional pre-computed student spatial feature map(s).
                May be a single tensor or a list/dict of feature maps; the last map is
                selected automatically via ``_get_last_feature_map``.

        Returns:
            torch.Tensor: Computed distillation loss
        """
        teacher_input = teacher_batch_input if teacher_batch_input is not None else batch_input
        device = batch_input.device

        if self.distillation_mode == "logits":
            if student_summary is not None:
                student_output = self.student_model.head(student_summary)
            else:
                student_output = self.student_model(batch_input)
            with torch.no_grad():
                teacher_output = self.teacher_model(teacher_input)
        elif self.distillation_mode == "spatial":
            if student_spatial is not None:
                student_output = self._get_last_feature_map(student_spatial)
            else:
                student_output = self.student_model.forward_feature_pyramid(batch_input)
                student_output = self._get_last_feature_map(student_output)
            with torch.no_grad():
                teacher_output = self.teacher_model.forward_feature_pyramid(teacher_input)
                teacher_output = self._get_last_feature_map(teacher_output)
            # normalize the teacher feature maps
            B, _, H, W = teacher_output.shape
            if teacher_valid_mask is not None:
                t_mask = self._build_mask(teacher_valid_mask, (H, W), device)
            else:
                t_mask = torch.ones(B, H, W, dtype=torch.bool, device=device)
            # Only accumulate normalization statistics during training so that
            # validation / sanity-check batches do not pollute the running stats.
            if self.training:
                self.phi_norm.update(None, teacher_output, t_mask)
            teacher_output = self.phi_norm.transform_targets(teacher_output)

            # align the shape of student and teacher feature maps
            if student_valid_mask is None:
                student_valid_mask = torch.ones(B, student_output.shape[2], student_output.shape[3],
                                                dtype=torch.float32, device=device)
            teacher_valid_mask = t_mask.float()

            student_output, teacher_output, eq_mask = self._align_features(
                student_output, teacher_output,
                student_valid_mask, teacher_valid_mask,
                spatial_transform,
            )
        elif self.distillation_mode == "combo":
            teacher_sig = inspect.signature(self.teacher_model.forward)
            student_sig = inspect.signature(self.student_model.forward)
            assert 'return_features' in teacher_sig.parameters, "Teacher model must support return_features in `combo` mode"
            assert 'return_features' in student_sig.parameters, "Student model must support return_features in `combo` mode"
            if student_summary is not None and student_spatial is not None:
                student_spatial = self._get_last_feature_map(student_spatial)
            else:
                student_summary, student_spatial = self.student_model.forward(batch_input, return_features=True)
                student_spatial = self._get_last_feature_map(student_spatial)
            with torch.no_grad():
                teacher_summary, teacher_spatial = self.teacher_model.forward(teacher_input, return_features=True)
                teacher_spatial = self._get_last_feature_map(teacher_spatial)

            # normalize the teacher feature maps
            B, _, H, W = teacher_spatial.shape
            if teacher_valid_mask is not None:
                t_mask = self._build_mask(teacher_valid_mask, (H, W), device)
            else:
                t_mask = torch.ones(B, H, W, dtype=torch.bool, device=device)
            # Only accumulate normalization statistics during training so that
            # validation / sanity-check batches do not pollute the running stats.
            if self.training:
                self.phi_norm.update(None, teacher_spatial, t_mask)
            teacher_spatial = self.phi_norm.transform_targets(teacher_spatial)
            # align the shape of student and teacher feature maps

            if student_valid_mask is None:
                student_valid_mask = torch.ones(B, student_spatial.shape[2], student_spatial.shape[3],
                                                dtype=torch.float32, device=device)
            teacher_valid_mask = t_mask.float()

            student_spatial, teacher_spatial, eq_mask = self._align_features(
                student_spatial, teacher_spatial,
                student_valid_mask, teacher_valid_mask,
                spatial_transform,
            )

            if self.projection_layer is not None:
                student_spatial = self.projection_layer(student_spatial)
                student_spatial = self._apply_phi_s(student_spatial)
            # spatial (feature distillation) loss — EVFM-style: element-wise (mse or dampened_mse), mean over C, masked mean
            loss_spatial = self._spatial_feature_loss(student_spatial, teacher_spatial, eq_mask)
            loss = self.fd_loss_weight * loss_spatial

            if self.summary_loss_weight != 0.0:
                student_summary = self._select_summary_token(student_summary, self.student_model)
                teacher_summary = self._select_summary_token(teacher_summary, self.teacher_model)
                if self.projection_layer_summary is not None:
                    student_summary = self.projection_layer_summary(student_summary)
                if self.teacher_norm is not None:
                    teacher_summary = self.teacher_norm(teacher_summary)

                # summary loss (CE with temperature, or EVFM-style angle/cosine/tangent_sphere)
                if self.summary_criterion is not None:
                    loss_summary = self.summary_criterion(student_summary, teacher_summary)
                else:
                    teacher_probs = F.softmax(teacher_summary / self.temperature, dim=-1)
                    loss_summary = self.criterions["CE"](student_summary / self.temperature, teacher_probs)
                loss = loss + self.summary_loss_weight * loss_summary
            return loss
        else:
            if student_summary is not None:
                student_output = student_summary
            else:
                student_output = self.student_model.forward_pre_logits(batch_input)
            with torch.no_grad():
                teacher_output = self.teacher_model.forward_pre_logits(teacher_input)

        # Handle projection for feature distillation
        if self.distillation_mode != "logits" and self.projection_layer is not None:
            student_output = self.projection_layer(student_output)
            if self.distillation_mode == "spatial":
                student_output = self._apply_phi_s(student_output)

        # Apply teacher normalization if specified
        if self.teacher_norm is not None and self.distillation_mode == "summary":
            teacher_output = self.teacher_norm(teacher_output)

        # Compute loss based on type
        if self.loss_type == "CE":
            # Cross entropy loss for logit distillation
            teacher_probs = F.softmax(teacher_output / self.temperature, dim=-1)
            loss = self.criterions["CE"](student_output / self.temperature, teacher_probs)
        elif self.loss_type == "KL":
            # KL divergence loss for logit distillation
            loss = self.criterions["KL"](student_output / self.temperature, teacher_output / self.temperature)
        else:
            # Direct loss computation for L1, L2, FD, CS, BALANCED
            loss = self.criterions[self.loss_type](student_output, teacher_output)

        return loss

    def get_loss_info(self) -> dict:
        """
        Get information about the configured loss.

        Returns:
            dict: Dictionary containing loss configuration details
        """
        return {
            "loss_type": self.loss_type,
            "distillation_mode": self.distillation_mode,
            "student_dim": self.student_dim,
            "teacher_dim": self.teacher_dim,
            "num_classes": self.num_classes,
            "temperature": self.temperature,
            "has_projection": self.projection_layer is not None,
        }
