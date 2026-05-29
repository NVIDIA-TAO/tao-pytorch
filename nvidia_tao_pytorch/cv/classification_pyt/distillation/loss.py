# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distillation Loss module for knowledge distillation."""
import os
import math
from typing import Union, List, Tuple, Dict
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.distillation.losses import LPCriterion, KLDivCriterion
from nvidia_tao_pytorch.cv.backbone_v2.radio import RADIO
from nvidia_tao_pytorch.cv.classification_pyt.utils.loss import Cross_Entropy
from nvidia_tao_pytorch.cv.classification_pyt.distillation.hadamard import get_hadamard_matrix
from nvidia_tao_pytorch.core.distributed.comm import get_global_rank, get_world_size


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
        """
        super().__init__()

        self.loss_type = loss_type.upper()
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
            valid_modes = ["logits", "summary", "spatial"]
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
        if self.student_dim != self.teacher_dim:
            if use_mlp:
                self.projection_layer = ProjectionMLP(self.student_dim, mlp_hidden_size, self.teacher_dim, num_inner=mlp_num_inner)
            else:
                self.projection_layer = nn.Linear(self.student_dim, self.teacher_dim, bias=True)
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

        if self.distillation_mode == "spatial":
            self.phi_norm = PHIStandardization(
                name='phi_norm',
                feature_dim=self.teacher_dim,
                ohem=False,
                update_period=100,  # Update projections every 100 batches
                rotate=True  # Use Hadamard rotation (default)
            )

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

    @staticmethod
    def _get_last_feature_map(features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]):
        """Extract the last feature map from a list/tuple/dict or return the tensor itself."""
        if isinstance(features, (list, tuple)):
            return features[-1]
        elif isinstance(features, dict):
            return list(features.values())[-1]
        return features

    def forward(self, batch_input: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss between student and teacher outputs.

        Args:
            batch_input (torch.Tensor): Input batch data to be passed through both models

        Returns:
            torch.Tensor: Computed distillation loss
        """
        # TODO(@yuw): Add summary + spatial loss
        # TODO(@yuw): add modify linear for inference
        # Get outputs based on distillation mode
        if self.distillation_mode == "logits":
            # Use standard forward pass for logits
            student_output = self.student_model(batch_input)
            with torch.no_grad():
                teacher_output = self.teacher_model(batch_input)
        elif self.distillation_mode == "spatial":
            student_output = self.student_model.forward_feature_pyramid(batch_input)
            student_output = self._get_last_feature_map(student_output)
            with torch.no_grad():
                teacher_output = self.teacher_model.forward_feature_pyramid(batch_input)
                teacher_output = self._get_last_feature_map(teacher_output)
            # normalize the teacher feature maps
            B, _, H, W = teacher_output.shape
            mask = torch.ones(B, H, W, dtype=torch.bool, device='cuda')
            self.phi_norm.update(None, teacher_output, mask)
            teacher_output = self.phi_norm.transform_targets(teacher_output)
            # align the shape of student and teacher feature maps
            if student_output.shape[2:] != teacher_output.shape[2:]:  # B, C, H, W
                max_shape = tuple(
                    min(s, t)  # diff from original code (max(s, t))
                    for s, t in zip(student_output.shape[2:], teacher_output.shape[2:])
                )
                student_output = self._interpolate_to_size(student_output, max_shape)
                teacher_output = self._interpolate_to_size(teacher_output, max_shape)
            # [B, C, H, W] -> [B, H*W, C]
            student_output = rearrange(student_output, 'b c h w -> b (h w) c')
            teacher_output = rearrange(teacher_output, 'b c h w -> b (h w) c')
        else:
            # Use forward_pre_logits for summary token distillation
            student_output = self.student_model.forward_pre_logits(batch_input)
            with torch.no_grad():
                teacher_output = self.teacher_model.forward_pre_logits(batch_input)

        # Handle projection for feature distillation
        if self.distillation_mode != "logits" and self.projection_layer is not None:
            student_output = self.projection_layer(student_output)

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
