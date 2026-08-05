# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def _masked_token_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over token-wise values with an optional spatial/token mask."""
    if mask is None:
        return values.mean()
    valid = mask.to(dtype=values.dtype)
    return (values * valid).sum() / valid.sum().clamp_min(1)


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
    """Per-element squared error (native reduction='none')."""
    return (pred - target) ** 2


def dampened_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dampened MSE (native): for |diff| < 1 use 0.5*diff^2, else 2*sqrt(|diff|+eps)-1.5.

    Reduces sensitivity to large residuals.
    """
    diff = pred - target
    abs_diff = torch.abs(diff)
    dampened = 2 * torch.sqrt(abs_diff + 1e-8) - 1.5
    mse = 0.5 * (diff ** 2)
    return torch.where(abs_diff.detach() < 1, mse, dampened)


def _fft_mse_loss(
    student_spatial: torch.Tensor,
    teacher_spatial: torch.Tensor,
    grid_hw: Tuple[int, int],
    reduction: str = "mean",
) -> torch.Tensor:
    """RADIO fft_mse spatial-frequency distillation loss for high-resolution spatial feature teachers.

    Faithful port of RADIO.feature_distillation_loss.fft_mse_loss: 2D FFT
    (norm='ortho') over the HxW grid, fftshift, complex magnitude-squared
    difference, mean over channels, then mean over all (B, H, W) — i.e.
    reduction='mean'. The gaussian ring weighting in RADIO is dead code
    (gauss_weight=1, commented out) and is intentionally omitted. No spatial
    mask or focal weighting is applied: RADIO applies neither to fft_mse, and
    masking frequency bins by a spatial token validity mask is not meaningful
    (the FFT mixes all spatial positions). This branch therefore deliberately
    bypasses the eq_mask/focal reduction used by the other spatial losses.

    Args:
        student_spatial: NLC student features, shape (B, H*W, C).
        teacher_spatial: NLC teacher features, shape (B, H*W, C).
        grid_hw: (H, W) of the aligned spatial grid.

    Returns:
        Scalar loss tensor.
    """
    height, width = grid_hw
    _, seq_len, _ = student_spatial.shape
    if seq_len != height * width:
        raise ValueError(f"fft_mse expects L == H*W, got L={seq_len}, H*W={height * width}")
    # NLC -> BCHW in fp32 (matches the reference implementation's pred.float()/target.float()).
    student_bchw = rearrange(student_spatial.float(), "b (h w) c -> b c h w", h=height, w=width)
    teacher_bchw = rearrange(teacher_spatial.float(), "b (h w) c -> b c h w", h=height, w=width)
    fft_s = torch.fft.fftshift(torch.fft.fft2(student_bchw, norm="ortho"), dim=(2, 3))
    fft_t = torch.fft.fftshift(torch.fft.fft2(teacher_bchw, norm="ortho"), dim=(2, 3))
    diff = fft_s - fft_t
    # complex magnitude-squared MSE, mean over channel dim -> [B, H, W]
    loss_map = (diff.real ** 2 + diff.imag ** 2).mean(dim=1)
    if reduction == "none":
        return loss_map
    if reduction == "mean":
        return loss_map.mean()
    raise ValueError(f"Unsupported fft_mse reduction: {reduction!r}")


def linear_cka(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute linear CKA between two feature tensors.

    The sample axis is formed by flattening all leading dimensions except the
    feature dimension. For spatial RADIO features, this compares the geometry of
    valid spatial tokens across the current local batch.
    """
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    if mask is not None:
        valid = mask.reshape(-1).to(dtype=torch.bool, device=pred.device)
        pred = pred[valid]
        target = target[valid]

    if pred.shape[0] < 2:
        # Degenerate batch (fewer than 2 valid samples): CKA is undefined.
        # Return a differentiable zero tied to `pred` so the autograd graph is
        # preserved. Returning a detached constant here (e.g. new_zeros) makes
        # the rank's loss lack a grad_fn, which crashes loss.backward() and
        # desyncs DDP collectives across all ranks.
        return pred.float().sum() * 0.0

    pred = pred.float()
    target = target.float()
    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)

    cross_cov = pred.T @ target
    pred_cov = pred.T @ pred
    target_cov = target.T @ target
    hsic = cross_cov.square().sum()
    pred_var = pred_cov.square().sum()
    target_var = target_cov.square().sum()
    return hsic / (pred_var.sqrt() * target_var.sqrt()).clamp_min(eps)


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
        self.freeze_updates = False

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
        if self.freeze_updates:
            return self.expected_mean

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
        if self.freeze_updates:
            return

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
        cache_dir = os.path.join(torch.hub.get_dir(), 'RADIO', 'fd_loss_states', 'whiten')
        # cache_dir = os.path.join(torch.hub.get_dir(), 'RADIO', 'fd_loss_states', 'whiten-4part')
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
        self.register_buffer('alpha', torch.tensor(1.0, dtype=torch.float32, device=H.device))

    def _update_projections(self, fwd_count: int):
        """Compute PHI whitening using mean eigenvalue scaling and optional rotation."""
        cov = self.covariance
        cov = torch.nan_to_num((cov + cov.T) * 0.5, nan=0.0, posinf=0.0, neginf=0.0)

        eye = self.eye.to(device=cov.device, dtype=cov.dtype)
        scale = torch.diagonal(cov).abs().mean().clamp_min(1.0)
        last_error = None
        # Solve the (feature_dim x feature_dim) eigenproblem on CPU. At high res the
        # GPU is near-full, so a GPU eigh OOMs and trips the diagonal fallback below,
        # which drops the decorrelating rotation V and silently degrades PHI (the
        # low-variance channels stop being equalized -> features under-supervised
        # while the loss keeps falling). The covariance is tiny (~feature_dim^2), so
        # a CPU fp64 solve is negligible and can never OOM the GPU; only a genuine
        # numerical failure now falls back to the diagonal scale.
        cov_cpu = cov.detach().to(device="cpu", dtype=torch.float64)
        eye_cpu = torch.eye(cov_cpu.shape[-1], dtype=torch.float64)
        scale_cpu = float(scale)
        for jitter in (0.0, 1e-8, 1e-6, 1e-4):
            try:
                L, V = torch.linalg.eigh(cov_cpu + jitter * scale_cpu * eye_cpu)
                L = L.to(device=cov.device, dtype=cov.dtype)
                V = V.to(device=cov.device, dtype=cov.dtype)
                break
            except RuntimeError as exc:
                last_error = exc
        else:
            logging.warning(
                "PHI eigensolve failed for %s at fwd_count=%s; using diagonal scale fallback. Error: %s",
                self.name,
                fwd_count,
                last_error,
            )
            L = torch.diagonal(cov).clamp_min(0)
            V = eye

        mask = L >= 0
        L = torch.where(mask, L, 0)

        mean_eigenvalue = L.mean().clamp_min(1e-12)
        alpha = mean_eigenvalue.rsqrt()
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


class CovarianceWhitening(WhitenNormState):
    """True covariance whitening for teacher spatial targets.

    ``mode='zca'`` whitens and rotates back toward the original teacher basis.
    ``mode='pca'`` whitens into the covariance eigenbasis. Both expose the same
    ``expected_mean`` and ``whiten`` interface used by the distillation loss.
    """

    def __init__(
        self,
        name: str,
        feature_dim: int,
        ohem: bool,
        update_period: int = 100,
        mode: str = "zca",
        freeze_after_steps: int = 0,
        shrinkage: float = 0.0,
        eigen_floor: float = 1.0e-6,
        max_gain: float = 0.0,
    ):
        super().__init__(name, feature_dim, ohem, update_period)
        mode = (mode or "zca").lower()
        if mode not in ("zca", "pca"):
            raise ValueError(f"Unsupported whitening mode: {mode}. Must be 'zca' or 'pca'.")
        self.mode = mode
        self.freeze_after_steps = int(freeze_after_steps or 0)
        self.shrinkage = float(shrinkage or 0.0)
        self.eigen_floor = float(eigen_floor or 0.0)
        self.max_gain = float(max_gain or 0.0)
        self.register_buffer('last_mean_eigenvalue', torch.tensor(1.0, dtype=torch.float64), persistent=True)
        self.register_buffer('last_max_gain', torch.tensor(1.0, dtype=torch.float64), persistent=True)

    @property
    def max_samples(self) -> int:
        """Max whitening-update samples: the freeze-after-steps cap when set, else the base value."""
        if self.freeze_after_steps > 0:
            return self.freeze_after_steps
        return super().max_samples

    def _update_projections(self, fwd_count: int):
        """Compute true PCA/ZCA whitening matrices from the running covariance."""
        cov = self.covariance
        cov = torch.nan_to_num((cov + cov.T) * 0.5, nan=0.0, posinf=0.0, neginf=0.0)
        eye = self.eye.to(device=cov.device, dtype=cov.dtype)

        diag_mean = torch.diagonal(cov).abs().mean().clamp_min(1e-12)
        if self.shrinkage > 0:
            cov = (1.0 - self.shrinkage) * cov + self.shrinkage * diag_mean * eye

        last_error = None
        for jitter in (0.0, 1e-8, 1e-6, 1e-4):
            try:
                L, V = torch.linalg.eigh(cov + jitter * diag_mean * eye)
                break
            except RuntimeError as exc:
                last_error = exc
        else:
            logging.warning(
                "Whitening eigensolve failed for %s at fwd_count=%s; using diagonal fallback. Error: %s",
                self.name,
                fwd_count,
                last_error,
            )
            L = torch.diagonal(cov).clamp_min(0)
            V = eye

        L = torch.clamp(L, min=0)
        mean_eigenvalue = L.mean().clamp_min(1e-12)
        floor = (self.eigen_floor * mean_eigenvalue).clamp_min(1e-12)
        L = L.clamp_min(floor)

        gains = L.rsqrt()
        if self.max_gain > 0:
            gains = gains.clamp(max=self.max_gain)
        inv_gains = gains.reciprocal()

        gain_diag = torch.diag(gains)
        inv_gain_diag = torch.diag(inv_gains)
        if self.mode == "pca":
            whiten = gain_diag @ V.T
            inv_whiten = V @ inv_gain_diag
        else:
            whiten = V @ gain_diag @ V.T
            inv_whiten = V @ inv_gain_diag @ V.T

        self.whiten.copy_(whiten)
        self.inv_whiten.copy_(inv_whiten)
        self.last_mean_eigenvalue.copy_(mean_eigenvalue)
        self.last_max_gain.copy_(gains.max())
        return L, V, L > floor

    def add_state_components(self, components):
        """Add whitening scalar components to the external state dictionary."""
        super().add_state_components(components)
        components[f'{self.mode}_mean_eigenvalue'] = self.last_mean_eigenvalue.item()
        components[f'{self.mode}_max_gain'] = self.last_max_gain.item()

    def _broadcast(self, src_rank: int, group: dist.ProcessGroup = None):
        """Broadcast whitening state, including persistent diagnostics."""
        super()._broadcast(src_rank, group)
        dist.broadcast(self.last_mean_eigenvalue, src_rank, group=group)
        dist.broadcast(self.last_max_gain, src_rank, group=group)


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


class ResidualProjectionMLP(nn.Module):
    """Constrained residual projector for student-to-teacher feature lifting.

    The base path is a normalized linear projection. The residual path is
    zero-initialized so the module starts as a pure linear lift, then learns
    bounded nonlinear corrections during projection-head warmup.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_inner: int = 1,
        residual_scale: float = 0.25,
        output_norm: bool = False,
        device: torch.device = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size, device=device)
        self.base = nn.Linear(input_size, output_size, device=device)
        self.residual_in = nn.Linear(input_size, hidden_size, device=device)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size, device=device),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size, device=device),
            )
            for _ in range(max(int(num_inner or 0), 0))
        ])
        self.residual_out = nn.Linear(hidden_size, output_size, device=device)
        self.output_norm = nn.LayerNorm(output_size, device=device) if output_norm else nn.Identity()
        self.residual_scale = float(residual_scale)

        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through a linear lift plus zero-init residual branch."""
        x = self.input_norm(x)
        base = self.base(x)
        residual = self.residual_in(x)
        for block in self.blocks:
            residual = residual + block(residual)
        residual = self.residual_out(F.gelu(residual))
        return self.output_norm(base + self.residual_scale * residual)


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
    """Cosine similarity loss for summary/embedding distillation (native).
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
    """Angle loss for summary distillation (native).
    loss = angle_sq / angle_variance, with running stats for teacher direction variance.
    """

    def __init__(self, feature_dim: int, max_samples: float = 1e7):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_samples = max_samples
        self.register_buffer('num_samples', torch.tensor(0, dtype=torch.float64))
        self.register_buffer('sum_direction', torch.zeros(feature_dim, dtype=torch.float64))
        self.register_buffer('sum_angle_variance', torch.tensor(0.0, dtype=torch.float64))
        self.freeze_updates = False
        # Reassigned to a per-resolution subgroup under distill.partitioned_ranks so this teacher's
        # direction-variance collectives reduce only over the ranks that run it (else global -> hang).
        self.dist_group = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the variance-normalized angular loss between ``pred`` and ``target``."""
        flat_pred = pred.flatten(1)
        flat_target = target.flatten(1)
        with torch.no_grad():
            if not self.freeze_updates and self.num_samples < self.max_samples:
                curr_num = torch.tensor(flat_target.shape[0], dtype=torch.float64, device=pred.device)
                curr_dir_sum = flat_target.detach().sum(dim=0, dtype=torch.float64)
                if dist.is_initialized():
                    dist.all_reduce(curr_num, op=dist.ReduceOp.SUM, group=self.dist_group)
                    dist.all_reduce(curr_dir_sum, op=dist.ReduceOp.SUM, group=self.dist_group)
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
                    dist.all_reduce(curr_angle_var, op=dist.ReduceOp.SUM, group=self.dist_group)
                self.sum_angle_variance.add_(curr_angle_var)
        angle_variance = (self.sum_angle_variance / self.num_samples).to(pred.dtype).clamp_min(1e-8)
        cos_theta = F.cosine_similarity(flat_pred, flat_target, dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        angle_sq = torch.acos(cos_theta).pow(2)
        loss = (angle_sq / angle_variance).mean()
        return loss


class SummaryTangentSphereLoss(nn.Module):
    """Tangent-space sphere loss for summary distillation (native).
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
        # Reassigned to a per-resolution subgroup under distill.partitioned_ranks (else global -> hang).
        self.dist_group = None

    def _update_phis(self, target: torch.Tensor) -> None:
        if self.fwd_ct > 0:
            return
        with torch.no_grad():
            target_n = F.normalize(target.flatten(1), dim=-1).to(torch.float64)
            batch_n = target_n.shape[0]
            if dist.is_initialized():
                n_t = torch.tensor(batch_n, dtype=torch.float64, device=target.device)
                dist.all_reduce(n_t, op=dist.ReduceOp.SUM, group=self.dist_group)
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
                dist.all_reduce(mean_dir, op=dist.ReduceOp.SUM, group=self.dist_group)
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
                dist.all_reduce(log_map_mean, op=dist.ReduceOp.SUM, group=self.dist_group)
            log_map_mean = log_map_mean / self.num_samples.to(samples.device)
            self.tan_mean.copy_(log_map_mean)
            centered = log_map - log_map_mean
            cov = centered.T @ centered
            if dist.is_initialized():
                dist.all_reduce(cov, op=dist.ReduceOp.SUM, group=self.dist_group)
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
        summary_mlp_version: Optional[str] = None,
        summary_num_inner: Optional[int] = None,
        summary_loss_weight: float = 1.0,
        fd_loss_weight: float = 1.0,
        summary_loss_type: str = "CE",
        spatial_loss_type: str = "mse",
        spatial_focal_weight: float = 0.0,
        spatial_focal_gamma: float = 1.0,
        spatial_focal_max_weight: float = 4.0,
        intermediate_loss_weight: float = 0.0,
        intermediate_loss_weights: Optional[List[float]] = None,
        intermediate_feature_dims: Optional[List[int]] = None,
        intermediate_focal_weight: float = 0.0,
        intermediate_mlp_version: str = "residual",
        intermediate_num_inner: Optional[int] = None,
        spatial_norm_type: str = "phi",
        spatial_whiten_update_period: int = 100,
        spatial_whiten_freeze_after_steps: int = 0,
        spatial_whiten_shrinkage: float = 0.0,
        spatial_whiten_eigen_floor: float = 1.0e-6,
        spatial_whiten_max_gain: float = 0.0,
        spatial_projector_residual_scale: float = 0.25,
        spatial_projector_output_norm: bool = False,
        summary_token_idx: Optional[int] = None,
        partitioned_ranks: bool = False,
        mosaic_inner_size: int = 0,
        mosaic_outer_size: int = 0,
        mosaic_downsample: int = 0,
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
            summary_mlp_version (Optional[str]): Projection head version for the
                combo-mode summary projector (``projection_layer_summary``).
                Defaults to ``spatial_mlp_version`` (partitioned training: the reference implementation's
                ``self.summary_mlp_version = summary_mlp_version or mlp_version``,
                teacher.py:118). Previously this was hardcoded to "v2"
                (``ProjectionMLP``) regardless of ``spatial_mlp_version``.
            summary_num_inner (Optional[int]): Number of inner layers for the
                summary projection head. Defaults to ``mlp_num_inner`` for
                "v2", 0 for "attn", 1 for "residual".
            summary_loss_weight (float): Weight for summary/CLS loss in combo mode. Default: 1.0
            fd_loss_weight (float): Weight for spatial/fd loss in combo mode. Default: 1.0
            summary_loss_type (str): Summary loss in combo mode. One of ["CE", "angle", "cosine", "tangent_sphere"].
                Default: "CE" (soft cross-entropy with temperature). native options: angle, cosine, tangent_sphere.
            spatial_loss_type (str): Spatial (feature map) loss in combo/spatial mode.
                One of ["mse", "dampened_mse", "balanced", "cosine", "gram", "channel_kl"].
                native: "mse" = per-element squared error; "dampened_mse" = dampened for large residuals.
                Default: "mse".
            spatial_focal_weight (float): Strength of optional teacher-saliency weighting for spatial loss.
                0 disables focal weighting; values closer to 1 focus more on high-saliency teacher tokens.
            spatial_focal_gamma (float): Exponent applied to normalized teacher token saliency.
            spatial_focal_max_weight (float): Optional clamp for focal token weights before re-normalization.
            intermediate_loss_weight (float): Global weight for optional intermediate student spatial maps.
            intermediate_loss_weights (Optional[List[float]]): Per-intermediate relative weights.
            intermediate_feature_dims (Optional[List[int]]): Channel dims for intermediate projectors.
            intermediate_focal_weight (float): Focal weighting strength for intermediate spatial losses.
            intermediate_mlp_version (str): Projection head type for intermediate spatial maps.
            intermediate_num_inner (Optional[int]): Inner blocks for intermediate projection heads.
            spatial_norm_type (str): Teacher spatial normalization. One of ["phi", "zca", "pca"].
            summary_token_idx (int): Optional RADIO summary-token slot for per-teacher summary distillation.
        """
        super().__init__()

        self.loss_type = loss_type.upper()
        self.summary_loss_type = (summary_loss_type or "CE").lower()
        self.spatial_loss_type = (spatial_loss_type or "mse").lower()
        self.spatial_norm_type = (spatial_norm_type or "phi").lower()
        self.spatial_focal_weight = float(spatial_focal_weight or 0.0)
        self.spatial_focal_gamma = float(spatial_focal_gamma or 0.0)
        self.spatial_focal_max_weight = float(spatial_focal_max_weight or 0.0)
        self.intermediate_loss_weight = float(intermediate_loss_weight or 0.0)
        self.intermediate_loss_weights = [float(w) for w in (intermediate_loss_weights or [])]
        self.intermediate_feature_dims = [int(d) for d in (intermediate_feature_dims or [])]
        self.intermediate_focal_weight = float(intermediate_focal_weight or 0.0)
        self.intermediate_mlp_version = (intermediate_mlp_version or "residual").lower()
        self.intermediate_num_inner = intermediate_num_inner
        self.summary_loss_weight = float(summary_loss_weight)
        self.fd_loss_weight = float(fd_loss_weight)
        self.summary_token_idx = summary_token_idx
        self.partitioned_ranks = bool(partitioned_ranks)
        self.mosaic_inner_size = int(mosaic_inner_size or 0)
        self.mosaic_outer_size = int(mosaic_outer_size or 0)
        self.mosaic_downsample = int(mosaic_downsample or 0)
        if self.mosaic_inner_size > 0:
            if self.mosaic_outer_size <= 0 or self.mosaic_downsample <= 0:
                raise ValueError(
                    "RADIO mosaic requires positive inner, outer, and downsample sizes"
                )
            if float(summary_loss_weight) != 0.0:
                raise ValueError(
                    "RADIO MosaicAdaptor repeats a canvas summary across tiles; "
                    "mosaic teachers must set summary_loss_weight=0"
                )
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.num_classes = num_classes
        self.temperature = temperature
        self.use_mlp = bool(use_mlp)
        self.mlp_hidden_size = int(mlp_hidden_size)
        self.mlp_num_inner = int(mlp_num_inner)
        self.spatial_projector_residual_scale = float(spatial_projector_residual_scale)
        self.spatial_projector_output_norm = bool(spatial_projector_output_norm)
        self.alignment_metrics_enabled = os.environ.get("RADIO_ALIGNMENT_METRICS", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.last_alignment_metrics = {}

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
        if self.distillation_mode == "spatial" and self.spatial_loss_type == "mse":
            spatial_loss_aliases = {
                "BALANCED": "balanced",
                "CS": "cosine",
            }
            self.spatial_loss_type = spatial_loss_aliases.get(self.loss_type, self.spatial_loss_type)

        # Validate configuration for feature distillation
        if self.loss_type in ["FD", "CS", "BALANCED", "MSE"] and self.distillation_mode == "logits":
            raise ValueError(f"Use L1, L2, KL or CE loss for logits distillation, but {self.loss_type} was specified.")

        if self.loss_type in ["FD", "CS", "BALANCED", "MSE"] and num_classes > 0:
            raise ValueError(f"Number of classes must be 0 when using '{self.loss_type}' for distillation")

        # Get model dimensions by checking available methods
        self.student_dim, self.teacher_dim = self._get_model_dimensions()
        # Reassigned to a per-resolution subgroup under distill.partitioned_ranks so the spatial
        # loss's cross-rank sqrt(count) reduction covers only the ranks running this teacher.
        self.dist_group = None
        logging.info(f"student_dim: {self.student_dim}, teacher_dim: {self.teacher_dim}")

        # Create projection layer if dimensions differ and we're doing feature distillation
        self.projection_layer = None
        self.projection_layer_summary = None
        self.intermediate_projection_layers = nn.ModuleList()
        spatial_mlp_version = (spatial_mlp_version or "v2").lower()
        if spatial_mlp_version not in ("v2", "attn", "residual"):
            raise ValueError(
                f"Unsupported spatial_mlp_version: {spatial_mlp_version}. "
                "Must be 'v2', 'attn', or 'residual'."
            )
        summary_mlp_version = (summary_mlp_version or spatial_mlp_version).lower()
        if summary_mlp_version not in ("v2", "attn", "residual"):
            raise ValueError(
                f"Unsupported summary_mlp_version: {summary_mlp_version}. "
                "Must be 'v2', 'attn', or 'residual'."
            )
        if self.intermediate_mlp_version not in ("v2", "attn", "residual"):
            raise ValueError(
                f"Unsupported intermediate_mlp_version: {self.intermediate_mlp_version}. "
                "Must be 'v2', 'attn', or 'residual'."
            )
        if spatial_num_inner is None:
            if spatial_mlp_version == "attn":
                spatial_num_inner = 0
            elif spatial_mlp_version == "residual":
                spatial_num_inner = 1
            else:
                spatial_num_inner = mlp_num_inner
        if summary_num_inner is None:
            if summary_mlp_version == "attn":
                summary_num_inner = 0
            elif summary_mlp_version == "residual":
                summary_num_inner = 1
            else:
                summary_num_inner = mlp_num_inner
        self.spatial_mlp_version = spatial_mlp_version
        self.summary_mlp_version = summary_mlp_version
        if self.intermediate_num_inner is None:
            if self.intermediate_mlp_version == "attn":
                self.intermediate_num_inner = 0
            elif self.intermediate_mlp_version == "residual":
                self.intermediate_num_inner = 1
            else:
                self.intermediate_num_inner = mlp_num_inner
        if self.student_dim != self.teacher_dim or isinstance(self.student_model, RADIO) or isinstance(self.teacher_model, RADIO):
            if use_mlp:
                if spatial_mlp_version == "attn":
                    self.projection_layer = AttnFDHead(
                        self.student_dim, mlp_hidden_size, self.teacher_dim,
                        num_inner=spatial_num_inner,
                    )
                elif spatial_mlp_version == "residual":
                    self.projection_layer = ResidualProjectionMLP(
                        self.student_dim, mlp_hidden_size, self.teacher_dim,
                        num_inner=spatial_num_inner,
                        residual_scale=spatial_projector_residual_scale,
                        output_norm=spatial_projector_output_norm,
                    )
                else:
                    self.projection_layer = ProjectionMLP(
                        self.student_dim, mlp_hidden_size, self.teacher_dim,
                        num_inner=spatial_num_inner,
                    )
                if self.distillation_mode == "combo":
                    student_dim_summary = self._summary_feature_dim(self.student_model, self.student_dim)
                    teacher_dim_summary = self._summary_feature_dim(self.teacher_model, self.teacher_dim)
                    if summary_mlp_version == "attn":
                        self.projection_layer_summary = AttnFDHead(
                            student_dim_summary, mlp_hidden_size, teacher_dim_summary,
                            num_inner=summary_num_inner,
                        )
                    elif summary_mlp_version == "residual":
                        self.projection_layer_summary = ResidualProjectionMLP(
                            student_dim_summary, mlp_hidden_size, teacher_dim_summary,
                            num_inner=summary_num_inner,
                            residual_scale=spatial_projector_residual_scale,
                            output_norm=spatial_projector_output_norm,
                        )
                    else:
                        self.projection_layer_summary = ProjectionMLP(
                            student_dim_summary, mlp_hidden_size, teacher_dim_summary,
                            num_inner=summary_num_inner,
                        )
            else:
                self.projection_layer = nn.Linear(self.student_dim, self.teacher_dim, bias=True)
                if self.distillation_mode == "combo":
                    student_dim_summary = self._summary_feature_dim(self.student_model, self.student_dim)
                    teacher_dim_summary = self._summary_feature_dim(self.teacher_model, self.teacher_dim)
                    self.projection_layer_summary = nn.Linear(student_dim_summary, teacher_dim_summary, bias=True)

        if self.intermediate_loss_weight > 0.0:
            if self.distillation_mode != "combo":
                raise ValueError("intermediate_loss_weight is only supported for combo distillation mode.")
            if not 0.0 <= self.intermediate_focal_weight <= 1.0:
                raise ValueError(
                    "intermediate_focal_weight must be in [0, 1], "
                    f"got {self.intermediate_focal_weight}"
                )
            dims = self.intermediate_feature_dims or self._infer_student_intermediate_dims()
            if not dims:
                raise ValueError(
                    "intermediate_loss_weight > 0 requires intermediate_feature_dims "
                    "or a student model exposing inferable intermediate feature maps."
                )
            self.intermediate_feature_dims = [int(dim) for dim in dims]
            if self.intermediate_loss_weights and len(self.intermediate_loss_weights) != len(self.intermediate_feature_dims):
                raise ValueError(
                    "intermediate_loss_weights must match intermediate_feature_dims length: "
                    f"{len(self.intermediate_loss_weights)} vs {len(self.intermediate_feature_dims)}"
                )
            for input_dim in self.intermediate_feature_dims:
                self.intermediate_projection_layers.append(
                    self._build_spatial_projector(
                        input_dim=input_dim,
                        version=self.intermediate_mlp_version,
                        num_inner=int(self.intermediate_num_inner or 0),
                    )
                )
            logging.info(
                "Using intermediate spatial supervision: weight=%s dims=%s weights=%s projector=%s focal=%s",
                self.intermediate_loss_weight,
                self.intermediate_feature_dims,
                self.intermediate_loss_weights or "uniform",
                self.intermediate_mlp_version,
                self.intermediate_focal_weight,
            )
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
            if self.spatial_norm_type == "phi":
                self.phi_norm = PHIStandardization(
                    name='phi_norm',
                    feature_dim=self.teacher_dim,
                    ohem=False,
                    update_period=spatial_whiten_update_period,
                    rotate=True
                )
            elif self.spatial_norm_type in ("zca", "pca"):
                self.phi_norm = CovarianceWhitening(
                    name=f'{self.spatial_norm_type}_whiten',
                    feature_dim=self.teacher_dim,
                    ohem=False,
                    update_period=spatial_whiten_update_period,
                    mode=self.spatial_norm_type,
                    freeze_after_steps=spatial_whiten_freeze_after_steps,
                    shrinkage=spatial_whiten_shrinkage,
                    eigen_floor=spatial_whiten_eigen_floor,
                    max_gain=spatial_whiten_max_gain,
                )
            else:
                raise ValueError(
                    "spatial_norm_type must be one of ('phi', 'zca', 'pca'), "
                    f"got {self.spatial_norm_type!r}"
                )
            valid_spatial_losses = ("mse", "dampened_mse", "balanced", "cosine", "gram", "channel_kl", "fft_mse")
            if self.spatial_loss_type not in valid_spatial_losses:
                raise ValueError(
                    f"spatial_loss_type must be one of {valid_spatial_losses}, got {self.spatial_loss_type!r}"
                )
            logging.info(f"Using spatial_loss_type={self.spatial_loss_type} for feature map distillation")
            if not 0.0 <= self.spatial_focal_weight <= 1.0:
                raise ValueError(
                    "spatial_focal_weight must be in [0, 1], "
                    f"got {self.spatial_focal_weight}"
                )
            if self.spatial_focal_weight > 0.0:
                logging.info(
                    "Using teacher-saliency spatial focal weighting: weight=%s gamma=%s max_weight=%s",
                    self.spatial_focal_weight,
                    self.spatial_focal_gamma,
                    self.spatial_focal_max_weight,
                )

        # Summary loss criterion for combo mode (native: angle, cosine, tangent_sphere)
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
                teacher_dim_summary = self._summary_feature_dim(self.teacher_model, self.teacher_dim)
                if self.summary_loss_type == "angle":
                    self.summary_criterion = SummaryAngleLoss(feature_dim=teacher_dim_summary)
                elif self.summary_loss_type == "cosine":
                    self.summary_criterion = SummaryCosineLoss()
                else:
                    self.summary_criterion = SummaryTangentSphereLoss(feature_dim=teacher_dim_summary)

    def _build_spatial_projector(self, input_dim: int, version: str, num_inner: int) -> nn.Module:
        """Build a student-to-teacher projector for a spatial token stream."""
        if not self.use_mlp:
            return nn.Linear(input_dim, self.teacher_dim, bias=True)
        if version == "attn":
            return AttnFDHead(
                input_dim,
                self.mlp_hidden_size,
                self.teacher_dim,
                num_inner=num_inner,
            )
        if version == "residual":
            return ResidualProjectionMLP(
                input_dim,
                self.mlp_hidden_size,
                self.teacher_dim,
                num_inner=num_inner,
                residual_scale=self.spatial_projector_residual_scale,
                output_norm=self.spatial_projector_output_norm,
            )
        return ProjectionMLP(
            input_dim,
            self.mlp_hidden_size,
            self.teacher_dim,
            num_inner=num_inner,
        )

    def _infer_student_intermediate_dims(self) -> List[int]:
        """Infer intermediate feature dims from the student model without a forward pass."""
        raw_model = getattr(self.student_model, "_model", None)
        if raw_model is None:
            return []

        dims = []
        norm = getattr(raw_model, "norm", None)
        if hasattr(norm, "num_features"):
            dims.append(int(norm.num_features))
        if getattr(raw_model, "output_stride3", False):
            norm3 = getattr(raw_model, "bn_norm3", None)
            if hasattr(norm3, "num_features"):
                dims.append(int(norm3.num_features))
        if getattr(raw_model, "output_stride2", False):
            norm2 = getattr(raw_model, "bn_norm2", None)
            if hasattr(norm2, "num_features"):
                dims.append(int(norm2.num_features))
        return dims

    def _summary_feature_dim(self, model: nn.Module, fallback_dim: int) -> int:
        """Return the summary dimension used by this loss for a model.

        Args:
            model (nn.Module): Student or teacher model that may be a RADIO
                model with multiple summary tokens.
            fallback_dim (int): Dimension to use for non-RADIO models.

        Returns:
            int: Per-token summary feature dimension used by the loss.
        """
        summary_features = getattr(model, "num_summary_features", None)
        if summary_features is not None:
            return int(summary_features)

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

    @staticmethod
    def _get_last_feature_map(features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]):
        """Extract the last feature map from a list/tuple/dict or return the tensor itself."""
        if isinstance(features, (list, tuple)):
            return features[-1]
        elif isinstance(features, dict):
            return list(features.values())[-1]
        return features

    @staticmethod
    def _get_intermediate_feature_maps(features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]):
        """Return all feature maps before the final spatial map."""
        if isinstance(features, (list, tuple)):
            return list(features[:-1])
        if isinstance(features, dict):
            values = list(features.values())
            return values[:-1]
        return []

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
            # Accept a pooled feature token only when every contributing input
            # pixel is valid.
            mask = mask == 1.0
        return mask.to(device)

    def _align_features(self, student_feat, teacher_feat,
                        student_valid_mask, teacher_valid_mask,
                        spatial_transform):
        """Align student and teacher feature maps."""
        if spatial_transform is not None:
            s_valid = student_valid_mask.float()
            t_valid = teacher_valid_mask.float()

            if (
                self.partitioned_ranks and
                teacher_feat.shape[-1] > student_feat.shape[-1]
            ):
                grid = generate_homography_grid(
                    torch.linalg.inv(spatial_transform), student_feat.shape
                )
                teacher_feat = F.grid_sample(
                    teacher_feat, grid, mode="bilinear", align_corners=True
                )
                t_valid = F.grid_sample(
                    t_valid.unsqueeze(1), grid, mode="bilinear", align_corners=True
                ).squeeze(1)
                s_valid = F.adaptive_avg_pool2d(
                    s_valid.unsqueeze(1), student_feat.shape[-2:]
                ).squeeze(1)
            else:
                grid = generate_homography_grid(spatial_transform, teacher_feat.shape)
                student_feat = F.grid_sample(
                    student_feat, grid, mode='bilinear', align_corners=True,
                )
                s_valid = F.grid_sample(
                    s_valid.unsqueeze(1), grid, mode='bilinear', align_corners=True,
                ).squeeze(1)

            valid_mask = s_valid * t_valid
            eps = 1e-8 if self.partitioned_ranks else 1e-6
            valid_mask = valid_mask * (1 - eps) + eps
            if self.partitioned_ranks:
                eq_valid = torch.all(grid.abs() <= 1, dim=-1)
                valid_mask = torch.where(eq_valid, valid_mask, 0.0)

            grid_h, grid_w = teacher_feat.shape[-2], teacher_feat.shape[-1]
            student_feat = rearrange(student_feat, 'b c h w -> b (h w) c')
            teacher_feat = rearrange(teacher_feat, 'b c h w -> b (h w) c')
            valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
            return student_feat, teacher_feat, valid_mask, (grid_h, grid_w)
        else:
            if student_feat.shape[2:] != teacher_feat.shape[2:]:
                target = tuple(
                    (max if self.partitioned_ranks else min)(s, t)
                    for s, t in zip(student_feat.shape[2:], teacher_feat.shape[2:])
                )
                align_corners = bool(self.partitioned_ranks)
                student_feat = F.interpolate(
                    student_feat, size=target, mode='bilinear', align_corners=align_corners
                )
                teacher_feat = F.interpolate(
                    teacher_feat, size=target, mode='bilinear', align_corners=align_corners
                )
            grid_h, grid_w = teacher_feat.shape[-2], teacher_feat.shape[-1]
            student_feat = rearrange(student_feat, 'b c h w -> b (h w) c')
            teacher_feat = rearrange(teacher_feat, 'b c h w -> b (h w) c')
            return student_feat, teacher_feat, None, (grid_h, grid_w)

    def _spatial_feature_loss(
        self,
        student_spatial: torch.Tensor,
        teacher_spatial: torch.Tensor,
        eq_mask: Optional[torch.Tensor],
        focal_weight: Optional[float] = None,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """native spatial (feature map) loss: per-position loss, then masked mean.

        Supports "mse" (per-element squared error), "dampened_mse" (dampened for
        large residuals), "balanced" (cosine plus SmoothL1), and "fft_mse".
        Partitioned training uses a masked teacher-group global-token denominator.
        """
        if self.spatial_loss_type == "cosine":
            loss_per_pos = 1.0 - F.cosine_similarity(
                student_spatial, teacher_spatial, dim=-1, eps=1e-8
            )
        elif self.spatial_loss_type == "gram":
            return self._spatial_gram_loss(student_spatial, teacher_spatial, eq_mask)
        elif self.spatial_loss_type == "channel_kl":
            return self._spatial_channel_kl_loss(student_spatial, teacher_spatial, eq_mask)
        elif self.spatial_loss_type == "fft_mse":
            if grid_hw is None:
                raise ValueError(
                    "fft_mse spatial loss requires grid_hw=(H, W); thread it "
                    "from _align_features into _spatial_feature_loss."
                )
            if self.partitioned_ranks:
                loss_per_pos = _fft_mse_loss(
                    student_spatial, teacher_spatial, grid_hw, reduction="none"
                )
            else:
                return _fft_mse_loss(student_spatial, teacher_spatial, grid_hw)
        elif self.spatial_loss_type == "balanced":
            loss_cos = 1.0 - F.cosine_similarity(
                student_spatial, teacher_spatial, dim=-1, eps=1e-8
            )
            loss_l1 = F.smooth_l1_loss(
                student_spatial, teacher_spatial, beta=2.0, reduction="none"
            ).mean(dim=-1)
            loss_per_pos = 0.9 * loss_cos + 0.1 * loss_l1
        elif self.spatial_loss_type == "dampened_mse":
            element_wise = dampened_mse_loss(student_spatial, teacher_spatial)
            loss_per_pos = element_wise.mean(dim=-1)
        else:
            element_wise = _mse_element_wise(student_spatial, teacher_spatial)
            loss_per_pos = element_wise.mean(dim=-1)
        if self.partitioned_ranks:
            lp = loss_per_pos.reshape(loss_per_pos.shape[0], -1).float()
            if eq_mask is None:
                mask = torch.ones_like(lp)
            else:
                mask = eq_mask.reshape(eq_mask.shape[0], -1).float()
            local_num_valid = mask.sum(dim=1)
            global_num_valid = local_num_valid.sum()
            if dist.is_initialized():
                dist.all_reduce(
                    global_num_valid, op=dist.ReduceOp.SUM, group=self.dist_group
                )
            group_world = get_world_size(self.dist_group)
            weight = (
                group_world * local_num_valid.shape[0]
            ) / global_num_valid.clamp_min(1.0)
            return ((lp * mask).sum(dim=1) * weight).mean()

        if eq_mask is not None:
            focal_weights = self._spatial_focal_weights(teacher_spatial, eq_mask, focal_weight=focal_weight)
            if focal_weights is not None:
                loss_per_pos = loss_per_pos * focal_weights
            # RADIO per-example size weighting (per_ex_weight_alpha=0.5): take each
            # example's masked-mean loss, weight it by sqrt(#valid tokens), then a
            # weighted mean across the batch. This is between flat-per-image
            # (count^0) and flat-per-token (count^1) reduction, matching
            # feature_distillation_loss.py (per_ex_counts ** 0.5). The previous
            # global masked mean (loss.sum()/mask.sum()) is the count^1 special case.
            lp = loss_per_pos.reshape(loss_per_pos.shape[0], -1)
            # Count/normalize in fp32 so the B=1 case is an *exact* no-op vs the
            # old global masked mean and bf16 doesn't round token counts.
            m = eq_mask.reshape(eq_mask.shape[0], -1).float()
            lp = lp.float()
            per_ex_valid = m.sum(dim=1)
            per_ex_loss = (lp * m).sum(dim=1) / per_ex_valid.clamp(min=1.0)
            per_ex_weight = per_ex_valid.clamp(min=0.0).sqrt()
            # RADIO cross-rank normalization: sum the sqrt(count) weights over the ranks running THIS
            # teacher (self.dist_group = its per-resolution subgroup under partitioning, else global)
            # and scale by global_batch_size, so DDP gradient-averaging yields a single cross-rank
            # sqrt(token-count)-weighted mean (the deployed high-token-count arm is not under-weighted).
            total_weight = per_ex_weight.sum()
            if dist.is_initialized():
                dist.all_reduce(total_weight, op=dist.ReduceOp.SUM, group=self.dist_group)
            global_batch_size = per_ex_loss.shape[0] * get_world_size(self.dist_group)
            per_ex_weight = (global_batch_size * per_ex_weight) / total_weight.clamp(min=1.0)
            loss_spatial = (per_ex_weight * per_ex_loss).mean()
        else:
            focal_weights = self._spatial_focal_weights(teacher_spatial, None, focal_weight=focal_weight)
            if focal_weights is not None:
                loss_per_pos = loss_per_pos * focal_weights
            loss_spatial = loss_per_pos.mean()
        return loss_spatial

    def _spatial_focal_weights(
        self,
        teacher_spatial: torch.Tensor,
        eq_mask: Optional[torch.Tensor],
        focal_weight: Optional[float] = None,
        eps: float = 1e-6,
    ) -> Optional[torch.Tensor]:
        """Build mean-one token weights from teacher spatial saliency.

        This is a label-free analogue of focal feature distillation: every valid
        token still contributes, but tokens with larger teacher activation norm
        get a larger share of the spatial feature loss. The final weights are
        normalized to mean one over valid tokens, so enabling this does not
        silently change the global spatial-loss scale.
        """
        weight = self.spatial_focal_weight if focal_weight is None else float(focal_weight)
        if weight <= 0.0:
            return None

        saliency = teacher_spatial.detach().float().pow(2).mean(dim=-1).sqrt()
        if eq_mask is None:
            valid = torch.ones_like(saliency)
        else:
            valid = eq_mask.to(dtype=saliency.dtype, device=saliency.device)

        count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        saliency = saliency * valid
        saliency_mean = saliency.sum(dim=1, keepdim=True) / count
        focus = (saliency / saliency_mean.clamp_min(eps)).clamp_min(eps)
        if self.spatial_focal_gamma != 1.0:
            focus = focus.pow(self.spatial_focal_gamma)
        focus = focus * valid
        focus_mean = focus.sum(dim=1, keepdim=True) / count
        focus = focus / focus_mean.clamp_min(eps)

        weights = (1.0 - weight) + weight * focus
        weights = weights * valid

        if self.spatial_focal_max_weight > 0.0:
            weights = weights.clamp(max=self.spatial_focal_max_weight) * valid
            weight_mean = weights.sum(dim=1, keepdim=True) / count
            weights = weights / weight_mean.clamp_min(eps)
            weights = weights * valid

        if self.alignment_metrics_enabled:
            with torch.no_grad():
                valid_bool = valid > 0
                if valid_bool.any():
                    valid_weights = weights[valid_bool]
                    self.last_alignment_metrics.update({
                        "spatial_focal_weight_mean": valid_weights.mean().detach(),
                        "spatial_focal_weight_max": valid_weights.max().detach(),
                    })
        return weights

    def _normalized_intermediate_weights(self, count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return mean-scale-stable relative weights for intermediate losses."""
        if self.intermediate_loss_weights:
            weights = torch.tensor(self.intermediate_loss_weights, device=device, dtype=dtype)
        else:
            weights = torch.ones(count, device=device, dtype=dtype)
        if weights.numel() != count:
            raise ValueError(f"Expected {count} intermediate weights, got {weights.numel()}")
        weight_sum = weights.sum().clamp_min(torch.finfo(dtype).eps)
        return weights / weight_sum

    def _intermediate_spatial_feature_loss(
        self,
        student_features: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
        teacher_spatial: torch.Tensor,
        student_valid_mask: Optional[torch.Tensor],
        teacher_valid_mask: Optional[torch.Tensor],
        spatial_transform: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Distill lower-resolution student maps against downsampled teacher features."""
        if self.intermediate_loss_weight <= 0.0 or not self.intermediate_projection_layers:
            return None

        student_maps = self._get_intermediate_feature_maps(student_features)
        num_levels = len(self.intermediate_projection_layers)
        if len(student_maps) < num_levels:
            raise RuntimeError(
                f"Student returned {len(student_maps)} intermediate feature maps, "
                f"but {num_levels} intermediate projection heads are configured."
            )

        student_maps = student_maps[:num_levels]
        weights = self._normalized_intermediate_weights(
            num_levels,
            teacher_spatial.device,
            torch.float32,
        )
        total = teacher_spatial.new_zeros((), dtype=torch.float32)
        per_level_losses = []

        B = teacher_spatial.shape[0]
        if teacher_valid_mask is None:
            teacher_valid_mask = torch.ones(
                B,
                teacher_spatial.shape[-2],
                teacher_spatial.shape[-1],
                dtype=torch.float32,
                device=teacher_spatial.device,
            )

        for level_idx, (student_map, projector, level_weight) in enumerate(
            zip(student_maps, self.intermediate_projection_layers, weights)
        ):
            if student_map.ndim != 4:
                raise RuntimeError(
                    f"Intermediate feature {level_idx} must be BCHW, got shape {list(student_map.shape)}"
                )
            if student_valid_mask is None:
                level_student_mask = torch.ones(
                    student_map.shape[0],
                    student_map.shape[-2],
                    student_map.shape[-1],
                    dtype=torch.float32,
                    device=student_map.device,
                )
            else:
                level_student_mask = self._build_mask(
                    student_valid_mask,
                    student_map.shape[-2:],
                    student_map.device,
                ).float()

            # RADIO order: project at the student map's native resolution, THEN warp
            # (mirrors the main spatial path above; keeps the projected map's gradient
            # constraining the full student rep rather than an already-warped copy).
            _h, _w = student_map.shape[2], student_map.shape[3]
            student_map = rearrange(student_map, 'b c h w -> b (h w) c')
            student_map = projector(student_map)
            student_map = rearrange(student_map, 'b (h w) c -> b c h w', h=_h, w=_w)

            student_level, teacher_level, eq_mask, level_grid_hw = self._align_features(
                student_map,
                teacher_spatial,
                level_student_mask,
                teacher_valid_mask,
                spatial_transform,
            )
            level_loss = self._spatial_feature_loss(
                student_level,
                teacher_level,
                eq_mask,
                focal_weight=self.intermediate_focal_weight,
                grid_hw=level_grid_hw,
            )
            total = total + level_weight * level_loss.float()
            per_level_losses.append(level_loss.detach())

        if self.alignment_metrics_enabled:
            with torch.no_grad():
                self.last_alignment_metrics["intermediate_spatial_loss"] = total.detach()
                for idx, level_loss in enumerate(per_level_losses):
                    self.last_alignment_metrics[f"intermediate_spatial_loss_{idx}"] = level_loss
        return total.to(dtype=teacher_spatial.dtype)

    @torch.no_grad()
    def compute_spatial_cka_stats(
        self,
        batch_input: torch.Tensor,
        teacher_batch_input: Optional[torch.Tensor] = None,
        student_valid_mask: Optional[torch.Tensor] = None,
        teacher_valid_mask: Optional[torch.Tensor] = None,
        spatial_transform: Optional[torch.Tensor] = None,
        student_spatial: Optional[torch.Tensor] = None,
    ) -> Optional[dict]:
        """Compute sufficient statistics for linear CKA between *raw* student and
        teacher spatial feature maps.

        This is intended as a projector-independent validation proxy of backbone
        quality: it operates on the raw backbone features (no projection layer,
        no PHI/teacher normalization), so the resulting CKA is directly
        comparable across runs regardless of which distillation loss they train
        with. Student and teacher may have different feature widths; CKA handles
        that natively.

        Returns a dict of accumulators (cross/self covariance sums, per-feature
        sums and a token count, all float64) that can be summed across batches
        and ranks. The final CKA is computed from these by the caller. Returns
        ``None`` when no valid tokens are present.
        """
        teacher_input = teacher_batch_input if teacher_batch_input is not None else batch_input
        device = batch_input.device

        if student_spatial is not None:
            student_feat = self._get_last_feature_map(student_spatial)
        else:
            student_feat = self._get_last_feature_map(
                self.student_model.forward_feature_pyramid(batch_input)
            )
        teacher_feat = self._get_last_feature_map(
            self.teacher_model.forward_feature_pyramid(teacher_input)
        )

        B, _, H, W = teacher_feat.shape
        if teacher_valid_mask is not None:
            t_mask = self._build_mask(
                teacher_valid_mask,
                (H, W),
                device,
            ).float()
        else:
            t_mask = torch.ones(B, H, W, dtype=torch.float32, device=device)
        if student_valid_mask is None:
            s_mask = torch.ones(
                B, student_feat.shape[2], student_feat.shape[3],
                dtype=torch.float32, device=device,
            )
        else:
            s_mask = student_valid_mask.float()

        student_feat, teacher_feat, eq_mask, _ = self._align_features(
            student_feat, teacher_feat, s_mask, t_mask, spatial_transform,
        )

        x = student_feat.reshape(-1, student_feat.shape[-1]).float()
        y = teacher_feat.reshape(-1, teacher_feat.shape[-1]).float()
        if eq_mask is not None:
            valid = eq_mask.reshape(-1) > 0.5
            x = x[valid]
            y = y[valid]
        if x.shape[0] == 0:
            return None

        x = x.double()
        y = y.double()
        return {
            "sum_xy": x.T @ y,
            "sum_xx": x.T @ x,
            "sum_yy": y.T @ y,
            "sum_x": x.sum(dim=0),
            "sum_y": y.sum(dim=0),
            "count": torch.tensor(float(x.shape[0]), dtype=torch.float64, device=device),
        }

    def _spatial_gram_loss(
        self,
        student_spatial: torch.Tensor,
        teacher_spatial: torch.Tensor,
        eq_mask: Optional[torch.Tensor],
        max_tokens: int = 512,
    ) -> torch.Tensor:
        """Match token-token relations instead of exact per-channel values."""
        num_tokens = student_spatial.shape[1]
        if num_tokens > max_tokens:
            idx = torch.linspace(
                0,
                num_tokens - 1,
                steps=max_tokens,
                device=student_spatial.device,
            ).long()
            student_spatial = student_spatial.index_select(1, idx)
            teacher_spatial = teacher_spatial.index_select(1, idx)
            if eq_mask is not None:
                eq_mask = eq_mask.index_select(1, idx)

        student_norm = F.normalize(student_spatial.float(), dim=-1, eps=1e-8)
        teacher_norm = F.normalize(teacher_spatial.float(), dim=-1, eps=1e-8)
        student_gram = torch.bmm(student_norm, student_norm.transpose(1, 2))
        teacher_gram = torch.bmm(teacher_norm, teacher_norm.transpose(1, 2))
        loss = (student_gram - teacher_gram).pow(2)

        if eq_mask is None:
            return loss.mean()

        valid = eq_mask.float()
        pair_valid = valid.unsqueeze(1) * valid.unsqueeze(2)
        return (loss * pair_valid).sum() / pair_valid.sum().clamp(min=1.0)

    def _spatial_channel_kl_loss(
        self,
        student_spatial: torch.Tensor,
        teacher_spatial: torch.Tensor,
        eq_mask: Optional[torch.Tensor],
        temperature: float = 4.0,
    ) -> torch.Tensor:
        """Match each channel's distribution over spatial positions."""
        student_logits = student_spatial.float().transpose(1, 2) / temperature
        teacher_logits = teacher_spatial.float().transpose(1, 2) / temperature

        if eq_mask is not None:
            invalid = ~eq_mask.bool().unsqueeze(1)
            student_logits = student_logits.masked_fill(invalid, -1e4)
            teacher_logits = teacher_logits.masked_fill(invalid, -1e4)

        teacher_prob = F.softmax(teacher_logits, dim=-1)
        student_log_prob = F.log_softmax(student_logits, dim=-1)
        loss = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1)
        return loss.mean() * (temperature ** 2)

    def _pack_mosaic(self, images: torch.Tensor):
        """Pack outer views into RADIO MosaicAdaptor's fixed inner canvas."""
        if self.mosaic_inner_size <= 0:
            return images, None

        batch, channels, height, width = images.shape
        if height != self.mosaic_outer_size or width != self.mosaic_outer_size:
            raise ValueError(
                "RADIO mosaic received the wrong outer view size: "
                f"expected {self.mosaic_outer_size}, got {(height, width)}"
            )
        inner = self.mosaic_inner_size
        stride = self.mosaic_downsample
        outer_grid_h = height // stride
        outer_grid_w = width // stride
        num_per_col = (inner // stride) // outer_grid_h
        num_per_row = (inner // stride) // outer_grid_w
        if num_per_col <= 0 or num_per_row <= 0:
            raise ValueError(
                f"Mosaic canvas {inner} cannot contain outer view {(height, width)} "
                f"at feature stride {stride}"
            )
        num_per_image = num_per_col * num_per_row
        canvas_batch = int(math.ceil(batch / num_per_image))
        tile_h = inner // num_per_col
        tile_w = inner // num_per_row
        canvas = images.new_zeros((canvas_batch, channels, inner, inner))
        for idx in range(batch):
            canvas_idx = idx // num_per_image
            tile_idx = idx % num_per_image
            row = tile_idx // num_per_row
            col = tile_idx % num_per_row
            y = row * tile_h
            x = col * tile_w
            canvas[canvas_idx, :, y:y + height, x:x + width] = images[idx]
        state = {
            "batch": batch,
            "num_per_image": num_per_image,
            "num_per_row": num_per_row,
            "tile_grid_h": tile_h // stride,
            "tile_grid_w": tile_w // stride,
            "outer_grid_h": outer_grid_h,
            "outer_grid_w": outer_grid_w,
        }
        return canvas, state

    @staticmethod
    def _unpack_mosaic_features(features: torch.Tensor, state):
        """Slice one teacher feature window back out for every outer view."""
        if state is None:
            return features
        windows = []
        for idx in range(state["batch"]):
            canvas_idx = idx // state["num_per_image"]
            tile_idx = idx % state["num_per_image"]
            row = tile_idx // state["num_per_row"]
            col = tile_idx % state["num_per_row"]
            y = row * state["tile_grid_h"]
            x = col * state["tile_grid_w"]
            windows.append(
                features[
                    canvas_idx,
                    :,
                    y:y + state["outer_grid_h"],
                    x:x + state["outer_grid_w"],
                ]
            )
        return torch.stack(windows, dim=0)

    @staticmethod
    def _unpack_mosaic_summary(summary: torch.Tensor, state):
        """Repeat canvas summaries for their corresponding outer views."""
        if state is None:
            return summary
        return torch.repeat_interleave(
            summary, state["num_per_image"], dim=0
        )[:state["batch"]]

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
        teacher_input, mosaic_state = self._pack_mosaic(teacher_input)
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
                teacher_output = self._unpack_mosaic_features(
                    teacher_output, mosaic_state
                )
            # normalize the teacher feature maps
            B, _, H, W = teacher_output.shape
            if teacher_valid_mask is not None:
                t_mask = self._build_mask(
                    teacher_valid_mask,
                    (H, W),
                    device,
                )
            else:
                t_mask = torch.ones(B, H, W, dtype=torch.bool, device=device)
            if self.training:
                self.phi_norm.update(None, teacher_output, t_mask)
            teacher_output = self.phi_norm.transform_targets(teacher_output)

            # align the shape of student and teacher feature maps
            if student_valid_mask is None:
                student_valid_mask = torch.ones(B, student_output.shape[2], student_output.shape[3],
                                                dtype=torch.float32, device=device)
            teacher_valid_mask = t_mask.float()

            student_output, teacher_output, eq_mask, grid_hw = self._align_features(
                student_output, teacher_output,
                student_valid_mask, teacher_valid_mask,
                spatial_transform,
            )
            if self.projection_layer is not None:
                student_output = self.projection_layer(student_output)
        elif self.distillation_mode == "combo":
            self.last_alignment_metrics = {}
            teacher_sig = inspect.signature(self.teacher_model.forward)
            student_sig = inspect.signature(self.student_model.forward)
            assert 'return_features' in teacher_sig.parameters, "Teacher model must support return_features in `combo` mode"
            assert 'return_features' in student_sig.parameters, "Student model must support return_features in `combo` mode"
            if student_summary is not None and student_spatial is not None:
                student_spatial_features = student_spatial
                student_spatial = self._get_last_feature_map(student_spatial)
            else:
                return_intermediates = self.intermediate_loss_weight > 0.0
                if return_intermediates and "return_intermediate_features" not in student_sig.parameters:
                    raise RuntimeError(
                        "intermediate_loss_weight > 0 requires the student forward to accept "
                        "return_intermediate_features."
                    )
                if return_intermediates:
                    student_summary, student_spatial_features = self.student_model.forward(
                        batch_input,
                        return_features=True,
                        return_intermediate_features=True,
                    )
                else:
                    student_summary, student_spatial_features = self.student_model.forward(batch_input, return_features=True)
                student_spatial = self._get_last_feature_map(student_spatial_features)
            with torch.no_grad():
                teacher_summary, teacher_spatial = self.teacher_model.forward(teacher_input, return_features=True)
                teacher_spatial = self._get_last_feature_map(teacher_spatial)
                teacher_spatial = self._unpack_mosaic_features(
                    teacher_spatial, mosaic_state
                )
                teacher_summary = self._unpack_mosaic_summary(
                    teacher_summary, mosaic_state
                )

            # normalize the teacher feature maps
            B, _, H, W = teacher_spatial.shape
            if teacher_valid_mask is not None:
                t_mask = self._build_mask(
                    teacher_valid_mask,
                    (H, W),
                    device,
                )
            else:
                t_mask = torch.ones(B, H, W, dtype=torch.bool, device=device)
            if self.training:
                self.phi_norm.update(None, teacher_spatial, t_mask)
            teacher_spatial = self.phi_norm.transform_targets(teacher_spatial)
            teacher_spatial_map = teacher_spatial
            # align the shape of student and teacher feature maps

            if student_valid_mask is None:
                student_valid_mask = torch.ones(B, student_spatial.shape[2], student_spatial.shape[3],
                                                dtype=torch.float32, device=device)
            teacher_valid_mask = t_mask.float()

            # RADIO order: project the student at its NATIVE resolution, THEN warp/align.
            # tao previously ran _align_features first -- warping the raw student down to
            # the teacher grid -- and projected the already-downsampled student. On the
            # deployed (1920/120x120) arm that under-supervised the full-resolution rep's
            # high frequencies: with identical inputs the deployed-student gradient under
            # the two orders diverges (cos 0.79) and tao's order carries less high-freq
            # energy (0.615 vs RADIO 0.727) while the projected loss is ~unchanged (ratio
            # 0.95). The deployed rep therefore drifts as training continues past unfreeze
            # (peak-then-decline) even though the loss looks healthy. Projecting at native
            # resolution first matches RADIO (feature_distillation_loss.py forward, which
            # applies the student adaptor before resize_fn) and keeps the loss gradient
            # constraining the full-resolution deployed backbone.
            if self.projection_layer is not None:
                _h, _w = student_spatial.shape[2], student_spatial.shape[3]
                student_spatial = rearrange(student_spatial, 'b c h w -> b (h w) c')
                student_spatial = self.projection_layer(student_spatial)
                student_spatial = rearrange(
                    student_spatial, 'b (h w) c -> b c h w', h=_h, w=_w
                )

            student_spatial, teacher_spatial, eq_mask, grid_hw = self._align_features(
                student_spatial, teacher_spatial,
                student_valid_mask, teacher_valid_mask,
                spatial_transform,
            )
            if self.alignment_metrics_enabled:
                with torch.no_grad():
                    spatial_cos = F.cosine_similarity(
                        student_spatial.float(), teacher_spatial.float(), dim=-1, eps=1e-8
                    )
                    spatial_mse = (student_spatial.float() - teacher_spatial.float()).pow(2).mean(dim=-1)
                    spatial_cka = linear_cka(student_spatial, teacher_spatial, eq_mask)
                    self.last_alignment_metrics.update({
                        "spatial_cosine": _masked_token_mean(spatial_cos, eq_mask).detach(),
                        "spatial_mse": _masked_token_mean(spatial_mse, eq_mask).detach(),
                        "spatial_cka": spatial_cka.detach(),
                        "spatial_student_norm": _masked_token_mean(
                            student_spatial.float().norm(dim=-1), eq_mask
                        ).detach(),
                        "spatial_teacher_norm": _masked_token_mean(
                            teacher_spatial.float().norm(dim=-1), eq_mask
                        ).detach(),
                    })
            # spatial (feature distillation) loss — native: element-wise (mse or dampened_mse), mean over C, masked mean
            loss_spatial = self._spatial_feature_loss(student_spatial, teacher_spatial, eq_mask, grid_hw=grid_hw)
            loss = self.fd_loss_weight * loss_spatial
            if self.alignment_metrics_enabled:
                with torch.no_grad():
                    self.last_alignment_metrics["spatial_loss"] = loss_spatial.detach().float()
            loss_intermediate = self._intermediate_spatial_feature_loss(
                student_spatial_features,
                teacher_spatial_map,
                student_valid_mask,
                teacher_valid_mask,
                spatial_transform,
            )
            if loss_intermediate is not None:
                loss = loss + self.intermediate_loss_weight * loss_intermediate

            if self.summary_loss_weight != 0.0:
                student_summary = self._select_summary_token(student_summary, self.student_model)
                teacher_summary = self._select_summary_token(teacher_summary, self.teacher_model)
                if self.projection_layer_summary is not None:
                    student_summary = self.projection_layer_summary(student_summary)
                if self.teacher_norm is not None:
                    teacher_summary = self.teacher_norm(teacher_summary)
                if self.alignment_metrics_enabled:
                    with torch.no_grad():
                        self.last_alignment_metrics.update({
                            "summary_cosine": F.cosine_similarity(
                                student_summary.float(), teacher_summary.float(), dim=-1, eps=1e-8
                            ).mean().detach(),
                            "summary_mse": (
                                student_summary.float() - teacher_summary.float()
                            ).pow(2).mean().detach(),
                            "summary_student_norm": student_summary.float().norm(dim=-1).mean().detach(),
                            "summary_teacher_norm": teacher_summary.float().norm(dim=-1).mean().detach(),
                        })

                # summary loss (CE with temperature, or native angle/cosine/tangent_sphere)
                if self.summary_criterion is not None:
                    loss_summary = self.summary_criterion(student_summary, teacher_summary)
                else:
                    teacher_probs = F.softmax(teacher_summary / self.temperature, dim=-1)
                    loss_summary = self.criterions["CE"](student_summary / self.temperature, teacher_probs)
                loss = loss + self.summary_loss_weight * loss_summary
                if self.alignment_metrics_enabled:
                    with torch.no_grad():
                        self.last_alignment_metrics["summary_loss"] = loss_summary.detach().float()
            if self.alignment_metrics_enabled:
                with torch.no_grad():
                    self.last_alignment_metrics["combo_loss"] = loss.detach().float()
            return loss
        else:
            if student_summary is not None:
                student_output = student_summary
            else:
                student_output = self.student_model.forward_pre_logits(batch_input)
            with torch.no_grad():
                teacher_output = self.teacher_model.forward_pre_logits(teacher_input)

        if (
            self.distillation_mode != "logits" and
            self.distillation_mode != "spatial" and
            self.projection_layer is not None
        ):
            student_output = self.projection_layer(student_output)

        if self.distillation_mode == "spatial" and (
            self.spatial_loss_type != "mse" or self.loss_type == "MSE"
        ):
            return self._spatial_feature_loss(student_output, teacher_output, eq_mask, grid_hw=grid_hw)

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
