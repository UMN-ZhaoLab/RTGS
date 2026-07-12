"""
RTGS-style speedups for GS-ICP-SLAM:
- adaptive ICP point-cloud downsample
- mapping train throttle (fewer random views)
- late gradient / opacity-aware Gaussian pruning
"""

from __future__ import annotations

import numpy as np
import torch


def default_rtgs_cfg(enabled: bool = False):
    if not enabled:
        return dict(enable=False)
    return dict(
        enable=True,
        # Adaptive ICP density (higher rate = fewer points)
        enable_adaptive_downsample=True,
        downsample_min_rate=5,
        downsample_max_rate=8,
        adaptive_warm_frames=20,
        motion_trans_low=0.01,
        motion_trans_high=0.05,
        # Mapping: fewer random BA iters between keyframes
        enable_mapping_throttle=True,
        map_random_train_prob=0.35,
        map_new_kf_extra_iters=2,
        # Late pruning of low-contribution Gaussians
        enable_gradient_pruning=True,
        pruning_start_progress=0.70,
        gradient_prune_every=150,
        target_reduction_ratio=0.10,
        max_prune_fraction_per_step=0.04,
        gradient_keep_percentile=0.55,
        min_opacity_protect=0.80,
        min_gaussians_to_prune=8000,
        # Slightly sparser mapping keyframes
        keyframe_freq_scale=1.5,
    )


def select_downsample_rate(frame_idx, prev_pose, curr_pose, base_rate, rtgs):
    """Return integer downsample rate (>= base) for ICP point cloud."""
    if not rtgs.get("enable") or not rtgs.get("enable_adaptive_downsample", False):
        return int(base_rate)

    min_r = int(rtgs.get("downsample_min_rate", base_rate))
    max_r = int(rtgs.get("downsample_max_rate", max(base_rate, min_r + 2)))
    min_r = max(min_r, int(base_rate))
    if min_r >= max_r:
        return min_r

    if frame_idx < int(rtgs.get("adaptive_warm_frames", 20)):
        return min_r

    if prev_pose is None or curr_pose is None:
        return min_r

    trans = float(np.linalg.norm(curr_pose[:3, 3] - prev_pose[:3, 3]))
    low = float(rtgs.get("motion_trans_low", 0.01))
    high = float(rtgs.get("motion_trans_high", 0.05))
    t = (trans - low) / max(high - low, 1e-6)
    t = max(0.0, min(1.0, t))
    # High motion -> denser (lower rate); low motion -> sparser
    rate = max_r - (max_r - min_r) * t
    return int(round(rate))


def should_run_random_mapping(rtgs, is_new_keyframe):
    """Decide whether to run a mapping train step on a random keyframe."""
    if not rtgs.get("enable") or not rtgs.get("enable_mapping_throttle", False):
        return True
    if is_new_keyframe:
        return True
    return np.random.rand() < float(rtgs.get("map_random_train_prob", 0.35))


def prune_low_gradient_gaussians(gaussians, frame_progress, train_iter, rtgs):
    """
    Remove Gaussians with weakest xyz gradient signal, protecting high opacity.
    Returns number pruned.
    """
    if not rtgs.get("enable") or not rtgs.get("enable_gradient_pruning", False):
        return 0

    start_p = float(rtgs.get("pruning_start_progress", 0.70))
    if frame_progress < start_p:
        return 0

    every = int(rtgs.get("gradient_prune_every", 150))
    if every <= 0 or (train_iter % every) != 0:
        return 0

    n = gaussians.get_xyz.shape[0]
    if n < int(rtgs.get("min_gaussians_to_prune", 8000)):
        return 0

    denom = gaussians.denom.float().clamp_min(1.0)
    grad_signal = (gaussians.xyz_gradient_accum.float() / denom).squeeze()
    if grad_signal.ndim == 0 or grad_signal.numel() != n:
        return 0

    opacity = gaussians.get_opacity.squeeze()
    active = (frame_progress - start_p) / max(1.0 - start_p, 1e-6)
    active = max(0.0, min(1.0, active))
    max_frac = float(rtgs.get("max_prune_fraction_per_step", 0.04))
    target_ratio = float(rtgs.get("target_reduction_ratio", 0.10))
    budget = max(1, int(n * max_frac * max(active, 0.25)))
    to_remove_n = min(budget, max(0, int(n * target_ratio * max(active, 0.25))))
    if to_remove_n <= 0:
        return 0

    keep_pct = float(rtgs.get("gradient_keep_percentile", 0.55))
    min_opacity_protect = float(rtgs.get("min_opacity_protect", 0.80))

    eligible = denom.squeeze() >= 1.0
    n_eligible = int(eligible.sum().item())
    if n_eligible <= to_remove_n:
        to_remove_n = max(0, n_eligible - 1)
    if to_remove_n <= 0:
        return 0

    protected = ~eligible
    if eligible.any():
        grad_thresh = torch.quantile(grad_signal[eligible], keep_pct)
        protected = protected | (grad_signal >= grad_thresh)
    protected = protected | (opacity >= min_opacity_protect)

    candidates = ~protected
    n_cand = int(candidates.sum().item())
    if n_cand == 0:
        return 0

    to_remove_n = min(to_remove_n, n_cand)
    score = grad_signal.clone()
    score[protected] = float("inf")
    _, idx = torch.topk(score, to_remove_n, largest=False)
    mask = torch.zeros(n, dtype=torch.bool, device=score.device)
    mask[idx] = True
    gaussians.prune_points(mask)
    torch.cuda.empty_cache()
    return to_remove_n
