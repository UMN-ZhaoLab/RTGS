"""
RTGS-style speedups for SplaTAM:
- tracking early-stop on pose delta
- adaptive tracking resolution helper
- late gradient-based Gaussian pruning
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.slam_external import remove_points, accumulate_mean2d_gradient


def pose_delta_norm(params, time_idx, prev_rot, prev_tran):
    """L2 norm of camera rotation (quat) + translation change since last check."""
    curr_rot = params["cam_unnorm_rots"][..., time_idx].detach()
    curr_tran = params["cam_trans"][..., time_idx].detach()
    d_rot = torch.norm(curr_rot - prev_rot).item()
    d_tran = torch.norm(curr_tran - prev_tran).item()
    return d_rot + d_tran, curr_rot.clone(), curr_tran.clone()


def select_tracking_scale(
    time_idx,
    params,
    num_gaussians,
    config,
):
    """
    Return tracking resolution scale in [min_scale, max_scale].
    Higher motion / keyframe proximity -> higher resolution.
    """
    rtgs = config.get("rtgs", {})
    if not rtgs.get("enable_adaptive_tracking_resolution", False):
        return 1.0

    min_scale = float(rtgs.get("tracking_resolution_min_scale", 0.6))
    max_scale = float(rtgs.get("tracking_resolution_max_scale", 1.0))
    if min_scale >= max_scale:
        return max_scale

    # Early frames: keep high quality
    if time_idx < int(rtgs.get("adaptive_warm_frames", 10)):
        return max_scale

    # Motion from previous pose estimate
    motion_need = 0.0
    if time_idx > 0:
        prev_tran = params["cam_trans"][..., time_idx - 1].detach()
        curr_tran = params["cam_trans"][..., time_idx].detach()
        prev_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx - 1].detach())
        curr_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
        trans_motion = torch.norm(curr_tran - prev_tran).item()
        # quaternion chordal distance proxy
        rot_motion = torch.norm(curr_rot - prev_rot).item()
        trans_low = float(rtgs.get("motion_trans_low", 0.008))
        trans_high = float(rtgs.get("motion_trans_high", 0.04))
        rot_low = float(rtgs.get("motion_rot_low", 0.01))
        rot_high = float(rtgs.get("motion_rot_high", 0.05))
        trans_t = (trans_motion - trans_low) / max(trans_high - trans_low, 1e-6)
        rot_t = (rot_motion - rot_low) / max(rot_high - rot_low, 1e-6)
        motion_need = max(0.0, min(1.0, max(trans_t, rot_t)))

    keyframe_every = int(config.get("keyframe_every", 5))
    since_kf = time_idx % keyframe_every
    kf_need = 1.0 if since_kf >= keyframe_every - 1 else 0.0

    gauss_ref = float(rtgs.get("tracking_gaussian_ref", 40000))
    gauss_pressure = min(
        1.0, max(0.0, (num_gaussians - gauss_ref * 0.75) / max(gauss_ref * 0.45, 1.0))
    )

    quality_need = max(motion_need, kf_need)
    scale = min_scale + (max_scale - min_scale) * quality_need
    scale -= (max_scale - min_scale) * 0.35 * gauss_pressure * (1.0 - 0.5 * quality_need)
    return max(min_scale, min(max_scale, scale))


def downsample_frame_data(curr_data, scale: float):
    """Create a resolution-scaled copy of curr_data for tracking."""
    if scale >= 0.999:
        return curr_data

    color = curr_data["im"]
    depth = curr_data["depth"]
    _, H, W = color.shape
    new_h = max(2, int(round(H * scale)))
    new_w = max(2, int(round(W * scale)))
    new_h -= new_h % 2
    new_w -= new_w % 2
    s_h = new_h / H
    s_w = new_w / W

    color_ds = F.interpolate(
        color.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    depth_ds = F.interpolate(
        depth.unsqueeze(0), size=(new_h, new_w), mode="nearest"
    ).squeeze(0)

    intrinsics = curr_data["intrinsics"].clone()
    intrinsics[0, 0] *= s_w
    intrinsics[1, 1] *= s_h
    intrinsics[0, 2] *= s_w
    intrinsics[1, 2] *= s_h

    from utils.recon_helpers import setup_camera

    w2c = curr_data["w2c"]
    cam = setup_camera(new_w, new_h, intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    out = dict(curr_data)
    out.update(
        {
            "cam": cam,
            "im": color_ds,
            "depth": depth_ds,
            "intrinsics": intrinsics,
        }
    )
    return out


def prune_low_gradient_gaussians(
    params,
    variables,
    optimizer,
    time_idx,
    num_frames,
    rtgs_cfg,
):
    """
    After mapping, remove Gaussians with lowest mapping view-space gradient
    signal (xyz/means2D accum / denom), protecting high-opacity / high-grad points.
    """
    if not rtgs_cfg.get("enable_gradient_pruning", False):
        return params, variables, 0

    progress = time_idx / max(num_frames - 1, 1)
    start_progress = float(rtgs_cfg.get("pruning_start_progress", 0.72))
    if progress < start_progress:
        return params, variables, 0

    if (time_idx + 1) % int(rtgs_cfg.get("gradient_prune_every", 5)) != 0:
        return params, variables, 0

    n = params["means3D"].shape[0]
    if n < int(rtgs_cfg.get("min_gaussians_to_prune", 5000)):
        return params, variables, 0

    denom = variables["denom"].float().clamp_min(1.0)
    grad_signal = (variables["means2D_gradient_accum"].float() / denom).squeeze()
    if grad_signal.ndim == 0:
        return params, variables, 0

    opacity = torch.sigmoid(params["logit_opacities"]).squeeze()
    target_ratio = float(rtgs_cfg.get("target_reduction_ratio", 0.12))
    max_frac = float(rtgs_cfg.get("max_prune_fraction_per_step", 0.04))
    active = (progress - start_progress) / max(1.0 - start_progress, 1e-6)
    active = max(0.0, min(1.0, active))
    budget = max(1, int(n * max_frac * max(active, 0.25)))
    to_remove_n = min(budget, max(0, int(n * target_ratio * max(active, 0.25))))
    if to_remove_n <= 0:
        return params, variables, 0

    keep_pct = float(rtgs_cfg.get("gradient_keep_percentile", 0.58))
    min_opacity_protect = float(rtgs_cfg.get("min_opacity_protect", 0.85))

    eligible = denom.squeeze() >= float(rtgs_cfg.get("min_mapping_observations", 1))
    # Prefer pruning older Gaussians if timestep is available
    if "timestep" in variables:
        age = variables["timestep"].float().squeeze()
        age_thresh = float(time_idx) - float(rtgs_cfg.get("protect_recent_frames", 8))
        eligible = eligible & (age < age_thresh)

    n_eligible = int(eligible.sum().item())
    if n_eligible <= to_remove_n:
        to_remove_n = max(0, n_eligible - 1)
    if to_remove_n <= 0:
        return params, variables, 0

    protected = ~eligible
    if eligible.any():
        grad_thresh = torch.quantile(grad_signal[eligible], keep_pct)
        protected = protected | (grad_signal >= grad_thresh)
    protected = protected | (opacity >= min_opacity_protect)

    candidates = ~protected
    n_cand = int(candidates.sum().item())
    if n_cand == 0:
        return params, variables, 0

    to_remove_n = min(to_remove_n, n_cand)
    score = grad_signal.clone()
    score[protected] = float("inf")
    _, idx = torch.topk(score, to_remove_n, largest=False)
    mask = torch.zeros(n, dtype=torch.bool, device=score.device)
    mask[idx] = True
    params, variables = remove_points(mask, params, variables, optimizer)
    torch.cuda.empty_cache()
    return params, variables, to_remove_n


def maybe_accumulate_mapping_grads(variables, rtgs_cfg):
    if rtgs_cfg.get("enable_gradient_pruning", False):
        if variables.get("means2D") is not None and variables["means2D"].grad is not None:
            variables = accumulate_mean2d_gradient(variables)
    return variables
