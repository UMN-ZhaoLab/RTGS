import os

# Full-sequence Baseline SplaTAM on TUM freiburg1_desk
primary_device = "cuda:0"
seed = 0
scene_name = "freiburg1_desk"

map_every = 1
keyframe_every = 5
mapping_window_size = 20
tracking_iters = 200
mapping_iters = 30
scene_radius_depth_ratio = 2

_TUM_BASE = os.environ.get("SPLATAM_TUM_BASEDIR", "/mnt/hdd/datasets")
group_name = "TUM_FULL"
run_name = f"{scene_name}_baseline_full_seed{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    mapping_window_size=mapping_window_size,
    report_global_progress_every=500,
    eval_every=500,
    scene_radius_depth_ratio=scene_radius_depth_ratio,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=100,
    use_wandb=False,
    wandb=dict(
        entity="local", project="SplaTAM", group=group_name, name=run_name,
        save_qual=False, eval_save_qual=False,
    ),
    data=dict(
        basedir=_TUM_BASE,
        gradslam_data_cfg=f"./configs/data/TUM/{scene_name}.yaml",
        sequence=f"rgbd_dataset_{scene_name}",
        desired_image_height=480,
        desired_image_width=640,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
    ),
    tracking=dict(
        use_gt_poses=False,
        forward_prop=True,
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0, rgb_colors=0.0, unnorm_rotations=0.0,
            logit_opacities=0.0, log_scales=0.0,
            cam_unnorm_rots=0.002, cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(im=0.5, depth=1.0),
        lrs=dict(
            means3D=0.0001, rgb_colors=0.0025, unnorm_rotations=0.001,
            logit_opacities=0.05, log_scales=0.001,
            cam_unnorm_rots=0.0000, cam_trans=0.0000,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0, remove_big_after=0, stop_after=20, prune_every=20,
            removal_opacity_threshold=0.005, final_removal_opacity_threshold=0.005,
            reset_opacities=False, reset_opacities_every=500,
        ),
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500, remove_big_after=3000, stop_after=5000, densify_every=100,
            grad_thresh=0.0002, num_to_split_into=2,
            removal_opacity_threshold=0.005, final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),
    rtgs=dict(),
    viz=dict(
        render_mode='color', offset_first_viz_cam=True, show_sil=False,
        visualize_cams=True, viz_w=600, viz_h=340, viz_near=0.01, viz_far=100.0,
        view_scale=2, viz_fps=5, enter_interactive_post_online=False,
    ),
)
