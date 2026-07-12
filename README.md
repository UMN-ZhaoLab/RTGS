# GS-ICP-SLAM + RTGS

RTGS speedups on [GS-ICP-SLAM](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM) (ECCV 2024): adaptive ICP downsample, mapping train throttle, late gradient pruning.

## Setup

```bash
source /mnt/hdd/lls/miniconda3/etc/profile.d/conda.sh
conda activate gsicpslam
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$(pwd)/submodules/fast_gicp/build:$LD_LIBRARY_PATH
```

Default TUM scene: `/mnt/hdd/datasets/rgbd_dataset_freiburg1_desk`  
Override with `GSICP_TUM_SCENE=/path/to/rgbd_dataset_freiburg1_desk`.

## Reproduce

```bash
bash run_full_fr1_compare.sh
```

Or:

```bash
# Baseline (unlimited FPS)
python -W ignore gs_icp_slam_unlimit.py \
  --dataset_path /mnt/hdd/datasets/rgbd_dataset_freiburg1_desk \
  --config configs/TUM/rgbd_dataset_freiburg1_desk.txt \
  --output_path experiments/baseline \
  --keyframe_th 0.81 --downsample_rate 5 --save_results

# RTGS
python -W ignore gs_icp_slam_unlimit.py \
  --dataset_path /mnt/hdd/datasets/rgbd_dataset_freiburg1_desk \
  --config configs/TUM/rgbd_dataset_freiburg1_desk.txt \
  --output_path experiments/rtgs \
  --keyframe_th 0.81 --downsample_rate 5 --save_results --rtgs
```

## Reference results (freiburg1_desk, ~592 frames, unlimited)

| Method | FPS | Wall | PSNR | ATE |
|--------|-----|------|------|-----|
| Baseline | 75.62 | 35.7 s | 16.96 dB | 2.67 cm |
| RTGS | 150.77 | 30.8 s | 16.51 dB | 2.55 cm |

FPS **×1.99** · PSNR **−0.45 dB** · ATE **−0.12 cm**
