# RTGS: Full MonoGS Pipeline (MonoGS_fullend)

Full tracking + mapping MonoGS pipeline with **RTGS** speedups (adaptive tracking resolution, mapping iter scaling, gradient pruning, early stop).

Repository: [UMN-ZhaoLab/RTGS](https://github.com/UMN-ZhaoLab/RTGS)  
Branch: `MonoGS_fullend`

---

## Quick Start

```bash
git clone https://github.com/UMN-ZhaoLab/RTGS.git
cd RTGS
git checkout MonoGS_fullend

# Environment
python3 -m venv /path/to/rtgs_venv
source /path/to/rtgs_venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" opencv-python matplotlib scipy munch trimesh evo==1.11.0 \
  open3d==0.17.0 torchmetrics imgviz PyOpenGL glfw PyGLM lpips rich plyfile \
  tqdm pyyaml wandb psutil wheel ninja

cd Baseline
pip install --no-build-isolation submodules/simple-knn submodules/diff-gaussian-rasterization
cd ../MonoRTGS
pip install --no-build-isolation submodules/simple-knn submodules/diff-gaussian-rasterization
cd ..

chmod +x run_slam.sh
```

Prepare TUM `fr1_desk` (or edit `dataset_path` in the YAML):

```bash
mkdir -p /mnt/hdd/datasets && cd /mnt/hdd/datasets
wget -O fr1_desk.tgz https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar -xzf fr1_desk.tgz
```

---

## How to Run

```bash
source /path/to/rtgs_venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)

# MonoGS Baseline (full pipeline)
./run_slam.sh baseline configs/rgbd/tum/fr1_desk_full.yaml

# RTGS (full pipeline, ~2× FPS)
./run_slam.sh monortgs configs/rgbd/tum/fr1_desk_2x_target.yaml
```

Results are written under `Baseline/results/` and `MonoRTGS/results/`. Logs include `Total FPS`, `mean psnr`, and `RMSE ATE`.

---

## Example Results

fr1_desk, 100 frames, full MonoGS pipeline (tracking + mapping + eval rendering). Fair comparison: both disable keyframe sleep throttle.

| Method | FPS | Wall | PSNR | ATE |
|--------|-----|------|------|-----|
| Baseline | 0.898 | 111.4 s | 20.04 dB | 3.17 cm |
| RTGS | 1.798 | 55.6 s | 19.96 dB | 3.41 cm |

- FPS: **×2.00**
- PSNR: **−0.07 dB**
- ATE: **+0.24 cm**

---

## What RTGS Changes (full pipeline)

| Technique | Role |
|-----------|------|
| Adaptive tracking downsample | Lower tracking render res when motion / load allow |
| Tracking early-stop | Stop pose opt after min iters + convergence |
| Mapping iter scaling (÷~2.1) | Fewer mapping iters; init BA protected |
| Late gradient pruning | Drop low-gradient stale Gaussians |
| Skip full-res finalize | Full-res render only when creating a keyframe |
| `kf_interval: 6`, idle mapping skip | Fewer / lighter mapping calls |

Config entry: `MonoRTGS/configs/rgbd/tum/fr1_desk_2x_target.yaml`

---

## Related

- [MonoGS](https://github.com/muskie82/MonoGS.git)
- Main branch README covers the lighter demo / multi-scene Soft benchmark
