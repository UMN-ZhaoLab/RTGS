# SplaTAM + RTGS

RTGS speedups on [SplaTAM](https://github.com/spla-tam/SplaTAM): adaptive tracking resolution, tracking early-stop, lighter mapping, late gradient pruning.

## Setup

```bash
python3 -m venv /path/to/splatam_venv && source /path/to/splatam_venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "numpy<2" scipy "opencv-python-headless<4.10"
```

Use a **dedicated** env (`diff-gaussian-rasterization-w-depth`).

```bash
export SPLATAM_VENV=/path/to/splatam_venv
export SPLATAM_TUM_BASEDIR=/path/to/parent_of_rgbd_dataset_freiburg1_desk
# default data path: /mnt/hdd/datasets/rgbd_dataset_freiburg1_desk
```

## Reproduce

```bash
bash run_full_fr1_compare.sh
```

Or:

```bash
python scripts/splatam.py configs/tum/splatam_baseline_full.py
python scripts/splatam.py configs/tum/splatam_rtgs_full.py
```

## Reference results (freiburg1_desk, ~592 frames)

| Method | FPS | Wall | PSNR | ATE |
|--------|-----|------|------|-----|
| Baseline | 0.271 | 2186.2 s | 21.83 dB | 3.39 cm |
| RTGS | 0.787 | 752.0 s | 21.35 dB | 3.65 cm |

FPS **×2.91** · PSNR **−0.48 dB** · ATE **+0.26 cm**
