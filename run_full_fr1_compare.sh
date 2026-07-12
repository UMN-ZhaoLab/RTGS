#!/usr/bin/env bash
# Baseline vs RTGS full-sequence compare on TUM freiburg1_desk (unlimited FPS mode)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
source /mnt/hdd/lls/miniconda3/etc/profile.d/conda.sh
conda activate gsicpslam

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ROOT/submodules/fast_gicp/build:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

DATASET="${GSICP_TUM_SCENE:-/mnt/hdd/datasets/rgbd_dataset_freiburg1_desk}"
CONFIG="$ROOT/configs/TUM/rgbd_dataset_freiburg1_desk.txt"
OUT="$ROOT/experiments/fr1_compare"
mkdir -p "$OUT"

COMMON=(
  --dataset_path "$DATASET"
  --config "$CONFIG"
  --keyframe_th 0.81
  --knn_maxd 99999.0
  --overlapped_th 1e-3
  --max_correspondence_distance 0.03
  --trackable_opacity_th 0.09
  --overlapped_th2 1e-3
  --downsample_rate 5
  --save_results
)

echo "===== Baseline full fr1/desk ====="
/usr/bin/time -f 'WALL_SEC %e' -o "$OUT/baseline_time.txt" \
  python -W ignore gs_icp_slam_unlimit.py \
    "${COMMON[@]}" \
    --output_path "$OUT/baseline" \
    > "$OUT/baseline_full.log" 2>&1

echo "===== RTGS full fr1/desk ====="
/usr/bin/time -f 'WALL_SEC %e' -o "$OUT/rtgs_time.txt" \
  python -W ignore gs_icp_slam_unlimit.py \
    "${COMMON[@]}" \
    --rtgs \
    --output_path "$OUT/rtgs" \
    > "$OUT/rtgs_full.log" 2>&1

python - <<'PY'
import re
from pathlib import Path

OUT = Path("/home/lls/workspace/MICRO/GS_ICP_SLAM/experiments/fr1_compare")

def parse(name, log_name, time_name):
    text = (OUT / log_name).read_text(errors="ignore")
    wall = (OUT / time_name).read_text(errors="ignore")
    fps = re.search(r"System FPS:\s*([0-9.]+)", text)
    ate = re.search(r"ATE RMSE:\s*([0-9.]+)", text)
    psnr = re.search(r"PSNR:\s*([0-9.]+)", text)
    w = re.search(r"WALL_SEC\s*([0-9.]+)", wall)
    return {
        "name": name,
        "fps": float(fps.group(1)) if fps else float("nan"),
        "ate": float(ate.group(1)) if ate else float("nan"),
        "psnr": float(psnr.group(1)) if psnr else float("nan"),
        "wall": float(w.group(1)) if w else float("nan"),
    }

rows = [
    parse("Baseline", "baseline_full.log", "baseline_time.txt"),
    parse("RTGS", "rtgs_full.log", "rtgs_time.txt"),
]
md = [
    "# GS-ICP-SLAM fr1/desk compare (unlimited)",
    "",
    "| Method | FPS | Wall | PSNR | ATE |",
    "|--------|-----|------|------|-----|",
]
for r in rows:
    md.append(
        f"| {r['name']} | {r['fps']:.3f} | {r['wall']:.1f} s | {r['psnr']:.2f} dB | {r['ate']:.2f} cm |"
    )
b, t = rows[0], rows[1]
md += [
    "",
    "## Delta (RTGS − Baseline)",
    "",
    f"- FPS: {t['fps'] - b['fps']:+.3f} (x{t['fps']/b['fps']:.2f})" if b['fps'] > 0 else "- FPS: n/a",
    f"- Wall: {t['wall'] - b['wall']:+.1f} s",
    f"- PSNR: {t['psnr'] - b['psnr']:+.2f} dB",
    f"- ATE: {t['ate'] - b['ate']:+.2f} cm",
]
(OUT / "compare.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
PY
