#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SPLATAM_VENV:-/mnt/hdd/lls/splatam_venv}"
OUT="$ROOT/experiments/TUM_FULL"
mkdir -p "$OUT"

source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export SPLATAM_TUM_BASEDIR="${SPLATAM_TUM_BASEDIR:-/mnt/hdd/datasets}"

cd "$ROOT"

echo "===== Baseline full fr1/desk ====="
python scripts/splatam.py configs/tum/splatam_baseline_full.py \
  > "$OUT/baseline_full.log" 2>&1

echo "===== RTGS full fr1/desk ====="
python scripts/splatam.py configs/tum/splatam_rtgs_full.py \
  > "$OUT/rtgs_full.log" 2>&1

python3 - <<'PY'
import re
from pathlib import Path
out = Path("experiments/TUM_FULL")
rows = []
for name, log in [("Baseline", "baseline_full.log"), ("RTGS", "rtgs_full.log")]:
    t = (out / log).read_text(errors="ignore")
    fps = re.search(r"Overall SLAM FPS \(track\+map\):\s*([\d.]+)", t)
    wall = re.search(r"Overall SLAM Wall Time:\s*([\d.]+)", t)
    ate = re.search(r"Final Average ATE RMSE:\s*([\d.]+)\s*cm", t)
    psnr = re.search(r"Average PSNR:\s*([\d.]+)", t)
    row = {
        "name": name,
        "fps": float(fps.group(1)) if fps else None,
        "wall": float(wall.group(1)) if wall else None,
        "ate": float(ate.group(1)) if ate else None,
        "psnr": float(psnr.group(1)) if psnr else None,
        "err": "Traceback" in t,
    }
    rows.append(row)
    print(f"{name}: FPS={row['fps']} Wall={row['wall']}s PSNR={row['psnr']} ATE={row['ate']}cm err={row['err']}")

b, r = rows[0], rows[1]
md = f"""# SplaTAM Full Scene — freiburg1_desk

| Method | FPS | Wall | PSNR | ATE |
|--------|-----|------|------|-----|
| Baseline | {b['fps']:.3f} | {b['wall']:.1f} s | {b['psnr']:.2f} dB | {b['ate']:.2f} cm |
| RTGS | {r['fps']:.3f} | {r['wall']:.1f} s | {r['psnr']:.2f} dB | {r['ate']:.2f} cm |

## Delta (RTGS − Baseline)
- FPS: ×{r['fps']/b['fps']:.2f} ({(r['fps']/b['fps']-1)*100:+.1f}%)
- PSNR: {r['psnr']-b['psnr']:+.2f} dB
- ATE: {r['ate']-b['ate']:+.2f} cm
- Wall: {(r['wall']-b['wall'])/b['wall']*100:+.1f}%
"""
(out / "summary.md").write_text(md)
print(md)
PY
