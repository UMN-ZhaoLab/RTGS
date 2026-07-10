#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${RTGS_VENV:-${ROOT}/.venv}"
OUT_DIR="$ROOT/comparison_results/full_scenes"
mkdir -p "$OUT_DIR"

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Virtualenv not found at: $VENV"
  echo "Set RTGS_VENV to your venv path, e.g.:"
  echo "  export RTGS_VENV=/path/to/rtgs_venv"
  exit 1
fi

source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT"

SCENES=(
  "fr1_desk:configs/rgbd/tum/fr1_desk.yaml"
  "fr2_xyz:configs/rgbd/tum/fr2_xyz.yaml"
  "fr3_office:configs/rgbd/tum/fr3_office.yaml"
)

is_done() {
  local scene="$1"
  local version="$2"
  python3 - <<PY "$ROOT" "$scene" "$version"
import json, sys
from pathlib import Path
root, scene, version = sys.argv[1:4]
tree = Path(root) / ("Baseline" if version == "baseline" else "MonoRTGS")
files = sorted(tree.glob("results/**/slam_results.json"))
if not files:
    raise SystemExit(1)
data = json.loads(files[-1].read_text())
metrics = data.get("frame_metrics") or []
if not metrics or "num_gaussians" not in metrics[0]:
    raise SystemExit(1)
dataset = str(data.get("dataset", "")).lower()
scene_key = scene.replace("_", "")
if scene_key not in dataset and scene not in dataset:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

run_scene() {
  local scene="$1"
  local config="$2"
  local version="$3"
  local log="$OUT_DIR/${scene}_${version}.log"
  local time_log="$OUT_DIR/${scene}_${version}_time.txt"

  if is_done "$scene" "$version"; then
    echo "===== [$scene] $version (skip, already done) ====="
    return 0
  fi

  echo "===== [$scene] $version ====="
  cd "$ROOT"
  /usr/bin/time -v bash -c "./run_slam.sh $version $config" > "$log" 2> "$time_log"
}

for entry in "${SCENES[@]}"; do
  scene="${entry%%:*}"
  config="${entry##*:}"
  run_scene "$scene" "$config" "baseline"
  run_scene "$scene" "$config" "monortgs"
done

python3 "$ROOT/comparison_results/plot_full_comparison.py"
echo "All scenes completed."
