#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${RTGS_VENV:-${ROOT}/.venv}"
OUT_DIR="$ROOT/comparison_results/full_pipeline"
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
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

SCENE="${1:-fr1_desk}"
case "$SCENE" in
  fr1_desk)
    CONFIG="configs/rgbd/tum/fr1_desk_full.yaml"
    ;;
  *)
    echo "Unknown scene: $SCENE (currently only fr1_desk is configured)"
    exit 1
    ;;
esac

run_one() {
  local version="$1"
  local log="$OUT_DIR/${SCENE}_${version}.log"
  local time_log="$OUT_DIR/${SCENE}_${version}_time.txt"

  echo "===== [$SCENE] full MonoGS pipeline: $version ====="
  cd "$ROOT"
  /usr/bin/time -v bash -c "./run_slam.sh $version $CONFIG" > "$log" 2> "$time_log"
}

run_one baseline
run_one monortgs

python3 "$ROOT/comparison_results/collect_full_pipeline.py" --scene "$SCENE" --out-dir "$OUT_DIR"
echo "Full pipeline comparison completed for $SCENE."
