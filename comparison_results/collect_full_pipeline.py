#!/usr/bin/env python3
"""Collect metrics from full MonoGS pipeline runs (--eval mode)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def latest_result_dir(tree: Path, scene_hint: str) -> Path | None:
    candidates = sorted(tree.glob("results/**/*"), key=lambda p: p.stat().st_mtime)
    result_dirs = [p for p in candidates if p.is_dir() and (p / "config.yml").exists()]
    scene_key = scene_hint.replace("_", "").lower()
    for path in reversed(result_dirs):
        cfg = path / "config.yml"
        if scene_key in cfg.read_text().lower():
            return path
    return result_dirs[-1] if result_dirs else None


def load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def parse_log_metrics(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    metrics: dict[str, float] = {}

    m = re.search(r"Total time[^\d]*([\d.]+)", text)
    if m:
        metrics["wall_time_s"] = float(m.group(1))
    m = re.search(r"Total FPS[^\d]*([\d.]+)", text)
    if m:
        metrics["fps"] = float(m.group(1))
    m = re.search(r"Before optimization - PSNR:\s*([\d.]+)", text)
    if m:
        metrics["psnr_before_opt_db"] = float(m.group(1))
    m = re.search(r"After optimization - PSNR:\s*([\d.]+)", text)
    if m:
        metrics["psnr_after_opt_db"] = float(m.group(1))
    m = re.search(r"RMSE ATE.*?\[m\]\s*([\d.]+)", text)
    if m:
        metrics["ate_m"] = float(m.group(1))
        metrics["ate_cm"] = float(m.group(1)) * 100.0
    return metrics


def collect_one(root: Path, version: str, scene: str, log_dir: Path) -> dict:
    tree = root / ("Baseline" if version == "baseline" else "MonoRTGS")
    result_dir = latest_result_dir(tree, scene)
    if result_dir is None:
        raise FileNotFoundError(f"No result directory found under {tree}/results")

    before = load_json(result_dir / "psnr/before_opt/final_result.json")
    after = load_json(result_dir / "psnr/after_opt/final_result.json")
    ate_stats = load_json(result_dir / "plot/stats_final.json")
    log_metrics = parse_log_metrics(log_dir / f"{scene}_{version}.log")

    ply = result_dir / "point_cloud/final/point_cloud.ply"
    gaussian_count = None
    if ply.exists():
        # Rough count from ply header
        for line in ply.read_text(errors="replace").splitlines()[:20]:
            if line.startswith("element vertex"):
                gaussian_count = int(line.split()[-1])
                break

    row = {
        "version": version,
        "result_dir": str(result_dir),
        "psnr_before_opt_db": before.get("mean_psnr") if before else log_metrics.get("psnr_before_opt_db"),
        "ssim_before_opt": before.get("mean_ssim") if before else None,
        "lpips_before_opt": before.get("mean_lpips") if before else None,
        "psnr_after_opt_db": after.get("mean_psnr") if after else log_metrics.get("psnr_after_opt_db"),
        "ssim_after_opt": after.get("mean_ssim") if after else None,
        "lpips_after_opt": after.get("mean_lpips") if after else None,
        "ate_rmse_m": ate_stats.get("rmse") if ate_stats else log_metrics.get("ate_m"),
        "ate_cm": (ate_stats.get("rmse") * 100.0 if ate_stats and ate_stats.get("rmse") is not None else log_metrics.get("ate_cm")),
        "fps": log_metrics.get("fps"),
        "wall_time_s": log_metrics.get("wall_time_s"),
        "gaussian_count": gaussian_count,
    }
    return row


def write_summary(out_dir: Path, scene: str, rows: list[dict]) -> None:
    lines = [
        f"# Full MonoGS Pipeline Comparison — {scene}",
        "",
        "Metrics from multiprocessing SLAM (`single_thread: False`) with `--eval` rendering.",
        "",
        "| Method | PSNR before (dB) | PSNR after (dB) | ATE (cm) | FPS | Wall (s) | Gaussians |",
        "|--------|------------------|-----------------|----------|-----|----------|-----------|",
    ]
    labels = {"baseline": "MonoGS Baseline", "monortgs": "RTGS Soft"}
    for row in rows:
        label = labels.get(row["version"], row["version"])
        lines.append(
            f"| {label} "
            f"| {fmt(row.get('psnr_before_opt_db'))} "
            f"| {fmt(row.get('psnr_after_opt_db'))} "
            f"| {fmt(row.get('ate_cm'))} "
            f"| {fmt(row.get('fps'))} "
            f"| {fmt(row.get('wall_time_s'))} "
            f"| {fmt(row.get('gaussian_count'), nd=0)} |"
        )
    if len(rows) == 2:
        b, r = rows[0], rows[1]
        lines.extend(["", "## Delta (RTGS − Baseline)", ""])
        for key, scale, unit in [
            ("psnr_before_opt_db", 1, "dB"),
            ("psnr_after_opt_db", 1, "dB"),
            ("ate_cm", 1, "cm"),
            ("fps", 1, "FPS"),
            ("wall_time_s", 1, "s"),
            ("gaussian_count", 1, ""),
        ]:
            if b.get(key) is not None and r.get(key) is not None:
                delta = r[key] - b[key]
                pct = (delta / b[key] * 100.0) if b[key] else 0.0
                suffix = f" ({pct:+.1f}%)" if key != "psnr_before_opt_db" else ""
                lines.append(f"- **{key}**: {delta * scale:+.3f}{unit}{suffix}")

    summary_path = out_dir / f"{scene}_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    json_path = out_dir / f"{scene}_results.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")


def fmt(val, nd: int = 3):
    if val is None:
        return "—"
    if nd == 0:
        return f"{int(val):,}"
    return f"{val:.{nd}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="fr1_desk")
    parser.add_argument("--out-dir", default="comparison_results/full_pipeline")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        collect_one(root, "baseline", args.scene, out_dir),
        collect_one(root, "monortgs", args.scene, out_dir),
    ]
    write_summary(out_dir, args.scene, rows)


if __name__ == "__main__":
    main()
