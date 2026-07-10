#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "comparison_results" / "full_scenes"
PLOT_DIR = OUT_DIR / "plots"

SCENES = {
    "fr1_desk": "rgbd_dataset_freiburg1_desk",
    "fr2_xyz": "rgbd_dataset_freiburg2_xyz",
    "fr3_office": "rgbd_dataset_freiburg3_long_office_household",
}


def find_result(tree: Path, dataset_name: str) -> Optional[Path]:
    matches = []
    for path in tree.glob("results/**/slam_results.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("dataset") == dataset_name:
            matches.append((path.stat().st_mtime, path))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def parse_wall_seconds(time_file: Path) -> Optional[float]:
    if not time_file.exists():
        return None
    text = time_file.read_text()
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\n]+)", text)
    if not m:
        return None
    token = m.group(1).strip()
    if ":" in token:
        parts = token.split(":")
        if len(parts) == 3:
            h, m_, s = parts
            return int(h) * 3600 + int(m_) * 60 + float(s)
        m_, s = parts
        return int(m_) * 60 + float(s)
    return float(token)


def parse_rss_gb(time_file: Path) -> Optional[float]:
    if not time_file.exists():
        return None
    text = time_file.read_text()
    peak = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    return int(peak.group(1)) / (1024 * 1024) if peak else None


def load_run(scene: str, version: str) -> Dict:
    tree = ROOT / ("Baseline" if version == "baseline" else "MonoRTGS")
    dataset_name = SCENES[scene]
    result_path = find_result(tree, dataset_name)
    time_path = OUT_DIR / f"{scene}_{version}_time.txt"

    data = {"result_path": str(result_path) if result_path else None}
    if result_path and result_path.exists():
        payload = json.loads(result_path.read_text())
        data.update(
            {
                "dataset": payload.get("dataset"),
                "average_psnr_db": payload.get("average_psnr_db"),
                "average_ate_cm": payload.get("average_ate_cm"),
                "average_fps": payload.get("average_fps"),
                "total_time_seconds": payload.get("total_time_seconds"),
                "peak_memory_gb": payload.get("peak_memory_gb"),
                "frame_metrics": payload.get("frame_metrics", []),
            }
        )

    rss = parse_rss_gb(time_path)
    wall = parse_wall_seconds(time_path)
    if rss is not None:
        data["rss_peak_gb"] = rss
    if wall is not None:
        data["wall_time_seconds"] = wall
    if not data.get("peak_memory_gb"):
        data["peak_memory_gb"] = rss
    return data


def gaussian_curve(frame_metrics: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    frames, counts = [], []
    for item in frame_metrics:
        if "num_gaussians" in item:
            frames.append(item["frame"])
            counts.append(item["num_gaussians"])
    return np.array(frames), np.array(counts)


def plot_gaussian_curves(results: Dict[str, Dict[str, Dict]]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (scene_key, dataset_name) in zip(axes, SCENES.items()):
        baseline = results[scene_key]["baseline"]
        monortgs = results[scene_key]["monortgs"]
        b_frames, b_counts = gaussian_curve(baseline.get("frame_metrics", []))
        m_frames, m_counts = gaussian_curve(monortgs.get("frame_metrics", []))

        if len(b_frames):
            ax.plot(
                b_frames,
                b_counts,
                label="MonoGS Baseline",
                color="#1f77b4",
                linewidth=2,
            )
        if len(m_frames):
            ax.plot(
                m_frames,
                m_counts,
                label="RTGS Soft",
                color="#d62728",
                linewidth=2,
                linestyle="--",
                alpha=0.9,
            )

        ax.set_title(scene_key)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Gaussian Count")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Gaussian Number Over Time: Baseline vs RTGS Soft", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gaussian_count_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    colors = {"baseline": "#1f77b4", "monortgs": "#d62728"}
    for ax, (scene_key, _) in zip(axes, SCENES.items()):
        for version, label in [("baseline", "Baseline"), ("monortgs", "RTGS Soft")]:
            frames, counts = gaussian_curve(
                results[scene_key][version].get("frame_metrics", [])
            )
            if len(frames):
                ax.plot(
                    frames,
                    counts,
                    label=label,
                    color=colors[version],
                    linewidth=1.8,
                    linestyle="-" if version == "baseline" else "--",
                )
        ax.set_title(scene_key)
        ax.set_ylabel("Gaussian Count")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Frame")
    fig.suptitle("Gaussian Number by Scene", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gaussian_count_by_scene.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Ratio plot: RTGS / Baseline gaussian count
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (scene_key, _) in zip(axes, SCENES.items()):
        b_frames, b_counts = gaussian_curve(results[scene_key]["baseline"].get("frame_metrics", []))
        m_frames, m_counts = gaussian_curve(results[scene_key]["monortgs"].get("frame_metrics", []))
        if len(b_frames) and len(m_frames):
            n = min(len(b_counts), len(m_counts))
            ratio = m_counts[:n] / np.maximum(b_counts[:n], 1)
            ax.plot(b_frames[:n], ratio, color="#d62728", linewidth=2)
            ax.axhline(0.5, color="gray", linestyle=":", label="50% target")
        ax.set_title(f"{scene_key} RTGS/Baseline ratio")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Gaussian Ratio")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Gaussian Count Ratio (RTGS / Baseline)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gaussian_ratio_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary_table(results: Dict[str, Dict[str, Dict]]) -> Dict:
    summary = {"scenes": {}, "average": {}}
    metric_keys = ["average_psnr_db", "average_ate_cm", "average_fps", "peak_memory_gb"]
    out_keys = ["psnr_db", "ate_cm", "fps", "memory_gb"]

    for scene_key in SCENES:
        summary["scenes"][scene_key] = {}
        for version in ("baseline", "monortgs"):
            row = results[scene_key][version]
            summary["scenes"][scene_key][version] = {
                "psnr_db": row.get("average_psnr_db"),
                "ate_cm": row.get("average_ate_cm"),
                "fps": row.get("average_fps"),
                "memory_gb": row.get("peak_memory_gb"),
                "wall_time_s": row.get("wall_time_seconds") or row.get("total_time_seconds"),
                "result_path": row.get("result_path"),
            }

    for metric_key, out_key in zip(metric_keys, out_keys):
        b_vals = [
            summary["scenes"][s]["baseline"].get(out_key)
            for s in SCENES
            if isinstance(summary["scenes"][s]["baseline"].get(out_key), (int, float))
        ]
        m_vals = [
            summary["scenes"][s]["monortgs"].get(out_key)
            for s in SCENES
            if isinstance(summary["scenes"][s]["monortgs"].get(out_key), (int, float))
        ]
        summary["average"][out_key] = {
            "baseline": float(np.mean(b_vals)) if b_vals else None,
            "monortgs": float(np.mean(m_vals)) if m_vals else None,
        }

    return summary


def save_markdown_summary(summary: Dict) -> None:
    lines = [
        "# RTGS Full Scene Comparison",
        "",
        "## Per-Scene Metrics",
        "",
        "| Scene | Method | PSNR (dB) | ATE (cm) | FPS | Memory (GB) | Wall Time (s) |",
        "|-------|--------|-----------|----------|-----|-------------|---------------|",
    ]

    for scene_key in SCENES:
        for version, label in [("baseline", "MonoGS Baseline"), ("monortgs", "RTGS Soft")]:
            row = summary["scenes"][scene_key][version]
            lines.append(
                f"| {scene_key} | {label} | "
                f"{row.get('psnr_db', 0):.3f} | "
                f"{row.get('ate_cm', 0):.3f} | "
                f"{row.get('fps', 0):.3f} | "
                f"{row.get('memory_gb', 0):.3f} | "
                f"{row.get('wall_time_s', 0):.1f} |"
            )

    lines.extend(
        [
            "",
            "## Average Across Scenes",
            "",
            "| Metric | MonoGS Baseline | RTGS Soft | Delta |",
            "|--------|-----------------|-----------|-------|",
        ]
    )

    for key, name in [
        ("psnr_db", "PSNR (dB)"),
        ("ate_cm", "ATE (cm)"),
        ("fps", "FPS"),
        ("memory_gb", "Memory (GB)"),
    ]:
        b = summary["average"][key]["baseline"]
        m = summary["average"][key]["monortgs"]
        delta = m - b if isinstance(b, (int, float)) and isinstance(m, (int, float)) else None
        lines.append(f"| {name} | {b:.3f} | {m:.3f} | {delta:+.3f} |")

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        scene: {
            "baseline": load_run(scene, "baseline"),
            "monortgs": load_run(scene, "monortgs"),
        }
        for scene in SCENES
    }

    plot_gaussian_curves(results)
    summary = build_summary_table(results)
    report = {"results": results, "summary": summary}
    (OUT_DIR / "full_comparison_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    save_markdown_summary(summary)

    print("=" * 80)
    print("Full Scene Comparison Summary")
    print("=" * 80)
    for scene_key in SCENES:
        print(f"\n[{scene_key}]")
        for version, label in [("baseline", "Baseline"), ("monortgs", "RTGS Soft")]:
            row = summary["scenes"][scene_key][version]
            print(
                f"  {label:12s}  PSNR={row.get('psnr_db', 0):.2f} dB  "
                f"ATE={row.get('ate_cm', 0):.2f} cm  FPS={row.get('fps', 0):.2f}  "
                f"Mem={row.get('memory_gb', 0):.2f} GB  "
                f"Time={row.get('wall_time_s', 0):.0f}s"
            )

    print("\n[Averages]")
    for key, name in [("psnr_db", "PSNR"), ("ate_cm", "ATE"), ("fps", "FPS"), ("memory_gb", "Memory")]:
        b = summary["average"][key]["baseline"]
        m = summary["average"][key]["monortgs"]
        print(f"  {name:8s}  Baseline={b:.3f}  RTGS={m:.3f}  Delta={m-b:+.3f}")

    print(f"\nPlots saved to: {PLOT_DIR}")
    print(f"Summary saved to: {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
