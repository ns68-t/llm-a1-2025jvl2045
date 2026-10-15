"""Create the two A1 submission figures from completed result files."""

import argparse
import json
from pathlib import Path

import pandas as pd
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Allows result packaging in minimal environments.
    plt = None


def pillow_part_a(frame, values, params, memory, output_path):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (1600, 720), "white")
    draw = ImageDraw.Draw(image)
    panels = [(95, 80, 760, 610, "Trainable parameters (%) - log scale"),
              (910, 80, 1510, 610, "Peak GPU memory (GB)")]
    for left, top, right, bottom, label in panels:
        draw.rectangle((left, top, right, bottom), outline="#222", width=2)
        draw.text((left, bottom + 22), label, fill="#222")
        draw.text((left - 55, top - 28), "Validation exact micro F1", fill="#222")
        for tick in (0, 0.5, 1.0):
            y = bottom - int((bottom - top) * tick)
            draw.line((left, y, right, y), fill="#dddddd")
            draw.text((left - 35, y - 8), str(tick), fill="#555")
    valid = [(str(run), float(x), float(y)) for run, x, y in zip(frame["run_id"], params, values)
             if pd.notna(x) and pd.notna(y) and x > 0]
    if valid:
        lo, hi = min(x for _, x, _ in valid), max(x for _, x, _ in valid)
        lo, hi = __import__("math").log10(lo), __import__("math").log10(hi)
        span = max(hi - lo, 0.01)
        for index, (run, x, y) in enumerate(valid):
            px = 95 + int((__import__("math").log10(x) - lo) / span * (760 - 95))
            py = 610 - int(y * (610 - 80))
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill="#3567a7")
            draw.text((px + 9, py - 10 - 14 * (index % 3)), run, fill="#222")
        for tick in sorted({x for _, x, _ in valid}):
            px = 95 + int((__import__("math").log10(tick) - lo) / span * (760 - 95))
            draw.text((px - 13, 618), f"{tick:.2f}", fill="#555")
    valid_mem = [(str(run), float(x), float(y)) for run, x, y in zip(frame["run_id"], memory, values)
                 if pd.notna(x) and pd.notna(y)]
    if valid_mem:
        lo, hi = min(x for _, x, _ in valid_mem), max(x for _, x, _ in valid_mem)
        span = max(hi - lo, 0.01)
        for index, (run, x, y) in enumerate(valid_mem):
            px = 910 + int((x - lo) / span * (1510 - 910))
            py = 610 - int(y * (610 - 80))
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill="#cb6d3c")
            draw.text((px + 9, py - 10 - 14 * (index % 3)), run, fill="#222")
        for tick in sorted({x for _, x, _ in valid_mem}):
            px = 910 + int((tick - lo) / span * (1510 - 910))
            draw.text((px - 13, 618), f"{tick:.1f}", fill="#555")
    image.save(output_path)
    image.save(Path(output_path).with_suffix(".pdf"), "PDF", resolution=200.0)


def pillow_part_c(rows, output_path):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (1000, 650), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 100, 65, 920, 540
    draw.rectangle((left, top, right, bottom), outline="#222", width=2)
    draw.text((left, bottom + 28), "Canary repetitions", fill="#222")
    draw.text((left, top - 28), "Mean exposure (bits)", fill="#222")
    max_y = max(value for _, _, value in rows) or 1.0
    for value in (0, max_y / 2, max_y):
        y = bottom - int(value / max_y * (bottom - top))
        draw.line((left, y, right, y), fill="#dddddd")
        draw.text((left - 55, y - 8), f"{value:.1f}", fill="#555")
    for reps in (1, 4, 16, 64):
        x = left + int((reps - 1) / 63 * (right - left))
        draw.text((x - 8, bottom + 6), str(reps), fill="#555")
    colors = {"V-raw": "#3567a7", "V-san": "#cb6d3c"}
    for name in ("V-raw", "V-san"):
        group = sorted((r for r in rows if r[0] == name), key=lambda r: r[1])
        points = []
        for _, reps, exposure in group:
            x = left + int((reps - 1) / 63 * (right - left))
            y = bottom - int(exposure / max_y * (bottom - top))
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=colors[name], width=3)
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=colors[name])
    draw.text((700, 575), "Blue: V-raw    Orange: V-san", fill="#222")
    image.save(output_path)
    image.save(Path(output_path).with_suffix(".pdf"), "PDF", resolution=200.0)


def part_a_figure(csv_path, output_path):
    frame = pd.read_csv(csv_path)
    if "run_id" in frame:
        frame = frame.sort_values("run_id")
    values = pd.to_numeric(frame.get("val_entity_f1"), errors="coerce")
    if values.isna().all():
        raise ValueError("Part A entity F1 is unavailable. Run sanitizer validation before plotting.")
    params = pd.to_numeric(frame["trainable_pct"], errors="coerce")
    memory = pd.to_numeric(frame["peak_gpu_mem_gb"], errors="coerce")
    if plt is None:
        pillow_part_a(frame, values, params, memory, output_path)
        return
    fig, (ax, ax_memory) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax.scatter(params, values, s=65, color="#3567a7")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (%) [log scale]")
    ax.set_ylabel("Validation exact micro F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    for run_id, x, y in zip(frame["run_id"], params, values):
        if pd.notna(x) and pd.notna(y):
            ax.annotate(str(run_id), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    valid_memory = memory.notna() & values.notna()
    ax_memory.scatter(memory[valid_memory], values[valid_memory], s=65, color="#cb6d3c")
    ax_memory.set_xlabel("Peak GPU memory (GB)")
    ax_memory.set_ylabel("Validation exact micro F1")
    ax_memory.set_ylim(0, 1)
    ax_memory.grid(alpha=0.25)
    for run_id, x, y in zip(frame.loc[valid_memory, "run_id"], memory[valid_memory], values[valid_memory]):
        ax_memory.annotate(str(run_id), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    fig.savefig(Path(output_path).with_suffix(".pdf"))


def part_c_figure(report_path, output_path):
    reports = json.loads(Path(report_path).read_text(encoding="utf-8"))
    name_map = {"victim_raw": "V-raw", "victim_san": "V-san"}
    rows = []
    for key, report in reports.items():
        if key in name_map:
            for reps, stat in report.get("canary", {}).get("by_n_reps", {}).items():
                rows.append((name_map[key], int(reps), stat["mean_exposure"]))
    if not rows:
        raise ValueError("V-raw and V-san canary results are required for the exposure figure.")
    if plt is None:
        pillow_part_c(rows, output_path)
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ("V-raw", "V-san"):
        group = sorted((r for r in rows if r[0] == name), key=lambda r: r[1])
        ax.plot([r[1] for r in group], [r[2] for r in group], marker="o", linewidth=2, label=name)
    ax.set_xlabel("Canary repetitions")
    ax.set_ylabel("Mean exposure (bits)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    fig.savefig(Path(output_path).with_suffix(".pdf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part_a_csv", required=True)
    parser.add_argument("--part_c_report", required=True)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    part_a_figure(args.part_a_csv, out / "partA_fig.png")
    part_c_figure(args.part_c_report, out / "partC_exposure.png")


if __name__ == "__main__":
    main()
