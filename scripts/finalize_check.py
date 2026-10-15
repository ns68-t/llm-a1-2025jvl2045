#!/usr/bin/env python3
"""Fail closed when required Assignment 1 submission artifacts are missing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "README.md",
    "SUBMISSION_STATUS.md",
    "requirements.txt",
    "run_all.sh",
    "src/peft_sweep.py",
    "src/sanitize_eval.py",
    "src/sanitize_corpus.py",
    "src/train_victim.py",
    "src/attack_extract.py",
    "src/plot_results.py",
    "scripts/update_part_a_metrics.py",
    "scripts/merge_partc.py",
    "adapters/best_sanitizer/adapter_model.safetensors",
    "adapters/victim_raw/adapter_model.safetensors",
    "adapters/victim_san/adapter_model.safetensors",
)

REQUIRED_RESULTS = (
    "results/partA_sweep.csv",
    "results/partA_fig.png",
    "results/partA_fig.pdf",
    "results/partA_A6_validation_metrics.json",
    "results/partB_predictions.jsonl",
    "results/partB_metrics.json",
    "results/partB_metrics_ood.json",
    "results/partB_regex_metrics.json",
    "results/partB_presidio_metrics.json",
    "results/partC_attacks.json",
    "results/partC_attacks_base.json",
    "results/partC_attacks_san.json",
    "results/partC_comparison.json",
    "results/partC_exposure.png",
    "results/partC_exposure.pdf",
    "report.pdf",
)

PART_A_REQUIRED_COLUMNS = (
    "run_id",
    "trainable_params",
    "total_params",
    "trainable_pct",
    "peak_gpu_mem_gb",
    "train_wallclock_s",
    "tokens_per_s",
    "final_train_loss",
    "val_loss",
    "val_entity_f1",
    "val_leak_rate",
)


def missing_paths(paths: tuple[str, ...]) -> list[str]:
    return [item for item in paths if not (ROOT / item).is_file()]


def check_part_a() -> list[str]:
    path = ROOT / "results/partA_sweep.csv"
    if not path.is_file():
        return ["results/partA_sweep.csv is missing"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        columns = set(rows[0]) if rows else set()
    problems = [f"Part A CSV lacks column: {column}" for column in PART_A_REQUIRED_COLUMNS if column not in columns]
    by_run = {row.get("run_id"): row for row in rows}
    for run in ("A1", "A2", "A3", "A4", "A5", "A6"):
        row = by_run.get(run)
        if row is None:
            problems.append(f"Part A CSV has no {run} row")
            continue
        for column in PART_A_REQUIRED_COLUMNS[1:]:
            if not (row.get(column) or "").strip():
                problems.append(f"{run} has an incomplete value for {column}")
    return problems


def check_json(path: str, required_keys: tuple[str, ...]) -> list[str]:
    file_path = ROOT / path
    if not file_path.is_file():
        return [f"{path} is missing"]
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path} is not valid JSON: {exc}"]
    return [f"{path} lacks top-level key: {key}" for key in required_keys if key not in payload]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Return non-zero when any finalization item is missing.")
    args = parser.parse_args()

    problems = [f"Missing required package file: {item}" for item in missing_paths(REQUIRED_FILES)]
    problems.extend(f"Missing required final result: {item}" for item in missing_paths(REQUIRED_RESULTS))
    problems.extend(check_part_a())
    json_checks = (
        ("results/partB_metrics.json", ("exact", "overlap_iou50", "leak_rate", "well_formed_rate")),
        ("results/partB_metrics_ood.json", ("exact", "overlap_iou50", "leak_rate", "well_formed_rate")),
        ("results/partA_A6_validation_metrics.json", ("exact", "overlap_iou50", "leak_rate", "well_formed_rate")),
        ("results/partC_attacks.json", ("victim", "heldout_ppl", "discoverable", "untargeted", "canary", "targeted")),
        ("results/partC_attacks_base.json", ("victim", "heldout_ppl", "discoverable", "untargeted", "canary", "targeted")),
        ("results/partC_attacks_san.json", ("victim", "heldout_ppl", "discoverable", "untargeted", "canary", "targeted")),
        ("results/partC_comparison.json", ("victim_base", "victim_raw", "victim_san")),
    )
    for path, keys in json_checks:
        if (ROOT / path).is_file():
            problems.extend(check_json(path, keys))

    if problems:
        print("NOT READY")
        for problem in problems:
            print(f"- {problem}")
        return 1 if args.strict else 0
    print("READY: package structure and required generated artifacts are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
