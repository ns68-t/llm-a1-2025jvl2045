"""Attach sanitizer validation metrics to the matching Part A sweep row."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--metrics_file", required=True)
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    report = json.loads(Path(args.metrics_file).read_text(encoding="utf-8"))
    frame = pd.read_csv(csv_path)
    mask = frame["run_id"].astype(str) == args.run_id
    if mask.sum() != 1:
        raise ValueError(f"Expected one row for {args.run_id}; found {mask.sum()}.")
    frame.loc[mask, "val_entity_f1"] = report["exact"]["micro_f1"]
    frame.loc[mask, "val_leak_rate"] = report["leak_rate"]
    frame.loc[mask, "validation_metric_status"] = "sanitizer validation evaluation"
    frame.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
