"""Merge independently generated Part C reports after validating their model keys."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--raw", type=Path, required=True)
parser.add_argument("--san", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

merged = {}
for expected, path in (("victim_base", args.base), ("victim_raw", args.raw), ("victim_san", args.san)):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("victim") != expected:
        raise ValueError(f"{path} has victim={report.get('victim')!r}; expected {expected!r}")
    merged[expected] = report

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
