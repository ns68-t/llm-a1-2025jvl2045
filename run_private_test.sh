#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash run_private_test.sh /path/to/private_test.jsonl [output_dir]"
  exit 2
fi

test_file="$1"
output_dir="${2:-ta_results}"
mkdir -p "$output_dir"
python src/sanitize_eval.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path adapters/best_sanitizer \
  --test_file "$test_file" \
  --system finetuned \
  --output_predictions "$output_dir/predictions.jsonl" \
  --report_file "$output_dir/metrics.json"
