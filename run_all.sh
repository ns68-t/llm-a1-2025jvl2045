#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${DATA_DIR:-data}"
SANITIZED_CORPUS="${SANITIZED_CORPUS:-data/partC_corpus_sanitized.jsonl}"

# Each split can be overridden independently for TA/private verification. The
# defaults reproduce the instructor-provided bundle layout under DATA_DIR.
PARTA_TRAIN_FILE="${PARTA_TRAIN_FILE:-$DATA_DIR/partA_train.jsonl}"
PARTA_VALIDATION_FILE="${PARTA_VALIDATION_FILE:-$DATA_DIR/partA_validation.jsonl}"
PARTB_TEST_FILE="${PARTB_TEST_FILE:-$DATA_DIR/partB_test.jsonl}"
PARTB_OOD_TEST_FILE="${PARTB_OOD_TEST_FILE:-$DATA_DIR/partB_ood_test.jsonl}"
PARTC_RAW_CORPUS_FILE="${PARTC_RAW_CORPUS_FILE:-$DATA_DIR/partC_corpus_raw.jsonl}"
PARTC_HELDOUT_FILE="${PARTC_HELDOUT_FILE:-$DATA_DIR/partC_heldout.jsonl}"
PARTC_CANARIES_FILE="${PARTC_CANARIES_FILE:-$DATA_DIR/partC_canaries.jsonl}"
PARTC_TARGETED_FILE="${PARTC_TARGETED_FILE:-$DATA_DIR/partC_targeted.jsonl}"

for required in "$PARTA_TRAIN_FILE" "$PARTA_VALIDATION_FILE" "$PARTB_TEST_FILE" "$PARTB_OOD_TEST_FILE" "$PARTC_RAW_CORPUS_FILE" "$PARTC_HELDOUT_FILE" "$PARTC_CANARIES_FILE" "$PARTC_TARGETED_FILE"; do
  [[ -f "$required" ]] || { echo "Missing required input: $required"; exit 1; }
done

# Fresh Kaggle/TA environments can include torchao, which is not used here and
# can conflict with the PEFT/Transformers stack. Remove it before installation.
if python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torchao') else 1)"; then
  python -m pip uninstall -y torchao
fi

# Install the declared runtime packages and Presidio's default English NLP model.
python -m pip install -r requirements.txt
if ! python -c "import spacy; spacy.load('en_core_web_lg')"; then
  python -m spacy download en_core_web_lg
fi

mkdir -p adapters results
rm -f results/partA_sweep.csv
SAN=Qwen/Qwen2.5-1.5B-Instruct
VICTIM=Qwen/Qwen2.5-0.5B
TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
BEST_RUN="${BEST_RUN:-A4}"
case "$BEST_RUN" in A1|A2|A3|A4|A5|A6) ;; *) echo "BEST_RUN must be A1 through A6"; exit 1;; esac
for spec in "A1 q_proj,v_proj 8 16 none" "A2 q_proj,k_proj,v_proj,o_proj 8 16 none" "A3 $TARGETS 8 16 none" "A4 $TARGETS 32 64 none" "A5 $TARGETS 32 16 none" "A6 $TARGETS 32 64 4bit"; do
  read -r run modules rank alpha quant <<< "$spec"
  python src/peft_sweep.py --model "$SAN" --train_file "$PARTA_TRAIN_FILE" --val_file "$PARTA_VALIDATION_FILE" --target_modules "$modules" --lora_r "$rank" --lora_alpha "$alpha" --lora_dropout 0.05 --quant "$quant" --seed 42 --epochs 3 --run_id "$run" --output_dir "adapters/$run" --results_csv results/partA_sweep.csv
done
# Evaluate every Part A candidate on the validation data before selecting A4.
for run in A1 A2 A3 A4 A5 A6; do
  python src/sanitize_eval.py --base_model "$SAN" --adapter_path "adapters/$run" --test_file "$PARTA_VALIDATION_FILE" --system finetuned --output_predictions "results/partA_${run}_validation_predictions.jsonl" --report_file "results/partA_${run}_validation_metrics.json" --overwrite
  python scripts/update_part_a_metrics.py --results_csv results/partA_sweep.csv --run_id "$run" --metrics_file "results/partA_${run}_validation_metrics.json"
done
rm -rf adapters/best_sanitizer
cp -a "adapters/$BEST_RUN" adapters/best_sanitizer
for split in id ood; do
  for system in finetuned regex presidio zeroshot; do
    adapter_args=(); [[ "$system" == finetuned ]] && adapter_args=(--adapter_path adapters/best_sanitizer)
    if [[ "$split" == ood ]]; then
      test_file="$PARTB_OOD_TEST_FILE"
      suffix="_ood"
    else
      test_file="$PARTB_TEST_FILE"
      suffix=""
    fi
    if [[ "$system" == finetuned ]]; then
      output_prefix="partB"
    else
      output_prefix="partB_${system}"
    fi
    python src/sanitize_eval.py --base_model "$SAN" "${adapter_args[@]}" --test_file "$test_file" --system "$system" --output_predictions "results/${output_prefix}_predictions${suffix}.jsonl" --report_file "results/${output_prefix}_metrics${suffix}.json" --overwrite
  done
done
# A1 requires V-san to be trained on a model-sanitized copy of the exact raw
# corpus. This preserves all metadata and does not special-case canaries.
if [[ ! -f "$SANITIZED_CORPUS" ]]; then
  python src/sanitize_corpus.py --base_model "$SAN" --adapter_path adapters/best_sanitizer --input_file "$PARTC_RAW_CORPUS_FILE" --output_file "$SANITIZED_CORPUS" --batch_size 4 --max_new_tokens 1024 --seed 42
fi
python src/train_victim.py --model "$VICTIM" --corpus "$PARTC_RAW_CORPUS_FILE" --heldout "$PARTC_HELDOUT_FILE" --victim_name V-raw --target_modules "$TARGETS" --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --epochs 4 --batch_size 4 --grad_accum 4 --lr 2e-4 --seed 42 --output_dir adapters/victim_raw
python src/train_victim.py --model "$VICTIM" --corpus "$SANITIZED_CORPUS" --heldout "$PARTC_HELDOUT_FILE" --victim_name V-san --target_modules "$TARGETS" --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --epochs 4 --batch_size 4 --grad_accum 4 --lr 2e-4 --seed 42 --output_dir adapters/victim_san
ATTACK=(--victim_model "$VICTIM" --ref_model "$VICTIM" --heldout "$PARTC_HELDOUT_FILE" --canaries "$PARTC_CANARIES_FILE" --targeted_file "$PARTC_TARGETED_FILE" --attack all --num_samples 5000 --seed 42)
PII=(); [[ -n "${PII_INDEX:-}" ]] && PII=(--pii_index "$PII_INDEX")
python src/attack_extract.py "${ATTACK[@]}" --corpus "$PARTC_RAW_CORPUS_FILE" "${PII[@]}" --report_file results/partC_attacks_base.json
python src/attack_extract.py "${ATTACK[@]}" --adapter_path adapters/victim_raw --corpus "$PARTC_RAW_CORPUS_FILE" "${PII[@]}" --report_file results/partC_attacks.json
python src/attack_extract.py "${ATTACK[@]}" --adapter_path adapters/victim_san --corpus "$SANITIZED_CORPUS" "${PII[@]}" --report_file results/partC_attacks_san.json
python scripts/merge_partc.py --base results/partC_attacks_base.json --raw results/partC_attacks.json --san results/partC_attacks_san.json --output results/partC_comparison.json
python src/plot_results.py --part_a_csv results/partA_sweep.csv --part_c_report results/partC_comparison.json --output_dir results
