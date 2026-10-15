Bundled inputs for direct script verification:
- Part A: partA_train.jsonl and partA_validation.jsonl
- Part B: partB_test.jsonl and partB_ood_test.jsonl
- Part C: partC_corpus_raw.jsonl, partC_corpus_sanitized.jsonl, partC_heldout.jsonl, partC_canaries.jsonl, and partC_targeted.jsonl
- The supplied bundle does not contain partC_pii_index.jsonl. PII_INDEX remains optional and PII recovery is reported as unavailable when it is absent.
- Base-model weights are intentionally not included; the scripts fetch the A1-specified Qwen base models at runtime.
