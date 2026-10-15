# Assignment data

The original JSONL datasets are not included in this submission. To rerun the full pipeline, place the eight supplied files in this `data/` folder without changing their rows or order:

| File | Rows |
| --- | ---: |
| `partA_train.jsonl` | 20,000 |
| `partA_validation.jsonl` | 2,000 |
| `partB_test.jsonl` | 2,000 |
| `partB_ood_test.jsonl` | 1,000 |
| `partC_corpus_raw.jsonl` | 8,007 |
| `partC_heldout.jsonl` | 500 |
| `partC_canaries.jsonl` | 20 |
| `partC_targeted.jsonl` | 400 |

`manifest.jsonl` lists the expected row counts and SHA-256 hashes. The canaries file is evaluation metadata; do not train on it separately.
