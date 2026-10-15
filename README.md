# Assignment 1 — 2025JVL2045

This repository contains the complete Part A, B, and C code and report. Original datasets and large weights are not stored in GitHub. All eight trained adapters (A1–A6, V-raw, V-san) are public at [Neeraj77/llm-a1-adapters](https://huggingface.co/Neeraj77/llm-a1-adapters). The scripts download the required base models and adapters automatically from Hugging Face.

## Test on TA data

Use Python 3.10+ and an NVIDIA GPU. Install packages, then give the test file to the script:

```bash
python -m pip install -r requirements.txt
bash run_private_test.sh /path/to/private_test.jsonl
```

Replace `/path/to/private_test.jsonl` with the TA file. Output goes to `ta_results/`. For OOD data, run the same command with its file and a different output folder:

```bash
bash run_private_test.sh /path/to/private_ood.jsonl ta_results_ood
```

Each JSONL row needs a unique `id`, `source_text`, and `privacy_mask` (entity `start`, `end`, `label`). Any filename and row count work. The first model run downloads Qwen2.5-1.5B-Instruct and the selected A4 adapter. No weight files need to be copied by hand.

## Reproduce training and results

Place the eight assignment input files in `data/` as named in [data/README.md](data/README.md), then run `bash run_all.sh`. It trains A1–A6 and both victim adapters and runs the evaluations. This takes much longer. Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B download automatically. `src/attack_extract.py` also downloads a victim adapter from Hugging Face when its local copy is absent.

