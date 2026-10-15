"""Part B evaluation: explicit span alignment and maximum-total-IoU matching.

Uses the provided privacy_mask as gold. No gold-dependent label remapping.
See EVALUATION_RULES for the alignment and taxonomy limitations.
"""
import argparse
from collections import Counter
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time

VERSION = "partb-corrected-1"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MASK = re.compile(r"\[([A-Z][A-Z0-9_]*?)\]")
BASELINE_LABELS = {
    "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "TEL", "IP_ADDRESS": "IP",
    "US_SSN": "SOCIALNUMBER", "US_PASSPORT": "PASSPORT",
    "US_DRIVER_LICENSE": "DRIVERLICENSE", "CREDIT_CARD": "CREDITCARDNUMBER",
    "DATE_TIME": "DATE", "IBAN_CODE": "IBAN",
}
EVALUATION_RULES = {
    "version": VERSION,
    "gold": "Unmodified supplied privacy_mask; original class labels retained.",
    "alignment": (
        "Replace each recognized output placeholder with a unique sentinel and align "
        "source/output using SequenceMatcher(autojunk=False), first locking the exact "
        "common prefix and suffix. A changed source interval "
        "with one placeholder gives one predicted span. Multiple placeholders in one "
        "unresolved interval are omitted and counted as ambiguous; zero-width insertions "
        "give no span. Literal rewrites and deletions still count toward over-masking. "
        "Repeated text can cause ambiguous alignment. Literal placeholder-like source "
        "text can be mistaken for a generated placeholder."
    ),
    "labels": "Strip only a trailing underscore plus digits (e.g. EMAIL_1); retain GIVENNAME1.",
    "matching": "One-to-one exact matching; overlap matching maximizes total IoU over same-label pairs with IoU >= 0.5.",
    "macro": "Unweighted mean over the union of gold and predicted classes; zero undefined P/R/F1.",
    "leak": "Count gold entity occurrences whose values remain after lowercasing and collapsing whitespace.",
    "over_masking": "Changed non-PII source characters / all non-PII source characters; character unions avoid double counting.",
    "well_formed": "No unmatched square brackets, code fences, chat control tokens, or ambiguous multi-placeholder intervals. A formatting proxy, not semantic correctness.",
    "baseline_labels": BASELINE_LABELS,
    "baseline_limitations": (
        "Same fixed mapping on test and OOD. Presidio PERSON/LOCATION remain coarse "
        "labels and cannot match fine-grained names/addresses exactly. DATE_TIME maps "
        "to DATE. OOD taxonomy differences are not remapped using test gold."
    ),
}


def normalize(text):
    return " ".join(str(text).lower().split())


def make_prompt(text):
    return f"<|im_start|>user\nSanitize the following text:\n{text}<|im_end|>\n<|im_start|>assistant\n"


def recover_spans(source, output):
    labels = {}
    pieces = []
    cursor = 0
    marker_code = 0xF0000
    for match in MASK.finditer(output):
        while chr(marker_code) in source or chr(marker_code) in output:
            marker_code += 1
        marker = chr(marker_code)
        marker_code += 1
        labels[marker] = re.sub(r"_\d+$", "", match.group(1))
        pieces.extend([output[cursor:match.start()], marker])
        cursor = match.end()
    pieces.append(output[cursor:])
    masked_output = "".join(pieces)
    predicted, changed, ambiguous = [], set(), 0
    prefix = 0
    while prefix < min(len(source), len(masked_output)) and source[prefix] == masked_output[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < min(len(source), len(masked_output)) - prefix
           and source[len(source) - suffix - 1] == masked_output[len(masked_output) - suffix - 1]):
        suffix += 1
    source_end, output_end = len(source) - suffix, len(masked_output) - suffix
    matcher = difflib.SequenceMatcher(None, source[prefix:source_end], masked_output[prefix:output_end], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        i1, i2, j1, j2 = i1 + prefix, i2 + prefix, j1 + prefix, j2 + prefix
        changed.update(range(i1, i2))
        found = [labels[c] for c in masked_output[j1:j2] if c in labels]
        if len(found) == 1 and i2 > i1:
            predicted.append({"start": i1, "end": i2, "label": found[0]})
        elif len(found) > 1:
            ambiguous += 1
    residual = MASK.sub("", output)
    well_formed = not any(x in residual for x in ("[", "]", "```", "<|")) and ambiguous == 0
    return predicted, changed, ambiguous, well_formed


def iou(a, b):
    intersection = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = a["end"] - a["start"] + b["end"] - b["start"] - intersection
    return intersection / union if union else 0.0


def maximum_weight_pairs(weights):
    """Rectangular Hungarian assignment, with zero-weight dummy columns."""
    n = len(weights)
    if not n or not weights[0]:
        return []
    original_m = len(weights[0])
    m = max(n, original_m)
    cost = [[-float(v) for v in row] + [0.0] * (m - original_m) for row in weights]
    u, v, p, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv, used = [float("inf")] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, original_m + 1)
            if p[j] and weights[p[j] - 1][j - 1] > 0]


def match_entities(predicted, gold, overlap=False):
    weights = []
    for a in predicted:
        row = []
        for b in gold:
            score = 0.0
            if a["label"] == b["label"]:
                if overlap:
                    value = iou(a, b)
                    score = value if value >= 0.5 else 0.0
                elif a["start"] == b["start"] and a["end"] == b["end"]:
                    score = 1.0
            row.append(score)
        weights.append(row)
    return maximum_weight_pairs(weights)


def prf(tp, predicted, gold):
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"p": precision, "r": recall, "f1": f1}


def make_prediction(row, text, latency_ms, direct_spans=None, output_limit_hit=False):
    spans, changed, ambiguous, formed = recover_spans(row["source_text"], text)
    if direct_spans is not None:
        spans = direct_spans
        changed = {i for s in spans for i in range(s["start"], s["end"])}
        ambiguous, formed = 0, True
    gold = row["privacy_mask"]
    gold_chars = {i for s in gold for i in range(s["start"], s["end"])}
    normalized_output = normalize(text)
    leaked = []
    for s in gold:
        value = str(s.get("value", row["source_text"][s["start"]:s["end"]]))
        if normalize(value) and normalize(value) in normalized_output:
            leaked.append(value)
    return {
        "id": row["id"], "language": row.get("language", "English"),
        "source_text": row["source_text"], "predicted_text": text,
        "predicted_spans": spans, "gold_spans": gold, "leaked_values": leaked,
        "latency_ms": latency_ms, "well_formed": formed,
        "over_masked_chars": len(changed - gold_chars),
        "non_pii_chars": len(row["source_text"]) - len(gold_chars),
        "ambiguous_alignment_regions": ambiguous, "output_limit_hit": output_limit_hit,
    }


def aggregate(predictions, system, adapter, test_set):
    gold_counts, pred_counts = Counter(), Counter()
    exact_counts, overlap_counts, errors = Counter(), Counter(), Counter()
    for row in predictions:
        gold, pred = row["gold_spans"], row["predicted_spans"]
        gold_counts.update(s["label"] for s in gold)
        pred_counts.update(s["label"] for s in pred)
        exact = match_entities(pred, gold)
        overlap = match_entities(pred, gold, overlap=True)
        exact_counts.update(gold[j]["label"] for _, j in exact)
        overlap_counts.update(gold[j]["label"] for _, j in overlap)
        matched_gold = {j for _, j in exact}
        for j, g in enumerate(gold):
            if j in matched_gold:
                continue
            if any(p["label"] == g["label"] and iou(p, g) > 0 for p in pred):
                errors["boundary_error"] += 1
            elif any(p["label"] != g["label"] and iou(p, g) >= 0.5 for p in pred):
                errors["label_error"] += 1
            else:
                errors["missed_entity"] += 1
        errors["unmatched_predictions_exact"] += len(pred) - len(exact)
        errors["empty_outputs"] += not row["predicted_text"].strip()
        errors["ambiguous_alignment_regions"] += row["ambiguous_alignment_regions"]
        errors["output_limit_hits"] += row["output_limit_hit"]
    classes = sorted(gold_counts.keys() | pred_counts.keys())
    per_class = {}
    for c in classes:
        exact = prf(exact_counts[c], pred_counts[c], gold_counts[c])
        overlap = prf(overlap_counts[c], pred_counts[c], gold_counts[c])
        per_class[c] = {**exact, "support": gold_counts[c], "predicted": pred_counts[c],
                        "exact": exact, "overlap_iou50": overlap}
    report = {
        "system": system, "adapter": adapter, "test_set": test_set, "n": len(predictions),
        "per_class": per_class,
        "top10_classes": sorted(gold_counts, key=lambda c: (-gold_counts[c], c))[:10],
        "error_taxonomy": dict(errors), "evaluation_rules": EVALUATION_RULES,
    }
    for key, counts in (("exact", exact_counts), ("overlap_iou50", overlap_counts)):
        micro = prf(sum(counts.values()), sum(pred_counts.values()), sum(gold_counts.values()))
        report[key] = {f"micro_{k}": value for k, value in micro.items()}
        for k in ("p", "r", "f1"):
            report[key][f"macro_{k}"] = sum(per_class[c][key][k] for c in classes) / len(classes) if classes else 0.0
    total_gold = sum(gold_counts.values())
    non_pii = sum(r["non_pii_chars"] for r in predictions)
    report["leak_rate"] = sum(len(r["leaked_values"]) for r in predictions) / total_gold if total_gold else 0.0
    report["over_masking_rate"] = sum(r["over_masked_chars"] for r in predictions) / non_pii if non_pii else 0.0
    report["well_formed_rate"] = sum(r["well_formed"] for r in predictions) / len(predictions) if predictions else 0.0
    return report


REGEX_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "TEL": r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "IP": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    "CREDITCARDNUMBER": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
    "DATE": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
    "URL": r'https?://[^\s<>"]+|www\.[^\s<>"]+',
}


def nonoverlapping(candidates):
    chosen = []
    for s in sorted(candidates, key=lambda s: (-s.get("score", 1.0), -(s["end"] - s["start"]), s["start"], s["label"])):
        if all(s["end"] <= t["start"] or s["start"] >= t["end"] for t in chosen):
            chosen.append(s)
    return [{k: s[k] for k in ("start", "end", "label")} for s in sorted(chosen, key=lambda s: s["start"])]


def render_masks(text, spans):
    output, last = [], 0
    for s in spans:
        output.extend([text[last:s["start"]], f'[{s["label"]}]'])
        last = s["end"]
    output.append(text[last:])
    return "".join(output)


def regex_predict(text):
    candidates = [{"start": m.start(), "end": m.end(), "label": label}
                  for label, pattern in REGEX_PATTERNS.items()
                  for m in re.finditer(pattern, text, flags=re.IGNORECASE)]
    spans = nonoverlapping(candidates)
    return render_masks(text, spans), spans


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    source = Path(args.test_file)
    records = [json.loads(l) for l in source.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert records, "Input JSONL contains no records."
    assert len({r["id"] for r in records}) == len(records), "Duplicate input IDs."
    prediction_file, report_file = Path(args.output_predictions), Path(args.report_file)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file = prediction_file.with_suffix(".meta.json")
    if args.overwrite:
        prediction_file.unlink(missing_ok=True)
        meta_file.unlink(missing_ok=True)
        report_file.unlink(missing_ok=True)
    settings = {
        "version": VERSION, "input_sha256": sha256(source), "system": args.system,
        "model": args.base_model, "seed": args.seed, "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens, "prompt": make_prompt("{source_text}"),
        "input_truncation": False, "do_sample": False, "num_beams": 1,
        "repetition_penalty": 1.0, "dtype": "float16", "label_mapping": BASELINE_LABELS,
    }
    if args.system == "finetuned":
        from hub_assets import ensure_adapter
        adapter = ensure_adapter(args.adapter_path)
        args.adapter_path = str(adapter)
        cfg = json.loads((adapter / "adapter_config.json").read_text())
        assert cfg["base_model_name_or_path"] == args.base_model, "Wrong adapter base model."
        settings["adapter_config"] = cfg
        settings["adapter_sha256"] = sha256(adapter / "adapter_model.safetensors")
    for package in ("torch", "transformers", "peft", "presidio-analyzer", "spacy", "en-core-web-lg"):
        try:
            settings[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if meta_file.exists():
        assert json.loads(meta_file.read_text()) == settings, "Settings changed. Use a new output filename to avoid mixing runs."
    else:
        assert not prediction_file.exists() or prediction_file.stat().st_size == 0, "Existing predictions have no run metadata. Use a new filename."
        meta_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    saved = []
    if prediction_file.exists():
        saved = [json.loads(l) for l in prediction_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(saved) <= len(records)
    assert [r["id"] for r in saved] == [r["id"] for r in records[:len(saved)]], "Resume ID mismatch."
    analyzer = model = tokenizer = None
    if len(saved) < len(records) and args.system == "presidio":
        import spacy
        spacy.require_cpu()
        from presidio_analyzer import AnalyzerEngine
        analyzer = AnalyzerEngine()
    if len(saved) < len(records) and args.system in ("finetuned", "zeroshot"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        assert torch.cuda.is_available(), "GPU is required for this notebook's model evaluation."
        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map={"": 0})
        if args.system == "finetuned":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_path)
        model.eval()
        stop_ids = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")})
        config = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False,
                                  num_beams=1, num_return_sequences=1, repetition_penalty=1.0,
                                  no_repeat_ngram_size=0, pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=stop_ids, use_cache=True)
    step = args.batch_size if args.system in ("finetuned", "zeroshot") else 1
    with prediction_file.open("a", encoding="utf-8") as f:
        for start in range(len(saved), len(records), step):
            batch = records[start:start + step]
            if args.system in ("finetuned", "zeroshot"):
                inputs = tokenizer([make_prompt(r["source_text"]) for r in batch], padding=True,
                                   truncation=False, add_special_tokens=False, return_tensors="pt").to(model.device)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=config)
                torch.cuda.synchronize()
                latency = 1000 * (time.perf_counter() - t0) / len(batch)
                tokens = generated[:, inputs["input_ids"].shape[1]:].cpu()
                texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
                new_rows = [make_prediction(r, text.strip(), latency, output_limit_hit=not any(t in stop_ids for t in seq.tolist()))
                            for r, text, seq in zip(batch, texts, tokens)]
                del inputs, generated, tokens
            else:
                r = batch[0]
                t0 = time.perf_counter()
                if args.system == "regex":
                    text, spans = regex_predict(r["source_text"])
                else:
                    hits = analyzer.analyze(text=r["source_text"], language="en")
                    spans = nonoverlapping([{"start": h.start, "end": h.end, "score": h.score,
                                             "label": BASELINE_LABELS.get(h.entity_type, h.entity_type)} for h in hits])
                    text = render_masks(r["source_text"], spans)
                latency = 1000 * (time.perf_counter() - t0)
                new_rows = [make_prediction(r, text, latency, direct_spans=spans)]
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            saved.extend(new_rows)
            if len(saved) % 100 == 0 or len(saved) == len(records):
                print(f"{args.system} {source.name}: saved {len(saved)}/{len(records)}", flush=True)
    report = aggregate(saved, args.system, args.adapter_path if args.system == "finetuned" else "none", source.name)
    report["run_settings"] = settings
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"DONE: {report_file}\nExact F1={report['exact']['micro_f1']:.4f}; leak={report['leak_rate']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=MODEL_ID)
    parser.add_argument("--adapter_path", default="adapters/best_sanitizer")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--system", required=True, choices=["finetuned", "regex", "presidio", "zeroshot"])
    parser.add_argument("--output_predictions", required=True)
    parser.add_argument("--report_file", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing prediction/report pair.")
    args = parser.parse_args()
    assert args.batch_size > 0 and args.max_new_tokens > 0
    run(args)


if __name__ == "__main__":
    main()
