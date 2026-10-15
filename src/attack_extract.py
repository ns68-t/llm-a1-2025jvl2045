
import argparse
import difflib
import json
import math
import random
import zlib
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    set_seed,
)

def compute_token_weighted_ppl(model, tokenizer, heldout_path, batch_size=8):
    heldout_dataset = load_dataset("json", data_files={"heldout": str(heldout_path)})["heldout"]

    def tokenize_fn(batch):
        return tokenizer(
            [text + tokenizer.eos_token for text in batch["text"]],
            truncation=True,
            max_length=512,
        )

    encoded = heldout_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=heldout_dataset.column_names,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(encoded, batch_size=batch_size, collate_fn=collator)

    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(**batch)

            # Causal-LM loss is computed on shifted labels.  Count exactly the
            # labels that contribute to that loss rather than estimating from
            # the attention mask (PAD may equal EOS).
            valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            if valid_tokens > 0:
                total_nll += outputs.loss.item() * valid_tokens
                total_tokens += valid_tokens

    if total_tokens == 0:
        return float("inf")
    return round(math.exp(total_nll / total_tokens), 4)

def run_discoverable_memorization(model, tokenizer, corpus_path, batch_size=32):
    rows = [json.loads(line) for line in Path(corpus_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    device = next(model.parameters()).device

    n_documents = len(rows)
    n_excluded_short = 0
    valid_samples = []

    for row in rows:
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        if len(ids) < 100:
            n_excluded_short += 1
            continue
        valid_samples.append({
            "prefix_ids": ids[:50],
            "target_ids": ids[50:100],
            "n_reps": str(row["n_reps"]),
        })

    exact_matches = 0
    total_edit_sim = 0.0
    by_reps = defaultdict(lambda: {"n": 0, "exact_matches": 0, "sim_sum": 0.0})

    model.eval()
    with torch.inference_mode():
        for i in range(0, len(valid_samples), batch_size):
            batch = valid_samples[i:i + batch_size]
            input_tensors = [torch.tensor(item["prefix_ids"], dtype=torch.long) for item in batch]
            input_ids = torch.stack(input_tensors).to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                generated = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=50,
                    min_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=None,
                )

            gen_completions = generated[:, 50:].tolist()

            for item, gen_ids in zip(batch, gen_completions):
                is_exact = (gen_ids == item["target_ids"])
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                target_text = tokenizer.decode(item["target_ids"], skip_special_tokens=True)
                sim = difflib.SequenceMatcher(None, gen_text, target_text).ratio()

                if is_exact:
                    exact_matches += 1
                total_edit_sim += sim

                rep = item["n_reps"]
                by_reps[rep]["n"] += 1
                if is_exact:
                    by_reps[rep]["exact_matches"] += 1
                by_reps[rep]["sim_sum"] += sim

    n_valid = len(valid_samples)
    exact_match_rate = round(exact_matches / n_valid, 4) if n_valid > 0 else 0.0
    mean_edit_sim = round(total_edit_sim / n_valid, 4) if n_valid > 0 else 0.0

    by_n_reps_res = {}
    for rep, data in sorted(by_reps.items(), key=lambda x: int(x[0])):
        by_n_reps_res[rep] = {
            "n": data["n"],
            "exact_match_rate": round(data["exact_matches"] / data["n"], 4) if data["n"] > 0 else 0.0,
            "mean_edit_sim": round(data["sim_sum"] / data["n"], 4) if data["n"] > 0 else 0.0,
        }

    return {
        "n_documents": n_documents,
        "n_excluded_short": n_excluded_short,
        "exact_match_rate": exact_match_rate,
        "mean_edit_sim": mean_edit_sim,
        "by_n_reps": by_n_reps_res,
    }

def compute_sequence_nll(model, tokenizer, text, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    if enc.input_ids.shape[1] < 2:
        return 100.0

    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(**enc)
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = enc.input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
    return loss.item()

def run_untargeted_extraction(
    victim_model,
    ref_model,
    tokenizer,
    corpus_path,
    pii_index_path,
    num_samples=5000,
    batch_size=32,
):
    device = next(victim_model.parameters()).device
    corpus_rows = [json.loads(line) for line in Path(corpus_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus_texts = [r["text"].strip() for r in corpus_rows]

    pii_values = set()
    pii_index_available = bool(pii_index_path and Path(pii_index_path).is_file())
    if pii_index_available:
        pii_records = [json.loads(line) for line in Path(pii_index_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        for rec in pii_records:
            if "value" in rec:
                pii_values.add(rec["value"].strip().lower())

    prompt_ids = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
    generated_texts = []

    victim_model.eval()
    with torch.inference_mode():
        for _ in tqdm(range(0, num_samples, batch_size), desc="Generating untargeted samples"):
            current_bs = min(batch_size, num_samples - len(generated_texts))
            batch_prompts = prompt_ids.repeat(current_bs, 1)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                gens = victim_model.generate(
                    batch_prompts,
                    max_new_tokens=64,
                    min_new_tokens=16,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for gen in gens:
                decoded = tokenizer.decode(gen[1:], skip_special_tokens=True).strip()
                if decoded:
                    generated_texts.append(decoded)

    unique_texts = list(set(generated_texts))

    scored = []
    for text in tqdm(unique_texts, desc="Scoring membership metrics"):
        lv_x = compute_sequence_nll(victim_model, tokenizer, text, device)
        lref_x = compute_sequence_nll(ref_model, tokenizer, text, device)
        lv_lower = compute_sequence_nll(victim_model, tokenizer, text.lower(), device)

        utf8_bytes = text.encode("utf-8")
        compressed_len = len(zlib.compress(utf8_bytes))
        h_zlib = (8.0 * compressed_len) / len(utf8_bytes) if len(utf8_bytes) > 0 else 1.0

        scored.append({
            "text": text,
            "ppl": lv_x,
            # Appendix A defines likelihood-ratio scores from NLL
            # differences, not a quotient of two NLL values.
            "ratio_ref": math.exp(lv_x - lref_x),
            "ratio_lower": math.exp(lv_x - lv_lower),
            "ratio_zlib": lv_x / (h_zlib + 1e-8),
        })

    def is_in_training(text):
        if len(text) < 20:
            return False
        return any(text in c or c in text for c in corpus_texts)

    precision_at_100 = {}
    top_samples = set()

    for metric in ["ppl", "ratio_ref", "ratio_lower", "ratio_zlib"]:
        sorted_samples = sorted(scored, key=lambda x: x[metric])[:100]
        hits = sum(1 for item in sorted_samples if is_in_training(item["text"]))
        precision_at_100[metric] = round(hits / 100.0, 4)
        for item in sorted_samples:
            top_samples.add(item["text"])

    distinct_docs_recovered = 0
    for doc in corpus_texts:
        if any(sample in doc or doc in sample for sample in top_samples):
            distinct_docs_recovered += 1

    recovered_pii = set()
    if pii_values:
        for sample in top_samples:
            sample_lower = sample.lower()
            for val in pii_values:
                if val in sample_lower:
                    recovered_pii.add(val)

    pii_recovery_rate = round(len(recovered_pii) / len(pii_values), 4) if pii_values else None

    return {
        "n_generated": len(generated_texts),
        "n_unique": len(unique_texts),
        "precision_at_100": precision_at_100,
        "distinct_docs_recovered": distinct_docs_recovered,
        "pii_index_status": "provided" if pii_index_available else "not_provided",
        "distinct_pii_values_recovered": len(recovered_pii) if pii_values else None,
        "pii_recovery_rate": pii_recovery_rate,
    }

def score_canary_continuations(model, tokenizer, prefix, candidates, device, batch_size=128):
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    scores = []

    model.eval()
    with torch.inference_mode():
        for i in range(0, len(candidates), batch_size):
            batch_cand = candidates[i:i + batch_size]
            input_list = []
            cand_token_lens = []

            for cand in batch_cand:
                cand_ids = tokenizer.encode(cand, add_special_tokens=False)
                input_list.append(torch.tensor(prefix_ids + cand_ids, dtype=torch.long))
                cand_token_lens.append(len(cand_ids))

            padded = torch.nn.utils.rnn.pad_sequence(
                input_list, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=padded)

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = padded[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss_matrix = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_matrix = loss_matrix.view(shift_labels.size())

            pref_len = len(prefix_ids)
            for b_idx, clen in enumerate(cand_token_lens):
                cand_loss = loss_matrix[b_idx, pref_len - 1 : pref_len - 1 + clen].mean().item()
                scores.append(cand_loss)

    return scores

def run_canary_exposure(model, tokenizer, canaries_path):
    canaries = [json.loads(line) for line in Path(canaries_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    device = next(model.parameters()).device

    exposure_by_rep = defaultdict(list)
    total_exposure = 0.0

    for canary in tqdm(canaries, desc="Evaluating canary exposures"):
        secret = str(canary["secret"]).strip()
        template = canary["template"]
        prefix = template.split("{secret}")[0]
        n_reps = str(canary["n_reps"])
        canary_seed = int(canary.get("seed", 42))

        rng = random.Random(canary_seed)
        alternatives = set()
        while len(alternatives) < 9999:
            cand = f"{rng.randint(0, 999999):06d}"
            if cand != secret:
                alternatives.add(cand)

        all_candidates = [secret] + sorted(alternatives)
        nlls = score_canary_continuations(model, tokenizer, prefix, all_candidates, device)

        secret_nll = nlls[0]
        alt_nlls = nlls[1:]

        rank = 1 + sum(1 for alt in alt_nlls if alt < secret_nll)
        exposure = math.log2(10000.0) - math.log2(rank)

        total_exposure += exposure
        exposure_by_rep[n_reps].append(exposure)

    by_n_reps = {}
    for rep, exp_list in sorted(exposure_by_rep.items(), key=lambda x: int(x[0])):
        by_n_reps[rep] = {
            "n": len(exp_list),
            "mean_exposure": round(sum(exp_list) / len(exp_list), 4),
        }

    return {
        "n_canaries": len(canaries),
        "mean_exposure": round(total_exposure / len(canaries), 4) if canaries else 0.0,
        "by_n_reps": by_n_reps,
    }

def run_targeted_extraction(model, tokenizer, targeted_path, batch_size=16):
    records = [json.loads(line) for line in Path(targeted_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    device = next(model.parameters()).device

    stats = {
        "member": {"total": 0, "greedy_leak": 0, "sampled_any10_leak": 0},
        "control": {"total": 0, "greedy_leak": 0, "sampled_any10_leak": 0},
    }

    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(0, len(records), batch_size), desc="Running targeted attacks"):
            batch = records[i:i + batch_size]
            prompts = [item["prompt"] for item in batch]
            gold_values = [str(item["gold_value"]).strip().lower() for item in batch]
            splits = [item["split"] for item in batch]

            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                greedy_out = model.generate(
                    **enc,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for b_idx in range(len(batch)):
                split = splits[b_idx]
                gold = gold_values[b_idx]
                stats[split]["total"] += 1

                prompt_len = enc.input_ids[b_idx].shape[0]
                greedy_text = tokenizer.decode(greedy_out[b_idx][prompt_len:], skip_special_tokens=True).strip().lower()
                if gold in greedy_text:
                    stats[split]["greedy_leak"] += 1

                single_input = enc.input_ids[b_idx:b_idx + 1].repeat(10, 1)
                single_mask = enc.attention_mask[b_idx:b_idx + 1].repeat(10, 1)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    sampled_out = model.generate(
                        input_ids=single_input,
                        attention_mask=single_mask,
                        max_new_tokens=32,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                sampled_leak = False
                for s_idx in range(10):
                    cand_text = tokenizer.decode(sampled_out[s_idx][prompt_len:], skip_special_tokens=True).strip().lower()
                    if gold in cand_text:
                        sampled_leak = True
                        break

                if sampled_leak:
                    stats[split]["sampled_any10_leak"] += 1

    return {
        "n_member": stats["member"]["total"],
        "n_control": stats["control"]["total"],
        "member": {
            "greedy_leak_rate": round(stats["member"]["greedy_leak"] / stats["member"]["total"], 4) if stats["member"]["total"] > 0 else 0.0,
            "sampled_any10_leak_rate": round(stats["member"]["sampled_any10_leak"] / stats["member"]["total"], 4) if stats["member"]["total"] > 0 else 0.0,
        },
        "control": {
            "greedy_leak_rate": round(stats["control"]["greedy_leak"] / stats["control"]["total"], 4) if stats["control"]["total"] > 0 else 0.0,
            "sampled_any10_leak_rate": round(stats["control"]["sampled_any10_leak"] / stats["control"]["total"], 4) if stats["control"]["total"] > 0 else 0.0,
        },
    }

def main():
    parser = argparse.ArgumentParser(description="Part C Extraction Attacks Suite")
    parser.add_argument("--victim_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--ref_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--canaries", required=True)
    parser.add_argument("--pii_index", default=None)
    parser.add_argument("--targeted_file", required=True)
    parser.add_argument(
        "--attack",
        choices=["discoverable", "untargeted", "canary", "targeted", "all"],
        default="all",
    )
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_file", required=True)
    args = parser.parse_args()
    assert args.num_samples >= 100, "--num_samples must be at least 100 for Precision@100."

    if args.adapter_path:
        from hub_assets import ensure_adapter
        args.adapter_path = str(ensure_adapter(args.adapter_path))

    set_seed(args.seed)

    if not args.adapter_path:
        victim_name = "victim_base"
    else:
        norm_path = str(args.adapter_path).lower()
        # FIXED NAMING LOGIC: Checks for specific folder names so v_raw isn't marked as san
        if "v_san" in norm_path or "victim_san" in norm_path:
            victim_name = "victim_san"
        elif "v_raw" in norm_path or "victim_raw" in norm_path:
            victim_name = "victim_raw"
        else:
            victim_name = Path(args.adapter_path).name

    tokenizer = AutoTokenizer.from_pretrained(args.victim_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Base Victim: {args.victim_model}")
    base_victim = AutoModelForCausalLM.from_pretrained(
        args.victim_model,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )

    if args.adapter_path:
        print(f"Applying LoRA adapter from: {args.adapter_path}")
        victim_model = PeftModel.from_pretrained(base_victim, args.adapter_path)
    else:
        victim_model = base_victim

    print(f"Computing held-out token-weighted PPL for {victim_name}...")
    heldout_ppl = compute_token_weighted_ppl(victim_model, tokenizer, args.heldout)

    results = {
        "victim": victim_name,
        "heldout_ppl": heldout_ppl,
        "discoverable": {},
        "untargeted": {},
        "canary": {},
        "targeted": {},
    }

    if args.attack in ["discoverable", "all"]:
        print("\n--- Running Discoverable Memorization ---")
        results["discoverable"] = run_discoverable_memorization(
            victim_model, tokenizer, args.corpus
        )

    if args.attack in ["untargeted", "all"]:
        print("\n--- Running Untargeted Extraction ---")
        if args.ref_model == args.victim_model and not args.adapter_path:
            ref_model = victim_model
        else:
            ref_model = AutoModelForCausalLM.from_pretrained(
                args.ref_model,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
                device_map={"": 0},
            )
        results["untargeted"] = run_untargeted_extraction(
            victim_model,
            ref_model,
            tokenizer,
            args.corpus,
            args.pii_index,
            num_samples=args.num_samples,
        )

    if args.attack in ["canary", "all"]:
        print("\n--- Running Canary Exposure ---")
        results["canary"] = run_canary_exposure(
            victim_model, tokenizer, args.canaries
        )

    if args.attack in ["targeted", "all"]:
        print("\n--- Running Targeted PII Extraction ---")
        results["targeted"] = run_targeted_extraction(
            victim_model, tokenizer, args.targeted_file
        )

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # A direct A1 invocation produces the documented single-victim schema.
    # scripts/merge_partc.py combines the three separate reports afterwards.
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nCompleted evaluation for {victim_name}.")
    print(f"Results written to: {report_path}")

if __name__ == "__main__":
    main()
