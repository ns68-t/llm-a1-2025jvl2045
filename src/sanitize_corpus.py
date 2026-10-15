"""Create the Part C sanitized corpus with the selected Part A adapter.

Every input record is copied verbatim except for ``text``.  This deliberately
does not apply rules for canaries or other special rows: sanitizer failures are
part of the Part C measurement.
"""
import argparse
import json
from pathlib import Path
import random

from sanitize_eval import make_prompt, sha256
from hub_assets import ensure_adapter


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    assert args.batch_size > 0 and args.max_new_tokens > 0
    args.adapter_path = str(ensure_adapter(args.adapter_path))

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    assert torch.cuda.is_available(), "GPU is required to create the sanitized corpus."
    input_path, output_path = Path(args.input_file), Path(args.output_file)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {output_path}; pass --overwrite.")
    records = read_jsonl(input_path)
    assert records and all("doc_id" in row and "text" in row for row in records), "Invalid Part C corpus."
    assert len({(row["doc_id"], row.get("repetition_index", 0)) for row in records}) == len(records), "Duplicate corpus rows."

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()
    stop_ids = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")})
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
        num_return_sequences=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids, use_cache=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("w", encoding="utf-8") as stream:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            prompts = [make_prompt(row["text"]) for row in batch]
            encoded = tokenizer(prompts, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**encoded, generation_config=config)
            continuation = generated[:, encoded["input_ids"].shape[1]:].cpu()
            outputs = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            for row, text in zip(batch, outputs):
                sanitized = dict(row)
                sanitized["text"] = text.strip()
                stream.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
            if (start // args.batch_size + 1) % 100 == 0 or start + len(batch) == len(records):
                print(f"Sanitized {start + len(batch)}/{len(records)} rows", flush=True)
    written = read_jsonl(temporary)
    assert len(written) == len(records)
    assert [(r["doc_id"], r.get("repetition_index", 0)) for r in written] == [(r["doc_id"], r.get("repetition_index", 0)) for r in records]
    temporary.replace(output_path)
    metadata = {
        "input_sha256": sha256(input_path), "output_sha256": sha256(output_path),
        "base_model": args.base_model, "adapter_path": args.adapter_path,
        "adapter_config_sha256": sha256(Path(args.adapter_path) / "adapter_config.json"),
        "seed": args.seed, "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "do_sample": False, "num_beams": 1, "n_rows": len(records),
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved sanitized corpus: {output_path}")


if __name__ == "__main__":
    main()
