
import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_TARGETS = (
    "q_proj,k_proj,v_proj,o_proj,"
    "gate_proj,up_proj,down_proj"
)

def tokenize_json(json_path, tokenizer, split_name):
    dataset = load_dataset(
        "json",
        data_files={split_name: str(json_path)},
    )[split_name]

    def tokenize_fn(examples):
        texts = [text + tokenizer.eos_token for text in examples["text"]]
        return tokenizer(texts, truncation=True, max_length=512)

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

def token_weighted_ppl(model, tokenized_heldout, tokenizer):
    """PPL = exp(total next-token NLL / total valid next-token count)."""
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(
        tokenized_heldout,
        batch_size=8,
        collate_fn=collator,
    )

    device = next(model.parameters()).device
    total_nll = 0.0
    total_next_tokens = 0

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)

            # Count the shifted labels actually used by the causal-LM loss.
            # This remains correct when PAD and EOS share an ID.
            valid_next_tokens = int(
                (batch["labels"][:, 1:] != -100).sum().item()
            )

            if valid_next_tokens > 0:
                total_nll += outputs.loss.item() * valid_next_tokens
                total_next_tokens += valid_next_tokens

    if total_next_tokens == 0:
        raise RuntimeError("No valid next-token labels in held-out data.")

    return math.exp(total_nll / total_next_tokens), total_next_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--output_dir", required=True)
    # Optional convenience label: the required A1 interface does not include it.
    parser.add_argument("--victim_name", choices=["V-raw", "V-san"], default=None)

    parser.add_argument("--target_modules", default=DEFAULT_TARGETS)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Identical to V-raw.
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    corpus_path = Path(args.corpus)
    heldout_path = Path(args.heldout)
    output_dir = Path(args.output_dir)
    victim_name = args.victim_name
    if victim_name is None:
        lowered = output_dir.name.lower()
        victim_name = "V-raw" if "raw" in lowered else "V-san" if "san" in lowered else output_dir.name

    assert torch.cuda.is_available(), "Enable GPU in Kaggle."
    assert corpus_path.exists(), f"Missing corpus: {corpus_path}"
    assert heldout_path.exists(), f"Missing held-out file: {heldout_path}"

    corpus_rows = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    required = {
        "doc_id", "text", "n_reps",
        "is_canary", "canary_id", "repetition_index",
    }
    assert corpus_rows, "Victim-training corpus contains no rows."
    assert all(required.issubset(row) for row in corpus_rows), "Metadata missing."
    assert all(str(row["text"]).strip() for row in corpus_rows), "Empty text found."

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()

    print(f"Loading V-base: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Same Base loading configuration as V-raw.
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    heldout_tokens = tokenize_json(heldout_path, tokenizer, "heldout")

    # V-base PPL before adding LoRA or training.
    print("Computing V-base held-out token-weighted PPL...")
    vbase_ppl, heldout_next_tokens = token_weighted_ppl(
        base_model, heldout_tokens, tokenizer
    )
    print(f"V-base PPL: {vbase_ppl:.6f}")

    targets = [module.strip() for module in args.target_modules.split(",")]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    train_tokens = tokenize_json(corpus_path, tokenizer, "train")
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Same V-raw TrainingArguments.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_strategy="no",
        fp16=True,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokens,
        data_collator=collator,
    )

    print(f"Training {victim_name}...")
    train_result = trainer.train()

    print(f"Computing {victim_name} held-out token-weighted PPL...")
    trained_ppl, _ = token_weighted_ppl(model, heldout_tokens, tokenizer)
    print(f"{victim_name} PPL: {trained_ppl:.6f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()

    metrics = {
        "victim": victim_name,
        "base_model": args.model,
        "corpus_rows": len(corpus_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": targets,
        "final_train_loss": train_result.training_loss,
        "vbase_heldout_token_weighted_ppl": vbase_ppl,
        "trained_heldout_token_weighted_ppl": trained_ppl,
        "heldout_next_tokens": heldout_next_tokens,
    }

    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nCOMPLETE")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {victim_name} adapter: {output_dir}")

if __name__ == "__main__":
    main()
