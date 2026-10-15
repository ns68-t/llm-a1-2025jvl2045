import argparse
import time
import os
from pathlib import Path
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    parser = argparse.ArgumentParser()
    # Required by assignment specification
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--target_modules", type=str, required=True)
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--quant", type=str, choices=["none", "4bit"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--results_csv", type=str, required=True)
    
    # Extra arguments for memory management
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--validation_metrics_file", type=str, default="")
    
    return parser.parse_args()

def prepare_tokenized_dataset(raw_dataset, tokenizer, max_length=1024):
    def tokenize_row(batch):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for source, target in zip(batch["source_text"], batch["target_text"]):
            prompt = (
                f"<|im_start|>user\nSanitize the following text:\n{source}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            completion = f"{target}<|im_end|>"

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)

            input_ids = prompt_ids + completion_ids
            attention_mask = [1] * len(input_ids)
            # Mask the prompt so loss is computed on completion tokens only
            labels = [-100] * len(prompt_ids) + completion_ids

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels
        }

    return raw_dataset.map(
        tokenize_row,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing and masking prompts"
    )

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure we use exactly the splits requested
    train_raw = load_dataset("json", data_files=args.train_file, split="train")
    val_raw = load_dataset("json", data_files=args.val_file, split="train")

    train_dataset = prepare_tokenized_dataset(train_raw, tokenizer, args.max_length)
    val_dataset = prepare_tokenized_dataset(val_raw, tokenizer, args.max_length)

    quantization_config = None
    if args.quant == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(model)

    target_modules = [m.strip() for m in args.target_modules.split(",")]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # Required by assignment: explicitly print trainable parameters
    model.print_trainable_parameters()
    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_pct = 100.0 * trainable_params / all_params

    is_sanity_check = args.max_steps > 0
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="no", 
        num_train_epochs=args.epochs if not is_sanity_check else 1,
        max_steps=args.max_steps if is_sanity_check else -1,
        learning_rate=args.learning_rate,
        seed=args.seed,
        fp16=True,
        logging_steps=25 if not is_sanity_check else 5,
        save_strategy="no" if is_sanity_check else "steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    device = torch.cuda.current_device() if torch.cuda.is_available() else 0
    torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    validation = trainer.evaluate()

    train_wallclock_s = end_time - start_time
    
    # Required by assignment: explicitly compute max memory allocated
    peak_gpu_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = {
        "run_id": args.run_id,
        "method": "QLORA" if args.quant == "4bit" else "LORA",
        "target_modules": args.target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "quant": args.quant,
        "trainable_params": trainable_params,
        "total_params": all_params,
        "trainable_pct": trainable_pct,
        "peak_gpu_mem_gb": peak_gpu_mem_gb,
        "train_wallclock_s": train_wallclock_s,
        "tokens_per_s": train_result.metrics.get("train_samples_per_second"),
        "final_train_loss": train_result.metrics.get("train_loss"),
        "val_loss": validation.get("eval_loss"),
        "val_entity_f1": None,
        "val_leak_rate": None,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "validation_metric_status": "requires sanitizer predictions"
    }

    if args.validation_metrics_file:
        with open(args.validation_metrics_file, "r", encoding="utf-8") as f:
            external = __import__("json").load(f)
        metrics["val_entity_f1"] = external["exact"]["micro_f1"]
        metrics["val_leak_rate"] = external["leak_rate"]
        metrics["validation_metric_status"] = "loaded from sanitizer evaluation"

    df = pd.DataFrame([metrics])
    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
    # The required interface appends one completed training run.
    df.to_csv(args.results_csv, mode="a", header=not os.path.exists(args.results_csv), index=False)
    print(f"\n[SUCCESS] Run {args.run_id} finished successfully. Metrics saved to {args.results_csv}")

if __name__ == "__main__":
    main()
