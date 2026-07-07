"""LoRA supervised fine-tune of a small base model on the 8 Track-1 categories.

Run on the AMD GPU pod (ROCm PyTorch). Teaches a small model to answer each
category correctly AND in our exact output format, so at eval time more tasks
are answered locally for 0 tokens. Produces a merged model ready for GGUF export.

    pip install -r finetune/requirements.txt
    python finetune/build_jsonl.py
    python finetune/train_lora.py --base Qwen/Qwen2.5-3B-Instruct --data finetune/train.jsonl

Notes:
- Keep the base small (~1.5-3B) so the exported GGUF stays runnable on the eval VM.
- trl's SFT API shifts across versions; if SFTConfig rejects an arg, move it to
  SFTTrainer(...) or drop it — the LoRA config + chat formatting are the core.
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="finetune/train.jsonl")
    ap.add_argument("--out", default="finetune/out")
    ap.add_argument("--epochs", type=float, default=3.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto")

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)},
                remove_columns=ds.column_names)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-4, bf16=True, logging_steps=10, save_strategy="epoch",
        max_seq_length=1024, packing=False, dataset_text_field="text")

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, tokenizer=tok)
    trainer.train()

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.out + "-merged")
    tok.save_pretrained(args.out + "-merged")
    print("Merged model saved to", args.out + "-merged", "-> export to GGUF next (see finetune/README.md)")


if __name__ == "__main__":
    main()
