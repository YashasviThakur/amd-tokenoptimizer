"""LoRA supervised fine-tune of the local model on the 8 Track-1 categories.

Run on the AMD GPU pod (ROCm PyTorch). Teaches Qwen2.5-3B to answer each category
correctly AND in our exact output format, so at eval time MORE tasks are answered
locally for 0 Fireworks tokens (raising fully-local accuracy toward safe 0-token
operation). Produces a merged model ready for GGUF export.

    pip install -r finetune/requirements.txt      # torch (ROCm) installed separately
    python finetune/train_lora.py --base Qwen/Qwen2.5-3B-Instruct
    # -> finetune/out-merged  (then convert to GGUF: see finetune/README.md)

Uses trl's high-level SFTTrainer: it consumes the conversational dataset
({"messages":[...]}) directly and applies the chat template itself, so this stays
robust across trl/transformers versions. If bf16 isn't supported, pass --fp16.
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="finetune/train.jsonl")
    ap.add_argument("--out", default="finetune/out")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--fp16", action="store_true", help="use fp16 instead of bf16")
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.data, split="train")
    print(f"training examples: {len(ds)}")

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    sft = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        bf16=not args.fp16,
        fp16=args.fp16,
    )

    trainer = SFTTrainer(model=args.base, args=sft, train_dataset=ds, peft_config=lora)
    trainer.train()

    adapter = args.out + "-adapter"
    trainer.save_model(adapter)
    print("adapter saved ->", adapter)

    # merge LoRA into the base so it can be exported to GGUF
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    merged = AutoPeftModelForCausalLM.from_pretrained(
        adapter, torch_dtype=torch.float16).merge_and_unload()
    merged.save_pretrained(args.out + "-merged")
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out + "-merged")
    print("MERGED MODEL ->", args.out + "-merged", "(export to GGUF next; see finetune/README.md)")


if __name__ == "__main__":
    main()
