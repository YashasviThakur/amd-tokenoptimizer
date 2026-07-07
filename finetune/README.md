# Fine-tuning kit — the podium lever

**Why:** Track 1 ranks by *raw Fireworks tokens*, and local tokens are free. So the
single biggest lever is making the **local model answer more tasks correctly on
its own** — every task it nails locally is 0 tokens. Fine-tuning a small model on
the 8 categories does exactly that, and it's explicitly allowed by the rules.

This kit fine-tunes a small base model on 200 in-format examples (all 8
categories), exports it to GGUF, and bundles it into the agent container in place
of the stock `qwen2.5-coder:3b`.

## Prereqs
- The **AMD GPU pod** (48 GB) from `notebooks.amd.com/hackathon` (ROCm PyTorch).
- Keep the base **small (~1.5–3B)** so the exported GGUF stays runnable on the
  eval VM (a 3B q4 GGUF is ~2 GB).

## Steps

```bash
# 0) on the AMD GPU pod, install ROCm torch first, then deps
pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch
pip install -r finetune/requirements.txt

# 1) build the chat-format training file (uses the agent's own prompts)
python finetune/build_jsonl.py            # -> finetune/train.jsonl (200 rows)

# 2) LoRA fine-tune + merge  (~minutes on the 48GB GPU)
python finetune/train_lora.py --base Qwen/Qwen2.5-3B-Instruct
#   -> finetune/out-merged/  (a full HF model)

# 3) export to GGUF + quantize (needs llama.cpp)
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && pip install -r requirements.txt
python convert_hf_to_gguf.py ../finetune/out-merged --outfile tokenopt.gguf --outtype f16
./llama-quantize tokenopt.gguf tokenopt-q4.gguf q4_k_m     # ~2GB

# 4) make it an Ollama model
printf 'FROM ./tokenopt-q4.gguf\n' > Modelfile
ollama create tokenoptimizer-local -f Modelfile
```

## Bundle into the container
In `docker/agent.Dockerfile`, replace the model bake:

```dockerfile
ENV LOCAL_MODEL=tokenoptimizer-local
COPY tokenopt-q4.gguf /models/tokenopt-q4.gguf
RUN printf 'FROM /models/tokenopt-q4.gguf\n' > /models/Modelfile && \
    (ollama serve & sleep 6 && ollama create tokenoptimizer-local -f /models/Modelfile)
```

Then let CI rebuild + push `:latest`. The router is unchanged — it just now has a
much stronger free local model, so it escalates less and spends fewer tokens
**while staying above the accuracy gate**.

## Growing the dataset
`build_jsonl.py` reads `finetune/dataset_raw.json` (200 examples). Add more
categories/examples there (same shape) and re-run — more, higher-quality
in-format data = a better local model = fewer tokens.
