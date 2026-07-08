# Fine-tuning the local model — toward safe 0-token operation

Local-model answers are **free** (0 Fireworks tokens). The higher the local model's
per-category accuracy, the more tasks we can answer locally — and the closer we get
to running **fully local at 0 tokens** while still clearing the accuracy gate. This
kit LoRA-fine-tunes `Qwen2.5-3B-Instruct` on the 8 Track-1 categories, in our exact
output format, then exports a GGUF to bundle in the agent image.

The training set (`train.jsonl`, 400 examples, 50/category) was generated and
**adversarially verified** for correctness, and formatted with the agent's own
system prompts so the tuned model answers *our* prompts.

## 0. Where to run
On the **AMD GPU pod** (ROCm). A single 3B LoRA run is quick (minutes on one GPU).

## 1. Setup
```bash
git clone https://github.com/YashasviThakur/amd-tokenoptimizer && cd amd-tokenoptimizer
# ROCm PyTorch — match the pod's ROCm version:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
pip install -r finetune/requirements.txt
```

## 2. Train (LoRA) + merge
```bash
python finetune/train_lora.py --base Qwen/Qwen2.5-3B-Instruct --epochs 3
# -> finetune/out-merged   (full HF model with the LoRA merged in)
```
Tips: pass `--fp16` if bf16 is unsupported; bump `--epochs` to 4-5 if underfitting.

## 3. Export to GGUF (Q4) for llama-cpp
```bash
git clone https://github.com/ggerganov/llama.cpp && pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py finetune/out-merged --outfile tokenopt-3b-f16.gguf --outtype f16
# build llama.cpp then quantize to Q4_K_M (fits the 4GB grading box)
cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build --target llama-quantize -j
./llama.cpp/build/bin/llama-quantize tokenopt-3b-f16.gguf tokenopt-3b-q4_k_m.gguf Q4_K_M
```

## 4. Host it, then bundle it in the image
The GGUF (~1.9GB) is too big for git. Upload it to a **public HF repo** you own:
```bash
huggingface-cli upload <you>/tokenopt-3b-gguf tokenopt-3b-q4_k_m.gguf
```
Then point the image at it — edit `docker/agent.Dockerfile`:
```dockerfile
RUN python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('<you>/tokenopt-3b-gguf','tokenopt-3b-q4_k_m.gguf', local_dir='/models')"
ENV LOCAL_MODEL_PATH=/models/tokenopt-3b-q4_k_m.gguf
```
Push to `main`; CI rebuilds and pushes `:latest`.

## 5. MEASURE before trusting it (do not skip)
Fine-tuning helps most on format-heavy categories (NER, sentiment, factual). Verify
it actually raised accuracy before leaning on it:
```bash
# with the new GGUF at LOCAL_MODEL_PATH:
python -m eval.harness --tasks eval/datasets/stress_tasks.json \
                       --expected eval/datasets/stress_expected.json
```
Compare fully-local accuracy vs the 83% baseline (Qwen2.5-3B stock). Only lower the
escalation (toward 0 tokens) once the leaderboard confirms we clear the gate with
margin — never flip to fully-local on a blind bet.

## Notes
- Keep the **stock** Qwen2.5-3B as the fallback: if the fine-tune doesn't beat 83%,
  don't ship it. Fine-tuning a 3B mainly fixes format/recall, not deep multi-step
  reasoning (that's what Fireworks escalation is for).
- Want a bigger jump? Regenerate `train.jsonl` with more examples (the generator
  workflow), or A/B a stronger reasoning base model that still fits 4GB.
