#!/usr/bin/env bash
# Fine-tune the local model on the AMD GPU pod, export GGUF Q4, publish to a PUBLIC
# HuggingFace repo. Run this ON the pod (notebooks.amd.com/hackathon).
#
# BEFORE RUNNING: export a HuggingFace WRITE token (hf.co/settings/tokens):
#     export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
# Then:
#     git clone https://github.com/YashasviThakur/amd-tokenoptimizer && \
#       bash amd-tokenoptimizer/finetune/run_on_pod.sh
#
# Fine-tune is minutes on one GPU; building llama.cpp + quantizing is the long pole
# (~10-20 min). When it finishes it prints the HF repo id — send me that string.
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN first:  export HF_TOKEN=hf_...   (write token from hf.co/settings/tokens)}"

REPO_URL="https://github.com/YashasviThakur/amd-tokenoptimizer"

echo "==> [1/6] Repo"
if [ ! -d amd-tokenoptimizer ]; then git clone "$REPO_URL"; fi
cd amd-tokenoptimizer && git pull --ff-only 2>/dev/null || true

echo "==> [2/6] Python deps"
# Pods usually ship a working GPU torch; only install the ROCm build if it's missing.
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "    installing ROCm torch (if this fails, match rocm6.1 to the pod's ROCm version)"
  pip install -q torch --index-url https://download.pytorch.org/whl/rocm6.1
fi
pip install -q -r finetune/requirements.txt huggingface_hub
# peft's LoRA path version-checks torchao and ERRORS if an old one is present
# (Colab ships 0.10; peft wants >0.16). We don't use torchao for LoRA -> remove it.
pip uninstall -y torchao 2>/dev/null || true

echo "==> [3/6] LoRA fine-tune (Qwen2.5-3B-Instruct, 3 epochs) -> finetune/out-merged"
# T4/P100 (Colab/Kaggle free tier) have no bf16 -> auto-fall back to fp16;
# AMD MI / NVIDIA A100 keep bf16. No user thinking required either way.
FP16=""
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_bf16_supported() else 1)" 2>/dev/null || FP16="--fp16"
echo "    precision: ${FP16:-bf16}"
python finetune/train_lora.py --base Qwen/Qwen2.5-3B-Instruct --epochs 3 $FP16

echo "==> [4/6] Build llama.cpp quantizer + convert to GGUF Q4_K_M"
if [ ! -d llama.cpp ]; then git clone https://github.com/ggerganov/llama.cpp; fi
pip install -q -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py finetune/out-merged --outfile tokenopt-3b-f16.gguf --outtype f16
cmake -B llama.cpp/build -S llama.cpp -DGGML_NATIVE=OFF >/dev/null
cmake --build llama.cpp/build --target llama-quantize -j >/dev/null
./llama.cpp/build/bin/llama-quantize tokenopt-3b-f16.gguf tokenopt-3b-q4_k_m.gguf Q4_K_M
ls -lh tokenopt-3b-q4_k_m.gguf

echo "==> [5/6] Publish GGUF to a PUBLIC HuggingFace repo"
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
user = api.whoami()["name"]
repo = f"{user}/tokenopt-3b-gguf"
api.create_repo(repo, private=False, exist_ok=True, repo_type="model")
api.upload_file(path_or_fileobj="tokenopt-3b-q4_k_m.gguf",
                path_in_repo="tokenopt-3b-q4_k_m.gguf", repo_id=repo)
open("HF_REPO.txt", "w").write(repo)
print("PUBLISHED ->", repo)
PY

echo "==> [6/6] DONE — send me this repo id:"
cat HF_REPO.txt
