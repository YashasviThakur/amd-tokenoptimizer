# Running live on AMD Developer Cloud (ROCm)

The whole point of Track 1 is that the local tier runs **on an AMD GPU**. Here is
the shortest reliable path to a live demo where `rocm-smi` and the cockpit show
the AMD GPU lighting up on local answers.

## 1. Get a GPU instance
- Provision an AMD Instinct (e.g. MI300X) instance on **AMD Developer Cloud**.
- Verify the GPU is visible:
  ```bash
  rocm-smi
  ```

## 2. Serve Gemma on ROCm (the local tier)
Option A — Docker (recommended):
```bash
export HF_TOKEN=hf_...            # needed to pull Gemma weights
export FIREWORKS_API_KEY=fw_...
cd docker && docker compose up --build
```
Option B — bare vLLM already installed on the ROCm image:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-4b-it --dtype float16 --max-model-len 8192 --port 8000
```
If the 4B model is slow to bring up, fall back to `google/gemma-3-1b-it` or
`google/gemma-2-2b-it` — **still on the AMD GPU**, which is all that matters.

## 3. Point the gateway at it
```bash
export TOKENOPT_MODE=live
export LOCAL_BASE_URL=http://localhost:8000/v1
export LOCAL_MODEL=google/gemma-3-4b-it
export FIREWORKS_API_KEY=fw_...
python run.py
```
Open `http://<instance-ip>:4321`. In `live` mode the cockpit reads the **real**
GPU via `rocm-smi`.

## 4. The money shot for the video
Put a terminal running `watch -n1 rocm-smi` next to the browser cockpit. Send a
simple query — the AMD GPU utilization jumps, the answer returns with **0 remote
tokens**, and the savings counter ticks up. Then send a hard query and watch it
escalate to Fireworks. Same workload, a fraction of the cost.

## Tips
- Budget the **$50 Fireworks credit**: keep `max_tokens` modest and pre-run the
  remote-path queries once so the cache covers them if you hit a rate limit.
- Keep a recorded backup clip of the live GPU spike in case the network flakes
  during judging.
