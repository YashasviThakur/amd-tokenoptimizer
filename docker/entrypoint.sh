#!/usr/bin/env bash
# Start the local model server, wait until it's ready, then run the batch agent.
set -e

nohup ollama serve >/tmp/ollama.log 2>&1 &

# wait for the local OpenAI-compatible endpoint (must be ready well within 60s)
for i in $(seq 1 50); do
  if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# warm the model so the first task isn't slow
ollama run "${LOCAL_MODEL:-gemma2:2b}" "ok" >/dev/null 2>&1 || true

exec python -m agent.main
