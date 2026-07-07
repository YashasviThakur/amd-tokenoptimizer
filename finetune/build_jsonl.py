"""Turn the generated per-category examples into a chat-format SFT JSONL.

Each training row uses the SAME system prompt the agent sends at inference
(`system_for(category)`), so the fine-tuned model learns to answer in exactly
the format we use at eval time — correct and minimal, which is what wins on
raw token count.

    python finetune/build_jsonl.py            # dataset_raw.json -> train.jsonl
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agent.prompts import system_for  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "dataset_raw.json")
OUT = os.path.join(HERE, "train.jsonl")


def main():
    data = json.load(open(RAW, encoding="utf-8"))
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for group in data:
            cat = group["category"]
            for ex in group.get("examples", []):
                if not ex.get("prompt") or not ex.get("output"):
                    continue
                row = {"messages": [
                    {"role": "system", "content": system_for(cat)},
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["output"]},
                ]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} chat examples -> {OUT}")


if __name__ == "__main__":
    main()
