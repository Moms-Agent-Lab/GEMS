"""OneIG-Bench (Chinese) — omni-dimensional evaluation for image generation.

Same categories as OneIG-EN plus Multilingualism (200 prompts).
Data: OneIG-Bench-ZH.json with fields {category, id, prompt_en (Chinese despite the key name)}.
Repo: https://github.com/OneIG-Bench/OneIG-Benchmark
HF:   https://huggingface.co/datasets/OneIG-Bench/OneIG-Bench
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = str(REPO_ROOT.parent / "OneIG-Bench" / "OneIG-Bench-ZH.json")


def load_prompts(n: int, data_path: str | None = None) -> list[dict]:
    path = data_path or os.environ.get("ONEIG_ZH_DATA", DEFAULT_DATA_PATH)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for i, row in enumerate(data):
        if i >= n:
            break
        prompt = row.get("prompt_en", row.get("prompt_zh", ""))
        items.append({
            "prompt": prompt,
            "idx": i,
            "meta": row,
        })
    return items


ONEIG_ZH_CONFIG = {
    "name": "OneIG-Bench (ZH)",
    "short_name": "oneig-zh",
    "default_n_prompts": 1320,
    "load_prompts": load_prompts,
    "data_env_var": "ONEIG_ZH_DATA",
    "default_data_path": DEFAULT_DATA_PATH,
}
