"""OneIG-Bench (English) — omni-dimensional evaluation for image generation.

Categories: Anime & Stylization (245), Portrait (244), General Object (206),
            Text Rendering (200), Knowledge & Reasoning (225).
Data: OneIG-Bench.json with fields {category, id, prompt_en, type, prompt_length, class}.
Repo: https://github.com/OneIG-Bench/OneIG-Benchmark
HF:   https://huggingface.co/datasets/OneIG-Bench/OneIG-Bench
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = str(REPO_ROOT.parent / "OneIG-Bench" / "OneIG-Bench.json")


def load_prompts(n: int, data_path: str | None = None) -> list[dict]:
    path = data_path or os.environ.get("ONEIG_EN_DATA", DEFAULT_DATA_PATH)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for i, row in enumerate(data):
        if i >= n:
            break
        items.append({
            "prompt": row["prompt_en"],
            "idx": i,
            "meta": row,
        })
    return items


ONEIG_EN_CONFIG = {
    "name": "OneIG-Bench (EN)",
    "short_name": "oneig-en",
    "default_n_prompts": 1120,
    "load_prompts": load_prompts,
    "data_env_var": "ONEIG_EN_DATA",
    "default_data_path": DEFAULT_DATA_PATH,
}
