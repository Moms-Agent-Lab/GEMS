"""WISE — World Knowledge-Informed Semantic Evaluation (1,000 prompts).

Categories: Cultural Common Sense (1-400), Time (401-567), Space (568-700),
            Biology (701-800), Physics (801-900), Chemistry (901-1000).
Data: prompts.json — list of objects with {prompt_id, prompt} or JSONL.
Repo: https://github.com/PKU-YuanGroup/WISE
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = str(REPO_ROOT.parent / "WISE" / "prompts.json")


def load_prompts(n: int, data_path: str | None = None) -> list[dict]:
    path = data_path or os.environ.get("WISE_DATA", DEFAULT_DATA_PATH)

    with open(path, encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):
        data = json.loads(content)
        items = []
        for i, row in enumerate(data):
            if i >= n:
                break
            prompt = row.get("prompt", row.get("text", ""))
            items.append({
                "prompt": prompt,
                "idx": i,
                "meta": row,
            })
        return items

    items = []
    for i, line in enumerate(content.splitlines()):
        if i >= n:
            break
        row = json.loads(line)
        prompt = row.get("prompt", row.get("text", ""))
        items.append({"prompt": prompt, "idx": i, "meta": row})
    return items


WISE_CONFIG = {
    "name": "WISE",
    "short_name": "wise",
    "default_n_prompts": 1000,
    "load_prompts": load_prompts,
    "data_env_var": "WISE_DATA",
    "default_data_path": DEFAULT_DATA_PATH,
}
