"""GenEval2 — 800 compositional text-to-image prompts (Meta FAIR).

Data: geneval2_data.jsonl with fields {prompt, atom_count, vqa_list, skills}.
Repo: https://github.com/facebookresearch/GenEval2
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = str(REPO_ROOT.parent / "GenEval2" / "geneval2_data.jsonl")


def load_prompts(n: int, data_path: str | None = None) -> list[dict]:
    path = data_path or os.environ.get("GENEVAL2_DATA", DEFAULT_DATA_PATH)
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            items.append({"prompt": row["prompt"], "idx": i, "meta": row})
    return items


GENEVAL2_CONFIG = {
    "name": "GenEval2",
    "short_name": "geneval2",
    "default_n_prompts": 800,
    "load_prompts": load_prompts,
    "data_env_var": "GENEVAL2_DATA",
    "default_data_path": DEFAULT_DATA_PATH,
}
