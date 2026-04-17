"""DPG-Bench — 1,065 dense prompts for text-to-image adherence evaluation.

Data: HuggingFace dataset Jialuo21/DPG-Bench (parquet) or local text files.
Repo: https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = str(REPO_ROOT.parent / "DPG-Bench" / "prompts.jsonl")


def load_prompts(n: int, data_path: str | None = None) -> list[dict]:
    """Load DPG-Bench prompts.

    Supports two formats:
    - JSONL: one JSON object per line with a "prompt" field
    - Directory of .txt files: each file contains one prompt
    """
    path = data_path or os.environ.get("DPG_BENCH_DATA", DEFAULT_DATA_PATH)

    if os.path.isdir(path):
        items = []
        txt_files = sorted(Path(path).glob("*.txt"), key=lambda p: p.stem.zfill(10))
        for i, txt_file in enumerate(txt_files):
            if i >= n:
                break
            prompt = txt_file.read_text(encoding="utf-8").strip()
            items.append({"prompt": prompt, "idx": i, "meta": {"file": txt_file.name}})
        return items

    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            items.append({"prompt": row["prompt"], "idx": i, "meta": row})
    return items


DPG_BENCH_CONFIG = {
    "name": "DPG-Bench",
    "short_name": "dpg-bench",
    "default_n_prompts": 1065,
    "load_prompts": load_prompts,
    "data_env_var": "DPG_BENCH_DATA",
    "default_data_path": DEFAULT_DATA_PATH,
}
