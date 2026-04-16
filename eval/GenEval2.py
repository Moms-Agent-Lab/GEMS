"""GenEval2 benchmark runner.

Supports two agent backends; pick via ``--agent``:

* ``gems``      — original HTTP line (unchanged; default behaviour
                  preserved bit-for-bit).
* ``comfygems`` — new ComfyUI line.  Extra flags:
                  ``--model``, ``--comfyui_servers``,
                  ``--workers_per_server``.

Examples
--------

Original HTTP line (unchanged from before)::

    python eval/GenEval2.py --name my_run --agent gems --max_iterations 5

ComfyUI line, 1 ComfyUI server × 2 client workers::

    python eval/GenEval2.py \\
        --name my_run_comfy --agent comfygems \\
        --model z-image-turbo \\
        --comfyui_servers 127.0.0.1:8188 \\
        --workers_per_server 2 \\
        --max_iterations 5

ComfyUI line, fan out across 4 ComfyUI servers::

    python eval/GenEval2.py \\
        --name my_run_multi --agent comfygems \\
        --model qwen-image-2512 \\
        --comfyui_servers host1:8188,host2:8188,host3:8188,host4:8188
"""

import os
import sys
import json

# Make `from agent.*` resolvable regardless of CWD (and, crucially, inside
# spawn-started child processes whose sys.path[0] is this script's dir).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch.multiprocessing as mp
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument(
    "--agent",
    type=str,
    required=True,
    choices=("gems", "comfygems"),
    help="Which image-generation backend to use.",
)
parser.add_argument("--max_iterations", type=int, default=5)
parser.add_argument("--max_nodes", type=int, default=5)

# --- comfygems-only flags (ignored when --agent=gems) ---
parser.add_argument(
    "--model",
    type=str,
    default="qwen-image-2512",
    help="[comfygems] qwen-image-2512 / z-image-turbo / flux-klein-9b / longcat-image",
)
parser.add_argument(
    "--comfyui_servers",
    type=str,
    default=os.environ.get("COMFYUI_SERVER", "127.0.0.1:8188"),
    help="[comfygems] comma-separated host:port list.",
)
parser.add_argument(
    "--workers_per_server",
    type=int,
    default=1,
    help="[comfygems] client workers PER ComfyUI server.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="[comfygems] fix the ComfyUI sampler seed.",
)
parser.add_argument(
    "--workflow_timeout",
    type=int,
    default=600,
    help="[comfygems] seconds to wait for each ComfyUI job.",
)
args = parser.parse_args()


# Original defaults kept unchanged.  These only apply to --agent gems so the
# legacy invocation `python eval/GenEval2.py --name ... --agent gems ...`
# continues to produce identical output.
NUM_WORKERS = 2
DATA_PATH = "eval/GenEval2_data/geneval2_data.jsonl"
OUTPUT_DIR = os.path.join("eval/GenEval2_data/results", args.name)
TRACE_DIR = os.path.join("eval/GenEval2_data/results", args.name + "_traces")
MAPPING_FILE = os.path.join(OUTPUT_DIR, "image_paths.json")
gen_url = "http://localhost:8000/generate"
mllm_url = ""
max_iterations = args.max_iterations


def _build_agent(rank: int, worker_cfg: dict):
    """Instantiate either GEMS or ComfyGEMS based on *worker_cfg['agent']*.

    Imports are local so a missing dependency on one line (e.g. no
    ComfyUI deps) doesn't block the other line.
    """
    if worker_cfg["agent"] == "gems":
        from agent.GEMS import GEMS
        return GEMS(
            gen_url=worker_cfg["gen_url"],
            mllm_url=worker_cfg["mllm_url"],
            max_iterations=worker_cfg["max_iterations"],
        )

    if worker_cfg["agent"] == "comfygems":
        from agent.comfy_gems import ComfyGEMS
        server = worker_cfg["servers"][rank % len(worker_cfg["servers"])]
        print(f"[Worker {rank}] ComfyGEMS → {server} (model={worker_cfg['model']})")
        return ComfyGEMS(
            model=worker_cfg["model"],
            comfyui_server=server,
            max_iterations=worker_cfg["max_iterations"],
            seed=worker_cfg["seed"],
            workflow_timeout=worker_cfg["workflow_timeout"],
            # workflow JSONs go under this prompt's trace dir — set per-prompt below
            workflow_log_dir=None,
        )

    raise ValueError(f"unknown agent: {worker_cfg['agent']}")


def agent_worker(rank, jobs, return_dict, worker_cfg=None):
    # Backward compat: legacy callsite passed no worker_cfg (--agent gems).
    if worker_cfg is None:
        worker_cfg = {
            "agent": "gems",
            "gen_url": gen_url,
            "mllm_url": mllm_url,
            "max_iterations": max_iterations,
        }

    agent = _build_agent(rank, worker_cfg)

    local_mapping = {}
    pbar = tqdm(jobs, desc=f"Worker {rank}", position=rank, leave=False)

    for global_idx, item in pbar:
        try:
            prompt = item['prompt']

            # For comfygems: dump every submitted workflow into this prompt's
            # trace dir.  Resets per prompt so numbering is local.
            if worker_cfg["agent"] == "comfygems":
                prompt_dir = os.path.join(TRACE_DIR, f"prompt_{global_idx:05d}")
                os.makedirs(prompt_dir, exist_ok=True)
                agent.workflow_log_dir = os.path.join(prompt_dir, "workflows")
                os.makedirs(agent.workflow_log_dir, exist_ok=True)
                agent._workflow_counter = 0

            result = agent.run_with_trace(item)

            img_name = f"img_idx_{global_idx:05d}.png"
            img_path = os.path.join(OUTPUT_DIR, img_name)
            with open(img_path, "wb") as f:
                f.write(result["best_image"])

            prompt_dir = os.path.join(TRACE_DIR, f"prompt_{global_idx:05d}")
            os.makedirs(prompt_dir, exist_ok=True)

            for round_idx, img_bytes in enumerate(result["all_images"], start=1):
                round_img_path = os.path.join(prompt_dir, f"round_{round_idx}.png")
                with open(round_img_path, "wb") as f:
                    f.write(img_bytes)

            best_img_path = os.path.join(prompt_dir, "best.png")
            with open(best_img_path, "wb") as f:
                f.write(result["best_image"])

            trace_path = os.path.join(prompt_dir, "trace.json")
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(result["trace"], f, indent=2, ensure_ascii=False)

            local_mapping[prompt] = img_path
        except Exception as e:
            print(f"\nWorker {rank} error processing item {global_idx}: {e}")

    return_dict[rank] = local_mapping


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TRACE_DIR, exist_ok=True)

    all_data_with_ids = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            all_data_with_ids.append((idx, json.loads(line)))

    total_count = len(all_data_with_ids)
    print(f"Total prompts in dataset: {total_count}")

    existing_mapping = {}
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                existing_mapping = json.load(f)
            print(f"Found existing mapping with {len(existing_mapping)} records.")
        except Exception as e:
            print(f"Error reading mapping file: {e}. Starting fresh.")

    to_process = []
    for global_idx, item in all_data_with_ids:
        if item['prompt'] not in existing_mapping:
            to_process.append((global_idx, item))

    if not to_process:
        print("All items have been processed already.")
        return

    print(f"Items remaining to process: {len(to_process)}")

    # Decide worker topology based on --agent.
    if args.agent == "gems":
        num_workers = NUM_WORKERS
        worker_cfg = {
            "agent": "gems",
            "gen_url": gen_url,
            "mllm_url": mllm_url,
            "max_iterations": max_iterations,
        }
    else:  # comfygems
        servers = [s.strip() for s in args.comfyui_servers.split(",") if s.strip()]
        if not servers:
            raise SystemExit("--comfyui_servers is empty")
        num_workers = max(1, args.workers_per_server) * len(servers)
        worker_cfg = {
            "agent": "comfygems",
            "servers": servers,
            "model": args.model,
            "max_iterations": max_iterations,
            "seed": args.seed,
            "workflow_timeout": args.workflow_timeout,
        }
        print(
            f"[comfygems] servers={servers} "
            f"workers_per_server={args.workers_per_server} "
            f"total_workers={num_workers}"
        )

    chunks = [to_process[i::num_workers] for i in range(num_workers)]

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for rank in range(num_workers):
        if rank < len(chunks) and len(chunks[rank]) > 0:
            p = mp.Process(
                target=agent_worker,
                args=(rank, chunks[rank], return_dict, worker_cfg),
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    final_mapping = existing_mapping.copy()
    new_generated_count = 0
    for rank_map in return_dict.values():
        final_mapping.update(rank_map)
        new_generated_count += len(rank_map)

    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_mapping, f, indent=4, ensure_ascii=False)

    print(f"\n[Task Completed] New images added this round: {new_generated_count}.")
    print(f"Current overall progress: {len(final_mapping)} / {total_count}")
    print(f"Mapping file updated to: {MAPPING_FILE}")


if __name__ == "__main__":
    main()
