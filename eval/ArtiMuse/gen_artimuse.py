import os
import json
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm

import torch
from agent.GEMS import GEMS

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("metadata_file", type=str, help="JSONL file containing lines of metadata for each prompt")
    parser.add_argument("--outdir", type=str, default="outputs", help="Base directory to write results to")
    parser.add_argument("--name", type=str, required=True, help="Experiment name, used for sub-directory")
    
    parser.add_argument("--agent", type=str, default="gems", help="Agent strategy (currently set to gems)")
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=256, help="Number of concurrent multiprocessing workers")
    parser.add_argument("--gen_url", type=str, default="")
    parser.add_argument("--mllm_url", type=str, default="")
    
    return parser.parse_args()


def get_agent(args):
    gen_url = args.gen_url
    mllm_url = args.mllm_url
    max_iterations = getattr(args, 'max_iterations', 8)

    return GEMS(gen_url=gen_url, mllm_url=mllm_url, max_iterations=max_iterations)


def agent_worker(rank, jobs, args, return_dict):    
    agent = get_agent(args)
    local_mapping = {}
    pbar = tqdm(jobs, desc=f"Worker {rank}", position=rank, leave=False)
    
    experiment_dir = os.path.join(args.outdir, args.name)
    
    for global_idx, metadata in pbar:
        try:
            prompt = metadata['prompt']
            
            meta_file_path = os.path.join(experiment_dir, f"{global_idx:05d}.json")
            with open(meta_file_path, "w", encoding="utf-8") as fp:
                json.dump(metadata, fp, ensure_ascii=False)
            
            img_data = agent.run(metadata)
            
            img_file_path = os.path.join(experiment_dir, f"{global_idx:05d}.png")
            with open(img_file_path, "wb") as f:
                f.write(img_data)
            
            local_mapping[prompt] = img_file_path
            
        except Exception as e:
            print(f"\nWorker {rank} error processing item {global_idx}: {e}")
            
    return_dict[rank] = local_mapping


def main():
    args = parse_args()
    
    OUTPUT_DIR = os.path.join(args.outdir, args.name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    MAPPING_FILE = os.path.join(OUTPUT_DIR, "image_paths.json")
    
    all_data_with_ids = []
    with open(args.metadata_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
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

    chunks = [to_process[i::args.num_workers] for i in range(args.num_workers)]

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    for rank in range(args.num_workers):
        if rank < len(chunks) and len(chunks[rank]) > 0:
            p = mp.Process(target=agent_worker, args=(rank, chunks[rank], args, return_dict))
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
        
    print(f"\nTask completed. New items: {new_generated_count}")
    print(f"Total progress: {len(final_mapping)} / {total_count}")
    print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()