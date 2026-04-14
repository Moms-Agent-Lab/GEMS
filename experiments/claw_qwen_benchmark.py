#!/usr/bin/env python3
"""
ComfyClaw benchmark with Qwen Image 2512 as base model.
Uses the official Qwen Image pipeline: UNETLoader + CLIPLoader + VAELoader +
ModelSamplingAuraFlow + Lightning LoRA + EmptySD3LatentImage.
"""
import copy, json, logging, os, sys, time

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("qwen_bench")

# ── Config ────────────────────────────────────────────────────────────────
GENEVAL2_PATH = os.environ.get("GENEVAL2_DATA", str(REPO_ROOT.parent / "GenEval2" / "geneval2_data.jsonl"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(REPO_ROOT.parent / "benchmark_qwen_10"))
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
COMFYUI_ADDR = "127.0.0.1:8188"
LLM_MODEL = "anthropic/claude-sonnet-4-5"
SKILLS_DIR = str(REPO_ROOT / "comfyclaw" / "skills")

N_PROMPTS = 10
MAX_ITERATIONS = 2
WARM_START = True

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# ── Qwen Image 2512 base workflow (API format) ───────────────────────────
# Faithfully reconstructed from image_qwen_Image_2512.json
#
# Pipeline: UNETLoader → LoraLoaderModelOnly → ModelSamplingAuraFlow → KSampler
#           CLIPLoader → CLIPTextEncode (pos) + CLIPTextEncode (neg)
#           VAELoader → VAEDecode
#           EmptySD3LatentImage → KSampler
#
QWEN_BASE_WORKFLOW = {
    "1": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "qwen_image_2512_fp8_e4m3fn.safetensors",
            "weight_dtype": "default",
        },
        "_meta": {"title": "UNET Loader"},
    },
    "2": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image",
            "device": "default",
        },
        "_meta": {"title": "CLIP Loader"},
    },
    "3": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "qwen_image_vae.safetensors",
        },
        "_meta": {"title": "VAE Loader"},
    },
    "4": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors",
            "strength_model": 1.0,
        },
        "_meta": {"title": "Lightning LoRA"},
    },
    "5": {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {
            "model": ["4", 0],
            "shift": 3.1,
        },
        "_meta": {"title": "Model Sampling AuraFlow"},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["2", 0],
            "text": "",
        },
        "_meta": {"title": "Positive Prompt"},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["2", 0],
            "text": "low resolution, low quality, deformed limbs, deformed fingers, oversaturated, wax figure, faceless, overly smooth, AI feel, cluttered composition, blurry text, distorted",
        },
        "_meta": {"title": "Negative Prompt"},
    },
    "8": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
            "width": 1328,
            "height": 1328,
            "batch_size": 1,
        },
        "_meta": {"title": "Empty Latent (1328x1328)"},
    },
    "9": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["5", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["8", 0],
            "seed": 42,
            "steps": 4,
            "cfg": 4.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        },
        "_meta": {"title": "KSampler"},
    },
    "10": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["9", 0],
            "vae": ["3", 0],
        },
        "_meta": {"title": "VAE Decode"},
    },
    "11": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["10", 0],
            "filename_prefix": "QwenClaw",
        },
        "_meta": {"title": "Save Image"},
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────

RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

def load_results() -> list[dict]:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []

def save_results(results: list[dict]):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

def load_prompts(n: int) -> list[dict]:
    items = []
    with open(GENEVAL2_PATH) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            items.append(json.loads(line))
    return items


def run_one(prompt: str, idx: int) -> dict:
    from comfyclaw.harness import ClawHarness, HarnessConfig

    t0 = time.time()
    cfg = HarnessConfig(
        api_key=API_KEY,
        server_address=COMFYUI_ADDR,
        model=LLM_MODEL,
        max_iterations=MAX_ITERATIONS,
        success_threshold=0.95,
        sync_port=0,
        image_model=None,  # don't override — workflow already has correct model
        stage_gated=True,
        skills_dir=SKILLS_DIR,
        max_nodes=20,
        baseline_first=WARM_START,
    )

    init_wf = copy.deepcopy(QWEN_BASE_WORKFLOW) if WARM_START else {}
    harness = ClawHarness.from_workflow_dict(init_wf, cfg)
    with harness:
        image_bytes = harness.run(prompt)

    best_score = 0.0
    baseline_score = 0.0
    for entry in harness.evolution_log.entries:
        if entry.verifier_score is not None:
            best_score = max(best_score, entry.verifier_score)
    for attempt in harness.memory.attempts:
        if attempt.verifier_score > best_score:
            best_score = attempt.verifier_score
        if attempt.iteration == 0:
            baseline_score = attempt.verifier_score

    passed_all, failed_all = [], []
    for attempt in harness.memory.attempts:
        if attempt.verifier_score == best_score:
            passed_all = attempt.passed
            failed_all = attempt.failed
            break

    img_path = None
    if image_bytes:
        img_path = os.path.join(OUTPUT_DIR, "images", f"qwen_{idx:02d}.png")
        with open(img_path, "wb") as f:
            f.write(image_bytes)

    elapsed = time.time() - t0
    node_count = (
        harness.evolution_log.entries[-1].node_count_after
        if harness.evolution_log.entries else 11
    )

    return {
        "idx": idx,
        "prompt": prompt,
        "baseline_score": baseline_score,
        "best_score": best_score,
        "passed": passed_all,
        "failed": failed_all,
        "elapsed_s": round(elapsed, 1),
        "node_count": node_count,
        "iterations": len(harness.evolution_log.entries),
        "image_path": img_path,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("ComfyClaw + Qwen Image 2512 Benchmark — %d prompts", N_PROMPTS)
    log.info("LLM: %s  Diffusion: Qwen Image 2512 (FP8 + Lightning LoRA)", LLM_MODEL)
    log.info("Max iterations: %d  Warm-start: %s", MAX_ITERATIONS, WARM_START)
    log.info("=" * 70)

    prompts = load_prompts(N_PROMPTS)
    existing = load_results()
    completed_idx = {r["idx"] for r in existing if r.get("best_score", -1) >= 0}

    log.info("Prompts: %d total, %d already completed", len(prompts), len(completed_idx))
    results = list(existing)

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        if i in completed_idx:
            r = next(r for r in results if r["idx"] == i)
            log.info("[%2d/%d] CACHED  base=%.3f best=%.3f  %s",
                     i + 1, N_PROMPTS, r.get("baseline_score", 0), r["best_score"], prompt)
            continue

        log.info("")
        log.info("[%2d/%d] RUNNING  %s", i + 1, N_PROMPTS, prompt)
        try:
            r = run_one(prompt, i)
            results.append(r)
            save_results(results)
            log.info("[%2d/%d] DONE  base=%.3f best=%.3f  time=%ds  nodes=%d",
                     i + 1, N_PROMPTS, r["baseline_score"], r["best_score"],
                     r["elapsed_s"], r["node_count"])
        except Exception as exc:
            log.error("[%2d/%d] FAILED: %s", i + 1, N_PROMPTS, exc, exc_info=True)
            results.append({
                "idx": i, "prompt": prompt, "baseline_score": 0.0, "best_score": 0.0,
                "passed": [], "failed": [str(exc)], "error": str(exc),
                "elapsed_s": 0.0, "node_count": 0, "iterations": 0,
            })
            save_results(results)

    # ── Summary ───────────────────────────────────────────────────────
    results_sorted = sorted(results, key=lambda r: r["idx"])
    valid = [r for r in results_sorted if not r.get("error")]
    baseline_scores = [r.get("baseline_score", 0) for r in valid]
    best_scores = [r["best_score"] for r in valid]

    log.info("")
    log.info("=" * 90)
    log.info("RESULTS — Qwen Image 2512")
    log.info("=" * 90)
    log.info("%-3s  %-40s  %8s  %8s  %7s  %6s", "#", "Prompt", "Baseline", "Best", "Delta", "Time")
    log.info("-" * 90)
    for r in results_sorted:
        b = r.get("baseline_score", 0)
        s = r["best_score"]
        d = s - b
        tag = "UP" if d > 0.01 else ("ERR" if r.get("error") else "")
        log.info("%-3d  %-40s  %8.3f  %8.3f  %+7.3f  %5.0fs  %s",
                 r["idx"], r["prompt"][:38], b, s, d, r.get("elapsed_s", 0), tag)
    log.info("-" * 90)

    if valid:
        mb = sum(baseline_scores) / len(baseline_scores)
        ms = sum(best_scores) / len(best_scores)
        improved = sum(1 for b, s in zip(baseline_scores, best_scores) if s > b + 0.01)
        log.info("Valid: %d/%d | baseline=%.3f best=%.3f delta=+%.3f (+%.1f%%)",
                 len(valid), N_PROMPTS, mb, ms, ms - mb, (ms - mb) / max(mb, 0.001) * 100)
        log.info("Improved: %d/%d | Perfect(>=0.85): %d/%d",
                 improved, len(valid),
                 sum(1 for r in valid if r["best_score"] >= 0.85), len(valid))

    log.info("Results saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
