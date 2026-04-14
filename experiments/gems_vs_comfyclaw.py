#!/usr/bin/env python3
"""
GEMS vs ComfyClaw Benchmark — Run the same 10 GenEval2 prompts through both
systems using the same backend (Claude + ComfyUI/DreamShaper).

GEMS baseline: prompt-only refinement loop (plan → decompose → generate →
verify → refine prompt).  No structural workflow changes.

ComfyClaw: full topology evolution (stage-gated agent builds/modifies the
ComfyUI workflow graph including LoRA/ControlNet/node wiring).

This isolates the key contribution: workflow-level agency vs prompt-level agency.
"""

import base64
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("benchmark")

import litellm

# ── Config ─────────────────────────────────────────────────────────────────

GENEVAL2_PATH = os.environ.get("GENEVAL2_DATA", str(REPO_ROOT.parent / "GenEval2" / "geneval2_data.jsonl"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(REPO_ROOT.parent / "benchmark_gems_vs_comfyclaw"))
CHECKPOINT = "DreamShaper_8_pruned.safetensors"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SKILLS_DIR = str(REPO_ROOT / "comfyclaw" / "skills")
LLM_MODEL = "anthropic/claude-sonnet-4-5"
COMFYUI_ADDR = "127.0.0.1:8188"
N_PROMPTS = 10
MAX_ITERATIONS_GEMS = 3
MAX_ITERATIONS_CLAW = 2  # fewer iters — topology changes are heavier than prompt edits
CLAW_WARM_START = True   # give ComfyClaw the same base workflow as GEMS

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "gems_images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "comfyclaw_images"), exist_ok=True)

if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# GEMS Baseline — Faithful reimplementation using Claude + ComfyUI
# ══════════════════════════════════════════════════════════════════════════

# A fixed 7-node SD1.5 workflow — GEMS never touches the graph structure
BASE_WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": CHECKPOINT},
        "_meta": {"title": "Load Checkpoint"},
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["1", 1], "text": ""},
        "_meta": {"title": "Positive Prompt"},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["1", 1], "text": "blurry, ugly, deformed, low quality"},
        "_meta": {"title": "Negative Prompt"},
    },
    "4": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
        "_meta": {"title": "Empty Latent"},
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": 42,
            "steps": 25,
            "cfg": 7.0,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1.0,
        },
        "_meta": {"title": "KSampler"},
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        "_meta": {"title": "VAE Decode"},
    },
    "7": {
        "class_type": "SaveImage",
        "inputs": {"images": ["6", 0], "filename_prefix": "GEMS"},
        "_meta": {"title": "Save Image"},
    },
}


def llm_call(prompt: str, images: list[bytes] | None = None, max_tokens: int = 4096) -> str:
    """Call the LLM with optional images."""
    content: list[dict] = []
    if images:
        segments = prompt.split("<image>")
        for i, seg in enumerate(segments):
            if seg.strip():
                content.append({"type": "text", "text": seg})
            if i < len(images):
                b64 = base64.b64encode(images[i]).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
    else:
        content = [{"type": "text", "text": prompt}]

    resp = litellm.completion(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def comfyui_generate(prompt_text: str, seed: int = 42) -> bytes | None:
    """Generate an image with a fixed SD1.5 workflow (GEMS-style: no graph changes)."""
    import copy
    import urllib.request

    wf = copy.deepcopy(BASE_WORKFLOW)
    wf["2"]["inputs"]["text"] = prompt_text
    wf["5"]["inputs"]["seed"] = seed

    # Queue
    payload = json.dumps({"prompt": wf}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{COMFYUI_ADDR}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        prompt_id = data["prompt_id"]
    except Exception as exc:
        log.error("ComfyUI queue failed: %s", exc)
        return None

    # Poll for completion
    import time as _time
    for _ in range(300):
        _time.sleep(2)
        try:
            with urllib.request.urlopen(
                f"http://{COMFYUI_ADDR}/history/{prompt_id}", timeout=10
            ) as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                for node_id, node_out in outputs.items():
                    if "images" in node_out:
                        for img_info in node_out["images"]:
                            fname = img_info["filename"]
                            subfolder = img_info.get("subfolder", "")
                            img_type = img_info.get("type", "output")
                            url = (
                                f"http://{COMFYUI_ADDR}/view?"
                                f"filename={fname}&subfolder={subfolder}&type={img_type}"
                            )
                            with urllib.request.urlopen(url, timeout=30) as img_resp:
                                return img_resp.read()
                if history[prompt_id].get("status", {}).get("status_str") == "error":
                    log.error("ComfyUI execution error for prompt_id=%s", prompt_id)
                    return None
        except Exception:
            pass
    log.error("ComfyUI timeout for prompt_id=%s", prompt_id)
    return None


# ── GEMS Algorithm ─────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = (
    "Analyze the user's image generation prompt. "
    "Break it down into specific visual requirements. "
    "For each requirement, write a question that can be answered with a simple 'yes' or 'no'. "
    "The questions should verify if the requirement is met in an image. "
    "YOU MUST RESPOND ONLY WITH A JSON ARRAY OF STRINGS. "
    'Example format: ["Is there a cat?", "Is the cat black?", "Is it sitting on a rug?"]'
)

VERIFY_PROMPT_PREFIX = (
    "Answer the following question with only 'yes' or 'no' based on the provided image: "
)

SUMMARIZE_EXPERIENCE_TEMPLATE = (
    "Task: Summarize the experience of the current image generation attempt.\n"
    "--- CURRENT ATTEMPT ---\n"
    "Prompt used: {current_prompt}\n"
    "Passed requirements: {passed}\n"
    "Failed requirements: {failed}\n"
    "Image: <image>\n"
    "--- PREVIOUS EXPERIENCES ---\n"
    "{previous_experiences}\n"
    "--- ANALYSIS ---\n"
    "Based on the provided image, the verification results, and historical experiences, "
    "write a concise summary of what worked, what failed, and what strategy should be adopted "
    "in the next attempt. Keep it under 100 words. Do not include introductory phrases."
)

REFINE_PROMPT_TEMPLATE = (
    "Task: Refine the image generation prompt based on previous failed attempts and accumulated experiences.\n"
    "Original Intent: {original_prompt}\n\n"
    "--- ATTEMPT HISTORY ---\n"
    "{history_log}\n"
    "--- ANALYSIS ---\n"
    "Review the history above. Rewrite a new, comprehensive prompt. This prompt must:\n"
    "1. Explicitly reinforce the requirements that failed in the latest attempt.\n"
    "2. Maintain and protect the requirements that were successfully met in previous rounds.\n"
    "3. Adopt the strategies suggested in the 'Experience' section.\n"
    "4. Use clear, non-conflicting descriptive language.\n\n"
    "Return ONLY the prompt text itself. Do not include any conversational filler."
)


def gems_decompose(prompt: str) -> list[str]:
    """Decompose a prompt into yes/no verification questions."""
    task = f"{DECOMPOSE_PROMPT}\n\nUser Prompt: {prompt}"
    response = llm_call(task, max_tokens=2048)
    try:
        m = re.search(r"\[.*\]", response, re.DOTALL)
        questions = json.loads(m.group() if m else response)
        return [q for q in questions if isinstance(q, str)]
    except Exception:
        return [line.strip() for line in response.split("\n") if "?" in line]


def gems_verify(image_bytes: bytes, questions: list[str]) -> list[dict]:
    """Verify image against questions (parallel, like GEMS)."""
    def ask_one(q: str) -> dict:
        full_query = f"<image>\n{VERIFY_PROMPT_PREFIX} {q}"
        try:
            answer = llm_call(full_query, images=[image_bytes], max_tokens=50).lower()
            passed = "yes" in answer and "no" not in answer
            return {"question": q, "passed": passed}
        except Exception as exc:
            return {"question": q, "passed": False}

    with ThreadPoolExecutor(max_workers=min(len(questions), 5)) as pool:
        return list(pool.map(ask_one, questions))


def gems_run(prompt: str, max_iterations: int = 3) -> dict:
    """Run the full GEMS algorithm: decompose → generate → verify → refine loop."""
    t0 = time.time()
    log.info("[GEMS] Prompt: %s", prompt)

    # Step 1: Decompose
    questions = gems_decompose(prompt)
    log.info("[GEMS] Decomposed into %d questions: %s", len(questions), questions)

    if not questions:
        img = comfyui_generate(prompt)
        return {
            "prompt": prompt,
            "score": 0.0,
            "passed": [],
            "failed": [],
            "questions": [],
            "image_bytes": img,
            "elapsed_s": time.time() - t0,
            "iterations_used": 1,
        }

    current_prompt = prompt
    attempt_history: list[dict] = []
    best_image: bytes | None = None
    best_passed_count = -1
    best_passed: list[str] = []
    best_failed: list[str] = []

    for iteration in range(1, max_iterations + 1):
        log.info("[GEMS] --- Round %d/%d ---", iteration, max_iterations)
        log.info("[GEMS] Current prompt: %s", current_prompt[:120])

        # Generate
        seed = 42 + iteration * 1000
        image_bytes = comfyui_generate(current_prompt, seed=seed)
        if image_bytes is None:
            log.error("[GEMS] Generation failed at round %d", iteration)
            continue

        # Verify
        verifications = gems_verify(image_bytes, questions)
        passed_q = [v["question"] for v in verifications if v["passed"]]
        failed_q = [v["question"] for v in verifications if not v["passed"]]

        log.info("[GEMS] Passed: %d/%d", len(passed_q), len(questions))
        for v in verifications:
            log.info("[GEMS]   %s %s", "✅" if v["passed"] else "❌", v["question"])

        if len(passed_q) > best_passed_count:
            best_passed_count = len(passed_q)
            best_image = image_bytes
            best_passed = passed_q
            best_failed = failed_q

        # All passed → early stop
        if not failed_q:
            log.info("[GEMS] All requirements met!")
            break

        # Refine prompt (if not last iteration)
        if iteration < max_iterations:
            prev_exp = "\n".join(
                f"Round {r['iteration']}: {r['experience']}"
                for r in attempt_history
            ) or "None (First round)"

            summary_task = SUMMARIZE_EXPERIENCE_TEMPLATE.format(
                current_prompt=current_prompt,
                passed=", ".join(passed_q) or "None",
                failed=", ".join(failed_q) or "None",
                previous_experiences=prev_exp,
            )
            experience = llm_call(summary_task, images=[image_bytes], max_tokens=300)

            attempt_history.append({
                "iteration": iteration,
                "prompt": current_prompt,
                "experience": experience,
                "failed": failed_q,
                "passed": passed_q,
                "image_bytes": image_bytes,
            })

            history_log = ""
            history_images = []
            for record in attempt_history:
                history_log += (
                    f"Attempt {record['iteration']}:\n"
                    f"- Experience: {record['experience']}\n"
                    f"- Prompt: {record['prompt']}\n"
                    f"- Image Result: <image>\n"
                    f"- Failed Points: {', '.join(record['failed']) or 'None'}\n\n"
                )
                history_images.append(record["image_bytes"])

            refine_task = REFINE_PROMPT_TEMPLATE.format(
                original_prompt=prompt,
                history_log=history_log,
            )
            current_prompt = llm_call(refine_task, images=history_images, max_tokens=1024)
            log.info("[GEMS] Refined prompt: %s", current_prompt[:120])

    score = best_passed_count / len(questions) if questions else 0.0
    return {
        "prompt": prompt,
        "score": score,
        "passed": best_passed,
        "failed": best_failed,
        "questions": questions,
        "image_bytes": best_image,
        "elapsed_s": time.time() - t0,
        "iterations_used": min(iteration, max_iterations),
    }


# ══════════════════════════════════════════════════════════════════════════
# ComfyClaw Runner
# ══════════════════════════════════════════════════════════════════════════

def comfyclaw_run(prompt: str, max_iterations: int = 3, warm_start: bool = True) -> dict:
    """Run ComfyClaw's full topology-evolving agent."""
    import copy
    from comfyclaw.harness import ClawHarness, HarnessConfig

    t0 = time.time()
    log.info("[ComfyClaw] Prompt: %s  warm_start=%s", prompt, warm_start)

    cfg = HarnessConfig(
        api_key=API_KEY,
        server_address=COMFYUI_ADDR,
        model=LLM_MODEL,
        max_iterations=max_iterations,
        success_threshold=0.95,
        sync_port=0,
        image_model=CHECKPOINT,
        stage_gated=True,
        skills_dir=SKILLS_DIR,
        max_nodes=20,
        baseline_first=warm_start,
    )

    init_workflow = copy.deepcopy(BASE_WORKFLOW) if warm_start else {}
    harness = ClawHarness.from_workflow_dict(init_workflow, cfg)
    with harness:
        image_bytes = harness.run(prompt)

    best_score = 0.0
    passed_all = []
    failed_all = []
    # Check both evolution log and memory (baseline-first is only in memory)
    for entry in harness.evolution_log.entries:
        if entry.verifier_score is not None:
            best_score = max(best_score, entry.verifier_score)
    for attempt in harness.memory.attempts:
        if attempt.verifier_score > best_score:
            best_score = attempt.verifier_score
    for attempt in harness.memory.attempts:
        if attempt.verifier_score == best_score:
            passed_all = attempt.passed
            failed_all = attempt.failed
            break

    return {
        "prompt": prompt,
        "score": best_score,
        "passed": passed_all,
        "failed": failed_all,
        "image_bytes": image_bytes,
        "elapsed_s": time.time() - t0,
        "iterations_used": len(harness.evolution_log.entries),
        "node_count": (
            harness.evolution_log.entries[-1].node_count_after
            if harness.evolution_log.entries else 7
        ),
    }


# ══════════════════════════════════════════════════════════════════════════
# Main Benchmark
# ══════════════════════════════════════════════════════════════════════════

def load_prompts(n: int) -> list[dict]:
    items = []
    with open(GENEVAL2_PATH) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            items.append(json.loads(line))
    return items


def save_image(img_bytes: bytes | None, path: str) -> str | None:
    if img_bytes:
        with open(path, "wb") as f:
            f.write(img_bytes)
        return path
    return None


def _load_cache() -> dict:
    """Load partial results from previous runs so we can resume."""
    cache_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def main():
    log.info("=" * 70)
    log.info("GEMS vs ComfyClaw Benchmark — %d GenEval2 prompts", N_PROMPTS)
    log.info("Backend: Claude (%s) + ComfyUI (%s)", LLM_MODEL, CHECKPOINT)
    log.info("Max iterations: GEMS=%d  ComfyClaw=%d  warm_start=%s", MAX_ITERATIONS_GEMS, MAX_ITERATIONS_CLAW, CLAW_WARM_START)
    log.info("=" * 70)

    prompts = load_prompts(N_PROMPTS)
    for i, p in enumerate(prompts):
        log.info("  [%d] %s", i, p["prompt"])

    cache = _load_cache()
    cached_gems = cache.get("gems", {}).get("per_prompt", [])
    cached_claw = cache.get("comfyclaw", {}).get("per_prompt", [])

    # ── Phase 1: GEMS Baseline ────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("PHASE 1: GEMS BASELINE (prompt-only refinement)")
    log.info("=" * 70)

    gems_results: list[dict] = []
    for i, item in enumerate(prompts):
        prompt = item["prompt"]

        # Reuse cached result if image exists
        if i < len(cached_gems) and cached_gems[i].get("score", -1) >= 0:
            img_file = os.path.join(OUTPUT_DIR, "gems_images", f"gems_{i:02d}.png")
            if os.path.exists(img_file):
                log.info("[GEMS %d/%d] CACHED score=%.3f — %s", i + 1, N_PROMPTS, cached_gems[i]["score"], prompt)
                gems_results.append(cached_gems[i])
                continue

        log.info("\n[GEMS %d/%d] %s", i + 1, N_PROMPTS, prompt)
        try:
            r = gems_run(prompt, max_iterations=MAX_ITERATIONS_GEMS)
            img_path = save_image(
                r["image_bytes"],
                os.path.join(OUTPUT_DIR, "gems_images", f"gems_{i:02d}.png"),
            )
            r["image_path"] = img_path
            del r["image_bytes"]
            gems_results.append(r)
            log.info(
                "[GEMS %d/%d] score=%.3f (%d/%d passed) time=%.1fs",
                i + 1, N_PROMPTS, r["score"],
                len(r["passed"]), len(r["passed"]) + len(r["failed"]),
                r["elapsed_s"],
            )
        except Exception as exc:
            log.error("[GEMS %d/%d] FAILED: %s", i + 1, N_PROMPTS, exc)
            gems_results.append({
                "prompt": prompt,
                "score": 0.0,
                "passed": [],
                "failed": [str(exc)],
                "error": str(exc),
                "elapsed_s": 0.0,
            })

    # ── Phase 2: ComfyClaw ────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: COMFYCLAW (topology-evolving agent, stage-gated)")
    log.info("=" * 70)

    claw_results: list[dict] = []
    for i, item in enumerate(prompts):
        prompt = item["prompt"]

        # Reuse cached result if image exists
        if i < len(cached_claw) and cached_claw[i].get("score", -1) >= 0:
            img_file = os.path.join(OUTPUT_DIR, "comfyclaw_images", f"claw_{i:02d}.png")
            if os.path.exists(img_file):
                log.info("[ComfyClaw %d/%d] CACHED score=%.3f — %s", i + 1, N_PROMPTS, cached_claw[i]["score"], prompt)
                claw_results.append(cached_claw[i])
                continue

        log.info("\n[ComfyClaw %d/%d] %s", i + 1, N_PROMPTS, prompt)
        try:
            r = comfyclaw_run(prompt, max_iterations=MAX_ITERATIONS_CLAW, warm_start=CLAW_WARM_START)
            img_path = save_image(
                r.get("image_bytes"),
                os.path.join(OUTPUT_DIR, "comfyclaw_images", f"claw_{i:02d}.png"),
            )
            r["image_path"] = img_path
            r.pop("image_bytes", None)
            claw_results.append(r)
            log.info(
                "[ComfyClaw %d/%d] score=%.3f time=%.1fs nodes=%d",
                i + 1, N_PROMPTS, r["score"], r["elapsed_s"],
                r.get("node_count", 0),
            )
        except Exception as exc:
            log.error("[ComfyClaw %d/%d] FAILED: %s", i + 1, N_PROMPTS, exc)
            claw_results.append({
                "prompt": prompt,
                "score": 0.0,
                "passed": [],
                "failed": [str(exc)],
                "error": str(exc),
                "elapsed_s": 0.0,
            })

    # ── Summary ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("BENCHMARK RESULTS")
    log.info("=" * 70)

    gems_scores = [r["score"] for r in gems_results]
    claw_scores = [r["score"] for r in claw_results]
    gems_mean = sum(gems_scores) / len(gems_scores) if gems_scores else 0
    claw_mean = sum(claw_scores) / len(claw_scores) if claw_scores else 0
    gems_times = [r.get("elapsed_s", 0) for r in gems_results]
    claw_times = [r.get("elapsed_s", 0) for r in claw_results]

    log.info("")
    log.info("%-4s  %-35s  %8s  %8s  %8s  %8s", "#", "Prompt", "GEMS", "Claw", "GEMS_t", "Claw_t")
    log.info("-" * 100)
    for i in range(N_PROMPTS):
        p = prompts[i]["prompt"][:33]
        gs = gems_scores[i] if i < len(gems_scores) else 0
        cs = claw_scores[i] if i < len(claw_scores) else 0
        gt = gems_times[i] if i < len(gems_times) else 0
        ct = claw_times[i] if i < len(claw_times) else 0
        winner = "←" if gs > cs else ("→" if cs > gs else "=")
        log.info(
            "%-4d  %-35s  %8.3f  %8.3f  %7.0fs  %7.0fs  %s",
            i, p, gs, cs, gt, ct, winner,
        )

    log.info("-" * 100)
    log.info(
        "%-4s  %-35s  %8.3f  %8.3f  %7.0fs  %7.0fs",
        "", "MEAN", gems_mean, claw_mean,
        sum(gems_times) / len(gems_times) if gems_times else 0,
        sum(claw_times) / len(claw_times) if claw_times else 0,
    )

    delta = claw_mean - gems_mean
    log.info("")
    if delta > 0:
        log.info("ComfyClaw wins by +%.3f (%.1f%% relative improvement)", delta, delta / max(gems_mean, 0.001) * 100)
    elif delta < 0:
        log.info("GEMS wins by +%.3f", -delta)
    else:
        log.info("Tie!")

    # GEMS wins/losses
    gems_wins = sum(1 for i in range(N_PROMPTS) if gems_scores[i] > claw_scores[i])
    claw_wins = sum(1 for i in range(N_PROMPTS) if claw_scores[i] > gems_scores[i])
    ties = N_PROMPTS - gems_wins - claw_wins
    log.info("Win/Loss/Tie: GEMS %d / ComfyClaw %d / Tie %d", gems_wins, claw_wins, ties)

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "config": {
            "n_prompts": N_PROMPTS,
            "max_iterations_gems": MAX_ITERATIONS_GEMS,
            "max_iterations_claw": MAX_ITERATIONS_CLAW,
            "claw_warm_start": CLAW_WARM_START,
            "llm_model": LLM_MODEL,
            "checkpoint": CHECKPOINT,
        },
        "gems": {
            "mean_score": gems_mean,
            "scores": gems_scores,
            "mean_time_s": sum(gems_times) / len(gems_times) if gems_times else 0,
            "per_prompt": gems_results,
        },
        "comfyclaw": {
            "mean_score": claw_mean,
            "scores": claw_scores,
            "mean_time_s": sum(claw_times) / len(claw_times) if claw_times else 0,
            "per_prompt": claw_results,
        },
        "comparison": {
            "gems_mean": gems_mean,
            "comfyclaw_mean": claw_mean,
            "delta": delta,
            "gems_wins": gems_wins,
            "comfyclaw_wins": claw_wins,
            "ties": ties,
        },
    }

    results_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("\nResults saved to %s", results_path)


if __name__ == "__main__":
    main()
