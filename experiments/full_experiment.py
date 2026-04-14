#!/usr/bin/env python3
"""
Full NeurIPS experiment: baseline → stage-gated → self-evolution → post-evo.

Runs 5 GenEval2 prompts through each configuration, collects scores,
then executes one round of skill self-evolution and re-tests.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("experiment")

from comfyclaw.harness import ClawHarness, HarnessConfig
from comfyclaw.evolve import SkillEvolver

# ── Config ────────────────────────────────────────────────────────────────

GENEVAL2_PATH = os.environ.get("GENEVAL2_DATA", str(REPO_ROOT.parent / "GenEval2" / "geneval2_data.jsonl"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(REPO_ROOT.parent / "experiment_results"))
CHECKPOINT = "DreamShaper_8_pruned.safetensors"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SKILLS_DIR = str(REPO_ROOT / "comfyclaw" / "skills")
N_PROMPTS = 5
MAX_ITER = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set")
    sys.exit(1)


# ── Load GenEval2 prompts ──────────────────────────────────────────────────

def load_prompts(n: int) -> list[dict]:
    items = []
    with open(GENEVAL2_PATH) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            items.append(json.loads(line))
    return items


# ── Run one prompt through ClawHarness ────────────────────────────────────

def run_single(prompt: str, stage_gated: bool, label: str, idx: int) -> dict:
    """Run a single prompt and return result dict."""
    cfg = HarnessConfig(
        api_key=API_KEY,
        server_address="127.0.0.1:8188",
        model="anthropic/claude-sonnet-4-5",
        max_iterations=MAX_ITER,
        success_threshold=0.95,
        sync_port=0,
        image_model=CHECKPOINT,
        stage_gated=stage_gated,
        skills_dir=SKILLS_DIR,
    )

    harness = ClawHarness.from_workflow_dict({}, cfg)

    t0 = time.time()
    with harness:
        image_bytes = harness.run(prompt)
    elapsed = time.time() - t0

    best_score = 0.0
    passed_all = []
    failed_all = []
    feedback_texts = []
    for entry in harness.evolution_log.entries:
        if entry.verifier_score is not None:
            best_score = max(best_score, entry.verifier_score)

    for attempt in harness.memory.attempts:
        passed_all.extend(attempt.passed)
        failed_all.extend(attempt.failed)
        if attempt.experience:
            feedback_texts.append(attempt.experience)

    img_path = None
    if image_bytes:
        img_path = os.path.join(OUTPUT_DIR, f"{label}_{idx:02d}.png")
        with open(img_path, "wb") as f:
            f.write(image_bytes)

    return {
        "index": idx,
        "prompt": prompt,
        "label": label,
        "score": best_score,
        "elapsed_s": elapsed,
        "has_image": image_bytes is not None,
        "image_path": img_path,
        "passed": list(set(passed_all)),
        "failed": list(set(failed_all)),
        "feedback": "; ".join(feedback_texts[:3]),
        "evolution_log": harness.evolution_log.format(),
        "stage_gated": stage_gated,
    }


# ── Run a batch of prompts ────────────────────────────────────────────────

def run_batch(prompts: list[dict], stage_gated: bool, label: str) -> list[dict]:
    """Run all prompts and return list of result dicts."""
    results = []
    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        log.info("[%s] Prompt %d/%d: %s", label, i + 1, len(prompts), prompt)
        try:
            r = run_single(prompt, stage_gated, label, i)
            results.append(r)
            log.info("[%s] Prompt %d: score=%.3f (%.1fs)", label, i + 1, r["score"], r["elapsed_s"])
        except Exception as e:
            log.error("[%s] Prompt %d failed: %s", label, i + 1, e)
            results.append({
                "index": i,
                "prompt": prompt,
                "label": label,
                "score": 0.0,
                "error": str(e),
                "passed": [],
                "failed": [str(e)],
                "feedback": f"Error: {e}",
            })
    return results


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    scores = [r["score"] for r in results]
    completed = [r for r in results if r.get("has_image")]
    return {
        "n_prompts": len(results),
        "n_completed": len(completed),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "scores": scores,
    }


# ── Main experiment ────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("FULL NEURIPS EXPERIMENT")
    log.info("=" * 70)

    prompts = load_prompts(N_PROMPTS)
    log.info("Loaded %d GenEval2 prompts:", len(prompts))
    for i, p in enumerate(prompts):
        log.info("  [%d] %s (skills: %s)", i, p["prompt"], p.get("skills", []))

    all_results = {}
    experiment_start = time.time()

    # ── Phase 1: Baseline (no stage gating) ────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("PHASE 1: BASELINE (stage_gated=False)")
    log.info("=" * 70)

    baseline_results = run_batch(prompts, stage_gated=False, label="baseline")
    baseline_metrics = compute_metrics(baseline_results)
    all_results["baseline"] = {
        "results": baseline_results,
        "metrics": baseline_metrics,
    }
    log.info("Baseline: mean=%.3f, min=%.3f, max=%.3f",
             baseline_metrics["mean_score"], baseline_metrics["min_score"],
             baseline_metrics["max_score"])

    # ── Phase 2: Stage-gated ───────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: STAGE-GATED (stage_gated=True)")
    log.info("=" * 70)

    staged_results = run_batch(prompts, stage_gated=True, label="staged")
    staged_metrics = compute_metrics(staged_results)
    all_results["staged"] = {
        "results": staged_results,
        "metrics": staged_metrics,
    }
    log.info("Stage-gated: mean=%.3f, min=%.3f, max=%.3f",
             staged_metrics["mean_score"], staged_metrics["min_score"],
             staged_metrics["max_score"])

    # ── Phase 3: Self-Evolution ────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: SELF-EVOLUTION (1 cycle)")
    log.info("=" * 70)

    # Combine all results for evolution input
    evo_input = []
    for r in baseline_results + staged_results:
        evo_input.append({
            "prompt": r["prompt"],
            "score": r["score"],
            "passed": r.get("passed", []),
            "failed": r.get("failed", []),
            "feedback": r.get("feedback", ""),
        })

    evolver = SkillEvolver(
        skills_dir=SKILLS_DIR,
        llm_model="anthropic/claude-sonnet-4-5",
        api_key=API_KEY,
        min_improvement=0.02,
        max_mutations_per_cycle=3,
    )

    log.info("Running evolution cycle with %d results (mean score: %.3f)...",
             len(evo_input),
             sum(r["score"] for r in evo_input) / len(evo_input))

    evo_report = evolver.run_cycle(
        results=evo_input,
        run_validation_fn=None,  # Accept optimistically (no re-run)
        cycle=1,
    )

    log.info("Evolution report:")
    log.info(evo_report.summary())

    all_results["evolution"] = {
        "pre_mean_score": evo_report.pre_mean_score,
        "post_mean_score": evo_report.post_mean_score,
        "mutations_proposed": evo_report.mutations_proposed,
        "mutations_accepted": evo_report.mutations_accepted,
        "failure_clusters": [
            {
                "name": c.name,
                "description": c.description,
                "failure_count": c.failure_count,
                "mean_score": c.mean_score,
                "affected_prompts": c.affected_prompts,
                "existing_skill": c.existing_skill,
            }
            for c in evo_report.failure_clusters
        ],
        "mutations": [
            {
                "type": m.mutation_type,
                "targets": m.target_skills,
                "rationale": m.rationale,
                "cluster": m.failure_cluster,
                "accepted": m.accepted,
            }
            for m in evo_report.mutations
        ],
    }

    # ── Phase 4: Post-evolution benchmark ──────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("PHASE 4: POST-EVOLUTION (stage_gated=True, evolved skills)")
    log.info("=" * 70)

    post_results = run_batch(prompts, stage_gated=True, label="post_evo")
    post_metrics = compute_metrics(post_results)
    all_results["post_evolution"] = {
        "results": post_results,
        "metrics": post_metrics,
    }
    log.info("Post-evolution: mean=%.3f, min=%.3f, max=%.3f",
             post_metrics["mean_score"], post_metrics["min_score"],
             post_metrics["max_score"])

    # ── Final summary ──────────────────────────────────────────────────
    total_time = time.time() - experiment_start

    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT COMPLETE")
    log.info("=" * 70)

    phases = [
        ("Baseline", baseline_metrics),
        ("Stage-gated", staged_metrics),
        ("Post-evolution", post_metrics),
    ]

    log.info("\n%-20s  %8s  %8s  %8s  %s", "Phase", "Mean", "Min", "Max", "Scores")
    log.info("-" * 80)
    for name, m in phases:
        scores_str = ", ".join(f"{s:.2f}" for s in m["scores"])
        log.info("%-20s  %8.3f  %8.3f  %8.3f  [%s]",
                 name, m["mean_score"], m["min_score"], m["max_score"], scores_str)

    if evo_report.failure_clusters:
        log.info("\nFailure clusters identified:")
        for c in evo_report.failure_clusters:
            log.info("  [%s] %s (count=%d, mean_score=%.2f, existing_skill=%s)",
                     c.name, c.description, c.failure_count, c.mean_score,
                     c.existing_skill or "none")

    if evo_report.mutations:
        log.info("\nSkill mutations applied:")
        for m in evo_report.mutations:
            status = "ACCEPTED" if m.accepted else "REJECTED"
            log.info("  [%s] %s %s — %s",
                     status, m.mutation_type, m.target_skills, m.rationale[:100])

    log.info("\nTotal experiment time: %.1f min", total_time / 60)

    # Save all results
    results_path = os.path.join(OUTPUT_DIR, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
