---
name: self-evolve
description: >-
  Self-evolution engine for ComfyClaw's skill system. Analyzes failure patterns
  across benchmark runs and autonomously proposes skill mutations: creating new
  skills for uncovered failure modes, updating underperforming skills, merging
  overlapping skills, or deleting skills that hurt performance. Use when running
  evolution cycles, optimizing skill coverage, or when benchmark scores plateau
  and the system needs to adapt its own capabilities.
license: MIT
metadata:
  author: comfyclaw
  version: "1.0.0"
tags: [meta]
---

# Self-Evolve — Autonomous Skill Evolution Engine

This skill drives the core self-evolution loop: run benchmarks, analyze
failures, propose skill mutations, validate improvements, commit or rollback.

## Evolution cycle overview

```
1. Run benchmark batch (N prompts from GenEval2/CREA/OneIG)
2. Collect results: scores, failure patterns, verifier feedback
3. Cluster failures by type (anatomy, spatial, style, text, etc.)
4. For each failure cluster:
   a. Check if an existing skill covers this pattern
   b. If no coverage → propose CREATE mutation
   c. If covered but underperforming → propose UPDATE mutation
   d. If two skills overlap significantly → propose MERGE mutation
   e. If a skill consistently hurts scores → propose DELETE mutation
5. Apply mutation to skill store
6. Re-run benchmark on held-out validation set
7. If score improved → commit mutation
8. If score degraded → rollback to previous version
9. Repeat until convergence or max cycles reached
```

## Failure clustering

Failures are grouped by analyzing verifier feedback across multiple runs:

| Cluster | Signals | Typical mutation |
|---------|---------|------------------|
| **Anatomy** | "bad anatomy", "extra fingers", "deformed" | Update `lora-enhancement` or create `anatomy-fix` |
| **Spatial** | "wrong position", "layout error", "overlapping" | Update `spatial` skill instructions |
| **Text rendering** | "blurry text", "wrong text", "unreadable" | Update `text-rendering` skill |
| **Style mismatch** | "wrong style", "not photorealistic", "cartoon" | Create style-specific skill |
| **Composition** | "bad composition", "cluttered", "empty space" | Create `composition` skill |
| **Color/lighting** | "wrong colors", "too dark", "oversaturated" | Create `lighting-color` skill |
| **Detail/quality** | "blurry", "low detail", "artifacts" | Update `high-quality` or `hires-fix` |

## Mutation proposal format

Each proposed mutation is structured as:

```json
{
  "mutation_type": "create|update|merge|delete",
  "target_skills": ["skill-name"],
  "rationale": "Why this mutation addresses the failure cluster",
  "failure_cluster": "anatomy",
  "affected_prompts": 5,
  "mean_score_on_cluster": 0.42,
  "proposed_changes": {
    "name": "anatomy-fix",
    "description": "...",
    "body": "..."
  }
}
```

## Validation protocol

After each mutation:
1. Run the same benchmark prompts with the mutated skill set
2. Compare mean score vs. pre-mutation baseline
3. Require >= 0.02 improvement to commit (avoid noise)
4. If multiple mutations proposed, apply and validate one at a time

## Convergence criteria

Stop evolving when:
- Mean score improvement < 0.01 over last 3 cycles
- Maximum evolution cycles reached (default: 10)
- All failure clusters have < 5% occurrence rate
- Manual stop from operator

## Integration

The evolution engine is invoked via:

```python
from comfyclaw.evolve import SkillEvolver

evolver = SkillEvolver(
    skills_dir="comfyclaw/skills",
    benchmark_config=bench_cfg,
    llm_model="anthropic/claude-sonnet-4-5",
)

# Run a full evolution cycle
report = evolver.run_cycle(
    train_prompts=train_set,
    val_prompts=val_set,
    max_mutations=3,
)
```
