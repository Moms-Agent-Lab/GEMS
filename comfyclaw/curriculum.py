"""
Curriculum — Voyager-style automatic curriculum for skill emergence in T2I.

Organises GenEval2-style prompts into difficulty levels and drives progressive
skill discovery: skills learned at level N serve as foundations for level N+1.

Levels
------
1. single_object      — "a red car"  (basic workflow construction)
2. attribute_binding   — "a wooden chair"  (unusual materials / colours)
3. counting            — "three blue birds"  (exact count satisfaction)
4. spatial             — "a cat left of a dog"  (spatial relations)
5. complex_composition — "four rabbits and a sheep on a hill"  (multi-skill)

Each level runs a mini-benchmark, evolves skills, measures **skill transfer
rate** (what fraction of skills from level N help at level N+1), then
advances.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .evolve import EvolutionReport, SkillEvolver

log = logging.getLogger(__name__)


# ── Difficulty levels ──────────────────────────────────────────────────────

@dataclass
class CurriculumLevel:
    """One difficulty level in the curriculum."""

    name: str
    description: str
    difficulty: int  # 1-5
    prompt_filter: Callable[[dict], bool]
    min_prompts: int = 5
    max_prompts: int = 20


LEVELS: list[CurriculumLevel] = [
    CurriculumLevel(
        name="single_object",
        description="Single object, no special attributes",
        difficulty=1,
        prompt_filter=lambda p: p.get("atom_count", 0) <= 2
            and len(p.get("prompt", "").split()) <= 5,
        min_prompts=5,
        max_prompts=15,
    ),
    CurriculumLevel(
        name="attribute_binding",
        description="Objects with unusual material/colour/texture attributes",
        difficulty=2,
        prompt_filter=lambda p: any(
            kw in p.get("prompt", "").lower()
            for kw in ("metal", "stone", "glass", "wooden", "green", "blue",
                        "red", "golden", "silver", "crystal", "rubber")
        ),
        min_prompts=5,
        max_prompts=15,
    ),
    CurriculumLevel(
        name="counting",
        description="Exact count of objects required",
        difficulty=3,
        prompt_filter=lambda p: any(
            w in p.get("prompt", "").lower().split()
            for w in ("two", "three", "four", "five", "six", "seven", "eight",
                       "nine", "ten", "2", "3", "4", "5", "6", "7", "8", "9", "10")
        ),
        min_prompts=5,
        max_prompts=15,
    ),
    CurriculumLevel(
        name="spatial",
        description="Spatial relationships between objects",
        difficulty=4,
        prompt_filter=lambda p: any(
            kw in p.get("prompt", "").lower()
            for kw in ("left of", "right of", "above", "below", "behind",
                        "in front of", "next to", "on top of", "between")
        ),
        min_prompts=5,
        max_prompts=15,
    ),
    CurriculumLevel(
        name="complex_composition",
        description="Multiple distinct objects with attributes, counts, and relations",
        difficulty=5,
        prompt_filter=lambda p: (
            p.get("atom_count", 0) >= 4
            or (" and " in p.get("prompt", "").lower()
                and len(p.get("prompt", "").split()) >= 6)
        ),
        min_prompts=5,
        max_prompts=15,
    ),
]


# ── Result types ───────────────────────────────────────────────────────────

@dataclass
class LevelResult:
    """Results from running one curriculum level."""

    level: str
    difficulty: int
    num_prompts: int
    mean_score: float
    scores: list[float]
    evolution_report: EvolutionReport | None
    skills_before: list[str]
    skills_after: list[str]
    skills_created: list[str]
    skills_updated: list[str]
    duration_s: float = 0.0


@dataclass
class CurriculumReport:
    """Full report across all curriculum levels."""

    levels: list[LevelResult]
    skill_transfer_rates: dict[str, float]  # level_name -> transfer rate
    total_skills_created: int
    total_duration_s: float

    def summary(self) -> str:
        lines = ["Curriculum Report", "=" * 50]
        for lr in self.levels:
            lines.append(
                f"  L{lr.difficulty} {lr.level}: mean={lr.mean_score:.3f} "
                f"({lr.num_prompts} prompts, {lr.duration_s:.0f}s) "
                f"created={lr.skills_created}"
            )
        lines.append("")
        lines.append("Skill Transfer Rates:")
        for level_name, rate in self.skill_transfer_rates.items():
            lines.append(f"  {level_name}: {rate:.1%}")
        lines.append(f"\nTotal skills created: {self.total_skills_created}")
        lines.append(f"Total time: {self.total_duration_s:.0f}s")
        return "\n".join(lines)


# ── CurriculumRunner ───────────────────────────────────────────────────────

class CurriculumRunner:
    """Drive progressive skill emergence through a difficulty curriculum.

    Parameters
    ----------
    all_prompts : Full benchmark prompt list (GenEval2 format: dicts with
                  ``prompt``, ``atom_count``, ``vqa_list``).
    evolver     : Configured SkillEvolver instance.
    run_prompt_fn : Callable that runs a single prompt and returns a result
                    dict with at minimum ``prompt``, ``score``, ``passed``,
                    ``failed``, ``feedback``, ``node_count``.
    """

    def __init__(
        self,
        all_prompts: list[dict],
        evolver: SkillEvolver,
        run_prompt_fn: Callable[[str], dict[str, Any]],
    ) -> None:
        self.all_prompts = all_prompts
        self.evolver = evolver
        self.run_prompt_fn = run_prompt_fn

    def run(
        self,
        levels: list[CurriculumLevel] | None = None,
        evolution_cycles: int = 1,
    ) -> CurriculumReport:
        """Execute the full curriculum, level by level."""
        levels = levels or LEVELS
        level_results: list[LevelResult] = []
        transfer_rates: dict[str, float] = {}
        total_created = 0
        t0 = time.time()

        for level in sorted(levels, key=lambda l: l.difficulty):
            log.info("=" * 60)
            log.info("CURRICULUM LEVEL %d: %s", level.difficulty, level.name)
            log.info("=" * 60)

            lr = self._run_level(level, evolution_cycles)
            level_results.append(lr)
            total_created += len(lr.skills_created)

            # Compute skill transfer rate from previous level
            if len(level_results) >= 2:
                prev = level_results[-2]
                rate = self._compute_transfer_rate(prev, lr)
                transfer_rates[level.name] = rate
                log.info(
                    "Skill transfer rate from L%d->L%d: %.1f%%",
                    prev.difficulty, lr.difficulty, rate * 100,
                )

        return CurriculumReport(
            levels=level_results,
            skill_transfer_rates=transfer_rates,
            total_skills_created=total_created,
            total_duration_s=time.time() - t0,
        )

    def _run_level(
        self, level: CurriculumLevel, evolution_cycles: int
    ) -> LevelResult:
        """Run one curriculum level: benchmark + evolve."""
        t0 = time.time()

        # Select prompts for this level
        matching = [p for p in self.all_prompts if level.prompt_filter(p)]
        if len(matching) < level.min_prompts:
            log.warning(
                "Only %d prompts match level %s (min %d), using all",
                len(matching), level.name, level.min_prompts,
            )
        prompts = matching[: level.max_prompts]
        if not prompts:
            log.warning("No prompts for level %s, skipping", level.name)
            return LevelResult(
                level=level.name,
                difficulty=level.difficulty,
                num_prompts=0,
                mean_score=0.0,
                scores=[],
                evolution_report=None,
                skills_before=[],
                skills_after=[],
                skills_created=[],
                skills_updated=[],
                duration_s=time.time() - t0,
            )

        skills_before = self.evolver.store.list_skills()

        # Run benchmark
        results: list[dict] = []
        for i, prompt_data in enumerate(prompts):
            prompt_text = prompt_data["prompt"]
            log.info("[L%d] Prompt %d/%d: %s", level.difficulty, i + 1, len(prompts), prompt_text)
            try:
                r = self.run_prompt_fn(prompt_text)
                results.append(r)
                log.info("[L%d] Score: %.3f", level.difficulty, r.get("score", 0))
            except Exception as exc:
                log.error("[L%d] Prompt failed: %s", level.difficulty, exc)
                results.append({
                    "prompt": prompt_text,
                    "score": 0.0,
                    "passed": [],
                    "failed": [str(exc)],
                    "feedback": f"Error: {exc}",
                })

        scores = [r.get("score", 0.0) for r in results]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        log.info("[L%d] Mean score: %.3f", level.difficulty, mean_score)

        # Evolve skills based on this level's results
        evo_report: EvolutionReport | None = None
        if evolution_cycles > 0:
            for cycle in range(1, evolution_cycles + 1):
                evo_report = self.evolver.run_cycle(
                    results=results,
                    run_validation_fn=None,
                    cycle=cycle,
                )
                log.info("[L%d] Evolution cycle %d: %s", level.difficulty, cycle, evo_report.summary())

        skills_after = self.evolver.store.list_skills()
        created = [s for s in skills_after if s not in skills_before]
        updated = [
            m.target_skills[0]
            for m in (evo_report.mutations if evo_report else [])
            if m.accepted and m.mutation_type == "update" and m.target_skills
        ]

        return LevelResult(
            level=level.name,
            difficulty=level.difficulty,
            num_prompts=len(prompts),
            mean_score=mean_score,
            scores=scores,
            evolution_report=evo_report,
            skills_before=skills_before,
            skills_after=skills_after,
            skills_created=created,
            skills_updated=updated,
            duration_s=time.time() - t0,
        )

    @staticmethod
    def _compute_transfer_rate(prev: LevelResult, curr: LevelResult) -> float:
        """Fraction of skills created at prev level that are in curr's skill set.

        A transfer rate of 1.0 means all skills from the previous level
        persisted and were available for the current level.  If the current
        level deleted some, the rate drops.
        """
        if not prev.skills_created:
            return 1.0
        still_present = sum(
            1 for s in prev.skills_created if s in curr.skills_after
        )
        return still_present / len(prev.skills_created)
