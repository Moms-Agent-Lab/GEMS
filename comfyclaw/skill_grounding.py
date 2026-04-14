"""
SkillGrounding — Verified acceptance of skill mutations.

Each proposed mutation must prove its worth against auto-generated test
prompts before being committed.  This prevents regressions where a
mutation sounds reasonable but actually produces worse/broken workflows.

Usage::

    grounding = SkillGrounding(run_prompt_fn=my_runner, min_improvement=0.05)
    accepted = grounding.verify_mutation(
        mutation=proposal,
        baseline_scores={"prompt1": 0.3, "prompt2": 0.5},
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .evolve import MutationProposal

log = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of verifying one mutation against test prompts."""

    mutation: MutationProposal
    test_prompts: list[str]
    baseline_scores: dict[str, float]
    post_scores: dict[str, float]
    mean_baseline: float
    mean_post: float
    improvement: float
    accepted: bool
    reason: str


class SkillGrounding:
    """Verify skill mutations against concrete test prompts.

    Parameters
    ----------
    run_prompt_fn : Callable that runs a single prompt through the full
                    harness and returns a result dict with at least ``score``.
    min_improvement : Minimum mean score improvement to accept the mutation.
    max_regression : Maximum allowed score drop on any individual prompt
                     (-1 = no individual-prompt check).
    """

    def __init__(
        self,
        run_prompt_fn: Callable[[str], dict[str, Any]],
        min_improvement: float = 0.03,
        max_regression: float = -0.10,
    ) -> None:
        self.run_prompt_fn = run_prompt_fn
        self.min_improvement = min_improvement
        self.max_regression = max_regression

    def run_baseline(self, prompts: list[str]) -> dict[str, float]:
        """Run test prompts before mutation and record baseline scores."""
        scores: dict[str, float] = {}
        for p in prompts:
            try:
                result = self.run_prompt_fn(p)
                scores[p] = result.get("score", 0.0)
            except Exception as exc:
                log.error("Baseline run failed for %r: %s", p, exc)
                scores[p] = 0.0
        return scores

    def run_post(self, prompts: list[str]) -> dict[str, float]:
        """Run test prompts after mutation and record post scores."""
        return self.run_baseline(prompts)  # same mechanism

    def verify_mutation(
        self,
        mutation: MutationProposal,
        baseline_scores: dict[str, float],
        post_scores: dict[str, float],
    ) -> GroundingResult:
        """Compare pre and post scores to decide if the mutation is accepted."""
        prompts = list(baseline_scores.keys())
        mean_base = sum(baseline_scores.values()) / len(baseline_scores) if baseline_scores else 0.0
        mean_post = sum(post_scores.get(p, 0.0) for p in prompts) / len(prompts) if prompts else 0.0
        improvement = mean_post - mean_base

        # Check for severe regressions on individual prompts
        worst_regression = 0.0
        for p in prompts:
            delta = post_scores.get(p, 0.0) - baseline_scores.get(p, 0.0)
            worst_regression = min(worst_regression, delta)

        if improvement >= self.min_improvement:
            if self.max_regression < 0 and worst_regression < self.max_regression:
                accepted = False
                reason = (
                    f"Mean improved by {improvement:.3f} but prompt-level "
                    f"regression {worst_regression:.3f} exceeds threshold "
                    f"{self.max_regression:.3f}"
                )
            else:
                accepted = True
                reason = f"Mean improvement {improvement:.3f} >= {self.min_improvement:.3f}"
        else:
            accepted = False
            reason = (
                f"Mean improvement {improvement:.3f} < "
                f"threshold {self.min_improvement:.3f}"
            )

        log.info(
            "Grounding verdict for mutation %s: %s (%.3f -> %.3f, delta=%.3f)",
            mutation.mutation_type,
            "ACCEPT" if accepted else "REJECT",
            mean_base, mean_post, improvement,
        )

        return GroundingResult(
            mutation=mutation,
            test_prompts=prompts,
            baseline_scores=baseline_scores,
            post_scores=post_scores,
            mean_baseline=mean_base,
            mean_post=mean_post,
            improvement=improvement,
            accepted=accepted,
            reason=reason,
        )

    def verify_full(
        self,
        mutation: MutationProposal,
        test_prompts: list[str],
        apply_fn: Callable[[], None],
        rollback_fn: Callable[[], None],
    ) -> GroundingResult:
        """Full workflow: baseline -> apply -> post -> verify -> maybe rollback.

        Parameters
        ----------
        mutation     : The mutation under test.
        test_prompts : Prompts to evaluate.
        apply_fn     : Callable that applies the mutation.
        rollback_fn  : Callable that rolls back the mutation.
        """
        log.info("Running baseline on %d test prompts", len(test_prompts))
        baseline = self.run_baseline(test_prompts)

        log.info("Applying mutation: %s", mutation.mutation_type)
        apply_fn()

        log.info("Running post-mutation on %d test prompts", len(test_prompts))
        post = self.run_post(test_prompts)

        result = self.verify_mutation(mutation, baseline, post)

        if not result.accepted:
            log.info("Rolling back mutation: %s", result.reason)
            rollback_fn()
        else:
            log.info("Mutation accepted: %s", result.reason)

        return result
