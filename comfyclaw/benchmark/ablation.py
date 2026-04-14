"""
AblationRunner — Systematic ablation matrix for NeurIPS experiments.

Runs a configurable matrix of ablation variables across a benchmark,
generating a LaTeX-ready results table.

Ablation dimensions:
- stage_gating: {True, False}
- skills_enabled: {True, False}
- evolution: {True, False}
- iterations: {1, 2, 3, 5}
- llm_backend: {"claude-sonnet-4-5", "gpt-4o", "gpt-4o-mini"}

Usage::

    python -m comfyclaw.benchmark.ablation \
        --suite geneval2 --data-path prompts.json \
        --output ablation_results/ \
        --max-prompts 50
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .runner import BenchmarkConfig, BenchmarkResult, BenchmarkRunner

log = logging.getLogger(__name__)


@dataclass
class AblationVariable:
    """One variable in the ablation matrix."""

    name: str
    values: list[Any]
    description: str = ""


@dataclass
class AblationCondition:
    """One row in the ablation matrix (one experiment configuration)."""

    name: str
    variables: dict[str, Any]

    def short_name(self) -> str:
        parts = []
        for k, v in sorted(self.variables.items()):
            if isinstance(v, bool):
                parts.append(f"{k[0].upper()}" if v else f"no-{k[0].upper()}")
            else:
                parts.append(f"{k}={v}")
        return "_".join(parts)


@dataclass
class AblationResult:
    """Results from the ablation study."""

    conditions: list[AblationCondition]
    results: dict[str, BenchmarkResult]  # condition.name -> result
    total_duration_s: float

    def to_latex_table(self) -> str:
        """Generate a LaTeX table for the paper."""
        headers = list(self.conditions[0].variables.keys()) + [
            "Score", "Completed", "Latency"
        ]
        header_line = " & ".join(headers) + " \\\\"

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Ablation study results}",
            "\\label{tab:ablation}",
            f"\\begin{{tabular}}{{{'l' * len(self.conditions[0].variables)}rrr}}",
            "\\toprule",
            header_line,
            "\\midrule",
        ]

        for cond in self.conditions:
            result = self.results.get(cond.name)
            if result is None:
                continue
            vals = [str(v) for v in cond.variables.values()]
            vals.extend([
                f"{result.mean_score:.3f}",
                f"{result.completed}/{result.total_prompts}",
                f"{result.mean_latency_s:.0f}s",
            ])
            lines.append(" & ".join(vals) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        return "\n".join(lines)

    def summary(self) -> str:
        lines = ["Ablation Study Results", "=" * 60]
        for cond in self.conditions:
            result = self.results.get(cond.name)
            if result:
                lines.append(
                    f"  {cond.name:<40} score={result.mean_score:.4f} "
                    f"({result.completed}/{result.total_prompts})"
                )
            else:
                lines.append(f"  {cond.name:<40} FAILED")
        lines.append(f"\nTotal time: {self.total_duration_s:.0f}s")
        return "\n".join(lines)


# Default ablation matrix for the paper
DEFAULT_ABLATION_VARIABLES = [
    AblationVariable(
        name="stage_gating",
        values=[False, True],
        description="Progressive tool disclosure via stage-gating",
    ),
    AblationVariable(
        name="skills_enabled",
        values=[False, True],
        description="Whether skills are loaded and used by the agent",
    ),
    AblationVariable(
        name="max_iterations",
        values=[1, 3, 5],
        description="Number of generate-verify-improve iterations",
    ),
]


def build_ablation_conditions(
    variables: list[AblationVariable] | None = None,
) -> list[AblationCondition]:
    """Generate all combinations of ablation variables."""
    variables = variables or DEFAULT_ABLATION_VARIABLES
    names = [v.name for v in variables]
    value_lists = [v.values for v in variables]

    conditions: list[AblationCondition] = []
    for combo in itertools.product(*value_lists):
        var_dict = dict(zip(names, combo))
        name = "_".join(f"{k}={v}" for k, v in var_dict.items())
        conditions.append(AblationCondition(name=name, variables=var_dict))

    return conditions


class AblationRunner:
    """Run a full ablation matrix."""

    def __init__(
        self,
        suite: str,
        data_path: str,
        output_dir: str = "ablation_results",
        api_key: str = "",
        server_address: str = "127.0.0.1:8188",
        model: str = "anthropic/claude-sonnet-4-5",
        image_model: str | None = None,
        workflow_path: str | None = None,
        max_prompts: int | None = None,
        conditions: list[AblationCondition] | None = None,
    ) -> None:
        self.suite = suite
        self.data_path = data_path
        self.output_dir = output_dir
        self.api_key = api_key
        self.server_address = server_address
        self.model = model
        self.image_model = image_model
        self.workflow_path = workflow_path
        self.max_prompts = max_prompts
        self.conditions = conditions or build_ablation_conditions()

    def run(self) -> AblationResult:
        """Execute all ablation conditions."""
        os.makedirs(self.output_dir, exist_ok=True)
        t0 = time.time()
        results: dict[str, BenchmarkResult] = {}

        for i, cond in enumerate(self.conditions):
            log.info(
                "Ablation %d/%d: %s", i + 1, len(self.conditions), cond.name
            )
            try:
                result = self._run_condition(cond)
                results[cond.name] = result
                log.info(
                    "  -> score=%.4f (%d/%d)",
                    result.mean_score, result.completed, result.total_prompts,
                )
            except Exception as exc:
                log.error("Condition %s failed: %s", cond.name, exc)

        final = AblationResult(
            conditions=self.conditions,
            results=results,
            total_duration_s=time.time() - t0,
        )

        # Save results
        summary_path = os.path.join(self.output_dir, "ablation_summary.json")
        self._save(final, summary_path)

        latex_path = os.path.join(self.output_dir, "ablation_table.tex")
        with open(latex_path, "w") as f:
            f.write(final.to_latex_table())
        log.info("LaTeX table saved to %s", latex_path)

        return final

    def _run_condition(self, cond: AblationCondition) -> BenchmarkResult:
        """Run one ablation condition."""
        variables = cond.variables
        bench_cfg = BenchmarkConfig(
            suite=self.suite,
            name=f"ablation_{cond.name}",
            data_path=self.data_path,
            output_dir=self.output_dir,
            max_iterations=variables.get("max_iterations", 3),
            server_address=self.server_address,
            api_key=self.api_key,
            model=variables.get("llm_backend", self.model),
            image_model=self.image_model,
            workflow_path=self.workflow_path,
            stage_gated=variables.get("stage_gating", False),
            max_prompts=self.max_prompts,
        )
        runner = BenchmarkRunner(bench_cfg)
        return runner.run()

    def _save(self, result: AblationResult, path: str) -> None:
        data: dict[str, Any] = {
            "conditions": [
                {"name": c.name, "variables": c.variables}
                for c in result.conditions
            ],
            "results": {},
            "total_duration_s": result.total_duration_s,
        }
        for name, br in result.results.items():
            data["results"][name] = {
                "mean_score": br.mean_score,
                "completed": br.completed,
                "failed": br.failed,
                "total_prompts": br.total_prompts,
                "scores": br.scores,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation matrix")
    parser.add_argument("--suite", type=str, default="geneval2")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output", type=str, default="ablation_results")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--image-model", type=str, default=None)
    parser.add_argument("--workflow", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    runner = AblationRunner(
        suite=args.suite,
        data_path=args.data_path,
        output_dir=args.output,
        api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
        image_model=args.image_model,
        workflow_path=args.workflow,
        max_prompts=args.max_prompts,
    )
    result = runner.run()
    print(result.summary())
    print("\nLaTeX Table:")
    print(result.to_latex_table())


if __name__ == "__main__":
    main()
