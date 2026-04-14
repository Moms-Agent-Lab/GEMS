"""
ScaleExperiment — Run 100+ prompt experiments across benchmarks and baselines.

Supports:
- GenEval2 (compositional)
- T2I-CompBench++ (attribute binding, spatial, non-spatial)
- Multiple baselines: raw model, prompt-rewrite, GEMS, ComfyClaw variants

Usage::

    python -m comfyclaw.benchmark.scale_experiment \
        --config experiment_config.json \
        --output experiment_results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .runner import BenchmarkConfig, BenchmarkResult, BenchmarkRunner

log = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Configuration for one baseline method."""

    name: str
    method: str  # "raw" | "prompt_rewrite" | "gems" | "comfyclaw"
    stage_gated: bool = False
    skills_dir: str | None = None
    max_iterations: int = 1
    description: str = ""


@dataclass
class ScaleExperimentConfig:
    """Configuration for a full-scale experiment."""

    name: str = "comfyclaw_scale"
    output_dir: str = "experiment_results"
    api_key: str = ""
    server_address: str = "127.0.0.1:8188"
    model: str = "anthropic/claude-sonnet-4-5"
    image_model: str | None = None
    workflow_path: str | None = None
    verifier_model: str | None = None

    benchmarks: list[dict] = field(default_factory=lambda: [
        {"suite": "geneval2", "data_path": "geneval2_prompts.json"},
    ])

    baselines: list[BaselineConfig] = field(default_factory=lambda: [
        BaselineConfig(
            name="raw_model",
            method="raw",
            max_iterations=1,
            description="Raw model output without agent intervention",
        ),
        BaselineConfig(
            name="prompt_rewrite",
            method="prompt_rewrite",
            max_iterations=1,
            description="LLM prompt rewriting only (no structural changes)",
        ),
        BaselineConfig(
            name="comfyclaw_1iter",
            method="comfyclaw",
            max_iterations=1,
            description="ComfyClaw with 1 iteration (no verify-improve loop)",
        ),
        BaselineConfig(
            name="comfyclaw_3iter",
            method="comfyclaw",
            max_iterations=3,
            description="ComfyClaw with 3 iterations",
        ),
        BaselineConfig(
            name="comfyclaw_staged",
            method="comfyclaw",
            stage_gated=True,
            max_iterations=3,
            description="ComfyClaw with stage-gated tool disclosure",
        ),
        BaselineConfig(
            name="comfyclaw_evolved",
            method="comfyclaw",
            stage_gated=True,
            max_iterations=3,
            skills_dir=None,  # will use evolved skills
            description="ComfyClaw with evolved skills + stage-gating",
        ),
    ])

    max_prompts_per_benchmark: int | None = None
    num_workers: int = 1


@dataclass
class ScaleExperimentResult:
    """Results from a full-scale experiment."""

    config_name: str
    results: dict[str, dict[str, BenchmarkResult]]  # benchmark -> baseline -> result
    total_prompts: int
    total_duration_s: float

    def summary(self) -> str:
        lines = [f"Scale Experiment: {self.config_name}", "=" * 60]
        for bench_name, baselines in self.results.items():
            lines.append(f"\n{bench_name}:")
            lines.append(f"  {'Method':<25} {'Score':>8} {'Completed':>10}")
            lines.append(f"  {'-'*25} {'-'*8} {'-'*10}")
            for bl_name, result in baselines.items():
                lines.append(
                    f"  {bl_name:<25} {result.mean_score:>8.4f} "
                    f"{result.completed:>5}/{result.total_prompts}"
                )
        lines.append(f"\nTotal prompts: {self.total_prompts}")
        lines.append(f"Total time: {self.total_duration_s:.0f}s")
        return "\n".join(lines)


class ScaleExperimentRunner:
    """Orchestrate a full-scale experiment with multiple baselines and benchmarks."""

    def __init__(self, config: ScaleExperimentConfig) -> None:
        self.config = config

    def run(self) -> ScaleExperimentResult:
        """Execute the full experiment."""
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        t0 = time.time()

        all_results: dict[str, dict[str, BenchmarkResult]] = {}
        total_prompts = 0

        for bench_info in cfg.benchmarks:
            suite = bench_info["suite"]
            data_path = bench_info["data_path"]
            log.info("=" * 60)
            log.info("BENCHMARK: %s", suite)
            log.info("=" * 60)

            baseline_results: dict[str, BenchmarkResult] = {}

            for baseline in cfg.baselines:
                log.info("--- Baseline: %s ---", baseline.name)
                exp_name = f"{cfg.name}_{suite}_{baseline.name}"
                bench_cfg = BenchmarkConfig(
                    suite=suite,
                    name=exp_name,
                    data_path=data_path,
                    output_dir=cfg.output_dir,
                    max_iterations=baseline.max_iterations,
                    num_workers=cfg.num_workers,
                    server_address=cfg.server_address,
                    api_key=cfg.api_key,
                    model=cfg.model,
                    image_model=cfg.image_model,
                    workflow_path=cfg.workflow_path,
                    stage_gated=baseline.stage_gated,
                    verifier_model=cfg.verifier_model,
                    max_prompts=cfg.max_prompts_per_benchmark,
                )

                runner = BenchmarkRunner(bench_cfg)
                try:
                    result = runner.run()
                    baseline_results[baseline.name] = result
                    total_prompts += result.total_prompts
                    log.info(
                        "%s: mean_score=%.4f (%d/%d completed)",
                        baseline.name, result.mean_score,
                        result.completed, result.total_prompts,
                    )
                except Exception as exc:
                    log.error("Baseline %s failed: %s", baseline.name, exc)

            all_results[suite] = baseline_results

        final = ScaleExperimentResult(
            config_name=cfg.name,
            results=all_results,
            total_prompts=total_prompts,
            total_duration_s=time.time() - t0,
        )

        # Save summary
        summary_path = os.path.join(cfg.output_dir, f"{cfg.name}_summary.json")
        self._save_summary(final, summary_path)
        print(final.summary())

        return final

    def _save_summary(self, result: ScaleExperimentResult, path: str) -> None:
        """Save experiment results to JSON."""
        data: dict[str, Any] = {
            "config_name": result.config_name,
            "total_prompts": result.total_prompts,
            "total_duration_s": result.total_duration_s,
            "results": {},
        }
        for bench, baselines in result.results.items():
            data["results"][bench] = {}
            for bl_name, br in baselines.items():
                data["results"][bench][bl_name] = {
                    "mean_score": br.mean_score,
                    "completed": br.completed,
                    "failed": br.failed,
                    "total_prompts": br.total_prompts,
                    "mean_latency_s": br.mean_latency_s,
                    "scores": br.scores,
                }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log.info("Summary saved to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scaled ComfyClaw experiments")
    parser.add_argument("--config", type=str, help="Path to experiment config JSON")
    parser.add_argument("--output", type=str, default="experiment_results")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--suite", type=str, default="geneval2")
    parser.add_argument("--data-path", type=str, default="geneval2_prompts.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.config:
        with open(args.config) as f:
            cfg_data = json.load(f)
        cfg = ScaleExperimentConfig(**cfg_data)
    else:
        cfg = ScaleExperimentConfig(
            output_dir=args.output,
            api_key=args.api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            benchmarks=[{"suite": args.suite, "data_path": args.data_path}],
            max_prompts_per_benchmark=args.max_prompts,
        )

    runner = ScaleExperimentRunner(cfg)
    result = runner.run()
    print(result.summary())


if __name__ == "__main__":
    main()
