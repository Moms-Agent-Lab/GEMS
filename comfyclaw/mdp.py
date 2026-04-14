r"""
Skill-Augmented Workflow MDP — Formal formulation for ComfyClaw.

This module encodes the theoretical framework for the NeurIPS paper:
ComfyClaw's agentic image generation as a Markov Decision Process with
stage-gating as action space reduction and skills as reusable macro-actions.

Mathematical Formulation
========================

We define the **Skill-Augmented Workflow MDP** as a 6-tuple:

    M = (S, A, T, R, γ, Σ)

State Space  S
--------------
A state s ∈ S is a tuple (G, p, k, h) where:
  - G = (V, E) is the current ComfyUI workflow DAG (nodes V, edges E)
  - p ∈ P is the text prompt
  - k ∈ {1,...,K} is the current pipeline stage
  - h = (s₁,a₁,r₁,...,sₜ) is the interaction history

Action Space  A
---------------
The unrestricted action space A is the union of all node-level operations:

    A = A_add ∪ A_connect ∪ A_param ∪ A_delete ∪ A_meta

where:
  - A_add     = {add_node(class_type, inputs)}
  - A_connect = {connect(src, slot, dst, input)}
  - A_param   = {set_param(node, param, value)}
  - A_delete  = {delete_node(node_id)}
  - A_meta    = {finalize, inspect, validate, query_models, report_strategy}

**Stage-gating** restricts the available actions at each stage:

    A(k) ⊂ A,    |A(k)| ≪ |A|

For K=5 stages (planning, construction, conditioning, enhancement,
finalization), the average action space reduction is:

    E[|A(k)|] / |A| ≈ 0.3

This is equivalent to an attention mask in the action selection process.

Transition Dynamics  T
----------------------
    T: S × A → Δ(S)

The transition is deterministic for graph operations (add/connect/param/delete)
but stochastic for the external verification step:

    s' = (G', p, k', h')  where  G' = apply(a, G)
                                  k' = stage_advance(k, a)
                                  h' = h ∥ (s, a, r)

Reward Function  R
------------------
    R(s, a, s') = R_quality(s') + R_complexity(s') + R_feasibility(s')

where:
  - R_quality    = VLM_score(image(G'), p)                    ∈ [0, 1]
  - R_complexity = -λ · max(0, |V'| - N_base) / N_base       ∈ (-∞, 0]
  - R_feasibility = -∞  if G' fails ComfyUI validation        {-∞, 0}

The VLM verifier provides a decomposed reward:

    R_quality = w_req · (Σᵢ pass(reqᵢ) / |req|) + w_detail · score_detail

Discount Factor  γ
-------------------
    γ = 0 (episodic — each prompt is a fresh episode)

Skill Library  Σ
-----------------
A skill σ ∈ Σ is a reusable macro-action (option in the options framework):

    σ = (Iσ, πσ, βσ, dσ)

where:
  - Iσ ⊂ S : initiation set (when the skill is applicable)
  - πσ : S → A : skill policy (the SKILL.md instructions)
  - βσ : S → [0,1] : termination condition
  - dσ : textual description used for skill retrieval

The agent selects skills based on the current state using a retrieval function:

    retrieve: S → 2^Σ

Currently keyword-based; upgradeable to embedding-based Skill-RAG (see
``skill_rag.py``).

Stage-Gating as Action Space Reduction
---------------------------------------
Stage-gating defines a partition of the tool set:

    A = ⊔_{k=1}^{K} A(k)    (disjoint union, with overlap for meta-tools)

The stage router π_stage : S → {1,...,K} determines the current stage based
on the workflow structure:

    k = π_stage(G) = argmax_k score_k(G)

where score_k(G) measures how "complete" stage k is in the current graph.

This provably reduces the branching factor:

    Theorem 1 (Action Space Reduction):
    Let b = |A| be the total branching factor and bₖ = |A(k)| for stage k.
    The expected number of LLM evaluations per step is O(bₖ) instead of O(b).
    For K=5 balanced stages, this gives ≈ 3.3× speedup.

Self-Evolution as MDP Policy Improvement
-----------------------------------------
Skill self-evolution corresponds to policy improvement in the options framework:

    Σ_{n+1} = improve(Σ_n, D_n)

where D_n is the experience dataset from cycle n. The improvement operator:
  1. Clusters failures in D_n by pattern
  2. Proposes skill mutations (create/update/merge/delete)
  3. Validates mutations against held-out prompts (skill grounding)
  4. Commits only if R improves above threshold

This is equivalent to a form of **evolutionary strategy** over the skill space
with fitness = R_quality - λ·R_complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .stage_router import STAGES, STAGE_TOOLS


@dataclass
class MDPState:
    """Represents a state in the Skill-Augmented Workflow MDP."""

    workflow: dict  # G = (V, E) as ComfyUI API dict
    prompt: str  # p
    stage: str  # k ∈ {planning, construction, conditioning, enhancement, finalization}
    iteration: int  # t
    history: list[dict] = field(default_factory=list)  # h

    @property
    def node_count(self) -> int:
        return len(self.workflow)

    @property
    def stage_index(self) -> int:
        return STAGES.index(self.stage) if self.stage in STAGES else 0


@dataclass
class MDPAction:
    """Represents an action in the MDP."""

    action_type: str  # add_node, connect, set_param, delete_node, meta
    tool_name: str
    arguments: dict = field(default_factory=dict)
    stage: str = ""


@dataclass
class MDPReward:
    """Decomposed reward signal."""

    quality: float = 0.0  # VLM verifier score
    complexity: float = 0.0  # -λ * excess_nodes / baseline
    feasibility: float = 0.0  # 0 if valid, -inf if broken
    total: float = 0.0

    @classmethod
    def compute(
        cls,
        verifier_score: float,
        node_count: int,
        baseline_nodes: int = 10,
        complexity_lambda: float = 0.02,
        is_valid: bool = True,
    ) -> MDPReward:
        quality = verifier_score
        excess = max(0, node_count - baseline_nodes)
        complexity = -complexity_lambda * excess / max(baseline_nodes, 1)
        feasibility = 0.0 if is_valid else -1e6
        return cls(
            quality=quality,
            complexity=complexity,
            feasibility=feasibility,
            total=quality + complexity + feasibility,
        )


def action_space_reduction_factor() -> dict[str, float]:
    """Compute the action space reduction factor for each stage.

    Returns a dict mapping stage name to the fraction of total tools
    available at that stage. Lower = more constrained = faster search.
    """
    all_tools: set[str] = set()
    for tools in STAGE_TOOLS.values():
        all_tools.update(tools)
    total = len(all_tools)

    result: dict[str, float] = {}
    for stage in STAGES:
        stage_tools = STAGE_TOOLS.get(stage, set())
        result[stage] = len(stage_tools) / total if total else 1.0

    result["mean"] = sum(result.values()) / len(result) if result else 1.0
    return result


@dataclass
class Skill:
    """Formal representation of a skill (option) in the MDP."""

    name: str
    description: str  # dσ: used for retrieval
    applicable_stages: list[str] = field(default_factory=list)  # Iσ
    keywords: list[str] = field(default_factory=list)
    body: str = ""  # πσ: the policy instructions

    @property
    def initiation_set_description(self) -> str:
        return f"Stages: {self.applicable_stages}, Keywords: {self.keywords}"
