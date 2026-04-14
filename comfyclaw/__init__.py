"""
ComfyClaw — agentic harness for self-evolving ComfyUI workflows.

Public API
----------
>>> from comfyclaw import ClawHarness, HarnessConfig
>>> cfg = HarnessConfig(api_key="sk-ant-...", max_iterations=3)
>>> with ClawHarness.from_workflow_file("workflow_api.json", cfg) as h:
...     image_bytes = h.run("a red fox at dawn, photorealistic")
"""

from pathlib import Path

from .client import ComfyClient
from .curriculum import CurriculumRunner
from .evolve import SkillEvolver
from .experience_db import ExperienceDB
from .harness import ClawHarness, HarnessConfig
from .mdp import MDPReward, MDPState
from .memory import ClawMemory
from .model_orchestrator import ModelOrchestrator
from .skill_grounding import SkillGrounding
from .skill_rag import SkillRAG
from .skill_store import SkillStore
from .stage_router import StageRouter
from .sync_server import SyncServer
from .verifier import ClawVerifier, RegionIssue, VerifierResult
from .workflow import WorkflowManager


def custom_node_path() -> Path:
    """Return the filesystem path to the bundled ComfyClaw-Sync ComfyUI custom node."""
    return Path(__file__).resolve().parent / "custom_node"


__all__ = [
    "ClawHarness",
    "HarnessConfig",
    "ClawMemory",
    "ComfyClient",
    "ClawVerifier",
    "CurriculumRunner",
    "ExperienceDB",
    "MDPReward",
    "MDPState",
    "ModelOrchestrator",
    "SkillGrounding",
    "SkillRAG",
    "VerifierResult",
    "RegionIssue",
    "SkillEvolver",
    "SkillStore",
    "StageRouter",
    "SyncServer",
    "WorkflowManager",
    "custom_node_path",
]

__version__ = "0.1.0"
