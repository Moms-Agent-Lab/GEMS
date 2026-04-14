"""
ModelOrchestrator — Multi-model orchestration for ComfyClaw.

Implements a two-phase generation strategy:
1. Fast exploration: use a lightweight model (SD1.5, SDXL-Turbo) with
   few steps to rapidly iterate on workflow topology.
2. Quality generation: once the topology is stable, switch to a
   high-quality model (SDXL, Flux, Qwen-Image) for final output.

Also handles:
- Automatic GPU/VRAM detection and model tier selection.
- Dynamic model swapping during the harness loop.
- OOM recovery by downgrading to a smaller model.

Usage::

    orchestrator = ModelOrchestrator(server_address="127.0.0.1:8188")
    explore_model = orchestrator.get_exploration_model()
    quality_model = orchestrator.get_quality_model()
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Detected GPU capabilities."""

    name: str = "unknown"
    vram_mb: int = 0
    compute_capability: str = ""
    driver_version: str = ""
    cuda_version: str = ""

    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024.0

    @property
    def tier(self) -> str:
        vram = self.vram_gb
        if vram < 6:
            return "minimal"
        elif vram < 8:
            return "basic"
        elif vram < 12:
            return "standard"
        elif vram < 16:
            return "high"
        elif vram < 24:
            return "premium"
        elif vram < 48:
            return "full"
        else:
            return "unrestricted"


@dataclass
class ModelConfig:
    """Configuration for a ComfyUI model."""

    name: str
    filename: str
    architecture: str  # sd15, sdxl, flux, qwen, sd3
    vram_required_mb: int
    default_resolution: tuple[int, int] = (512, 512)
    default_steps: int = 20
    default_cfg: float = 7.0
    weight_dtype: str = "default"
    sampler: str = "euler"
    scheduler: str = "normal"
    is_fast: bool = False

    @property
    def resolution_str(self) -> str:
        return f"{self.default_resolution[0]}x{self.default_resolution[1]}"


# Model catalog
MODELS: dict[str, ModelConfig] = {
    "sd15_fast": ModelConfig(
        name="SD 1.5 (Fast)",
        filename="DreamShaper_8_pruned.safetensors",
        architecture="sd15",
        vram_required_mb=4096,
        default_resolution=(512, 512),
        default_steps=10,
        default_cfg=7.0,
        sampler="euler",
        is_fast=True,
    ),
    "sd15_quality": ModelConfig(
        name="SD 1.5 (Quality)",
        filename="DreamShaper_8_pruned.safetensors",
        architecture="sd15",
        vram_required_mb=4096,
        default_resolution=(512, 512),
        default_steps=25,
        default_cfg=7.0,
        sampler="dpmpp_2m",
    ),
    "sdxl_turbo": ModelConfig(
        name="SDXL Turbo",
        filename="sd_xl_turbo_1.0_fp16.safetensors",
        architecture="sdxl",
        vram_required_mb=8192,
        default_resolution=(1024, 1024),
        default_steps=4,
        default_cfg=1.0,
        sampler="euler",
        is_fast=True,
    ),
    "sdxl_quality": ModelConfig(
        name="SDXL 1.0",
        filename="sd_xl_base_1.0.safetensors",
        architecture="sdxl",
        vram_required_mb=10240,
        default_resolution=(1024, 1024),
        default_steps=25,
        default_cfg=7.0,
        sampler="dpmpp_2m",
    ),
    "flux_dev_fp8": ModelConfig(
        name="Flux Dev (FP8)",
        filename="flux1-dev-fp8.safetensors",
        architecture="flux",
        vram_required_mb=16384,
        default_resolution=(1024, 1024),
        default_steps=20,
        default_cfg=1.0,
        weight_dtype="fp8_e4m3fn",
        sampler="euler",
    ),
    "qwen_image_fp8": ModelConfig(
        name="Qwen Image (FP8)",
        filename="qwen_image_2512_fp8_e4m3fn.safetensors",
        architecture="qwen",
        vram_required_mb=16384,
        default_resolution=(1328, 1328),
        default_steps=25,
        default_cfg=7.0,
        weight_dtype="fp8_e4m3fn",
        sampler="dpmpp_2m",
    ),
}


class ModelOrchestrator:
    """Multi-model orchestration engine.

    Parameters
    ----------
    server_address : ComfyUI server address for model availability checks.
    force_gpu_info : Override GPU detection with provided info.
    """

    def __init__(
        self,
        server_address: str = "127.0.0.1:8188",
        force_gpu_info: GPUInfo | None = None,
    ) -> None:
        self.server_address = server_address
        self.gpu_info = force_gpu_info or self._detect_gpu()
        self._available_models: list[str] | None = None

        log.info(
            "GPU: %s (%.1f GB VRAM, tier=%s)",
            self.gpu_info.name, self.gpu_info.vram_gb, self.gpu_info.tier,
        )

    def _detect_gpu(self) -> GPUInfo:
        """Detect GPU capabilities via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    return GPUInfo(
                        name=parts[0].strip(),
                        vram_mb=int(float(parts[1].strip())),
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as exc:
            log.warning("GPU detection failed: %s", exc)

        return GPUInfo(name="unknown", vram_mb=8192)

    def get_available_models(self) -> list[str]:
        """Query ComfyUI for available checkpoint files."""
        if self._available_models is not None:
            return self._available_models

        try:
            import urllib.request
            url = f"http://{self.server_address}/object_info/CheckpointLoaderSimple"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
                ckpt_info = data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {})
                ckpt_names = ckpt_info.get("ckpt_name", [[]])[0]
                self._available_models = ckpt_names
                return ckpt_names
        except Exception as exc:
            log.warning("Failed to query available models: %s", exc)
            return []

    def get_exploration_model(self) -> ModelConfig:
        """Select the best fast model for topology exploration."""
        available = self.get_available_models()
        tier = self.gpu_info.tier

        preferences = ["sd15_fast", "sdxl_turbo"]
        if tier in ("premium", "full", "unrestricted"):
            preferences = ["sdxl_turbo", "sd15_fast"]

        for model_key in preferences:
            cfg = MODELS[model_key]
            if cfg.vram_required_mb <= self.gpu_info.vram_mb:
                if not available or cfg.filename in available:
                    log.info("Exploration model: %s", cfg.name)
                    return cfg

        fallback = MODELS["sd15_fast"]
        log.info("Exploration model (fallback): %s", fallback.name)
        return fallback

    def get_quality_model(self) -> ModelConfig:
        """Select the best quality model for final generation."""
        available = self.get_available_models()
        tier = self.gpu_info.tier

        # Ordered by quality (best first)
        quality_order = ["qwen_image_fp8", "flux_dev_fp8", "sdxl_quality", "sd15_quality"]

        for model_key in quality_order:
            cfg = MODELS[model_key]
            if cfg.vram_required_mb <= self.gpu_info.vram_mb:
                if not available or cfg.filename in available:
                    log.info("Quality model: %s", cfg.name)
                    return cfg

        fallback = MODELS["sd15_quality"]
        log.info("Quality model (fallback): %s", fallback.name)
        return fallback

    def get_model_for_iteration(self, iteration: int, max_iterations: int) -> ModelConfig:
        """Select the appropriate model for a given iteration.

        Strategy:
        - First half of iterations: use fast exploration model.
        - Last iteration(s): switch to quality model.
        """
        if max_iterations <= 1:
            return self.get_quality_model()

        if iteration < max_iterations:
            return self.get_exploration_model()
        else:
            return self.get_quality_model()

    def handle_oom(self, current_model: ModelConfig) -> ModelConfig | None:
        """Suggest a smaller model after an OOM error.

        Returns None if no smaller model is available.
        """
        smaller: dict[str, list[str]] = {
            "qwen": ["flux_dev_fp8", "sdxl_quality", "sd15_quality"],
            "flux": ["sdxl_quality", "sd15_quality"],
            "sdxl": ["sd15_quality"],
            "sd15": [],
        }

        fallbacks = smaller.get(current_model.architecture, [])
        for model_key in fallbacks:
            cfg = MODELS[model_key]
            if cfg.vram_required_mb <= self.gpu_info.vram_mb:
                log.info("OOM recovery: downgrading from %s to %s", current_model.name, cfg.name)
                return cfg

        log.warning("No smaller model available for OOM recovery")
        return None

    def apply_to_workflow(
        self,
        workflow: dict,
        model_config: ModelConfig,
    ) -> list[tuple[str, str, Any]]:
        """Apply model configuration to a workflow.

        Returns a list of (node_id, param_name, new_value) changes.
        """
        changes: list[tuple[str, str, Any]] = []

        for nid, node in workflow.items():
            ct = node.get("class_type", "")
            inputs = node.get("inputs", {})

            if ct in ("CheckpointLoaderSimple", "CheckpointLoader"):
                inputs["ckpt_name"] = model_config.filename
                changes.append((nid, "ckpt_name", model_config.filename))

            elif ct == "KSampler":
                inputs["steps"] = model_config.default_steps
                inputs["cfg"] = model_config.default_cfg
                inputs["sampler_name"] = model_config.sampler
                inputs["scheduler"] = model_config.scheduler
                changes.append((nid, "steps", model_config.default_steps))
                changes.append((nid, "cfg", model_config.default_cfg))

            elif ct == "EmptyLatentImage":
                w, h = model_config.default_resolution
                inputs["width"] = w
                inputs["height"] = h
                changes.append((nid, "width", w))
                changes.append((nid, "height", h))

        return changes
