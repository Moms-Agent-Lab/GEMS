---
name: compute
description: >-
  Resource-aware model selection and configuration based on local GPU/VRAM
  availability. Use when setting up a new ComfyUI session, choosing between
  model architectures, or when encountering out-of-memory errors. Detects
  GPU capabilities via nvidia-smi or ComfyUI system info and recommends
  the optimal model tier, weight dtype, batch size, and resolution for the
  available hardware.
license: MIT
metadata:
  author: comfyclaw
  version: "1.0.0"
tags: [agent]
---

# Compute — Resource-Aware Model Selection

This skill detects the available GPU hardware and recommends the optimal
model configuration for ComfyUI.

## GPU detection

Query GPU info using one of these methods:
1. `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`
2. ComfyUI `/system_stats` endpoint (if available)
3. Fall back to conservative defaults if detection fails

## Model tier selection

| VRAM | Recommended tier | Models | weight_dtype |
|------|-----------------|--------|-------------|
| < 6 GB | Minimal | SD 1.5 (512x512) | default |
| 6-8 GB | Basic | SD 1.5, SDXL (with attention slicing) | default |
| 8-12 GB | Standard | SDXL, Qwen-Image (FP8) | fp8_e4m3fn |
| 12-16 GB | High | SDXL, Flux-dev (FP8), Qwen-Image (FP8) | fp8_e4m3fn |
| 16-24 GB | Premium | Flux-dev, Qwen-Image, SD3 | fp8_e4m3fn |
| 24-48 GB | Full | All models at native precision | default or fp8_e4m3fn |
| 48+ GB | Unrestricted | All models, large batch sizes | default |

## Configuration adjustments

### Resolution scaling
- < 8 GB VRAM: Max 512x512 (SD 1.5) or 768x768
- 8-12 GB: Max 1024x1024
- 12-16 GB: Max 1328x1328 (Qwen native)
- 16+ GB: Full resolution for any model

### Batch size
- < 12 GB: batch_size=1
- 12-24 GB: batch_size=1-2
- 24+ GB: batch_size=1-4

### Weight dtype selection
| Platform | VRAM | Recommended dtype |
|----------|------|-------------------|
| CUDA | < 12 GB | fp8_e4m3fn |
| CUDA | 12+ GB | fp8_e4m3fn or default |
| Apple MPS | Any | default (FP8 not supported on MPS) |
| CPU | Any | default |

### KSampler tuning for low VRAM
- Use fewer steps (15 instead of 25 for SDXL)
- Prefer `euler` sampler (faster, lower peak VRAM vs. `dpmpp_2m`)
- Disable hires-fix if VRAM < 12 GB
- Use `--lowvram` or `--novram` ComfyUI flags when VRAM < 6 GB

## API selection for external models

When local GPU is insufficient, recommend cloud API alternatives:
- VRAM < 6 GB and user wants Flux/SDXL → suggest API-based generation
- ComfyUI server running on remote machine → adjust for network latency
- Multiple GPUs → suggest load balancing across ComfyUI instances

## Error recovery

| Error | Likely cause | Fix |
|-------|-------------|-----|
| CUDA out of memory | Model too large for VRAM | Switch to FP8, reduce resolution, or use a smaller model |
| Float8_e4m3fn MPS error | FP8 not supported on Apple Silicon | Set weight_dtype to "default" |
| Allocation failed | Peak VRAM exceeded during sampling | Reduce steps, disable hires-fix, or lower resolution |
| Slow generation (>5 min) | Model offloaded to CPU/disk | Use a smaller model or add `--lowvram` flag |
