---
name: explore
description: >-
  Explore the ComfyUI node ecosystem and discover pipeline stages autonomously.
  Use when you need to understand what ComfyUI nodes are available, classify them
  into workflow stages (loading, conditioning, sampling, post-processing, output),
  or build a stage map that determines which tools are relevant at each phase of
  workflow construction. Trigger this skill before building unfamiliar pipelines,
  when encountering unknown node types, or when the agent needs to discover its
  own toolset from scratch.
license: MIT
metadata:
  author: comfyclaw
  version: "1.0.0"
tags: [meta]
---

# Explore — ComfyUI Node and Workflow Stage Discovery

This skill enables autonomous exploration of the ComfyUI node ecosystem.
The agent queries the server's `/object_info` endpoint to enumerate every
available node class, then classifies nodes into pipeline stages based on
their input/output type signatures.

## When to use

- At the start of a session to discover what the server supports
- Before building a workflow for an unfamiliar architecture
- When self-evolve needs to understand which tools map to which stages
- When a new custom node pack is installed and its capabilities are unknown

## Exploration protocol

### Step 1 — Enumerate nodes

Run the `explore_nodes` tool (or the helper script `scripts/explore_nodes.py`)
against the ComfyUI server. This queries `/object_info` and returns a JSON
manifest of every registered node class with:

- `class_type`: The node's registered name (e.g. `KSampler`, `VAEDecode`)
- `category`: The UI category path (e.g. `sampling`, `latent/transform`)
- `input_types`: Map of input names to their expected types
- `output_types`: List of output type strings
- `description`: Human-readable description if available

### Step 2 — Classify into pipeline stages

Analyze each node's type signature and assign it to one or more pipeline stages:

| Stage | Key I/O types | Example nodes |
|-------|--------------|---------------|
| **Loading** | Outputs MODEL, CLIP, VAE, CONTROL_NET | CheckpointLoaderSimple, UNETLoader, VAELoader, CLIPLoader, LoraLoader, ControlNetLoader |
| **Conditioning** | Consumes CLIP, outputs CONDITIONING | CLIPTextEncode, ConditioningCombine, ConditioningAverage, FluxGuidance |
| **Sampling** | Consumes MODEL + CONDITIONING + LATENT, outputs LATENT | KSampler, KSamplerAdvanced, SamplerCustom |
| **Latent ops** | Consumes/outputs LATENT | EmptyLatentImage, LatentUpscaleBy, LatentComposite, LatentBlend |
| **Decoding** | Consumes LATENT + VAE, outputs IMAGE | VAEDecode, VAEEncode |
| **Image post-processing** | Consumes/outputs IMAGE | ImageScale, ImageSharpen, ImageBlend, SaveImage, PreviewImage |
| **Control** | Consumes IMAGE, outputs CONTROL_NET hints | CannyEdgePreprocessor, DepthAnythingPreprocessor, OpenPosePreprocessor |

Classification rules:
1. If a node outputs MODEL, CLIP, or VAE as its primary output -> **Loading**
2. If a node outputs CONDITIONING -> **Conditioning**
3. If a node consumes MODEL + CONDITIONING and outputs LATENT -> **Sampling**
4. If a node only handles LATENT <-> LATENT transforms -> **Latent ops**
5. If a node consumes LATENT + VAE and outputs IMAGE -> **Decoding**
6. If a node primarily handles IMAGE -> **Image post-processing**
7. If a node produces preprocessed hints for ControlNet -> **Control**

### Step 3 — Map agent tools to stages

Based on the node classification, map ComfyClaw's 16 tools to pipeline stages:

| Stage | Agent tools |
|-------|-------------|
| **Planning** | `inspect_workflow`, `report_evolution_strategy`, `read_skill`, `query_available_models` |
| **Construction** | `add_node`, `connect_nodes`, `delete_node`, `set_param` |
| **Conditioning** | `set_prompt`, `add_regional_attention`, `add_controlnet` |
| **Enhancement** | `add_lora_loader`, `add_hires_fix`, `add_inpaint_pass` |
| **Finalization** | `validate_workflow`, `finalize_workflow` |

### Step 4 — Output stage map

Write the results to a `stage_map.json` with this structure:

```json
{
  "server_address": "127.0.0.1:8188",
  "timestamp": "2026-04-13T12:00:00Z",
  "total_nodes_discovered": 150,
  "stages": {
    "loading": {
      "description": "Model, CLIP, VAE, and adapter loading",
      "node_classes": ["CheckpointLoaderSimple", "UNETLoader", "..."],
      "agent_tools": ["query_available_models", "add_node", "set_param"]
    },
    "conditioning": {
      "description": "Prompt encoding and conditioning control",
      "node_classes": ["CLIPTextEncode", "ConditioningCombine", "..."],
      "agent_tools": ["set_prompt", "add_regional_attention", "add_controlnet"]
    }
  },
  "unclassified_nodes": ["CustomNode1", "..."]
}
```

## Using the stage map

Once the stage map is generated, it serves as input for:
1. **workflow.skill**: Determines which tools to expose at each pipeline phase
2. **self-evolve.skill**: Identifies which stages have weak coverage and need new skills
3. **The agent itself**: When building a workflow, consult the stage map to understand what nodes are available for each step

## Helper script

Run `scripts/explore_nodes.py` for automated exploration:

```bash
python -m comfyclaw.skills.explore.scripts.explore_nodes \
  --server 127.0.0.1:8188 \
  --output stage_map.json
```
