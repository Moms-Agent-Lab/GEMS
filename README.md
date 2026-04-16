
<div align="center">


# <img src="assets/logo.png" width="40" style="vertical-align: -25%; margin-right: -5px;"> GEMS: Agent-Native Multimodal Generation with Memory and Skills

<a href="https://arxiv.org/abs/2603.28088"><img src="https://img.shields.io/badge/arXiv-paper-b31b1b?logo=arxiv&logoColor=white" alt="Paper"></a>&nbsp;&nbsp;<a href="https://gems-gen.github.io"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project-Page-2563eb" alt="Project Page"></a>&nbsp;&nbsp;
<a href="https://huggingface.co/papers/2603.28088"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-ffc107" alt="Paper"></a>



![Main Image](assets/main.png)

</div>

## Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Setup — MLLM (shared by both lines)](#setup--mllm-shared-by-both-lines)
- [Line 1 — HTTP Generation (`infer.py`)](#line-1--http-generation-inferpy)
- [Line 2 — ComfyUI Generation (`infer_comfy.py`)](#line-2--comfyui-generation-infer_comfypy)
- [Evaluation](#evaluation)
- [Skills](#skills)
- [Citation](#citation)

---

## Project Overview

GEMS ships with **two interchangeable image-generation lines**. Both share the same
decompose → generate → verify → refine pipeline and the same `SkillManager` / planner;
only the `generate()` backend differs.

| Line | Entry | Backend | Supported generators |
|---|---|---|---|
| **HTTP line** | `infer.py` | `agent/GEMS.py` → FastAPI server (`POST /generate?prompt=…`) | Qwen-Image-2512, Z-Image-Turbo |
| **ComfyUI line** | `infer_comfy.py` | `agent/comfy_gems.py` → ComfyUI REST API (`/prompt`, `/history`, `/view`) | Qwen-Image-2512, Z-Image-Turbo, FLUX.2 [klein] 9B, LongCat-Image |

```text
GEMS/
├── agent/
│   ├── server/                 # FastAPI image-gen servers (HTTP line)
│   │   ├── kimi.sh             # Kimi-K2.5 MLLM
│   │   ├── qwen_image.py       # Qwen-Image-2512 server
│   │   └── z_image.py          # Z-Image-Turbo server
│   ├── skills/                 # prompt-routing skills
│   │   ├── aesthetic_drawing/  #   legacy GEMS skills
│   │   ├── creative_drawing/
│   │   ├── spatial/
│   │   ├── text_rendering/
│   │   ├── qwen-image-2512/    #   ComfyUI model skills
│   │   ├── z-image-turbo/
│   │   ├── flux-klein-9b/
│   │   └── longcat-image/
│   ├── base_agent.py           # BaseAgent + LiteLLM config
│   ├── GEMS.py                 # pipeline (HTTP line)
│   ├── comfy_gems.py           # ComfyGEMS subclass (ComfyUI line)
│   ├── comfy_client.py         # minimal ComfyUI HTTP client
│   ├── comfy_workflow.py       # ComfyUI API-format templates
│   └── skill_manager.py        # loads SKILL.md (both formats)
├── eval/                       # GenEval2, CREA, ArtiMuse
├── infer.py                    # HTTP-line demo
└── infer_comfy.py              # ComfyUI-line demo
```

---

## Quick Start

```bash
git clone https://github.com/lcqysl/GEMS.git
cd GEMS
pip install requests litellm torch diffusers transformers fastapi uvicorn accelerate tqdm
```

Additional requirement for the ComfyUI line: a separate ComfyUI installation reachable
over HTTP (see [Line 2](#line-2--comfyui-generation-infer_comfypy)).

---

## Setup — MLLM (shared by both lines)

GEMS uses an MLLM for reasoning, verification, and prompt refinement. Pick one option.

### Option A — Cloud API via LiteLLM (recommended)

Supports Claude, GPT-4o, Gemini, and [any model LiteLLM covers](https://docs.litellm.ai/docs/providers).
The default config uses **Claude Sonnet 4.6**.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The model is set in `agent/base_agent.py`:

```python
LITELLM_MODEL = "anthropic/claude-sonnet-4-6"
```

Change this to switch providers (e.g. `"openai/gpt-4o"`, `"gemini/gemini-2.0-flash"`).

### Option B — Self-hosted via SGLang

Run [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) locally (requires 8× GPU):

```bash
pip install sglang
MODEL_PATH=/path/to/Kimi-K2.5 bash agent/server/kimi.sh
# starts on http://localhost:30000
```

Then set `mllm_url` in `infer.py` and switch `base_agent.py` back to the
OpenAI-compatible client.

---

## Line 1 — HTTP Generation (`infer.py`)

Uses a dedicated FastAPI server to serve a single model.

### 1. Download the generator weights

```bash
# Qwen-Image-2512
huggingface-cli download Qwen/Qwen-Image-2512 --local-dir /path/to/Qwen-Image-2512

# Z-Image-Turbo (faster, ~9 steps)
huggingface-cli download Tongyi-MAI/Z-Image-Turbo --local-dir /path/to/Z-Image-Turbo
```

### 2. Start the image-generation server

**Qwen-Image:**

```bash
MODEL_PATH=/path/to/Qwen-Image-2512 NUM_GPUS=1 python agent/server/qwen_image.py
# http://localhost:8000
```

- `NUM_GPUS` — number of GPU workers (load-balanced).
- `MODEL_PATH` — local weights dir.

**Z-Image-Turbo** (faster, 9 steps vs 50):

```bash
MODEL_PATH=/path/to/Z-Image-Turbo NUM_GPUS=1 PORT=8000 python agent/server/z_image.py
# http://localhost:8000 (default port is 8001)
```

Verify:

```bash
curl -X POST "http://localhost:8000/generate?prompt=a+cat+on+a+rooftop" --output test.png
```

### 3. Run inference

Edit `infer.py`:

```python
gen_url        = "http://localhost:8000/generate"
max_iterations = 5
```

Then:

```bash
python infer.py
# → infer_results/test_output.png
```

GEMS decomposes the prompt into verification questions, generates an image, checks
each requirement, and iteratively refines the prompt based on failures — repeating
up to `max_iterations` rounds.

---

## Line 2 — ComfyUI Generation (`infer_comfy.py`)

Produces **ComfyUI API-format workflows** and submits them to a running ComfyUI
server. The decompose / verify / refine loop is inherited verbatim from `GEMS`;
only `generate()` is replaced — it builds a workflow dict, submits to
`/prompt`, polls `/history`, and fetches the output via `/view`.

### Supported models

| Model | `model=` value | Template highlights | Default size |
|---|---|---|---|
| Qwen-Image-2512 | `qwen-image-2512` | 20B MMDiT FP8 · `ModelSamplingAuraFlow` → `KSampler` (steps=50, cfg=4.0, euler/simple) | 1328×1328 |
| Z-Image-Turbo | `z-image-turbo` | 6B S3-DiT BF16 · `ConditioningZeroOut` on negatives · `KSampler` (steps=8, cfg=1, res_multistep) | 1024×1024 |
| FLUX.2 [klein] 9B | `flux-klein-9b` | `SamplerCustomAdvanced` + `CFGGuider` + `Flux2Scheduler` + `RandomNoise` + `KSamplerSelect` (4 steps) | 1024×1024 |
| LongCat-Image | `longcat-image` | 6B DiT BF16 · `FluxGuidance` on both conds + `CFGNorm` · `KSampler` (steps=20, cfg=4) | 1024×1024 |

Each model has a corresponding `SKILL.md` under `agent/skills/<model>/`
(imported as-is from `comfyclaw`, YAML-frontmatter format) so the planner
can route the user's prompt through the model's own recipe.

### 1. Install and start ComfyUI

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI && pip install -r requirements.txt
python main.py --listen 127.0.0.1 --port 8188
```

ComfyUI must be recent enough to provide `QwenImage*`, `FluxGuidance`, `CFGNorm`,
`Flux2Scheduler`, and `ModelSamplingAuraFlow` as built-in nodes (ComfyUI ≥ 0.19).

### 2. Place the model weights

Drop each file under the matching folder in your ComfyUI installation. Filenames
are **hard-coded in `agent/comfy_workflow.py`** and must match exactly:

| Model | `ComfyUI/models/unet/` | `ComfyUI/models/text_encoders/` (or `clip/`) | `ComfyUI/models/vae/` |
|---|---|---|---|
| `qwen-image-2512` | `qwen_image_2512_fp8_e4m3fn.safetensors` | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `qwen_image_vae.safetensors` |
| `z-image-turbo` | `z_image_turbo_bf16.safetensors` | `qwen_3_4b.safetensors` | `ae.safetensors` |
| `longcat-image` | `longcat_image_bf16.safetensors` | `qwen_2.5_vl_7b.safetensors` | `ae.safetensors` |
| `flux-klein-9b` | `flux-2-klein-9b.safetensors` | `qwen_3_8b_fp8mixed.safetensors` | `flux2-vae.safetensors` |

If you need different filenames, either symlink them to the names above, or edit
the corresponding template in `agent/comfy_workflow.py`.

> The `SKILL.md` for each model (`agent/skills/<model>/SKILL.md`) lists the
> download URLs and sampler-specific tips — worth a read before you first run
> that model.

### 3. Run inference

**Via the example script** (env-var driven, no code edits needed):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export COMFYUI_SERVER=127.0.0.1:8188      # host:port of your ComfyUI
export GEMS_COMFY_MODEL=z-image-turbo     # qwen-image-2512 | z-image-turbo | flux-klein-9b | longcat-image
export GEMS_MAX_ITERATIONS=5
python infer_comfy.py
```

Output:

- `infer_results/test_output_comfy.png` — final (best) image
- `infer_results/workflows/workflow_NNN.json` — every workflow submitted this
  run, pretty-printed for inspection / replay in the ComfyUI web UI

**Programmatic use:**

```python
from agent.comfy_gems import ComfyGEMS

agent = ComfyGEMS(
    model="qwen-image-2512",           # or: "z-image-turbo" / "flux-klein-9b" / "longcat-image"
    comfyui_server="127.0.0.1:8188",
    max_iterations=5,
    workflow_log_dir="run_workflows",  # optional: dump every submitted workflow
    seed=42,                           # optional: pin KSampler / RandomNoise seed
    default_negative=None,             # optional: override per-model negative prompt
    workflow_timeout=600,              # optional: seconds to wait for one ComfyUI job
)

# Full pipeline (decompose → generate → verify → refine):
image_bytes = agent.run({"prompt": "a cozy cabin in a snowy pine forest at dusk"})
with open("out.png", "wb") as f:
    f.write(image_bytes)

# Or just build the workflow (don't submit) for offline inspection:
wf_dict = agent.build_workflow("a cozy cabin in a snowy pine forest at dusk")
```

### 4. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `litellm.BadRequestError: LLM Provider NOT provided` | Missing `ANTHROPIC_API_KEY` (or whichever provider `LITELLM_MODEL` points at). |
| `ConnectionRefusedError` / `HTTP 0` | ComfyUI server not running at `COMFYUI_SERVER`. |
| `Prompt has no outputs` | ComfyUI version too old (missing one of the custom nodes above). |
| `Value not in list: unet_name` | Weight file not placed under `ComfyUI/models/unet/` with the exact filename shown in the table above. |
| Generation hangs | Raise `workflow_timeout`, or check ComfyUI logs for OOM / VAE decode errors. |

### 4b. Run a batch of prompts in parallel

`infer_comfy.py` is a single-prompt demo.  For real experiments use
`run_comfy_batch.py`, which shards prompts across one or more ComfyUI
servers and/or multiple client workers per server (each worker owns its
own `ComfyGEMS` instance, so the full decompose / verify / refine loop
runs independently per prompt).

```bash
# 1 ComfyUI server, 2 client workers overlapping MLLM + ComfyUI:
python run_comfy_batch.py \
    --prompts prompts.jsonl \
    --output-dir results/my_run \
    --model z-image-turbo \
    --comfyui-servers 127.0.0.1:8188 \
    --workers-per-server 2 \
    --max-iterations 5

# 4 ComfyUI servers (one worker each, auto round-robin):
python run_comfy_batch.py \
    --prompts prompts.jsonl \
    --output-dir results/my_run \
    --model qwen-image-2512 \
    --comfyui-servers host1:8188,host2:8188,host3:8188,host4:8188
```

**Input** (`--prompts`): a `.jsonl` file with one `{"prompt": "..."}` per
line (extra fields are forwarded to the agent), or a plain `.txt` file
with one prompt per non-empty line.

**Output layout:**

```text
results/my_run/
├── index.json                  # {prompt: rel_path} — enables --resume
├── images/
│   ├── prompt_00000.png        # best image per prompt
│   └── ...
├── traces/
│   └── prompt_00000/
│       ├── trace.json
│       ├── best.png
│       ├── round_1.png ...     # if --save-all-rounds
│       └── workflows/          # one submitted workflow per iteration
└── logs/
    ├── worker_0.log            # stdout/stderr per worker (isolated)
    └── ...
```

**Resume:** re-running with the same `--output-dir` skips any prompt
already listed in `index.json`.  `index.json` is flushed to disk after
every successful prompt, so a crash/OOM loses at most one in-flight
item per worker.

**Notes on parallelism:**

- A single ComfyUI server serialises its job queue, so raising
  `--workers-per-server` above 1 *won't* make image generation faster,
  but it **does** overlap MLLM HTTP latency (decompose / verify / refine)
  with ComfyUI work — usually a 1.3–1.8× speed-up on short prompts.
- For true parallelism, launch N ComfyUI servers (different GPUs /
  hosts) and pass them all to `--comfyui-servers`.

### Scope (by design)

Only `generate()` is new. The refine loop still edits only the **positive prompt**,
verification is still the stock MLLM yes/no decomposition, and the workflow itself
is **not** topologically evolved (no LoRA / ControlNet auto-insertion, no sampler
LLM-tuning, no repair loop). Use [`comfyclaw`](https://github.com/...) if you want
topology evolution on top of ComfyUI.

---

## Evaluation

Images are first generated with GEMS, then scored with task-specific methods.

### GenEval2

```bash
# Option A — from Hugging Face
hf download Jialuo21/GenEval2 --repo-type dataset --local-dir /path/to/GenEval2

# Option B — from GitHub
git clone https://github.com/facebookresearch/GenEval2.git /path/to/GenEval2
```

Set `DATA_PATH`, `OUTPUT_DIR`, `gen_url`, `mllm_url` at the top of
`eval/GenEval2.py`, then pick a backend via `--agent`:

```bash
# HTTP line (unchanged; NUM_WORKERS=2, reads gen_url at top of file)
python eval/GenEval2.py --name my_run --agent gems --max_iterations 5

# ComfyUI line, single ComfyUI server × 2 client workers
python eval/GenEval2.py \
    --name my_run_comfy --agent comfygems \
    --model z-image-turbo \
    --comfyui_servers 127.0.0.1:8188 \
    --workers_per_server 2 \
    --max_iterations 5

# ComfyUI line, fan out across 4 ComfyUI servers (one worker each)
python eval/GenEval2.py \
    --name my_run_multi --agent comfygems \
    --model qwen-image-2512 \
    --comfyui_servers host1:8188,host2:8188,host3:8188,host4:8188
```

### CREA

```bash
python eval/CREA/CREA.py --name my_run --agent gems --max_iterations 5 --n_samples 25
```

### ArtiMuse

```bash
python eval/ArtiMuse/gen_artimuse.py \
    --gen_url http://localhost:8000/generate \
    --mllm_url http://localhost:30000/v1 \
    --max_iterations 5
```

Occasional server errors (e.g. timeouts) may leave a few tasks empty — just re-run,
the scripts skip already-completed items. Full evaluation code is provided for
**CREA** and **ArtiMuse**; for other tasks, evaluations follow their official
settings.

---

## Skills

![Skill](assets/skill_demo.png)

Our Skills are summarized from previous works and tested on downstream tasks. You
can add your own by creating a folder under `agent/skills/`:

```text
agent/skills/
└── <skill_id>/             # folder name is the skill ID
    └── SKILL.md
```

`SkillManager` auto-detects and supports **both** of the formats below.

**Format A — legacy GEMS template:**

```markdown
# Skill: <Name>

## Description
Provide a concise summary of what this skill does.

## Instructions
Provide detailed domain-specific guidance, prompts, or constraints here.
The code captures all content below this header.
```

**Format B — Agent-Skills YAML frontmatter** (used by the ComfyUI-line model skills
imported from `comfyclaw`):

```markdown
---
name: <skill-id>
description: >-
  One- or multi-line description of the skill.
---

The entire body below the closing `---` becomes the skill's instructions.
```

The folder name is always surfaced as the `SKILL_ID` in the planner manifest.

---

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{he2026gems,
  title={GEMS: Agent-Native Multimodal Generation with Memory and Skills},
  author={He, Zefeng and Huang, Siyuan and Qu, Xiaoye and Li, Yafu and Zhu, Tong and Cheng, Yu and Yang, Yang},
  journal={arXiv preprint arXiv:2603.28088},
  year={2026}
}
```
