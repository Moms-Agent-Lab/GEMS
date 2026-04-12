# ComfyClaw

[![CI](https://github.com/davidliuk/comfyclaw/actions/workflows/ci.yml/badge.svg)](https://github.com/davidliuk/comfyclaw/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Agentic harness for self-evolving ComfyUI image-generation workflows.**

ComfyClaw wraps ComfyUI in an LLM agent loop that _builds and grows_ workflow
topologies in response to image quality feedback — constructing pipelines from
scratch or adding LoRA loaders, ControlNet branches, regional conditioning, and
hires-fix passes until a configurable quality threshold is reached.

### Key features

- **Generate from ComfyUI** — type a prompt, click Generate in the built-in
  panel, and watch the agent work — no terminal interaction needed
- **Build from scratch or evolve** — the agent can construct an entire ComfyUI
  workflow from zero or iterate on an existing one
- **Incremental visualization** — watch nodes appear one-by-one on the ComfyUI
  canvas as the agent builds
- **Human-in-the-loop** — choose VLM-only, human-only, or hybrid verification;
  give subjective feedback directly from the ComfyUI panel
- **Any LLM, any provider** — swap agent and verifier models independently via
  [LiteLLM](https://docs.litellm.ai/docs/providers) (Anthropic, OpenAI, Gemini,
  Ollama, 100+ more)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           ClawHarness loop                              │
│                                                                         │
│  ┌──────────┐  evolve   ┌──────────────┐  submit  ┌──────────────┐     │
│  │  Agent   │ ────────► │  ComfyUI API │ ───────► │  Image out   │     │
│  │  (LLM)   │           │  (HTTP/WS)   │          └──────┬───────┘     │
│  └────┬─────┘           └──────────────┘                 │             │
│       │  feedback                                  verify│             │
│       │ ◄─────────────────────────────────────────── ▼  │             │
│       │                                       ┌──────────────┐         │
│       │                                       │  Verifier    │         │
│  ┌────┴─────┐                                 │ VLM / Human  │         │
│  │  Memory  │  broadcast   ┌────────────────┐ │  / Hybrid    │         │
│  │          │              │ ComfyClaw-Sync  │ └──────────────┘         │
│  └──────────┘              │ (ComfyUI panel) │                         │
│                            │ ● live graph    │ ◄─── trigger_generation │
│                            │ ● node-by-node  │ ◄─── human_feedback     │
│                            │ ● generate btn  │ ───► generation_status  │
│                            └────────────────┘                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Table of contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [Usage guide](#usage-guide)
  - [Serve mode — generate from ComfyUI (recommended)](#serve-mode--generate-from-comfyui-recommended)
  - [CLI run — one-shot from terminal](#cli-run--one-shot-from-terminal)
  - [Human-in-the-loop verification](#human-in-the-loop-verification)
  - [Choosing an LLM provider](#choosing-an-llm-provider)
- [CLI reference](#cli-reference)
- [Python API](#python-api)
- [Architecture](#architecture)
- [Skills](#skills)
- [Development](#development)
- [Project structure](#project-structure)

---

## Quick start

Four steps from zero to generating images inside ComfyUI:

```bash
# 1. Clone and install
git clone https://github.com/davidliuk/comfyclaw.git
cd comfyclaw
uv sync                                    # or: pip install -e ".[sync]"

# 2. Configure (set at least one LLM API key)
cp .env.example .env                       # then edit .env
# ANTHROPIC_API_KEY=sk-ant-...             # ← required
# COMFYUI_ADDR=127.0.0.1:8188             # ← your ComfyUI address

# 3. Install the ComfyUI plugin (one-time), then restart ComfyUI
comfyclaw install-node

# 4. Start the ComfyClaw server
comfyclaw serve
```

Now open ComfyUI in your browser. You'll see:
- Bottom-right: status badge shows **🟢 ComfyClaw: live**
- Top-right: the **🐾 ComfyClaw panel** with a prompt box and Generate button

Type a prompt, click **▶ Generate**, and watch the agent build a workflow
node-by-node on the canvas, generate the image, and score it — all without
leaving ComfyUI.

> **How it works:** `comfyclaw serve` starts a persistent background server.
> The ComfyUI plugin connects to it via WebSocket. When you click Generate,
> the plugin sends your prompt to the server; the server runs an LLM agent
> that builds/evolves a workflow, submits it to ComfyUI, verifies the output,
> and iterates. You see every step live on the canvas.
>
> **If the badge shows 🔴 disconnected:** make sure `comfyclaw serve` is
> running in a terminal. The plugin is just a frontend — it needs the Python
> server to be active.

For one-shot CLI usage (without the ComfyUI panel):

```bash
comfyclaw run --prompt "a red fox at dawn, photorealistic, DSLR"
```

---

## Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | 3.12+ recommended |
| **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** | Desktop app or server, running and accessible via HTTP |
| **LLM API key** | For your chosen provider (see [Choosing an LLM provider](#choosing-an-llm-provider)) |

### Step 1 — Install ComfyClaw

**With [uv](https://docs.astral.sh/uv/) (recommended):**

```bash
git clone https://github.com/davidliuk/comfyclaw.git
cd comfyclaw
uv sync                      # runtime dependencies
uv sync --group dev          # + dev tools (pytest, ruff, mypy, …)
```

**With pip:**

```bash
git clone https://github.com/davidliuk/comfyclaw.git
cd comfyclaw
pip install -e ".[sync]"    # editable install with WebSocket support
```

**Dependency extras:**

| Extra | Packages | When needed |
|---|---|---|
| *(none)* | `litellm`, `python-dotenv` | Always |
| `sync` | `websockets>=12` | Live graph updates in ComfyUI canvas |
| `providers` | `anthropic>=0.25` | Direct Anthropic SDK (optional; litellm handles it) |
| `dev` (group) | `pytest`, `ruff`, `mypy`, … | Development & CI |

### Step 2 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```ini
# Required: at least one LLM provider key
ANTHROPIC_API_KEY=sk-ant-...        # for Anthropic (default provider)
# OPENAI_API_KEY=sk-...             # for OpenAI
# GEMINI_API_KEY=...                # for Google Gemini
# (no key needed for local Ollama)

# Required: ComfyUI server address
COMFYUI_ADDR=127.0.0.1:8188        # adjust to your ComfyUI port

# Optional: ComfyUI install path (for plugin installation)
COMFYUI_DIR=~/Documents/ComfyUI

# Optional: model and behavior overrides
# COMFYCLAW_MODEL=anthropic/claude-sonnet-4-5
# COMFYCLAW_VERIFIER_MODEL=openai/gpt-4o
# COMFYCLAW_VERIFIER_MODE=vlm
# COMFYCLAW_MAX_ITERATIONS=3
# COMFYCLAW_THRESHOLD=0.85
# COMFYCLAW_SYNC_PORT=8765
```

All CLI flags can also be set as environment variables. `.env` is auto-loaded
at startup.

### Step 3 — Install the ComfyUI plugin

The plugin is bundled inside the package. Install it once, then **restart
ComfyUI** so it loads the new extension.

```bash
# Automatic (recommended)
comfyclaw install-node

# With an explicit ComfyUI path
comfyclaw install-node --comfyui-dir ~/Documents/ComfyUI
```

<details>
<summary>Manual alternatives</summary>

```bash
# Symlink (edits take effect immediately — best for development)
ln -s "$(comfyclaw node-path)" ~/Documents/ComfyUI/custom_nodes/ComfyClaw-Sync

# Or copy
cp -r "$(comfyclaw node-path)" ~/Documents/ComfyUI/custom_nodes/ComfyClaw-Sync
```

</details>

### Step 4 — Verify installation

1. Start the server: `comfyclaw serve`
2. Open ComfyUI in your browser. You should see:

| UI element | Location | What to check |
|---|---|---|
| **Status badge** | Bottom-right corner | Shows **🟢 ComfyClaw: live** (not 🔴 or 🔄) |
| **🐾 ComfyClaw panel** | Top-right corner | Prompt box, mode toggle, Generate button visible |

3. Type a test prompt (e.g. "a cute cat") in the panel and click **▶ Generate**.

If the badge stays at **🔴 disconnected**, verify that `comfyclaw serve` is running
and the port matches (default 8765). See [Troubleshooting connection](#serve-mode--generate-from-comfyui-recommended) for details.

---

## Usage guide

### Serve mode — generate from ComfyUI (recommended)

This is the primary way to use ComfyClaw. You start the server once, then do
everything from within ComfyUI's browser interface.

**Step 1: Start the server** (leave it running in a terminal):

```bash
comfyclaw serve
```

You can configure the server with the same flags as `run`:

```bash
comfyclaw serve \
  --model openai/gpt-4o \
  --iterations 5 \
  --threshold 0.9
```

**Step 2: Open ComfyUI** in your browser. The status badge (bottom-right) should
show **🟢 live**. If it shows 🔴, the server isn't running or the port doesn't
match (default: `ws://localhost:8765`).

**Step 3: Use the 🐾 ComfyClaw panel** (top-right corner, click header to
expand/collapse):

```
┌─────────────────────────────────┐
│ 🐾 ComfyClaw                  ▼│
├─────────────────────────────────┤
│                                 │
│ Prompt                          │
│ ┌─────────────────────────────┐ │
│ │ a cute cat sitting on a     │ │
│ │ windowsill at sunset...     │ │
│ └─────────────────────────────┘ │
│                                 │
│ Mode                            │
│ [✨ From Scratch] [🔧 Improve]  │
│                                 │
│ ▸ Settings                      │
│   Iterations: [3]               │
│   Verifier:   [VLM ▾]          │
│                                 │
│ [        ▶ Generate          ]  │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ ✅ Done! Score: 0.89 (1 it)│ │
│ └─────────────────────────────┘ │
└─────────────────────────────────┘
```

| Element | What it does |
|---|---|
| **Prompt** | Multi-line text — describe what you want to generate |
| **✨ From Scratch** | Agent builds the entire workflow from zero |
| **🔧 Improve Current** | Agent evolves whatever is currently on the canvas |
| **Settings** | Override iterations count and verifier mode (VLM / Human / Hybrid) per-run |
| **▶ Generate** | Send prompt to agent — workflow builds live on canvas |
| **Status area** | Real-time progress: idle → running → verifying → complete |
| **■ Stop** | Cancel the current run (appears while running) |

**What happens when you click Generate:**

1. Panel sends your prompt + mode + settings to the server via WebSocket
2. Server creates a fresh LLM agent
3. Agent queries ComfyUI for available models, reads skill recipes
4. Nodes appear one-by-one on the canvas (highlighted in blue)
5. Workflow is submitted to ComfyUI → image generated
6. Vision LLM (or you, in human mode) scores the image
7. If below threshold, agent iterates with feedback
8. Status area shows final score; server waits for next trigger

**Troubleshooting connection:**

| Symptom | Cause | Fix |
|---|---|---|
| 🔴 disconnected | Server not running | Run `comfyclaw serve` in a terminal |
| 🔄 connecting (stuck) | Port mismatch | Check `--sync-port` matches (default 8765) |
| 🔴 after server crash | Port still held | Wait a few seconds or `lsof -ti :8765 \| xargs kill` |
| Panel not visible | Plugin not installed | Run `comfyclaw install-node` and restart ComfyUI |

### CLI run — one-shot from terminal

For scripting or batch jobs, you can run a single generation from the CLI
without using the ComfyUI panel:

```bash
# Build from scratch (no workflow file needed)
comfyclaw run \
  --prompt "a red fox at dawn, photorealistic, DSLR" \
  --iterations 3

# Or evolve an existing workflow
comfyclaw run \
  --workflow my_workflow_api.json \
  --prompt "a red fox at dawn, photorealistic, DSLR" \
  --iterations 3

# Dry-run (agent builds workflow, no ComfyUI execution — good for testing)
comfyclaw dry-run --prompt "a cute cat"
```

The agent loop is identical to serve mode. The only difference is that the
prompt and settings come from CLI flags instead of the ComfyUI panel, and
the process exits after one run.

### Human-in-the-loop verification

By default, a vision LLM scores each generated image. You can add human
judgement — either replacing the LLM entirely or reviewing its assessment.

| Mode | Flag | Behavior |
|---|---|---|
| **VLM** (default) | `--verifier-mode vlm` | Vision LLM scores automatically |
| **Human** | `--verifier-mode human` | You score via ComfyUI panel (terminal fallback if no panel) |
| **Hybrid** | `--verifier-mode hybrid` | VLM scores first → you review and accept or override |

```bash
# Human-only verification
comfyclaw run \
  --prompt "portrait of a girl in golden hour light" \
  --verifier-mode human \
  --iterations 3

# Hybrid: VLM proposes, you approve or correct
comfyclaw run \
  --prompt "portrait of a girl in golden hour light" \
  --verifier-mode hybrid

# In serve mode: selectable per-run from the panel's Settings dropdown
comfyclaw serve --iterations 3
```

When feedback is requested, a **floating panel** appears in ComfyUI:

- Prompt and iteration number displayed at top
- VLM assessment summary (hybrid mode only)
- **Score buttons**: 👍 Good (0.9) · 👌 OK (0.6) · 👎 Needs Work (0.3)
- **Text area** for specific feedback ("make the lighting warmer", "fix the hands")
- **Submit** sends feedback → agent adapts next iteration
- **Accept as-is** approves the current result

The agent treats human feedback as high-priority subjective input and focuses
its next iteration on the specific issues you raised.

### Choosing an LLM provider

ComfyClaw uses [LiteLLM](https://docs.litellm.ai/docs/providers) to route to
any provider. Set the matching environment variable and use the model string
with provider prefix:

| Provider | Model string | Required env var |
|---|---|---|
| **Anthropic** (default) | `anthropic/claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| **OpenAI** | `openai/gpt-4o` | `OPENAI_API_KEY` |
| **Google Gemini** | `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |
| **Groq** | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| **Azure OpenAI** | `azure/<deployment>` | `AZURE_API_KEY` + `AZURE_API_BASE` |
| **Local Ollama** | `ollama/llama3.1` | *(none)* |

You can **mix providers** — use a cheap/fast model for the agent and a strong
vision model for the verifier:

```bash
# Cloud agent + cloud verifier (highest quality)
comfyclaw run --model anthropic/claude-sonnet-4-5 --prompt "..."

# Local agent + cloud verifier (saves agent API costs)
comfyclaw run \
  --model ollama/llama3.1 \
  --verifier-model anthropic/claude-sonnet-4-5 \
  --prompt "..."

# Fully local (no API keys needed, requires capable local models)
comfyclaw run \
  --model ollama/llama3.1 \
  --verifier-model ollama/llava \
  --prompt "..."
```

> **Vision requirement**: the `--verifier-model` must support image inputs.
> Good choices: `anthropic/claude-*`, `openai/gpt-4o`, `gemini/gemini-*`,
> `ollama/llava`.

---

## CLI reference

```
comfyclaw serve         Persistent server — trigger from ComfyUI panel (recommended)
comfyclaw run           One-shot agent → generate → verify loop from terminal
comfyclaw dry-run       Agent-only (no ComfyUI execution — useful for testing)
comfyclaw install-node  Symlink the ComfyClaw-Sync plugin into ComfyUI
comfyclaw node-path     Print path to the bundled plugin directory
```

### Options for `run` / `dry-run` / `serve`

| Flag | Default | Description |
|---|---|---|
| `--workflow PATH` | *(optional)* | API-format workflow JSON; omit to build from scratch |
| `--prompt TEXT` | *(required for run/dry-run; ignored by serve)* | Image generation prompt; in serve mode the prompt comes from the ComfyUI panel |
| `--model MODEL` | `anthropic/claude-sonnet-4-5` | LiteLLM model for the agent |
| `--verifier-model MODEL` | *(same as --model)* | LiteLLM model for the vision verifier |
| `--verifier-mode MODE` | `vlm` | `vlm`, `human`, or `hybrid` |
| `--image-model NAME` | *(from workflow)* | Pin ComfyUI checkpoint filename |
| `--iterations N` | `3` | Max agent–generate–verify cycles |
| `--threshold SCORE` | `0.85` | Stop early when score ≥ threshold |
| `--max-repair-attempts N` | `2` | Auto-repair attempts per iteration |
| `--sync-port PORT` | `8765` | WebSocket port for live sync |
| `--no-sync` | off | Disable live sync |
| `--skills-dir DIR` | *(built-in)* | Custom skill directory |
| `--reset-each-iter` | off | Reset to base workflow each iteration |
| `--output-dir DIR` | `./comfyclaw_output/` | Where to save the best image |

### Environment variables

All flags have environment variable equivalents:

| Variable | Default | Maps to |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic provider auth |
| `OPENAI_API_KEY` | — | OpenAI provider auth |
| `GEMINI_API_KEY` | — | Google Gemini provider auth |
| `COMFYUI_DIR` | `~/Documents/ComfyUI` | `install-node` target |
| `COMFYUI_ADDR` | `127.0.0.1:8188` | `--` (server address) |
| `COMFYCLAW_MODEL` | `anthropic/claude-sonnet-4-5` | `--model` |
| `COMFYCLAW_VERIFIER_MODEL` | *(same as model)* | `--verifier-model` |
| `COMFYCLAW_VERIFIER_MODE` | `vlm` | `--verifier-mode` |
| `COMFYCLAW_MAX_ITERATIONS` | `3` | `--iterations` |
| `COMFYCLAW_THRESHOLD` | `0.85` | `--threshold` |
| `COMFYCLAW_SYNC_PORT` | `8765` | `--sync-port` |

---

## Sample runs

### Claude Sonnet 4.5 — Wildlife photography (serve mode)

Start the server once and generate from ComfyUI:

```bash
comfyclaw serve --iterations 2
```

In the ComfyUI panel, enter the prompt:
> A majestic red fox sitting in a misty ancient forest at dawn, photorealistic wildlife photography

Select **✨ From Scratch** and click **▶ Generate**.

Equivalent one-shot CLI command:

```bash
comfyclaw run \
  --workflow qwen_workflow_api.json \
  --prompt "A majestic red fox sitting in a misty ancient forest at dawn, photorealistic wildlife photography" \
  --iterations 2
```

The agent read `qwen-image-2512` + `photorealistic` skills, expanded the
prompt with camera specs (300 mm f/2.8, shallow DoF, National Geographic
aesthetic), added a Chinese negative prompt, and set resolution to 1472×1104
for Qwen's optimal Lightning bucket.

| Metric | Value |
|--------|-------|
| Score | **0.89 / 1.00** |
| Outcome | ✅ Stopped early after iteration 1 |
| Passed | Fox, red colour, sitting pose, forest, mist, dawn lighting, photorealistic |

### Ollama Gemma4 — Cyberpunk city

```bash
comfyclaw run \
  --workflow qwen_workflow_api.json \
  --model ollama/gemma4:e4b \
  --verifier-model ollama/gemma4:e4b \
  --iterations 3 \
  --prompt "a futuristic cyberpunk city skyline at night, neon lights, rain, 8k"
```

| Iteration | Score | Notes |
|-----------|-------|-------|
| 1 | 0.36 | Atmosphere present; missing neon, reflections |
| 2 | 0.36 | Agent planned ControlNet but failed to execute tool calls |
| 3 | **0.49** | Rain/reflections improved; neon still weak |

**Takeaway:** Gemma4 is a capable *vision verifier* but less reliable at
complex tool-call execution. Use a stronger model (Claude, GPT-4o) for the
agent, and reserve local models for `--verifier-model`.

---

## Python API

### Minimal usage

```python
from comfyclaw import ClawHarness, HarnessConfig

cfg = HarnessConfig(
    server_address="127.0.0.1:8188",
    model="anthropic/claude-sonnet-4-5",
    max_iterations=3,
    success_threshold=0.85,
)

# From a workflow file
with ClawHarness.from_workflow_file("workflow_api.json", cfg) as h:
    image_bytes = h.run("a red fox at dawn, photorealistic")

# Or build from scratch (empty dict)
with ClawHarness.from_workflow_dict({}, cfg) as h:
    image_bytes = h.run("a red fox at dawn, photorealistic")

if image_bytes:
    open("output.png", "wb").write(image_bytes)
```

### HarnessConfig

```python
@dataclass
class HarnessConfig:
    api_key: str = ""                   # or set provider env var
    server_address: str = "127.0.0.1:8188"
    model: str = "anthropic/claude-sonnet-4-5"
    verifier_model: str | None = None   # None = same as model
    max_iterations: int = 3
    success_threshold: float = 0.85
    sync_port: int = 8765               # 0 = disable live sync
    skills_dir: str | None = None       # None = built-in skills
    evolve_from_best: bool = True       # accumulate topology across iters
    max_images: int = 5
    score_weights: tuple = (0.6, 0.4)   # (requirement, detail) blend
    image_model: str | None = None      # pin checkpoint/UNET filename
    max_repair_attempts: int = 2
    verifier_mode: str = "vlm"          # "vlm", "human", or "hybrid"
```

### Topology accumulation

When `evolve_from_best=True` (default), each iteration starts from the **best
workflow snapshot** so far:

```
Iter 1:  base(3 nodes) → +LoRA         → 4 nodes   score=0.62
Iter 2:  4-node snapshot → +ControlNet → 6 nodes   score=0.81
Iter 3:  6-node snapshot → +hires-fix  → 8 nodes   score=0.91 ✅
```

### WorkflowManager

```python
from comfyclaw.workflow import WorkflowManager

wm = WorkflowManager.from_file("workflow_api.json")

print(wm)                                      # repr with node count
print(WorkflowManager.summarize(wm.workflow))   # human-readable table
errors = wm.validate()                          # check graph integrity

nid = wm.add_node("LoraLoader", nickname="My LoRA",
                   lora_name="detail.safetensors",
                   strength_model=0.8, strength_clip=0.8)
wm.connect("1", 0, nid, "model")
wm.set_param("3", "steps", 30)
wm.delete_node("2")
```

### ComfyClient

```python
from comfyclaw.client import ComfyClient

client = ComfyClient("127.0.0.1:8188")
resp   = client.queue_prompt(wm.workflow)
entry  = client.wait_for_completion(resp["prompt_id"], timeout=300)
images = client.collect_images(entry)    # list[bytes]
```

---

## Architecture

### The agent loop

Each iteration of the harness follows this cycle:

1. **Agent evolves** — the LLM reads skills, inspects the workflow, calls tools
   (`add_node`, `connect_nodes`, `set_param`, etc.) to modify the graph
2. **Validate** — `finalize_workflow` auto-validates; blocks if errors found
3. **Submit** — workflow is sent to ComfyUI's `/prompt` endpoint
4. **Repair** (if needed) — ComfyUI errors trigger up to N repair attempts
   where the agent sees the error and fixes the topology
5. **Generate** — ComfyUI runs the workflow and produces an image
6. **Verify** — VLM / human / hybrid verifier scores the image
7. **Feedback** — score and suggestions fed back to agent for next iteration

### Incremental visualization

Changes appear **node by node** on the ComfyUI canvas. Each new node is briefly
highlighted in blue. The sync protocol uses an efficient diff algorithm:

| Message | When sent | Content |
|---|---|---|
| `workflow_update` | First load / reconnect | Full workflow snapshot |
| `workflow_diff` | Subsequent mutations | Granular ops: `add_node`, `remove_node`, `update_node` |

Adjust animation speed: `localStorage.setItem('comfyclaw_op_delay', '200')` (default 400 ms).

### WebSocket protocol

Bidirectional communication between the Python process and ComfyUI extension:

**Server → Client:**

| Message | Purpose |
|---|---|
| `workflow_update` | Full workflow snapshot |
| `workflow_diff` | Incremental ops |
| `request_feedback` | Ask human for feedback |
| `generation_status` | Progress: `running` / `verifying` / `repairing` |
| `generation_complete` | Final score and image path |
| `generation_error` | Error details |

**Client → Server:**

| Message | Purpose |
|---|---|
| `human_feedback` | Score, text, action (accept/override) |
| `trigger_generation` | Start a run from the ComfyUI panel |

### Agent tools (15)

| Category | Tool | Purpose |
|---|---|---|
| Inspect | `inspect_workflow` | Show all nodes and connections |
| Inspect | `query_available_models` | List installed models/LoRAs/ControlNets |
| Validate | `validate_workflow` | Check dangling refs, wrong slots, missing outputs |
| Basic | `set_param` | Set a scalar input |
| Basic | `add_node` | Append a new node |
| Basic | `connect_nodes` | Wire output → input |
| Basic | `delete_node` | Remove node + clean up links |
| LoRA | `add_lora_loader` | Insert LoRA with auto re-wiring |
| ControlNet | `add_controlnet` | Add ControlNet pipeline |
| Regional | `add_regional_attention` | Foreground/background conditioning split |
| Refinement | `add_hires_fix` | Upscale + second KSampler |
| Refinement | `add_inpaint_pass` | Targeted region inpainting |
| Skills | `read_skill` | Load skill instructions on demand |
| Control | `report_evolution_strategy` | Declare plan before changes |
| Control | `finalize_workflow` | Complete iteration (auto-validates) |

---

## Skills

ComfyClaw's skills follow the [Agent Skills spec](https://agentskills.dev/specification).
Each skill is a directory with a `SKILL.md` file containing YAML frontmatter.

**Progressive disclosure** keeps context lean:
1. **Startup** — only `name` + `description` from frontmatter appear in the system prompt
2. **On demand** — agent calls `read_skill("name")` to load full instructions

### Built-in skills

| Skill | When activated |
|---|---|
| `workflow-builder` | Building from scratch (architecture recipes + slot reference) |
| `qwen-image-2512` | Qwen-Image-2512 model (Lightning 4-step pipeline) |
| `high-quality` | "high quality", "sharp", "detailed", "8K" |
| `photorealistic` | "photo", "DSLR", "realistic", "cinematic" |
| `creative` | "creative", "artistic", "fantasy", "concept art" |
| `aesthetic-drawing` | "masterpiece", "award-winning", "professional art" |
| `creative-drawing` | "cool", "dreamy", "futuristic" |
| `lora-enhancement` | Texture / lighting / anatomy defects |
| `controlnet-control` | Flat background, blurry edges, wrong pose |
| `regional-control` | Subject–background style bleed |
| `hires-fix` | Low resolution, soft detail |
| `spatial` | Multiple objects with spatial relationships |
| `text-rendering` | Quoted text, signs, labels |

### Adding custom skills

```
my_skills/
└── portrait-lighting/
    └── SKILL.md
```

```markdown
---
name: portrait-lighting
description: >-
  Optimise lighting for portrait photography. Activate when the user mentions
  "portrait", "face", "studio lighting".
---

1. Append `, dramatic studio lighting, rim light, catchlights` to the positive prompt.
2. Set KSampler CFG to 8.0–9.0.
3. Consider adding a normal-map ControlNet for skin texture depth.
```

```bash
comfyclaw run --prompt "..." --skills-dir ./my_skills/
```

---

## Workflow format

ComfyClaw uses the **API format** (not the UI format):

```json
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    "_meta": { "title": "Load Checkpoint" },
    "inputs": { "ckpt_name": "v1-5-pruned.ckpt" }
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": { "clip": ["1", 1], "text": "a red fox" }
  }
}
```

Export from ComfyUI: **Workflow → Export (API)** in the menu.

`from_workflow_file()` also handles prompt-keyed saves (`{"prompt": {...}}`)
and UI format (looks for sibling `*_api.json`; falls back to conversion).

---

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

```bash
uv sync --group dev                                      # bootstrap
uv run pre-commit install                                # commit hooks
uv run pre-commit install --hook-type pre-push           # push hooks
uv run pytest -ra -q                                     # 192 tests, < 1 s
uv run ruff check --fix . && uv run ruff format .        # lint + format
uv build                                                 # build wheel
```

---

## Project structure

```
comfyclaw/
├── pyproject.toml
├── README.md
├── comfyclaw/
│   ├── __init__.py           Public re-exports
│   ├── cli.py                CLI entry point (run / serve / dry-run / install-node)
│   ├── harness.py            ClawHarness — orchestrates the agent loop
│   ├── agent.py              ClawAgent — LLM tool-use loop (15 tools)
│   ├── verifier.py           ClawVerifier — vision LLM scoring
│   ├── human_verifier.py     HumanVerifier + HybridVerifier
│   ├── workflow.py           WorkflowManager — graph mutations + validation
│   ├── client.py             ComfyClient — HTTP + polling
│   ├── memory.py             ClawMemory — per-run attempt history
│   ├── sync_server.py        SyncServer — bidirectional WebSocket
│   ├── skill_manager.py      SkillManager — Agent Skills spec loader
│   ├── custom_node/          Bundled ComfyUI plugin (v4.0)
│   │   ├── __init__.py
│   │   └── js/
│   │       └── comfy_claw_sync.js
│   └── skills/               Built-in skills
│       ├── workflow-builder/  Architecture recipes
│       ├── qwen-image-2512/  Qwen model config
│       ├── photorealistic/   … and 10 more
│       └── ...
└── tests/                    192 tests (all offline, < 1 s)
    ├── test_agent.py
    ├── test_harness.py
    ├── test_workflow.py
    ├── test_sync_server.py
    ├── test_human_verifier.py
    └── ...
```

---

## Known constraints

- **Apple MPS + FP8 models**: `Float8_e4m3fn` is not supported on Apple Silicon.
  The agent auto-detects and repairs by setting `weight_dtype: "default"`. If the
  model file is natively fp8, the hardware incompatibility persists.
- **Serve mode requires WebSocket**: Do not use `--no-sync` with `comfyclaw serve`.
- **Auto-repair**: Up to `--max-repair-attempts` (default 2) per iteration.
  Transient infrastructure faults (e.g. BrokenPipe) are retried once
  automatically.
- **ComfyUI versions**: The JS extension tries `app.loadApiJson` (≥ 0.2),
  `app.loadGraphData`, and `app.graph.configure` in order.
- **Workflow format**: Only API format is sent to `/prompt`. UI format is
  converted on load.
