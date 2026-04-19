# LoRA — usage guide

ComfyClaw's agent can insert LoRA adapters into any workflow it builds or
evolves, across four architecture families:

| Family | LoRA tool | LoRA node |
|---|---|---|
| **SD 1.5 / SDXL / Flux** (standard) | `add_lora_loader` | `LoraLoader` (MODEL + CLIP) |
| **Qwen-Image-2512** (20B MMDiT) | `add_lora_loader` | `LoraLoaderModelOnly` |
| **Z-Image-Turbo** (6B S3-DiT) | `add_lora_loader` | `LoraLoaderModelOnly` |
| **LongCat-Image** (6B pipeline) | — | *(blocked — no MODEL tensor)* |

You always call the **same agent tool** (`add_lora_loader`). The harness
detects the active architecture from the workflow graph (checkpoint filename,
`CLIPLoader.type`, custom loader `class_type`) and dispatches to the correct
node automatically. See `comfyclaw/skills/<arch>/arch.yaml` (loaded into
`ARCH_REGISTRY` in `comfyclaw/agent.py`) for the
detection rules.

---

## 1. When to reach for LoRA

LoRA is most useful when the VLM verifier reports a structural defect that
prompt-tuning alone can't fix.

| Verifier complaint | Best fix | Tool |
|---|---|---|
| Plasticky skin, flat lighting, soft overall quality | Add a detail / realism / lighting LoRA | `add_lora_loader` |
| Wrong style, anime / oil paint intent | Style LoRA | `add_lora_loader` |
| Model too slow (30+ steps) | Speed / acceleration LoRA (Lightning, Turbo) | `add_lora_loader` |

The agent's `lora-enhancement` skill also suggests these fixes based on
verifier `fix_strategy` tokens.

---

## 2. Where to put the weights

The agent discovers files via `query_available_models("loras")`. This surfaces
the standard ComfyUI folder:

```
ComfyUI/models/
└── loras/                   LoRA safetensors for all archs
```

Recommended starter set (all verified to run end-to-end against ComfyUI):

| Arch | Role | Filename | Size | Source |
|---|---|---|---|---|
| Qwen-Image-2512 | LoRA (photorealism) | `real_life_qwen.safetensors` | 203 MB | Civitai #2056953 |
| Qwen-Image-2512 | LoRA (aesthetic) | `Qwen_art_lora.safetensors` | 563 MB | Civitai #2010520 |
| Qwen-Image-2512 | LoRA (speed, 4-step) | `Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors` | 811 MB | `lightx2v/Qwen-Image-2512-Lightning` |
| Z-Image-Turbo   | LoRA (realism) | `Z-Image-Turbo-Radiant-Realism-Pro.safetensors` | 163 MB | community |
| Z-Image-Turbo   | LoRA (realism, lighter) | `Z-Image-Turbo-Realism-LoRA.safetensors` | 82 MB | community |

Download HuggingFace LoRAs with `huggingface-cli download <repo> <file> --local-dir <target>`
or `wget` on the `resolve/main/…` URL. For Civitai LoRAs, use
`python experiments/setup.py --model qwen` which handles both sources automatically.

---

## 3. Letting the agent do it (recommended)

Just tell the agent what you want in the ComfyClaw panel or on the CLI. The
agent reads the relevant skill (`lora-enhancement`, `qwen-image-2512`,
`z-image-turbo`), queries installed weights, and calls `add_lora_loader`
with the right arguments.

Serve mode:
```bash
comfyclaw serve
```

Then in the ComfyUI panel, try prompts like:

- _"photoreal portrait at golden hour — use a realism LoRA if one is
  available"_
- _"fast generation, Lightning 4-step"_

CLI one-shot:
```bash
comfyclaw run \
  --workflow qwen_workflow_api.json \
  --prompt "a red fox at dawn, photorealistic, DSLR" \
  --iterations 2
```

The agent also repairs automatically: if the LoRA file is missing, the
CLIPLoader `type` is wrong, or a required node is absent, it'll try
to fix the graph up to `--max-repair-attempts` times.

---

## 4. Calling the tools directly (Python API)

If you want deterministic control, drive the agent's dispatcher yourself.

```python
from comfyclaw.agent import ClawAgent
from comfyclaw.workflow import WorkflowManager
from comfyclaw.memory import ClawMemory

wm    = WorkflowManager.from_file("qwen_workflow_api.json")
agent = ClawAgent(api_key="", model="anthropic/claude-sonnet-4-5")
mem   = ClawMemory()

agent._dispatch_tool(
    "add_lora_loader",
    {
        "lora_name":      "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
        "model_node_id":  "37",       # UNETLoader node ID
        "strength_model": 1.0,
    },
    wm, mem,
)

wm.to_file("qwen_with_lora.json")
```

### Tool arguments

`add_lora_loader`:

| Arg | Required | SD / SDXL / Flux | Qwen / Z-Image | LongCat |
|---|---|---|---|---|
| `lora_name` | yes | filename from `loras/` | filename from `loras/` | *(blocked)* |
| `model_node_id` | yes | `CheckpointLoaderSimple` or `UNETLoader` (or previous LoRA) | `UNETLoader` (or previous LoRA) | — |
| `clip_node_id` | yes here, ignored elsewhere | `CheckpointLoaderSimple` or `CLIPLoader` | omit / ignored | — |
| `strength_model` | optional (default 1.0) | 0.4–1.0 typical | 0.4–1.0 typical | — |
| `strength_clip` | optional | 0.4–1.0 typical | ignored | — |

For all archs, call `query_available_models("loras")` first so you use an
exact filename that is actually installed.

---

## 5. End-to-end test on a live ComfyUI

Two scripts under `tests/` validate the full stack against a running server.
They are not part of the default pytest suite (offline-only).

```bash
# 1. Graph-shape / validation E2E (doesn't require the real weights to run,
#    errors out cleanly with "value_not_in_list" if files are missing).
python tests/e2e_lora_controlnet.py

# 2. Real-weight E2E — requires the starter weight set from § 2 to be
#    installed. Generates actual PNGs in ComfyUI/output/.
python tests/e2e_lora_controlnet_real.py
```

Both scripts honour `COMFYUI_ADDR` from `.env` (default `127.0.0.1:8188`).

---

## 6. Troubleshooting

| Symptom | Diagnosis | Fix |
|---|---|---|
| `value_not_in_list: … not in [...]` from `/prompt` | The filename isn't present on the ComfyUI server | `ls ComfyUI/models/loras/` and match exactly — names are case-sensitive |
| LoRA silently ignored | Wrong arch (e.g. using an SDXL LoRA with Qwen) | LoRAs are **not** cross-compatible across archs — match the base model |
| Qwen image comes out over-saturated at cfg=1 | Too strong a speed LoRA | Reduce `strength_model` to 0.75, or switch to 8-step Lightning |
| LongCat complains that LoRA isn't supported | Expected — LongCat pipeline nodes expose no MODEL tensor | Use prompt-level guidance or switch to Qwen / Z-Image |

---

## 7. Reference: where this is implemented

- `comfyclaw/skills/<arch>/arch.yaml` — per-model registry entries (detection + LoRA config)
- `comfyclaw/agent.py` — `ARCH_REGISTRY = load_arch_registry()`, `_detect_arch`, `_add_lora`
- `comfyclaw/skills/lora-enhancement/SKILL.md` — agent-facing playbook
- `comfyclaw/skills/qwen-image-2512/SKILL.md` — full Qwen recipe
- `comfyclaw/skills/z-image-turbo/SKILL.md` — full Z-Image recipe
- `comfyclaw/skills/longcat-image/SKILL.md` — LongCat limitations
- `tests/test_lora_controlnet_archs.py` — offline unit tests for the dispatcher
- `tests/e2e_lora_controlnet.py` — live-server graph-shape E2E
- `tests/e2e_lora_controlnet_real.py` — live-server real-weight E2E
