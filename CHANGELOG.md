# Changelog

All notable changes to ComfyClaw are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — 2026-04-09

### Added

#### Multi-provider LLM support via LiteLLM
- Replaced the hard `anthropic` SDK dependency with `litellm>=1.0` — any of the
  100+ providers supported by LiteLLM can now be used as the agent or verifier
  model (Anthropic, OpenAI, Google Gemini, Azure, Groq, local Ollama, vLLM, …).
- New `--verifier-model` CLI flag and `COMFYCLAW_VERIFIER_MODEL` env-var allow
  the vision verifier to use a different model from the agent (e.g. run the
  agent on a cheap local model, but verify with a strong vision model).
- New `ClawVerifier.complete()` method for lightweight text completions reused
  by the harness experience-summary, replacing direct SDK access.
- `HarnessConfig` gains an optional `verifier_model` field (defaults to `None`,
  meaning the verifier uses the same model as the agent).
- Default model string updated to `anthropic/claude-sonnet-4-5` (explicit
  provider prefix) for unambiguous LiteLLM routing.

#### Backward compatibility
- Existing `ANTHROPIC_API_KEY` environment variable continues to work unchanged.
- Bare Claude model names (e.g. `claude-sonnet-4-5`) are still auto-detected by
  LiteLLM when `ANTHROPIC_API_KEY` is set.
- The `api_key` field on `HarnessConfig` / `ClawAgent` / `ClawVerifier` remains
  functional (written into `ANTHROPIC_API_KEY` env-var if provided).

---


### Added

#### Qwen-Image-2512 support
- New model skill `comfyclaw/skills/qwen-image-2512/SKILL.md` — covers the
  native ComfyUI FP8 pipeline (`UNETLoader` + `CLIPLoader` + `VAELoader` +
  `KSampler` + `EmptySD3LatentImage`), Lightning LoRA 4-step mode, recommended
  aspect-ratio buckets, prompt-engineering guidance, and per-issue iteration
  strategies.
- New base workflow `qwen_workflow_api.json` — ready-to-use API-format workflow
  for Qwen-Image-2512 (FP8, Lightning LoRA enabled, 16:9 landscape default).
- Automatic Qwen model detection in `ClawAgent._build_user_message`: the agent
  inspects the workflow for `UNETLoader` nodes whose `unet_name` contains
  `"qwen_image"` (native FP8 format) or for legacy `QwenImageModelLoader`/
  `RH_QwenImageGenerator` nodes, and automatically prepends `qwen-image-2512`
  to the suggested-skills list.
- System-prompt heuristic updated: when the workflow contains Qwen nodes the
  agent is instructed to `read_skill("qwen-image-2512")` before any parameter
  tuning.

#### DreamShaper 8 LCM model skill
- New model skill `comfyclaw/skills/dreamshaper8-lcm/SKILL.md` — documents LCM
  sampler (`lcm`), scheduler (`sgm_uniform`), recommended steps (4–8), CFG
  (1.5–2.5), and LCM-compatible hires-fix configuration.
- System-prompt heuristic: active model name containing `"lcm"` triggers
  automatic skill suggestion.

#### Agentic error-repair loop (`harness.py`, `cli.py`)
- **Queue-error repair**: when `queue_prompt` returns an HTTP 4xx rejection,
  the agent receives the exact error message and gets up to
  `max_repair_attempts` (default 2) chances to inspect and fix the workflow
  topology before the iteration is abandoned.
- **Execution-error repair**: when ComfyUI reports an execution-time error
  (wrong types, invalid connections, missing inputs, etc.), the same repair
  loop applies to the running graph.
- **Infrastructure-fault detection**: errors matching `[Errno 32] Broken pipe`
  or `BrokenPipeError` are classified as transient ComfyUI infrastructure
  faults (caused by tqdm writing to a closed stderr pipe). These bypass the
  agent-repair loop entirely; the harness waits 5 s and retries the same
  workflow once without asking the agent to modify anything.
- New `_build_repair_feedback` helper produces structured, actionable feedback
  for the agent: verbatim error string, step-by-step fix instructions, list of
  common root causes, and previous verifier feedback for context.
- New `--max-repair-attempts N` CLI flag (also `COMFYCLAW_MAX_REPAIR_ATTEMPTS`
  env var) to tune or disable the repair loop at runtime.

#### Remote-access fixes for ComfyClaw-Sync
- `SyncServer` default bind host changed from `127.0.0.1` to `0.0.0.0`, so the
  WebSocket server is reachable when ComfyUI is accessed over a remote tunnel
  or container port-forward.
- `comfy_claw_sync.js` default WebSocket URL changed from the hardcoded
  `ws://127.0.0.1:8765` to `ws://${window.location.hostname}:8765`, which
  automatically follows the browser's current hostname.
- `custom_node/__init__.py` docstring updated to explain the dynamic URL
  resolution.

#### Active-model injection in agent context
- `ClawAgent._build_user_message` now extracts the active checkpoint/UNET model
  name directly from the workflow (checking `CheckpointLoaderSimple`,
  `CheckpointLoader`, and `UNETLoader` nodes) and surfaces it in the user
  message so the agent can match model-specific skills without needing to call
  `inspect_workflow` first.

### Fixed

#### VAE output-slot bug in `_add_hires_fix` and `_add_inpaint_pass`
- Previously the VAE input on every new `VAEDecode` node was hardcoded to slot
  `0`, which is the `MODEL` output on `CheckpointLoaderSimple` (VAE is slot
  `2`). The fix dynamically copies the `vae` connection from an existing
  `VAEDecode` node in the graph, so both `CheckpointLoaderSimple` (slot 2) and
  standalone `VAELoader` (slot 0) are handled correctly.

### Tests

- Added `TestRepairLoop` test class in `tests/test_harness.py` (6 new cases):
  - `test_queue_error_triggers_agent_repair` — queue rejection wires through to
    `plan_and_patch`.
  - `test_repair_feedback_contains_error_message` — repair feedback includes the
    verbatim error string.
  - `test_repair_exhausted_records_error` — all repair attempts exhausted leaves
    the error in memory and continues to the next iteration.
  - `test_repair_success_produces_image` — a repair that fixes the workflow
    produces an image and records it normally.
  - `test_execution_error_triggers_repair` — execution-time errors also feed the
    repair loop.
  - `test_build_repair_feedback_content` — feedback string contains the error
    message, instructions, and common-causes list.

---

## [0.1.0] — 2025-12-10

Initial release. See commit `3342c7b` for full details.

- Agent-driven ComfyUI workflow generation and evolution loop.
- Claude Vision verifier with region-level analysis and iteration feedback.
- LoRA injection, ControlNet, hires-fix, and inpaint-pass topology tools.
- ComfyClaw-Sync custom node for live canvas updates in the ComfyUI web UI.
- Built-in skills: `photorealistic`, `high-quality`, `hires-fix`,
  `lora-enhancement`.
