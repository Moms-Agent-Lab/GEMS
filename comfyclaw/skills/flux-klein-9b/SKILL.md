---
name: flux-klein-9b
description: >-
  Configuration guide for FLUX.2 [klein] 9B, Black Forest Labs' fast 9B DiT
  text-to-image model (Apr 2026). Uses an ADVANCED sampling pipeline:
  SamplerCustomAdvanced + CFGGuider + Flux2Scheduler — NOT standard KSampler.
  Detect when UNETLoader contains "flux-2-klein" or "flux2-klein" in the model
  name, or CLIPLoader type is "flux2". LoRA and ControlNet are not supported
  for this model.
license: Apache-2.0
metadata:
  author: davidliuk
  version: "1.0.0"
  base_arch: FLUX.2 [klein] 9B (DiT, BF16)
  diffusion_model: flux-2-klein-9b.safetensors        → ComfyUI/models/diffusion_models/
  text_encoder:    qwen_3_8b_fp8mixed.safetensors      → ComfyUI/models/text_encoders/
  vae:             flux2-vae.safetensors                → ComfyUI/models/vae/
---

FLUX.2 [klein] is Black Forest Labs' distilled fast-inference model family.
The 9B variant produces high-quality images in just 4 steps. It uses a
**completely different sampling pipeline** from standard KSampler models.

---

## ⚠️ CRITICAL: This model does NOT use KSampler

FLUX.2 Klein uses `SamplerCustomAdvanced` with `CFGGuider` and `Flux2Scheduler`.
**Do NOT add KSampler, KSamplerAdvanced, or any standard sampler node.**
**Do NOT try to rewire the pipeline to use KSampler — it will fail.**

| Feature | Standard models | FLUX.2 Klein 9B |
|---|---|---|
| Sampler node | `KSampler` | **`SamplerCustomAdvanced`** |
| Guidance | cfg param on KSampler | **`CFGGuider` (separate node)** |
| Scheduler | scheduler param on KSampler | **`Flux2Scheduler` (separate node)** |
| Noise | built into KSampler | **`RandomNoise` (separate node)** |
| Sampler select | sampler_name on KSampler | **`KSamplerSelect` (separate node)** |
| Latent | `EmptySD3LatentImage` | **`EmptyFlux2LatentImage`** |
| Negative cond | `CLIPTextEncode` | **`ConditioningZeroOut`** |
| Steps | varies | **4** (fixed, do not change) |
| CFG | varies | **1** (fixed, do not change) |
| Text encoder type | varies | **`flux2`** |
| LoRA | yes | **NOT supported** |
| ControlNet | yes | **NOT supported** |

---

## 1. Node graph structure

```
UNETLoader ("flux-2-klein-9b.safetensors", weight_dtype="default")
    └──► CFGGuider (cfg=1)
             ◄── CLIPTextEncode (positive) ← CLIPLoader (type="flux2")
             ◄── ConditioningZeroOut ← CLIPTextEncode (empty text) ← CLIPLoader
         └──► SamplerCustomAdvanced
                  ◄── RandomNoise (noise_seed=42)
                  ◄── KSamplerSelect (sampler_name="euler")
                  ◄── Flux2Scheduler (steps=4, width=1024, height=1024)
                  ◄── EmptyFlux2LatentImage (1024×1024)
              └──► VAEDecode ◄── VAELoader ("flux2-vae.safetensors")
                       └──► SaveImage
```

### Connection map (node IDs from base workflow)

| Node ID | Class | Key inputs |
|---|---|---|
| 1 | `UNETLoader` | unet_name, weight_dtype="default" |
| 2 | `CLIPLoader` | clip_name, type="flux2" |
| 3 | `VAELoader` | vae_name |
| 4 | `CLIPTextEncode` | clip=[2,0], text="<positive prompt>" |
| 5 | `CLIPTextEncode` | clip=[2,0], text="" |
| 6 | `ConditioningZeroOut` | conditioning=[5,0] |
| 7 | `CFGGuider` | model=[1,0], positive=[4,0], negative=[6,0], cfg=1 |
| 8 | `Flux2Scheduler` | steps=4, width=1024, height=1024 |
| 9 | `EmptyFlux2LatentImage` | width=1024, height=1024, batch_size=1 |
| 10 | `RandomNoise` | noise_seed=42 |
| 11 | `KSamplerSelect` | sampler_name="euler" |
| 12 | `SamplerCustomAdvanced` | noise=[10,0], guider=[7,0], sampler=[11,0], sigmas=[8,0], latent_image=[9,0] |
| 13 | `VAEDecode` | samples=[12,0], vae=[3,0] |
| 14 | `SaveImage` | images=[13,0] |

---

## 2. What you CAN change

Only these modifications are safe:

### Prompt text
```
set_param("<CLIPTextEncode positive ID>", "text", "<improved prompt>")
```

### Resolution (both Flux2Scheduler AND EmptyFlux2LatentImage must match)
```
set_param("<Flux2Scheduler ID>", "width", 1280)
set_param("<Flux2Scheduler ID>", "height", 720)
set_param("<EmptyFlux2LatentImage ID>", "width", 1280)
set_param("<EmptyFlux2LatentImage ID>", "height", 720)
```

### Seed
```
set_param("<RandomNoise ID>", "noise_seed", 12345)
```

---

## 3. What you MUST NOT change

- **Do NOT change CFGGuider cfg** — must stay at 1
- **Do NOT change Flux2Scheduler steps** — must stay at 4
- **Do NOT change KSamplerSelect sampler_name** — must stay at "euler"
- **Do NOT add KSampler** — this model uses SamplerCustomAdvanced exclusively
- **Do NOT add LoRA** — not supported for FLUX.2 Klein architecture
- **Do NOT add ControlNet** — not supported for FLUX.2 Klein architecture
- **Do NOT add ModelSamplingAuraFlow** — not needed for this model
- **Do NOT replace ConditioningZeroOut with CLIPTextEncode for negative** — model requires zeroed negative conditioning
- **Do NOT disconnect or rewire the SamplerCustomAdvanced inputs** — all 5 inputs (noise, guider, sampler, sigmas, latent_image) must be connected

Violating any of these will cause `prompt_outputs_failed_validation` errors or
garbage output.

---

## 4. Resolution buckets

| Aspect | Width | Height |
|---|---|---|
| 1:1 (default) | 1024 | 1024 |
| 16:9 | 1280 | 720 |
| 9:16 | 720 | 1280 |
| 4:3 | 1152 | 864 |
| 3:4 | 864 | 1152 |

**IMPORTANT:** When changing resolution, update BOTH `Flux2Scheduler` AND
`EmptyFlux2LatentImage` — they must match.

---

## 5. Prompt engineering

FLUX.2 Klein uses Qwen3-8B as text encoder. Use natural language descriptions.
Keep prompts concise — this is a fast model optimised for 4 steps, so overly
complex prompts may degrade quality.

```
Good:
"A green backpack and a pig on a white studio background, professional product photography"

Poor:
"green backpack, pig, 8k, masterpiece, highly detailed, ultra realistic"
```

**Negative prompt:** Not used — FLUX.2 Klein uses `ConditioningZeroOut`.

---

## 6. Iteration strategy

Since LoRA and ControlNet are NOT available, improvements come entirely from
prompt engineering and resolution changes.

| Verifier issue | Fix strategy |
|---|---|
| Wrong subject | Rewrite prompt with clearer subject description |
| Wrong count | Use explicit counting: "exactly three cats" |
| Wrong attribute | Emphasize attribute: "a bright RED car" |
| Wrong composition | Add spatial language: "on the left", "in the foreground" |
| Low detail | Increase resolution to 1280×720 or 1152×864 |
| Blurry | Simplify prompt — fewer subjects, clearer description |

**NEVER modify the sampling pipeline nodes.** The only tools available are
`set_param` for prompt text, resolution, and seed changes.

---

## 7. Example

```
inspect_workflow()

set_param("<CLIPTextEncode positive ID>", "text",
    "A majestic golden eagle perched on a snow-covered pine branch. "
    "Mountain landscape in the background, dramatic sunset sky, "
    "photorealistic, sharp detail.")

set_param("<Flux2Scheduler ID>", "width", 1280)
set_param("<Flux2Scheduler ID>", "height", 720)
set_param("<EmptyFlux2LatentImage ID>", "width", 1280)
set_param("<EmptyFlux2LatentImage ID>", "height", 720)

finalize_workflow()
```
