---
name: learned-errors
description: Read this when building or modifying ComfyUI workflows to avoid common validation and execution errors
---

# ComfyUI Workflow Error Prevention

## Critical: Prompt Must Have Outputs

`Prompt has no outputs` means **no terminal node exists** in the workflow.

**Fix**: Every workflow MUST include at least one output node:
- `SaveImage`
- `PreviewImage`

**Without a terminal node, the workflow is invalid.**

## Required Input Errors - TOP PRIORITY

`Required input is missing` means a node has an unwired mandatory input or missing parameter in `inputs` dict.

### VAEDecode - MOST COMMON ERROR

**ALWAYS verify VAEDecode has BOTH inputs wired:**
- **`samples`** - LATENT from KSampler slot 0 or other LATENT source
  - **NEVER pass string literal, file path, null, or leave unwired**
  - **NEVER pass IMAGE data - only accepts LATENT type**
  - **NEVER use wrong wire reference format - MUST be `["node_id", 0]` array**
  - `'str' object has no attribute 'shape'` = **wrong data type passed to samples input**
- **`vae`** - VAE from CheckpointLoaderSimple slot 2

**Before submitting, ALWAYS check every VAEDecode node has valid `samples` wire in `["node_id", int_slot]` format from a LATENT source.**

### KSampler - ALL Inputs Required

**4 WIRED inputs (all mandatory - NEVER leave unwired):**
- **`model`** - from CheckpointLoaderSimple slot 0 (MODEL type - NEVER pass string/path)
  - `'str' object has no attribute 'get_model_object'` = you passed string instead of MODEL wire
- **`positive`** - from CLIPTextEncode slot 0 or conditioning chain
  - **NEVER leave unwired or null - ALWAYS wire from CONDITIONING source**
- **`negative`** - from CLIPTextEncode slot 0 or conditioning chain
- **`latent_image`** - from EmptyLatentImage slot 0 or VAEEncode slot 0

**6 PARAMETERS in `inputs` dict (all mandatory - NEVER omit):**
- **`seed`** - **MUST be integer >= 0** (NEVER -1, null, float like `123.0`, or string like `"123"`)
  - `Failed to convert an input value to a INT value` = you passed float/string instead of int
  - **ALWAYS use Python int type: `123` not `123.0` or `"123"`**
- **`steps`** - integer 1-10000
- **`cfg`** - float 0.0-30.0
- **`denoise`** - float 0.0-1.0
- **`sampler_name`** - string
- **`scheduler`** - string

### EmptyLatentImage - All Parameters Required

**3 PARAMETERS in `inputs` dict (all mandatory):**
- **`width`** - **MUST be integer** (64-16384, typically 512, 1024)
  - `Failed to convert an input value to a INT value` = you passed float/string instead of int
- **`height`** - **MUST be integer** (64-16384, typically 512, 1024)
- **`batch_size`** - **MUST be integer** (1-4096, typically 1)

### CLIPTextEncode - Both Required

- **`text`** - **MUST be non-empty string in `inputs` dict** (NEVER null, empty string, or omitted)
- **`clip`** - wired from CheckpointLoaderSimple slot 1

### ControlNetApply - All Three Required

- **`conditioning`** - wired from CLIPTextEncode slot 0 or previous ControlNetApply slot 0
- **`control_net`** - wired from ControlNetLoader slot 0
- **`image`** - wired from LoadImage slot 0 or preprocessor output

### FluxGuidance - Both Required

- **`guidance`** - **MUST be float parameter in `inputs` dict** (typically 3.0-4.0)
  - **NEVER omit - this is REQUIRED parameter, not optional**
  - `Required input is missing: guidance` = you forgot to add `guidance` to `inputs` dict
- **`conditioning`** - wired CONDITIONING input from CLIPTextEncode slot 0

### Other Common Missing Inputs

- **SaveImage/PreviewImage**: `images` (from VAEDecode slot 0 - MUST be IMAGE type)
- **VAEEncode**: `pixels` (from LoadImage slot 0), `vae` (from checkpoint slot 2)

## Critical Node Output Slots

**CheckpointLoaderSimple**:
- Slot 0: MODEL
- Slot 1: CLIP
- Slot 2: VAE

**KSampler**: Slot 0: LATENT
**VAEDecode**: Slot 0: IMAGE
**VAEEncode**: Slot 0: LATENT
**CLIPTextEncode**: Slot 0: CONDITIONING
**EmptyLatentImage**: Slot 0: LATENT
**LoadImage**: Slot 0: IMAGE, Slot 1: MASK
**ControlNetLoader**: Slot 0: CONTROL_NET
**ControlNetApply**: Slot 0: CONDITIONING
**FluxGuidance**: Slot 0: CONDITIONING

## Wire Reference Format - CRITICAL

**ALL wire references MUST be `["node_id", slot_index]` where:**
- `node_id` is **string** matching existing node's `id`
- `slot_index` is **integer** (0, 1, or 2)
- Array has exactly 2 non-null elements

**Common errors:**
- Using dict: `{"node": "1", "slot": 0}` ❌ → `["1", 0]` ✓
- String slot: `["1", "0"]` ❌ → `["1", 0]` ✓
- Null values: `["1", null]` ❌ → `["1", 0]` ✓

## Parameter Type Errors - CRITICAL

**Integer parameters** (use Python int in `inputs` dict - NEVER float/string):
- **`seed`** - **MUST be Python int >= 0** (❌ `123.0`, `"123"`, `-1` | ✓ `123`)
- **`width`**, **`height`**, **`batch_size`** - **MUST be Python int** (❌ `512.0`, `"512"` | ✓ `512`)
- **`steps`** - **MUST be Python int**
- `Failed to convert an input value to a INT value` = **you passed float or string instead of int**

**Float parameters** (use Python float in `inputs` dict):
- **`guidance`** (FluxGuidance - **REQUIRED**, typically 3.0-4.0)
- **`conditioning_to_strength`** - **MUST be 0.0-1.0, NEVER exceed 1.0**
  - `Value X bigger than max of 1.0` = you set conditioning_to_strength > 1.0
- `denoise` (0.0-1.0), `cfg` (0.0-30.0)

**String parameters** (non-empty string in `inputs` dict):
- **`text`** (for CLIPTextEncode - REQUIRED, NEVER null/empty/omitted)
- `sampler_name`, `scheduler`

## Runtime Type Errors

**"'str' object has no attribute 'shape'"** means:
- VAEDecode `samples` input has wrong data type (string/IMAGE instead of LATENT)
- **FIX**: Verify wire connects to correct output slot of correct node type

**"'str' object has no attribute 'get_model_object'"** means:
- You passed a string instead of MODEL wire to `model` input
- **FIX**: Wire from CheckpointLoaderSimple slot 0, NEVER