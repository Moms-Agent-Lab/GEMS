---
name: learned-errors
description: Read this when building or modifying ComfyUI workflows to avoid common validation and wiring errors
---

# ComfyUI Error Prevention Guide

## Exception During Inner Validation Errors

**Most common cause**: Incorrect node wiring or missing connections

These errors (`exception_during_inner_validation`) typically indicate:
1. **Wrong input type connected** - e.g., passing IMAGE where LATENT expected
2. **Missing required connection** - a mandatory input is unwired
3. **Invalid slot reference** - referring to output slot that doesn't exist
4. **Node ID doesn't exist** - referencing a node that isn't in the workflow

### Systematic Debugging for Inner Validation Errors
When you see `exception_during_inner_validation` on a specific node ID:

1. **Verify the node exists** - Check that the node ID is actually defined in your workflow
2. **Check ALL inputs are wired** - Every required input must have a valid connection
3. **Verify connection types match** - MODEL→MODEL, CLIP→CLIP, VAE→VAE, LATENT→LATENT, IMAGE→IMAGE
4. **Validate source slot indices** - Ensure you're using the correct output slot from upstream nodes
5. **Check source nodes exist** - All nodes referenced in connections must be present

### Common Patterns That Cause This Error
❌ Connecting to a node ID that doesn't exist in the workflow
❌ Using wrong slot index (e.g., slot 2 when node only has slots 0-1)
❌ Type mismatch (CLIP connected to VAE input)
❌ Forgetting to wire a required input

## Node Output Slot Indices

**Critical: Slots are 0-indexed. Verify each node's output order.**

### Standard Loader Nodes

**CheckpointLoaderSimple**
- Slot 0: MODEL
- Slot 1: CLIP
- Slot 2: VAE

**LoraLoader**
- Slot 0: MODEL
- Slot 1: CLIP

**VAELoader**
- Slot 0: VAE

### Encoder/Decoder Nodes

**CLIPTextEncode**
- Slot 0: CONDITIONING

**VAEEncode**
- Slot 0: LATENT

**VAEDecode**
- Slot 0: IMAGE

### Sampler Nodes

**KSampler**
- Slot 0: LATENT

**KSamplerAdvanced**
- Slot 0: LATENT

### Common Wiring Mistakes

❌ **WRONG**: `CheckpointLoaderSimple` slot 1 → MODEL input (slot 1 is CLIP!)
✅ **CORRECT**: `CheckpointLoaderSimple` slot 0 → MODEL input

❌ **WRONG**: Connecting CLIP output to VAE input
✅ **CORRECT**: Match types exactly - check both slot index AND data type

❌ **WRONG**: Referencing slot 2 on a node with only 2 outputs
✅ **CORRECT**: 2 outputs = slots [0, 1] only

## Parameter Value Constraints

### Numeric Ranges
- **Strength/weight/denoise parameters**: Must be in [0.0, 1.0]
- **Common violations**:
  - `conditioning_to_strength` in `ConditioningSetTimestepRange` must be ≤ 1.0
  - Setting values > 1.0 causes `value_bigger_than_max` errors
- **Fix**: Clamp to valid range before setting

### ConditioningSetTimestepRange Specific
- `start` must be < `end`
- Both `start` and `end` must be in [0.0, 1.0]

### KSampler Parameters
- `denoise`: [0.0, 1.0]
- `steps`: positive integer
- Requires: MODEL, positive CONDITIONING, negative CONDITIONING, LATENT (all wired)

## Pre-Submission Checklist

Before submitting ANY workflow:
- [ ] All referenced node IDs actually exist in the workflow
- [ ] Every required input is connected
- [ ] All connections use correct slot indices (0-indexed)
- [ ] Data types match: MODEL→MODEL, CLIP→CLIP, VAE→VAE, LATENT→LATENT, IMAGE→IMAGE
- [ ] Checkpoint loader connections: slot 0=MODEL, slot 1=CLIP, slot 2=VAE
- [ ] All strength/weight/denoise values are in [0.0, 1.0]
- [ ] No references to non-existent slots (don't use slot 2 if node only has 2 outputs)

## Quick Reference: Output Slots by Node Type

```
CheckpointLoaderSimple → [0: MODEL, 1: CLIP, 2: VAE]
LoraLoader           → [0: MODEL, 1: CLIP]
VAELoader            → [0: VAE]
CLIPTextEncode       → [0: CONDITIONING]
VAEEncode            → [0: LATENT]
VAEDecode            → [0: IMAGE]
KSampler             → [0: LATENT]
```

**When connecting nodes**: Always verify the source node's slot index matches the required input type.