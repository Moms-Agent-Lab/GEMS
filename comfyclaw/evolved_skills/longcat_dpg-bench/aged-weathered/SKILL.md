---
name: aged-weathered
description: >-
  Restructure prompts to depict age, wear, deterioration, and weathered conditions on objects and subjects that models typically render as pristine or new.
license: MIT
metadata:
  cluster: "attribute_depiction_failure"
  origin: "self-evolve"
---

# Aged & Weathered Attribute Skill

## When to Use
Trigger when the user requests:
- Age indicators: "elderly", "old", "ancient", "aged"
- Wear states: "weathered", "worn", "battered", "distressed"
- Damage: "dented", "scratched", "chipped", "cracked", "rusted"
- Degradation: "faded", "dusty", "dirty", "stained", "tarnished"
- Time effects: "vintage", "antique", "decaying", "deteriorated"

## Problem
Diffusion models have strong priors toward clean, pristine, idealized versions of objects. Simply adding "old" or "weathered" often gets ignored or produces minimal effect.

## Solution Strategy

### 1. Anatomical Age Features (for living subjects)
For elderly animals/people, add explicit age markers:
- "gray/white fur", "wrinkled skin", "cloudy eyes"
- "grizzled muzzle", "sagging features", "thinning coat"
- Position age terms early: "(elderly:1.3) raccoon with gray-streaked fur"

### 2. Surface Wear Tokens
For objects, layer multiple wear descriptors:
- "heavily scratched", "deep dents", "peeling paint"
- "surface rust", "oxidized metal", "patina"
- "cracked leather", "frayed edges", "worn finish"

### 3. Environmental Weathering
Add context clues:
- "covered in dust and grime", "sun-faded colors"
- "water stains", "mud splatter", "road worn"
- "exposed to elements", "years of neglect"

### 4. Emphasis Syntax
Use ComfyUI emphasis for critical terms:
- "(weathered:1.4)", "(dented and scratched:1.3)"
- "(faded blue paint:1.2)", "(elderly:1.3)"

### 5. Negative Prompts
Counter the pristine bias:
- "pristine, new, clean, polished, shiny, mint condition"
- "flawless, perfect, undamaged, factory fresh"

### 6. Style Anchors
Reference photography styles that capture wear:
- "documentary photography", "gritty realism"
- "urban decay aesthetic", "found object photography"

## Example Transformations

**Before:** "an elderly raccoon with a top hat"
**After:** "(elderly:1.3) raccoon with (gray-streaked fur:1.2), wrinkled muzzle, cloudy eyes, wearing a top hat, documentary photography style"
Negative: "young, pristine, glossy fur, bright eyes"

**Before:** "stacked pickup trucks, red, blue, and white, dented and scratched"
**After:** "precarious stack of three (heavily weathered:1.3) pickup trucks: bottom truck (rust-spotted red paint:1.2) with deep dents, middle truck (sun-faded blue:1.2) with scratches, top truck (dusty white:1.2) with chipped paint, (damaged body panels:1.2), dirt and grime, years of hard use"
Negative: "new, shiny, polished, clean, showroom condition, pristine paint"

## Implementation
This is a **prompt-only** skill. Modify the positive and negative conditioning text before CLIPTextEncode nodes. No workflow topology changes needed.

## Pair With
- **photorealistic**: For realistic wear documentation
- **high-quality**: To ensure texture detail is visible
- **creative**: For stylized decay/patina effects