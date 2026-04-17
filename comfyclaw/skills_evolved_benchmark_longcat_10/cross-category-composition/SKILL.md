---
name: cross-category-composition
description: >-
  Generate scenes mixing distinct object categories using regional prompts, attention balancing, and category isolation to prevent merging or omission
license: MIT
metadata:
  cluster: "multi_object_composition"
  origin: "self-evolve"
---

# Cross-Category Composition

## When to Use
Trigger when prompts mix distinct categories:
- Animals + objects (pig and backpack, bears and donut)
- Multiple animal types (rabbits and sheep, cars and kangaroo)
- Objects + plants (car and flowers)
- Any scene where 2+ semantically distant concepts appear together

Also trigger when verifier reports: missing objects, merged objects, category contamination, or wrong object types.

## Core Problem
Diffusion models collapse distinct categories into hybrid forms or omit weaker concepts entirely. "Pig and backpack" becomes "pig-shaped backpack" or just "pig".

## Solution Strategy

### 1. Regional Prompt Separation
Split each category into its own ConditioningSetArea:
```
Prompt 1 (left/top region): "[COUNT] [OBJECT_A], highly detailed"
Prompt 2 (right/bottom region): "[COUNT] [OBJECT_B], highly detailed"
Base prompt: "[OBJECT_A] and [OBJECT_B] together in one scene"
```
Use ConditioningCombine to merge regional prompts with strength 0.9 each.

### 2. Attention Isolation
For each regional prompt, add negative tokens of the OTHER category:
- Region A negative: "no [OBJECT_B], without [OBJECT_B]"
- Region B negative: "no [OBJECT_A], without [OBJECT_A]"
This prevents semantic bleeding.

### 3. Explicit Enumeration
When counts are involved, spell out each item:
- "four rabbits" → "first rabbit, second rabbit, third rabbit, fourth rabbit"
- "five bears" → "bear 1, bear 2, bear 3, bear 4, bear 5"

### 4. Spatial Anchoring
Add physical layout cues to each region:
- "on the left side", "in the foreground"
- "on the right side", "in the background"
- "arranged in a row", "grouped together"

### 5. Workflow Modifications
- Set CFG to 8.5-9.5 (higher guidance prevents category collapse)
- Use steps >= 35 for complex multi-object scenes
- If using regional prompts, divide image into 60/40 or 50/50 splits
- Consider MultiAreaConditioning if available for 3+ regions

### 6. Fallback: Iterative Composition
If regional prompts fail (not available or still failing):
- Generate each category separately with identical scene context
- Use image composition tools or inpainting to merge
- This is slower but guarantees category preservation

## Example Transformation
Input: "a green backpack and a pig"
Output:
- Region 1 (left 50%): "a green backpack, outdoor gear, detailed textures, realistic materials" + negative "no pig, no animal"
- Region 2 (right 50%): "a pig, farm animal, natural fur texture, realistic anatomy" + negative "no backpack, no bag"
- Base: "green backpack and pig together in outdoor scene, photorealistic"
- CFG: 9.0, steps: 40

## Validation
After generation, verify:
1. Both categories present (not merged)
2. Correct count for each category
3. No hybrid artifacts
4. Each object maintains category-appropriate features