---
name: exact-counting
description: >-
  Force generation of exact object counts using repetition syntax, regional prompting, and latent composition when users specify precise numerosity (e.g., 'five cats', 'seven croissants', 'twelve stars').
license: MIT
metadata:
  cluster: "counting_failure"
  origin: "self-evolve"
---

# Exact Counting Skill

## When to use
Trigger when:
- User specifies exact numeric quantities: "five", "seven", "twelve", "3 cats", "8 croissants"
- Verifier reports wrong object count or missing instances
- fix_strategy contains "fix_counting" or "add_regional_for_count"
- Numbers > 3 (diffusion models naturally handle 1-3 objects)

## Core problem
Diffusion models lack true counting ability. They approximate numerosity through semantic associations but fail at exact counts beyond 3-4 objects.

## Solution patterns

### Pattern 1: Repetition syntax (for counts 4-6)
Rewrite prompt to repeat the object multiple times:
- "seven green croissants" → "green croissant, green croissant, green croissant, green croissant, green croissant, green croissant, green croissant"
- Add to positive prompt: "exactly [N], [N] distinct objects, all [N] visible"
- Add to negative prompt: "fewer than [N], more than [N], merged objects, overlapping"

### Pattern 2: Regional prompting (for counts 4-8)
Use RegionalPromptSimple or RegionalConditioningSimple to divide the image into N regions:
1. Plan a grid layout (2×3 for 6 objects, 2×4 for 8 objects)
2. Create one region per object with individual mask
3. Apply same object prompt to each region
4. Use base prompt for background/context

### Pattern 3: Latent composition (for counts 7+)
Generate objects in smaller batches, then compose:
1. Generate 3-4 objects in first pass
2. Generate remaining objects in second pass
3. Use LatentComposite or image inpainting to merge
4. Requires calling workflow tools multiple times

### Pattern 4: Layout enforcement
Combine with spatial skill:
- Arrange objects in clear patterns: "arranged in a circle", "in two rows", "evenly spaced"
- Use ControlNet depth/canny to lock positions if available

## Implementation priority
1. For counts 4-6: Try repetition syntax first (fastest)
2. For counts 7-9: Use regional prompting
3. For counts 10+: Warn user that accuracy drops significantly; suggest latent composition or multiple images

## Example transformations
- "five red apples" → "red apple, red apple, red apple, red apple, red apple, arranged in a row, exactly 5 apples, all 5 visible"
- "seven green croissants" → Use RegionalPromptSimple with 7 regions (3×3 grid minus 2), each region conditioned on "green croissant, distinct object"

## Node recommendations
- RegionalPromptSimple (from ComfyUI-Impact-Pack)
- RegionalConditioningSimple
- ConditioningSetMask for manual region assignment
- Pair with spatial skill for layout language
- Pair with unusual-attributes for non-standard colors/materials

## Failure modes to avoid
- Don't just add number words without structural changes (ineffective)
- Don't use counting for tiny background details (focus on main subjects)
- Warn user when count exceeds 10 (model capability ceiling)