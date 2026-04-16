---
name: counting-objects
description: >-
  Restructure prompts to enforce precise object counts (especially 4-7 items) using numerical emphasis, layout specification, and regional prompting for mixed object scenes.
license: MIT
metadata:
  cluster: "counting_multiple_objects"
  origin: "self-evolve"
---

# Counting Objects

Diffusion models fail at counting beyond 2-3 items due to token-level numeracy limits. For counts ≥4, explicit spatial layout and emphasis are required.

## When to Use
- User specifies exact counts: "four rabbits", "seven croissants", "six cars"
- Counts of 4 or more objects (success rate drops dramatically above 3)
- Mixed object scenes: "four rabbits AND a sheep", "five bears AND a donut"

## Technique

### For counts 4-7:
1. **Expand the number into emphatic tokens**: "four" → "(four:1.4), (4:1.3), exactly four"
2. **Add explicit layout language**: "arranged in a row", "spread across the scene", "clustered together", "in a grid pattern"
3. **Repeat the object noun with count**: "four brown monkeys, 4 monkeys, four of them"
4. **Use negative prompt**: "three monkeys, five monkeys, wrong count"

### For mixed objects (e.g., "four rabbits and a sheep"):
1. **Trigger regional-control skill** to assign separate regions
2. **Split into**: "(four rabbits:1.4), exactly 4 rabbits | (one sheep:1.3), single sheep"
3. **Use list syntax**: "four rabbits arranged in foreground, one sheep in background"

### Example transformations:
- "seven green croissants" → "(seven:1.4) (7:1.3) green croissants, exactly seven croissants arranged in a row, (green color:1.2)"
- "four rabbits and a sheep" → "(four white rabbits:1.4), exactly 4 rabbits in foreground | (one sheep:1.3), single sheep in background" + trigger regional-control
- "six cars and a kangaroo" → "(six cars:1.4), 6 cars spread across scene | (one kangaroo:1.3), single kangaroo" + trigger regional-control

## Integration
- Always combine with **regional-control** when multiple object types are present
- Always combine with **unusual-attributes** if colors/materials are atypical
- Add to negative prompt: common miscounts ("three", "five" when asking for four)

## Node impact
Modifies CLIPTextEncode positive/negative prompts only. No topology changes unless regional-control is triggered.