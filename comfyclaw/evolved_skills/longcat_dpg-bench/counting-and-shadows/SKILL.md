---
name: counting-and-shadows
description: >-
  Ensure accurate object counts and explicit shadow rendering through prompt restructuring and regional separation. Use when scenes require specific numbers of objects or shadows with particular characteristics.
license: MIT
metadata:
  cluster: "counting_and_shadow_rendering"
  origin: "self-evolve"
---

# Counting and Shadows

## When to Use
Trigger when:
- User specifies exact counts: 'three trucks', 'five apples', 'two cats'
- Verifier reports wrong number of objects generated
- Prompt describes shadow characteristics: 'long shadows', 'soft shadows', 'cast shadows', 'shadows on [surface]'
- fix_strategy contains 'fix_counting' or 'fix_shadows'
- Scene involves stacked/arranged objects where count matters

## Counting Strategy

### Prompt Restructuring
1. **Enumerate explicitly**: Replace 'three trucks' with 'first truck, second truck, third truck'
2. **Add positional anchors**: 'bottom truck painted red, middle truck faded blue, top truck dusty white'
3. **Use ordinal descriptors**: 'the first is red, the second is blue, the third is white'
4. **Negative prompt**: Add '(fewer objects:1.3), (extra objects:1.3), wrong count'

### Regional Prompting for Counts
When objects exceed 2, use regional-control skill to separate each object:
- Divide canvas into regions matching object positions
- Assign one prompt per object with unique identifiers
- Example: Region 1 'red pickup truck at bottom', Region 2 'blue pickup truck in middle', Region 3 'white pickup truck on top'

## Shadow Strategy

### Explicit Shadow Tokens
1. **Shadow type**: 'hard shadows', 'soft diffused shadows', 'contact shadows'
2. **Shadow direction**: 'shadows extending to the right', 'long shadows toward camera'
3. **Shadow surface**: 'shadows cast on gravel', 'shadow on concrete'
4. **Lighting cue**: Add 'late afternoon sun' or 'low angle lighting' to justify long shadows

### Workflow Additions
If shadows remain incorrect after prompt fixes:
- Consider controlnet-control with depth map to enforce shadow geometry
- Add '(no shadows:1.5)' to negative prompt if shadows appear where they shouldn't
- Increase CFG slightly (+0.5 to +1.0) to strengthen shadow adherence

## Example Transformations

**Before**: 'three trucks stacked with long shadows'

**After**: 'first pickup truck at the bottom painted bright red, second pickup truck stacked on top with faded blue paint, third pickup truck balanced at the very top with dusty white paint, hard-edged long shadows cast on gravel lot extending to the right, late afternoon sunlight, each truck clearly separated and distinct'

**Negative**: '(two trucks:1.4), (four trucks:1.4), (merged trucks:1.3), (no shadows:1.3), (short shadows:1.2)'

## Integration
- Always combine with spatial skill for multi-object layouts
- Layer with regional-control when count ≥ 3
- Use with photorealistic for natural shadow rendering
- Apply unusual-attributes if shadow colors/styles deviate from natural