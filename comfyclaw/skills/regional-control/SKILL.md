---
name: regional-control
description: >-
  Apply separate text prompts to distinct image regions to ensure all objects appear and maintain their individual characteristics. Use when generating multiple distinct objects or animals together (e.g., 'backpack AND pig', 'four rabbits AND sheep'), when the verifier reports missing objects, wrong counts, style contamination between subject and background, or when fix_strategy contains 'add_regional_prompt'. Also useful when the user explicitly asks for 'different styles in different parts' of the image.
license: MIT
metadata:
  author: "davidliuk"
  version: "0.2.0"
tags: [agent]
---

# regional-control

## Purpose
Apply separate conditioning to distinct image regions to ensure:
1. **All requested objects appear** (preventing object dropout in multi-object scenes)
2. **Correct counts** when multiple instances of different object types are specified
3. **Style isolation** between subject and background or between different objects
4. **Spatial coherence** in complex compositions

## When to Use
- User prompt contains multiple distinct objects or animals ("a cat and a dog", "backpack and pig", "four rabbits and one sheep")
- Verifier reports missing objects or incorrect counts
- Verifier reports style contamination or mismatched elements
- Background is too plain or generic relative to subject
- User explicitly requests "different styles in different parts"
- fix_strategy contains "add_regional_prompt"

## Detection Patterns
- Prompts with "AND" between object types
- Numeric counts with multiple object types ("four X and one Y")
- Multiple subjects that need different treatment ("person in foreground, mountains in background")

## Implementation Strategy

### Step 1: Parse Object Requirements
Analyze the prompt to identify:
- Distinct object types and their counts
- Spatial relationships (if specified)
- Style differences between elements

Example: "four rabbits and one sheep"
- Object A: rabbits (count: 4)
- Object B: sheep (count: 1)
- Total objects: 5 distinct entities needed

### Step 2: Design Region Layout
For multi-object scenes:
- **Horizontal split**: Use for objects that should be side-by-side ("backpack and pig")
  - Left region: 0.0-0.5 width
  - Right region: 0.5-1.0 width
- **Grid layout**: Use for multiple instances ("four rabbits and one sheep")
  - Divide canvas into proportional regions based on object counts
  - Example for 4+1: four smaller regions (0.0-0.5 width, split top/bottom) + one larger region (0.5-1.0)
- **Foreground/background split**: Use for subject vs. environment
  - Center region for subject
  - Outer region for background

### Step 3: Configure Regional Conditioning Nodes
Use ComfyUI regional conditioning nodes (search for "RegionalConditioner", "ConditioningSetArea", or "ConditioningCombine"):

```
For each distinct object/region:
1. Create separate CLIPTextEncode for that object's description
2. Apply ConditioningSetArea with calculated bounds:
   - x: horizontal start (0.0-1.0)
   - y: vertical start (0.0-1.0) 
   - width: region width (0.0-1.0)
   - height: region height (0.0-1.0)
   - strength: 1.0 for hard isolation, 0.7-0.9 for soft blending
3. Combine all regional conditionings with ConditioningCombine
4. Feed combined conditioning to KSampler
```

### Step 4: Craft Region-Specific Prompts
For each region, create focused prompts:
- **Object regions**: Describe ONLY that object with full detail
  - "a fluffy white rabbit, detailed fur, alert ears, sitting pose"
  - "a brown and white sheep, wooly texture, standing"
- **Avoid mentioning other objects** in each region's prompt
- **Background region**: Describe environment without mentioning subjects
  - "grassy meadow, soft lighting, natural environment"

### Step 5: Handle Counts Explicitly
When specific counts are required:
- Create one region per instance if count ≤ 3
- For counts > 3, use larger regions with explicit count in prompt: "four rabbits"
- Consider grid arrangements: 2×2 for four objects, 2×3 for six, etc.

### Step 6: Verify Coverage
Ensure:
- All regions sum to full canvas coverage (no gaps)
- No excessive overlap (causes double-conditioning artifacts)
- Each requested object type has dedicated region(s)
- Background region fills remaining space

## Example Workflow Modifications

### Example 1: "a green backpack and a pig"
```
Region A (left half, x=0.0, width=0.5):
  Prompt: "a green backpack, canvas material, detailed straps and buckles"
  
Region B (right half, x=0.5, width=0.5):
  Prompt: "a pink pig, realistic farm animal, curly tail, snout"
  
Background (full canvas, strength=0.3):
  Prompt: "simple background, neutral setting, soft lighting"
```

### Example 2: "four rabbits and one sheep"
```
Region grid (2×2 for rabbits + 1 larger for sheep):
  Top-left (x=0.0, y=0.0, w=0.25, h=0.5): "white rabbit"
  Top-center-left (x=0.25, y=0.0, w=0.25, h=0.5): "brown rabbit"
  Bottom-left (x=0.0, y=0.5, w=0.25, h=0.5): "grey rabbit"
  Bottom-center-left (x=0.25, y=0.5, w=0.25, h=0.5): "spotted rabbit"
  Right side (x=0.5, y=0.0, w=0.5, h=1.0): "one sheep, wooly white sheep"
  
Background (full canvas, strength=0.2): "pastoral meadow, grass"
```

## Troubleshooting
- **Objects still missing**: Increase region strength to 1.0, ensure no overlap with background
- **Hard edges between regions**: Lower strength to 0.7-0.8, add small overlap (0.05)
- **Style bleed**: Make prompts more focused, remove shared descriptors
- **Wrong counts**: Create explicit regions per instance, or use count number in prompt

## Tools to Query
- `query_available_nodes('regional')` or `query_available_nodes('condition')`
- Look for: ConditioningSetArea, ConditioningCombine, RegionalConditioner, or similar

## Success Criteria
- All requested objects appear in generated image
- Object counts match user specification
- Each object maintains its described characteristics
- Clean separation or natural blending based on scene requirements