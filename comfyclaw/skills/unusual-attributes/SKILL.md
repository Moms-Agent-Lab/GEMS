---
name: unusual-attributes
description: >-
  Restructure prompts to force unusual or non-standard material, texture, and color attributes onto objects when they deviate from the object's typical appearance. Use when the user requests atypical colors (green croissants, purple trees, blue oranges), unusual materials (stone animals, glass furniture, metal food), or texture overrides (fuzzy rocks, smooth bark, transparent solid objects). Diffusion models have strong priors about object appearance — this skill uses emphasis syntax, material tokens, and strategic negative prompts to override those defaults.
license: MIT
metadata:
  cluster: "unusual_attribute_application"
  origin: "self-evolve"
tags: [agent]
---

---
name: unusual-attributes
description: Restructure prompts to force unusual or non-standard material, texture, and color attributes onto objects when they deviate from the object's typical appearance. Use when the user requests atypical colors (green croissants, purple trees, blue oranges), unusual materials (stone animals, glass furniture, metal food), or texture overrides (fuzzy rocks, smooth bark, transparent solid objects). Diffusion models have strong priors about object appearance — this skill uses emphasis syntax, material tokens, and strategic negative prompts to override those defaults.
---

# Unusual Attributes Skill

## Purpose
Diffusion models develop strong appearance priors during training. When you ask for "green croissants" the model wants to generate golden-brown pastries because that's what it has seen millions of times. This skill restructures prompts to force atypical attributes through emphasis, explicit material language, and negative prompts that suppress the model's default expectations.

## When to Use
Trigger this skill when the user requests:
- **Atypical colors**: green croissants, purple trees, blue oranges, black roses, rainbow bread
- **Unusual materials**: stone raccoons, glass chairs, metal toys, wooden clouds, crystal animals
- **Texture overrides**: fuzzy rocks, smooth tree bark, transparent solid objects, matte mirrors
- **Any deviation from typical object appearance** where the model's training data conflicts with the request

## Detection Patterns
Look for:
- `[material] [object]` where material is unusual: "stone raccoon", "metal toy", "glass apple"
- `[color] [object]` where color deviates from typical: "green croissant", "blue banana", "purple carrot"
- Texture descriptors that conflict with object nature: "smooth cactus", "fuzzy metal", "rough silk"

## Prompt Restructuring Strategy

### 1. Emphasis Syntax
Use parentheses with weight multipliers to force the unusual attribute:
```
Original: "green croissants"
Restructured: "(bright green:1.4) croissants, (green pastry:1.3), croissants with (vibrant green color:1.35)"

Original: "stone raccoons"
Restructured: "raccoons made of (solid stone:1.4), (stone texture:1.3), (gray stone material:1.35), raccoon sculptures"

Original: "metal toys"
Restructured: "toys made of (shiny metal:1.4), (metallic surface:1.3), (chrome finish:1.35), steel toys"
```

### 2. Material Token Injection
Explicitly name the material/attribute multiple times with variations:
- **Color overrides**: "bright green", "vibrant green", "lime green", "emerald green", "green-colored"
- **Material overrides**: "made of stone", "carved from stone", "solid stone", "stone texture", "stone material"
- **Texture overrides**: "smooth surface", "polished", "glossy", "matte", "rough texture"

### 3. Negative Prompts
Suppress the model's default expectations:
```
For green croissants:
Negative: "golden brown, tan, beige, normal croissant color, typical pastry color"

For stone raccoons:
Negative: "furry, soft fur, natural raccoon, living animal, brown fur, realistic fur texture"

For metal toys:
Negative: "plastic, wood, soft material, fabric, rubber, painted surface"
```

### 4. Context Reinforcement
Add scene context that makes the unusual attribute logical:
```
"(bright green:1.4) croissants on a bakery display, fantasy bakery, magical pastries, (green-colored baked goods:1.3)"

"raccoon sculptures made of (solid gray stone:1.4), (stone carving:1.3), museum exhibit of animal statues"

"children's toys made of (shiny chrome metal:1.4), (polished steel:1.3), metallic toy collection"
```

## Implementation Steps

1. **Parse the original prompt** for object + unusual attribute pairs
2. **Identify the conflict**: What does the model expect vs. what the user wants?
3. **Apply emphasis syntax** (1.3-1.5 range) to the unusual attribute
4. **Inject material/color tokens** — use 3-4 variations of the attribute
5. **Build negative prompt** listing the model's default expectations
6. **Add contextual support** — place the object in a setting where the attribute makes sense
7. **Return both positive and negative prompts** as structured fields

## Example Transformations

### Example 1: Seven Green Croissants
**Original**: "seven green croissants"

**Restructured Positive**:
```
"exactly (7:1.2) croissants, (bright green croissants:1.45), (vibrant lime green pastries:1.4), (emerald green baked goods:1.35), fantasy bakery display, magical green-colored croissants, studio photograph, centered composition, white background"
```

**Negative**:
```
"golden brown, tan, beige, normal croissant color, typical pastry color, brown pastry, yellow croissants, realistic croissant color, 8 croissants, more than 7, fewer than 7"
```

### Example 2: Two Stone Raccoons
**Original**: "two stone raccoons"

**Restructured Positive**:
```
"exactly (2:1.2) raccoons, raccoons made of (solid gray stone:1.45), (stone sculpture:1.4), (carved stone texture:1.35), stone animal statues, museum quality stone carving, detailed stone surface, granite material, photorealistic stone texture"
```

**Negative**:
```
"furry, soft fur, natural raccoon, living animal, brown fur, realistic fur texture, fluffy, real animal, mammal fur, 1 raccoon, 3 raccoons, more than 2, fewer than 2"
```

### Example 3: Two Metal Toys
**Original**: "two metal toys"

**Restructured Positive**:
```
"exactly (2:1.2) toys, toys made of (shiny chrome metal:1.45), (polished steel surface:1.4), (metallic finish:1.35), metal toy collection, reflective metal material, brushed aluminum toys, industrial metal texture, studio lighting on metal"
```

**Negative**:
```
"plastic, wood, soft material, fabric, rubber, painted surface, matte finish, non-metallic, toy plastic, wooden toys, 1 toy, 3 toys, more than 2, fewer than 2"
```

## Parameter Recommendations

When using this skill, also adjust sampler settings:
- **CFG Scale**: 8-10 (higher guidance helps override model priors)
- **Steps**: 35-50 (more iterations to resolve the attribute conflict)
- **Sampler**: DPM++ 2M Karras or Euler A (good at respecting weighted tokens)

If the unusual attribute still doesn't appear after first generation:
1. Increase emphasis weights to 1.5-1.6
2. Add the attribute to the negative prompt's inverse (e.g., negative: "not green, lacking green color")
3. Try adding "award-winning product photography" to positive prompt for better attention to details

## Output Format

Return a JSON object:
```json
{
  "positive_prompt": "<restructured prompt with emphasis and material tokens>",
  "negative_prompt": "<model's default expectations to suppress>",
  "cfg_scale": 8-10,
  "steps": 35-50,
  "rationale": "Applied unusual-attributes restructuring for [attribute] on [object]"
}
```

## Testing
This skill should be validated on:
- Color overrides: green croissants, purple trees, blue oranges
- Material overrides: stone animals, glass furniture, metal food
- Texture overrides: smooth rocks, fuzzy metal, transparent wood
- Count + attribute: "seven green croissants", "three glass apples"

Success = unusual attribute clearly visible in >80% of generations.