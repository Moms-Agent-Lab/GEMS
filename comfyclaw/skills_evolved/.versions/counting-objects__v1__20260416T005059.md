---
name: counting-objects
description: >-
  Restructure prompts to enforce precise object counts, especially for quantities greater than two where diffusion models systematically fail due to numerical tokenization limits.
license: MIT
metadata:
  cluster: "counting_multiple_objects"
  origin: "self-evolve"
---

# Counting Objects Skill

## When to Use
Trigger when:
- User specifies exact counts: "three cats", "five apples", "seven croissants"
- Numbers ≥ 3 (models handle "two" better than higher counts)
- Verifier reports wrong object count or missing objects
- fix_strategy contains "fix_count" or "add_counting"

## Why Models Fail at Counting
- Tokenizers break numbers into subword units, losing numeric meaning
- Attention diffuses across repeated objects, making exact counts probabilistic
- No explicit counting mechanism in U-Net architecture

## Prompt Restructuring Rules

### 1. Repetition with Enumeration
Replace: "four rabbits"
With: "rabbit, rabbit, rabbit, rabbit, exactly four rabbits in total"

Replace: "seven green croissants"
With: "green croissant, green croissant, green croissant, green croissant, green croissant, green croissant, green croissant, precisely seven green croissants, 7 croissants"

### 2. Emphasis Syntax
Wrap count in parentheses with weight:
"(exactly four:1.4) brown monkeys, (4 monkeys:1.3)"

### 3. Negative Prompts for Wrong Counts
For "four rabbits":
Positive: "(exactly 4:1.3) rabbits, four rabbits"
Negative: "three rabbits, 3 rabbits, five rabbits, 5 rabbits, two rabbits, six rabbits"

### 4. Spatial Distribution Hints
For larger counts, add layout cues:
"seven croissants arranged in a row"
"four monkeys, two in front and two in back"
"six cars parked in two rows of three"

### 5. Combine with Regional-Control
For counts ≥4, consider regional prompting:
- Divide image into zones
- Assign specific objects to each zone
- Example: "four rabbits" → left region: "two rabbits", right region: "two rabbits"

## Node-Level Actions
1. Rewrite CLIPTextEncode positive prompt using repetition + emphasis
2. Add count-specific negative prompts to negative CLIPTextEncode
3. If count ≥5, recommend regional-control skill for zoned generation
4. Increase CFG slightly (+0.5 to +1.0) to strengthen prompt adherence
5. Consider seed variation if first attempt miscounts

## Example Transformations

Input: "four brown monkeys"
Output positive: "brown monkey, brown monkey, brown monkey, brown monkey, (exactly four:1.4) brown monkeys, (4 monkeys:1.3), four primates"
Output negative: "three monkeys, 3 monkeys, five monkeys, 5 monkeys, two monkeys, six monkeys"

Input: "six cars and a kangaroo"
Output positive: "car, car, car, car, car, car, (exactly six:1.4) cars, (6 cars:1.3), one kangaroo, (1 kangaroo:1.3)"
Output negative: "five cars, seven cars, 5 cars, 7 cars, four cars, two kangaroos, multiple kangaroos"

## Limitations
Counts above 10 remain unreliable even with these techniques. For such cases, recommend regional-control or inpainting workflows.