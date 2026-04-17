---
name: object-anchoring
description: >-
  Enforce precise object placement on surfaces and in specific spatial positions using depth control and regional masking. Use when objects must be positioned on windowsills, tables, shelves, or held/floating at specific locations, or when the verifier reports objects appearing in wrong positions, floating incorrectly, or lacking proper surface contact.
license: MIT
metadata:
  cluster: "spatial_positioning_failure"
  origin: "self-evolve"
---

# Object Anchoring Skill

## When to Use
Trigger when:
- Objects must sit ON specific surfaces (windowsill, table, shelf, desk)
- Items need to be held or float at precise positions (glasses floating, phone held by invisible hands)
- Verifier reports "wrong position", "floating incorrectly", "not on surface", "lack of contact"
- User describes placement with "on the", "sits on", "rests on", "placed on", "held by"
- Multiple depth layers need strict separation (foreground object + background surface)

## Strategy
1. **Prompt restructuring for anchoring**:
   - Make surface explicit and prominent: "wooden windowsill in sharp focus" not just "windowsill"
   - Add contact language: "resting on", "supported by", "sitting firmly on"
   - Specify viewpoint to clarify depth: "viewed from front", "side angle showing contact"
   - For floating objects, describe the negative space: "empty space where invisible figure stands, glasses floating at head height, phone floating at chest height"

2. **Regional prompt separation**:
   - Use regional-control skill to isolate the support surface (windowsill, table) in one region with high weight
   - Place the object (plant, glasses) in an overlapping region with explicit position relative to surface
   - This prevents the model from "forgetting" one element while rendering the other

3. **ControlNet depth for structure** (when available):
   - If controlnet-control skill can be combined: use depth preprocessing to lock the surface plane at correct z-distance
   - For windowsill scenes: depth map ensures window recess and sill have correct 3D relationship
   - For floating objects: depth map can enforce consistent height/distance from viewer

4. **Negative prompts for physical correctness**:
   - Add: "floating in air, levitating, no support, disconnected, hovering" to negative when object MUST be on surface
   - For invisible holder scenes, invert: add "visible hands, solid figure, person" to negative

## Example Transformations

**Before**: "a potted plant with delicate small flowers sits on a wooden windowsill"

**After**: "a ceramic potted plant with vibrant purple flowers resting firmly on a solid wooden windowsill, the plant's base making clear contact with the horizontal windowsill surface, viewed from a slight angle showing the depth of the window recess and the sill's thickness, sharp focus on the point of contact between pot and wood"

**Before**: "horn-rimmed glasses floating in mid-air, pearl necklace below, smartphone held by unseen figure"

**After**: "empty space occupying the center frame where an invisible figure stands, horn-rimmed glasses suspended at precise head height (5.5 feet from ground), pearl bead necklace draped at neck height below the glasses, smartphone floating at chest level as if held by invisible hands in operating position, consistent vertical alignment of all three objects suggesting a standing human form, negative space clearly defined"

## Node-Level Actions
- If regional-control is available: call it to separate surface and object into distinct conditioned regions
- If controlnet-control is available: recommend depth ControlNet with preprocessor to enforce surface geometry
- Always update the positive prompt with contact/anchoring language
- Always add anti-floating terms to negative prompt (unless floating is intentional)

## Coordination
- Run BEFORE high-quality (this establishes structure, quality enhances it)
- Run AFTER spatial (spatial handles relationships between objects, this handles object-to-surface physics)
- Can combine with controlnet-control for maximum positional accuracy
- Can combine with regional-control when multiple objects need individual anchoring