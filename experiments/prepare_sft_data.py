#!/usr/bin/env python3
"""
Post-process raw SFT traces into training-ready JSONL.

Reads sft_traces_all.jsonl (or per-prompt sft_traces.jsonl files) and produces
a cleaned JSONL where:

  1. Skill content from read_skill tool calls is inlined into the system prompt
     (so the model learns the *knowledge*, not the act of fetching it).
  2. The read_skill tool call + tool result turn pair is removed from the
     conversation, since the content is now in-context.
  3. The read_skill tool is removed from any tool definitions if present.
  4. Optionally filters traces by minimum verifier score.
  5. Outputs in OpenAI chat-completion fine-tuning format.

Usage:
    python prepare_sft_data.py --input  ../benchmark_qwen_10/sft_traces_all.jsonl \
                               --output ../benchmark_qwen_10/sft_training.jsonl \
                               --min-score 0.3

    # Or process individual prompt folders:
    python prepare_sft_data.py --input-dir ../benchmark_qwen_detailed \
                               --output    ../benchmark_qwen_10/sft_training.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _extract_skill_reads(messages: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Remove read_skill call/result pairs and return (cleaned_messages, skills_dict).

    skills_dict maps skill_name -> full skill body text.
    """
    skills: dict[str, str] = {}

    # First pass: identify tool_call_ids that correspond to read_skill calls
    read_skill_tc_ids: dict[str, str] = {}  # tool_call_id -> skill_name
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            if fn.get("name") == "read_skill":
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                skill_name = args.get("skill_name", "unknown")
                read_skill_tc_ids[tc["id"]] = skill_name

    # Second pass: collect skill bodies from tool result messages
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") in read_skill_tc_ids:
            skill_name = read_skill_tc_ids[msg["tool_call_id"]]
            skills[skill_name] = msg.get("content", "")

    # Third pass: rebuild messages without read_skill calls and their results
    cleaned: list[dict] = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") in read_skill_tc_ids:
            continue

        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            filtered_tcs = [
                tc for tc in msg["tool_calls"]
                if tc["id"] not in read_skill_tc_ids
            ]
            if not filtered_tcs and not msg.get("content"):
                # Assistant message had only read_skill calls and no text — skip entirely
                continue
            new_msg = dict(msg)
            if filtered_tcs:
                new_msg["tool_calls"] = filtered_tcs
            else:
                new_msg.pop("tool_calls", None)
            cleaned.append(new_msg)
        else:
            cleaned.append(msg)

    return cleaned, skills


def _build_skills_section(skills: dict[str, str]) -> str:
    """Format inlined skill content for injection into the system prompt."""
    if not skills:
        return ""
    parts = ["\n\n## Loaded Skills (reference material)\n"]
    for name, body in sorted(skills.items()):
        parts.append(f"### Skill: {name}\n{body}\n")
    return "\n".join(parts)


def _inline_skills(messages: list[dict]) -> list[dict]:
    """Move read_skill content into the system prompt and remove the tool calls."""
    cleaned, skills = _extract_skill_reads(messages)
    if not skills:
        return cleaned

    skills_section = _build_skills_section(skills)

    # Inject into the system prompt (first message)
    if cleaned and cleaned[0].get("role") == "system":
        cleaned[0] = dict(cleaned[0])
        cleaned[0]["content"] = cleaned[0]["content"] + skills_section
    else:
        cleaned.insert(0, {"role": "system", "content": skills_section})

    return cleaned


def _remove_read_skill_from_tools(tools: list[dict]) -> list[dict]:
    """Filter out the read_skill tool definition."""
    return [t for t in tools if t.get("function", {}).get("name") != "read_skill"]


def process_trace(trace: dict, min_score: float | None = None) -> dict | None:
    """Process a single raw trace into SFT training format.

    Returns None if the trace should be filtered out.
    """
    score = trace.get("verifier_score")
    if min_score is not None and (score is None or score < min_score):
        return None

    messages = trace.get("messages", [])
    if not messages:
        return None

    inlined = _inline_skills(messages)

    record: dict = {
        "messages": inlined,
    }

    # Preserve metadata for filtering / analysis downstream
    meta: dict = {}
    for key in ("prompt", "iteration", "type", "model", "token_usage",
                "verifier_score", "passed", "failed"):
        if key in trace:
            meta[key] = trace[key]
    if meta:
        record["metadata"] = meta

    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process raw SFT traces into training-ready JSONL"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str,
                       help="Path to sft_traces_all.jsonl (aggregated file)")
    group.add_argument("--input-dir", type=str,
                       help="Path to detailed results directory (scans for sft_traces.jsonl per prompt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path for training data")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Discard traces with verifier_score below this threshold")
    parser.add_argument("--include-repairs", action="store_true", default=False,
                        help="Include repair conversation traces (default: evolution only)")
    parser.add_argument("--include-workflows", action="store_true", default=False,
                        help="Keep workflow_before/workflow_after in output (large, usually not needed for SFT)")
    args = parser.parse_args()

    # Collect input lines
    raw_lines: list[str] = []
    if args.input:
        p = Path(args.input)
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)
        raw_lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"Error: {input_dir} is not a directory", file=sys.stderr)
            sys.exit(1)
        for trace_file in sorted(input_dir.rglob("sft_traces.jsonl")):
            raw_lines.extend(
                ln for ln in trace_file.read_text(encoding="utf-8").splitlines() if ln.strip()
            )

    print(f"Read {len(raw_lines)} raw traces")

    kept, skipped_score, skipped_repair, skipped_empty = 0, 0, 0, 0
    total_skills_inlined = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for line in raw_lines:
            trace = json.loads(line)

            # Filter repairs unless requested
            if not args.include_repairs and trace.get("type", "").startswith(("submission_repair", "execution_repair")):
                skipped_repair += 1
                continue

            # Strip workflow snapshots to save space unless requested
            if not args.include_workflows:
                trace.pop("workflow_before", None)
                trace.pop("workflow_after", None)

            record = process_trace(trace, min_score=args.min_score)
            if record is None:
                if trace.get("verifier_score") is not None and args.min_score and trace["verifier_score"] < args.min_score:
                    skipped_score += 1
                else:
                    skipped_empty += 1
                continue

            # Count inlined skills for stats
            msgs = record["messages"]
            if msgs and msgs[0].get("role") == "system":
                sys_content = msgs[0].get("content", "")
                total_skills_inlined += sys_content.count("### Skill: ")

            out_f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            kept += 1

    print(f"\nResults:")
    print(f"  Kept:              {kept}")
    print(f"  Skipped (score):   {skipped_score}")
    print(f"  Skipped (repairs): {skipped_repair}")
    print(f"  Skipped (empty):   {skipped_empty}")
    print(f"  Skills inlined:    {total_skills_inlined}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
