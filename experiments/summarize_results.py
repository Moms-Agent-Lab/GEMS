#!/usr/bin/env python3
import json, sys
sys.stdout.reconfigure(line_buffering=True)

with open("/workspace/benchmark_claw_20/results.json") as f:
    results = sorted(json.load(f), key=lambda r: r["idx"])

header = f"{'#':>3}  {'Prompt':<42}  {'Base':>8}  {'Best':>8}  {'Delta':>7}  {'Time':>6}  {'Nodes':>5}"
sys.stderr.write(header + "\n")
sys.stderr.write("-" * 95 + "\n")

base_scores, best_scores = [], []
improved = 0
for r in results:
    b = r.get("baseline_score", 0)
    s = r["best_score"]
    d = s - b
    t = r.get("elapsed_s", 0)
    tag = "UP" if d > 0.01 else ("ERR" if r.get("error") else " ")
    base_scores.append(b)
    best_scores.append(s)
    if d > 0.01:
        improved += 1
    line = f"{r['idx']:>3}  {r['prompt'][:40]:<42}  {b:8.3f}  {s:8.3f}  {d:+7.3f}  {t:5.0f}s  {r.get('node_count',0):>5}  {tag}"
    sys.stderr.write(line + "\n")

sys.stderr.write("-" * 95 + "\n")
mb = sum(base_scores) / len(base_scores)
ms = sum(best_scores) / len(best_scores)
md = ms - mb
sys.stderr.write(f"     {'MEAN':<42}  {mb:8.3f}  {ms:8.3f}  {md:+7.3f}\n\n")

valid = [r for r in results if not r.get("error")]
vb = sum(r.get("baseline_score", 0) for r in valid) / len(valid)
vs = sum(r["best_score"] for r in valid) / len(valid)
sys.stderr.write(f"Valid prompts (excl. API error): {len(valid)}/20\n")
sys.stderr.write(f"Mean baseline: {vb:.3f}  Mean best: {vs:.3f}  Delta = +{vs-vb:.3f} (+{(vs-vb)/max(vb,0.001)*100:.1f}%)\n")
sys.stderr.write(f"Improved: {improved}/{len(valid)} prompts\n")
sys.stderr.write(f"Perfect (best >= 0.85): {sum(1 for r in valid if r['best_score'] >= 0.85)}/{len(valid)}\n")
