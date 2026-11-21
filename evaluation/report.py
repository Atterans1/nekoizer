"""Render results.md from results.json and apply the pre-registered set rule (>=75% pass, every target >=1 pass)."""
import json
import os
from collections import defaultdict

EV = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(EV, "results.json"), encoding="utf-8") as fh:
    data = json.load(fh)
rows = data["rows"]
judge = {}
jpath = os.path.join(EV, "judge.json")
if os.path.exists(jpath):
    with open(jpath, encoding="utf-8") as fh:
        judge = json.load(fh)


def F(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


lines = ["# Voice-conversion evaluation: results", "",
         "Pre-registered thresholds: preregistered_thresholds.json (sha256 " + data["thresholds_sha256"][:12] + "...), registered before any conversion.",
         "Defaults: shipped params (chunk 320 ms, cfm_steps 4, f0_preserve off, pitch 0, blend 1.0). Metrics on 16 kHz; similarity from resemblyzer (independent) and CAMPPlus (same model the system conditions on, NOT independent).",
         "", "## Per conversion", "",
         "| # | target | source | indep gain | CAMP conv-tgt | CAMP gain | F0 r | joint frames | energy r | dur ratio | result | failed checks |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if r["error"]:
        lines.append(f"| {r['n']} | {r['target']} | {r['source']} | ERROR | | | | | | | FAIL | {r['error'][:120]} |")
        continue
    s, p = r["similarity"], r["prosody"]
    failed = [k for k, v in r["checks"].items() if not v]
    lines.append(f"| {r['n']} | {r['target']} | {r['source']} ({r['source_gender']}) | {F(s['resemblyzer_gain'])} | {F(s['campplus_conv_tgt'])} | {F(s['campplus_gain'])} | {F(p['f0_logcorr'])} | {p['joint_voiced_frames']} | {F(p['energy_logcorr'])} | {F(p['duration_ratio'])} | {'PASS' if r['passed'] else 'FAIL'} | {', '.join(failed) or '-'} |")

n = len(rows)
passed = sum(1 for r in rows if r["passed"])
rate = passed / n if n else 0.0
per_target = defaultdict(lambda: [0, 0])
for r in rows:
    per_target[r["target"]][0] += 1
    per_target[r["target"]][1] += int(r["passed"])
set_pass = rate >= 0.75 and all(v[1] > 0 for v in per_target.values()) and n > 0
lines += ["", "## Summary", "",
          f"- Conversions: {n}; passed: {passed}; pass rate: {rate:.1%}",
          f"- Set rule (>=75% pass and no target with 0 passes): **{'PASS' if set_pass else 'FAIL'}**",
          f"- Skipped same-speaker pairs: {data.get('skipped_same_speaker', [])}", "",
          "### Per target", "", "| target | conversions | passed |", "|---|---|---|"]
for t, (tot, ok) in sorted(per_target.items()):
    lines.append(f"| {t} | {tot} | {ok} |")

if judge:
    lines += ["", "## Judge (scores measured evidence; does NOT listen to audio)", "",
              "| # | voice score | prosody score | listener accepts | disagrees with thresholds |", "|---|---|---|---|---|"]
    for r in rows:
        j = judge.get(str(r["n"]))
        if not j:
            continue
        lines.append(f"| {r['n']} | {j.get('voice', 'n/a')} | {j.get('prosody', 'n/a')} | {j.get('accept', 'n/a')} | {'YES' if j.get('disagree') else 'no'} |")

with open(os.path.join(EV, "results.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("wrote results.md", n, "conversions", passed, "passed", "set", "PASS" if set_pass else "FAIL")
