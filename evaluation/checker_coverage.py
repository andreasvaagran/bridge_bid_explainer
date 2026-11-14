#!/usr/bin/env python3
"""What the consistency checker verifies: pass rates with and without the strength checks, false claims
per claim type, and the share of numbers about the hand that some check verifies, for the teacher, the fine-tuned
and the base explainer."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import collections
import json
from pathlib import Path

from nlg import consistency_checks as C
from nlg.prompt import check_style

ROOT = Path(__file__).resolve().parent.parent


def _old_checks(frame, text):
    """check_explanation without the strength/arithmetic check (check_strength_claims)."""
    saved = C.check_strength_claims
    C.check_strength_claims = lambda f, e: None
    try:
        return C.check_explanation(frame, text)
    finally:
        C.check_strength_claims = saved


def score(items):
    """items: iterable of (frame, text) -> summary dict."""
    n = old_pass = new_pass = style = 0
    found, false = collections.Counter(), collections.Counter()
    cov = collections.Counter()
    examples = collections.defaultdict(list)
    for frame, text in items:
        n += 1
        st = not check_style(text)
        style += st
        old_pass += (not _old_checks(frame, text)) and st
        new_ok = not C.check_explanation(frame, text)
        new_pass += new_ok and st
        for c in C.claim_report(frame, text):
            found[c["type"]] += 1
            if not c["ok"]:
                false[c["type"]] += 1
                if len(examples[c["type"]]) < 3:
                    examples[c["type"]].append({"claim": c["text"], "why": c["why"], "text": text,
                                                "bid": frame.get("bid"), "hand": frame.get("hand_str")})
        for k, v in C.number_coverage(frame, text).items():
            cov[k] += v
    return {"n": n, "pass_old_checks": old_pass / max(n, 1), "pass_new_checks": new_pass / max(n, 1),
            "style": style / max(n, 1),
            "claims": {k: {"found": found[k], "false": false[k]} for k in sorted(found)},
            "numbers": dict(cov), "number_coverage": cov["verified"] / max(cov["numbers"], 1),
            "false_examples": dict(examples)}


def teacher_items(d: Path):
    for split in ("train", "val", "test"):
        fp, lp = d / f"{split}_frames.jsonl", d / f"{split}_labels.jsonl"
        if not fp.exists():
            continue
        frames = {(f["deal_id"], f["step"]): f for f in map(json.loads, fp.open())}
        for lab in map(json.loads, lp.open()):
            fr = frames.get((lab["deal_id"], lab["step"]))
            if fr is not None:
                yield fr, lab["explanation"]


def generation_items(gen: Path, frames_path: Path, key="qwen"):
    frames = {(f["deal_id"], f["step"]): f for f in map(json.loads, frames_path.open())}
    for g in map(json.loads, gen.open()):
        fr = frames.get((g["deal_id"], g["step"]))
        if fr is not None and g.get(key):
            yield fr, g[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher_dirs", nargs="*", default=["runs/explainer_data/teacher", "runs/explainer_data/teacher_selfplay"])
    ap.add_argument("--pre_fix_teacher_dirs", nargs="*", default=[],
                    help="optional: teacher datasets of an earlier teacher version, scored the same way")
    ap.add_argument("--qwen_dir", default="runs/evaluation/qwen")
    args = ap.parse_args()
    res = {}
    for tag, dirs in (("teacher (current)", args.teacher_dirs), ("teacher (earlier version)",
                                                                  args.pre_fix_teacher_dirs)):
        dirs = [ROOT / d for d in dirs if (ROOT / d).exists()]
        if dirs:
            res[tag] = score(x for d in dirs for x in teacher_items(d))
            print(f"[checker] {tag}: {res[tag]['n']:,}  old {res[tag]['pass_old_checks']:.4f}  "
                  f"new {res[tag]['pass_new_checks']:.4f}  coverage {res[tag]['number_coverage']:.4f}", flush=True)
    q = ROOT / args.qwen_dir
    frames = q / "testsplit_data" / "test_frames.jsonl"
    for tag, sub in (("Qwen3-8B + adapter (first attempt)", "eval_adapter_testsplit"),
                     ("Qwen3-8B, no fine-tuning (first attempt)", "eval_base_testsplit")):
        gen = q / sub / "generations.jsonl"
        if gen.exists() and frames.exists():
            res[tag] = score(generation_items(gen, frames))
            print(f"[checker] {tag}: {res[tag]['n']:,}  old {res[tag]['pass_old_checks']:.4f}  "
                  f"new {res[tag]['pass_new_checks']:.4f}  coverage {res[tag]['number_coverage']:.4f}", flush=True)
    if frames.exists():
        gen = q / "eval_adapter_testsplit" / "generations.jsonl"
        res["teacher on the same 550 positions (as stored for the eval)"] = score(
            generation_items(gen, frames, key="teacher"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
