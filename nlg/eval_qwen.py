#!/usr/bin/env python3
"""Scores a Qwen explainer on held-out teacher positions (stratified per situation): first-try pass rate of
the checks, token-F1 against the teacher, samples.md, and with --retries the deployed generate -> check -> retry
(`explain_checked`)."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import collections
import json
import random
import re
import time
from pathlib import Path

import torch
from peft import PeftModel

from nlg.consistency_checks import check_explanation, strip_tautological_clauses
from nlg.prompt import build_llm_prompt, check_style
from nlg.qwen_model import DEFAULT_MODEL_ID, load_llm
from nlg.train_qwen import load_pairs
from nlg.explanation_inputs import bid_category


def _tokens(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())


def token_f1(a: str, b: str) -> float:
    ta, tb = collections.Counter(_tokens(a)), collections.Counter(_tokens(b))
    common = sum((ta & tb).values())
    if not common:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)


def _generate(model, tok, prompt: str, max_new_tokens: int, sample: bool) -> str:
    enc = tok(prompt, return_tensors="pt").to("cuda:0")
    kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
    with torch.no_grad():
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **kw)
    return strip_tautological_clauses(tok.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())


def _problems(frame, text: str):
    """(consistency warnings, style problem or None) for one output."""
    if not text:
        return ["empty output"], "empty output"
    return check_explanation(frame, text), check_style(text)


def explain_checked(model, tok, frame, selected, retries: int = 3, max_new_tokens: int = 160) -> dict:
    """Deployed explainer: greedy, then up to `retries` sampled attempts if
    the output fails nlg.consistency_checks / the style check; the first passing
    output is kept (else the greedy one)."""
    prompt = tok.apply_chat_template(build_llm_prompt(frame, selected), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    greedy = _generate(model, tok, prompt, max_new_tokens, sample=False)
    warnings, style = _problems(frame, greedy)
    out = {"greedy": greedy, "warnings": warnings, "style": style, "final": greedy,
           "final_ok": not warnings and style is None, "attempts": 1}
    while not out["final_ok"] and out["attempts"] <= retries:
        cand = _generate(model, tok, prompt, max_new_tokens, sample=True)
        out["attempts"] += 1
        w, st = _problems(frame, cand)
        if not w and st is None:
            out.update(final=cand, final_ok=True)
    if not out["final_ok"]:
        fixed = repair(frame, out["final"])
        if fixed:
            out.update(final=fixed, final_ok=True, repaired=True)
    return out


# a false clause the explainer tends to repeat through every retry, which can be cut out without touching
# the rest: "..., and the hand is semi-balanced (1-2-5-5)." on a hand with a singleton
_SHAPE_CLAUSE_RE = re.compile(r"(?:,? and|;|,) the hand is (?:semi-)?balanced \(\d-\d-\d-\d\)")


def repair(frame, text: str):
    """The text with a known false clause cut out or replaced by the teacher's own
    fallback, if that makes it pass the checks; else None. Two clauses are handled:
    the shape claim (cut), and the role minimum written for a hand above it ("With
    10 HCP South has no more than a minimum response (at most 9), already shown by
    1D" -> "South has 10 HCP and has already described the hand with 1D", what the
    teacher writes there). Both are errors the explainer can repeat through all retries."""
    for fix in (lambda s: _SHAPE_CLAUSE_RE.sub("", s), _fix_role_minimum):
        fixed = fix(text)
        if fixed == text:
            continue
        w, st = _problems(frame, fixed)
        if not w and st is None:
            return fixed
    return None


_ROLE_MIN_RE = re.compile(r"\b[Ww]ith (\d+) HCP (\w+) has no more than a minimum \w+ \(at most \d+\)(?:, already shown by (\S+?))?(?=[.;,]|$)")


def _fix_role_minimum(text: str) -> str:
    def sub(m):
        n, seat, call = m.group(1), m.group(2), m.group(3)
        return f"{seat} has {n} HCP and has already described the hand with {call}" if call else f"{seat} has {n} HCP"
    return _ROLE_MIN_RE.sub(sub, text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (train_qwen's adapter/ or adapter-best/)")
    ap.add_argument("--no_adapter", action="store_true", help="evaluate the base model instead")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_category", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retries", type=int, default=0, help="sampled retries after a failed greedy output")
    args = ap.parse_args()
    if not args.no_adapter and not args.adapter:
        ap.error("--adapter is required unless --no_adapter")

    by_cat = collections.defaultdict(list)
    for frame, selected, text in load_pairs(args.labels, args.frames):
        by_cat[bid_category(frame)].append((frame, selected, text))
    rng = random.Random(args.seed)
    items = [(c, x) for c in sorted(by_cat) for x in rng.sample(by_cat[c], min(args.per_category, len(by_cat[c])))]

    llm = load_llm(args.model_id)
    model = llm.model if args.no_adapter else PeftModel.from_pretrained(llm.model, args.adapter)
    model.eval()
    tok = llm.tokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, (cat, (frame, selected, teacher)) in enumerate(items):
        r = explain_checked(model, tok, frame, selected, retries=args.retries, max_new_tokens=args.max_new_tokens)
        rows.append({"category": cat, "deal_id": frame["deal_id"], "step": frame["step"], "bid": frame["bid"],
                     "auction": " ".join(b["bid"] for b in frame["auction_before"]), "hand": frame["hand_str"],
                     "teacher": teacher, "qwen": r["greedy"], "consistent": not r["warnings"],
                     "style_ok": r["style"] is None,
                     "warnings": r["warnings"] + ([r["style"]] if r["style"] else []),
                     "qwen_final": r["final"], "passes_with_retries": r["final_ok"], "attempts": r["attempts"],
                     "token_f1": token_f1(r["greedy"], teacher)})
        if i % 20 == 0:
            ok = sum(r["consistent"] and r["style_ok"] for r in rows)
            print(f"[eval_qwen] {i + 1}/{len(items)}  passing {ok}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)

    per_cat = {}
    for c in sorted(by_cat):
        rs = [r for r in rows if r["category"] == c]
        if rs:
            per_cat[c] = {"n": len(rs), "pass": sum(r["consistent"] and r["style_ok"] for r in rs) / len(rs),
                          "token_f1": sum(r["token_f1"] for r in rs) / len(rs)}
    n = len(rows)
    summary = {"adapter": None if args.no_adapter else args.adapter, "n": n,
               "pass_rate": sum(r["consistent"] and r["style_ok"] for r in rows) / n,
               "consistent_rate": sum(r["consistent"] for r in rows) / n,
               "style_rate": sum(r["style_ok"] for r in rows) / n,
               "token_f1": sum(r["token_f1"] for r in rows) / n,
               "retries": args.retries,
               "pass_rate_with_retries": sum(r["passes_with_retries"] for r in rows) / n,
               "mean_attempts": sum(r["attempts"] for r in rows) / n,
               "worst_situations": sorted(per_cat.items(), key=lambda kv: kv[1]["pass"])[:8],
               "per_category": per_cat}
    (out / "eval.json").write_text(json.dumps(summary, indent=2))
    with open(out / "generations.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = [f"# Qwen explainer vs teacher ({n} held-out decisions)", "",
             f"pass (consistent + style): {summary['pass_rate']:.1%}  |  consistent: {summary['consistent_rate']:.1%}"
             f"  |  style: {summary['style_rate']:.1%}  |  token F1 vs teacher: {summary['token_f1']:.2f}", ""]
    for c in sorted(per_cat):
        lines.append(f"## {c} (pass {per_cat[c]['pass']:.0%})")
        for r in [r for r in rows if r["category"] == c]:
            flag = "" if r["consistent"] and r["style_ok"] else f"  \n  **FAIL**: {'; '.join(map(str, r['warnings']))[:300]}"
            if r["qwen_final"] != r["qwen"]:
                flag += f"  \n  after {r['attempts'] - 1} retr{'y' if r['attempts'] == 2 else 'ies'} " \
                        f"({'passes' if r['passes_with_retries'] else 'still failing'}): {r['qwen_final']}"
            lines.append(f"- `{r['auction'] or '(first call)'} -> {r['bid']}` | {r['hand']}  \n"
                         f"  teacher: {r['teacher']}  \n  qwen: {r['qwen']}{flag}")
        lines.append("")
    (out / "samples.md").write_text("\n".join(lines) + "\n")
    print(f"[eval_qwen] with up to {args.retries} retries: pass {summary['pass_rate_with_retries']:.1%} "
          f"(mean {summary['mean_attempts']:.2f} attempts)", flush=True)
    print(f"[eval_qwen] done: pass {summary['pass_rate']:.1%}, consistent {summary['consistent_rate']:.1%}, "
          f"style {summary['style_rate']:.1%}, token F1 {summary['token_f1']:.2f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
