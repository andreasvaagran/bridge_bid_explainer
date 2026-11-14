#!/usr/bin/env python3
"""Compares three follow-up systems on held-out, indirect and hand-written questions: the keyword parser
with templates, base Qwen, and the fine-tuned explainer."""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the bidding model on the CPU, Qwen on the GPU

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from nlg.followup_questions import (PARSE_SYSTEM, Decision, _positions, answer_messages,
                          check_answer, facts_for, norm_query, parse_messages, parse_rule, query_json, say_call,
                          say_card, say_seat, teacher_answer, SUIT_NAME, SUIT_SING)
from explain.counterfactual import softmax_masked
from nlg.eval_qwen import token_f1
from nlg.explanation_frames import BIDS

REPO = Path(__file__).resolve().parent.parent

# question templates written separately from followup.QUESTIONS (loose, colloquial);
# placeholders: {alt} {chosen} {who} {call} {out} {in} {suit} {suit_s}
HANDWRITTEN = {
    "why_not": ["honestly I'd have bid {alt} here. what am I missing?", "{alt} seems like the obvious bid to me",
                "is {alt} completely off the table?", "couldn't we just bid {alt}?", "what's the case against {alt}?",
                "Why wasn't {alt} good enough?", "{alt} would have been my choice. Why is it wrong?",
                "Doesn't this hand want to bid {alt}?", "I don't get why you avoided {alt}",
                "Is there a reason you skipped {alt}?"],
    "change_any": ["Which card in my hand is the deciding one?", "How close was this to being a different bid?",
                   "What would it take for you to bid something else?", "Would a slightly different hand change the call?"],
    "change_stronger": ["Say I had an extra ace, any difference?", "If my hand were a king stronger, what would you do?",
                        "How about with a few more high-card points?"],
    "change_weaker": ["And if I were a jack weaker?", "With a couple of points less, would you still bid {chosen}?"],
    "change_longer": ["What if my {suit} were a card longer?", "Imagine one extra {suit_s} in my hand, what then?"],
    "change_shorter": ["With a {suit_s} fewer would it change?", "And if I was one {suit_s} short?"],
    "swap": ["Say my {out} was the {in}, what then?", "Would you still bid {chosen} if the {out} were the {in}?",
             "Trade the {out} for the {in}: new bid?", "if {out} became {in}?"],
    "meaning": ["{who} bid {call} earlier, what was that about?", "I didn't follow {who}'s {call}",
                "What's {who}'s {call} telling us?", "Is {who}'s {call} natural?",
                "How strong is {who} for the {call} bid?"],
    "trust": ["How likely is it that this call is a mistake?", "Should I double-check this bid with someone?",
              "How much do you believe in {chosen} here?", "Is the agent guessing here?",
              "On a scale of sure to unsure, where is this?"],
    "explain": ["What's the idea behind {chosen}?", "Talk me through why {chosen} is right", "So why {chosen} exactly?"],
    "other": ["Who's going to be declarer?", "What should I lead against this?", "Do we make game?", "What time is it?",
              "Can you recommend a bridge book?"],
}


def handwritten_item(d: Decision, rng: random.Random):
    frame = d.frame
    prior = frame.get("auction_before") or []
    you = (frame.get("table") or {}).get("you")
    others = [b for b in prior if b["seat"] != you and b["bid"] != "P"]
    kinds = [k for k in HANDWRITTEN if k != "meaning" or others]
    k = rng.choice(kinds)
    tpl = rng.choice(HANDWRITTEN[k])
    chosen = say_call(frame["bid"], rng)
    if k == "why_not":
        order = [int(i) for i in np.argsort(-d.probs) if d.legal[i] and int(i) != d.action]
        alt = BIDS[order[0]] if rng.random() < 0.6 or len(order) < 3 else BIDS[rng.choice(order[1:4])]
        return tpl.format(alt=say_call(alt, rng), chosen=chosen), {"type": "why_not", "call": alt}
    if k.startswith("change_"):
        kind = k.split("_")[1]
        q = {"type": "change", "kind": kind}
        if kind in ("longer", "shorter"):
            s = rng.choice("SHDC")
            q["suit"] = s
            return tpl.format(suit=SUIT_NAME[s], suit_s=SUIT_SING[s], chosen=chosen), q
        return tpl.format(chosen=chosen), q
    if k == "swap":
        held = [s + r for s, rs in d.hand.cards.items() for r in rs]
        from explain.counterfactual import FULL_DECK
        o = rng.choice(held)
        i = rng.choice([c for c in FULL_DECK if c not in held and (c[0] == o[0] or c[1] in "AKQ")])
        t = tpl.format(out=say_card(o, rng), **{"in": say_card(i, rng)}, chosen=chosen)
        return t.replace("my the ", "my ").replace("the the ", "the "), {"type": "swap", "out": o, "in": i}
    if k == "meaning":
        b = rng.choice(others)
        return (tpl.format(who=say_seat(frame, b["seat"], rng), call=say_call(b["bid"], rng)),
                {"type": "meaning", "seat": b["seat"], "call": b["bid"]})
    return tpl.format(chosen=chosen), {"type": k}


# ---------------------------------------------------------------------------

PARSE_SHOTS = [
    ("Why not 3NT?", {"type": "why_not", "call": "3NT"}),
    ("What if I had the king of hearts instead of the 4 of clubs?", {"type": "swap", "out": "C4", "in": "HK"}),
    ("What would make you bid differently?", {"type": "change", "kind": "any"}),
    ("What does partner's 2C mean?", {"type": "meaning", "seat": "N", "call": "2C"}),
    ("How sure are you?", {"type": "trust"}),
    ("What's the best opening lead?", {"type": "other"}),
]


class LLMRunner:
    def __init__(self, adapter: str | None):
        import torch  # noqa: F401
        from nlg.qwen_model import load_llm
        self.llm = load_llm()
        self.tok = self.llm.tokenizer
        self.model = self.llm.model
        self.adapter = None
        if adapter:
            self.use_adapter(adapter)

    def use_adapter(self, adapter: str):
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.llm.model, adapter)
        self.model.eval()
        self.adapter = adapter

    def gen(self, messages, max_new_tokens: int, base: bool = False) -> str:
        """Greedy generation; base=True runs the untrained model (the adapter switched off)."""
        import contextlib
        import torch
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = self.tok(text, return_tensors="pt").to("cuda:0")
        off = self.model.disable_adapter() if base and self.adapter else contextlib.nullcontext()
        with torch.no_grad(), off:
            ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def few_shot_parse(frame, question):
    """The base model's parse prompt: the schema plus six worked examples (seats as in a fixed example table)."""
    msgs = [{"role": "system", "content": PARSE_SYSTEM}]
    ex_frame = {"bid": "1NT", "table": {"you": "S", "partner": "N", "opponents": ["W", "E"]},
                "auction_before": [{"seat": "N", "bid": "2C"}, {"seat": "E", "bid": "P"}],
                "hand_str": "S:AQ4 H:K93 D:QJ5 C:K843", "hcp": 15, "vulnerability": "None", "dealer": "N"}
    for qtext, q in PARSE_SHOTS:
        m = parse_messages(ex_frame, qtext)
        msgs += [m[1], {"role": "assistant", "content": query_json(q)}]
    return msgs + [parse_messages(frame, question)[1]]


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return norm_query(json.loads(m.group(0)))
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REPO / "runs/explainer_data/followup/test.jsonl"))
    ap.add_argument("--adapter", default=str(REPO / "runs/qwen/adapter-best"))
    ap.add_argument("--model_path", default=str(REPO / "runs/sl/independent/params-best.pkl"))
    ap.add_argument("--prepared", default=str(REPO / "data/wbridge5/prepared/train"))
    ap.add_argument("--systems", nargs="+", default=["rule", "tuned", "base"])
    ap.add_argument("--base_every", type=int, default=2,
                    help="the untrained model answers every k-th question only (it is slow and its results are clear)")
    ap.add_argument("--n", type=int, default=400, help="generated test questions")
    ap.add_argument("--n_hand", type=int, default=150, help="hand-written questions")
    ap.add_argument("--out_dir", default=str(REPO / "runs/evaluation/followup"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import jax
    from training.checkpoint import load_checkpoint
    from nlg.explanation_frames import Enricher
    ck = load_checkpoint(args.model_path)
    en = Enricher(ck)
    fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))
    offsets, obs_bits, legal_bits = _positions(Path(args.prepared))

    def decision(frame):
        g, s = int(frame["deal_id"]), int(frame["step"])
        r = int(offsets[g]) + s
        o = np.unpackbits(obs_bits[r:r + 1], axis=1)[0, :480].astype(np.float32)
        lg = np.unpackbits(legal_bits[r:r + 1], axis=1)[0, :38].astype(bool)
        out = {k: np.asarray(v[0]) for k, v in fwd(o[None]).items()}
        probs = softmax_masked(np.asarray(out["policy_logits"], np.float64), lg)
        return Decision(frame, o, lg, probs, en.policy_logits_fn, np.asarray(out["concept_acts"]), en.W,
                        list(ck.hard_concepts), "SL independent")

    rows = [json.loads(l) for l in open(args.data)][:args.n]
    rng = random.Random(args.seed)
    items = [{"group": ("indirect" if r.get("indirect") else "train-phrasing" if r["phrasing"] == "train"
                        else "held-out phrasing"), "frame": r["frame"],
              "question": r["question"], "gold": norm_query(r["query"])} for r in rows]
    for r in rng.sample(rows, min(args.n_hand, len(rows))):
        d = decision(r["frame"])
        qtext, q = handwritten_item(d, rng)
        items.append({"group": "hand-written", "frame": r["frame"], "question": qtext, "gold": norm_query(q)})
    print(f"[followup_eval] {len(items)} questions: " + ", ".join(
        f"{g} {sum(1 for x in items if x['group'] == g)}" for g in ("train-phrasing", "held-out phrasing", "indirect",
                                                                      "hand-written")),
        flush=True)

    def run_answer(d, q, question, gen_answer):
        f = facts_for(d, q)
        if q["type"] in ("explain", "other"):
            text = teacher_answer(d.frame, q, f)  # fixed: the explanation itself / the out-of-scope reply
        else:
            text = gen_answer(d.frame, question, q, f)
        return f, text, check_answer(d.frame, q, f, text)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    runner = None
    for system in args.systems:
        t0 = time.time()
        if system == "rule":
            parse = lambda fr, qt: parse_rule(fr, qt)  # noqa: E731
            answer = lambda fr, qt, q, f: teacher_answer(fr, q, f)  # noqa: E731
        else:
            if runner is None:
                runner = LLMRunner(None)
            if system == "tuned" and runner.adapter is None:
                runner.use_adapter(args.adapter)
            if system == "base":
                parse = lambda fr, qt: parse_json(runner.gen(few_shot_parse(fr, qt), 60, base=True))  # noqa: E731
                answer = lambda fr, qt, q, f: runner.gen(answer_messages(fr, qt, q, f), 200, base=True)  # noqa: E731
            else:
                parse = lambda fr, qt: parse_json(runner.gen(parse_messages(fr, qt), 60))  # noqa: E731
                answer = lambda fr, qt, q, f: runner.gen(answer_messages(fr, qt, q, f), 200)  # noqa: E731
        recs = []
        for k, it in enumerate(items):
            if system == "base" and k % args.base_every:
                continue
            d = decision(it["frame"])
            try:
                pq = norm_query(parse(it["frame"], it["question"]))
            except Exception:
                pq = None
            ok_parse = pq == it["gold"]
            # the answer step alone, with the gold query
            f, text, probs = run_answer(d, it["gold"], it["question"], answer)
            teacher = teacher_answer(it["frame"], it["gold"], f)
            # end to end: the answer to what was PARSED
            if ok_parse:
                e2e_text, e2e_probs = text, probs
            elif pq is None:
                e2e_text, e2e_probs = "", ["no query"]
            else:
                try:
                    _, e2e_text, e2e_probs = run_answer(d, pq, it["question"], answer)
                except Exception as e:
                    e2e_text, e2e_probs = "", [f"error {e!r}"]
            recs.append({"i": k, "group": it["group"], "question": it["question"], "gold": it["gold"], "parsed": pq,
                         "parse_ok": ok_parse, "answer": text, "answer_ok": not probs, "answer_problems": probs,
                         "teacher": teacher, "token_f1": token_f1(text, teacher),
                         "e2e_answer": e2e_text, "e2e_ok": ok_parse and not e2e_probs,
                         "hand": it["frame"]["hand_str"], "bid": it["frame"]["bid"],
                         "auction": " ".join(b["bid"] for b in it["frame"]["auction_before"])})
            if k % 50 == 0:
                print(f"[followup_eval] {system} {k + 1}/{len(items)} parse {np.mean([r['parse_ok'] for r in recs]):.3f} "
                      f"answer {np.mean([r['answer_ok'] for r in recs]):.3f} ({time.time() - t0:.0f}s)", flush=True)
        summ = {}
        for g in ("all", "train-phrasing", "held-out phrasing", "indirect", "hand-written"):
            rs = [r for r in recs if g == "all" or r["group"] == g]
            if not rs:
                continue
            gen = [r for r in rs if r["gold"]["type"] not in ("explain", "other")]
            from evaluation.common import wilson
            summ[g] = {"n": len(rs), "parse_acc": float(np.mean([r["parse_ok"] for r in rs])),
                       "parse_ci95": list(wilson(sum(r["parse_ok"] for r in rs), len(rs))[1:]),
                       "e2e_ci95": list(wilson(sum(r["e2e_ok"] for r in rs), len(rs))[1:]),
                       "answer_pass": float(np.mean([r["answer_ok"] for r in gen])) if gen else None,
                       "answer_token_f1": float(np.mean([r["token_f1"] for r in gen])) if gen else None,
                       "e2e": float(np.mean([r["e2e_ok"] for r in rs]))}
        by_type = {}
        for t in ("why_not", "change", "swap", "meaning", "trust", "explain", "other"):
            rs = [r for r in recs if r["gold"]["type"] == t]
            if rs:
                by_type[t] = {"n": len(rs), "parse_acc": float(np.mean([r["parse_ok"] for r in rs])),
                              "answer_pass": float(np.mean([r["answer_ok"] for r in rs])),
                              "e2e": float(np.mean([r["e2e_ok"] for r in rs]))}
        results[system] = {"summary": summ, "by_type": by_type, "seconds": round(time.time() - t0)}
        with open(out / f"records_{system}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[followup_eval] {system}: " + json.dumps(summ), flush=True)
        (out / "followup_eval.json").write_text(json.dumps({"adapter": args.adapter, "questions": len(items),
                                                            "results": results}, indent=1))
    # samples for reading
    lines = ["# Follow-up questions: rule / base / tuned", ""]
    recs_by = {s: {r["i"]: r for r in map(json.loads, open(out / f"records_{s}.jsonl"))} for s in args.systems}
    for i in range(0, len(items), max(1, len(items) // 40)):
        it = items[i]
        auction = " ".join(b["bid"] for b in it["frame"]["auction_before"])
        lines.append(f"- **{it['group']}** `{auction or '(first call)'} -> "
                     f"{it['frame']['bid']}` | {it['frame']['hand_str']}  \n  Q: {it['question']}  \n"
                     f"  gold: `{query_json(it['gold'])}`")
        for s in args.systems:
            r = recs_by[s].get(i)
            if r is None:
                continue
            lines.append(f"  - {s}: parsed `{query_json(r['parsed']) if r['parsed'] else None}` "
                         f"({'ok' if r['parse_ok'] else 'WRONG'}) -> {r['e2e_answer'] or '(none)'}"
                         + ("" if r["e2e_ok"] else f"  **[{'; '.join(r['answer_problems'])[:160] if r['parse_ok'] else 'wrong query'}]**"))
        lines.append("")
    (out / "samples.md").write_text("\n".join(lines) + "\n")
    print(f"[followup_eval] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
