#!/usr/bin/env python3
"""Are the explanations faithful, correct and useful? On the model's own calls the deployed pipeline
runs as in the GUI. Measured: the stated reasons' share of the call's margin and the effect of removing them
(against random and top-k concepts), the check pass rate, and how well per-decision signals flag departures from
WBridge5. --medians reads saved records."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import collections
import json
import pickle
import random
import time
from pathlib import Path

import jax
import numpy as np

from evaluation.common import (BIDS, MODELS, SEATS, auroc, cluster_bootstrap, load_test_split, masked_argmax,
                                  sample_rows, selfplay, wilson)
from nlg.consistency_checks import check_explanation
from nlg.explanation_frames import Enricher, decision_frame
from nlg.prompt import check_style
from nlg.explanation_inputs import (_gt, bid_category, bid_meaning_concepts, deviation_notes, hand_dependent_concepts,
                           select_reasoned_concepts)
from nlg.teacher import explain
from training.checkpoint import load_checkpoint


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _p_bid(a, W, b, lg, bid):
    L = a @ W + b
    return _softmax(np.where(lg, L, -np.inf))[bid], int(np.argmax(np.where(lg, L, -np.inf)))


def light_category(pos, i, bid) -> str:
    before, seat, dl = pos.auction(i)
    t = {"you": seat, "partner": SEATS[(SEATS.index(seat) + 2) % 4]}
    return bid_category({"table": t, "auction_before": before, "bid": bid, "hand_str": "", "dealer": SEATS[dl]})


def analyse_decision(ck, en, fwd, pos, i, rng) -> dict:
    obs = pos.obs[i].astype(np.float32)
    lg = pos.legal[i]
    out = {k: np.asarray(v[0]) for k, v in fwd(obs[None]).items()}
    before, seat, dl = pos.auction(i)
    frame = decision_frame(en, obs, lg, out, int(pos.deal[i]), int(pos.step[i]), seat, SEATS[dl], dl,
                           pos.vul[i], before, counterfactuals=False)
    bid = BIDS.index(frame["bid"])
    rec = {"row": int(i), "deal": int(pos.deal[i]), "bid": frame["bid"], "category": bid_category(frame),
           "conf": float(_softmax(np.where(lg, np.asarray(out["policy_logits"], np.float64), -np.inf))[bid])}
    if pos.targets is not None:
        rec["wb5_bid"] = BIDS[int(pos.action[i])]
        rec["error"] = rec["wb5_bid"] != frame["bid"]
    try:
        text = explain(frame)
        rec["explained"] = True
    except Exception as e:  # counted, not fatal
        text, rec["explained"], rec["teacher_error"] = "", False, repr(e)[:200]
    if rec["explained"]:
        problems = check_explanation(frame, text)
        style = check_style(text)
        rec["consistent"] = not problems
        rec["style_ok"] = style is None
        rec["problems"] = problems + ([style] if style else [])
        rec["words"] = len(text.split())
        rec["text"] = text
    rec["deviation"] = bool(deviation_notes(frame))

    # ---- faithfulness of the cited reasons (the linear concepts->bid head the bid is computed from)
    W, b = en.W.astype(np.float64), en.b.astype(np.float64)
    a = np.asarray(out["concept_acts"], np.float64)
    runner = BIDS.index(frame["runner_up"]["bid"])
    m = a * (W[:, bid] - W[:, runner])
    pos_total = m[m > 0].sum()
    names = list(ck.hard_concepts)
    gt = _gt(frame)
    dep = hand_dependent_concepts(frame)
    reasons = select_reasoned_concepts(frame)
    meaning = bid_meaning_concepts(frame)
    R = [names.index(n) for n in reasons if n in names]
    rec["n_reasons"] = len(R)
    rec["reasons"] = reasons
    rec["meaning"] = meaning
    p0, _ = _p_bid(a, W, b, lg, bid)
    rec["p_bid_head"] = float(p0)

    def removal(idx):
        a2 = a.copy(); a2[idx] = 0.0
        p, new = _p_bid(a2, W, b, lg, bid)
        return float(p0 - p), new != bid, float(m[idx].sum() / pos_total) if pos_total > 0 else 0.0

    if R:
        rec["reason_compr"], rec["reason_flip"], rec["reason_share"] = removal(R)
        pool = [k for k, n in enumerate(names) if a[k] >= 0.5 and gt.get(n) and not n.startswith("bid_is_in_")]
        draws = [removal(rng.sample(pool, len(R))) for _ in range(20)] if len(pool) >= len(R) else []
        if draws:
            rec["random_compr"] = float(np.mean([d[0] for d in draws]))
            rec["random_flip"] = float(np.mean([d[1] for d in draws]))
            rec["random_share"] = float(np.mean([d[2] for d in draws]))
        top = list(np.argsort(-m)[:len(R)])
        rec["top_compr"], rec["top_flip"], rec["top_share"] = removal(top)
    # everything the explanation states from the concept layer: the hand-based
    # reasons AND what the call means here (bid_meaning_concepts)
    S = sorted(set(R) | {names.index(n) for n in meaning if n in names})
    if S:
        rec["n_stated"] = len(S)
        rec["stated_compr"], rec["stated_flip"], rec["stated_share"] = removal(S)
        pool = [k for k, n in enumerate(names) if a[k] >= 0.5 and gt.get(n) and not n.startswith("bid_is_in_")]
        draws = [removal(rng.sample(pool, len(S))) for _ in range(20)] if len(pool) >= len(S) else []
        if draws:
            rec["random_stated_compr"] = float(np.mean([d[0] for d in draws]))
            rec["random_stated_flip"] = float(np.mean([d[1] for d in draws]))
            rec["random_stated_share"] = float(np.mean([d[2] for d in draws]))
        rec["top_stated_compr"], rec["top_stated_flip"], rec["top_stated_share"] = removal(list(np.argsort(-m)[:len(S)]))
    # where the margin comes from (positive contributions)
    grp = collections.Counter()
    for k, n in enumerate(names):
        if m[k] <= 0:
            continue
        g = ("strain descriptor" if n.startswith("bid_is_in_") else
             "hand-dependent" if dep.get(n) else "call meaning (hand-independent)")
        grp[g] += m[k]
    rec["margin_groups"] = {g: float(v / pos_total) for g, v in grp.items()} if pos_total > 0 else {}
    rec["margin"] = float(m.sum() + b[bid] - b[runner])
    # ---- per-decision trust signals
    rec["concept_entropy"] = float(np.mean(-(a * np.log(a + 1e-9) + (1 - a) * np.log(1 - a + 1e-9))))
    top5 = np.argsort(-m)[:5]
    rec["relies_on_rule_false"] = bool(any(a[k] >= 0.5 and m[k] > 0 and not gt.get(names[k], False)
                                           and not names[k].startswith("bid_is_in_") for k in top5))
    rec["min_margin_concept_uncertainty"] = float(min(abs(a[k] - 0.5) for k in top5))
    return rec


def summarise(model, pos, recs, strat) -> dict:
    def mean_ci(key, rs, cond=lambda r: True):
        rs = [r for r in rs if key in r and cond(r)]
        if not rs:
            return None
        v = np.array([float(r[key]) for r in rs])
        e, lo, hi = cluster_bootstrap(v, np.array([r["deal"] for r in rs]))
        return {"value": e, "ci95": [lo, hi], "n": len(rs)}

    uni = [r for r in recs if not r.get("stratified")]
    res = {"model": model, "positions": pos.source, "n_uniform": len(uni), "n_stratified": len(strat)}
    res["teacher_coverage"] = mean_ci("explained", uni)
    res["consistent"] = mean_ci("consistent", uni)
    res["style_ok"] = mean_ci("style_ok", uni)
    res["passes_all_checks"] = mean_ci("_pass", [dict(r, _pass=r.get("consistent", False) and r.get("style_ok", False))
                                                for r in uni])
    res["has_reason"] = mean_ci("_has", [dict(r, _has=r["n_reasons"] > 0) for r in uni])
    for k in ("reason_share", "random_share", "top_share", "reason_compr", "random_compr", "top_compr",
              "reason_flip", "random_flip", "top_flip",
              "stated_share", "random_stated_share", "top_stated_share", "stated_compr", "random_stated_compr",
              "top_stated_compr", "stated_flip", "random_stated_flip", "top_stated_flip", "n_stated", "n_reasons"):
        res[k] = mean_ci(k, uni)
        res[k + "_nonpass"] = mean_ci(k, uni, lambda r: r["bid"] != "P")
    groups = collections.defaultdict(list)
    for r in uni:
        for g, v in r["margin_groups"].items():
            groups[g].append(v)
    res["margin_groups_mean"] = {g: float(np.sum(v) / len(uni)) for g, v in groups.items()}
    res["margin_groups_mean_nonpass"] = {
        g: float(np.sum([r["margin_groups"].get(g, 0.0) for r in uni if r["bid"] != "P"]) /
                 max(1, sum(r["bid"] != "P" for r in uni))) for g in groups}
    res["words_mean"] = float(np.mean([r["words"] for r in uni if r.get("explained")]))
    probs = collections.Counter(p.split(":")[0][:60] for r in uni for p in r.get("problems", []))
    res["problem_types"] = dict(probs.most_common(15))
    if "error" in uni[0]:
        err = np.array([r["error"] for r in uni])
        res["model_vs_wbridge5_disagreement"] = float(err.mean())
        sig = {"low confidence (1 - p)": np.array([1 - r["conf"] for r in uni]),
               "concept uncertainty (mean entropy)": np.array([r["concept_entropy"] for r in uni]),
               "uncertain top concept (0.5 - min |c - 0.5|)": np.array([0.5 - r["min_margin_concept_uncertainty"] for r in uni])}
        res["error_detection_auroc"] = {k: auroc(v, err) for k, v in sig.items()}
        flags = {"relies on a concept the rules reject": np.array([r["relies_on_rule_false"] for r in uni]),
                 "deviation note": np.array([r["deviation"] for r in uni]),
                 "explanation fails a check": np.array([not (r.get("consistent", False) and r.get("style_ok", False))
                                                         for r in uni])}
        res["error_flags"] = {}
        for k, f in flags.items():
            res["error_flags"][k] = {
                "flag_rate": float(f.mean()),
                "error_rate_if_flag": wilson(int((err & f).sum()), int(f.sum())),
                "error_rate_if_no_flag": wilson(int((err & ~f).sum()), int((~f).sum())),
                "recall_of_errors": float((err & f).sum() / max(err.sum(), 1))}
    per_cat = collections.defaultdict(list)
    for r in strat:
        per_cat[r["category"]].append(r)
    res["per_category"] = {c: {"n": len(rs),
                               "explained": float(np.mean([r["explained"] for r in rs])),
                               "passes": float(np.mean([r.get("consistent", False) and r.get("style_ok", False) for r in rs])),
                               "has_reason": float(np.mean([r["n_reasons"] > 0 for r in rs])),
                               "reason_share": float(np.mean([r["reason_share"] for r in rs if "reason_share" in r]))
                               if any("reason_share" in r for r in rs) else None,
                               "hand_share": float(np.mean([r["margin_groups"].get("hand-dependent", 0.0) for r in rs])),
                               "error_rate": float(np.mean([r["error"] for r in rs])) if "error" in rs[0] else None}
                           for c, rs in sorted(per_cat.items())}
    return res


def stated_share_medians(specs) -> None:
    """Median stated-concept margin share of non-pass bids, per name=records.pkl."""
    for spec in specs:
        name, path = spec.split("=", 1)
        recs = [r for r in pickle.load(open(path, "rb"))["uniform"] if r["bid"] != "P"]
        u = [r for r in recs if "stated_share" in r]
        share = np.median([r["stated_share"] for r in u])
        rand = np.nanmedian([r.get("random_stated_share", np.nan) for r in u])
        flip = np.mean([r["stated_flip"] for r in u])
        rflip = np.nanmean([r.get("random_stated_flip", np.nan) for r in u])
        note = np.mean([r["deviation"] for r in recs])
        print(f"{name:34s} n={len(u):5d}  share median {share:.2f} (random {rand:.2f})  "
              f"bid changes {flip:.2f} (random {rflip:.2f})  deviation note {note:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="SL independent")
    ap.add_argument("--n", type=int, default=3000, help="uniformly sampled decisions")
    ap.add_argument("--per_category", type=int, default=40, help="extra decisions per bidding situation")
    ap.add_argument("--source", choices=("test", "selfplay"), default="test")
    ap.add_argument("--selfplay_deals", type=int, default=3000)
    ap.add_argument("--random_vulnerability", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir")
    ap.add_argument("--medians", nargs="+", metavar="NAME=RECORDS_PKL",
                    help="only summarise saved records by their median shares (no new run)")
    args = ap.parse_args()
    if args.medians:
        return stated_share_medians(args.medians)
    if not args.out_dir:
        ap.error("--out_dir is required")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = MODELS.get(args.model, args.model)
    ck = load_checkpoint(str(path))
    en = Enricher(ck)
    fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))
    pos = load_test_split() if args.source == "test" else selfplay(path, args.selfplay_deals, args.seed,
                                                                   args.random_vulnerability)
    rows = sample_rows(pos, args.n, args.seed)
    # stratified extra sample: categorise the model's call at every position
    pl = np.concatenate([np.asarray(fwd(pos.obs_f32(np.arange(s, min(s + 8192, len(pos)))))["policy_logits"])
                         for s in range(0, len(pos), 8192)])
    pred = masked_argmax(pl, pos.legal)
    cats = collections.defaultdict(list)
    t0 = time.time()
    for i in range(len(pos)):
        cats[light_category(pos, i, BIDS[pred[i]])].append(i)
    print(f"[faithfulness] categorised {len(pos):,} decisions into {len(cats)} situations ({time.time() - t0:.0f}s)",
          flush=True)
    rr = random.Random(args.seed)
    strat_rows = sorted(i for c, v in cats.items() for i in rr.sample(v, min(args.per_category, len(v))))
    rng = random.Random(args.seed)
    recs, strat = [], []
    t0 = time.time()
    for j, i in enumerate(rows):
        recs.append(analyse_decision(ck, en, fwd, pos, i, rng))
        if j % 500 == 0:
            print(f"[faithfulness] uniform {j:,}/{len(rows):,} ({time.time() - t0:.0f}s)", flush=True)
    for j, i in enumerate(strat_rows):
        r = analyse_decision(ck, en, fwd, pos, i, rng)
        r["stratified"] = True
        strat.append(r)
        if j % 500 == 0:
            print(f"[faithfulness] stratified {j:,}/{len(strat_rows):,} ({time.time() - t0:.0f}s)", flush=True)
    res = summarise(args.model, pos, recs, strat)
    res["situations_total"] = {c: len(v) for c, v in sorted(cats.items())}
    tag = f"{args.model.replace(' ', '_').replace('/', '_')}__{args.source}"  # a path model: no subdirectories
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    with open(out / f"{tag}_records.pkl", "wb") as f:
        pickle.dump({"uniform": recs, "stratified": strat}, f)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_category", "situations_total")}, indent=1))


if __name__ == "__main__":
    main()
