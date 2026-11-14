#!/usr/bin/env python3
"""Every strength comparison of the thesis with one protocol: 10,000 duplicate boards per match
(duplicate_match.py) and each pair's agreement with WBridge5 and with each other. --all_pairs gives the full matrix
and --distill the distilled models' matches."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path

import numpy as np

from evaluation.duplicate_match import evaluate
from evaluation.common import (DISTILLED, MODELS, REPO, checkpoint_stamp, cluster_bootstrap, load_test_split,
                                  masked_argmax, policy_fn, same_checkpoint)

MATCHES = [
    # (A, B, vulnerability settings) -- the cost of concepts: SL concept models vs brl's SL network
    ("SL joint", "brl SL", ("none", "random")),
    ("SL independent", "brl SL", ("none", "random")),
    ("SL sequential", "brl SL", ("none", "random")),
    # the three modes against each other
    ("SL independent", "SL joint", ("none",)),
    ("SL sequential", "SL joint", ("none",)),
    # what PPO adds over its own SL start
    ("PPO joint", "SL joint", ("random",)),
    ("PPO independent", "SL independent", ("random",)),
    ("PPO sequential", "SL sequential", ("random",)),
    # PPO against the published brl networks
    ("PPO joint", "brl SL", ("random",)),
    ("PPO independent", "brl SL", ("random",)),
    ("PPO sequential", "brl SL", ("random",)),
    ("PPO joint", "brl RL+PFSP", ("random",)),
    ("PPO independent", "brl RL+PFSP", ("random",)),
    ("PPO sequential", "brl RL+PFSP", ("random",)),
    # reference: how much brl's own RL added
    ("brl RL", "brl SL", ("random",)),
    ("brl RL+PFSP", "brl SL", ("random",)),
    ("brl RL+FSP", "brl SL", ("random",)),
    ("PPO independent", "brl RL+FSP", ("random",)),
]
# the unanchored PPO models: against their SL start, their anchored counterpart and the published networks; SL and PPO
# side by side -- the modes against each other after PPO too, every PPO model against brl's best network, and the
# plain network against its concept counterparts
for _mode in ("joint", "independent", "sequential"):
    _u = f"PPO {_mode} unanchored"
    MATCHES += [(_u, f"SL {_mode}", ("random",)), (_u, f"PPO {_mode}", ("random",)),
                (_u, "brl SL", ("random",)), (_u, "brl RL+PFSP", ("random",)), (_u, "brl RL+FSP", ("random",))]
MATCHES += [("PPO joint", "brl RL+FSP", ("random",)), ("PPO sequential", "brl RL+FSP", ("random",)),
            ("PPO independent", "PPO joint", ("random",)), ("PPO sequential", "PPO joint", ("random",)),
            ("PPO independent unanchored", "PPO joint unanchored", ("random",)),
            ("PPO sequential unanchored", "PPO joint unanchored", ("random",)),
            ("SL plain", "SL joint", ("none",)), ("SL independent", "SL plain", ("none",)),
            ("SL plain", "brl SL", ("none", "random")), ("brl RL scratch", "brl SL", ("random",)),
            ("PPO plain unanchored", "SL plain", ("random",)),
            ("PPO plain unanchored", "PPO joint unanchored", ("random",)),
            ("PPO plain unanchored", "brl SL", ("random",))]
MATCHES = [m for m in MATCHES if m[0] in MODELS and m[1] in MODELS]
# --all_pairs: every model of the thesis's strength heatmap against every other, so that tab:strength is a
# full matrix, under the thesis's protocol: two supervised models play non-vulnerable (as their data were), any pair
# with an RL model random vulnerability; a pair already played in either direction is not played again
HEATMAP_MODELS = ["brl SL", "SL plain", "SL joint", "SL independent", "SL sequential", "PPO joint", "PPO independent",
                  "PPO sequential", "PPO plain unanchored", "PPO joint unanchored", "PPO independent unanchored",
                  "PPO sequential unanchored", "brl RL scratch", "brl RL", "brl RL+PFSP", "brl RL+FSP"]


SUPERVISED = {"brl SL", "SL plain", "SL joint", "SL independent", "SL sequential"}
# --distill: the concept models distilled from brl RL+FSP's games (scripts/distill_rl_fsp.py,
# evaluation.common.DISTILLED) against the models the thesis compares them with and against each other -- the
# heatmap of the thesis's distillation section (tab:distill_strength). Every pair involves a model whose data come
# from RL self-play, so all are played with random vulnerability, as the distillation's own matches were; those are
# imported, not replayed. The distilled models are registered here only, so that no other evaluation's default
# model list picks them up.
DISTILL_REFERENCES = ["brl SL", "SL joint", "SL independent", "SL sequential", "PPO independent",
                      "PPO joint unanchored", "brl RL+FSP", "brl RL scratch"]


def protocol_vulnerability(a: str, b: str) -> str:
    return "none" if a in SUPERVISED and b in SUPERVISED else "random"


def all_pairs() -> list:
    have = {(frozenset((a, b)), x) for a, b, v in MATCHES for x in v}
    ms = [m for m in HEATMAP_MODELS if m in MODELS]
    return [(a, b, (protocol_vulnerability(a, b),)) for i, a in enumerate(ms) for b in ms[i + 1:]
            if (frozenset((a, b)), protocol_vulnerability(a, b)) not in have]


def distill_pairs() -> list:
    """Every distilled model against every reference and every other distilled model, random vulnerability."""
    MODELS.update({n: p for n, p in DISTILLED.items() if p.exists()})
    rows = [m for m in DISTILLED if m in MODELS]
    refs = [m for m in DISTILL_REFERENCES if m in MODELS]
    return ([(a, b, ("random",)) for a in rows for b in refs]
            + [(a, b, ("random",)) for i, a in enumerate(rows) for b in rows[i + 1:]])


def distill_matches(pairs: list) -> list:
    """The distillation's own duplicate matches (runs/distill/eval; the same
    protocol: 10,000 boards of the 500K table, seed 0) for the pairs asked for, in this script's format, so that they
    are not replayed."""
    want = {(Path(MODELS[a]).resolve(), Path(MODELS[b]).resolve()): (a, b) for a, b, _ in pairs}
    out = []
    for f in sorted((REPO / "runs/distill/eval").glob("*_vs_*_random.json")):
        r = json.loads(f.read_text())
        pa, pb = (Path(r[k]) if Path(r[k]).is_absolute() else REPO / r[k] for k in ("a", "b"))
        key = (pa.resolve(), pb.resolve())
        if key not in want or r.get("deals") != 10_000 or r.get("seed") != 0 or r.get("vulnerability") != "random":
            continue
        a, b = want[key]
        r.update({"A": a, "B": b, "source": str(f.relative_to(REPO)), **checkpoint_stamp(MODELS[a], "a"),
                  **checkpoint_stamp(MODELS[b], "b")})
        out.append(r)
    return out


def imitation(models, pos) -> dict:
    preds = {}
    for m in models:
        f, _ = policy_fn(MODELS[m])
        preds[m] = np.concatenate([masked_argmax(f(pos.obs_f32(np.arange(s, min(s + 8192, len(pos))))),
                                                 pos.legal[s:s + 8192]) for s in range(0, len(pos), 8192)])
    acc = {}
    for m, p in preds.items():
        e, lo, hi = cluster_bootstrap((p == pos.action).astype(float), pos.deal)
        acc[m] = {"value": e, "ci95": [lo, hi]}
    return acc, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_deals", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all_pairs", action="store_true", help="also every pair of HEATMAP_MODELS (random vulnerability)")
    ap.add_argument("--distill", action="store_true",
                    help="only the distilled models' matches (DISTILLED x DISTILL_REFERENCES and each other); use its "
                         "own --out, e.g. strength/strength_distill.json")
    args = ap.parse_args()
    matches = distill_pairs() if args.distill else MATCHES + (all_pairs() if args.all_pairs else [])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {"matches": []}
    # a match whose models' checkpoints changed since it was played is replayed, not kept
    res["matches"] = [r for r in res["matches"] if r["A"] in MODELS and r["B"] in MODELS
                      and same_checkpoint(r, MODELS[r["A"]], "a") and same_checkpoint(r, MODELS[r["B"]], "b")]
    done = {(r["A"], r["B"], r["vulnerability"]) for r in res["matches"]}

    pos = load_test_split()
    names = sorted({m for a, b, _ in matches for m in (a, b)})
    acc, preds = imitation(names, pos)
    res["wbridge5_agreement"] = acc
    res["test_decisions"] = len(pos)
    print("[strength] WBridge5 agreement: " + ", ".join(f"{m} {v['value']:.4f}" for m, v in acc.items()), flush=True)
    if args.distill:
        for r in distill_matches(matches):
            if (r["A"], r["B"], r["vulnerability"]) not in done:
                r["call_agreement_on_wbridge5_positions"] = float((preds[r["A"]] == preds[r["B"]]).mean())
                res["matches"].append(r)
                done.add((r["A"], r["B"], r["vulnerability"]))
                print(f"[strength] {r['A']} vs {r['B']}: imported from {r['source']} ({r['imps_per_board']:+.3f})",
                      flush=True)
        out.write_text(json.dumps(res, indent=1))

    for a, b, vuls in matches:
        for vul in vuls:
            if (a, b, vul) in done:
                continue
            t0 = time.time()
            r = evaluate(str(MODELS[a]), str(MODELS[b]), args.num_deals, args.batch, args.seed, vulnerability=vul)
            r.update({"A": a, "B": b, "vulnerability": vul,
                      "call_agreement_on_wbridge5_positions": float((preds[a] == preds[b]).mean()),
                      **checkpoint_stamp(MODELS[a], "a"), **checkpoint_stamp(MODELS[b], "b")})
            res["matches"].append(r)
            out.write_text(json.dumps(res, indent=1))
            print(f"[strength] {a} vs {b} ({vul}): {r['imps_per_board']:+.3f} +- {r['se']:.3f} IMPs/board, "
                  f"A better {r['a_won']:.1%} / tied {r['tied']:.1%} / B better {r['b_won']:.1%}, "
                  f"same call {r['call_agreement_on_wbridge5_positions']:.1%} ({time.time() - t0:.0f}s)", flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
