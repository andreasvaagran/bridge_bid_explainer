#!/usr/bin/env python3
"""The PPO anchor ablation: runs that differ only in the KL anchor (and lr), each measured at several
checkpoints for strength (IMPs vs its SL start and brl SL), drift (KL to SL, agreement with WBridge5, calls that
break WBridge5's system) and concept F1 on its own calls."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from evaluation.common import MODELS, cluster_bootstrap, load_test_split, masked_argmax, policy_fn, selfplay

ROOT = Path(__file__).resolve().parent.parent


def _update(p: Path) -> int:
    m = re.search(r"params-(\d+)", p.name)
    return int(m.group(1)) if m else 0  # the SL start (params-best.pkl)


def checkpoints(run: Path, points: int, max_update: int | None = None):
    ck = sorted((run / "rl_params").glob("params-*.pkl"), key=_update)
    if max_update is not None:
        ck = [c for c in ck if _update(c) <= max_update]
    if not ck:
        return []
    idx = sorted({int(round(x)) for x in np.linspace(0, len(ck) - 1, points + 1)[1:]})
    return [ck[i] for i in idx]


def local_path(p: str | None) -> str | None:
    """A checkpoint path written in another checkout (results hold absolute paths): the same file under this
    checkout's runs/, when the recorded path does not exist here but that one does."""
    if not p or Path(p).exists():
        return p
    m = re.search(r"(?:^|/)(runs/.+)$", p)
    return str(ROOT / m.group(1)) if m and (ROOT / m.group(1)).exists() else p


RUN_NAME = re.compile(r"(ind|joint|seq)_a([\d.]+)_lr([\de.-]+)")   # pipeline/train.sh's PPO run names


def run_meta(run: Path) -> dict:
    """mode / anchor / lr from the run's name, and the SL start it was launched from."""
    m = RUN_NAME.match(run.parent.name)
    meta = {"mode": {"ind": "independent", "joint": "joint", "seq": "sequential"}[m.group(1)],
            "anchor": float(m.group(2)), "lr": float(m.group(3))}
    first = json.loads((run / "launches.jsonl").read_text().splitlines()[0])
    meta["sl"] = local_path(str(ROOT / first["initial_model_path"]))
    log = run.parent.with_suffix(".log")
    if log.exists():
        meta["log"] = str(log)
    return meta


def metrics_at(run: Path, update: int) -> dict:
    """KL to SL / same-call rate on the rollouts nearest to `update` (mean of the last 5 logged rows)."""
    mp = run / "metrics.jsonl"
    if not mp.exists():
        return {}
    rows = [json.loads(l) for l in mp.open() if '"kl_to_sl"' in l]
    # resumed runs restart the counter: keep the LAST occurrence of each update
    by = {}
    for r in rows:
        by[r["update"]] = r
    us = sorted(u for u in by if u <= update)[-5:]
    if not us:
        return {}
    return {"kl_to_sl": float(np.mean([by[u]["kl_to_sl"] for u in us])),
            "same_call_as_sl": float(np.mean([by[u]["sl_agreement"] for u in us]))}


def concept_f1_selfplay(path: str, sp) -> float | None:
    import jax
    from training.checkpoint import load_checkpoint
    from evaluation.concept_f1 import f1_table, labels_for
    ck = load_checkpoint(str(path))
    K = len(ck.hard_concepts)
    fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))
    acts = np.concatenate([np.asarray(fwd(sp.obs_f32(np.arange(s, min(s + 8192, len(sp)))))["concept_acts"])
                           for s in range(0, len(sp), 8192)])
    gt = labels_for(sp, np.arange(len(sp)), sp.action)[:, :K]
    return f1_table(acts, gt, ck.hard_concepts)["macro_f1"]


_WB5_CARD = None


def wb5_card():
    """WBridge5's convention card, fitted on the test split's 10,000 deals as in
    evaluation/convention_ranges.py (2.5th-97.5th HCP percentile and shortest bid-suit
    length per bidding situation and call)."""
    global _WB5_CARD
    if _WB5_CARD is None:
        from evaluation.convention_ranges import calls, fit
        _WB5_CARD = fit(calls(load_test_split()))
    return _WB5_CARD


def card_measure(sp) -> dict:
    """How much of an agent's own self-play bidding WBridge5's card describes: over all
    its non-pass calls (a call in a situation WBridge5 never bids counts as not described)
    and over the calls whose (situation, call) has an entry in WBridge5's card."""
    from evaluation.convention_ranges import calls, describe
    rows = calls(sp)
    card = wb5_card()
    sup = [x for x in rows if x[1] in card]
    return {"wbridge5_card_described": describe(card, rows)["described"],
            "wbridge5_card_described_supported": describe(card, sup)["described"] if sup else None,
            "calls": len(rows), "supported_share": len(sup) / max(len(rows), 1)}


def evaluate_point(path: Path, meta: dict, test, args) -> dict:
    from evaluation.duplicate_match import evaluate as duplicate
    from evaluation.convention_drift import analyse as drift
    t0 = time.time()
    r = {"checkpoint": str(path), "update": _update(path)}
    f, _ = policy_fn(path)
    pred = np.concatenate([masked_argmax(f(test.obs_f32(np.arange(s, min(s + 8192, len(test))))),
                                         test.legal[s:s + 8192]) for s in range(0, len(test), 8192)])
    e, lo, hi = cluster_bootstrap((pred == test.action).astype(float), test.deal)
    r["wbridge5_agreement"] = {"value": e, "ci95": [lo, hi]}
    for tag, opp in (("vs_own_sl", meta["sl"]), ("vs_brl_sl", str(MODELS["brl SL"]))):
        if tag == "vs_own_sl" and str(path) == str(opp):
            r[tag] = {"imps_per_board": 0.0, "ci95": [0.0, 0.0], "se": 0.0}
            continue
        d = duplicate(str(path), str(opp), num_deals=args.strength_deals, vulnerability="random")
        r[tag] = {k: d[k] for k in ("imps_per_board", "ci95", "se", "tied")}
    sp = selfplay(path, args.deals, seed=0, random_vulnerability=False)
    dr = drift(path.parent.parent.name + "@" + str(r["update"]), sp)
    r["deviation_rate_non_pass"] = dr["deviation_rate_non_pass"]
    r["opening_calls"] = dr["opening_calls"]
    r["profiles"] = {k: dr["profiles"][k] for k in ("opening_1_major", "opening_pass") if k in dr["profiles"]}
    r["concept_f1_selfplay"] = concept_f1_selfplay(path, sp)
    r["cards"] = card_measure(sp)
    r["seconds"] = round(time.time() - t0, 1)
    print(f"[ablation] {path}: agree {e:.4f}, vs SL {r['vs_own_sl']['imps_per_board']:+.3f}, "
          f"vs brl SL {r['vs_brl_sl']['imps_per_board']:+.3f}, deviation {dr['deviation_rate_non_pass']['value']:.4f}, "
          f"concept F1 {r['concept_f1_selfplay']:.3f} ({r['seconds']:.0f}s)", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=["runs/ppo"])
    ap.add_argument("--only", nargs="*", default=None, help="run names to (re)evaluate; others are kept from --out")
    ap.add_argument("--points", type=int, default=3)
    ap.add_argument("--deals", type=int, default=2000, help="self-play deals for drift / concept F1")
    ap.add_argument("--strength_deals", type=int, default=10_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--same_points_as", default=None, metavar="JSON",
                    help="evaluate each run at the checkpoints an earlier result recorded (for a rerun after unused "
                         "intermediate checkpoints were deleted: --points would space its picks over the files that "
                         "are left and choose other updates)")
    ap.add_argument("--cards_only", action="store_true",
                    help="only add the convention-card measure to the points already in --out")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {"runs": {}}
    if args.cards_only:
        for name, run in res["runs"].items():
            if args.only and name not in args.only:
                continue
            for pt in run["points"]:
                if "cards" in pt:
                    continue
                sp = selfplay(Path(pt["checkpoint"]), args.deals, seed=0, random_vulnerability=False)
                pt["cards"] = card_measure(sp)
                print(f"[ablation cards] {name} @{pt['update']}: WBridge5's card describes "
                      f"{pt['cards']['wbridge5_card_described']:.3f} "
                      f"(supported {pt['cards']['wbridge5_card_described_supported']:.3f})", flush=True)
                out.write_text(json.dumps(res, indent=1))
        print(f"[done] -> {out}")
        return
    test = load_test_split()
    runs = {}
    for d in args.runs:
        for run in sorted((ROOT / d).glob("*/ppo")):
            if (run / "rl_params").is_dir() and RUN_NAME.match(run.parent.name):   # not the plain network's run
                runs[run.parent.name] = run
    for name, run in runs.items():
        if args.only and name not in args.only:
            continue
        meta = run_meta(run)
        pts = []
        sl_key = f"SL {meta['mode']} (start)"
        if sl_key not in res["runs"]:
            res["runs"][sl_key] = {"meta": {"mode": meta["mode"]},
                                   "points": [evaluate_point(Path(meta["sl"]), meta, test, args)]}
            out.write_text(json.dumps(res, indent=1))
        conv = run.parent / "converged.json"      # stop at the converged checkpoint
        last = json.loads(conv.read_text())["update"] if conv.exists() else None
        same = json.loads(Path(args.same_points_as).read_text())["runs"].get(name) if args.same_points_as else None
        cks = [Path(local_path(pt["checkpoint"])) for pt in same["points"]] if same else checkpoints(run, args.points, max_update=last)
        for ck in cks:
            p = evaluate_point(ck, meta, test, args)
            p.update(metrics_at(run, p["update"]))
            pts.append(p)
            res["runs"][name] = {"meta": meta, "points": pts}
            out.write_text(json.dumps(res, indent=1))
    out.write_text(json.dumps(res, indent=1))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
