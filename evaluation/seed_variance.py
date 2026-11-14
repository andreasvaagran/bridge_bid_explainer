#!/usr/bin/env python3
"""Are the mode differences larger than seed-to-seed variation? Three SL seeds per mode: agreement, concept
F1 and duplicate matches, with per-mode and paired-difference spread."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.common import MODELS, REPO, RUNS, cluster_bootstrap, load_test_split, masked_argmax, policy_fn

ROOT = Path(__file__).resolve().parent.parent
MODES = ("joint", "independent", "sequential")


def seed_models(seeds_dir: Path) -> dict:
    out = {}
    for m in MODES:
        paths = [("thesis", REPO / RUNS["sl"].format(mode=m) / "params-best.pkl")]
        paths += [(p.parent.name.split("_s")[-1], p) for p in sorted(seeds_dir.glob(f"{m}_s*/params-best.pkl"))]
        out[m] = [(tag, p) for tag, p in paths if p is not None and p.exists()]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds_dir", default="runs/sl")
    ap.add_argument("--deals", type=int, default=10_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from evaluation.duplicate_match import evaluate as duplicate
    test = load_test_split()
    models = seed_models(ROOT / args.seeds_dir)
    res = {"models": {}, "summary": {}, "paired": {}}
    for m, lst in models.items():
        for tag, p in lst:
            f, _ = policy_fn(p)
            pred = np.concatenate([masked_argmax(f(test.obs_f32(np.arange(s, min(s + 8192, len(test))))),
                                                 test.legal[s:s + 8192]) for s in range(0, len(test), 8192)])
            e, lo, hi = cluster_bootstrap((pred == test.action).astype(float), test.deal)
            te = p.parent / "test_eval.json"
            f1 = json.loads(te.read_text()).get("concept_macro_f1") if te.exists() else None
            d = duplicate(str(p), str(MODELS["brl SL"]), num_deals=args.deals, vulnerability="none")
            res["models"][f"{m} {tag}"] = {"path": str(p), "agreement": e, "ci95": [lo, hi], "concept_macro_f1": f1,
                                           "vs_brl_sl": {k: d[k] for k in ("imps_per_board", "ci95", "se")}}
            print(f"[seeds] {m} {tag}: agree {e:.4f}, F1 {f1}, vs brl SL {d['imps_per_board']:+.3f}", flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(res, indent=1))
    for m, lst in models.items():
        rows = [res["models"][f"{m} {t}"] for t, _ in lst]
        res["summary"][m] = {
            "n_seeds": len(rows),
            "agreement_mean": float(np.mean([r["agreement"] for r in rows])),
            "agreement_sd": float(np.std([r["agreement"] for r in rows], ddof=1)) if len(rows) > 1 else None,
            "vs_brl_sl_mean": float(np.mean([r["vs_brl_sl"]["imps_per_board"] for r in rows])),
            "vs_brl_sl_sd": float(np.std([r["vs_brl_sl"]["imps_per_board"] for r in rows], ddof=1)) if len(rows) > 1 else None,
            "concept_f1_mean": float(np.mean([r["concept_macro_f1"] for r in rows if r["concept_macro_f1"] is not None]))
            if any(r["concept_macro_f1"] is not None for r in rows) else None}
    joint = dict(models["joint"])
    for m in ("independent", "sequential"):
        diffs = []
        for tag, p in models[m]:
            if tag not in joint:
                continue
            d = duplicate(str(p), str(joint[tag]), num_deals=args.deals, vulnerability="none")
            diffs.append({"seed": tag, "imps_per_board": d["imps_per_board"], "ci95": d["ci95"],
                          "agreement_diff": res["models"][f"{m} {tag}"]["agreement"] - res["models"][f"joint {tag}"]["agreement"]})
            print(f"[seeds] {m} vs joint, seed {tag}: {d['imps_per_board']:+.3f}", flush=True)
        v = [x["imps_per_board"] for x in diffs]
        a = [x["agreement_diff"] for x in diffs]
        res["paired"][f"{m} vs joint"] = {"per_seed": diffs, "imps_mean": float(np.mean(v)) if v else None,
                                          "imps_sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                          "imps_range": [float(min(v)), float(max(v))] if v else None,
                                          "agreement_diff_mean": float(np.mean(a)) if a else None,
                                          "agreement_diff_sd": float(np.std(a, ddof=1)) if len(a) > 1 else None}
        Path(args.out).write_text(json.dumps(res, indent=1))
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
