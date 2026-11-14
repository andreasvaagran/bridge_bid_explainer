#!/usr/bin/env python3
"""The three-way table: one checkpoint per bottleneck mode, evaluated on every decision of the test split
(evaluation/split_accuracy.py), with each mode's errors broken down by type."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json

from evaluation.split_accuracy import evaluate

_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--joint", required=True)
    p.add_argument("--independent", required=True)
    p.add_argument("--sequential", required=True)
    p.add_argument("--split", default=_os.path.join(_REPO, "data", "wbridge5", "prepared", "test"),
                   help="a prepared split directory (every decision is evaluated)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = {}
    for mode in ("joint", "independent", "sequential"):
        path = getattr(args, mode)
        print(f"[compare_modes] evaluating {mode}: {path}", flush=True)
        results[mode] = evaluate(path, args.split)

    n = results["joint"]["n_decisions"]
    print(f"\n{'=' * 104}\n  Concept-bottleneck mode comparison ({n:,} test decisions, every decision)\n{'=' * 104}")
    print(f"{'mode':<12} {'policy acc':>10} {'NLL':>7} {'concept F1 macro':>17} {'concept-only acc':>17} "
          f"{'agree':>7} {'wrong strain':>13} {'call type':>10} {'other':>7}")
    for mode, r in results.items():
        e = r["error_breakdown"]
        print(f"{mode:<12} {r['acc']:>10.4f} {r['nll']:>7.3f} {r['concept_macro_f1']:>17.3f} "
              f"{r['concept_only_acc']:>17.4f} {r['concept_policy_agreement']:>7.3f} "
              f"{e['same_level_wrong_strain']:>13.0%} {e['call_type']:>10.0%} {e['other']:>7.0%}")
    print("  (error columns: share of that mode's errors that are right-level-wrong-strain / "
          "pass-double-bid confusions / anything else)")
    gap = results["joint"]["acc"] - results["independent"]["acc"]
    print(f"\n[interpretation]\n  joint - independent policy accuracy: {gap:+.4f}. In 'independent'/'sequential' "
          f"the policy IS a linear map of the concepts (concept-only acc == policy acc), so this gap is the "
          f"measured cost of a true bottleneck with this vocabulary. In 'joint', concept-only acc is the "
          f"diagnostic head's accuracy: how much of the full policy the concepts alone account for.")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({m: {k: v for k, v in r.items() if k != "per_concept_f1"} for m, r in results.items()},
                      f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
