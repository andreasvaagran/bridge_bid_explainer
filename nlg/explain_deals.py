#!/usr/bin/env python3
"""Explains every call of held-out deals in the terminal, with the evidence nlg/decision_record.py records
(teacher text, and with --adapter Qwen's checked text)."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", required=True, help="bids for North-South (and East-West unless --opponents)")
    ap.add_argument("--opponents", default=None, help="another model for East-West")
    ap.add_argument("--adapter", default=None, help="trained Qwen LoRA adapter (nlg/train_qwen.py) for LLM explanations")
    ap.add_argument("--deals", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0, help="first deal (deal i is seed + i)")
    ap.add_argument("--vulnerability", default="None", choices=("random", "None", "NS", "EW", "Both"),
                    help="default None: the WBridge5 SL data is non-vulnerable (use random for PPO models)")
    ap.add_argument("--dealer", default=None, choices=("N", "E", "S", "W"))
    ap.add_argument("--dds_path", default=str(_REPO / "dds_results" / "dds_results_500K.npy"))
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--out", default=None, help="also write the explained records (JSON lines) here")
    args = ap.parse_args()
    if args.adapter:  # before JAX is imported: its GPU preallocation would leave no room for Qwen
        _os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from gui.game import Game, Registry
    from nlg.decision_record import format_record
    paths = {"a": args.model_path}
    if args.opponents:
        paths["b"] = args.opponents
    reg = Registry.of(paths)
    seats = {s: ("b" if args.opponents and s in "EW" else "a") for s in "NESW"}

    llm = None
    if args.adapter:
        from peft import PeftModel
        from nlg.eval_qwen import explain_checked
        from nlg.qwen_model import load_llm
        base = load_llm()
        model = PeftModel.from_pretrained(base.model, args.adapter)
        model.eval()
        llm = lambda rec: explain_checked(model, base.tokenizer, rec["frame"], rec["reasons"], retries=args.retries)  # noqa: E731

    out = open(args.out, "w") if args.out else None
    for i in range(args.deals):
        g = Game(reg, seats, seed=args.seed + i, dealer=args.dealer, vulnerability=args.vulnerability,
                 dds_path=args.dds_path)
        g.auto()
        j = g.to_json()
        print(f"\n=== Deal {j['seed']}: dealer {j['dealer']}, vulnerable {j['vulnerability']}; "
              f"N-S {reg.specs[seats['N']]['label']}, E-W {reg.specs[seats['E']]['label']}")
        for s in "NESW":
            print(f"  {s}: {j['hands'][s]}")
        print("  Auction: " + " ".join(f"{a['seat']}:{a['call']}" for a in j["auction"]))
        for a in j["auction"]:
            rec = g.records[a["record"]]
            if llm:
                r = llm(rec)
                rec["llm_explanation"] = r["final"] + ("" if r["final_ok"] else "  [did not pass the checks]")
            print("\n" + format_record(rec))
            if out:
                out.write(json.dumps({"deal": j["seed"], **{k: v for k, v in rec.items() if k != "frame"}},
                                     ensure_ascii=False, default=str) + "\n")
        r = j["result"]
        print(f"\n  Result: {r['contract']}" + (f" by {r['declarer']}, {r['tricks']} tricks double-dummy, "
                                              f"N-S score {r['score_ns']:+d}" if r.get("declarer") else ""))
    if out:
        out.close()


if __name__ == "__main__":
    main()
