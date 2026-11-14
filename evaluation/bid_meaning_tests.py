#!/usr/bin/env python3
"""What a model's bids mean in its own system, one subcommand each:
  artificial  natural vs artificial suit bids on the cards learned from play, and what each artificial bid conveys;
  swap        does partner read an opening by its meaning (1C/1D swapped in the observation)?
  contracts   where the partnership plays after each one-of-a-minor opening."""
from __future__ import annotations

import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import collections
import json

import numpy as np

SU = "SHDC"


# ---------------------------------------------------------------------------
# artificial: natural vs artificial suit bids on a learned card
# ---------------------------------------------------------------------------

def _strain(c):
    return c[1] if len(c) == 2 and c[0].isdigit() and c[1] in SU else None


def _context_totals(ctx):
    tot = {s: np.zeros(14) for s in SU}
    hc = np.zeros(38)
    for v in ctx["shows"].values():
        for s in SU:
            tot[s] += v["len_dist"][s]
        hc += v["hcp_dist"]
    return tot, hc


def _median(d):
    c = np.cumsum(d)
    return int(np.searchsorted(c, c[-1] / 2))


def classify(card, min_n=200):
    """{(auction key, call): natural or not, and what the bid conveys} for every suit bid made min_n times."""
    out = {}
    for key, ctx in card["contexts"].items():
        tot, hc = _context_totals(ctx)
        for call, v in ctx.get("shows", {}).items():
            s = _strain(call)
            if not s or v["n"] < min_n:
                continue
            ld = np.array(v["len_dist"][s])
            p3 = ld[3:].sum() / ld.sum()
            conv = {"hcp": [_median(v["hcp_dist"]), _median(hc)]}
            for t in SU:
                a, b = np.array(v["len_dist"][t]), tot[t]
                conv[t] = [round(a[4:].sum() / a.sum(), 2), round(b[4:].sum() / b.sum(), 2)]
            out[(key, call)] = {"auction": ctx["auction"], "n": v["n"], "p3": round(float(p3), 2),
                                "natural": p3 >= 0.9, "hcp": v["hcp"], "conveys": conv}
    return out


def artificial(args):
    wb = classify(json.load(open(args.wb5_card)))
    res = {}
    for p in args.cards:
        c = classify(json.load(open(p)))
        name = p.split("/")[-1][:-5]
        n_all = sum(v["n"] for v in c.values())
        n_art = sum(v["n"] for v in c.values() if not v["natural"])
        became = [k for k in c if k in wb and wb[k]["natural"] and not c[k]["natural"]]
        n_became = sum(c[k]["n"] for k in became)
        res[name] = {"artificial_share": n_art / n_all, "became_artificial_share": n_became / n_all,
                     "became": sorted([[c[k]["n"], c[k]["auction"], k[1], c[k]["p3"], wb[k]["p3"], c[k]["hcp"],
                                        c[k]["conveys"]] for k in became], reverse=True)}
        print(f"{name:28s} artificial {n_art/n_all:6.1%}   natural in WBridge5 but artificial here: "
              f"{len(became):3d} bids, {n_became/n_all:6.1%} of suit bids")
    json.dump(res, open(args.out, "w"), indent=1, default=str)


# ---------------------------------------------------------------------------
# swap: the message swap at partner's reply to a one-of-a-minor opening
# ---------------------------------------------------------------------------

def _first_call(calls):
    k = 0
    while k < len(calls) and calls[k] == "P":
        k += 1
    return k


def swap(args):
    from concepts.auction_obs import BID_NAMES, auction_from_obs
    from evaluation.convention_cards_from_play import F, hand_features
    from evaluation.common import BIDS, MODELS, load_test_split, policy_fn

    listeners = args.listeners.split(";")
    pos = load_test_split(args.split)
    rows, opening = [], []
    for i in range(len(pos)):
        calls, _ = auction_from_obs(pos.obs[i])
        k = _first_call(calls)
        if k < 3 and calls[k:] in (["1C", "P"], ["1D", "P"]):
            rows.append(i)
            opening.append(calls[k])
    rows, opening = np.array(rows), np.array(opening)
    obs = pos.obs[rows].copy()
    hf = hand_features(obs)

    def legal_after(op):
        b = BID_NAMES.index(op)
        legal = np.zeros(38, bool)
        legal[0] = True
        legal[3 + b + 1:] = True
        return legal

    def swapped(o, frm, to):
        o = o.copy()
        o[:, 8 + BID_NAMES.index(frm) * 12 + 2] = 0
        o[:, 8 + BID_NAMES.index(to) * 12 + 2] = 1
        return o, np.tile(legal_after(to), (len(o), 1))

    def stats(a, sel):
        c = [BIDS[x] for x in a[sel]]
        n = len(c)
        return {"n": n, "pass": sum(x == "P" for x in c) / n,
                "bids_clubs": sum(x.endswith("C") for x in c) / n,
                "bids_diamonds": sum(x.endswith("D") and x[0].isdigit() for x in c) / n,
                "game_or_higher_or_2nt+": sum(x[0].isdigit() and (int(x[0]) >= 3 or x == "2NT") for x in c) / n}

    groups = {"all": np.ones(len(hf), bool),
              "clubs>=5,diamonds<=3": (hf[:, F["len_C"]] >= 5) & (hf[:, F["len_D"]] <= 3),
              "diamonds>=5,clubs<=3": (hf[:, F["len_D"]] >= 5) & (hf[:, F["len_C"]] <= 3),
              "hcp>=12": hf[:, F["hcp"]] >= 12, "hcp<=6": hf[:, F["hcp"]] <= 6}
    res, acts = {}, {}
    for spec in listeners:
        name, path = spec.split("=") if "=" in spec else (spec, MODELS[spec])
        logits, _ = policy_fn(path)
        out = {}
        for real, other in (("1C", "1D"), ("1D", "1C")):
            m = opening == real
            o_real, l_real = obs[m], np.tile(legal_after(real), (m.sum(), 1))
            o_sw, l_sw = swapped(o_real, real, other)
            a_real = np.where(l_real, logits(o_real.astype(np.float32)), -np.inf).argmax(1)
            a_sw = np.where(l_sw, logits(o_sw.astype(np.float32)), -np.inf).argmax(1)
            acts.setdefault(name, {})[real] = (a_real, a_sw)
            out[f"{real} (real) vs {other} (swapped in)"] = {
                g: {"real": stats(a_real, s[m]), "swapped": stats(a_sw, s[m])}
                for g, s in groups.items() if s[m].sum() >= 30}
        res[name] = out
    for name, out in res.items():
        print("==", name)
        for k, v in out.items():
            print("  ", k)
            for g, d in v.items():
                r, s = d["real"], d["swapped"]
                print(f"     {g:24s} n={r['n']:5d}  pass {r['pass']:.2f}->{s['pass']:.2f}  "
                      f"clubs {r['bids_clubs']:.2f}->{s['bids_clubs']:.2f}  "
                      f"diamonds {r['bids_diamonds']:.2f}->{s['bids_diamonds']:.2f}  "
                      f"3-level/2NT+ {r['game_or_higher_or_2nt+']:.2f}->{s['game_or_higher_or_2nt+']:.2f}")
    first = listeners[0].split("=")[0]
    summ = {}
    for name in acts:
        summ[name] = {}
        for real in ("1C", "1D"):
            ar, asw = acts[name][real]
            fr, fsw = acts[first][real]
            summ[name][real] = {"changes_when_swapped": float((ar != asw).mean()),
                                f"same_as_{first}_real": float((ar == fr).mean()),
                                f"same_as_{first}_swapped": float((asw == fsw).mean())}
    print(json.dumps(summ, indent=1))
    res["_summary"] = summ
    json.dump(res, open(args.out, "w"), indent=1)


# ---------------------------------------------------------------------------
# contracts: where the partnership plays after a one-of-a-minor opening
# ---------------------------------------------------------------------------

def contracts(args):
    from concepts.auction_obs import auction_from_obs
    from evaluation.common import load_test_split

    for split in args.splits:
        pos = load_test_split(split)
        last = np.r_[np.flatnonzero(np.diff(pos.deal)), len(pos) - 1]
        res = collections.defaultdict(collections.Counter)
        for i in last:
            a = int(pos.action[i])
            calls, _ = auction_from_obs(pos.obs[i])
            calls = calls + [["P", "X", "XX"][a] if a < 3 else None]
            if calls[-1] is None:
                continue
            k = _first_call(calls)
            if k >= len(calls) or calls[k] not in ("1C", "1D"):
                continue
            j, c = [(j, c) for j, c in enumerate(calls) if c[0].isdigit()][-1]
            side = "openers" if (j - k) % 2 == 0 else "opponents"
            res[calls[k]][(side, c[1:])] += 1
        print("==", split)
        for op, cnt in res.items():
            n = sum(cnt.values())
            ours = {s: v / n for (sd, s), v in cnt.items() if sd == "openers"}
            theirs = sum(v for (sd, _), v in cnt.items() if sd == "opponents") / n
            print(f"  {op} n={n}  openers declare: "
                  + ", ".join(f"{s} {ours.get(s, 0):.1%}" for s in ["C", "D", "H", "S", "NT"])
                  + f"  opponents {theirs:.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("artificial", help="natural vs artificial suit bids on learned cards")
    p.add_argument("wb5_card")
    p.add_argument("cards", nargs="+")
    p.add_argument("--out", default="runs/bid_meaning/natural.json")
    p.set_defaults(fn=artificial)
    p = sub.add_parser("swap", help="the message swap at partner's reply to 1C/1D")
    p.add_argument("split", help="a prepared split of the model's own games")
    p.add_argument("listeners", help="';'-separated model names or name=checkpoint; the first is the reference")
    p.add_argument("out")
    p.set_defaults(fn=swap)
    p = sub.add_parser("contracts", help="final contracts after a one-of-a-minor opening")
    p.add_argument("splits", nargs="+")
    p.set_defaults(fn=contracts)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
