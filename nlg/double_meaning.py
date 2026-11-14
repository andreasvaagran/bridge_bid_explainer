#!/usr/bin/env python3
"""What a double means in WBridge5's system, named by what partner does with it: penalty if left in
>= 60% of the time, competitive if <= 30% (double_stats.json, rebuilt with --build)."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import functools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
STATS_PATH = Path(__file__).resolve().parent / "double_stats.json"
MIN_N = 30

_PASS, _DBL, _RDBL = ("P", "Pass"), ("X", "Dbl"), ("XX", "RDbl")


def _norm(call: str) -> str:
    if call in _PASS:
        return "P"
    if call in _DBL:
        return "X"
    if call in _RDBL:
        return "XX"
    return call.replace("NT", "N")


def shape_key(calls: Iterable[Tuple[str, str]]) -> str:
    """calls: (who, call) with who in m/p/o, in auction order, before the call."""
    calls = [(w, _norm(c)) for w, c in calls]
    while calls and calls[0][1] == "P":
        calls.pop(0)
    owner: Dict[str, str] = {}
    out: List[str] = []
    for who, c in calls:
        if c in ("P", "X", "XX"):
            out.append(who + c)
        elif c.endswith("N"):
            out.append(f"{who}{c[0]}N")
        else:
            s = c[1]
            tag = "new" if s not in owner else "=" + owner[s]
            owner.setdefault(s, who)
            out.append(f"{who}{c[0]}{tag}")
    return " ".join(out)


def frame_key(frame: Dict[str, Any]) -> str:
    t = frame.get("table") or {}
    you, partner = t.get("you"), t.get("partner")
    return shape_key((("m" if b["seat"] == you else "p" if b["seat"] == partner else "o"), b["bid"])
                     for b in frame.get("auction_before") or [])


@functools.lru_cache(maxsize=1)
def _stats() -> Dict[str, List[int]]:
    try:
        return json.loads(STATS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def leave_in_rate(frame: Dict[str, Any]) -> Optional[Tuple[float, int]]:
    """(fraction of WBridge5 doubles in this shape that partner left in, n),
    or None when the shape was seen fewer than MIN_N times."""
    left, n = _stats().get(frame_key(frame), (0, 0))
    return (left / n, n) if n >= MIN_N else None


# ---------------------------------------------------------------------------

def build(data_path: Path, out: Path, max_deals: int) -> Dict[str, List[int]]:
    """Scan OpenSpiel-format WBridge5 deals (52 dealing actions, then the
    auction: 52 = pass, 53 = double, 54 = redouble, 55 + 5*(level-1) + strain)."""
    strains = "CDHSN"

    def call(a: int) -> str:
        if a == 52:
            return "P"
        if a == 53:
            return "X"
        if a == 54:
            return "XX"
        b = a - 55
        return f"{b // 5 + 1}{strains[b % 5]}"

    stats: Dict[str, List[int]] = {}
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= max_deals:
                break
            auction = []
            for a in map(int, line.split()[52:]):
                if a < 52:
                    break
                auction.append(call(a))
            for j, c in enumerate(auction):
                if c != "X" or j + 2 >= len(auction) or auction[j + 1] != "P":
                    continue
                me = j % 4
                who = lambda k: "m" if k % 4 == me else "p" if (k - me) % 4 == 2 else "o"  # noqa: E731
                key = shape_key((who(k), auction[k]) for k in range(j))
                s = stats.setdefault(key, [0, 0])
                s[0] += auction[j + 2] == "P"
                s[1] += 1
    kept = {k: v for k, v in sorted(stats.items()) if v[1] >= 20}
    out.write_text(json.dumps(kept, separators=(",", ":")))
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--data", default=str(_REPO / "data" / "wbridge5" / "official_train.txt"))
    ap.add_argument("--max_deals", type=int, default=2_000_000)
    ap.add_argument("--out", default=str(STATS_PATH))
    args = ap.parse_args()
    if args.build:
        kept = build(Path(args.data), Path(args.out), args.max_deals)
        print(f"{len(kept)} auction shapes with 20+ doubles -> {args.out}")


if __name__ == "__main__":
    main()
