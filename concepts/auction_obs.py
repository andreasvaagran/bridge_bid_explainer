"""Rebuilds the auction so far from one 480-bit observation (pgx / OpenSpiel layout), relative to the
observer: 0 = observer, 1 = LHO, 2 = partner, 3 = RHO. Bits [0:4] vulnerability, [4:8] opening passes,
[8:428] the 35 bids with their doubles and redoubles, [428:480] the hand."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

_STRAINS = ["C", "D", "H", "S", "NT"]
BID_NAMES = [f"{lvl}{s}" for lvl in range(1, 8) for s in _STRAINS]  # 35 bids, index = pgx bid index


def auction_from_obs(obs) -> Tuple[List[str], int]:
    """Returns (calls, dealer) with calls in pgx notation ("P", "X", "XX",
    "1C".."7NT") and `dealer` as a seat RELATIVE to the observer (0 =
    observer), chosen so AuctionContext(bids=calls, dealer=dealer) puts the
    observer on turn. The rule engine only uses relative seats, so this
    classifies identically to a context built from the real dealer."""
    o = np.asarray(obs).reshape(-1)[:480] > 0.5
    opening_passes = int(o[4:8].sum())
    recorded: List[Tuple[int, str]] = []  # (relative seat, call) for non-pass calls, in order
    # bid b's 12 bits are [bid by rel p (4)][double (4)][redouble (4)]; np.nonzero lists them in (bid, kind, seat)
    # order, so the first entry of each (bid, kind) is its lowest seat -- what a scan of each block of 4 would find
    # (one nonzero over the block instead of 105 small ones: the PPO concept step decodes 32,768 of these per update)
    last = None
    for b, kind, seat in zip(*(x.tolist() for x in np.nonzero(o[8:428].reshape(35, 3, 4)))):
        if (b, kind) != last:
            last = (b, kind)
            recorded.append((seat, (BID_NAMES[b], "X", "XX")[kind]))

    calls: List[str] = []
    if not recorded:
        calls = ["P"] * opening_passes
    else:
        calls = ["P"] * opening_passes
        prev = None
        for seat, call in recorded:
            if prev is not None:
                calls += ["P"] * ((seat - prev - 1) % 4)
            calls.append(call)
            prev = seat
        calls += ["P"] * ((0 - prev - 1) % 4)  # passes until the observer is on turn
    dealer = (-len(calls)) % 4
    return calls, dealer
