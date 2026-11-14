"""What a call means in its exact auction, like a bridge site's "Meaning" box: WBridge5's measured HCP
and suit-length ranges and forcing status (call_meanings.json.gz, never invented for rare auctions), and the
agent's own meaning (random hands weighted by its probability of its earlier calls)."""
from __future__ import annotations

import functools
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from explain.counterfactual import softmax_masked

HERE = Path(__file__).resolve().parent
TABLE_PATH = HERE / "call_meanings.json.gz"
SUITS = "SHDC"
SEATS = "NESW"
NB = ("P", "X", "XX")
MIN_N = 20

# partner passed the call (after a pass by the next opponent) ...
FORCING_MAX = 0.02      # ... in at most 2% of the cases: forcing
NEARLY_FORCING_MAX = 0.06  # ... up to 6%: nearly forcing (a takeout double, passed for penalties 6%;
                           #     a 1D opening, passed 9% by a responder too weak to answer, is not)
SIGNOFF_MIN = 0.90      # ... in at least 90%: to play


@functools.lru_cache(maxsize=1)
def table() -> Dict[str, Any]:
    if not TABLE_PATH.exists():
        return {}
    with gzip.open(TABLE_PATH, "rt") as f:
        return json.load(f)


def auction_key(calls: Sequence[str]) -> str:
    """The auction so far, from the dealer: ["1D", "2C", "2H", "3C", "P"] -> "1D 2C 2H 3C P"."""
    return " ".join({"Pass": "P", "Dbl": "X", "RDbl": "XX"}.get(c, c).replace("N", "NT").replace("NTT", "NT")
                    for c in calls)


def _strip_leading_passes(calls: Sequence[str]) -> List[str]:
    out, opened = [], False
    for c in calls:
        if c not in ("P", "Pass") or opened:
            opened = True
            out.append(c)
    return out


def wb5_entry(calls_before: Sequence[str], call: str) -> Optional[Dict[str, Any]]:
    """WBridge5's measured meaning of `call` after `calls_before` (from the dealer), or None."""
    T = table()
    if not T:
        return None
    call = auction_key([call])
    e = (T["exact"].get(auction_key(calls_before)) or {}).get(call)
    if e:
        return dict(e, match="exact")
    e = (T["no_leading_passes"].get(auction_key(_strip_leading_passes(calls_before))) or {}).get(call)
    if e:
        return dict(e, match="no_leading_passes")
    return None


def wb5_total(calls_before: Sequence[str], match: str = "exact") -> Optional[int]:
    """How many hands reached this auction (every call made after it, rare ones too), from the table
    `wb5_entry` matched in; None when the table has no totals."""
    T = table()
    tot = (T.get("totals") or {}).get(match)
    if tot is None:
        return None
    key = calls_before if match == "exact" else _strip_leading_passes(calls_before)
    return tot.get(auction_key(key))


# ---------------------------------------------------------------------------
# rendering, Funbridge style
# ---------------------------------------------------------------------------

def length_text(lo: int, hi: int) -> str:
    if lo == hi:
        return f"exactly {lo} card{'s' if lo != 1 else ''}"
    if lo <= 0:
        return f"at most {hi} card{'s' if hi != 1 else ''}"
    if hi >= 9:
        return f"at least {lo} cards"
    return f"{lo} to {hi} cards"


def hcp_text(lo: int, hi: int) -> str:
    return f"{lo} HCP" if lo == hi else f"{lo} to {hi} HCP"


def forcing_status(e: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """{"label", "rate", "n"} from how often partner passed the call, or None (not measured)."""
    pp = e.get("partner_passed")
    if not pp or pp[1] < 10:
        return None
    rate = pp[0] / pp[1]
    if rate <= FORCING_MAX:
        label = "forcing"
    elif rate <= NEARLY_FORCING_MAX:
        label = "nearly forcing"
    elif rate >= SIGNOFF_MIN:
        label = "to play"
    else:
        label = "non-forcing"
    return {"label": label, "rate": round(rate, 3), "n": pp[1]}


def summarize(e: Dict[str, Any], source: str) -> Dict[str, Any]:
    """A table/agent entry -> the shape the interface shows."""
    lo, mid, hi = e["hcp"]
    suits = []
    for s, (a, m, b) in zip(SUITS, e["len"]):
        suits.append({"suit": s, "min": int(a), "median": int(m), "max": int(b), "text": length_text(int(a), int(b))})
    out = {"source": source, "n": e.get("n"), "hcp": [int(lo), int(mid), int(hi)], "hcp_text": hcp_text(int(lo), int(hi)),
           "suits": suits}
    if e.get("match"):
        out["match"] = e["match"]
    if e.get("n_eff") is not None:
        out["n_eff"] = e["n_eff"]
    f = forcing_status(e)
    if f:
        out["forcing"] = f
    return out


def describe(calls_before: Sequence[str], call: str, frame_like: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything known about `call` here: the WBridge5 measurement and the
    teacher's phrase (when a frame-like context -- table, dealer -- is given)."""
    out: Dict[str, Any] = {"call": call}
    e = wb5_entry(calls_before, call)
    if e:
        out["wb5"] = summarize(e, "WBridge5")
        total = wb5_total(calls_before, e.get("match", "exact"))
        if total and total >= e["n"]:   # "1,370 times" alone has no scale: "10% of 13,361"
            out["wb5"]["of"] = int(total)
    if frame_like is not None:
        out["phrase"] = meaning_phrase_for(frame_like, call)
    return out


_PROBE_HANDS = ("S:AK32 H:K32 D:Q32 C:432", "S:J8765 H:Q2 D:K43 C:Q32", "S:2 H:AKJ98 D:AQ43 C:K32")


def meaning_phrase_for(frame_like: Dict[str, Any], call: str) -> Optional[str]:
    """The teacher's meaning phrase for `call` in this auction when it is the
    same for three different hands (a phrase that depends on the hand is not
    the call's meaning). frame_like: {"table", "auction_before", "dealer"}."""
    from nlg.followup_questions import _alt_frame, _meaning, decode_hcp
    phrases = set()
    for hs in _PROBE_HANDS:
        fr = _alt_frame(dict(frame_like, hand_str=hs, hcp=decode_hcp(hs), deal_id=frame_like.get("deal_id", 0),
                             step=frame_like.get("step", 0)), call)
        phrases.add(_meaning(fr))
    return phrases.pop() if len(phrases) == 1 else None


# ---------------------------------------------------------------------------
# the agent's own meaning: importance-weighted random hands
# ---------------------------------------------------------------------------

_BIT_RANK = np.arange(52) // 4
_BIT_SUIT = 3 - np.arange(52) % 4                  # index into "SHDC"
_BIT_HCP = np.where(_BIT_RANK >= 9, _BIT_RANK - 8, 0).astype(np.float64)
HAND_START = 428                                    # the last 52 of the 480 observation bits


def random_hands(n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 52) 0/1 rows, 13 random cards each (observation bit order)."""
    idx = np.argsort(rng.random((n, 52)), axis=1)[:, :13]
    bits = np.zeros((n, 52), np.float32)
    np.put_along_axis(bits, idx, 1.0, axis=1)
    return bits


def _weighted_quantiles(values: np.ndarray, w: np.ndarray, qs=(0.05, 0.5, 0.95)) -> List[int]:
    order = np.argsort(values, kind="stable")
    v, cw = values[order], np.cumsum(w[order])
    cw /= cw[-1]
    return [int(v[min(len(v) - 1, int(np.searchsorted(cw, q)))]) for q in qs]


def model_meanings(policy_batch, history: List[Tuple[np.ndarray, np.ndarray, int]], obs_now: np.ndarray,
                   legal_now: np.ndarray, n: int = 12000, seed: int = 0, min_eff: float = 30.0,
                   batch: int = 4000) -> Dict[int, Dict[str, Any]]:
    """{action: entry} for every legal call: the hands this model would make it
    with here. history: the bidder's OWN earlier decisions (obs, legal, action),
    whose probabilities weight each random hand. policy_batch(obs (B, 480)) ->
    logits (B, 38). Entries with an effective sample size below min_eff are
    left out (too few hands to say what the call shows)."""
    rng = np.random.default_rng(seed)
    hands = random_hands(n, rng)
    logw = np.zeros(n)
    for obs_t, legal_t, a_t in history:
        for i in range(0, n, batch):
            ob = np.repeat(np.asarray(obs_t, np.float32)[None, :480], len(hands[i:i + batch]), 0)
            ob[:, HAND_START:HAND_START + 52] = hands[i:i + batch]
            p = softmax_masked(np.asarray(policy_batch(ob), np.float64), np.asarray(legal_t, bool))
            logw[i:i + batch] += np.log(np.maximum(p[:, a_t], 1e-12))
    P = np.zeros((n, len(legal_now)))
    for i in range(0, n, batch):
        ob = np.repeat(np.asarray(obs_now, np.float32)[None, :480], len(hands[i:i + batch]), 0)
        ob[:, HAND_START:HAND_START + 52] = hands[i:i + batch]
        P[i:i + batch] = softmax_masked(np.asarray(policy_batch(ob), np.float64), np.asarray(legal_now, bool))
    base = np.exp(logw - logw.max())
    hcp = hands @ _BIT_HCP
    lengths = np.stack([hands[:, _BIT_SUIT == s].sum(1) for s in range(4)], 1)
    out = {}
    for a in np.flatnonzero(legal_now):
        w = base * P[:, a]
        if w.sum() <= 0:
            continue
        n_eff = float(w.sum() ** 2 / (w ** 2).sum())
        if n_eff < min_eff:
            continue
        out[int(a)] = {"n_eff": round(n_eff, 1), "share": float(w.sum() / max(base.sum(), 1e-300)),
                       "hcp": _weighted_quantiles(hcp, w),
                       "len": [_weighted_quantiles(lengths[:, s], w) for s in range(4)]}
    return out
