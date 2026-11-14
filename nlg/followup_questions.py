#!/usr/bin/env python3
"""Follow-up questions about one explained decision ("why not 3NT?", "what if I had the ace of spades instead
of the four of clubs?", "what does partner's 2C show?", "how sure are you?"). Every answer is computed:
  parse   question -> a query in a small schema (QUERY_TYPES; `parse_rule` is the keyword fallback);
  facts   query -> facts from the agent and the rules (`facts_for`), never from the language model;
  answer  facts -> two or three sentences, checked by `check_answer`.
The teacher templates are the training targets; --build writes the dataset, --calibrate the trust table."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import functools
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from concepts import gloss_for_concept
from concepts.hand import RANKS, decode_hand, replace_card
from explain.counterfactual import FULL_DECK, HCP_MAP, softmax_masked
from nlg.explanation_frames import BIDS
from nlg.explanation_inputs import (SEAT_NAME, SUIT_NAME, _gt, bid_category, deviation_notes)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WB5_CARD_PATH = HERE / "wbridge5_card.json"
TRUST_PATH = HERE / "trust_calibration.json"

QUERY_TYPES = ("why_not", "change", "swap", "meaning", "trust", "explain", "other")
CHANGE_KINDS = ("any", "stronger", "weaker", "longer", "shorter")
SUITS = "SHDC"
SUIT_SYM = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
SUIT_SING = {"S": "spade", "H": "heart", "D": "diamond", "C": "club"}
RANK_WORD = {"A": "ace", "K": "king", "Q": "queen", "J": "jack", "T": "ten", "9": "nine", "8": "eight",
             "7": "seven", "6": "six", "5": "five", "4": "four", "3": "three", "2": "two"}
NB = ("P", "X", "XX")
OUT_OF_SCOPE = ("I can only answer questions about this call: why not another call, what would change it, "
                "what an earlier call means, and how sure the agent is.")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def call_name(c: str) -> str:
    return {"P": "Pass", "X": "Double", "XX": "Redouble"}.get(c, c)


def card_text(c: str) -> str:
    """'SA' -> '♠A', 'CT' -> '♣10'."""
    return SUIT_SYM[c[0]] + ("10" if c[1] == "T" else c[1])


def pct(p: float) -> str:
    return "under 1%" if p < 0.005 else f"{round(100 * p)}%"


def shape_str(lengths: Dict[str, int]) -> str:
    return "-".join(str(lengths[s]) for s in SUITS)


def _hcp_of(card: str) -> int:
    return HCP_MAP.get(card[1], 0)


def _swap_cost(out_card: str, in_card: str) -> tuple:
    """Smallest change first: HCP moved, then a change of suit, then rank distance."""
    return (abs(_hcp_of(in_card) - _hcp_of(out_card)), out_card[0] != in_card[0],
            abs(RANKS.index(in_card[1]) - RANKS.index(out_card[1])))


def norm_query(q: Optional[dict]) -> Optional[dict]:
    """Canonical form of a query (for exact-match comparison); None if malformed."""
    if not isinstance(q, dict) or q.get("type") not in QUERY_TYPES:
        return None
    t = q["type"]
    try:
        if t == "why_not":
            return {"type": t, "call": norm_call(q["call"])}
        if t == "change":
            kind = q.get("kind", "any")
            if kind not in CHANGE_KINDS:
                return None
            out = {"type": t, "kind": kind}
            if q.get("call"):  # "what would I need to bid 3NT?": only changes that lead to that call
                out["call"] = norm_call(q["call"])
            if kind in ("longer", "shorter"):
                out["suit"] = str(q["suit"]).upper()[:1]
                if out["suit"] not in SUITS:
                    return None
            return out
        if t == "swap":
            return {"type": t, "out": norm_card(q["out"]), "in": norm_card(q["in"])}
        if t == "meaning":
            return {"type": t, "seat": str(q["seat"]).upper()[:1], "call": norm_call(q["call"])}
        return {"type": t}
    except (KeyError, TypeError, ValueError):
        return None


def norm_call(c: Any) -> str:
    s = str(c).upper().replace(" ", "").replace("NOTRUMP", "NT").replace("N", "NT").replace("NTT", "NT")
    s = {"PASS": "P", "DOUBLE": "X", "DBL": "X", "REDOUBLE": "XX", "RDBL": "XX"}.get(s, s)
    for sym, letter in SUIT_SYM.items():
        s = s.replace(letter, sym)
    if s not in BIDS:
        raise ValueError(f"not a call: {c!r}")
    return s


def norm_card(c: Any) -> str:
    s = str(c).upper().replace("10", "T").replace(" ", "")
    for sym, letter in SUIT_SYM.items():
        s = s.replace(letter, sym)
    if len(s) == 2 and s[0] in RANKS and s[1] in SUITS:  # "AS" -> "SA"
        s = s[1] + s[0]
    if len(s) != 2 or s[0] not in SUITS or s[1] not in RANKS:
        raise ValueError(f"not a card: {c!r}")
    return s


def query_json(q: dict) -> str:
    return json.dumps(q, ensure_ascii=False)


def parse_query_text(text: str) -> Optional[dict]:
    """The query in a language model's reply (the first JSON object), normalised; None if there is none."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return norm_query(json.loads(m.group(0)))
    except (ValueError, TypeError):
        return None


def describe_query(q: dict) -> str:
    """The query in words, shown with the answer ("understood as: why not 3NT")."""
    t = q["type"]
    if t == "why_not":
        return f"why not {call_name(q['call'])}"
    if t == "change" and q.get("call"):
        return f"what the hand would need for {call_name(q['call'])}"
    if t == "change":
        return {"any": "which change to the hand would change the call", "stronger": "a stronger hand",
                "weaker": "a weaker hand", "longer": f"one more {SUIT_SING.get(q.get('suit'), '')}",
                "shorter": f"one {SUIT_SING.get(q.get('suit'), '')} fewer"}[q["kind"]]
    if t == "swap":
        return f"the {card_text(q['in'])} instead of the {card_text(q['out'])}"
    if t == "meaning":
        return f"what {SEAT_NAME.get(q['seat'], q['seat'])}'s {call_name(q['call'])} means"
    return {"trust": "how sure the agent is", "explain": "why this call", "other": "not about this call"}[t]


# ---------------------------------------------------------------------------
# measured tables: WBridge5's own calls, and how often uncertain decisions err
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def wb5_card() -> Dict[str, Any]:
    return json.loads(WB5_CARD_PATH.read_text())["entries"] if WB5_CARD_PATH.exists() else {}


@functools.lru_cache(maxsize=None)
def trust_table() -> Dict[str, Any]:
    return json.loads(TRUST_PATH.read_text()) if TRUST_PATH.exists() else {}


def trust_rate(model_key: Optional[str], certainty: float) -> Optional[Tuple[float, int, float]]:
    """(rate of calls that differ from WBridge5's among test decisions whose least
    certain top concept is in the same bin, n in that bin, overall rate)."""
    t = trust_table().get(model_key or "") or trust_table().get("SL independent")
    if not t:
        return None
    for lo, hi, rate, n in t["bins"]:
        if lo <= certainty < hi or (hi >= 1.0 and certainty >= lo):
            return rate, n, t["overall"]
    return None


# ---------------------------------------------------------------------------
# one decision, with everything the facts need
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    frame: Dict[str, Any]
    obs: np.ndarray                     # (480,) float32
    legal: np.ndarray                   # (38,) bool
    probs: np.ndarray                   # (38,) the agent's probabilities
    policy: Callable                    # obs batch -> policy logits
    acts: Optional[np.ndarray] = None   # concept activations (concept models)
    W: Optional[np.ndarray] = None      # concepts->call weights (concept models)
    names: Optional[List[str]] = None   # concept names, vocabulary order
    model_key: Optional[str] = None     # evaluation.common name, for TRUST_TABLE

    @property
    def action(self) -> int:
        return BIDS.index(self.frame["bid"])

    @functools.cached_property
    def hand(self):
        return decode_hand(self.obs)

    @functools.cached_property
    def swaps(self):
        """All 507 one-card swaps: (pairs, new probabilities, new calls, new hands' hcp, new lengths)."""
        held = [s + r for s, rs in self.hand.cards.items() for r in rs]
        held_set = set(held)
        pairs = [(c, u) for c in held for u in FULL_DECK if u not in held_set]
        batch = np.stack([replace_card(self.obs, c, u) for c, u in pairs])
        p = softmax_masked(np.asarray(self.policy(batch), np.float64), self.legal[None])
        calls = p.argmax(1)
        return pairs, p, calls


def decision_from_record(bidder, obs, legal, frame: Dict[str, Any], model_key: Optional[str] = None) -> Decision:
    """From an nlg.decision_record.Bidder (GUI, nlg.explain_deals)."""
    out = bidder.outputs(obs)
    probs = softmax_masked(np.asarray(out["policy_logits"], np.float64), np.asarray(legal, bool))
    acts = W = names = None
    if bidder.kind == "concept" and bidder.enricher is not None and bidder.enricher.W is not None:
        acts, W, names = np.asarray(out["concept_acts"]), bidder.enricher.W, list(bidder.ck.hard_concepts)
    return Decision(frame, np.asarray(obs, np.float32)[:480], np.asarray(legal, bool), probs,
                    bidder.policy_logits_batch, acts, W, names, model_key)


# ---------------------------------------------------------------------------
# facts
# ---------------------------------------------------------------------------

def _alt_frame(frame: Dict[str, Any], call: str) -> Dict[str, Any]:
    """The decision as if `call` had been made: rule-true concepts as evidence
    (equal weight), so the teacher's meaning phrase describes THAT call."""
    fr = dict(frame, bid=call, runner_up=None)
    gt = _gt(fr)
    fr["concept_evidence"] = [{"name": n, "act": 1.0, "true": True, "contribution": 1.0} for n, v in gt.items() if v]
    return fr


def _meaning(frame: Dict[str, Any]) -> Optional[str]:
    from nlg.teacher import situation
    try:
        return situation(frame)["meaning"]
    except Exception:
        return None


def _illegal_reason(frame: Dict[str, Any], call: str) -> str:
    prior = frame.get("auction_before") or []
    real = [b for b in prior if b["bid"] not in NB]
    if call == "X":
        return "there is no opponents' bid to double"
    if call == "XX":
        return "only a doubled bid can be redoubled"
    last = real[-1]["bid"] if real else None
    return f"the auction is already at {last}, so {call} would be too low" if last else f"{call} is not allowed here"


def _evidence_vs(d: Decision, alt: int) -> List[str]:
    """The teacher's hand-evidence clauses for the chosen call over `alt`: the
    reasons are re-ranked by their exact contribution to THIS margin
    (a_i (W[i, call] - W[i, alt])) when the agent has a concepts->call layer."""
    from nlg.teacher import Ctx, evidence, _fallback_evidence, with_context
    frame = d.frame
    if d.acts is not None:
        gt = _gt(frame)
        contrib = d.acts * (d.W[:, d.action] - d.W[:, alt])
        keep = [i for i in range(len(d.names)) if d.acts[i] >= 0.5 or gt.get(d.names[i])]
        keep.sort(key=lambda i: -abs(contrib[i]))
        frame = dict(frame, runner_up={"bid": BIDS[alt], "p": round(float(d.probs[alt]), 3)},
                     concept_evidence=[{"name": d.names[i], "act": round(float(d.acts[i]), 3),
                                        "true": bool(gt.get(d.names[i])), "contribution": round(float(contrib[i]), 3)}
                                       for i in keep[:20]])
    c = Ctx(frame, random.Random(f"{frame.get('deal_id')}:{frame.get('step')}:{alt}"))
    try:
        return with_context(c, evidence(c) or _fallback_evidence(c))
    except Exception:
        return []


def _swap_fact(d: Decision, k: int) -> Dict[str, Any]:
    pairs, p, calls = d.swaps
    out_c, in_c = pairs[k]
    h = decode_hand(replace_card(d.obs, out_c, in_c))
    new = int(calls[k])
    return {"out": out_c, "in": in_c, "new_call": BIDS[new], "new_p": round(float(p[k, new]), 3),
            "hcp": int(h.hcp), "shape": shape_str(h.lengths)}


def _swap_filter(kind: str, suit: Optional[str]) -> Callable[[str, str], bool]:
    if kind == "stronger":
        return lambda o, i: _hcp_of(i) > _hcp_of(o)
    if kind == "weaker":
        return lambda o, i: _hcp_of(i) < _hcp_of(o)
    if kind == "longer":
        return lambda o, i: i[0] == suit and o[0] != suit
    if kind == "shorter":
        return lambda o, i: o[0] == suit and i[0] != suit
    return lambda o, i: True


def _minimal_swaps(d: Decision, accept: Callable[[int], bool], limit: int = 2) -> List[Dict[str, Any]]:
    """Up to `limit` distinct smallest one-card changes accepted by `accept(k)`."""
    pairs, p, calls = d.swaps
    ks = sorted((k for k in range(len(pairs)) if accept(k)), key=lambda k: _swap_cost(*pairs[k]))
    out, seen = [], set()
    for k in ks:
        f = _swap_fact(d, k)
        key = (f["new_call"], f["shape"], f["hcp"])
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
        if len(out) >= limit:
            break
    return out


def _pair_swap(d: Decision, target: int, shortlist: int = 40) -> Optional[Dict[str, Any]]:
    """The smallest TWO-card change that makes the agent choose `target`, when no single
    card does ("what would I need to bid 3NT?" on a 10-count): pairs of the `shortlist`
    one-card swaps that raise the target's probability most, all evaluated exactly."""
    pairs, p, _ = d.swaps
    top = [int(k) for k in np.argsort(-p[:, target])[:shortlist]]
    combos = [(a, b) for i, a in enumerate(top) for b in top[i + 1:]
              if pairs[a][0] != pairs[b][0] and pairs[a][1] != pairs[b][1]]
    if not combos:
        return None
    batch = np.stack([replace_card(replace_card(d.obs, *pairs[a]), *pairs[b]) for a, b in combos])
    pp = softmax_masked(np.asarray(d.policy(batch), np.float64), d.legal[None])
    hit = [i for i in range(len(combos)) if int(pp[i].argmax()) == target]
    if not hit:
        return None
    i = min(hit, key=lambda i: tuple(x + y for x, y in zip(_swap_cost(*pairs[combos[i][0]]), _swap_cost(*pairs[combos[i][1]]))))
    a, b = combos[i]
    h = decode_hand(batch[i])
    return {"swaps": [{"out": pairs[a][0], "in": pairs[a][1]}, {"out": pairs[b][0], "in": pairs[b][1]}],
            "new_call": BIDS[target], "new_p": round(float(pp[i, target]), 3), "hcp": int(h.hcp),
            "shape": shape_str(h.lengths)}


def _wb5_here(frame: Dict[str, Any], call: str) -> Optional[Dict[str, Any]]:
    """WBridge5's measured HCP range for `call` at this exact position (nlg.call_meaning),
    when it made the call here at least 30 times, and how often it chose the call here: "782
    times" alone has no scale, "18% of 4,300 hands" sits next to the agent's 36%."""
    from nlg.call_meaning import wb5_entry, wb5_total
    before = [b["bid"] for b in frame.get("auction_before") or []]
    e = wb5_entry(before, call)
    if not e or e["n"] < 30:
        return None
    total = wb5_total(before, e.get("match", "exact"))
    if not total or total < e["n"]:
        return None
    return {"hcp": [int(e["hcp"][0]), int(e["hcp"][2])], "n": int(e["n"]), "of": int(total),
            "share": round(e["n"] / total, 3)}


def _wb5_share_clause(c: str, w: Dict[str, Any]) -> str:
    """'Double was chosen here 18% of the time (782 of 4,300 hands), mostly with 9 to 11 HCP'
    (the range is the 5th to 95th percentile of those hands)."""
    return (f"{c} was chosen here {pct(w['share'])} of the time ({w['n']:,} of {w['of']:,} hands), "
            f"mostly with {w['hcp'][0]} to {w['hcp'][1]} HCP")


def _meaning_of_earlier(frame: Dict[str, Any], seat: str, call: str) -> Dict[str, Any]:
    prior = frame.get("auction_before") or []
    idx = [j for j, b in enumerate(prior) if b["seat"] == seat and b["bid"] == call]
    if not idx:
        made_by = [b["seat"] for b in prior if b["bid"] == call]
        return {"found": False, "seat": seat, "call": call, "made_by": made_by[-1] if made_by else None}
    j = idx[-1]
    seats = "NESW"
    i = seats.index(seat)
    sub = {"table": {"you": seat, "partner": seats[(i + 2) % 4], "opponents": [seats[(i + 1) % 4], seats[(i + 3) % 4]]},
           "auction_before": prior[:j], "bid": call, "dealer": frame.get("dealer"),
           "vulnerability": frame.get("vulnerability"), "deal_id": frame.get("deal_id"), "step": j}
    category = bid_category(dict(sub, hand_str="S:AK32 H:K32 D:Q32 C:432", hcp=12))
    # the meaning must not depend on the (unknown) hand: the same phrase for three probe hands
    phrases = set()
    for hs in ("S:AK32 H:K32 D:Q32 C:432", "S:J8765 H:Q2 D:K43 C:Q32", "S:2 H:AKJ98 D:AQ43 C:K32"):
        fr = _alt_frame(dict(sub, hand_str=hs, hcp=decode_hcp(hs)), call)
        phrases.add(_meaning(fr))
    phrase = phrases.pop() if len(phrases) == 1 else None
    suit = call[-1] if call[:1].isdigit() and call[-1] in SUITS else None
    # this exact auction first (nlg.call_meaning: "North's 3C ... 2 to 9 HCP, 3 to 4 clubs, 141 times"),
    # the 55-situation card only without it (coarser: "2 to 14 HCP" for the same 3C)
    from nlg.call_meaning import wb5_entry
    x = wb5_entry([b["bid"] for b in prior[:j]], call)
    if x and x["n"] >= 30:
        si = SUITS.index(suit) if suit else None
        wb5 = {"hcp": [int(x["hcp"][0]), int(x["hcp"][2])], "n": int(x["n"]), "exact": True, "suit": suit,
               "min_len": int(x["len"][si][0]) if suit else None, "max_len": int(x["len"][si][2]) if suit else None}
    else:
        e = wb5_card().get(f"{category}|{call}")
        if e and e["hcp"][1] - e["hcp"][0] > 12:  # a situation too coarse to say what THIS call shows
            e = None
        wb5 = {"hcp": e["hcp"], "min_len": e.get("min_len") if suit else None, "suit": suit, "n": e["n"]} if e else None
    return {"found": True, "seat": seat, "call": call,
            "situation": SITUATION_PHRASE.get(category, category.replace("_", " ")), "meaning": phrase, "wb5": wb5}


SITUATION_PHRASE = {
    "advancer_cuebid": "a cuebid by the partner of an overcaller", "advancer_double": "a double by the partner of an overcaller",
    "advancer_new_suit": "a new suit by the partner of an overcaller",
    "advancer_notrump": "a notrump bid by the partner of an overcaller", "advancer_pass": "a pass by the partner of an overcaller",
    "advancer_raise": "a raise of partner's overcall", "competitive_later_bid": "a bid later in a contested auction",
    "competitive_later_double": "a double later in a contested auction", "competitive_later_pass": "a pass later in a contested auction",
    "competitive_later_support": "support for partner later in a contested auction", "opener_double": "a double by the opening bidder",
    "opener_new_suit": "a new suit by the opening bidder", "opener_pass": "a pass by the opening bidder",
    "opener_pass_competitive": "a pass by the opening bidder in a contested auction",
    "opener_raises_responder": "the opening bidder's raise of partner's suit", "opener_rebid_notrump": "the opening bidder's notrump rebid",
    "opener_rebids_own_suit": "the opening bidder's rebid of the own suit", "opening_1_major": "an opening bid of one of a major",
    "opening_1_minor": "an opening bid of one of a minor", "opening_1nt": "a 1NT opening", "opening_2c_strong": "the strong 2C opening",
    "opening_2nt": "a 2NT opening", "opening_pass": "a pass in opening position", "opening_preempt": "a preemptive opening",
    "opening_weak_two": "a weak-two opening", "overcall_cuebid": "a cuebid of the opponents' suit", "overcall_jump": "a jump overcall",
    "overcall_notrump": "a notrump overcall", "overcall_suit": "a simple overcall", "pass_over_opponents_opening": "a pass over the opponents' opening",
    "redouble": "a redouble", "responder_bid_competitive": "a bid by the responder in a contested auction",
    "responder_later_notrump": "a later notrump bid by the responder", "responder_later_other": "a later bid by the responder",
    "responder_later_pass": "a later pass by the responder", "responder_later_support": "later support by the responder",
    "responder_negative_double": "a negative double", "responder_pass_competitive": "a pass by the responder in a contested auction",
    "responder_raise_competitive": "a raise by the responder in a contested auction", "response_1nt_other": "a response to 1NT",
    "response_1nt_pass": "a pass of partner's 1NT", "response_1nt_stayman": "Stayman over 1NT", "response_1nt_transfer": "a transfer over 1NT",
    "response_game_raise": "a raise to game", "response_jump_raise": "a jump raise", "response_jump_shift": "a jump shift",
    "response_new_suit_1_level": "a new suit at the one level", "response_new_suit_2_level": "a new suit at the two level",
    "response_notrump": "a notrump response", "response_pass": "a pass of partner's opening", "response_simple_raise": "a simple raise",
    "slam_bid": "a slam bid", "slam_blackwood_answer": "an answer to Blackwood", "slam_blackwood_ask": "Blackwood",
    "takeout_double": "a takeout double"}


def decode_hcp(hand_str: str) -> int:
    return sum(HCP_MAP.get(ch, 0) for part in hand_str.split() for ch in part.split(":")[1])


def facts_for(d: Decision, q: dict) -> Dict[str, Any]:
    """The facts that answer query `q` about decision `d` (all computed, never generated)."""
    frame, t = d.frame, q["type"]
    chosen = frame["bid"]
    f: Dict[str, Any] = {"type": t, "chosen": chosen, "p_chosen": round(float(d.probs[d.action]), 3)}
    if t == "why_not":
        c = q["call"]
        f["call"] = c
        if c == chosen:
            from nlg.teacher import explain
            f.update(same=True, explanation=explain(frame))
            return f
        ic = BIDS.index(c)
        if not d.legal[ic]:
            f.update(legal=False, reason=_illegal_reason(frame, c))
            return f
        alt = _alt_frame(frame, c)
        order = np.argsort(-d.probs)
        f.update(legal=True, p_call=round(float(d.probs[ic]), 3), rank=int(np.flatnonzero(order == ic)[0]) + 1,
                 runner_up=bool(order[1] == ic), meaning=_meaning(alt), notes=deviation_notes(alt),
                 evidence=_evidence_vs(d, ic))
        cf = _minimal_swaps(d, lambda k: int(d.swaps[2][k]) == ic, limit=1)
        f["swap_to_call"] = cf[0] if cf else None
        w = _wb5_here(frame, c)
        if w:
            f["wb5"] = w
        return f
    if t == "change":
        kind, suit = q.get("kind", "any"), q.get("suit")
        ok = _swap_filter(kind, suit)
        pairs, p, calls = d.swaps
        target = q.get("call")
        if target:
            it = BIDS.index(target)
            f.update(kind=kind, suit=suit, call=target, legal=bool(d.legal[it]))
            if not d.legal[it]:
                f.update(changes=[], reason=_illegal_reason(frame, target))
                return f
            f["changes"] = _minimal_swaps(d, lambda k: int(calls[k]) == it and ok(*pairs[k]), limit=2)
            f["p_call"] = round(float(d.probs[it]), 3)
            if not f["changes"]:  # two cards, else the one-card change that brings the call closest
                f["pair"] = _pair_swap(d, it)
                k = int(np.argmax(p[:, it]))
                if not f["pair"] and p[k, it] > d.probs[it] + 0.005:
                    f["closest"] = dict(_swap_fact(d, k), p_target=round(float(p[k, it]), 3))
            w = _wb5_here(frame, target)
            if w:
                f["wb5"] = w
            return f
        f.update(kind=kind, suit=suit,
                 changes=_minimal_swaps(d, lambda k: int(calls[k]) != d.action and ok(*pairs[k]), limit=2))
        return f
    if t == "swap":
        o, i = q["out"], q["in"]
        f.update(out=o, **{"in": i})
        held = {s + r for s, rs in d.hand.cards.items() for r in rs}
        if o not in held:
            f.update(valid=False, why=f"the hand does not hold the {card_text(o)}")
            return f
        if i in held:
            f.update(valid=False, why=f"the hand already holds the {card_text(i)}")
            return f
        ob = replace_card(d.obs, o, i)
        p = softmax_masked(np.asarray(d.policy(ob[None]), np.float64), d.legal[None])[0]
        h = decode_hand(ob)
        new = int(p.argmax())
        f.update(valid=True, new_call=BIDS[new], new_p=round(float(p[new]), 3), hcp=int(h.hcp),
                 shape=shape_str(h.lengths), same=new == d.action)
        return f
    if t == "meaning":
        f.update(_meaning_of_earlier(frame, q["seat"], q["call"]))
        return f
    if t == "trust":
        order = np.argsort(-d.probs)
        ru = int(order[1]) if d.legal.sum() > 1 else None
        f.update(runner_up=BIDS[ru] if ru is not None else None,
                 p_runner_up=round(float(d.probs[ru]), 3) if ru is not None else None)
        if d.acts is None:
            f["concepts"] = False
            return f
        # the five concepts with the largest |contribution| to the margin over the runner-up, over ALL
        # concepts (descriptors aside), exactly as TRUST_TABLE was measured (calibrate); the frame's stored
        # evidence leaves out inactive concepts, so its least certain one is not the measured one
        gt = _gt(frame)
        if ru is not None and d.W is not None:
            contrib = d.acts * (d.W[:, d.action] - d.W[:, ru])
            ev = [{"name": n, "act": float(d.acts[i]), "contribution": float(contrib[i])}
                  for i, n in enumerate(d.names) if not n.startswith("bid_is_in_")]
        else:
            ev = [e for e in frame.get("concept_evidence") or [] if not e["name"].startswith("bid_is_in_")]
        top = sorted(ev, key=lambda e: -abs(e["contribution"]))[:5]
        if not top:
            f["concepts"] = False
            return f
        least = min(top, key=lambda e: max(e["act"], 1 - e["act"]))
        cert = max(least["act"], 1 - least["act"])
        bad = [e for e in top if e["act"] >= 0.5 and e["contribution"] > 0 and not gt.get(e["name"])]
        f.update(concepts=True, least={"gloss": gloss_for_concept(least["name"]), "act": round(float(least["act"]), 3),
                                       "certainty": round(float(cert), 3)},
                 rule_rejects=[gloss_for_concept(e["name"]) for e in bad][:1])
        r = trust_rate(d.model_key, cert)
        if r:
            f.update(err_rate=round(r[0], 4), err_n=r[1], err_overall=round(r[2], 4))
        return f
    if t == "explain":
        from nlg.teacher import explain
        f["explanation"] = explain(frame)
        return f
    return f  # other


# ---------------------------------------------------------------------------
# the teacher's answer (training target; deterministic from the facts)
# ---------------------------------------------------------------------------

def _clause_join(cl: List[str]) -> str:
    if len(cl) <= 2:
        return ", and ".join(cl)
    return ", ".join(cl[:-1]) + ", and " + cl[-1]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


_LEN_WORD = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
             9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen"}


def change_desc(sw: Dict[str, Any], hcp_before: int) -> str:
    """What the swap changes, in words: 'five spades instead of six' / '2 HCP more'."""
    o, i = sw["out"][0], sw["in"][0]
    lens = [int(x) for x in sw["shape"].split("-")]
    new = dict(zip(SUITS, lens))
    parts = []
    d = sw["hcp"] - hcp_before
    if d:
        parts.append(f"{abs(d)} HCP {'more' if d > 0 else 'less'}")
    if o != i:
        n = new[o]
        parts.append(f"{_LEN_WORD[n]} {SUIT_NAME[o] if n != 1 else SUIT_SING[o]} instead of {_LEN_WORD[n + 1]}")
    return ", ".join(parts) if parts else "a different small card of the same suit"


def _seat_of(frame: Dict[str, Any]) -> str:
    return SEAT_NAME.get((frame.get("table") or {}).get("you"), "The bidder")


def act_phrase(c: str) -> str:
    return {"P": "pass", "X": "double", "XX": "redouble"}.get(c, f"bid {c}")


def _swap_phrase(sw: Dict[str, Any], hcp_before: Optional[int] = None) -> str:
    what = f"{change_desc(sw, hcp_before)}, " if hcp_before is not None else ""
    return f"with the {card_text(sw['in'])} instead of the {card_text(sw['out'])} ({what}{sw['hcp']} HCP, {sw['shape']})"


def teacher_answer(frame: Dict[str, Any], q: dict, f: Dict[str, Any]) -> str:
    t = q["type"]
    chosen = call_name(f["chosen"])
    if t == "why_not":
        c = call_name(f["call"])
        if f.get("same"):
            return f["explanation"]
        if not f.get("legal"):
            return f"{c} is not possible here: {f['reason']}."
        s = [f"{c} was {'the main alternative' if f['runner_up'] else 'possible'}"
             + (f", {f['meaning']}." if f.get("meaning") else ".")]
        if f.get("notes"):
            s.append(f"With this hand it would be unusual: {f['notes'][0]}")
        if f.get("evidence"):
            s.append(f"{chosen} fits the hand better: {_clause_join(f['evidence'][:2])}.")
        s.append(f"The agent gives {c} {pct(f['p_call'])} against {pct(f['p_chosen'])} for {chosen}.")
        if f.get("wb5") and not any("WBridge5 passes here" in n for n in (f.get("notes") or [])[:1]):  # (said once)
            w = f["wb5"]
            s.append(f"In WBridge5's games {_wb5_share_clause(c, w)}; {_seat_of(frame)} has {frame.get('hcp')} HCP.")
        if f.get("swap_to_call"):
            s.append(f"{_cap(_swap_phrase(f['swap_to_call'], frame.get('hcp')))}, it would {act_phrase(f['call'])}.")
        return " ".join(s)
    if t == "change" and f.get("call"):
        c = call_name(f["call"])
        if not f.get("legal"):
            return f"{c} is not possible here: {f['reason']}."
        if f["changes"]:
            a = f["changes"][0]
            s = [f"{_cap(_swap_phrase(a, frame.get('hcp')))}, the agent would {act_phrase(a['new_call'])}."]
            if len(f["changes"]) > 1:
                b = f["changes"][1]
                s.append(f"Another such change: the {card_text(b['in'])} instead of the {card_text(b['out'])} "
                         f"({change_desc(b, frame.get('hcp'))}, {b['hcp']} HCP, {b['shape']}), after which it would "
                         f"{act_phrase(b['new_call'])} too.")
        elif f.get("pair"):
            pr = f["pair"]
            (a1, a2) = pr["swaps"]
            s = [f"No single card change is enough (the agent gives {c} {pct(f['p_call'])} now), but two are: "
                 f"with the {card_text(a1['in'])} instead of the {card_text(a1['out'])} and the {card_text(a2['in'])} "
                 f"instead of the {card_text(a2['out'])} ({pr['hcp']} HCP, {pr['shape']}), the agent would "
                 f"{act_phrase(pr['new_call'])}."]
        else:
            s = [f"No single card change makes the agent {act_phrase(f['call'])}: it gives {c} {pct(f['p_call'])} now."]
            cl = f.get("closest")
            if cl:
                verb = ("still " + act_phrase(cl["new_call"])) if cl["new_call"] == f["chosen"] else act_phrase(cl["new_call"])
                s.append(f"The closest is {_swap_phrase(cl, frame.get('hcp'))}: the agent would {verb}, "
                         f"and {c} rises to {pct(cl['p_target'])}.")
        if f.get("wb5"):
            w = f["wb5"]
            s.append(f"In WBridge5's games {_wb5_share_clause(c, w)}; {_seat_of(frame)} has {frame.get('hcp')} HCP.")
        return " ".join(s)
    if t == "change":
        kind, suit = f["kind"], f.get("suit")
        if not f["changes"]:
            what = {"any": "", "stronger": " that adds strength", "weaker": " that takes strength away",
                    "longer": f" that adds a {SUIT_SING.get(suit, '')}", "shorter": f" that removes a {SUIT_SING.get(suit, '')}"}[kind]
            return f"No single card change{what} moves the agent off {chosen}: it would still bid {chosen}."
        a = f["changes"][0]
        s = [f"{_cap(_swap_phrase(a, frame.get('hcp')))}, the agent would {act_phrase(a['new_call'])} instead of {chosen}."]
        if len(f["changes"]) > 1:
            b = f["changes"][1]
            s.append(f"Another such change: the {card_text(b['in'])} instead of the {card_text(b['out'])} "
                     f"({change_desc(b, frame.get('hcp'))}, {b['hcp']} HCP, {b['shape']}), after which it would "
                     f"{act_phrase(b['new_call'])}.")
        return " ".join(s)
    if t == "swap":
        if not f.get("valid"):
            return f"That swap is not possible: {f['why']}."
        head = _cap(_swap_phrase(f, frame.get("hcp")))
        if f["same"]:
            return f"{head}, the agent would still {act_phrase(f['chosen'])} ({pct(f['new_p'])})."
        return f"{head}, the agent would {act_phrase(f['new_call'])} instead of {chosen} ({pct(f['new_p'])})."
    if t == "meaning":
        who = SEAT_NAME[f["seat"]]
        c = call_name(f["call"])
        if not f.get("found"):
            if f.get("made_by"):
                return f"{who} did not bid {c}; {SEAT_NAME[f['made_by']]} did."
            return f"Nobody has bid {c} in this auction."
        m = f.get("meaning")
        if m and m.startswith("showing "):
            s = [f"{who}'s {c} shows {m[len('showing '):]}."]
        elif m:
            s = [f"{who}'s {c} is {m}."]
        else:
            s = [f"{who}'s {c} is {f['situation']}."]
        w = f.get("wb5")
        if w and w.get("exact"):
            length = ""
            if w.get("suit") and w.get("min_len") is not None:
                lo_, hi_ = w["min_len"], w["max_len"]
                length = (f" and {lo_} to {hi_} {SUIT_NAME[w['suit']]}" if hi_ > lo_ and hi_ < 9
                          else f" and at least {lo_} {SUIT_NAME[w['suit']]}" if lo_ >= 1 else "")
            s.append(f"WBridge5 makes this call, in this exact auction, with {w['hcp'][0]} to "
                     f"{w['hcp'][1]} HCP{length} (the middle 90% of its {w['n']:,} cases).")
        elif w:
            length = (f" and at least {w['min_len']} {SUIT_NAME[w['suit']]}" if w.get("suit") and (w.get("min_len") or 0) >= 3 else "")
            s.append(f"In WBridge5's own games this call is made with {w['hcp'][0]} to {w['hcp'][1]} HCP{length} "
                     f"(95% of {w['n']:,} cases).")
        return " ".join(s)
    if t == "trust":
        ru = (f" and {call_name(f['runner_up'])} {pct(f['p_runner_up'])}" if f.get("runner_up") and f.get("p_runner_up", 0) >= 0.005
              else "")
        s = [f"The agent gives {chosen} {pct(f['p_chosen'])}{ru}" + (", a close decision." if f["p_chosen"] < 0.6 else ".")]
        if not f.get("concepts"):
            s.append("This network has no layer of bridge reasons to inspect, so its probabilities are all it "
                     "reports about its certainty.")
            return " ".join(s)
        least = f["least"]
        # the least certain concept may be one the model rules OUT (judged under 1%): it is not a
        # reason for the call (not "the least certain, that it shows first-round control ..., is judged
        # at under 1%")
        believed = least["act"] >= 0.5
        if least["certainty"] >= 0.9:
            s.append("The reading of the hand behind the call is clear-cut: even the least certain point, "
                     + (f"that it {least['gloss']}, is judged at {pct(least['act'])}." if believed
                        else f"whether it {least['gloss']}, is ruled out at {pct(least['act'])}."))
        elif least["certainty"] >= 0.7:
            s.append(f"It is fairly but not fully sure that the call {least['gloss']} ({pct(least['act'])})." if believed
                     else f"It is fairly but not fully sure that the call is not one that {least['gloss']} "
                          f"({pct(least['act'])}).")
        else:
            s.append(f"It is unsure whether the call {least['gloss']} ({pct(least['act'])}), and that point "
                     f"weighs on the decision.")
        if f.get("rule_rejects"):
            s.append(f"It also relies on the call being one that {f['rule_rejects'][0]}, which the rules do not "
                     f"confirm for this hand.")
        if f.get("err_rate") is not None:
            s.append(f"Decisions this certain differ from WBridge5's call in {pct(f['err_rate'])} of test cases "
                     f"(all decisions: {pct(f['err_overall'])}).")
        return " ".join(s)
    if t == "explain":
        return f["explanation"]
    return OUT_OF_SCOPE


# ---------------------------------------------------------------------------
# checks: hand facts like an explanation, and every stated number against the facts
# ---------------------------------------------------------------------------

_CF_RE = re.compile(r"(?:[Ww]ith )?[Tt]he ([♠♥♦♣](?:10|[2-9AKQJ])) instead of the ([♠♥♦♣](?:10|[2-9AKQJ]))"
                    r"(?: \((?:[^()]*?, )?(\d+) HCP, (\d+-\d+-\d+-\d+)\))?[^.]*?would (?:still )?"
                    r"(?:bid ([1-7](?:NT|[CDHS])|Pass|Double|Redouble)|(pass|double|redouble))")
# a two-card change in any of the explainer's phrasings: "with the A instead of the B and the C instead of
# the D (17 HCP, 1-2-5-5), the agent would bid 3NT", "Two-card change: the ... (...): the agent would bid 4H",
# "... (8 HCP, 3-2-2-6). With that change the agent would bid 3NT"
_CF2_RE = re.compile(r"(?:[Ww]ith )?[Tt]he ([♠♥♦♣](?:10|[2-9AKQJ])) instead of the ([♠♥♦♣](?:10|[2-9AKQJ])) and the "
                     r"([♠♥♦♣](?:10|[2-9AKQJ])) instead of the ([♠♥♦♣](?:10|[2-9AKQJ])) \((\d+) HCP, (\d+-\d+-\d+-\d+)\)"
                     r"(?:\.\s*[Ww]ith (?:that|this|these) changes?,?)?"
                     r"[^.]*?would (?:bid ([1-7](?:NT|[CDHS])|Pass|Double|Redouble)|(pass|double|redouble))")
_PCT_RE = re.compile(r"\b(\d{1,3})%")
_CALL_RE = re.compile(r"\b([1-7](?:NT|[CDHS]))\b")


def _card_from_text(s: str) -> str:
    sym = {v: k for k, v in SUIT_SYM.items()}[s[0]]
    return sym + ("T" if s[1:] == "10" else s[1:])


def _call_from_text(s: str) -> str:
    return {"Pass": "P", "Double": "X", "Redouble": "XX"}.get(s, s)


def check_answer(frame: Dict[str, Any], q: dict, f: Dict[str, Any], text: str) -> List[str]:
    """Problems with an answer: (1) hand claims, by the explanation checker's
    hand checks (the counterfactual "With the X instead of the Y (...)" clauses
    are about ANOTHER hand and are verified against the facts instead);
    (2) every card swap, call and percentage it states must be in the facts;
    (3) the answer must address the query (name the asked call, the new call);
    (4) style (no identifiers, no "concept", no "the model")."""
    from nlg import consistency_checks as K
    from nlg.prompt import check_style
    probs: List[str] = []
    if not text or not text.strip():
        return ["empty answer"]
    st = check_style(text)
    if st and not (q["type"] in ("explain",) or (q["type"] == "why_not" and f.get("same"))):
        probs.append(st)
    if q["type"] == "other":
        return probs + ([] if "only answer questions about this call" in text or "I can only" in text
                        else ["out-of-scope question answered as if in scope"])
    if q["type"] == "explain" or (q["type"] == "why_not" and f.get("same")):
        return probs + K.check_explanation(frame, text)
    # (2)/(3) counterfactual clauses
    allowed = []
    if q["type"] == "why_not" and f.get("swap_to_call"):
        allowed.append(f["swap_to_call"])
    if q["type"] == "change":
        allowed += f["changes"] + ([f["closest"]] if f.get("closest") else [])
    if q["type"] == "swap" and f.get("valid"):
        allowed.append(f)
    rest = text
    pair_free = text
    for m in _CF2_RE.finditer(text):
        pr = f.get("pair") if q["type"] == "change" else None
        got = [(_card_from_text(m.group(2)), _card_from_text(m.group(1))), (_card_from_text(m.group(4)), _card_from_text(m.group(3)))]
        want = [(x["out"], x["in"]) for x in (pr or {}).get("swaps", [])]
        if not pr or sorted(got) != sorted(want):
            probs.append(f"card change {m.group(0)!r} is not one the search found")
        elif int(m.group(5)) != pr["hcp"] or m.group(6) != pr["shape"]:
            probs.append(f"the changed hand is {pr['hcp']} HCP {pr['shape']}, not {m.group(5)} HCP {m.group(6)}")
        elif (_call_from_text(m.group(7)) if m.group(7) else {"pass": "P", "double": "X", "redouble": "XX"}[m.group(8)]) != pr["new_call"]:
            probs.append(f"with that change the agent bids {pr['new_call']}")
        pair_free = pair_free.replace(m.group(0), "")
    for m in _CF_RE.finditer(pair_free):
        i_c, o_c = _card_from_text(m.group(1)), _card_from_text(m.group(2))
        hit = [a for a in allowed if a["in"] == i_c and a["out"] == o_c]
        if not hit:
            probs.append(f"card change {m.group(0)!r} is not one the search found")
            continue
        a = hit[0]
        if m.group(3) and (int(m.group(3)) != a["hcp"] or m.group(4) != a["shape"]):
            probs.append(f"the changed hand is {a['hcp']} HCP {a['shape']}, not {m.group(3)} HCP {m.group(4)}")
        said = _call_from_text(m.group(5)) if m.group(5) else {"pass": "P", "double": "X", "redouble": "XX"}[m.group(6)]
        want = a.get("new_call") if q["type"] != "why_not" else f["call"]
        if q["type"] == "change" and f.get("call") and a is not f.get("closest"):
            want = f["call"]
        if q["type"] == "swap" and a.get("same"):
            want = f["chosen"]
        if said != want:
            probs.append(f"with that change the agent bids {want}, not {said}")
    # sentences with a counterfactual clause are about another hand: not checked against this one
    rest = " ".join(s for s in re.split(r"(?<=[.!?])\s+", text) if not _CF_RE.search(s) and not _CF2_RE.search(s))
    # percentages must be among the facts' probabilities / rates
    known = []
    for k in ("p_chosen", "p_call", "p_runner_up", "new_p", "err_rate", "err_overall"):
        if f.get(k) is not None:
            known.append(100 * f[k])
    if f.get("closest"):
        known.append(100 * f["closest"]["p_target"])
    if f.get("pair"):
        known.append(100 * f["pair"]["new_p"])
    if f.get("least"):
        known.append(100 * f["least"]["act"])
    if f.get("wb5") and f["wb5"].get("share") is not None:  # how often WBridge5 chose the call here
        known.append(100 * f["wb5"]["share"])
    for a in allowed:
        if a.get("new_p") is not None:
            known.append(100 * a["new_p"])
    for m in _PCT_RE.finditer(text):
        v = int(m.group(1))
        if q["type"] == "meaning" and v in (90, 95):
            continue
        if v == 90 and text[m.start() - 11:m.start()] == "the middle ":  # WBridge5's 5th-95th percentile range
            continue
        if not any(abs(v - k) <= 1 for k in known):
            probs.append(f"{v}% is not one of the agent's numbers")
    if q["type"] in ("why_not", "change") and f.get("call") and f["call"] not in ("P", "X", "XX"):
        if f["call"] not in _CALL_RE.findall(text):
            probs.append(f"the answer does not mention the asked call {f['call']}")
    if q["type"] == "meaning" and f.get("found") and f.get("wb5"):
        w = f["wb5"]
        if f"{w['hcp'][0]} to {w['hcp'][1]} HCP" not in text and str(w["hcp"][0]) in text:
            probs.append("WBridge5's range is misquoted")
    # (1) hand claims, sentence by sentence against the right frame: a "why not"
    # answer's first sentences describe the OTHER call (checked as if it had been
    # made, so its meaning is not compared with the chosen call's); a meaning
    # answer describes another player's call and makes no claim about this hand
    if q["type"] == "trust":  # the glosses describe the call (quoted from the facts), not the hand
        for g in [(f.get("least") or {}).get("gloss")] + list(f.get("rule_rejects") or []):
            if g:
                rest = rest.replace(g, "")
    if q["type"] != "meaning":
        alt = _alt_frame(frame, f["call"]) if q["type"] == "why_not" and f.get("legal") else None
        own, other = [], []
        for sent in re.split(r"(?<=[.!?])\s+", rest):
            if alt is not None and (re.search(r"was (?:the main alternative|possible)", sent)
                                    or sent.startswith("With this hand it would be unusual")):
                other.append(sent)
            else:
                own.append(sent)
        for fr, part in ((frame, " ".join(own)), (alt, " ".join(other))):
            if fr is not None and part.strip():
                probs += _hand_checks(fr, part)
    try:
        w = K.check_repeated_sentences(text) if q["type"] != "change" else None
    except Exception:
        w = None
    if w:
        probs.append(w)
    return probs


def _hand_checks(fr: Dict[str, Any], text: str) -> List[str]:
    from nlg import consistency_checks as K
    out = []
    for check in (K.check_suit_length_claims, K.check_hand_claims, K.check_partner_opponent_attribution,
                  K.check_role_bid_claims, K.check_convention_claims, K.check_length_comparisons,
                  K.check_vulnerability_claims, K.check_shape_claims):
        try:
            w = check(fr, text)
        except Exception:
            w = None
        if w:
            out.append(w)
    stripped = _strip_meaning_numbers(text)
    try:
        w = K.check_hcp_threshold_claims(fr.get("hcp"), stripped)
    except Exception:
        w = None
    if w:
        out.append(w)
    try:
        for c in K.claim_report(fr, stripped):
            if not c["ok"]:
                out.append(f"{c['type']} claim {c['text']!r} does not hold ({c['why']})")
                break
    except Exception:
        pass
    return out


def _strip_meaning_numbers(text: str) -> str:
    """Ranges that describe what a CALL promises ("showing a balanced 15 to 17
    HCP", "made with 11 to 21 HCP") are not claims about this hand."""
    text = re.sub(r"(?:showing|shows|promising|promises|made with|with)\s+(?:an? |a balanced |balanced |at least )?"
                  r"\d+\s*(?:to|-|–)\s*\d+\s*HCP", "", text)
    return re.sub(r"In WBridge5's own games[^.]*\.", "", text)


# ---------------------------------------------------------------------------
# prompts (identical in training and at inference)
# ---------------------------------------------------------------------------

PARSE_SYSTEM = (
    "You turn a bridge player's question about one bidding decision into a query. Answer with one JSON object "
    "and nothing else. Query types:\n"
    '{"type": "why_not", "call": C} -- why the agent did not make call C (C like "3NT", "2H", "P", "X", "XX")\n'
    '{"type": "change", "kind": K} -- what change to the hand would change the call; K is "any", "stronger", '
    '"weaker", or "longer"/"shorter" with "suit": S (S one of "S", "H", "D", "C")\n'
    '{"type": "change", "kind": "any", "call": C} -- what the hand would need for the agent to bid C\n'
    '{"type": "swap", "out": X, "in": Y} -- the call if card X of the hand were card Y (cards like "SA", "CT", "H4")\n'
    '{"type": "meaning", "seat": P, "call": C} -- what an earlier call C by seat P ("N", "E", "S", "W") means\n'
    '{"type": "trust"} -- how sure the agent is, how reliable the call is\n'
    '{"type": "explain"} -- why the agent made its call\n'
    '{"type": "other"} -- anything else (the play, the lead, the result, unrelated questions)')

ANSWER_SYSTEM = (
    "You answer a bridge player's follow-up question about one bidding decision of an AI agent, in two or three "
    "plain sentences, using ONLY the facts given. Do not add numbers, cards or calls that are not in the facts. "
    "Write cards as in the facts (♠A, ♣10), calls as 2H, 3NT, Pass, Double.")


def context_lines(frame: Dict[str, Any]) -> str:
    from nlg.explanation_inputs import narrate_auction, vulnerability_line
    t = frame.get("table") or {}
    you = SEAT_NAME.get(t.get("you"), "?")
    v = vulnerability_line(frame)
    return (f"Decision: {you} bid {frame['bid']}.\n"
            f"Seats: the bidder is {you}, partner is {SEAT_NAME.get(t.get('partner'), '?')}, opponents are "
            f"{' and '.join(SEAT_NAME.get(o, o) for o in t.get('opponents') or [])}.{(' Vulnerability: ' + v + '.') if v else ''}\n"
            f"Auction before the call: {narrate_auction(frame)}\n"
            f"Hand: {frame['hand_str']} ({frame.get('hcp')} HCP).")


def parse_messages(frame: Dict[str, Any], question: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": PARSE_SYSTEM},
            {"role": "user", "content": f"{context_lines(frame)}\n\nQuestion: {question}\n\nQuery:"}]


def render_facts(frame: Dict[str, Any], q: dict, f: Dict[str, Any]) -> List[str]:
    """The facts as prompt lines (everything the teacher's answer states)."""
    t, ch = q["type"], call_name(f["chosen"])
    L = [f"The agent bid {ch} ({pct(f['p_chosen'])})."]
    if t == "why_not":
        c = call_name(f["call"])
        if f.get("same"):
            L.append(f"{c} is the call the agent made. Its explanation: {f['explanation']}")
        elif not f.get("legal"):
            L.append(f"{c} is not a legal call here: {f['reason']}.")
        else:
            L.append(f"{c}: probability {pct(f['p_call'])}, rank {f['rank']} among the calls"
                     + (" (the main alternative)." if f["runner_up"] else "."))
            if f.get("meaning"):
                L.append(f"What {c} would mean here: {f['meaning']}.")
            for n in f.get("notes") or []:
                L.append(f"Why {c} would be unusual with this hand: {n}")
            if f.get("evidence"):
                L.append(f"Hand evidence for {ch} over {c}: " + "; ".join(f["evidence"][:2]) + ".")
            sw = f.get("swap_to_call")
            L.append(f"Smallest one-card change after which the agent bids {c}: " +
                     (f"the {card_text(sw['in'])} instead of the {card_text(sw['out'])} ({change_desc(sw, frame.get('hcp'))}, "
                      f"{sw['hcp']} HCP, {sw['shape']})." if sw else "none."))
            if f.get("wb5"):
                w = f["wb5"]
                L.append(f"In WBridge5's games at this exact point, {_wb5_share_clause(c, w)}; "
                         f"this hand has {frame.get('hcp')} HCP.")
    elif t == "change" and f.get("call"):
        c = call_name(f["call"])
        L.append(f"Asked about: what the hand would need for {c}.")
        if not f.get("legal"):
            L.append(f"{c} is not a legal call here: {f['reason']}.")
        else:
            L.append(f"{c} now: probability {pct(f['p_call'])}.")
            for a in f["changes"]:
                L.append(f"The {card_text(a['in'])} instead of the {card_text(a['out'])} ({change_desc(a, frame.get('hcp'))}, "
                         f"{a['hcp']} HCP, {a['shape']}): the agent bids {call_name(a['new_call'])} ({pct(a['new_p'])}).")
            if not f["changes"]:
                L.append(f"No single card change makes the agent bid {c}.")
            if f.get("pair"):
                (a1, a2), pr = f["pair"]["swaps"], f["pair"]
                L.append(f"Two-card change: the {card_text(a1['in'])} instead of the {card_text(a1['out'])} and the "
                         f"{card_text(a2['in'])} instead of the {card_text(a2['out'])} ({pr['hcp']} HCP, {pr['shape']}): "
                         f"the agent bids {call_name(pr['new_call'])} ({pct(pr['new_p'])}).")
            if f.get("closest"):
                cl = f["closest"]
                L.append(f"Closest one-card change: the {card_text(cl['in'])} instead of the {card_text(cl['out'])} "
                         f"({change_desc(cl, frame.get('hcp'))}, {cl['hcp']} HCP, {cl['shape']}): the agent bids "
                         f"{call_name(cl['new_call'])}, and {c} rises to {pct(cl['p_target'])}.")
            if f.get("wb5"):
                w = f["wb5"]
                L.append(f"In WBridge5's games at this exact point, {_wb5_share_clause(c, w)}; "
                         f"this hand has {frame.get('hcp')} HCP.")
    elif t == "change":
        what = {"any": "any change", "stronger": "a change that adds strength", "weaker": "a change that takes strength away",
                "longer": f"a change that adds a {SUIT_SING.get(f.get('suit'), '')}",
                "shorter": f"a change that removes a {SUIT_SING.get(f.get('suit'), '')}"}[f["kind"]]
        L.append(f"Asked about: {what}.")
        if not f["changes"]:
            L.append("No single card change of that kind changes the call.")
        for a in f["changes"]:
            L.append(f"The {card_text(a['in'])} instead of the {card_text(a['out'])} ({change_desc(a, frame.get('hcp'))}, "
                     f"{a['hcp']} HCP, {a['shape']}): the agent bids {call_name(a['new_call'])} ({pct(a['new_p'])}).")
    elif t == "swap":
        if not f.get("valid"):
            L.append(f"The swap is not possible: {f['why']}.")
        else:
            L.append(f"The {card_text(f['in'])} instead of the {card_text(f['out'])} ({change_desc(f, frame.get('hcp'))}, "
                     f"{f['hcp']} HCP, {f['shape']}): the agent bids {call_name(f['new_call'])} ({pct(f['new_p'])})"
                     + (", the same call." if f["same"] else ", a different call."))
    elif t == "meaning":
        who, c = SEAT_NAME[f["seat"]], call_name(f["call"])
        if not f.get("found"):
            L.append(f"{who} did not bid {c}." + (f" {SEAT_NAME[f['made_by']]} did." if f.get("made_by") else
                                                   f" Nobody bid {c}."))
        else:
            L.append(f"{who}'s {c}: {f['situation']}" + (f"; meaning: {f['meaning']}." if f.get("meaning") else "."))
            w = f.get("wb5")
            if w and w.get("exact"):
                length = (f", {w['min_len']} to {w['max_len']} {SUIT_NAME[w['suit']]}" if w.get("suit") else "")
                L.append(f"WBridge5's own games, this exact auction: {w['hcp'][0]} to {w['hcp'][1]} HCP{length} "
                         f"(the middle 90% of {w['n']:,} cases).")
            elif w:
                length = (f", at least {w['min_len']} {SUIT_NAME[w['suit']]}" if w.get("suit") and (w.get("min_len") or 0) >= 3 else "")
                L.append(f"WBridge5's own games: {w['hcp'][0]} to {w['hcp'][1]} HCP{length} (95% of {w['n']:,} cases).")
    elif t == "trust":
        if f.get("runner_up"):
            L.append(f"Main alternative: {call_name(f['runner_up'])} ({pct(f['p_runner_up'])}).")
        if not f.get("concepts"):
            L.append("The network has no layer of bridge reasons to inspect.")
        else:
            lt = f["least"]
            L.append((f"Least certain of the points that carry the call: the call {lt['gloss']} "
                      if lt["act"] >= 0.5 else
                      f"Least certain of the points that carry the call: whether the call {lt['gloss']} -- ruled out, "
                      ) + f"(judged {pct(lt['act'])}, certainty {pct(lt['certainty'])}).")
            for g in f.get("rule_rejects") or []:
                L.append(f"The agent relies on the call being one that {g}; the rules do not confirm this for the hand.")
            if f.get("err_rate") is not None:
                L.append(f"Test decisions this certain differ from WBridge5's call in {pct(f['err_rate'])} of cases "
                         f"(all decisions: {pct(f['err_overall'])}).")
    elif t == "explain":
        L.append(f"Explanation: {f['explanation']}")
    else:
        L.append("The question is not about this call.")
    return L


def answer_messages(frame: Dict[str, Any], question: str, q: dict, f: Dict[str, Any]) -> List[Dict[str, str]]:
    facts = "\n".join(f"- {x}" for x in render_facts(frame, q, f))
    return [{"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"{context_lines(frame)}\n\nQuestion: {question}\nQuery: {query_json(q)}\n\n"
                                        f"Facts (exact):\n{facts}\n\nAnswer:"}]


# ---------------------------------------------------------------------------
# questions in many phrasings (generator) -- TRAIN templates and HELD-OUT ones
# ---------------------------------------------------------------------------

_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
_STRAIN_WORDS = {"C": ["clubs", "C", "♣"], "D": ["diamonds", "D", "♦"], "H": ["hearts", "H", "♥"],
                 "S": ["spades", "S", "♠"], "NT": ["notrump", "NT", "no trump", "N"]}


def say_call(c: str, rng: random.Random) -> str:
    if c == "P":
        return rng.choice(["pass", "passing", "Pass"])
    if c == "X":
        return rng.choice(["double", "a double", "X", "Dbl"])
    if c == "XX":
        return rng.choice(["redouble", "a redouble", "XX"])
    lvl, strain = c[0], c[1:]
    w = rng.choice(_STRAIN_WORDS[strain])
    form = rng.random()
    if form < 0.5:
        return c if rng.random() < 0.7 else c.lower()
    if form < 0.75 and len(w) > 2:
        return f"{lvl} {w}"
    if form < 0.9 and len(w) > 2:
        return f"{_NUM[int(lvl)]} {w}"
    return f"{lvl}{w}" if len(w) <= 2 else f"{lvl} {w}"


def say_card(card: str, rng: random.Random) -> str:
    s, r = card[0], card[1]
    rw = RANK_WORD[r]
    sym = SUIT_SYM[s] + ("10" if r == "T" else r)
    return rng.choice([f"the {rw} of {SUIT_NAME[s]}", sym, f"the {sym}", f"the {SUIT_SING[s]} {rw}",
                       f"{card}", f"{('10' if r == 'T' else r)}{SUIT_SYM[s]}", f"the {('10' if r == 'T' else r)} of {SUIT_NAME[s]}"])


def say_suit(s: str, rng: random.Random, singular: bool = False) -> str:
    return SUIT_SING[s] if singular else rng.choice([SUIT_NAME[s], SUIT_NAME[s], SUIT_SYM[s]])


def say_seat(frame: Dict[str, Any], seat: str, rng: random.Random) -> str:
    t = frame.get("table") or {}
    if seat == t.get("partner"):
        return rng.choice(["partner", "my partner", SEAT_NAME[seat], "partner"])
    you = t.get("you")
    left = "NESW"[("NESW".index(you) + 1) % 4] if you else None
    if seat == left:
        return rng.choice([SEAT_NAME[seat], "my left-hand opponent", "LHO", SEAT_NAME[seat]])
    return rng.choice([SEAT_NAME[seat], "my right-hand opponent", "RHO", SEAT_NAME[seat]])


WHY_NOT = {
    "train": ["Why not {call}?", "Why didn't you bid {call}?", "Why not bid {call} instead?", "Wouldn't {call} be better?",
              "What's wrong with {call}?", "I would have bid {call}. Why didn't you?", "How about {call}?",
              "Why {chosen} and not {call}?", "Could you have bid {call} here?", "{call} looks more natural to me, why not?",
              "why no {call}", "Isn't {call} the normal bid here?", "why not {call}", "What about {call}?",
              "Why did you choose {chosen} over {call}?", "Shouldn't you have bid {call}?"],
    "heldout": ["What made you reject {call}?", "Was {call} ever an option?", "Explain why {call} is worse.",
                "My instinct says {call}. Thoughts?", "Would {call} have been wrong?",
                "Why did you prefer {chosen} to {call}?", "{call} seems obvious, so why not?"],
}
CHANGE = {
    "any": {"train": ["What would change your call?", "What would make you bid differently?",
                      "When would you bid something else?", "Which card would change your mind?",
                      "How different would the hand need to be for another call?", "what would change the bid"],
            "heldout": ["What's the smallest change to my hand that changes the bid?",
                        "Is there a hand close to this one where you'd bid otherwise?"]},
    "stronger": {"train": ["What if I had more points?", "What if the hand were stronger?",
                           "And with a bit more strength?", "What if I had an extra honour?", "with more HCP?",
                           "Give me an extra king: would that change anything?", "What if I held another queen?"],
                 "heldout": ["With more high cards, would you bid differently?", "What if my hand were a little better?"]},
    "weaker": {"train": ["What if I had fewer points?", "What if the hand were weaker?", "And with less strength?",
                         "What if I had one honour less?", "Take a king away: same call?", "With a jack less?"],
               "heldout": ["Suppose I was a queen lighter?", "What if my hand were a bit worse?"]},
    "longer": {"train": ["What if I had another {suit_s}?", "What if my {suit} were longer?",
                         "With one more {suit_s}, would you bid differently?", "what if i had an extra {suit_s}"],
               "heldout": ["Add a {suit_s} to my hand: what changes?", "Suppose the {suit} were one card longer?"]},
    "shorter": {"train": ["What if I had one {suit_s} fewer?", "What if my {suit} were shorter?",
                          "With one {suit_s} less, would you bid differently?"],
                "heldout": ["Take away a {suit_s}: does the call change?", "Suppose the {suit} were a card shorter?"]},
}
# "what would I need for X?": the smallest one-card changes that lead to THAT call (not "why not X",
# which answers with the chosen call's reasons)
NEED = {"train": ["What would I need to bid {call}?", "What would it take for you to bid {call}?",
                  "What hand would you need for {call}?", "When would you bid {call} here?",
                  "How much more would I need for {call}?", "what would make you bid {call}",
                  "What would the hand need for {call}?"],
        "heldout": ["What do I need to hold for {call}?", "Which hand would justify {call}?",
                    "What would have to change for {call}?"]}
SWAP = {"train": ["What if I had {in} instead of {out}?", "Swap my {out} for {in}: what would you bid?",
                  "And with {in} in place of {out}?", "Suppose {out} were {in}?", "what if {out} was {in}",
                  "If I held {in} rather than {out}, what would you bid?"],
        "heldout": ["Replace {out} by {in}. Same bid?", "If my {out} were the {in}, would you still bid {chosen}?",
                    "Change {out} into {in} -- what happens?"]}
MEANING = {"train": ["What does {who}'s {call} mean?", "What did {who}'s {call} show?", "What is {who}'s {call}?",
                     "Explain {who}'s {call}.", "What does {call} by {who} promise?", "what does {who}'s {call} show",
                     "How many points did {who} show with {call}?", "How strong is {who}'s {call}?",
                     "What strength does {who}'s {call} promise?"],
           "heldout": ["{who} bid {call} -- what does that tell me?", "How should I read {who}'s {call}?",
                       "What is {who} saying with {call}?"]}
TRUST = {"train": ["How sure are you?", "Can I trust this call?", "How confident is this bid?",
                   "Is this a close decision?", "Are you certain about {chosen}?", "how sure", "Is {chosen} a safe call?"],
         "heldout": ["How reliable is this call?", "Would you bet on this bid?", "Is there much doubt here?"]}
EXPLAIN = {"train": ["Why did you bid {chosen}?", "Explain your call.", "Why {chosen}?", "Explain {chosen}.",
                     "What's the reasoning behind {chosen}?", "why this bid"],
           "heldout": ["Walk me through this bid.", "What made you choose {chosen}?"]}
OTHER = {"train": ["How should I play this hand?", "Who will win this deal?", "What's the best opening lead?",
                   "What will the final contract be?", "Tell me a joke.", "How many tricks will we take?",
                   "What is the score?", "Which card should I play first?", "Can we make a slam?",
                   "Is game going to make?"],
         "heldout": ["Which card should I lead?", "Will this contract make?", "What's the weather like?"]}


# questions that name no call or card outright: the call is resolved from the
# auction ("raise partner" = the cheapest legal bid in partner's suit)
INDIRECT = {
    "raise": {"train": ["Why not raise partner?", "Why didn't you support partner's {psuit}?", "why not support partner",
                        "Why no raise of partner's suit?"],
              "heldout": ["Shouldn't you show support for partner?", "Why not just raise partner's suit?"]},
    "rebid": {"train": ["Why not rebid your {suit}?", "Why didn't you repeat your suit?", "why not bid my {suit} again"],
              "heldout": ["Shouldn't you show the long {suit} again?", "Why not repeat the {suit}?"]},
    "opening": {"train": ["What does the opening bid show?", "What did the opening bid mean?", "what was the opening",
                          "What did the opener's first call say?", "Tell me about the auction's first bid."],
                "heldout": ["What was the opening about?", "Explain the first bid of the auction."]},
    "partner_last": {"train": ["What did partner's last bid mean?", "What does partner's last call show?"],
                     "heldout": ["And partner's last call, what was that?", "What is partner telling me with the last bid?"]},
    "their_double": {"train": ["What did their double mean?", "What does the opponents' double show?"],
                     "heldout": ["Why did they double, what does it show?", "What is the opponents' double about?"]},
    "rank_swap": {"train": ["What if my {out} were the {rank}?", "And if the {out} were the {rank}?",
                            "Promote the {out} to the {rank}: what then?", "Turn my {out} into the {rank}?"],
                  "heldout": ["Make the {out} the {rank}: same bid?", "Upgrade the {out} to the {rank}?"]},
}


def _cheapest_in(d: "Decision", suit: str) -> Optional[str]:
    for lvl in range(1, 8):
        c = f"{lvl}{suit}"
        if c in BIDS and d.legal[BIDS.index(c)]:
            return c
    return None


def indirect_question(d: "Decision", rng: random.Random, split: str) -> Optional[Tuple[str, dict]]:
    """A question that refers to a call or card through the auction, or None if
    this decision offers none of the kind drawn."""
    from nlg.explanation_inputs import natural_partner_suits
    frame = d.frame
    t = frame.get("table") or {}
    you, partner = t.get("you"), t.get("partner")
    prior = frame.get("auction_before") or []
    kind = rng.choice(list(INDIRECT))
    tpl = rng.choice(INDIRECT[kind][split])
    if kind == "raise":
        ps = natural_partner_suits(frame)
        c = _cheapest_in(d, ps[-1]) if ps else None
        if not c or c == frame["bid"]:
            return None
        return tpl.format(psuit=SUIT_NAME[ps[-1]]), {"type": "why_not", "call": c}
    if kind == "rebid":
        mine = [b["bid"] for b in prior if b["seat"] == you and b["bid"][:1].isdigit() and b["bid"][-1] in SUITS]
        c = _cheapest_in(d, mine[-1][-1]) if mine else None
        if not c or c == frame["bid"]:
            return None
        return tpl.format(suit=SUIT_NAME[mine[-1][-1]]), {"type": "why_not", "call": c}
    if kind == "opening":
        real = [b for b in prior if b["bid"] not in NB]
        if not real or real[0]["seat"] == you:
            return None
        return tpl, {"type": "meaning", "seat": real[0]["seat"], "call": real[0]["bid"]}
    if kind == "partner_last":
        pb = [b for b in prior if b["seat"] == partner and b["bid"] != "P"]
        if not pb:
            return None
        return tpl, {"type": "meaning", "seat": partner, "call": pb[-1]["bid"]}
    if kind == "their_double":
        xs = [b for b in prior if b["seat"] not in (you, partner) and b["bid"] == "X"]
        if not xs:
            return None
        return tpl, {"type": "meaning", "seat": xs[-1]["seat"], "call": "X"}
    held = [s_ + r for s_, rs in d.hand.cards.items() for r in rs]
    out_c = rng.choice(held)
    ranks = [r for r in "AKQJT" if out_c[0] + r not in held]
    if not ranks:
        return None
    r = rng.choice(ranks)
    return (tpl.format(out=say_card(out_c, rng).replace("the the ", "the "), rank=RANK_WORD[r]),
            {"type": "swap", "out": out_c, "in": out_c[0] + r})


def _alt_calls(d: Decision, rng: random.Random) -> str:
    """An alternative call to ask about: mostly the likely ones, sometimes any legal call, rarely an illegal one."""
    legal = np.flatnonzero(d.legal)
    order = [int(i) for i in np.argsort(-d.probs) if d.legal[i] and int(i) != d.action]
    r = rng.random()
    if r < 0.45 and order:
        return BIDS[order[0]]
    if r < 0.7 and len(order) > 1:
        return BIDS[rng.choice(order[1:4])]
    if r < 0.93:
        return BIDS[int(rng.choice([i for i in legal if i != d.action] or list(legal)))]
    if r < 0.97:
        return d.frame["bid"]  # asking about the call it made
    illegal = [i for i in range(len(BIDS)) if not d.legal[i] and BIDS[i] != d.frame["bid"]]
    return BIDS[rng.choice(illegal)] if illegal else BIDS[order[0]] if order else d.frame["bid"]


def sample_question(d: Decision, rng: random.Random, split: str) -> Tuple[str, dict]:
    """(question text, gold query) in a TRAIN or HELD-OUT phrasing."""
    frame = d.frame
    prior = frame.get("auction_before") or []
    you = (frame.get("table") or {}).get("you")
    others = [b for b in prior if b["seat"] != you and b["bid"] != "P"]
    chosen_s = say_call(frame["bid"], rng)
    if rng.random() < 0.18:
        for _ in range(4):
            x = indirect_question(d, rng, split)
            if x:
                return x[0].replace("my the ", "my ").replace("the the ", "the "), dict(x[1], indirect=True)
    types = ["why_not"] * 30 + ["change"] * 16 + ["swap"] * 14 + ["meaning"] * (16 if others else 0) + \
            ["trust"] * 10 + ["explain"] * 7 + ["other"] * 7
    t = rng.choice(types)
    if t == "why_not":
        c = _alt_calls(d, rng)
        tpl = rng.choice(WHY_NOT[split])
        return tpl.format(call=say_call(c, rng), chosen=chosen_s), {"type": "why_not", "call": c}
    if t == "change" and rng.random() < 0.3:
        c = _alt_calls(d, rng)
        if c != frame["bid"]:
            return rng.choice(NEED[split]).format(call=say_call(c, rng)), {"type": "change", "kind": "any", "call": c}
    if t == "change":
        kind = rng.choice(["any", "any", "stronger", "weaker", "longer", "shorter"])
        q = {"type": "change", "kind": kind}
        tpl = rng.choice(CHANGE[kind][split])
        if kind in ("longer", "shorter"):
            s = rng.choice(SUITS)
            q["suit"] = s
            return tpl.format(suit=say_suit(s, rng), suit_s=say_suit(s, rng, singular=True)), q
        return tpl, q
    if t == "swap":
        held = [s + r for s, rs in d.hand.cards.items() for r in rs]
        out_c = rng.choice(held)
        pool = [c for c in FULL_DECK if c not in held]
        if rng.random() < 0.6:  # mostly a nearby or an honour card, as people ask
            pool = [c for c in pool if c[0] == out_c[0] or c[1] in "AKQJ"] or pool
        in_c = rng.choice(pool)
        if rng.random() < 0.05:  # a card the hand does not hold
            out_c = rng.choice([c for c in FULL_DECK if c not in held])
        tpl = rng.choice(SWAP[split])
        text = tpl.format(out=say_card(out_c, rng), **{"in": say_card(in_c, rng)}, chosen=chosen_s)
        text = text.replace("my the ", "my ").replace("the the ", "the ")
        return text, {"type": "swap", "out": out_c, "in": in_c}
    if t == "meaning":
        b = rng.choice(others)
        tpl = rng.choice(MEANING[split])
        return (tpl.format(who=say_seat(frame, b["seat"], rng), call=say_call(b["bid"], rng)),
                {"type": "meaning", "seat": b["seat"], "call": b["bid"]})
    if t == "trust":
        return rng.choice(TRUST[split]).format(chosen=chosen_s), {"type": "trust"}
    if t == "explain":
        return rng.choice(EXPLAIN[split]).format(chosen=chosen_s), {"type": "explain"}
    return rng.choice(OTHER[split]), {"type": "other"}


# ---------------------------------------------------------------------------
# the keyword parser (baseline; what a template system can do)
# ---------------------------------------------------------------------------

_CALL_PAT = re.compile(r"\b([1-7])\s?(nt|no ?trumps?|notrumps?|n|c|d|h|s|clubs?|diamonds?|hearts?|spades?|[♣♦♥♠])(?![a-z])",
                       re.IGNORECASE)
_WORDNUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7"}
_CARD_PAT = re.compile(r"(?:\b(ace|king|queen|jack|ten|nine|eight|seven|six|five|four|three|two|10|[2-9akqjt])\s*of\s*"
                       r"(spades?|hearts?|diamonds?|clubs?)\b)|(?:([♠♥♦♣])\s?(10|[2-9AKQJT]))|(?:\b(10|[2-9AKQJT])\s?([♠♥♦♣]))"
                       r"|(?:\b([SHDC])(10|[2-9AKQJT])\b)|(?:\b(spade|heart|diamond|club)\s+(ace|king|queen|jack|ten))",
                       re.IGNORECASE)
_RANK_OF = {"ace": "A", "king": "K", "queen": "Q", "jack": "J", "ten": "T", "nine": "9", "eight": "8", "seven": "7",
            "six": "6", "five": "5", "four": "4", "three": "3", "two": "2", "10": "T"}
_SUIT_OF_WORD = {"spade": "S", "heart": "H", "diamond": "D", "club": "C", "♠": "S", "♥": "H", "♦": "D", "♣": "C"}


def _calls_in(text: str) -> List[str]:
    t = re.sub(r"\b(one|two|three|four|five|six|seven)\b", lambda m: _WORDNUM[m.group(1).lower()], text, flags=re.I)
    out = []
    for m in _CALL_PAT.finditer(t):
        st = m.group(2).lower()
        strain = ("NT" if st.startswith("n") else _SUIT_OF_WORD.get(st[0], None) if st[0] in "♠♥♦♣"
                  else st[0].upper())
        if strain:
            out.append(m.group(1) + strain)
    for w, c in (("redouble", "XX"), ("rdbl", "XX"), ("double", "X"), ("dbl", "X"), ("pass", "P")):
        if re.search(rf"\b{w}\w*", text, re.I) and c not in out and not (c == "X" and "XX" in out):
            out.append(c)
    return out


def _cards_in(text: str) -> List[str]:
    out = []
    for m in _CARD_PAT.finditer(text):
        g = m.groups()
        if g[0]:
            out.append(_SUIT_OF_WORD[g[1].lower()[:-1] if g[1].lower().endswith("s") else g[1].lower()] + _RANK_OF.get(g[0].lower(), g[0].upper()))
        elif g[2]:
            out.append(_SUIT_OF_WORD[g[2]] + ("T" if g[3] == "10" else g[3].upper()))
        elif g[4]:
            out.append(_SUIT_OF_WORD[g[5]] + ("T" if g[4] == "10" else g[4].upper()))
        elif g[6]:
            out.append(g[6].upper() + ("T" if g[7] == "10" else g[7].upper()))
        elif g[8]:
            out.append(_SUIT_OF_WORD[g[8].lower()] + _RANK_OF[g[9].lower()])
    return out


def _suit_word(w: str) -> str:
    return _SUIT_OF_WORD[w[0] if w[0] in "♠♥♦♣" else w.rstrip("s")]


_NEED_RE = re.compile(r"\bwhat (?:would|do|does|did) (?:i|you|we|it|the hand|my hand|south|north|east|west) "
                      r"(?:need|have to (?:have|hold)|require)|\bwhat would it take\b|\bwhat hand would\b"
                      r"|\bwhich hand would\b|\bhow (?:much|many) more\b|\bwhen would (?:you|i|it) bid\b"
                      r"|\bwhat would (?:make you|have to change)|\bneed(?:ed)? (?:to|for) (?:bid|make)\b")


_FIRST_BID_RE = re.compile(r"\b(?:first|opening) (?:bid|call)\b|\bthe opening\b|\bthe opener'?s? (?:bid|call)\b")
_HOW_STRONG_RE = re.compile(r"\bhow (?:strong|many points|many hcp)\b|\bwhat strength\b")


def refine_query(frame: Dict[str, Any], question: str, q: Optional[dict]) -> Optional[dict]:
    """Repairs of a parse that the question itself contradicts, applied after either
    parser (the GUI; the evaluation reports the parsers before and after it):
    - "what would I need for X?" read as a plain "why not X" or "what would change the
      call": give it the call it names;
    - a question about the auction's first bid read as "explain your call": the meaning of
      the first real call;
    - "how strong is partner for 1NT?" read as "how sure are you": the meaning of the
      call named, when exactly one other seat made it."""
    ql = question.lower()
    prior = frame.get("auction_before") or []
    if (not q or q.get("type") in ("explain", "trust")) and _FIRST_BID_RE.search(ql):
        first = next((b for b in prior if b["bid"] not in ("P", "Pass")), None)
        if first:
            return {"type": "meaning", "seat": first["seat"], "call": first["bid"]}
    if q and q.get("type") == "trust" and _HOW_STRONG_RE.search(ql):
        you = (frame.get("table") or {}).get("you")
        named = set(_calls_in(question))
        made = {(b["seat"], b["bid"]) for b in prior if b["bid"] in named and b["seat"] != you}
        if len(made) == 1:
            seat, call = made.pop()
            return {"type": "meaning", "seat": seat, "call": call}
    if not q or q.get("type") not in ("why_not", "change") or q.get("call") and q["type"] == "change":
        return q
    if not _NEED_RE.search(question.lower()):
        return q
    alts = [c for c in _calls_in(question) if c != frame["bid"]]
    if not alts:
        return q
    return {"type": "change", "kind": "any", "call": alts[0]}


def parse_rule(frame: Dict[str, Any], question: str) -> dict:
    """Keyword parser (the baseline the explainer is compared with): a good-faith
    rule system -- two cards make a swap, a seat or role with a call asks for
    that call's meaning, any other call named is a "why not", and word lists for
    strength, suit length, certainty, explanation and out-of-scope questions."""
    q = question.strip()
    ql = q.lower()
    calls, cards = _calls_in(q), _cards_in(q)
    t = frame.get("table") or {}
    you = t.get("you") or "S"
    chosen = frame["bid"]
    if re.search(r"\b(lead|declarer|play(?:ing)?|make game|win|score|contract|tricks|time|book|weather|joke)\b", ql):
        return {"type": "other"}
    if len(cards) >= 2:
        if re.search(r"\binstead of\b|\brather than\b|\bin place of\b", ql):
            return {"type": "swap", "out": cards[1], "in": cards[0]}
        return {"type": "swap", "out": cards[0], "in": cards[1]}
    seat = None
    for s_, name in SEAT_NAME.items():
        if re.search(rf"\b{name.lower()}\b", ql):
            seat = s_
    if seat is None and "partner" in ql:
        seat = t.get("partner")
    if seat is None and re.search(r"\blho\b|left", ql):
        seat = "NESW"[("NESW".index(you) + 1) % 4]
    if seat is None and re.search(r"\brho\b|right", ql):
        seat = "NESW"[("NESW".index(you) + 3) % 4]
    if seat and seat != you and calls:
        return {"type": "meaning", "seat": seat, "call": calls[0]}
    alts = [c for c in calls if c != chosen]
    if alts and _NEED_RE.search(ql):
        return {"type": "change", "kind": "any", "call": alts[0]}
    if alts and not re.search(r"\b(still bid|would you still)\b", ql):
        prefer_last = re.search(r"\bover\b|\band not\b|\bprefer\b|\bto\b", ql) and len(alts) == 1
        return {"type": "why_not", "call": alts[-1] if prefer_last else alts[0]}
    m = re.search(r"\b(another|one more|extra|longer)\b.*\b(spades?|hearts?|diamonds?|clubs?|[♠♥♦♣])", ql) or \
        re.search(r"\b(spades?|hearts?|diamonds?|clubs?)\b.*\b(longer)\b", ql)
    if m:
        w = m.group(2) if m.group(1) in ("another", "one more", "extra", "longer") else m.group(1)
        return {"type": "change", "kind": "longer", "suit": _suit_word(w)}
    m = re.search(r"\b(fewer|shorter|less|short)\b.*\b(spades?|hearts?|diamonds?|clubs?|[♠♥♦♣])", ql) or \
        re.search(r"\b(spades?|hearts?|diamonds?|clubs?)\b.*\b(fewer|shorter|less|short)\b", ql)
    if m:
        w = m.group(2) if m.group(1) in ("fewer", "shorter", "less", "short") else m.group(1)
        return {"type": "change", "kind": "shorter", "suit": _suit_word(w)}
    if re.search(r"\b(more points|stronger|more strength|extra (?:honou?r|ace|king|queen|jack)|more hcp|better|"
                 r"more high|high-card points)", ql):
        return {"type": "change", "kind": "stronger"}
    if re.search(r"\b(fewer points|weaker|less strength|honou?r less|worse|lighter|points less)", ql):
        return {"type": "change", "kind": "weaker"}
    if re.search(r"\b(change|different|differently|otherwise|something else|mind|close to|deciding)\b", ql):
        return {"type": "change", "kind": "any"}
    if re.search(r"\b(sure|confiden|trust|certain|reliable|bet\b|doubt|believe|guess|mistake|safe|close decision)", ql):
        return {"type": "trust"}
    if re.search(r"\b(why|explain|reason|idea|walk me through|choose|chose)", ql) or chosen in calls:
        return {"type": "explain"}
    return {"type": "other"}


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

def _positions(split_dir: Path):
    offsets = np.load(split_dir / "deal_offsets.npy")
    obs_bits = np.load(split_dir / "obs_bits.npy", mmap_mode="r")
    legal_bits = np.load(split_dir / "legal_bits.npy", mmap_mode="r")
    return offsets, obs_bits, legal_bits


def build(args):
    """Questions on the teacher dataset's decisions (same model, same deal split):
    parse and answer examples for training, and the evaluation sets."""
    import jax
    from training.checkpoint import load_checkpoint
    from nlg.explanation_frames import Enricher
    ck = load_checkpoint(args.model_path)
    en = Enricher(ck)
    fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))
    offsets, obs_bits, legal_bits = _positions(Path(args.prepared))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    stats = {"model_path": args.model_path, "splits": {}}
    for split, src, n, phrasing in (("train", "train", args.n_train, "train"), ("val", "val", args.n_val, "train"),
                                    ("test", "test", args.n_test, "mixed")):
        frames = [json.loads(l) for l in open(Path(args.teacher_dir) / f"{src}_frames.jsonl")]
        rng.shuffle(frames)
        rows, fails, kinds = [], 0, {}
        for frame in frames:
            if len(rows) >= n:
                break
            g, s = int(frame["deal_id"]), int(frame["step"])
            r = int(offsets[g]) + s
            o = np.unpackbits(obs_bits[r:r + 1], axis=1)[0, :480].astype(np.float32)
            lg = np.unpackbits(legal_bits[r:r + 1], axis=1)[0, :38].astype(bool)
            out_ = {k: np.asarray(v[0]) for k, v in fwd(o[None]).items()}
            probs = softmax_masked(np.asarray(out_["policy_logits"], np.float64), lg)
            if BIDS[int(probs.argmax())] != frame["bid"]:
                continue  # the stored frame is from another checkpoint version
            d = Decision(frame, o, lg, probs, en.policy_logits_fn, np.asarray(out_["concept_acts"]), en.W,
                         list(ck.hard_concepts), "SL independent")
            ph = phrasing if phrasing != "mixed" else ("heldout" if rng.random() < 0.7 else "train")
            question, q = sample_question(d, rng, ph)
            indirect = bool(q.pop("indirect", False))
            try:
                f = facts_for(d, q)
                ans = teacher_answer(frame, q, f)
            except Exception as e:  # pragma: no cover - reported, never silently trained on
                fails += 1
                print(f"[followup] {split} deal {g} step {s} {q}: {e!r}", flush=True)
                continue
            problems = check_answer(frame, q, f, ans)
            if problems:
                fails += 1
                if fails <= 30:
                    print(f"[followup] teacher answer fails its checks ({q['type']}): {problems[0][:200]} | {ans}",
                          flush=True)
                continue
            kinds[q["type"]] = kinds.get(q["type"], 0) + 1
            rows.append({"deal_id": g, "step": s, "question": question, "query": q, "phrasing": ph,
                         "indirect": indirect, "facts": f,
                         "answer": ans, "frame": frame})
        with open(out / f"{split}.jsonl", "w") as fh:
            for x in rows:
                fh.write(json.dumps(x, ensure_ascii=False, default=_json_default) + "\n")
        # the chat rows nlg/train_qwen.py --sft trains on
        with open(out / f"sft_{split}.jsonl", "w") as fh:
            for x in sft_rows(json.loads(json.dumps(rows, default=_json_default))):
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        stats["splits"][split] = {"rows": len(rows), "teacher_fails": fails, "types": kinds}
        print(f"[followup] {split}: {len(rows)} questions ({kinds}), {fails} dropped", flush=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def sft_rows(rows: List[dict]) -> List[dict]:
    """Chat SFT rows {"messages", "target", "task"} for nlg/train_qwen.py --sft."""
    out = []
    for r in rows:
        out.append({"task": "parse", "messages": parse_messages(r["frame"], r["question"]), "target": query_json(r["query"])})
        if r["query"]["type"] not in ("explain", "other"):  # the explanation is its own task; "other" is fixed text
            out.append({"task": "answer", "messages": answer_messages(r["frame"], r["question"], r["query"], r["facts"]),
                        "target": r["answer"]})
    return out


# ---------------------------------------------------------------------------
# calibration: how often decisions as uncertain as this one differ from WBridge5
# ---------------------------------------------------------------------------

BINS = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.0)]


def calibrate(models: List[str]):
    """TRUST_TABLE: for each concept model, the test decisions binned by the
    certainty max(a, 1-a) of the least certain of the five concepts with the
    largest |contribution| to the call over its runner-up (the signal of
    evaluation/explanation_faithfulness.py), and the rate at which the model's
    call differs from WBridge5's in each bin."""
    import jax
    from evaluation.common import MODELS, load_test_split
    from training.checkpoint import load_checkpoint
    pos = load_test_split()
    res = json.loads(TRUST_PATH.read_text()) if TRUST_PATH.exists() else {}
    for m in models:
        ck = load_checkpoint(str(MODELS[m]))
        fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))
        W = np.asarray(ck.params["concept_only_policy_head"]["w"])
        descr = np.array([n.startswith("bid_is_in_") for n in ck.hard_concepts])
        cert_all, err_all = [], []
        for s in range(0, len(pos), 8192):
            rows = np.arange(s, min(s + 8192, len(pos)))
            out = fwd(pos.obs_f32(rows))
            pl = np.asarray(out["policy_logits"])
            acts = np.asarray(out["concept_acts"])
            co = np.asarray(out["concept_only_logits"])
            lg = pos.legal[rows]
            a = np.where(lg, pl, -np.inf).argmax(1)
            co2 = np.where(lg, co, -np.inf)
            co2[np.arange(len(rows)), a] = -np.inf
            r = co2.argmax(1)
            contrib = acts * (W[:, a].T - W[:, r].T)
            contrib[:, descr] = 0
            top = np.argsort(-np.abs(contrib), axis=1)[:, :5]
            ta = np.take_along_axis(acts, top, 1)
            cert = np.maximum(ta, 1 - ta).min(1)
            cert_all.append(cert)
            err_all.append(a != pos.action[rows])
        cert, err = np.concatenate(cert_all), np.concatenate(err_all)
        bins = []
        for lo, hi in BINS:
            m_ = (cert >= lo) & ((cert < hi) | (hi >= 1.0))
            bins.append([lo, hi, float(err[m_].mean()) if m_.any() else float(err.mean()), int(m_.sum())])
        res[m] = {"bins": bins, "overall": float(err.mean()), "decisions": int(len(err)),
                  "signal": "certainty of the least certain of the 5 most contributing non-descriptor concepts"}
        print(f"[calibrate] {m}: overall {err.mean():.4f}; " + ", ".join(f"[{lo},{hi}) {r:.3f} (n={n})" for lo, hi, r, n in bins),
              flush=True)
    TRUST_PATH.write_text(json.dumps(res, indent=1))


def build_wb5_card():
    """WB5_CARD: per (bidding situation, call), the HCP range (2.5th-97.5th
    percentile) and the shortest length of the bid suit (2.5th percentile) with
    which WBridge5 itself makes the call, on the test split's 10,000 deals
    (the same fit as evaluation/convention_ranges.py)."""
    from evaluation.convention_ranges import calls, fit
    from evaluation.common import load_test_split
    card = fit(calls(load_test_split()))
    WB5_CARD_PATH.write_text(json.dumps({"agent": "WBridge5", "source": "WBridge5 test split (10,000 deals)",
                                         "built_by": "python -m nlg.followup_questions --wb5_card",
                                         "entries": {f"{k[0]}|{k[1]}": v for k, v in sorted(card.items())}}, indent=0))
    print(f"[wb5_card] {len(card)} entries -> {WB5_CARD_PATH}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--calibrate", nargs="*", default=None)
    ap.add_argument("--wb5_card", action="store_true")
    ap.add_argument("--model_path", default=str(REPO / "runs/sl/independent/params-best.pkl"))
    ap.add_argument("--teacher_dir", default=str(REPO / "runs/explainer_data/teacher"))
    ap.add_argument("--prepared", default=str(REPO / "data/wbridge5/prepared/train"))
    ap.add_argument("--out_dir", default=str(REPO / "runs/explainer_data/followup"))
    ap.add_argument("--n_train", type=int, default=2600)
    ap.add_argument("--n_val", type=int, default=150)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.wb5_card:
        build_wb5_card()
    if args.calibrate is not None:
        calibrate(args.calibrate or ["SL independent", "SL joint", "SL sequential", "PPO independent", "PPO joint",
                                     "PPO sequential"])
    if args.build:
        build(args)


if __name__ == "__main__":
    main()
