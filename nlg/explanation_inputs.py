"""What an explanation is about: exact hand facts read in the auction's context, the bidding situation
(`bid_category`), notes for calls that break their situation's requirements (`deviation_notes`), and which concepts
are the reasons. Reasons are predicted, rule-true, hand-dependent concepts ranked by their exact contribution
a_i (W[i, call] - W[i, runner-up]), or over all alternatives when the runner-up has under 5%. Pure Python over a
frame (no JAX)."""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from concepts import AuctionContext, HandFeatures, create_sayc_state_authority, is_balanced_shape
from concepts.hand import HCP_VALUES, RANKS, is_stopper
from concepts.predicates_hand import losing_trick_count, quick_tricks, rule_of_20_satisfied, support_points, wb5_points
from concepts.predicates_auction import opponent_natural_suits
from concepts.vocabulary import NON_BIDS

SUIT_NAME = {"S": "spades", "H": "hearts", "D": "diamonds", "C": "clubs"}
SEAT_NAME = {"N": "North", "E": "East", "S": "South", "W": "West"}
_SEATS = "NESW"


def hand_from_str(hand_str: str) -> HandFeatures:
    """Inverse of nlg.explanation_frames.hand_str_letter_coded ("S:AK83 H:JT74 D:J86 C:AT",
    "-" for a void), built exactly like concepts.hand.decode_hand."""
    cards = {s: [] for s in "SHDC"}
    for part in hand_str.split():
        suit, ranks = part.split(":")
        cards[suit] = [] if ranks == "-" else list(ranks)
    lengths = {s: len(cards[s]) for s in "SHDC"}
    return HandFeatures(
        hcp=sum(HCP_VALUES.get(r, 0) for rs in cards.values() for r in rs),
        lengths=lengths, balanced=is_balanced_shape(lengths),
        stoppers={s: is_stopper(cards[s]) for s in "SHDC"},
        passed_hand=False, cards=cards,
    )


def auction_from_frame(frame: Dict[str, Any]) -> AuctionContext:
    prior = frame.get("auction_before") or []
    dealer = _SEATS.index(frame["dealer"]) if frame.get("dealer") in _SEATS else 0
    return AuctionContext(bids=[b["bid"] for b in prior], bid=frame["bid"], dealer=dealer)


@lru_cache(maxsize=1)
def _authority():
    return create_sayc_state_authority()


@lru_cache(maxsize=1)
def _probe_hands() -> Tuple[HandFeatures, ...]:
    """Fixed, deterministic set of varied hands: random deals plus shape/
    strength extremes, so both HCP- and shape-conditioned rules can flip."""
    deck = [s + r for s in "SHDC" for r in RANKS]
    rng = random.Random(0)
    hands = []
    for _ in range(40):
        pick = rng.sample(deck, 13)
        hands.append(" ".join(f"{s}:{''.join(c[1] for c in pick if c[0] == s) or '-'}" for s in "SHDC"))
    hands += [
        "S:AKQJT98 H:A2 D:K3 C:A2", "S:- H:AKQJT9 D:AKQJ2 C:32", "S:5432 H:5432 D:432 C:32",
        "S:AKQ H:AKQ D:AKJ2 C:K32", "S:T9876543 H:2 D:32 C:32", "S:K2 H:Q32 D:J432 C:5432",
        "S:AQ32 H:KJ2 D:Q32 C:K32", "S:32 H:32 D:32 C:AKQJ9876",
    ]
    return tuple(hand_from_str(h) for h in hands)


@lru_cache(maxsize=200_000)
def _probe_values(bids: Tuple[str, ...], bid: str, dealer: int) -> Dict[str, frozenset]:
    """Each concept's values over the probe hands for one auction + bid --
    independent of the real hand, so cached (48 classifications per
    decision would cost ~50 ms per teacher/prompt build)."""
    auction = AuctionContext(bids=list(bids), bid=bid, dealer=dealer)
    seen: Dict[str, set] = {}
    for h in _probe_hands():
        for name, v in _authority().classify(auction, h).items():
            seen.setdefault(name, set()).add(v)
    return {name: frozenset(vals) for name, vals in seen.items()}


def hand_dependent_concepts(frame: Dict[str, Any]) -> Dict[str, bool]:
    """{concept: True if its truth, in THIS auction for THIS bid, depends on
    the hand}. False = decided by the auction/bid alone (a restatement)."""
    auction = auction_from_frame(frame)
    probe = _probe_values(tuple(auction.bids), auction.bid, auction.dealer)
    own = _authority().classify(auction, hand_from_str(frame["hand_str"]))
    return {name: len(probe.get(name, frozenset()) | {v}) > 1 for name, v in own.items()}


def vulnerability_of(frame: Dict[str, Any]) -> Optional[Tuple[bool, bool]]:
    """(our side vulnerable, their side vulnerable), or None when the frame
    does not say (nlg/explanation_frames.py stores "None" / "NS" / "EW" / "Both")."""
    v, you = frame.get("vulnerability"), (frame.get("table") or {}).get("you")
    if v not in ("None", "NS", "EW", "Both") or not you or you not in "NESW":
        return None
    ns, ew = v in ("NS", "Both"), v in ("EW", "Both")
    return (ns, ew) if you in "NS" else (ew, ns)


def vulnerability_line(frame: Dict[str, Any]) -> Optional[str]:
    """Prompt line; None when nobody is vulnerable, which keeps the prompt of
    the (100% non-vulnerable) WBridge5-based data unchanged."""
    vul = vulnerability_of(frame)
    if not vul or vul == (False, False):
        return None
    return {(True, True): "both sides are vulnerable",
            (True, False): "your side is vulnerable, the opponents are not (unfavourable)",
            (False, True): "the opponents are vulnerable, your side is not (favourable)"}[vul]


def _suit_of(bid: str) -> Optional[str]:
    if bid in NON_BIDS or len(bid) < 2:
        return None
    return None if bid[1:] in ("N", "NT") else bid[-1]


def _holding(h: HandFeatures, s: str) -> str:
    cards = h.cards[s]
    name = SUIT_NAME[s][:-1] if len(cards) == 1 else SUIT_NAME[s]
    return f"{len(cards)} {name} ({' '.join(cards)})" if cards else f"no {SUIT_NAME[s]} (void)"


_NEXT_SUIT = {"D": "H", "H": "S"}


def _conventional_meaning(prior: List[Dict], idx: int, you: str, partner: str) -> Optional[str]:
    """Plain-language meaning of partner's call at prior[idx] when it is a
    standard SAYC convention rather than a natural suit (Stayman, Jacoby
    transfers, Blackwood replies, the 2D waiting response to 2C), so the
    facts never call a transfer "partner's suit" or offer support points in it."""
    call = prior[idx]["bid"]
    ours_before = [b for b in prior[:idx] if b["seat"] in (you, partner) and b["bid"] not in NON_BIDS]
    my_before = [b for b in ours_before if b["seat"] == you]
    if call == "2C" and not [b for b in prior[:idx] if b["bid"] not in NON_BIDS]:
        return "the strong, artificial 2C opening, not a natural club suit"
    if call in ("4NT", "4N") and any(_suit_of(b["bid"]) for b in ours_before):
        return "Blackwood, asking how many aces you hold"
    if not my_before:
        return None
    mine = my_before[-1]["bid"]
    opp_bid_between = any(b["seat"] not in (you, partner) and b["bid"] not in NON_BIDS
                          for b in prior[prior.index(my_before[-1]) + 1:idx])
    if mine in ("4NT", "4N") and call in ("5C", "5D", "5H", "5S"):
        return "a Blackwood reply to your 4NT, showing a number of aces, not a natural suit"
    if opp_bid_between:
        return None
    first_real = next((b for b in prior if b["bid"] not in NON_BIDS), None)
    you_opened = first_real is not None and first_real["seat"] == you
    if you_opened and len(ours_before) == 1 and mine in ("1NT", "1N"):
        if call == "2C":
            return "Stayman, asking whether you hold a 4-card major, not a natural club suit"
        if call in ("2D", "2H"):
            return (f"a Jacoby transfer asking you to bid {SUIT_NAME[_NEXT_SUIT[call[1]]]}, "
                    f"showing at least 5 {SUIT_NAME[_NEXT_SUIT[call[1]]]}, not {SUIT_NAME[call[1]]}")
    if you_opened and len(ours_before) == 1 and mine == "2C" and call == "2D":
        return "the artificial 2D waiting response to your strong 2C, not a natural diamond suit"
    return None


_ACE_REPLY = {"5C": (0, 3), "5D": (1, 4), "5H": (2,), "5S": (3,)}
_KING_REPLY = {"6C": 0, "6D": 1, "6H": 2, "6S": 3, "6NT": 4}  # WBridge5: 6NT = 4 kings (66/66 in the data)


def blackwood_context(frame: Dict[str, Any]) -> Optional[str]:
    """What partner's reply to OUR 4NT / 5NT showed, resolved against the
    actor's own aces (5C = 0 or 3 etc. in this system) -- the arithmetic a
    slam decision rests on."""
    table = frame.get("table") or {}
    you, partner = table.get("you"), table.get("partner")
    prior = frame.get("auction_before") or []
    mine = [b["bid"] for b in prior if b["seat"] == you and b["bid"] not in NON_BIDS]
    theirs = [b["bid"] for b in prior if b["seat"] == partner and b["bid"] not in NON_BIDS]
    if not mine or not theirs:
        return None
    h = hand_from_str(frame["hand_str"])
    my_aces = sum("A" in h.cards[x] for x in "SHDC")
    my_kings = sum("K" in h.cards[x] for x in "SHDC")
    if mine[-1] in ("4NT", "4N") and theirs[-1] in _ACE_REPLY:
        fits = [o for o in _ACE_REPLY[theirs[-1]] if o + my_aces <= 4]
        if fits:
            p = max(fits)
            return (f"Partner's {theirs[-1]} reply to your Blackwood 4NT shows "
                    f"{' or '.join(map(str, _ACE_REPLY[theirs[-1]]))} aces; with your {my_aces}, partner has {p}, "
                    f"so the partnership holds {my_aces + p} of the 4 aces.")
    if mine[-1] in ("5NT", "5N") and theirs[-1] in _KING_REPLY:
        k = _KING_REPLY[theirs[-1]]
        return (f"Partner's {theirs[-1]} reply to your 5NT king ask shows {k} king{'s' if k != 1 else ''}; with your {my_kings}, "
                f"the partnership holds {my_kings + k} of the 4 kings.")
    return None


def partnership_aces(frame: Dict[str, Any]) -> Optional[int]:
    """Aces the partnership holds when the actor asked with 4NT and partner
    replied (resolved as in blackwood_context), else None."""
    table = frame.get("table") or {}
    prior = frame.get("auction_before") or []
    mine = [b["bid"] for b in prior if b["seat"] == table.get("you") and b["bid"] not in NON_BIDS]
    theirs = [b["bid"] for b in prior if b["seat"] == table.get("partner") and b["bid"] not in NON_BIDS]
    if not mine or not theirs or mine[-1] not in ("4NT", "4N") or theirs[-1] not in _ACE_REPLY:
        return None
    my_aces = sum("A" in hand_from_str(frame["hand_str"]).cards[x] for x in "SHDC")
    fits = [o for o in _ACE_REPLY[theirs[-1]] if o + my_aces <= 4]
    return my_aces + max(fits) if fits else None


def hand_facts(frame: Dict[str, Any]) -> List[str]:
    """Exact, pre-computed facts about the hand, read in the auction's
    context -- every number here is arithmetic on hand_str/auction_before,
    so an explanation citing them stays checkable by nlg.consistency_checks."""
    h = hand_from_str(frame["hand_str"])
    table = frame.get("table") or {}
    you, partner = table.get("you"), table.get("partner")
    prior = frame.get("auction_before") or []
    shape = "-".join(str(h.lengths[s]) for s in "SHDC")
    facts = [
        f"{h.hcp} HCP, {'balanced' if h.balanced else 'unbalanced'} shape {shape} "
        f"(spades-hearts-diamonds-clubs): " + ", ".join(_holding(h, s) for s in "SHDC") + ".",
    ]
    longest = max(h.lengths.values())
    facts.append("Longest suit: " + " and ".join(SUIT_NAME[s] for s in "SHDC" if h.lengths[s] == longest)
                 + f" ({longest} cards).")

    own = [b["bid"] for b in prior if b["seat"] == you and b["bid"] not in NON_BIDS]
    partner_bids = [b["bid"] for b in prior if b["seat"] == partner and b["bid"] not in NON_BIDS]
    opp_bids = [b["bid"] for b in prior if b["seat"] not in (you, partner) and b["bid"] not in NON_BIDS]
    if own:
        facts.append(f"Your own earlier bid(s): {', '.join(own)}.")
    if partner_bids:
        ps = _suit_of(partner_bids[-1])
        line = f"Partner's last bid: {partner_bids[-1]}."
        p_idx = max(i for i, b in enumerate(prior) if b["seat"] == partner and b["bid"] == partner_bids[-1])
        cuebid = ps is not None and any(_suit_of(b["bid"]) == ps for b in prior[:p_idx]
                                        if b["seat"] not in (you, partner))
        conv = _conventional_meaning(prior, p_idx, you, partner)
        if conv:
            # artificial: partner's suit is NOT a trump suit, so no
            # "holding in partner's suit" / support-point line
            line += f" That is {conv}."
        elif cuebid:
            # bidding the opponents' suit is artificial, not a natural suit --
            # "support points with spades as trumps" would be nonsense here
            line += (f" That is a cuebid of the opponents' {SUIT_NAME[ps]}, not a natural {SUIT_NAME[ps]} suit; "
                     f"you hold {_holding(h, ps)}.")
        elif ps:
            line += (f" You hold {_holding(h, ps)} in partner's suit; "
                     f"support points with {SUIT_NAME[ps]} as trumps: {support_points(h, ps)}.")
        facts.append(line)
    elif any(b["seat"] == partner for b in prior):
        facts.append("Partner has only passed so far.")
    opp_suits = []
    for b in opp_bids:
        s = _suit_of(b)
        if s and s not in opp_suits:
            opp_suits.append(s)
    opp_doubles = [b["bid"] for b in prior if b["seat"] not in (you, partner) and b["bid"] in ("X", "XX")]
    if opp_suits:
        facts.append("Opponents have bid: " + ", ".join(opp_bids) + ". In their suit(s) you hold " + "; ".join(
            f"{_holding(h, s)}, {'a' if h.stoppers[s] else 'no'} stopper (A, Kx, Qxx or Jxxx)" for s in opp_suits) + ".")
    elif opp_bids:
        facts.append("Opponents have bid: " + ", ".join(opp_bids) + " (no suit).")
    elif opp_doubles:
        facts.append("The opponents have only doubled, not bid a suit.")
    else:
        facts.append("The opponents have not bid (uncontested so far).")
    bid_suits = {_suit_of(b["bid"]) for b in prior} - {None}
    unbid = [s for s in "SHDC" if s not in bid_suits]
    if bid_suits and unbid:
        facts.append("Unbid suits: " + ", ".join(f"{SUIT_NAME[s]} ({h.lengths[s]})" for s in unbid) + ".")
    facts.append(f"Losing trick count: {losing_trick_count(h)}; quick tricks: {quick_tricks(h):g}; "
                 f"points (HCP plus shape, the system's measure for preempts and 2C): {wb5_points(h)}.")
    aces = [SUIT_NAME[x][:-1] for x in "SHDC" if "A" in h.cards[x]]
    kings = [SUIT_NAME[x][:-1] for x in "SHDC" if "K" in h.cards[x]]
    facts.append(f"Aces held: {len(aces)}" + (f" ({', '.join(aces)})" if aces else "")
                 + f"; kings held: {len(kings)}" + (f" ({', '.join(kings)})." if kings else "."))
    bw = blackwood_context(frame)
    if bw:
        facts.append(bw)
    if all(b["bid"] in NON_BIDS for b in prior):  # an opening decision
        facts.append(f"Rule of 20 (HCP + two longest suits >= 20): "
                     f"{'met' if rule_of_20_satisfied(h) else 'not met'}.")
    return facts


# Structural concepts describing the action itself, not why it was chosen:
# fine as the sole concept, never worth stacking with a second one.
_STRUCTURAL_ONLY = {"bid_is_pass", "bid_is_double", "bid_is_redouble", "bid_is_conventional"}
_BID_CATEGORY = {"bid_is_pass": ("P", "Pass"), "bid_is_double": ("X", "Dbl"),
                 "bid_is_redouble": ("XX", "Rdbl", "RDbl")}


def _select_concepts(frame: Dict[str, Any]) -> List[str]:
    """Fallback for frames without concept_evidence: the model's top predicted
    concepts. A predicted bid_is_pass/double/redouble that contradicts the real
    bid is dropped (the concept and policy heads can disagree on an imperfect
    checkpoint), and a structural concept is never stacked with a second one."""
    top = frame.get("top_concepts") or []
    bid = frame.get("bid", "?")
    names = [n for n, p in top if p >= 0.5] or [n for n, _p in top[:1]]
    names = [n for n in names if bid in _BID_CATEGORY.get(n, (bid,))]
    if not names:
        return []
    selected = names[:1]
    if names[0] not in _STRUCTURAL_ONLY:
        selected += [n for n in names[1:4] if n not in _STRUCTURAL_ONLY and n != names[0]][:1]
    return selected


def select_reasoned_concepts(frame: Dict[str, Any], k: int = 3) -> List[str]:
    """Up to k concepts that are (a) predicted by the model, (b) true per
    the rule engine, (c) hand-dependent in this auction, ranked by their
    contribution to the chosen bid over the runner-up. Falls back to
    _select_concepts when the frame lacks the evidence fields
    (non-enriched frames) or nothing qualifies."""
    evidence = frame.get("concept_evidence")
    if not evidence:
        return _select_concepts(frame)
    dep = hand_dependent_concepts(frame)
    gt = _gt(frame)
    ck = contribution_key(frame)
    # truth from the CURRENT rules, not the frame's stored flag (frames written
    # before a rule fix would otherwise keep offering a concept that is now false)
    ranked = [e for e in evidence if gt.get(e["name"], False) and e["act"] >= 0.5 and dep.get(e["name"], False)
              and e[ck] > 0]
    ranked.sort(key=lambda e: -e[ck])
    chosen = [e["name"] for e in ranked[:k]]
    if chosen:
        return chosen
    # no hand-dependent reason: fall back to any TRUE concept the model used, never a false
    # prediction ("East passes: the 20 HCP are mainly defensive" when passing partner's 6S)
    # (never the strain descriptors: "is a bid in clubs" is not a reason)
    ranked = [e for e in evidence if gt.get(e["name"], False) and e["act"] >= 0.5 and e[ck] > 0
              and not e["name"].startswith("bid_is_in_")]
    ranked.sort(key=lambda e: -e[ck])
    return [e["name"] for e in ranked[:k]]


RUNNER_UP_MIN_P = 0.05


def contribution_key(frame: Dict[str, Any]) -> str:
    """Which stored contribution ranks the reasons: the margin over the runner-up
    when the runner-up is a real alternative (p >= 5%), else the margin over all the
    alternatives weighed by their probability ("contribution_alt", nlg/explanation_frames.py;
    older frames have only the runner-up margin)."""
    ev = frame.get("concept_evidence") or []
    p = (frame.get("runner_up") or {}).get("p", 1.0)
    if ev and all("contribution_alt" in e for e in ev) and p < RUNNER_UP_MIN_P:
        return "contribution_alt"
    return "contribution"


# Too generic to tell the reader anything beyond the bid itself.
_GENERIC_MEANING = {"bid_is_pass", "bid_is_double", "bid_is_redouble", "bid_shows_major_suit",
                    "bid_shows_minor_suit", "bid_shows_4_plus_cards", "bid_is_in_clubs", "bid_is_in_diamonds",
                    "bid_is_in_hearts", "bid_is_in_spades", "bid_is_in_notrump"}


def bid_meaning_concepts(frame: Dict[str, Any], k: int = 2) -> List[str]:
    """The auction-decided (hand-INdependent) concepts that pushed toward the
    bid: not reasons about the hand, but what the call MEANS here (e.g. a
    takeout double, a raise of partner's suit) -- context the explanation
    needs, kept separate so it can't masquerade as a hand-based reason."""
    evidence = frame.get("concept_evidence") or []
    if not evidence:
        return []
    dep = hand_dependent_concepts(frame)
    gt = _gt(frame)  # current rules, not the frame's stored flag
    ck = contribution_key(frame)
    cands = [e for e in evidence if gt.get(e["name"], False) and e["act"] >= 0.5 and not dep.get(e["name"], True)
             and e[ck] > 0 and e["name"] not in _GENERIC_MEANING]
    cands.sort(key=lambda e: -e[ck])
    return [e["name"] for e in cands[:k]]


_CALL_VERB = {"P": "passed", "Pass": "passed", "X": "doubled", "XX": "redoubled"}


def narrate_auction(frame: Dict[str, Any]) -> str:
    """The auction as plain sentences with each caller's ROLE spelled out
    ("South (opponent) opened 1C. West (you) doubled."): from a compact
    "S(OPP):1C W(YOU):X" string Qwen3-8B misattributes openings."""
    table = frame.get("table") or {}
    you, partner = table.get("you"), table.get("partner")
    prior = frame.get("auction_before") or []
    if not prior:
        return "Nobody has bid yet: this is the first call of the auction."
    role = lambda s: "you" if s == you else "partner" if s == partner else "opponent"  # noqa: E731
    opened, out = False, []
    for b in prior:
        call = b["bid"]
        if call in _CALL_VERB:
            verb = _CALL_VERB[call]
        elif not opened:
            verb, opened = f"opened {call}", True
        else:
            verb = f"bid {call}"
        out.append(f"{SEAT_NAME.get(b['seat'], b['seat'])} ({role(b['seat'])}) {verb}.")
    if not opened:
        out.append("Nobody has opened yet.")
    else:
        opener = next(b for b in prior if b["bid"] not in _CALL_VERB)
        side = "your side" if role(opener["seat"]) in ("you", "partner") else "the OPPONENTS (not partner)"
        out.append(f"So the auction was opened by {side}: "
                   f"{SEAT_NAME.get(opener['seat'], opener['seat'])} ({role(opener['seat'])}).")
    return " ".join(out)


def natural_partner_suits(frame: Dict[str, Any]) -> List[str]:
    """Suits partner has shown NATURALLY, in bidding order: artificial calls
    (strong 2C opening, Stayman/transfers over our notrump, the 2D waiting
    reply to our 2C, Blackwood replies, cuebids of the opponents' suit) are
    skipped -- "2C P 4C" is not a raise of partner's clubs."""
    table = frame.get("table") or {}
    you, partner = table.get("you"), table.get("partner")
    prior = frame.get("auction_before") or []
    real = [(i, b) for i, b in enumerate(prior) if b["bid"] not in NON_BIDS]
    out = []
    for k, (i, b) in enumerate(real):
        if b["seat"] != partner or not _suit_of(b["bid"]):
            continue
        before = real[:k]
        call = b["bid"]
        if not before and call == "2C":
            continue
        mine = [x["bid"] for _, x in before if x["seat"] == you]
        if mine:
            m = mine[-1]
            if (m in ("1NT", "1N") and call in ("2C", "2D", "2H")) or (m in ("2NT", "2N") and call in ("3C", "3D", "3H")) \
                    or (m == "2C" and call == "2D") or (m in ("4NT", "4N") and call[0] == "5"):
                continue
        opp_suits = {_suit_of(x["bid"]) for _, x in before if x["seat"] not in (you, partner)}
        if _suit_of(call) in opp_suits:
            continue
        # after our side has agreed a suit (both bid it, or partner's Jacoby
        # 2NT over our 1M), partner's new suit at the 4 level or higher is a
        # control cuebid, not a suit to support
        ours_before = [(x["seat"], x["bid"]) for _, x in before if x["seat"] in (you, partner)]
        mine_s = {_suit_of(bd) for st, bd in ours_before if st == you} - {None}
        part_s = {_suit_of(bd) for st, bd in ours_before if st == partner} - {None}
        agreed = mine_s & part_s
        # Jacoby 2NT either way round: 1M by one of us, 2NT by the other
        if len(ours_before) >= 2 and ours_before[0][1] in ("1H", "1S") and ours_before[1][1] in ("2NT", "2N") \
                and ours_before[0][0] != ours_before[1][0]:
            agreed |= {ours_before[0][1][-1]}
        if agreed and _suit_of(call) not in agreed and int(call[0]) >= 4:
            continue
        out.append(_suit_of(call))
    return out


def _level(call: str) -> int:
    return int(call[0]) if call and call[0].isdigit() else 0


def _strain(call: str) -> Optional[str]:
    if call in NON_BIDS or not call or not call[0].isdigit():
        return None
    return "N" if call[1:] in ("N", "NT") else call[1]


def _is_jump(bid: str, last_real: str) -> bool:
    """True if `bid` is higher than the cheapest legal bid in its strain
    over `last_real` (the auction's last non-pass/double call)."""
    order = "CDHSN"
    cheapest = _level(last_real) + (0 if order.index(_strain(bid)) > order.index(_strain(last_real)) else 1)
    return _level(bid) > cheapest


def bid_category(frame: Dict[str, Any]) -> str:
    """Deterministic bidding-situation label for one decision -- who opened,
    whether we're opener/responder/overcaller/advancer, and what kind of
    call this is. Labels the gold examples (nlg/gold_examples.jsonl),
    stratifies the teacher dataset and measures coverage across situations."""
    table = frame.get("table") or {}
    you, partner = table.get("you"), table.get("partner")
    prior = frame.get("auction_before") or []
    bid = frame["bid"]
    ours = lambda s: s in (you, partner)  # noqa: E731
    real = [b for b in prior if b["bid"] not in NON_BIDS]
    lvl, st = _level(bid), _strain(bid)
    partner_last = next((b["bid"] for b in reversed(prior) if b["seat"] == partner and b["bid"] != "P"), None)

    ours_suit_bid = lambda upto: any(  # noqa: E731
        ours(b["seat"]) and b["bid"] not in NON_BIDS and _suit_of(b["bid"]) for b in prior[:upto])
    # slam machinery first: it looks the same whoever opened. 4NT is only
    # Blackwood once our side has bid a suit; otherwise (e.g. over the
    # opponents' 1S-2C) it is the Unusual 4NT and 5C/5D picks a minor.
    if bid in ("4NT", "4N") and ours_suit_bid(len(prior)):
        return "slam_blackwood_ask"
    if partner_last in ("4NT", "4N") and bid in ("5C", "5D", "5H", "5S"):
        p4 = max(i for i, b in enumerate(prior) if b["seat"] == partner and b["bid"] in ("4NT", "4N"))
        if ours_suit_bid(p4):
            return "slam_blackwood_answer"
    if partner_last in ("5NT", "5N") and bid in ("6C", "6D", "6H", "6S", "6NT") and any(
            b["seat"] == partner and b["bid"] in ("4NT", "4N") for b in prior) and ours_suit_bid(len(prior)):
        return "slam_blackwood_answer"  # the king-ask reply (6C = 0 ... 6S = 3 kings), not a slam bid
    if lvl >= 6:
        return "slam_bid"
    if bid == "XX":
        # rare in every position (e.g. 1D X XX, showing 10+ HCP); one situation
        return "redouble"

    if not real:  # nobody has opened
        if bid == "P":
            return "opening_pass"
        if st == "N":
            return {1: "opening_1nt", 2: "opening_2nt"}.get(lvl, "opening_preempt")
        if bid == "2C":
            return "opening_2c_strong"
        if lvl == 2:
            return "opening_weak_two"
        if lvl >= 3:
            return "opening_preempt"
        return "opening_1_major" if st in ("H", "S") else "opening_1_minor"

    opener = real[0]
    opp_real = [b for b in real if not ours(b["seat"])]
    my_real = [b for b in real if b["seat"] == you]
    partner_real = [b for b in real if b["seat"] == partner]
    partner_strains = set(natural_partner_suits(frame)) | {
        "N" for b in partner_real if _strain(b["bid"]) == "N"}
    my_strains = {_strain(b["bid"]) for b in my_real} - {None}
    opp_strains = {_strain(b["bid"]) for b in opp_real} - {None, "N"}
    contested = bool(opp_real)
    opp_doubled = any(not ours(b["seat"]) and b["bid"] in ("X", "XX") for b in prior)
    partner_doubled = any(b["seat"] == partner and b["bid"] == "X" for b in prior)

    if ours(opener["seat"]):
        if opener["seat"] == you:  # I opened: this is a rebid
            if bid == "P":
                return "opener_pass_competitive" if contested else "opener_pass"
            if bid == "X":
                return "opener_double"
            if st in partner_strains:
                return "opener_raises_responder"
            if st in my_strains:
                return "opener_rebids_own_suit"
            if st == "N":
                return "opener_rebid_notrump"
            return "opener_new_suit"
        # partner opened: I'm responder
        if not my_real:
            if contested and opp_real[-1]["seat"] != partner:
                if bid == "X":
                    # a negative double is the FIRST call right over an
                    # overcall of partner's one-level suit opening; any other
                    # double here (after passing, over a preempt) is not one
                    # (a passed hand may still make one: P P P 1H 2C X)
                    o_idx = next(i for i, b in enumerate(prior) if b["bid"] not in NON_BIDS)
                    first_call = not any(b["seat"] == you for b in prior[o_idx + 1:])
                    if first_call and _level(opener["bid"]) == 1 and _strain(opener["bid"]) != "N":
                        return "responder_negative_double"
                    return "competitive_later_double"
                if bid == "P":
                    return "responder_pass_competitive"
                if st in partner_strains:
                    return "responder_raise_competitive"
                return "responder_bid_competitive"
            if _strain(opener["bid"]) == "N" and _level(opener["bid"]) == 1 and not opp_doubled:
                # after 1NT-X the system is off: those responses are competitive
                if bid == "2C":
                    return "response_1nt_stayman"
                if bid in ("2D", "2H"):
                    return "response_1nt_transfer"
                return "response_1nt_pass" if bid == "P" else "response_1nt_other"
            if bid == "P":
                return "response_pass"
            if st in partner_strains:
                jump = lvl - _level(opener["bid"])
                if st in ("H", "S") and lvl == 4:
                    return "response_game_raise"
                return "response_simple_raise" if jump <= 1 else "response_jump_raise"
            if st == "N":
                return "response_notrump"
            if _is_jump(bid, real[-1]["bid"]):
                return "response_jump_shift"
            return "response_new_suit_1_level" if lvl == 1 else "response_new_suit_2_level"
        # responder's later calls
        if bid == "P":
            return "responder_later_pass"
        if st in partner_strains:
            return "responder_later_support"
        if st == "N":
            return "responder_later_notrump"
        return "responder_later_other"

    # the opponents opened
    if not my_real and not partner_real and partner_doubled:
        # answering partner's takeout double (not an overcall)
        if bid == "P":
            return "advancer_pass"
        if bid == "X":
            return "advancer_double"
        if st == "N":
            return "advancer_notrump"
        if st in opp_strains:
            return "advancer_cuebid"
        return "advancer_new_suit"
    if not my_real and not partner_real:
        if bid == "X":
            return "takeout_double"
        if bid == "P":
            return "pass_over_opponents_opening"
        if st == "N":
            return "overcall_notrump"
        if st in opp_strains:
            return "overcall_cuebid"
        return "overcall_jump" if _is_jump(bid, real[-1]["bid"]) else "overcall_suit"
    if partner_real and not my_real:  # I'm the advancer
        if bid == "P":
            return "advancer_pass"
        if bid == "X":
            return "advancer_double"
        if st in partner_strains:
            return "advancer_raise"
        if st in opp_strains:
            return "advancer_cuebid"
        if st == "N":
            return "advancer_notrump"
        return "advancer_new_suit"
    if bid == "P":
        return "competitive_later_pass"
    if bid == "X":
        return "competitive_later_double"
    if st in partner_strains:
        return "competitive_later_support"
    return "competitive_later_bid"


# ---------------------------------------------------------------------------
# Deviation notes: bids that do not meet their own situation's requirements
# ---------------------------------------------------------------------------
# The policy imitates WBridge5 at ~83-95% accuracy, so some of its bids fit
# neither textbook SAYC nor WBridge5's system (e.g. a weak 2S with Kx of
# spades). An explanation must not invent a justification for those; the
# prompt instead gets an explicit note so the explanation can say the bid is
# unusual and why. WBridge5's own variants (5-card third-seat weak twos, its
# 2C and Blackwood schemes...) are part of the rule engine, so they are
# NOT flagged. Calibrated with scripts/audit_concepts.py-style replays.

@lru_cache(maxsize=200_000)
def _gt_cached(bids: Tuple[str, ...], bid: str, dealer: int, hand_str: str) -> Dict[str, bool]:
    return _authority().classify(AuctionContext(bids=list(bids), bid=bid, dealer=dealer), hand_from_str(hand_str))


def _gt(frame: Dict[str, Any]) -> Dict[str, bool]:
    """Concept truth under the CURRENT rules (cached; treat as read-only)."""
    a = auction_from_frame(frame)
    return _gt_cached(tuple(a.bids), a.bid, a.dealer, frame["hand_str"])


def _can_open(h: HandFeatures) -> bool:
    return h.hcp >= 12 or (h.hcp >= 11 and wb5_points(h) >= 13) or rule_of_20_satisfied(h)


def _clearly_too_weak_to_open(h: HandFeatures) -> bool:
    # WBridge5 opens light hands with a good long suit (10 HCP 1H with
    # AQJT65), so only a clear shortfall is unusual
    lens = sorted(h.lengths.values(), reverse=True)
    return h.hcp < 10 or (not _can_open(h) and wb5_points(h) < 13 and h.hcp + lens[0] + lens[1] < 19)


def _low_level_double_with_length(frame: Dict[str, Any], h: HandFeatures) -> bool:
    real = [b["bid"] for b in frame.get("auction_before") or [] if b["bid"] not in NON_BIDS]
    if any(_suit_of(b) is None for b in real):
        return False  # doubles in notrump auctions (e.g. of a transfer) are not takeout doubles
    last = real[-1] if real else None
    # WBridge5 doubles a 2-level bid holding length with strong hands (double
    # then bid its own suit), so only a 1-level double, or a weak one, is unusual
    return (bool(last) and _suit_of(last) is not None and h.lengths.get(_suit_of(last), 0) >= 4
            and (last[0] == "1" or (last[0] == "2" and h.hcp < 15)))


def _wrong_minor(h: HandFeatures, s: str) -> bool:
    """The opened minor is not the one the system opens (longer minor; with
    equal length 1D from four-four up, 1C with three-three), and only when
    no 5-card major changes the picture."""
    d, c = h.lengths["D"], h.lengths["C"]
    if max(h.lengths["S"], h.lengths["H"]) >= 5:
        return False
    expected = "D" if d > c else "C" if c > d else ("D" if d >= 4 else "C")
    return s != expected


def support_double_suit(frame: Dict[str, Any]) -> Optional[str]:
    """Partner's suit when this call is opener's support double: 1x (P or X)
    1y/2y (RHO's suit bid) X. In WBridge5's data opener's double there holds
    exactly three cards in partner's suit -- after LHO's pass, RHO at the
    1/2/3 level: 86%/97%/80% of 4,078/4,598/341 doubles; after LHO's takeout
    double, RHO at the 1/2 level: 88%/96% of 679/1,391 -- while the direct
    raise holds four, and any length in RHO's suit. The rules call it a
    penalty double. Not after an overcall by LHO too (60%/19%:
    a different, competitive double)."""
    if frame["bid"] not in ("X", "Dbl"):
        return None
    t = frame.get("table") or {}
    you, partner = t.get("you"), t.get("partner")
    seq = list(frame.get("auction_before") or [])
    while seq and seq[0]["bid"] in ("P", "Pass"):
        seq.pop(0)
    if len(seq) != 4:
        return None
    mine, lho, resp, rho = seq
    m, r, z = _suit_of(mine["bid"]), _suit_of(resp["bid"]), _suit_of(rho["bid"])
    if not (mine["seat"] == you and mine["bid"][:1] == "1" and m and lho["bid"] in ("P", "Pass", "X", "Dbl")
            and resp["seat"] == partner and resp["bid"][:1] in "12" and r and r != m
            and rho["seat"] not in (you, partner) and z and z not in (m, r)):
        return None
    top = 2 if lho["bid"] in ("X", "Dbl") else 3
    return r if int(rho["bid"][0]) <= top else None


def support_redouble_suit(frame: Dict[str, Any]) -> Optional[str]:
    """Partner's suit when this call is opener's support redouble: 1x (X or P)
    1y (X) XX -- WBridge5's opener held exactly three of partner's suit 290 of
    290 times after LHO's double, 3,090 of 3,140 after LHO's pass."""
    if frame["bid"] not in ("XX", "RDbl"):
        return None
    t = frame.get("table") or {}
    seq = list(frame.get("auction_before") or [])
    while seq and seq[0]["bid"] in ("P", "Pass"):
        seq.pop(0)
    if len(seq) != 4:
        return None
    mine, lho, resp, rho = seq
    m, r = _suit_of(mine["bid"]), _suit_of(resp["bid"])
    if (mine["seat"] == t.get("you") and mine["bid"][:1] == "1" and m and lho["bid"] in ("X", "Dbl", "P", "Pass")
            and resp["seat"] == t.get("partner") and resp["bid"][:1] == "1" and r and r != m
            and rho["bid"] in ("X", "Dbl")):
        return r
    return None


def responsive_double_suits(frame: Dict[str, Any]) -> Optional[List[str]]:
    """The unbid suits when this call is a responsive double: (1x) X (2x) X --
    WBridge5's advancer held four or more cards in two unbid suits 1,604 of
    1,604 times, with 5/8/10 HCP."""
    if frame["bid"] not in ("X", "Dbl"):
        return None
    t = frame.get("table") or {}
    seq = list(frame.get("auction_before") or [])
    while seq and seq[0]["bid"] in ("P", "Pass"):
        seq.pop(0)
    if len(seq) != 3:
        return None
    op, dbl, rho = seq
    o = _suit_of(op["bid"])
    if (op["seat"] not in (t.get("you"), t.get("partner")) and op["bid"][:1] == "1" and o
            and dbl["seat"] == t.get("partner") and dbl["bid"] in ("X", "Dbl")
            and rho["bid"] in ("2" + o, "3" + o)):
        return [x for x in "SHDC" if x != o]
    return None


def competitive_double_suit(frame: Dict[str, Any]) -> Optional[str]:
    """Partner's suit when this call is opener's competitive double after both
    opponents bid: 1x (overcall) 1y/2y (RHO's suit at the 2-3 level) X. In
    WBridge5's data partner bids on after it 94% of the time (passed 47 of
    792 at the 2 level, 44 of 383 at the 3), opener has 13-14 HCP median and
    any length in RHO's suit -- not the penalty double the rules name."""
    if frame["bid"] not in ("X", "Dbl"):
        return None
    t = frame.get("table") or {}
    you, partner = t.get("you"), t.get("partner")
    seq = list(frame.get("auction_before") or [])
    while seq and seq[0]["bid"] in ("P", "Pass"):
        seq.pop(0)
    if len(seq) != 4:
        return None
    mine, lho, resp, rho = seq
    m, o, r, z = (_suit_of(x["bid"]) for x in seq)
    if (mine["seat"] == you and mine["bid"][:1] == "1" and m and o and o != m
            and resp["seat"] == partner and resp["bid"][:1] in "12" and r and r not in (m, o)
            and rho["seat"] not in (you, partner) and z and z != r and rho["bid"][:1] in "23"):
        return r
    return None


STRONG_PASS_MIN_N, STRONG_PASS_MARGIN = 50, 2


def strong_pass_note(frame: Dict[str, Any], h=None) -> Optional[str]:
    """A pass with clearly more than WBridge5 passes with in this exact auction: its
    measured 95th percentile plus 2 HCP or more, on 50+ of its passes there (e.g. an
    advancer passing with 13 HCP and four-card support, where the concepts give no
    reason and "West has no reason to bid" would be wrong)."""
    if frame["bid"] not in ("P", "Pass"):
        return None
    from nlg.call_meaning import wb5_entry
    h = h or hand_from_str(frame["hand_str"])
    e = wb5_entry([b["bid"] for b in frame.get("auction_before") or []], "P")
    if not e or e["n"] < STRONG_PASS_MIN_N or h.hcp < int(e["hcp"][2]) + STRONG_PASS_MARGIN:
        return None
    return (f"WBridge5 passes here with {int(e['hcp'][0])} to {int(e['hcp'][2])} HCP (the middle 90% of its "
            f"{e['n']:,} passes in this exact auction); this hand has {h.hcp} HCP.")


def deviation_notes(frame: Dict[str, Any], explanation_notes: bool = True) -> List[str]:
    """Plain-language notes on how THIS bid departs from what its situation
    normally requires; [] when it is a normal bid for the system.

    explanation_notes=False leaves out the two notes that only the explanations use
    -- a natural notrump without a stopper in their suit (WBridge5 itself
    bids one 7% of the time) and a pass stronger than WBridge5's own passes in the
    auction -- so that the thesis's system-breaking measure (evaluation/convention_drift.py,
    the anchor ablation, the convention cards) keeps the definition all its numbers
    were measured with: counted in, WBridge5's own rate rose from 0.43% to 0.58%."""
    cat = bid_category(frame)
    h = hand_from_str(frame["hand_str"])
    bid = frame["bid"]
    s = _suit_of(bid)
    L = h.lengths.get(s, 0) if s else 0
    g = _gt(frame)
    notes: List[str] = []
    suit = SUIT_NAME.get(s, "")
    if L == 1 and suit:
        suit = suit[:-1]  # "1 spade", not "1 spades"
    if cat in ("opening_1_major", "opening_1_minor"):
        need = 5 if s in ("H", "S") else 3
        if L < need:
            notes.append(f"A {bid} opening normally promises {need}+ {suit}; this hand has only {L}.")
        seat = len(frame.get("auction_before") or []) + 1
        if _clearly_too_weak_to_open(h) and (seat <= 2 or h.hcp < 8):  # light 3rd/4th-seat openings are normal
            if rule_of_20_satisfied(h):  # (not "needs ... or the Rule of 20" right after "the Rule of 20 is met")
                notes.append(f"With only {h.hcp} HCP the hand is light even for a Rule of 20 opening.")
            else:
                notes.append(f"A one-level opening normally needs about 12+ HCP (or the Rule of 20); "
                             f"this hand has {h.hcp} HCP.")
        if s in ("C", "D") and _wrong_minor(h, s):  # (brl's RL model opened 1D on 4 diamonds and 5 clubs)
            notes.append(f"With {h.lengths['D']} diamonds and {h.lengths['C']} clubs the system opens "
                         f"1{'D' if s == 'C' else 'C'} (the longer minor; 1D with four-four, 1C with three-three).")
    elif cat == "opening_1nt" and not (14 <= h.hcp <= 18 and (h.balanced or sorted(h.lengths.values())[0] >= 2)):
        notes.append(f"A 1NT opening normally shows a balanced 15-17 HCP; this hand has {h.hcp} HCP"
                     f"{'' if h.balanced else ' and is unbalanced'}.")
    elif cat == "opening_2nt" and not 19 <= h.hcp <= 22:
        notes.append(f"A 2NT opening normally shows a balanced 20-21 HCP; this hand has {h.hcp} HCP.")
    elif cat == "opening_2c_strong" and not g.get("bid_is_strong_2cc_opening"):
        notes.append(f"A strong 2C opening normally needs about 21+ points or a 4-loser hand; this hand has "
                     f"{h.hcp} HCP and {wb5_points(h)} points.")
    elif cat == "opening_weak_two" and not g.get("bid_is_weak_two"):
        notes.append(f"A weak two normally shows a 6-card suit (5 in third seat) and under 12 HCP; this hand has "
                     f"{L} {suit} and {h.hcp} HCP.")
    elif cat == "opening_preempt" and s and not g.get("bid_is_unusual_4nt_opening"):
        need = 7 if (bid[0].isdigit() and int(bid[0]) >= 4) else 6
        if L < need or h.hcp >= 13:
            notes.append(f"A {bid} preempt normally shows {need}+ {suit} and under 12 HCP; this hand has "
                         f"{L} {suit} and {h.hcp} HCP.")
    elif cat == "opening_pass" and (h.hcp >= 13 or (h.hcp >= 12 and rule_of_20_satisfied(h))):
        notes.append(f"With {h.hcp} HCP the hand would normally open rather than pass.")
    elif cat == "overcall_suit" and s and L <= 3:
        notes.append(f"A suit overcall normally shows 5+ cards; this hand has only {L} {suit}.")
    elif cat == "overcall_jump" and s and L <= 4:
        notes.append(f"A jump overcall normally shows a long suit; this hand has only {L} {suit}.")
    elif cat == "takeout_double" and _low_level_double_with_length(frame, h):
        notes.append("A takeout double of a low-level bid normally shows shortness in the doubled suit; "
                     "this hand has 4+ cards there.")
    elif cat == "response_1nt_stayman" and not g.get("bid_stayman_with_4_card_major"):
        notes.append("Stayman normally shows a 4-card major; this hand has none.")
    elif cat == "response_1nt_transfer" and not g.get("bid_transfer_with_5_card_major"):
        target = {"D": "hearts", "H": "spades"}.get(s, "the next suit")
        notes.append(f"A transfer normally shows 5+ {target}; this hand has fewer.")
    elif cat == "response_new_suit_1_level" and s and L < 4:
        notes.append(f"A new-suit response normally shows 4+ cards; this hand has {L} {suit}.")
    elif cat == "response_simple_raise" and s and L < 3:
        notes.append(f"A raise normally shows 3+ card support; this hand has {L} {suit}.")
    elif cat == "slam_blackwood_answer" and bid[:1] == "6":
        kings = sum("K" in h.cards[x] for x in "SHDC")
        if _KING_REPLY.get(bid) != kings:
            notes.append(f"This reply to the 5NT king ask does not match the {kings} king{'s' if kings != 1 else ''} "
                         f"held (6C = 0, 6D = 1, 6H = 2, 6S = 3, 6NT = 4).")
    elif cat == "slam_blackwood_answer" and bid in _ACE_REPLY:
        # checked against the aces held, not the rule concepts, which fail after an opponent
        # doubles 4NT (5C with no ace would be called a mismatch)
        aces = sum("A" in h.cards[x] for x in "SHDC")
        if aces not in _ACE_REPLY[bid]:
            notes.append(f"This reply to Blackwood does not match the {aces} ace(s) held "
                         f"(5C = 0 or 3, 5D = 1 or 4, 5H = 2).")
    # a natural 1NT/3NT over the opponents' suit(s), the partnership's first notrump: WBridge5 holds a
    # stopper (A, Kx, Qxx, Jxxx) in every suit they bid 93% of the time (120,000 training deals);
    # without the note a 3NT without one gets explained as "the A 8 5 4 of diamonds stops the
    # opponents' suit" with 8 3 in their hearts
    real_ = [b for b in frame.get("auction_before") or [] if b["bid"] not in NON_BIDS]
    t_ = frame.get("table") or {}
    ours_ = (t_.get("you"), t_.get("partner"))
    if (explanation_notes and bid in ("1NT", "1N", "3NT", "3N") and not g.get("bid_is_conventional") and not g.get("bid_is_unusual_2nt")
            and not any(b["bid"].endswith(("NT", "N")) for b in real_ if b["seat"] in ours_)):
        theirs = []
        for b in real_:
            x = _suit_of(b["bid"])
            if b["seat"] not in ours_ and x and x in opponent_natural_suits(auction_from_frame(frame)) and x not in theirs:
                theirs.append(x)
        bare = [x for x in theirs if not is_stopper(h.cards[x])]
        if bare:
            x = bare[0]
            notes.append(f"A natural {bid} over the opponents' bidding normally shows a stopper in each of their suits; "
                         f"this hand has no stopper in {SUIT_NAME[x]} ({' '.join(h.cards[x]) or 'void'}).")
    note = strong_pass_note(frame, h) if explanation_notes else None
    if note and cat != "opening_pass":  # (that one has its own note above)
        notes.append(note)
    pa = partnership_aces(frame)
    if pa is not None and pa <= 2 and bid[:1] == "6":
        notes.append(f"A small slam is normally bid missing at most one ace; the partnership holds only {pa} "
                     f"of the 4 aces.")
    # 1NT (2x) 3y is forcing in WBridge5 (responder: 5+ cards, 10 HCP median,
    # 8 at the 10th percentile, 6 at least; nlg.teacher "nt_forcing_new_suit")
    real = [b for b in frame.get("auction_before") or [] if b["bid"] not in NON_BIDS]
    t = frame.get("table") or {}
    if (s and bid[:1] == "3" and len(real) == 2 and real[0]["seat"] == t.get("partner")
            and real[0]["bid"] in ("1NT", "1N") and real[1]["seat"] not in (t.get("you"), t.get("partner"))
            and real[1]["bid"][:1] == "2" and _suit_of(real[1]["bid"]) not in (None, s) and h.hcp <= 6):
        notes.append(f"A new suit at the three level over the interference is forcing and needs about 8+ HCP "
                     f"opposite the 1NT opening; this hand has {h.hcp} HCP.")
    sr = support_redouble_suit(frame)
    if sr and h.lengths[sr] != 3:
        notes.append(f"A support redouble normally shows exactly three cards in partner's suit; this hand has "
                     f"{h.lengths[sr]} {SUIT_NAME[sr] if h.lengths[sr] != 1 else SUIT_NAME[sr][:-1]}.")
    sd = support_double_suit(frame)
    if sd and h.lengths[sd] != 3:
        notes.append(f"A support double normally shows exactly three cards in partner's suit; this hand has "
                     f"{h.lengths[sd]} {SUIT_NAME[sd] if h.lengths[sd] != 1 else SUIT_NAME[sd][:-1]}.")
    return notes
