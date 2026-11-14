"""The rule-based teacher: writes the explanation of any decision in the style of the gold examples, so the
Qwen explainer can be trained at scale. Structure: headline (what the call means here) -> hand evidence for each
reason -> contrast with the runner-up -> a note for an unusual call. It cites only facts that are in the
explainer's prompt, and every output must pass nlg.consistency_checks and the style check."""
from __future__ import annotations

import functools
import random
import re
from typing import Any, Callable, Dict, List, Optional

from concepts import gloss_for_concept
from concepts.predicates_hand import losing_trick_count, quick_tricks, support_points, wb5_points
from concepts.hand import HCP_VALUES, is_stopper
from concepts.predicates_hand import is_semi_balanced, rule_of_20_satisfied
from nlg.double_meaning import leave_in_rate
from concepts.predicates_auction import natural_suits_of, opponent_natural_suits
from nlg.explanation_inputs import (
    SEAT_NAME, SUIT_NAME, _gt, _is_jump, _suit_of, auction_from_frame, bid_category, bid_meaning_concepts, blackwood_context,
    competitive_double_suit, deviation_notes, strong_pass_note, hand_from_str, natural_partner_suits, responsive_double_suits,
    select_reasoned_concepts, support_double_suit, support_redouble_suit,
    vulnerability_of,
)

# calls whose suit is NOT a natural suit of the bidder's (never "cite the suit")
_ARTIFICIAL = ("bid_is_conventional", "bid_is_stayman", "bid_is_jacoby_transfer", "bid_is_waiting_response",
               "bid_is_blackwood_response", "bid_is_control_bid", "bid_is_michaels_cuebid", "bid_is_cuebid",
               "bid_is_2nt_stayman", "bid_shows_stayman_major_structure", "bid_answers_king_ask",
               "bid_shows_aces_0_or_3", "bid_shows_aces_1_or_4", "bid_shows_aces_2", "bid_shows_aces_3",
               "bid_is_gerber", "bid_is_gerber_response", "bid_is_splinter", "bid_is_singleton_splinter",
               "bid_is_void_splinter")
_PREEMPTIVE = ("bid_is_weak_jump_overcall", "bid_shows_preemptive_intent", "bid_is_preemptive_jump_response",
               "bid_is_preemptive_raise", "bid_is_sacrifice_candidate", "bid_is_weak_two")

NB = ("P", "X", "XX", "Pass", "Dbl", "RDbl")
_WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
          9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen"}


# ---------------------------------------------------------------------------
# small phrase helpers (every number comes from the hand/auction)
# ---------------------------------------------------------------------------

def holding(h, s: str) -> str:
    cards = h.cards.get(s, [])
    if not cards:
        return f"a void in {SUIT_NAME[s]}"
    if len(cards) == 1:
        return f"a singleton {SUIT_NAME[s][:-1]} ({cards[0]})"
    return f"{len(cards)} {SUIT_NAME[s]} ({' '.join(cards)})"


def top_honors(h, s: str) -> str:
    hon = [r for r in h.cards.get(s, []) if r in "AKQJ"]
    names = {"A": "ace", "K": "king", "Q": "queen", "J": "jack"}
    if not hon:
        return f"no honor in {SUIT_NAME[s]}"
    return "the " + " and ".join(names[r] for r in hon[:2])


def rule_of_20(h) -> str:
    l1, l2 = sorted(h.lengths.values(), reverse=True)[:2]
    return f"{h.hcp} + {l1} + {l2} = {h.hcp + l1 + l2}"


def shape(h) -> str:
    return "-".join(str(h.lengths[s]) for s in "SHDC")


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def count_phrase(h, s: str) -> str:
    n = h.lengths[s]
    return f"no {SUIT_NAME[s]}" if n == 0 else f"1 {SUIT_NAME[s][:-1]}" if n == 1 else f"{n} {SUIT_NAME[s]}"


def _negative_double_majors(c) -> List[str]:
    """Opener's view: the major(s) partner's negative double promised --
    partner's only call is a double made directly over the opponents' 1- or
    2-level suit overcall of my one-level suit opening (1D 1H X: spades;
    1C 1D X: both majors; 1H 1S X: none)."""
    real = [b for b in c.prior if b["bid"] not in NB]
    if not real or real[0]["seat"] != c.you or c.partner_bids or len(real) < 2:
        return []
    opening, over = real[0]["bid"], real[1]
    if opening[0] != "1" or _suit_of(opening) is None or over["seat"] in (c.you, c.partner_seat) \
            or not _suit_of(over["bid"]) or over["bid"][0] not in "12":
        return []
    k = c.prior.index(over)
    if k + 1 >= len(c.prior) or c.prior[k + 1]["seat"] != c.partner_seat or c.prior[k + 1]["bid"] != "X":
        return []
    return [m for m in "HS" if m not in (_suit_of(opening), _suit_of(over["bid"]))]


def _partner_promise(c):
    """(partner's call, promised length) when partner's natural call in the
    suit was a preempt -- an opening weak two / 3- or 4-level opening, or a
    jump overcall -- which promises a long suit, so short support can do."""
    ps = c.partner_suit
    if not ps:
        return None
    real = [b for b in c.prior if b["bid"] not in NB]
    # a suit partner bid twice (and we never bid) promises six: 1S ... 2S
    twice = [b["bid"] for b in real if b["seat"] == c.partner_seat and _suit_of(b["bid"]) == ps]
    if len(twice) >= 2 and ps not in {_suit_of(b) for b in c.my_bids}:
        return " and ".join(twice[-2:]), 6
    for k in range(len(real) - 1, -1, -1):
        b = real[k]
        if b["seat"] != c.partner_seat or _suit_of(b["bid"]) != ps:
            continue
        lvl = int(b["bid"][0])
        if k == 0 and b["bid"] != "2C" and lvl >= 2:
            third_seat = c.prior.index(b) == 2
            return b["bid"], (5 if lvl == 2 and third_seat else 7 if lvl >= 4 else 6)
        ours_before = any(x["seat"] in (c.you, c.partner_seat) for x in real[:k])
        if k > 0 and not ours_before and real[k - 1]["seat"] not in (c.you, c.partner_seat) \
                and _is_jump(b["bid"], real[k - 1]["bid"]):
            return b["bid"], 6
        return None
    return None


def _direct_cuebid_suit(c, bidder: str) -> Optional[str]:
    """The opening suit when `bidder`'s first call was a direct Michaels-style
    cuebid of the opponents' one-level suit opening (1S 2S, 1D 2D)."""
    real = [b for b in c.prior if b["bid"] not in NB]
    if len(real) < 2 or real[0]["seat"] in (c.you, c.partner_seat):
        return None
    op, cb = real[0], real[1]
    if (cb["seat"] != bidder or op["bid"][0] != "1" or _suit_of(op["bid"]) is None
            or cb["bid"] != "2" + op["bid"][1:] or c.prior.index(cb) != c.prior.index(op) + 1):
        return None
    return _suit_of(op["bid"])


def _conventional_call(c) -> Optional[str]:
    """Conventional follow-ups the rule engine does not model: the advancer's 2NT after partner's Michaels
    (asks for the minor) and the Michaels bidder's answer; Stayman and
    transfers after partner's 2C-2D-2NT; returning to the agreed suit after
    partner's slam try (a sign-off, not "a preemptive raise") or game try.
    And a new suit at the three level over an overcall of partner's 1NT:
    forcing in WBridge5 (opener passed 7 of 566 times; responder holds a
    5+ suit and median 10 HCP), not the rules' "competes for the
    partscore"/"preemptive" (which would call 1NT (2S) 3H with 8 HCP "the
    preempt ... even though vulnerable"). A new suit
    at the two level there is competitive (opener passed 69% of 407; 5+
    cards, at most 9 HCP)."""
    b = c.bid
    real = [x for x in c.prior if x["bid"] not in NB]
    doubled = any(x["bid"] in ("X", "XX", "Dbl", "RDbl") for x in c.prior)
    # 1NT (P) 3D (P) 4C: the notrump opener's new suit at the 4 level agrees partner's
    # long suit and shows a control (not "showing a second suit ... 2 clubs (K 4)")
    if (c.my_bids[:1] in (["1NT"], ["1N"], ["2NT"], ["2N"]) and (c.opener or {}).get("seat") == c.you
            and b[:1] == "4" and _suit_of(b) and c.partner_suit and _suit_of(b) != c.partner_suit
            and _suit_of(b) not in {_suit_of(x) for x in c.my_bids} and _control_phrase(c)):
        return "nt_opener_control"
    # (1x) 1NT (P) 2C/2D/2H/2S: "systems on" after partner's 1NT overcall, as over a
    # 1NT opening -- WBridge5: 2D held 5+ hearts every time and the overcaller bid 2H
    # 76-85%; 2H the same for spades; 2C a 4-card major 64-100%; 2S -> 3C 92%. The
    # rules read 2D over their 1D as "a cuebid ..., a strong, forcing call"
    # ... and after a natural 2NT overcall of a weak two: 3C Stayman, 3D/3H transfers
    # (the rules already read 2NT (P) 3D (P) 3H as completing a transfer)
    nt_ov = {"1": ("1NT", "1N"), "2": ("2NT", "2N")}
    if (len(real) == 2 and real[0]["seat"] not in (c.you, c.partner_seat) and real[0]["bid"][:1] in nt_ov
            and _suit_of(real[0]["bid"]) and real[1]["seat"] == c.partner_seat
            and real[1]["bid"] in nt_ov[real[0]["bid"][:1]]
            and c.prior[-1]["bid"] in ("P", "Pass") and not c.my_bids and not doubled):
        lvl = str(int(real[1]["bid"][0]) + 1)
        conv = {lvl + "C": "nt_overcall_stayman", lvl + "D": "nt_overcall_transfer", lvl + "H": "nt_overcall_transfer",
                **({"2S": "nt_minor_puppet"} if lvl == "2" else {})}.get(b)
        if conv:
            return conv
    if (len(real) == 3 and real[0]["seat"] not in (c.you, c.partner_seat) and real[1]["seat"] == c.you
            and real[1]["bid"] in ("1NT", "1N", "2NT", "2N") and real[2]["seat"] == c.partner_seat and not doubled):
        pb = real[2]["bid"]
        lvl = pb[0]
        if pb[0] == str(int(real[1]["bid"][0]) + 1) and (pb[1:], b[1:]) in (("D", "H"), ("H", "S")) and b[0] == lvl:
            return "nt_overcall_transfer_accept"
        if (pb, b) == ("2S", "3C") and real[1]["bid"][0] == "1":
            return "nt_minor_puppet_accept"
    # 1NT 2D 2H 3NT (or 2H 2S 3NT): exactly five in the major and 9-13 HCP -- WBridge5's
    # opener passed all 6,051 times, even with four-card support (no choice of games)
    if (b in ("3NT", "3N") and c.my_bids in (["2D"], ["2H"]) and c.partner_bids[:1] in (["1NT"], ["1N"])
            and len(c.partner_bids) == 2 and c.partner_bids[1] == {"2D": "2H", "2H": "2S"}[c.my_bids[0]]
            and (c.opener or {}).get("seat") == c.partner_seat and not c.opp_bids):
        return "transfer_3nt"
    # 1NT (P) 2S: a puppet to 3C with a weak hand and a long minor (WBridge5:
    # opener bid 3C 1,567 of 1,714 times; responder held 6+ in a minor)
    if (b == "2S" and len(real) == 1 and real[0]["seat"] == c.partner_seat and real[0]["bid"] in ("1NT", "1N")
            and not any(x["bid"] in ("X", "XX", "Dbl", "RDbl") for x in c.prior)):
        return "nt_minor_puppet"
    if (b == "3C" and len(real) == 2 and real[0]["seat"] == c.you and real[0]["bid"] in ("1NT", "1N")
            and real[1]["seat"] == c.partner_seat and real[1]["bid"] == "2S"
            and not any(x["bid"] in ("X", "XX", "Dbl", "RDbl") for x in c.prior)):
        return "nt_minor_puppet_accept"
    # 1NT (P) 3x: forcing -- a long minor (5/6+ cards, 12 HCP median; opener went on
    # every time) or a 5+ major with 13-17 HCP (WBridge5)
    if (b in ("3C", "3D", "3H", "3S") and len(real) == 1 and real[0]["seat"] == c.partner_seat
            and real[0]["bid"] in ("1NT", "1N") and not any(x["bid"] in ("X", "XX", "Dbl", "RDbl") for x in c.prior)):
        return "nt_three_level_suit"
    # 1M (overcall) 3M: preemptive (WBridge5: 5/8/10 HCP at the 10th/50th/90th
    # percentile, three-card support 38%), unlike 1M (P) 3M, a limit raise (9/10/11)
    if (b in ("3H", "3S") and len(real) == 2 and real[0]["seat"] == c.partner_seat and real[0]["bid"] == "1" + b[1]
            and real[1]["seat"] not in (c.you, c.partner_seat) and _suit_of(real[1]["bid"]) not in (None, b[1])
            and not c.my_bids):
        ov = real[1]["bid"]  # 1H (2S) 3H is the cheapest raise, not a jump
        cheapest = int(ov[0]) + (0 if "CDHS".index(b[1]) > "CDHS".index(_suit_of(ov)) else 1)
        return "preemptive_jump_raise" if int(b[0]) > cheapest else "competitive_three_level_raise"
    # 1H/1S (P) 2NT: 13-15 HCP balanced, forcing to game (WBridge5: 13/14/15 HCP at
    # the 10th/50th/90th percentile; three of opener's major 67%, four 25% -- not Jacoby)
    if (b in ("2NT", "2N") and len(real) == 1 and real[0]["seat"] == c.partner_seat and real[0]["bid"] in ("1H", "1S")
            and not any(x["bid"] in ("X", "XX", "Dbl", "RDbl") for x in c.prior)):
        return "major_2nt_response"
    if (len(real) == 2 and real[0]["seat"] == c.partner_seat and real[0]["bid"] in ("1NT", "1N")
            and real[1]["seat"] not in (c.you, c.partner_seat) and real[1]["bid"][:1] == "2"
            and _suit_of(real[1]["bid"]) and b[:1] in "23" and _suit_of(b) and _suit_of(b) != _suit_of(real[1]["bid"])):
        return "nt_forcing_new_suit" if b[:1] == "3" else "nt_competitive_new_suit"
    # 1x (2z) 3y, a non-jump new suit: forcing (WBridge5: 9/11/15 HCP, 5+ cards
    # 98%, opener passed 7% of 3,911), not "preemptive intent" as the rules say
    if (len(real) == 2 and real[0]["seat"] == c.partner_seat and real[0]["bid"][:1] == "1" and _suit_of(real[0]["bid"])
            and real[1]["seat"] not in (c.you, c.partner_seat) and real[1]["bid"][:1] == "2" and _suit_of(real[1]["bid"])
            and b[:1] == "3" and _suit_of(b) and _suit_of(b) not in (_suit_of(real[0]["bid"]), _suit_of(real[1]["bid"]))
            and "CDHS".index(_suit_of(b)) < "CDHS".index(_suit_of(real[1]["bid"]))
            and not any(x["bid"] in ("X", "XX", "Dbl", "RDbl") for x in c.prior)):
        return "suit_forcing_new_suit"
    ms = _direct_cuebid_suit(c, c.partner_seat)
    if ms in ("H", "S") and b in ("2NT", "2N") and not c.my_bids:
        return "michaels_ask"
    mine = _direct_cuebid_suit(c, c.you)
    if mine in ("H", "S") and c.partner_bids[-1:] in (["2NT"], ["2N"]) and b in ("3C", "3D"):
        return "michaels_minor"
    if (c.partner_bids[:1] == ["2C"] and c.partner_bids[-1:] in (["2NT"], ["2N"]) and c.my_bids == ["2D"]
            and not c.opp_bids and (c.opener or {}).get("seat") == c.partner_seat):
        if b == "3C":
            return "stayman_after_2c"
        if b in ("3D", "3H"):
            return "transfer_after_2c"
    # the agreed suit is any suit partner bid naturally -- 1S 2S 3H (a game try) 3S
    # declines it; partner's LAST suit is hearts (not "inviting game")
    ps = c.bid_suit if c.bid_suit in natural_partner_suits(c.f) else None
    if ps and c.partner_bids and _suit_of(c.partner_bids[-1]) not in (ps, None):
        agreed = ps in {_suit_of(x) for x in c.my_bids} or sum(_suit_of(x) == ps for x in c.partner_bids) >= 2
        lvl = int(c.partner_bids[-1][0])
        game = {"C": "5", "D": "5", "H": "4", "S": "4"}[ps]
        if agreed and lvl >= 4 and b[0] == game:
            return "signoff"
        if agreed and lvl == 3 and ps in "HS":
            return "game_try_accept" if b[0] == "4" else "game_try_decline" if b[0] == "3" else None
    return None


class Ctx:
    """Everything a sentence template may use, derived once per decision."""

    def __init__(self, frame: Dict[str, Any], rng: random.Random):
        self.f = frame
        self.rng = rng
        self.h = hand_from_str(frame["hand_str"])
        t = frame.get("table") or {}
        self.you, self.partner_seat = t.get("you"), t.get("partner")
        self.seat = SEAT_NAME.get(self.you, self.you or "The bidder")
        self.bid = frame["bid"]
        self.cat = bid_category(frame)
        self.prior = frame.get("auction_before") or []
        real = [b for b in self.prior if b["bid"] not in NB]
        self.partner_bids = [b["bid"] for b in real if b["seat"] == self.partner_seat]
        self.my_bids = [b["bid"] for b in real if b["seat"] == self.you]
        self.opp_bids = [b["bid"] for b in real if b["seat"] not in (self.you, self.partner_seat)]
        # the opponents' NATURAL suits, in bidding order: their Stayman 2C is not
        # "the opponents' suit" (East's 6 clubs are not "the opponents' suit")
        natural = opponent_natural_suits(auction_from_frame(frame))
        self.opp_suits = []
        for b in self.opp_bids:
            s = _suit_of(b)
            if s and s in natural and s not in self.opp_suits:
                self.opp_suits.append(s)
        nat = natural_partner_suits(frame)
        self.partner_suit = nat[-1] if nat else None
        self.partner_doubled = any(b["seat"] == self.partner_seat and b["bid"] == "X" for b in self.prior)
        self.bid_suit = _suit_of(self.bid)
        self.gt = _gt(frame)
        self.opener = next((b for b in real), None)
        # the call a double applies to is the last real bid at the table
        last = real[-1] if real else None
        theirs = last is not None and last["seat"] not in (self.you, self.partner_seat)
        self.last_opp_bid = last["bid"] if theirs else None
        self.last_opp_suit = _suit_of(last["bid"]) if theirs else None
        self.implied_majors = _negative_double_majors(self)

    # lazy: the prompt's situation block (situation()) needs neither
    @functools.cached_property
    def conv(self) -> Optional[str]:
        return _conventional_call(self)

    @functools.cached_property
    def reasons(self) -> List[str]:
        return select_reasoned_concepts(self.f)

    @functools.cached_property
    def meaning(self) -> List[str]:
        return bid_meaning_concepts(self.f)

    def pick(self, *options: str) -> str:
        return self.rng.choice(options)


# ---------------------------------------------------------------------------
# 1. headline: seat + action + what the call means here
# ---------------------------------------------------------------------------

def _action(c: Ctx) -> str:
    if c.bid in ("P", "Pass"):
        return f"{c.seat} passes"
    if c.bid in ("X", "Dbl"):
        return f"{c.seat} doubles"
    if c.bid in ("XX", "RDbl"):
        return f"{c.seat} redoubles"
    if c.cat.startswith("opening_"):
        return f"{c.seat} opens {c.bid}"
    return f"{c.seat} bids {c.bid}"


def _gloss_as_apposition(concept: str) -> Optional[str]:
    """'is a takeout double of ...' -> 'a takeout double of ...' for use
    after the action ("West doubles, a takeout double of ...")."""
    g = gloss_for_concept(concept)
    for pre, rep in (("is a ", "a "), ("is an ", "an "), ("is the ", "the "), ("is ", ""),
                     ("shows ", "showing "), ("completes ", "completing "), ("raises ", "raising "),
                     ("makes ", "making "), ("asks ", "asking "), ("answers ", "answering "),
                     ("bids ", "bidding "), ("uses ", "using "), ("transfers ", "transferring "),
                     ("redoubles ", "redoubling "), ("doubles ", "doubling "), ("opens ", "opening "),
                     ("settles ", "settling ")):
        if g.startswith(pre):
            return rep + g[len(pre):]
    return None


# situation-specific meaning phrases where the concept gloss would be clumsy
def _doubled_cuebid_suit(c: Ctx) -> Optional[str]:
    """The suit of the opponents' cuebid of OUR suit that this double is of
    (1H (1S) P (3H) X: 3H raises their spades; nobody will play it), else
    None. The rules call such a double a penalty double, which would read
    "a penalty double, expecting to defeat the doubled contract ... the
    opponents are vulnerable, so every undertrick costs them more"."""
    if c.bid not in ("X", "Dbl") or not c.last_opp_suit or int(c.last_opp_bid[0]) > 4:
        return None
    if c.opp_bids[-2:-1] in (["4NT"], ["4N"]):
        return None  # a Blackwood reply, not a cuebid
    s = c.last_opp_suit
    ours = {_suit_of(b) for b in c.my_bids + c.partner_bids}
    theirs_before = [b for b in c.opp_bids[:-1] if _suit_of(b) == s]
    return s if s in ours and not theirs_before else None


def _is_artificial(c: Ctx) -> bool:
    """The call's suit is not a natural suit of the bidder's -- except a
    "control bid" the rules name without a control in the suit (4S with J T 9
    returning to partner's spades)."""
    if c.cat == "slam_blackwood_answer":
        return True
    return any(c.gt.get(x) for x in _ARTIFICIAL
               if not (x == "bid_is_control_bid" and not (c.bid_suit and _control_phrase(c))))


def _takeout_cuebid(c: Ctx) -> Optional[str]:
    """The advancer's cuebid of the suit partner doubled for takeout, which
    WBridge5 does not use as "a strong, forcing call" (the teacher said so of
    7 HCP): after RHO's 1NT or redouble it asks partner to choose between the
    unbid suits (5/7/10 HCP, two 4+ card unbid suits 95% of 336); after a pass
    or a bid it shows values (median 10-11 HCP) and asks for the strain
    ("choose" / "values", else None)."""
    if not (c.bid_suit and c.partner_doubled and not c.partner_bids and not c.my_bids and c.opener):
        return None
    if (c.opener["seat"] in (c.you, c.partner_seat) or c.bid_suit != _suit_of(c.opener["bid"])
            or c.opener["bid"][:1] not in "123"):  # also over a doubled weak two or preempt (7/9/13 HCP)
        return None
    after = [b for b in c.prior[[b is c.opener for b in c.prior].index(True) + 1:]]
    # partner's takeout double, directly or in the balancing seat ((1x) P (P) X)
    k = next((i for i, b in enumerate(after) if b["bid"] not in ("P", "Pass")), None)
    if k is None or after[k]["bid"] not in ("X", "Dbl") or after[k]["seat"] != c.partner_seat:
        return None
    rho = after[k + 1]["bid"] if len(after) > k + 1 else None
    return "choose" if rho in ("1NT", "1N", "XX", "RDbl") else "values"


def _own_suit(c: Ctx, s: str) -> bool:
    """A suit the bidder has bid NATURALLY (their Stayman 2C or a cuebid is not)."""
    a = auction_from_frame(c.f)
    return s in natural_suits_of(a, (a.actor_seat(),))


def _where(c: Ctx, s: str, theirs: str) -> str:
    """ "in partner's suit" when the opponents' last (artificial) bid was in a
    suit our side bid (not "3 diamonds (Q 8 7) in the opponents' suit" of a
    Blackwood reply in partner's diamonds)."""
    if s in natural_partner_suits(c.f):  # (partner's cuebid 3D of THEIR diamonds does not make them "partner's")
        return "in partner's suit"
    if _own_suit(c, s):
        return f"in {c.seat}'s own suit"
    return theirs


def _double_reading(c: Ctx) -> Optional[str]:
    """"penalty" / "competitive" when WBridge5's partner leaves a double in
    this auction shape in >= 60% / <= 30% of 30+ cases (nlg.double_meaning),
    else None (no data, or an optional double)."""
    if c.bid not in ("X", "Dbl"):
        return None
    r = leave_in_rate(c.f)
    if not r:
        # no data for this exact shape: a double of a 4-level+ suit contract is
        # left in 76-99% wherever WBridge5 has data (e.g. a 5D sacrifice doubled with 6 HCP)
        lb = c.last_opp_bid or ""
        if lb[:1] in "4567" and _suit_of(lb) and not _doubled_cuebid_suit(c) and c.opp_bids[-2:-1] not in (["4NT"], ["4N"]):
            return "penalty"
        return None
    return "penalty" if r[0] >= 0.6 else "competitive" if r[0] <= 0.3 else None


def _penaltyish(c: Ctx) -> bool:
    return bool(c.gt.get("bid_is_penalty_double") or c.gt.get("bid_double_with_length_in_their_suit"))


def _doubled_nt_convention(c: Ctx):
    """(bid, suit, name) when this double is of the opponents' Stayman or
    transfer right after their 1NT opening (1NT (P) 2D X: WBridge5's partner
    leaves it in 136 of 136 times -- it asks for the lead, it is not a
    takeout double of 2D)."""
    if c.bid not in ("X", "Dbl") or c.opp_bids[:1] not in (["1NT"], ["1N"]) or len(c.opp_bids) != 2:
        return None
    if c.my_bids or c.partner_bids or (c.opener or {}).get("bid") not in ("1NT", "1N"):
        return None
    b = c.last_opp_bid
    name = {"2C": "Stayman", "2D": "transfer", "2H": "transfer"}.get(b or "")
    return (b, b[1], name) if name else None


def _doubled_artificial_reply(c: Ctx) -> Optional[str]:
    """The suit of the opponents' reply to their own 4NT / 5NT ask that this
    double is of (6C = no king, doubled: "a penalty double of 6C")."""
    if c.bid not in ("X", "Dbl") or not c.last_opp_suit or len(c.opp_bids) < 2:
        return None
    return c.last_opp_suit if c.opp_bids[-2] in ("4NT", "4N", "5NT", "5N") else None


def _not_for_penalties(c: Ctx) -> bool:
    """Doubles the rules call penalty doubles that are not: of the
    opponents' cuebid of our suit or of their Stayman/transfer, opener's
    support and competitive doubles, and doubles WBridge5's partner takes
    out (nlg.double_meaning)."""
    return bool(_doubled_cuebid_suit(c) or support_double_suit(c.f) or competitive_double_suit(c.f)
                or responsive_double_suits(c.f) or _doubled_artificial_reply(c)
                or _doubled_nt_convention(c) or _double_reading(c) == "competitive")


def _ev_data_penalty(c: Ctx) -> str:
    """Why a double WBridge5's partner leaves in is for penalties."""
    lb = c.last_opp_bid or ""
    if lb in ("1NT", "1N") and (c.opener or {}).get("bid") in ("1NT", "1N") and c.h.hcp >= 15:
        return f"with {c.h.hcp} HCP {c.seat} is at least as strong as the 1NT opener"
    if lb in ("1NT", "1N") and c.partner_bids and c.h.hcp >= 9:
        return f"with {c.h.hcp} HCP opposite partner's opening, the partnership has most of the strength"
    s = c.last_opp_suit
    if s and c.h.lengths[s] >= 3:
        return f"{c.seat} has {c.h.hcp} HCP and {holding(c.h, s)} in their suit"
    return f"{c.seat} has {c.h.hcp} HCP"


def _meaning_override(c: Ctx) -> Optional[str]:
    cat, bid, s = c.cat, c.bid, c.bid_suit
    pb = c.partner_bids[-1] if c.partner_bids else None
    conv = c.conv
    if bid in ("P", "Pass") and c.gt.get("bid_pass_accepts_partners_contract"):
        # the concept's own meaning ("passes, leaving partner in the contract partner chose"), so the
        # headline is not a bare "South passes." when no auction-only concept has a positive contribution
        real = [b for b in c.prior if b["bid"] not in NB]
        if real and real[-1]["seat"] == c.partner_seat:
            return f"leaving partner in {real[-1]['bid']}"
    if conv == "michaels_ask":
        return "asking partner to name the minor of the Michaels two-suiter"
    if conv == "michaels_minor":
        return f"showing {SUIT_NAME[s]} as the minor of the Michaels cuebid"
    if conv == "stayman_after_2c":
        return "Stayman, asking partner for a four-card major"
    if conv == "transfer_after_2c":
        return f"a transfer to {SUIT_NAME['H' if s == 'D' else 'S']}"
    if conv == "signoff":
        return "signing off in game, declining partner's slam try"
    if conv == "transfer_3nt":
        m = "H" if c.my_bids == ["2D"] else "S"
        if c.h.lengths[m] != 5:  # (the system meaning would be false for this hand)
            return "bidding game in notrump after the transfer"
        return f"bidding game in notrump after the transfer, showing exactly five {SUIT_NAME[m]} and game values"
    if conv == "nt_opener_control":
        return f"a control cuebid in {SUIT_NAME[s]}, agreeing partner's {SUIT_NAME[c.partner_suit]}"
    if conv == "nt_overcall_stayman":
        return f"Stayman over partner's {c.partner_bids[-1]} overcall, asking for a four-card major"
    if conv == "nt_overcall_transfer":
        return f"a transfer to {SUIT_NAME['H' if s == 'D' else 'S']} over partner's {c.partner_bids[-1]} overcall"
    if conv == "nt_overcall_transfer_accept":
        return "completing partner's transfer"
    if conv == "nt_minor_puppet":
        return "an artificial relay asking partner to bid 3C, with a weak hand and a long minor"
    if conv == "nt_minor_puppet_accept":
        return "completing partner's 2S relay, as asked"
    if conv == "nt_three_level_suit":
        return (f"a natural, forcing bid showing a long {SUIT_NAME[s][:-1]} suit" if s in "CD"
                else f"showing five or more {SUIT_NAME[s]}, forcing to game")
    if conv == "preemptive_jump_raise":
        return (f"a preemptive jump raise over the overcall, showing support for partner's {SUIT_NAME[s]} "
                f"and about 5 to 10 HCP")
    if conv == "competitive_three_level_raise":
        return (f"a competitive raise to the three level, showing support for partner's {SUIT_NAME[s]} "
                f"and about 5 to 10 HCP")
    if conv == "major_2nt_response":
        return "showing a balanced 13 to 15 HCP, forcing to game"
    if conv == "nt_forcing_new_suit":
        return (f"a forcing bid over the interference, showing five or more {SUIT_NAME[s]} and enough to look "
                f"for game opposite partner's 1NT")
    if conv == "suit_forcing_new_suit":
        return f"a natural, forcing new suit over the overcall, showing five or more {SUIT_NAME[s]} and values"
    if conv == "nt_competitive_new_suit":
        return f"a natural, non-forcing bid to compete, showing five or more {SUIT_NAME[s]} and a weak hand"
    cued = _doubled_cuebid_suit(c)
    if cued and _own_suit(c, cued):  # WBridge5: 5+ cards 95%, an A/K/Q 97% of 10,981
        return f"a lead-directing double of the opponents' cuebid of {SUIT_NAME[cued]}, showing good {SUIT_NAME[cued]}"
    if cued:  # of partner's suit: 3+ cards 98% of 7,102, but no A/K/Q 43% -- support, not a lead
        return f"a double of the opponents' cuebid of partner's {SUIT_NAME[cued]}, showing support for them"
    sd = support_double_suit(c.f)
    if sd:
        return f"a support double, showing exactly three cards in partner's {SUIT_NAME[sd]}"
    sr = support_redouble_suit(c.f)
    if sr:
        return f"a support redouble, showing exactly three cards in partner's {SUIT_NAME[sr]}"
    if responsive_double_suits(c.f):
        return "a responsive double, showing the unbid suits after the opponents' raise"
    if competitive_double_suit(c.f):
        return "a competitive double, showing values and asking partner to bid again"
    art = _doubled_nt_convention(c)
    if art:
        return f"a lead-directing double of the opponents' {art[2]} {art[0]}, showing good {SUIT_NAME[art[1]]}"
    ar = _doubled_artificial_reply(c)
    if ar:
        return f"a lead-directing double of the opponents' artificial {c.last_opp_bid} reply, showing good {SUIT_NAME[ar]}"
    dr = _double_reading(c)
    if dr == "penalty" and c.last_opp_bid:  # e.g. of a 1NT overcall: left in 81% (not "a negative double")
        return f"a penalty double of {c.last_opp_bid}"
    if dr == "competitive" and _penaltyish(c):  # e.g. 1H (2C) 2H (3C) X: taken out 98% (not "a penalty double")
        return "a competitive double, asking partner to bid again"
    if conv == "game_try_accept":
        return "accepting partner's game try"
    if conv == "game_try_decline":
        return "declining partner's game try"
    if cat == "opening_1_major":
        return f"showing five or more {SUIT_NAME[s]}"
    if cat == "opening_1_minor":
        if c.h.lengths["D"] == c.h.lengths["C"]:
            # "the longer minor" with 4-4 taught Qwen "diamonds are longer than clubs (4 against 4)"
            return f"the system's choice with {_WORDS[c.h.lengths['D']]} cards in each minor"
        return "the longer minor" if c.gt.get("bid_shows_longer_minor") else f"a natural {SUIT_NAME[s][:-1]} opening"
    if cat == "opening_1nt":
        return "showing a balanced 15 to 17 HCP"
    if cat == "opening_2nt":
        return "showing a balanced 20 to 21 HCP"
    if cat == "opening_2c_strong":
        return "the strong, artificial opening"
    if cat == "opening_weak_two":
        return "a light third-seat weak two" if c.gt.get("bid_is_light_third_seat_weak_two") else "a weak two"
    if cat == "opening_preempt":
        if c.gt.get("bid_is_unusual_4nt_opening"):
            return "the Unusual 4NT, showing at least six cards in both minors"
        return f"a preempt at the {bid[0]} level"
    if cat == "response_simple_raise" and s:
        return f"a simple raise of partner's {SUIT_NAME[s]}"
    if cat in ("response_jump_raise",) and s:
        return f"a jump raise of partner's {SUIT_NAME[s]}"
    if cat == "response_game_raise" and s:
        return f"raising partner's {SUIT_NAME[s]} straight to game"
    if cat == "response_1nt_stayman" or c.gt.get("bid_is_stayman"):  # 2C over 1NT, 3C over 2NT
        return "Stayman, asking partner for a four-card major"
    if (cat == "response_1nt_transfer" or c.gt.get("bid_is_jacoby_transfer")) and s in ("D", "H"):
        return f"a transfer to {SUIT_NAME['H' if s == 'D' else 'S']}"
    if cat == "response_new_suit_1_level" and s:
        return f"a new suit at the one level, showing four or more {SUIT_NAME[s]}"
    if cat == "response_new_suit_2_level" and pb == "2C" and bid == "2D":
        return "the waiting response to partner's strong 2C"
    if cat == "response_new_suit_2_level" and pb == "2C":
        return "a positive response to partner's strong 2C"
    if cat == "response_new_suit_2_level" and s and bid[0] == "2":
        return "a new suit at the two level, showing ten or more HCP"
    if cat == "response_1nt_other" and bid in ("2NT", "2N") and not c.opp_bids:
        return "inviting partner to bid 3NT with a maximum"
    if cat == "response_notrump" and bid in ("1NT",):
        return "showing a limited hand with no fit to bid"
    if cat == "slam_blackwood_ask":
        return "Blackwood, asking partner how many aces"
    if cat == "slam_blackwood_answer" and bid[:1] == "6":
        return {"6C": "showing no king", "6D": "showing one king", "6H": "showing two kings",
                "6S": "showing three kings", "6NT": "showing all four kings"}.get(bid, "answering the king ask") + " in reply to partner's 5NT king ask"
    if cat == "slam_blackwood_answer":
        return {"5C": "showing 0 or 3 aces", "5D": "showing 1 or 4 aces", "5H": "showing 2 aces",
                "5S": "showing 3 aces"}.get(bid, "answering Blackwood")
    if cat == "takeout_double":
        return f"a takeout double of {c.opp_bids[-1]}" if c.opp_bids else "a takeout double"
    if cat == "responder_negative_double":
        if c.last_opp_suit and c.h.lengths[c.last_opp_suit] >= 4:
            return "a double showing values; with length in their suit, partner may pass it for penalties"
        return "a negative double"
    if cat == "redouble":
        return "showing useful values after the opponents' double"
    if cat == "overcall_suit" and s:
        return f"overcalling in {SUIT_NAME[s]}"
    doubled_first = any(b["seat"] == c.you and b["bid"] in ("X", "Dbl") for b in c.prior) and not c.my_bids
    if doubled_first and s and s not in c.opp_suits and s != c.partner_suit and cat.startswith(("overcall", "competitive")):
        # X then a new suit is not an overcall: it shows a hand too strong to overcall
        return (f"bidding {SUIT_NAME[s]} after doubling first, showing a strong hand with long {SUIT_NAME[s]}"
                if c.h.hcp >= 16 and c.h.lengths[s] >= 5 else f"bidding {SUIT_NAME[s]} after doubling first")
    if cat == "overcall_jump" and s:
        return "a weak jump overcall" if c.gt.get("bid_is_weak_jump_overcall") else "a jump overcall"
    if cat == "overcall_notrump":
        if c.gt.get("bid_is_unusual_2nt"):
            return "the Unusual 2NT for the two lowest unbid suits"
        if bid in ("4NT", "4N"):  # a jump to 4NT over the opponents' preempt is for the minors
            return "the Unusual 4NT, asking partner to choose a minor"
        return "a natural notrump overcall"
    if cat == "overcall_cuebid" and c.gt.get("bid_is_michaels_cuebid"):
        return "a Michaels cuebid showing two long suits"
    if s and c.partner_suit and (c.gt.get("bid_is_singleton_splinter") or c.gt.get("bid_is_void_splinter")):
        short = "a void" if c.gt.get("bid_is_void_splinter") else "a singleton"
        return (f"a splinter: a double jump showing support for partner's {SUIT_NAME[c.partner_suit]} "
                f"and {short} in {SUIT_NAME[s]}")
    tc = _takeout_cuebid(c)
    if tc == "choose":
        return f"a cuebid of the opponents' {SUIT_NAME[s]}, asking partner to choose between the unbid suits"
    if tc == "values":
        return f"a cuebid of the opponents' {SUIT_NAME[s]}, showing values and asking partner to choose the strain"
    if s and c.gt.get("bid_is_cuebid"):
        ps = c.partner_suit
        if ps and _own_suit(c, ps) and bid[:1] == "3":
            # the suit is agreed and was ours first: 1D (2C) 2H (3C) P (3D) is the overcaller's game try,
            # not "a limit raise or better in partner's clubs" -- that is the cuebid of the partner who has
            # not bid the suit. WBridge5 itself passes there (12,878
            # of ~13,000 times after 1x 2y .. 3y), so no measured strength is claimed for it.
            return f"a cuebid of the opponents' {SUIT_NAME[s]}, a game try in the agreed {SUIT_NAME[ps]}"
        if ps and c.h.lengths[ps] >= 3:
            return f"a cuebid of the opponents' {SUIT_NAME[s]}, showing a limit raise or better in partner's {SUIT_NAME[ps]}"
        doubled_first = any(b["seat"] == c.you and b["bid"] in ("X", "Dbl") for b in c.prior)
        if ps and not doubled_first and (c.opener or {}).get("seat") not in (c.you, c.partner_seat):
            # the advancer's cuebid without support: median 11 HCP in WBridge5's games, not "strong"
            return f"a cuebid of the opponents' {SUIT_NAME[s]}, a forcing call showing values"
        return f"a cuebid of the opponents' {SUIT_NAME[s]}, a strong, forcing call"
    if c.gt.get("bid_is_check_back") and s:
        # natural in WBridge5, not new minor forcing: after 1x 1M 1NT, responder's
        # 2C/2D held 4+ cards there 2,747 of 2,747 times, with 5+ in the major
        # 97%, and opener passed 11-28%
        m = next((_suit_of(b) for b in c.my_bids if _suit_of(b) in ("H", "S")), None)
        return (f"showing a second suit, {SUIT_NAME[s]}, with five or more {SUIT_NAME[m]}" if m
                else f"showing a second suit, {SUIT_NAME[s]}")
    if cat == "opener_double" and c.gt.get("bid_is_takeout_double"):
        return "a reopening takeout double, asking partner to choose a suit"
    if s and s in c.implied_majors and c.h.lengths[s] >= 4:
        return f"supporting the {SUIT_NAME[s]} partner's negative double promised"
    if cat.startswith("advancer_") and cat != "advancer_pass" and c.partner_doubled and not c.partner_bids:
        return "answering partner's takeout double"
    if cat.startswith("advancer_") and cat != "advancer_pass" and pb:
        return f"advancing partner's {pb}"
    if s and s in c.opp_suits and cat in ("overcall_cuebid", "responder_bid_competitive", "advancer_cuebid",
                                          "competitive_later_bid"):
        return f"a cuebid of the opponents' {SUIT_NAME[s]}"
    if cat == "response_jump_shift":
        if bid in ("4H", "4S", "5C", "5D"):
            return f"a jump to game in {SUIT_NAME[s]}"
        # a strong jump shift is ~16+ HCP
        if c.prior and c.prior[-1]["bid"] in ("X", "Dbl"):  # 1x (X) jump: 4/7/9 HCP, 6+ cards, passed 59%
            return "a weak jump shift over the double, showing a long suit and a weak hand"
        return "a strong jump shift" if c.h.hcp >= 16 and not c.opp_bids else "a jump response in a long suit"
    if c.gt.get("bid_is_control_bid") and s and _control_phrase(c):  # (4S with J T 9 is not "a control cuebid")
        return f"a control cuebid in {SUIT_NAME[s]} on the way to slam" if s else None
    if bid in ("5NT", "5N") and c.my_bids[-1:] in (["4NT"], ["4N"]):
        return "asking partner for kings"
    if cat == "opener_rebids_own_suit" and s:
        return f"rebidding the {SUIT_NAME[s]}"
    if cat == "opener_raises_responder" and s:
        if _own_suit(c, s):  # 1D (X) 2D (2S) 3D: opener's own diamonds, raised by partner
            return f"competing further in the agreed {SUIT_NAME[s]}"
        return f"raising partner's {SUIT_NAME[s]}"
    if cat == "opener_rebid_notrump":
        if bid in ("3NT", "3N"):
            return "bidding game in notrump"
        if c.h.balanced or is_semi_balanced(c.h):
            return "rebidding notrump to describe a balanced hand"
        return "rebidding notrump"  # (not "to describe a balanced hand": the hand may be 4-4-1-4)
    own = bool(s) and _own_suit(c, s)  # partner raised MY suit: "the agreed", not "partner's"
    if bid in ("3NT", "4H", "4S", "5C", "5D") and cat not in ("slam_blackwood_answer",):
        return "bidding game" + ((f" in the agreed {SUIT_NAME[s]}" if own else f" in partner's {SUIT_NAME[s]}")
                                 if s and s in natural_partner_suits(c.f) else "")
    if cat in ("responder_later_support", "competitive_later_support", "advancer_raise",
               "responder_raise_competitive") and s:
        if own:
            real_ = [b for b in c.prior if b["bid"] not in NB]
            contested = bool(real_) and real_[-1]["seat"] not in (c.you, c.partner_seat)
            if contested:  # the opponents bid over the raise: 3S competes, it does not invite
                return f"competing further in the agreed {SUIT_NAME[s]}"
            return (f"inviting game in the agreed {SUIT_NAME[s]}" if bid[0] == "3" and s in "HS"
                    else f"raising the agreed {SUIT_NAME[s]}")
        return f"supporting partner's {SUIT_NAME[s]}"
    if cat == "opener_new_suit" and c.gt.get("bid_completes_jacoby_transfer"):
        return "completing partner's transfer"
    if cat == "opener_new_suit" and c.gt.get("bid_is_reverse"):
        return "a reverse into a higher suit, showing extra strength"
    if cat == "opener_new_suit" and c.gt.get("bid_shows_stayman_major_structure"):
        return "answering partner's Stayman"
    if cat == "opener_new_suit" and c.my_bids == ["2C", "2NT"] and c.partner_bids == ["2D", "3C"] and bid in ("3D", "3H", "3S"):
        return "answering partner's Stayman"  # 2C 2D 2NT 3C: Stayman, not a new suit
    if cat == "opener_new_suit" and s and c.my_bids == ["2C"]:  # 2C was artificial: this is the real suit
        return f"showing the strong hand's suit, {SUIT_NAME[s]}"
    if cat == "opener_new_suit" and s:
        return f"showing a second suit, {SUIT_NAME[s]}"
    if cat in ("responder_later_notrump", "competitive_later_bid", "responder_later_other") and bid.endswith("NT") and bid[0] in "123":
        return "suggesting notrump"
    if cat == "slam_bid":
        return "bidding slam"
    return None


_VAGUE_MEANING = {"bid_shows_constructive_intent", "bid_shows_competitive_intent", "bid_is_rebid", "bid_is_game_bid",
                  "bid_limits_hand", "bid_limits_to_game_values", "bid_is_jump_bid", "bid_is_1_level_response",
                  "bid_is_2_level_response", "bid_is_2_level_opening", "bid_is_3_level_opening",
                  "bid_is_4plus_level_opening", "bid_is_1_level_opening", "bid_is_conventional",
                  "bid_shows_extra_values", "bid_shows_minimum_values"}


def meaning_phrase(c: Ctx) -> Optional[str]:
    """What the call means here ("Stayman, asking partner for a four-card
    major"), or None when nothing specific can be said."""
    phrase = _meaning_override(c)
    if phrase is None:
        for m in c.meaning:
            if m in _VAGUE_MEANING:
                continue
            phrase = _gloss_as_apposition(m)
            if phrase:
                break
    return phrase


def headline(c: Ctx) -> str:
    phrase = meaning_phrase(c)
    return f"{_action(c)}, {phrase}." if phrase else f"{_action(c)}."


# ---------------------------------------------------------------------------
# 2. evidence: one clause per hand-dependent reason, citing the hand
# ---------------------------------------------------------------------------

def _psuit(c: Ctx) -> Optional[str]:
    return c.partner_suit


def _good_suit(cards) -> bool:
    """Two of the top three honours, or three of the top five (K 7 6 4 3 is not "a good
    five-card suit")."""
    return sum(r in cards for r in "AKQ") >= 2 or sum(r in cards for r in "AKQJT") >= 3


def _ev_rule_of_20(c: Ctx) -> Optional[str]:
    """The Rule of 20 decides borderline hands only (not "the Rule of 20 is met
    (19 + 5 + 3 = 27)" on a 19-count); a balanced 18-19 opens a
    suit because it is too strong for 1NT."""
    if c.h.hcp <= 12:
        return f"the Rule of 20 is met ({rule_of_20(c.h)})"
    if c.h.hcp >= 18 and (c.h.balanced or is_semi_balanced(c.h)):
        return f"with {c.h.hcp} HCP the hand is too strong to open 1NT"
    return None


def _ev_support(c: Ctx) -> Optional[str]:
    s = _psuit(c) or c.bid_suit
    if not s:
        return None
    sp = support_points(c.h, s)
    if _own_suit(c, s):  # partner raised MY suit ("with 4 spades in partner's suit" of 1S 2S 3S)
        return f"{c.seat} holds {holding(c.h, s)} in the agreed suit, worth {sp} points counting shortness"
    return c.pick(f"{c.seat} holds {holding(c.h, s)} in partner's suit and {sp} support points",
                  f"with {holding(c.h, s)} in partner's suit, {c.seat} has real support ({sp} support points)")


def _ev_length(c: Ctx) -> Optional[str]:
    s = c.bid_suit
    if not s:
        return None
    return c.pick(f"{c.seat} holds {holding(c.h, s)}", f"the hand has {holding(c.h, s)}")


def _ev_pass_no_suit(c: Ctx) -> str:
    """Why a pass with no partner's suit to support, when the hand has a 5+ card suit (not "no good
    suit to show"): too weak for the level it needs, or it is the opponents' suit."""
    long_s = max("SHDC", key=lambda x: (c.h.lengths[x], -"SHDC".index(x)))
    if c.h.lengths[long_s] < 5:
        return f"with {c.h.hcp} HCP and no good suit to show, there is nothing safe to bid"
    if long_s in c.opp_suits:
        return f"the long suit, {holding(c.h, long_s)}, is the opponents' suit, so there is nothing to bid"
    real = [b["bid"] for b in c.prior if b["bid"] not in NB]
    last = real[-1] if real else None
    if not last:
        lvl = 1
    elif last[1:] in ("NT", "N"):
        lvl = int(last[0]) + 1
    else:
        lvl = int(last[0]) + (0 if "CDHS".index(long_s) > "CDHS".index(last[-1]) else 1)
    if lvl > 7:
        return f"with {c.h.hcp} HCP there is nothing safe to bid"
    return f"with {c.h.hcp} HCP, {holding(c.h, long_s)} are not enough to bid at the {_WORDS[lvl]} level"


def _nt_opening_range(c: Ctx):
    """(lo, hi) when the bidder opened 1NT/2NT: "minimum" is relative to it."""
    first = c.my_bids[0] if c.my_bids else None
    if not first or (c.opener or {}).get("seat") != c.you:
        return None
    return {"1NT": (15, 17), "1N": (15, 17), "2NT": (20, 21), "2N": (20, 21)}.get(first)


def _ev_strength(label: str) -> Callable[[Ctx], Optional[str]]:
    def fn(c: Ctx) -> Optional[str]:
        rng_ = _nt_opening_range(c)
        if rng_:  # a 1NT opener's 15 is its minimum
            lo, hi = rng_
            if "more than a minimum" in label or "extra" in label:
                return None if c.h.hcp <= lo else f"the {c.h.hcp} HCP are above the minimum of the {lo}-{hi} range"
            if "minimum" in label:
                return None if c.h.hcp > lo else f"the {c.h.hcp} HCP are the minimum of the {lo}-{hi} range"
        opener = (c.opener or {}).get("seat") == c.you
        if label.startswith("a minimum") and not opener and c.h.hcp >= 11:
            return None  # an advancer's or responder's 12-14 is not "a minimum" (only opener's is)
        if label == "invitational at most" and c.h.hcp <= 5:
            return None  # "the 1 HCP are invitational at most"
        return f"the {c.h.hcp} HCP are {label}"
    return fn


def _ev_fit(c: Ctx) -> Optional[str]:
    s = _psuit(c) or c.bid_suit
    if not s:
        return None
    if _own_suit(c, s):
        return f"with partner's raise, {holding(c.h, s)} give an eight-card fit or better"
    return f"{holding(c.h, s)} opposite partner's {SUIT_NAME[s]} give an eight-card fit or better"


def _ev_stopper(c: Ctx) -> Optional[str]:
    if not c.opp_suits or c.cat.startswith("slam_") or c.conv == "michaels_ask":
        return None  # stoppers are irrelevant to Blackwood / slam decisions
    parts = [f"{' '.join(c.h.cards[s]) or '-'} of {SUIT_NAME[s]}" for s in c.opp_suits]
    return f"the {' and the '.join(parts)} stop{'s' if len(parts) == 1 else ''} the opponents' suit{'s' if len(parts) > 1 else ''}"


def _ev_takeout(c: Ctx) -> Optional[str]:
    s = c.last_opp_suit or (c.opp_suits[-1] if c.opp_suits else None)
    if not s:
        return None
    mine = {_suit_of(b) for b in c.my_bids}
    unbid = [x for x in "SHDC" if x not in c.opp_suits and x not in mine and c.h.lengths[x] >= 3]
    sup = ", ".join(holding(c.h, x) for x in unbid[:3])
    n, cards = c.h.lengths[s], " ".join(c.h.cards[s])
    lead = (f"{c.seat} has a void in the opponents' {SUIT_NAME[s]}" if n == 0
            else f"{c.seat} has only a singleton {SUIT_NAME[s][:-1]} ({cards}) in the opponents' suit" if n == 1
            else f"{c.seat} is short in the opponents' {SUIT_NAME[s]} ({cards})" if n == 2
            else f"{c.seat} has only {holding(c.h, s)} in the opponents' suit")
    return (lead
            + (f" and has support for the other suits: {sup}" if sup else ""))


def _ev_cuebid(c: Ctx) -> Optional[str]:
    """A cuebid is about partner's suit, never about length in the cuebid
    suit (not "West holds 3 clubs" for 1C 1S P 4C)."""
    ps = c.partner_suit
    if ps and c.h.lengths[ps] >= 3:
        return _ev_support(c)
    return None


def _ev_negative_double(c: Ctx) -> Optional[str]:
    unbid = [x for x in "SHDC" if x not in c.opp_suits and x != _psuit(c)]
    majors = [x for x in "HS" if x in unbid]
    show = [x for x in (majors or unbid) if c.h.lengths[x] >= 3]
    if not show:
        return None
    what = "unbid major" if majors else "unbid suit"
    return (f"{c.seat} has {c.h.hcp} HCP and " + ", ".join(holding(c.h, x) for x in show)
            + f" in the {what}" + ("s" if len(show) > 1 else ""))


def _ev_pass_too_weak(c: Ctx) -> str:
    """Pass as responder: the reason depends on what partner opened."""
    pb = c.partner_bids[-1] if c.partner_bids else None
    ps = _psuit(c)
    fit = f" and {holding(c.h, ps)} in partner's suit" if ps else ""
    interfered = any(b["seat"] not in (c.you, c.partner_seat) and b["bid"] not in ("P", "Pass") for b in c.prior)
    if _partner_preempt(c):
        dbl = ", even after the double" if any(b["bid"] == "X" for b in c.prior) else ""
        if _pass_game_context(c):  # the numbers follow (with_context): "the preempt showed 4 to 10 HCP, ..."
            return f"partner's {pb} is a preempt and {c.seat} has {c.h.hcp} HCP{fit}"
        target = "slam" if _is_game(pb) else "game"
        return f"partner's {pb} is a preempt, and with {c.h.hcp} HCP{fit} there is no {target} to look for{dbl}"
    if interfered:
        return f"with {c.h.hcp} HCP{fit}, there is no reason to bid over the interference"
    nt = {"1NT": (15, 17), "1N": (15, 17), "2NT": (20, 21), "2N": (20, 21)}.get(pb or "")
    if nt:
        lo, hi = nt
        if c.h.hcp + hi < 25:
            return f"with only {c.h.hcp} HCP there is no game opposite partner's {lo} to {hi}"
        # 8 HCP opposite 15-17 can still make 25: "no game" would be false
        return (f"with only {c.h.hcp} HCP the partnership has at most {c.h.hcp + hi} opposite partner's "
                f"{lo} to {hi}, so game is unlikely")
    return f"with only {c.h.hcp} HCP there is no response to make"


def _ev_aces(c: Ctx) -> Optional[str]:
    aces = [SUIT_NAME[x][:-1] for x in "SHDC" if "A" in c.h.cards[x]]
    if not aces:
        return f"{c.seat} holds no ace"
    return f"{c.seat} holds {_WORDS[len(aces)]} ace{'s' if len(aces) > 1 else ''} ({', '.join(aces)})"


def _ev_kings(c: Ctx) -> str:
    kings = [SUIT_NAME[x][:-1] for x in "SHDC" if "K" in c.h.cards[x]]
    return (f"{c.seat} holds {_WORDS[len(kings)]} king{'s' if len(kings) != 1 else ''}"
            + (f" ({', '.join(kings)})" if kings else ""))


def _ev_blackwood_math(c: Ctx) -> Optional[str]:
    bw = blackwood_context(c.f)
    if not bw:
        return None
    # "...so the partnership holds N of the 4 aces." -> reuse the resolved numbers
    return bw.rstrip(".").replace("your ", f"{c.seat}'s ").replace("Partner's", "partner's")


def _minor_choice(c: Ctx) -> Optional[str]:
    d, cl = c.h.lengths["D"], c.h.lengths["C"]
    if [c.h.lengths[x] for x in "SHDC"] == [4, 4, 3, 2]:
        return "with 4-4 in the majors and 3-2 in the minors, the system opens 1D"
    if d > cl:
        return f"diamonds are longer than clubs ({d} against {cl})"
    if cl > d:
        return f"clubs are longer than diamonds ({cl} against {d})"
    # the headline already says "the system's choice with four cards in each minor"
    return "the system opens 1D with four-four or longer in the minors and 1C with three-three"


def _control_phrase(c: Ctx) -> Optional[str]:
    """The control the hand holds in the bid suit, or None when it holds none
    (not "a singleton heart (Q)" for Q J 9 8 4)."""
    s, cards = c.bid_suit, c.h.cards.get(c.bid_suit, [])
    name = SUIT_NAME[s][:-1]
    if "A" in cards:
        return f"the ace of {SUIT_NAME[s]} gives first-round control of the suit"
    if not cards:
        return f"a void in {SUIT_NAME[s]} gives first-round control of the suit"
    if "K" in cards:
        return f"the king of {SUIT_NAME[s]} gives second-round control of the suit"
    if len(cards) == 1:
        return f"a singleton {name} ({cards[0]}) gives second-round control of the suit"
    return None


_PREEMPT_CONCEPTS = ("bid_is_weak_jump_overcall", "bid_shows_preemptive_intent", "bid_is_preemptive_jump_response",
                     "bid_is_preemptive_raise", "bid_is_weak_two")


def _vul_fact(c: Ctx) -> Optional[str]:
    """Where vulnerability changes the bridge logic of THIS call, say how
    (a situation fact, so the explainer's prompt -- which also carries the
    Vulnerability line -- has it too). Nothing when nobody is vulnerable or
    the call type does not depend on it."""
    vul = vulnerability_of(c.f)
    if not vul or vul == (False, False):
        return None
    we, they = vul
    favourable, unfavourable = (not we and they), (we and not they)
    preempt = (c.cat in ("opening_weak_two", "opening_preempt", "overcall_jump") or any(
        c.gt.get(x) for x in _PREEMPT_CONCEPTS)) and c.conv not in ("signoff", "game_try_accept", "game_try_decline",
                                                                 "nt_forcing_new_suit", "suit_forcing_new_suit")
    if c.gt.get("bid_is_sacrifice_candidate"):
        if favourable:
            return "at favourable vulnerability, going down costs less than the opponents' vulnerable game"
        if unfavourable:
            return "the sacrifice is made even at unfavourable vulnerability, where going down doubled is expensive"
        return None
    if preempt and c.bid not in NB:
        what = "preemptive raise" if c.bid_suit and c.bid_suit == c.partner_suit else "preempt"
        if favourable:
            return f"at favourable vulnerability the {what} risks little: the opponents are the side with more to lose"
        if we:
            return f"the {what} is made even though vulnerable, where going down doubled is expensive"
        return None
    if c.bid in ("X", "Dbl") and they and not _not_for_penalties(c) and (
            _penaltyish(c) or _double_reading(c) == "penalty"):
        return "the opponents are vulnerable, so every undertrick costs them more"
    if c.bid in ("P", "Pass") and we and 7 <= c.h.hcp < 12 and c.cat in (
            "pass_over_opponents_opening", "advancer_pass", "competitive_later_pass",
            "responder_pass_competitive", "opener_pass_competitive"):
        long_s = max("SHDC", key=lambda x: (c.h.lengths[x], -"SHDC".index(x)))
        ours = {_suit_of(b) for b in c.my_bids + c.partner_bids}
        real = [b["bid"] for b in c.prior if b["bid"] not in NB]
        cheapest = (int(real[-1][0]) + (0 if real[-1][1:] not in ("NT", "N") and "CDHS".index(long_s)
                                        > "CDHS".index(real[-1][-1]) else 1)) if real else 1
        # only a suit nobody has bid that could still be shown at a sane level
        # (not "bidding the diamonds" at the 7-level by the 1D opener)
        if c.h.lengths[long_s] >= 5 and long_s not in c.opp_suits and long_s not in ours and cheapest <= 3:
            return (f"being vulnerable, bidding the {SUIT_NAME[long_s]} with {c.h.hcp} HCP would risk a costly "
                    f"penalty double")
    return None


def _nt_major_fact(c: Ctx) -> Optional[str]:
    """Over partner's notrump opening: no major to look for, or -- with a
    4-card major in a flat 4-3-3-3 -- why 3NT skips Stayman (the trained
    explainer wrote "no four-card major to look for" on such hands)."""
    if not _partner_nt_range(c) or c.opp_suits:
        return None
    majors = [m for m in "SH" if c.h.lengths[m] >= 4]
    if not majors:
        return f"there is no four-card major to look for ({count_phrase(c.h, 'S')}, {count_phrase(c.h, 'H')})"
    if c.bid in ("3NT", "3N") and sorted(c.h.lengths.values()) == [3, 3, 3, 4]:
        m = majors[0]
        return (f"with a flat 4-3-3-3 shape the {holding(c.h, m)} add no ruffing value, so 3NT is bid "
                f"directly instead of looking for a {SUIT_NAME[m][:-1]} fit")
    return None


def _ev_pass_weak(c: Ctx) -> str:
    return c.pick(f"with only {c.h.hcp} HCP the hand is too weak to act",
                  f"{c.h.hcp} HCP is not enough to bid")


def _role_of_first_bid(c: Ctx) -> Optional[tuple]:
    """(the bidder's first real bid, its role, the most HCP a minimum hand of that role
    has): concepts.rules' _role_minimum, which defines bid_pass_no_extra_values --
    the auction's opener 14, a responder to partner's opening 9, anyone else 11."""
    if not c.my_bids:
        return None
    real = [b for b in c.prior if b["bid"] not in NB]
    first = c.my_bids[0]
    if real and real[0]["seat"] == c.you:
        return first, "opening", 14
    if real and real[0]["seat"] == c.partner_seat:
        return first, "response", 9
    k = next(i for i, b in enumerate(c.prior) if b["seat"] == c.you and b["bid"] == first)
    if any(b["seat"] == c.you and b["bid"] in ("X", "Dbl") for b in c.prior[:k]):
        return first, "bid", 11
    partner_first = any(b["seat"] == c.partner_seat and b["bid"] not in NB for b in c.prior[:k])
    return first, ("advance" if partner_first else "overcall"), 11


def _ev_pass_no_extra(c: Ctx) -> Optional[str]:
    """bid_pass_no_extra_values, stated with the threshold the rule uses, so "the 10 HCP add nothing
    to what the earlier bidding already showed" says what that was."""
    r = _role_of_first_bid(c)
    if not r or c.h.hcp > r[2]:
        return None
    first, role, mx = r
    tail = ""
    fs = _suit_of(first)
    # "already shown by 2H" only when that bid said something about strength: a natural suit or
    # notrump bid, not a transfer, Stayman or a cuebid
    if first.endswith(("NT", "N")) and first[:1] in "123" or (fs and _own_suit(c, fs)):
        tail = f", already shown by {first}"
    return f"with {c.h.hcp} HCP {c.seat} has no more than a minimum {role} (at most {mx}){tail}"


def _partner_last_call_range(c: Ctx, game_ok: bool = False) -> Optional[tuple]:
    """(partner's call, lo, hi): the HCP WBridge5 made partner's last call with in this
    exact auction (5th-95th percentile, nlg.call_meaning), when partner made the
    auction's last real bid, below game (game_ok: up to the five level), measured on
    30+ hands and in a range no wider than 10 HCP (a wider range says nothing about game)."""
    real = [(i, b) for i, b in enumerate(c.prior) if b["bid"] not in NB]
    if not real:
        return None
    i, last = real[-1]
    pb = last["bid"]
    if last["seat"] != c.partner_seat or not pb[:1].isdigit() or int(pb[0]) > (5 if game_ok else 3) \
            or (not game_ok and pb in ("3NT", "3N")):
        return None
    from nlg.call_meaning import wb5_entry
    e = wb5_entry([b["bid"] for b in c.prior[:i]], pb)
    if not e or e["n"] < 30 or e["hcp"][2] - e["hcp"][0] > 10:
        return None
    return pb, int(e["hcp"][0]), int(e["hcp"][2])


# what a partnership needs, in HCP, for the next target above partner's contract
GAME_HCP, MINOR_GAME_HCP, SLAM_HCP = 25, 29, 33


def _is_game(call: str) -> bool:
    lvl, strain = int(call[0]), call[1:]
    return lvl >= 5 or (lvl == 4 and strain in ("H", "S", "NT", "N")) or (lvl == 3 and strain in ("NT", "N"))


def _partner_preempt(c: Ctx) -> Optional[str]:
    """Partner's opening, when it is a preempt (a suit opening at the two level or higher, not 2C)."""
    pb = c.partner_bids[-1] if c.partner_bids else None
    if pb and _psuit(c) and pb[0] in "234567" and pb != "2C" and pb == (c.opener or {}).get("bid"):
        return pb
    return None


def _pass_game_context(c: Ctx) -> Optional[str]:
    """Why passing partner's contract: what partner's call showed, added to the hand
    ("partner's 3C showed 2 to 9 HCP, so with 10 HCP the partnership has at most about
    19, short of game"; after partner's game, "short of slam"). Only when it IS short.
    Any pass of partner's last bid, not only after one's own bid ("with 16 HCP ... no reason to bid"
    alone says nothing about why)."""
    if c.bid not in ("P", "Pass"):
        return None
    pr = _partner_last_call_range(c, game_ok=True)
    if not pr:
        return None
    pb, lo, hi = pr
    tot = c.h.hcp + hi
    who = "the preempt" if _partner_preempt(c) == pb else f"partner's {pb}"
    head = f"{who} showed {lo} to {hi} HCP, so the partnership has at most about {tot}"
    if _is_game(pb):
        return f"{head}, short of slam" if tot < SLAM_HCP and int(pb[0]) < 6 else None
    if pb[0] == "4" and pb[1:] in ("C", "D"):
        return f"{head}, short of game at the five level" if tot < MINOR_GAME_HCP else None
    return f"{head}, short of game" if tot < GAME_HCP else None


_LEVEL_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
_STRAINS = ("C", "D", "H", "S", "NT")


def _cheapest_level(c: Ctx, s: str) -> Optional[int]:
    real = [b["bid"] for b in c.prior if b["bid"] not in NB]
    if not real:
        return 1
    last = real[-1].replace("N", "NT").replace("NTT", "NT")
    lvl, strain = int(last[0]), last[1:]
    lv = lvl if _STRAINS.index(s) > _STRAINS.index(strain) else lvl + 1
    return lv if lv <= 7 else None


def _ev_pass_no_good_call(c: Ctx) -> Optional[str]:
    """A pass with opening values over the opponents' bidding, before partner has
    bid: why no call fits -- no five-card suit to overcall, or the suit would have to
    go in at the three level or higher, and a shortage in an unbid suit that rules
    out a takeout double ("14 HCP but no convenient call" alone says nothing)."""
    if c.partner_bids or not c.opp_suits:
        return None
    longest = max(c.h.lengths.values())
    ours = [s for s in "SHDC" if s not in c.opp_suits]
    if not ours:
        return None
    s = max(ours, key=lambda x: (c.h.lengths[x], -"SHDC".index(x)))
    parts = []
    if c.h.lengths[s] == longest and longest < 5:
        parts.append(f"the longest suit is {holding(c.h, s)}, too short to overcall")
    elif c.h.lengths[s] >= 5:
        lv = _cheapest_level(c, s)
        if lv and lv >= 3:
            parts.append(f"{holding(c.h, s)} would have to be bid at the {_LEVEL_WORD[lv]} level")
    short = [u for u in ours if c.h.lengths[u] <= 2 and u in "SH"] or [u for u in ours if c.h.lengths[u] <= 1]
    if short:
        parts.append(f"with {count_phrase(c.h, short[0])} the hand is not shaped for a takeout double")
    if not parts:
        return None
    return f"{c.seat} has {c.h.hcp} HCP, but " + ", and ".join(parts)


_EVIDENCE: Dict[str, Callable[[Ctx], Optional[str]]] = {
    # length / shape
    "bid_shows_4_plus_cards": _ev_length,
    "bid_shows_5_plus_cards": _ev_length,
    "bid_shows_6_plus_cards": _ev_length,
    "bid_shows_7_plus_cards": _ev_length,
    "bid_shows_exactly_5_cards": _ev_length,
    "bid_shows_exactly_6_cards": _ev_length,
    "bid_shows_extra_length_on_rebid": lambda c: c.bid_suit and f"{c.seat} has {holding(c.h, c.bid_suit)}, more than the first bid promised",
    "bid_shows_extra_length_on_jump_raise_after_nt": _ev_length,
    # a meaning concept (a preempt PROMISES one long suit): state it only when the hand has it
    # (not "a void in clubs is the hand's only long suit" for a 5C opening)
    "bid_shows_single_suited": lambda c: (c.bid_suit and c.h.lengths[c.bid_suit] >= 6
                                          and sorted(c.h.lengths.values())[-2] <= 4
                                          and f"{holding(c.h, c.bid_suit)} is the hand's only long suit"),
    "bid_denies_shortness": lambda c: f"the {shape(c.h)} shape has no singleton or void",
    "bid_shows_notrump": lambda c: (f"the hand is {'balanced' if c.h.balanced else 'semi-balanced'} ({shape(c.h)})"
                                    if c.h.balanced or is_semi_balanced(c.h) else None),
    "bid_shows_minor_suit_only": lambda c: f"there is no four-card major ({count_phrase(c.h, 'S')}, {count_phrase(c.h, 'H')})",
    "bid_shows_longer_minor": lambda c: _minor_choice(c),
    "bid_shows_longer_major": lambda c: f"with {holding(c.h, 'S')} and {holding(c.h, 'H')}, the longer major (spades when equal) comes first",
    "bid_prefers_notrump_over_major": lambda c: "the balanced 15 to 17 HCP is described better by 1NT than by the five-card major",
    "bid_shows_5_card_major_in_1nt": lambda c: "the five-card major fits inside a balanced 1NT opening",
    # strength
    "bid_shows_weak_values": _ev_strength("a weak hand for this call"),
    "bid_shows_minimum_values": _ev_strength("a minimum for this position"),
    "bid_shows_invitational_values": _ev_strength("enough to invite game but not to insist on it"),
    "bid_shows_game_forcing_values": _ev_strength("enough to force to game"),
    "bid_shows_extra_values": _ev_strength("more than a minimum for this position"),
    "bid_limits_to_minimum": _ev_strength("a minimum response"),
    "bid_limits_to_invitational": _ev_strength("invitational at most"),
    "bid_limits_to_game_values": lambda c: None,  # too vague to explain with (contradicts "competes")
    "bid_limits_hand": lambda c: None,
    "bid_opening_meets_rule_of_20": lambda c: _ev_rule_of_20(c),
    "bid_opening_meets_ltc_standard": lambda c: f"the losing trick count is {losing_trick_count(c.h)}, normal for an opening",
    "bid_shows_quick_tricks_2plus": lambda c: f"the hand has {quick_tricks(c.h):g} quick tricks",
    "bid_competes_for_partscore": lambda c: f"the {c.h.hcp} HCP are enough to compete for the partscore but not to invite game",
    "bid_shows_defensive_values": lambda c: (f"the {c.h.hcp} HCP are mainly defensive" if c.bid in NB[:1] + ("Pass",)
                                             else f"the {c.h.hcp} HCP give enough values to act"),
    "bid_shows_jump_shift_strength": lambda c: (c.cat == "response_jump_shift"
                                                 and _ev_strength("strong enough for a jump shift")(c)),
    # fit / support
    "bid_shows_support": _ev_support,
    "bid_shows_3_card_support": _ev_support,
    "bid_shows_4_plus_support": _ev_support,
    "bid_promises_8_card_fit": _ev_fit,
    "bid_shows_support_points_invite_plus": lambda c: (_psuit(c) or c.bid_suit) and f"counting shortness, the hand is worth {support_points(c.h, _psuit(c) or c.bid_suit)} support points",
    "bid_shows_minor_fit_raise": _ev_support,
    "bid_is_opener_support_raise": _ev_support,
    "bid_raises_partners_overcall": _ev_support,
    "bid_is_simple_game_raise": _ev_support,
    "bid_denies_fit": lambda c: _psuit(c) and f"{c.seat} has only {holding(c.h, _psuit(c))} in partner's suit",
    "bid_denies_major_fit": lambda c: _psuit(c) and f"{c.seat} has only {holding(c.h, _psuit(c))} in partner's major",
    "bid_denies_3_card_support": lambda c: _psuit(c) and f"{c.seat} has only {holding(c.h, _psuit(c))} in partner's major",
    "bid_denies_major_fit_after_5_card_major": lambda c: _psuit(c) and f"{c.seat} has only {holding(c.h, _psuit(c))} in partner's major",
    # competitive
    "bid_shows_stopper_in_their_suit": _ev_stopper,
    "bid_is_takeout_double": _ev_takeout,
    "bid_is_negative_double": _ev_negative_double,
    "bid_is_cuebid": _ev_cuebid,
    # the suit of the doubled contract, not opp_suits[-1], the last NEW suit
    # (doubling 4H after 1H..4C..4H is about the hearts)
    "bid_double_with_length_in_their_suit": lambda c: c.last_opp_suit and f"{c.seat} holds {holding(c.h, c.last_opp_suit)} {_where(c, c.last_opp_suit, 'in the suit the opponents just bid')}",
    "bid_is_weak_jump_overcall": lambda c: f"with only {c.h.hcp} HCP the long suit is best used to take bidding room away",
    "bid_shows_preemptive_intent": lambda c: f"with only {c.h.hcp} HCP the long suit is best used to take bidding room away",
    "bid_is_preemptive_jump_response": lambda c: f"with only {c.h.hcp} HCP the jump is preemptive, taking bidding room away",
    "bid_is_preemptive_raise": lambda c: f"with only {c.h.hcp} HCP but long trumps, the high raise is mainly preemptive",
    "bid_is_sacrifice_candidate": lambda c: f"with only {c.h.hcp} HCP there is little defense, so bidding on may cost less than defending",
    "bid_is_natural_2nt_overcall": lambda c: f"{c.seat} has {c.h.hcp} HCP and the opponents' suit stopped",
    "bid_is_redouble": lambda c: f"{c.seat} has {c.h.hcp} HCP, enough to show the partnership has the balance of strength",
    "bid_is_penalty_double": lambda c: c.last_opp_suit and f"{c.seat} holds {holding(c.h, c.last_opp_suit)} in the doubled suit and {c.h.hcp} HCP",
    # openings
    "bid_is_1nt_opening": lambda c: f"{c.seat} has {c.h.hcp} HCP",
    "bid_is_2nt_opening": lambda c: f"{c.seat} has {c.h.hcp} HCP",
    "bid_is_strong_2cc_opening": lambda c: f"the hand has {c.h.hcp} HCP ({wb5_points(c.h)} points) and only {losing_trick_count(c.h)} losers",
    "bid_is_weak_two": lambda c: c.bid_suit and f"{c.seat} has {holding(c.h, c.bid_suit)} and {c.h.hcp} HCP, too weak to open at the one level",
    "bid_is_light_third_seat_weak_two": lambda c: c.bid_suit and (
        f"after partner's pass, {'a good' if _good_suit(c.h.cards[c.bid_suit]) else 'a'} five-card suit is enough "
        f"for a weak two in third seat"),
    "bid_is_unusual_4nt_opening": lambda c: f"{c.seat} has {holding(c.h, 'D')} and {holding(c.h, 'C')}",
    "bid_is_preempt_quality_good": lambda c: c.bid_suit and f"the suit is headed by {top_honors(c.h, c.bid_suit)}",
    "bid_is_preempt_quality_marginal": lambda c: c.bid_suit and f"the suit is weak at the top ({top_honors(c.h, c.bid_suit)})",
    # responses / conventions
    "bid_is_two_over_one_response": lambda c: c.bid_suit and f"{c.seat} has {holding(c.h, c.bid_suit)} and {c.h.hcp} HCP, enough for a two-level response",
    "bid_stayman_with_4_card_major": lambda c: f"{c.seat} holds {holding(c.h, 'H')} and {holding(c.h, 'S')}, so a 4-4 major fit is possible",
    "bid_transfer_with_5_card_major": lambda c: (c.bid_suit in ("D", "H")) and f"{c.seat} has {holding(c.h, 'H' if c.bid_suit == 'D' else 'S')}, so partner's stronger hand should declare",
    "bid_completes_transfer_without_super_accept": lambda c: c.bid_suit and f"with {holding(c.h, c.bid_suit)} and {c.h.hcp} HCP there is no reason to jump",
    "bid_waiting_response_without_positive": lambda c: f"with {c.h.hcp} HCP and no good five-card suit, there is no positive response to make",
    "bid_positive_response_to_2c": lambda c: (f"{c.seat} has {holding(c.h, c.bid_suit)} and {c.h.hcp} HCP" if c.bid_suit
                                              else f"{c.seat} has {c.h.hcp} HCP"),
    "bid_is_splinter": lambda c: None,
    "bid_is_singleton_splinter": lambda c: c.bid_suit and _psuit(c) and f"{c.seat} has {holding(c.h, c.bid_suit)} and {holding(c.h, _psuit(c))} in partner's suit",
    "bid_is_void_splinter": lambda c: c.bid_suit and _psuit(c) and f"{c.seat} has {holding(c.h, c.bid_suit)} and {holding(c.h, _psuit(c))} in partner's suit",
    "bid_shows_stayman_major_structure": lambda c: f"{c.seat} holds {holding(c.h, 'H')} and {holding(c.h, 'S')}",
    "bid_is_stayman_game_fit_raise": _ev_support,
    # slam
    "bid_shows_aces_0_or_3": _ev_aces,
    "bid_shows_aces_1_or_4": _ev_aces,
    "bid_shows_aces_2": _ev_aces,
    "bid_shows_aces_3": _ev_aces,
    "bid_blackwood_with_fit": lambda c: _ev_support(c),
    "bid_slam_with_enough_aces": _ev_blackwood_math,
    "bid_king_ask_with_all_aces": _ev_blackwood_math,
    "bid_small_slam_missing_a_king": _ev_blackwood_math,
    "bid_answers_king_ask": lambda c: _ev_kings(c),
    "bid_shows_first_control": lambda c: c.bid_suit and _control_phrase(c),
    "bid_shows_second_control": lambda c: c.bid_suit and _control_phrase(c),
    "bid_shows_control": lambda c: c.bid_suit and _control_phrase(c),
    # passes
    "bid_pass_lacks_opening_values": lambda c: f"with {c.h.hcp} HCP the hand is short of an opening (Rule of 20: {rule_of_20(c.h)})",
    "bid_pass_lacks_values_to_compete": lambda c: (
        None if not c.partner_bids  # a situation fact (_forced), so the explainer's prompt has the level too
        else f"with {c.h.hcp} HCP and {holding(c.h, _psuit(c)) if _psuit(c) else 'no fit'} in partner's suit, there is no reason to compete"),
    "bid_pass_too_weak_to_respond": _ev_pass_too_weak,
    "bid_pass_no_extra_values": lambda c: _ev_pass_no_extra(c) or f"the {c.h.hcp} HCP add nothing to what the earlier bidding already showed",
    "bid_pass_content_with_final_contract": lambda c: (
        f"partner's {c.partner_bids[-1]} is the final contract" if c.partner_bids and c.partner_bids[-1][:1] == "7"
        else f"with {c.h.hcp} HCP there is no reason to go past partner's {c.partner_bids[-1]}" if c.partner_bids
        else f"with {c.h.hcp} HCP there is no reason to go past partner's contract"),
}


_GROUP = {}
for _c in ("bid_shows_weak_values", "bid_shows_minimum_values", "bid_shows_invitational_values",
           "bid_shows_game_forcing_values", "bid_shows_extra_values", "bid_limits_to_minimum",
           "bid_limits_to_invitational", "bid_competes_for_partscore", "bid_shows_defensive_values",
           "bid_shows_jump_shift_strength", "bid_is_sacrifice_candidate", "bid_pass_no_extra_values"):
    _GROUP[_c] = "strength"
for _c in ("bid_shows_support", "bid_shows_3_card_support", "bid_shows_4_plus_support", "bid_shows_minor_fit_raise",
           "bid_is_opener_support_raise", "bid_raises_partners_overcall", "bid_is_simple_game_raise",
           "bid_is_stayman_game_fit_raise", "bid_blackwood_with_fit", "bid_shows_support_points_invite_plus",
           "bid_is_cuebid"):
    _GROUP[_c] = "support"
for _c in ("bid_shows_4_plus_cards", "bid_shows_5_plus_cards", "bid_shows_6_plus_cards", "bid_shows_7_plus_cards",
           "bid_shows_exactly_5_cards", "bid_shows_exactly_6_cards", "bid_shows_extra_length_on_jump_raise_after_nt",
           "bid_shows_single_suited"):
    _GROUP[_c] = "length"
for _c in ("bid_is_weak_jump_overcall", "bid_shows_preemptive_intent", "bid_is_preemptive_jump_response",
           "bid_is_preemptive_raise"):
    _GROUP[_c] = "preempt"
for _c in ("bid_shows_notrump", "bid_denies_shortness"):
    _GROUP[_c] = "nt_shape"
for _c in ("bid_shows_aces_0_or_3", "bid_shows_aces_1_or_4", "bid_shows_aces_2", "bid_shows_aces_3"):
    _GROUP[_c] = "aces"
for _c in ("bid_slam_with_enough_aces", "bid_king_ask_with_all_aces", "bid_small_slam_missing_a_king"):
    _GROUP[_c] = "blackwood_math"
for _c in ("bid_denies_fit", "bid_denies_major_fit", "bid_denies_3_card_support", "bid_denies_major_fit_after_5_card_major"):
    _GROUP[_c] = "no_fit"
for _c in ("bid_is_takeout_double", "bid_double_with_length_in_their_suit", "bid_is_penalty_double"):
    _GROUP[_c] = "their_suit"  # "short in their suit" and "holds 3 in their suit" never both
for _c in ("bid_is_singleton_splinter", "bid_is_void_splinter"):
    _GROUP[_c] = "bid_is_splinter"
for _c in ("bid_shows_first_control", "bid_shows_second_control", "bid_shows_control"):
    _GROUP[_c] = "control"
_STRENGTH_IN = ("HCP", "points", "Rule of 20")


def _stayman_answer(c: Ctx) -> str:
    """Opener's reply to Stayman: 2D denies a four-card major, 2H shows four
    hearts (spades not denied), 2S four spades and fewer than four hearts."""
    S, H = count_phrase(c.h, "S"), count_phrase(c.h, "H")
    if c.bid in ("2D", "3D"):
        return f"{c.bid} denies a four-card major ({S}, {H})"
    if c.bid in ("2H", "3H"):
        return (f"{c.bid} shows four or more hearts" + (", bid before the spades with both majors"
                                                       if c.h.lengths["S"] >= 4 else "")
                + f"; {c.seat} holds {holding(c.h, 'H')}")
    return f"{c.bid} shows four or more spades and denies four hearts; {c.seat} holds {holding(c.h, 'S')}"


def _trap_pass(c: Ctx) -> Optional[str]:
    """A strong pass with length and honours in the suit the right-hand
    opponent just bid: the hand wants to defend (e.g. a pass of 15 HCP with A Q 3 2 over East's 2H,
    where the small hearts push toward the pass in Integrated Gradients), not "no reason to bid"."""
    if c.bid not in ("P", "Pass") or c.h.hcp < 12 or not c.prior:
        return None
    rho = c.prior[-1]
    s = _suit_of(rho["bid"]) if rho["bid"] not in NB else None
    if not s or rho["seat"] in (c.you, c.partner_seat):
        return None
    cards = c.h.cards[s]
    suit_hcp = sum(HCP_VALUES.get(r, 0) for r in cards)
    if not ((len(cards) >= 4 and suit_hcp >= 5) or (len(cards) >= 3 and suit_hcp >= 6)):
        return None
    tail = (f": if partner reopens with a double, {c.seat} can leave it in for penalties"
            if int(rho["bid"][0]) <= 3 else "")
    return f"with {holding(c.h, s)} sitting over the {SUIT_NAME[s]} bidder, {c.seat} would rather defend{tail}"


def _forced(c: Ctx) -> List[tuple]:
    """(group, clause) facts that DEFINE the situation and so are stated
    whatever the model's top reasons are: a cuebid's support for partner, a
    takeout double's shape, a negative double's unbid suits, the ace/king
    arithmetic around 5NT, the major partner's negative double promised,
    and the length partner's preempt promised when support is short."""
    out = []
    if c.cat == "slam_blackwood_answer" and c.bid in ("5C", "5D", "5H", "5S"):
        n_aces = sum("A" in c.h.cards[x] for x in "SHDC")
        out.append(("aces", f"{c.seat} holds {_WORDS[n_aces]} ace{'s' if n_aces != 1 else ''}"))
    if c.bid in ("P", "Pass") and not c.partner_bids and c.gt.get("bid_pass_lacks_values_to_compete"):
        out.append(("pass_reason", _ev_pass_no_suit(c)))
    if c.conv == "nt_opener_control":
        out.append(("control", _control_phrase(c)))
    if c.my_bids == ["2C", "2NT"] and c.partner_bids == ["2D", "3C"] and c.bid in ("3D", "3H", "3S"):
        out.append(("stayman", _stayman_answer(c)))
    if c.conv == "nt_overcall_stayman":
        S, H = c.h.lengths["S"], c.h.lengths["H"]
        out.append(("length", f"{c.seat} holds {holding(c.h, 'H')} and {holding(c.h, 'S')}"
                              + (", so a 4-4 major fit is possible" if max(S, H) >= 4 else "")))
    if c.conv in ("nt_overcall_transfer", "nt_overcall_transfer_accept"):
        t = _target_suit(c)
        if t and c.conv == "nt_overcall_transfer":
            out.append(("length", f"{c.seat} has {holding(c.h, t)}, so partner's stronger hand should declare"))
        elif t:
            out.append(("length", f"{c.seat} holds {holding(c.h, t)}"))
    if c.conv == "nt_minor_puppet":
        m = "C" if c.h.lengths["C"] >= c.h.lengths["D"] else "D"
        out.append(("length", f"{c.seat} holds {holding(c.h, m)} and {c.h.hcp} HCP, too weak to look for game"))
    if _takeout_cuebid(c):
        unbid = [x for x in "SHDC" if x not in c.opp_suits and x != c.bid_suit]
        two = sorted(unbid, key=lambda x: (-c.h.lengths[x], "SHDC".index(x)))[:2]
        if len(two) == 2:  # (after RHO's new suit only one unbid suit may be left)
            out.append(("their_suit", f"{c.seat} has {holding(c.h, two[0])} and {holding(c.h, two[1])} in the unbid suits"))
        elif two:
            out.append(("their_suit", f"{c.seat} has {holding(c.h, two[0])} in the unbid suit"))
    if c.conv == "nt_minor_puppet_accept":
        out.append(("relay", f"the 3C reply is forced whatever {c.seat} holds, so that the weak hand can play "
                             f"in its long minor"))
    if c.conv == "major_2nt_response" and c.partner_suit:
        out.append(("support", f"{c.seat} holds {holding(c.h, c.partner_suit)} in partner's suit, and {c.h.hcp} HCP "
                               f"opposite an opening bid are enough for game"))
    if c.conv == "nt_forcing_new_suit":
        out.append(("strength", f"with {c.h.hcp} HCP opposite partner's 15 to 17, the partnership has "
                                f"{c.h.hcp + 15} to {c.h.hcp + 17}"
                                + (", enough for game" if c.h.hcp + 15 >= 25 else "")))
    cued = _doubled_cuebid_suit(c)
    if cued:
        out.append(("their_suit", f"{c.seat} holds {holding(c.h, cued)}"
                                  + ("" if _own_suit(c, cued) else " in partner's suit")))
    sr = support_redouble_suit(c.f)
    if sr:
        out.append(("their_suit", f"{c.seat} holds {holding(c.h, sr)} in partner's suit"
                                  + (f"; with four {SUIT_NAME[sr]} the system raises directly instead"
                                     if c.h.lengths[sr] == 3 else "")))
    rd = responsive_double_suits(c.f)
    if rd:
        two = sorted(rd, key=lambda x: (-c.h.lengths[x], "SHDC".index(x)))[:2]
        out.append(("their_suit", f"{c.seat} has {holding(c.h, two[0])} and {holding(c.h, two[1])} in the unbid suits"))
    cd = competitive_double_suit(c.f)
    if cd:
        out.append(("their_suit", f"{c.seat} has {c.h.hcp} HCP and {holding(c.h, cd)} in partner's suit"))
    sd = support_double_suit(c.f)
    if sd:
        out.append(("their_suit", f"{c.seat} holds {holding(c.h, sd)} in partner's suit"
                                  + (f"; with four {SUIT_NAME[sd]} the system raises directly instead"
                                     if c.h.lengths[sd] == 3 else "")))
    if c.gt.get("bid_is_cuebid") and not c.gt.get("bid_is_michaels_cuebid"):
        out.append(("support", _ev_cuebid(c)))
    dr = _double_reading(c)
    if c.bid in ("X", "Dbl") and c.gt.get("bid_is_takeout_double") and dr != "penalty":
        out.append(("their_suit", _ev_takeout(c)))
    if dr == "penalty" and not _penaltyish(c) and c.last_opp_bid and not _doubled_nt_convention(c):
        out.append(("their_suit", _ev_data_penalty(c)))
    art = _doubled_nt_convention(c)
    if art:
        out.append(("their_suit", f"{c.seat} holds {holding(c.h, art[1])}"))
    ar = _doubled_artificial_reply(c)
    if ar:
        out.append(("their_suit", f"{c.seat} holds {holding(c.h, ar)}"))
    if dr == "competitive" and _penaltyish(c) and not (support_double_suit(c.f) or competitive_double_suit(c.f)
                                                       or _doubled_cuebid_suit(c)):
        ps = c.partner_suit
        where = "the agreed suit" if ps and _own_suit(c, ps) else "partner's suit"
        out.append(("their_suit", f"{c.seat} has {c.h.hcp} HCP"
                                  + (f" and {holding(c.h, ps)} in {where}" if ps else "")))
    if c.cat == "responder_negative_double" and dr != "penalty":
        out.append(("bid_is_negative_double", _ev_negative_double(c)))
    if c.bid in ("5NT", "5N") and c.my_bids[-1:] in (["4NT"], ["4N"]):
        out.append(("blackwood_math", _ev_blackwood_math(c)))
    if c.cat == "slam_bid" and blackwood_context(c.f):
        out.append(("blackwood_math", _ev_blackwood_math(c)))
    if c.cat == "slam_blackwood_ask":
        ps = c.partner_suit
        ls = max("SHDC", key=lambda x: (c.h.lengths[x], -"SHDC".index(x)))
        where = "the agreed suit" if ps and _own_suit(c, ps) else "partner's suit"  # (my 3D, partner's 4D raise)
        what = (f"{holding(c.h, ps)} in {where}" if ps and c.h.lengths[ps] >= 3
                else holding(c.h, ls) if c.h.lengths[ls] >= 6 else None)
        n_aces = sum("A" in c.h.cards[x] for x in "SHDC")
        aces = f"{_WORDS[n_aces]} ace{'s' if n_aces != 1 else ''}"
        out.append(("support" if what and ps and c.h.lengths[ps] >= 3 else "aces",
                    (f"{c.seat} has {what} and {aces}" if what else f"{c.seat} holds {aces}")
                    + f", so {c.seat} needs to know how many aces partner holds before bidding slam"))
    if _natural_nt(c):
        out.append(("nt_points", _nt_points(c)))
        out.append(("nt_major", _nt_major_fact(c)))
        if c.opp_suits:
            out.append(("bid_shows_stopper_in_their_suit", _nt_stoppers(c)))
    if c.gt.get("bid_is_waiting_response"):
        out.append(("waiting", f"with {c.h.hcp} HCP, the artificial 2D keeps the bidding low so partner can "
                               f"describe the strong hand first"))
    if c.cat == "slam_blackwood_answer" and c.bid[:1] == "6":
        out.append(("bid_answers_king_ask", _ev_kings(c)))
    if c.bid_suit and c.partner_suit and (c.gt.get("bid_is_singleton_splinter") or c.gt.get("bid_is_void_splinter")):
        out.append(("bid_is_splinter", f"{c.seat} has {holding(c.h, c.bid_suit)} and "
                                       f"{holding(c.h, c.partner_suit)} in partner's suit"))
    if c.bid_suit and c.gt.get("bid_is_control_bid"):
        out.append(("control", _control_phrase(c)))
    if c.cat == "overcall_notrump" and c.bid in ("4NT", "4N"):
        out.append(("minors", f"{c.seat} has {holding(c.h, 'D')} and {holding(c.h, 'C')}"))
    if c.gt.get("bid_is_check_back"):
        m = next((_suit_of(b) for b in c.my_bids if _suit_of(b) in ("H", "S")), None)
        if m and c.bid_suit:  # a natural second suit in WBridge5 (see _meaning_override)
            out.append(("check_back", f"{c.seat} has {holding(c.h, m)} and {holding(c.h, c.bid_suit)}"))
    if c.gt.get("bid_stayman_with_4_card_major"):
        out.append(("bid_stayman_with_4_card_major", _EVIDENCE["bid_stayman_with_4_card_major"](c)))
    if c.gt.get("bid_transfer_with_5_card_major"):
        out.append(("bid_transfer_with_5_card_major", _EVIDENCE["bid_transfer_with_5_card_major"](c)))
    # opener answering Stayman: my 1NT OPENED the auction (not 1S P 1NT P 2C P 2S)
    if ((c.my_bids == ["1NT"] and c.partner_bids == ["2C"] and c.bid in ("2D", "2H", "2S"))
            or (c.my_bids == ["2NT"] and c.partner_bids == ["3C"] and c.bid in ("3D", "3H", "3S"))) \
            and (c.opener or {}).get("seat") == c.you and not c.opp_bids:
        out.append(("bid_shows_stayman_major_structure", _stayman_answer(c)))
    if c.my_bids in (["2C"], ["3C"]) and _partner_nt_range(c) and len(c.partner_bids) == 2:
        ans = c.partner_bids[-1]  # partner's answer to my Stayman
        if ans[1:] in ("H", "S") and c.h.lengths[ans[-1]] < 4 and c.bid in ("3NT", "3N"):
            out.append(("no_fit", f"with only {holding(c.h, ans[-1])} there is no {SUIT_NAME[ans[-1]][:-1]} fit, "
                                  f"so the hand plays in notrump"))
    s = c.bid_suit
    if s and s in c.implied_majors and c.h.lengths[s] >= 4:
        out.append(("length", f"{c.seat} holds {holding(c.h, s)}, so with partner's four the partnership "
                              f"has an eight-card fit or better"))
    conv = c.conv
    if conv == "michaels_ask":
        pm = "H" if _direct_cuebid_suit(c, c.partner_seat) == "S" else "S"
        out.append(("support", f"{c.seat} has {holding(c.h, pm)} in partner's known major and "
                               f"{holding(c.h, 'D')} and {holding(c.h, 'C')} in the minors"))
    elif conv == "michaels_minor":
        out.append(("length", f"{c.seat} holds {holding(c.h, c.bid_suit)}"))
    elif conv == "stayman_after_2c":
        out.append(("bid_stayman_with_4_card_major",
                    f"{c.seat} holds {holding(c.h, 'H')} and {holding(c.h, 'S')}"))
    elif conv == "transfer_after_2c":
        out.append(("length", f"{c.seat} has {holding(c.h, 'H' if c.bid_suit == 'D' else 'S')}"))
    elif conv == "signoff":
        if c.h.hcp <= 13:
            out.append(("strength", f"with only {c.h.hcp} HCP {c.seat} has nothing extra for slam"))
        else:  # (not "with only 18 HCP ... nothing extra" for a responder's 18-count)
            out.append(("strength", f"{c.seat} holds {holding(c.h, c.bid_suit)} and {c.h.hcp} HCP, and stops in game"))
    elif conv in ("game_try_accept", "game_try_decline"):
        t = _suit_of(c.partner_bids[-1])
        out.append(("strength", f"{c.seat} has {holding(c.h, t)} in partner's try suit and {c.h.hcp} HCP"))
    real = [b for b in c.prior if b["bid"] not in NB]
    px = [i for i, b in enumerate(c.prior) if b["seat"] == c.partner_seat and b["bid"] in ("X", "Dbl")]
    if (c.bid in ("P", "Pass") and px and all(b["bid"] in ("P", "Pass") for b in c.prior[px[-1] + 1:])
            and real and real[-1]["seat"] not in (c.you, c.partner_seat)):
        doubled = real[-1]["bid"]
        s_d = _suit_of(doubled)
        out.append(("penalty", f"passing leaves partner's double of {doubled} in, playing for penalties"
                               + (f" with {holding(c.h, s_d)} in their suit" if s_d and c.h.lengths[s_d] >= 3 else "")))
    tp = _trap_pass(c)
    if tp:
        out.append(("trap", tp))
    vf = _vul_fact(c)
    if vf:
        out.append(("vulnerability", vf))
    pr = _partner_promise(c)
    if s and pr and s == c.partner_suit and c.h.lengths[s] <= 2:
        pb, n = pr
        out.append(("promise", f"partner's {pb} promised at least {_WORDS[n]} {SUIT_NAME[s]}"
                               + (f", so {holding(c.h, s)} is enough support" if n + c.h.lengths[s] >= 8 else "")))
    return [(g, cl) for g, cl in out if cl]


_HOLDING_RE = re.compile(r"\b(?:\d+ \w+|a singleton \w+|a void in \w+) \([^)]*\)")
_PLAIN_LENGTH_RE = re.compile(r"^(?:\w+ holds|the hand has|\w+ has) (?:\d+ \w+|a singleton \w+) \([^)]*\)$")


def evidence(c: Ctx, k: int = 2) -> List[str]:
    """Up to k clauses, one per reason GROUP (so "balanced" and "no
    singleton" or two strength labels never both appear), each citing the
    hand; a fit reason is folded into the support clause; for a real bid a
    strength fact is added if no chosen clause mentions HCP/points. The
    situation's defining facts (_forced) come first, except the
    vulnerability, which comes last, after the hand."""
    clauses, groups = [], set()
    forced = _forced(c)
    vul_cl = next((cl for g, cl in forced if g == "vulnerability"), None)
    preempt = c.cat in ("overcall_jump", "opening_weak_two", "opening_preempt")  # (a competitive raise does compete)
    for g, cl in forced:
        if g not in groups:
            clauses.append(cl); groups.add(g)
    if clauses:
        k = min(3, len(clauses) + k)
    reasons = list(c.reasons)
    if c.cat == "slam_blackwood_answer" and c.bid[:1] == "5":
        reasons = []  # the reply counts aces: its suit's length or quality says nothing
    for concept in reasons:
        if c.conv in ("signoff", "game_try_accept", "game_try_decline") and concept in _PREEMPTIVE:
            continue  # returning to the agreed suit after a slam/game try is not a preempt
        if c.conv in ("nt_forcing_new_suit", "suit_forcing_new_suit") and (
                concept in _PREEMPTIVE or concept == "bid_competes_for_partscore"):
            continue  # forcing opposite a strong notrump: neither a preempt nor a partscore bid
        if concept in ("bid_is_penalty_double", "bid_double_with_length_in_their_suit") and _not_for_penalties(c):
            continue  # a double of their cuebid of our suit, or a support double ("in the suit the opponents just bid")
        if concept in ("bid_is_takeout_double", "bid_is_negative_double") and _double_reading(c) == "penalty":
            continue  # WBridge5's partner leaves this double in (e.g. of a 1NT overcall)
        if concept == "bid_competes_for_partscore" and preempt:
            continue  # a preempt (a 9-card weak jump overcall) is more than "enough to compete for the partscore"
        if concept == "bid_competes_for_partscore" and (c.gt.get("bid_is_cuebid") or "cuebid" in c.cat):
            continue  # a cuebid is a limit raise or better: "... not to invite game" would contradict it
        if concept == "bid_competes_for_partscore" and any(
                b[:1].isdigit() and _suit_of(b) in c.opp_suits for b in c.my_bids):
            continue  # after this player's own cuebid of their suit (a limit raise or better), 3S is no
            #           "partscore, not invite" bid: P 1D 1S X 2D P 2H 3C 3S
        fn = _EVIDENCE.get(concept)
        g = _GROUP.get(concept, concept)
        if fn is None or g in groups:
            continue
        try:
            clause = fn(c)
        except Exception:
            clause = None
        if not clause:
            continue
        if g == "support" and "bid_promises_8_card_fit" in reasons:
            clause += ", so the partnership has an eight-card fit or better"
            groups.add("bid_promises_8_card_fit")
        if concept == "bid_promises_8_card_fit" and "support" in groups:
            continue
        if " HCP" in clause and any(" HCP" in x for x in clauses):
            continue  # one strength statement is enough
        held = set(_HOLDING_RE.findall(clause))
        dup = [i for i, x in enumerate(clauses) if held & set(_HOLDING_RE.findall(x))]
        if dup:  # not "The hand has 6 hearts (...); North has 6 hearts (...) and 10 HCP, too weak..."
            i = dup[0]
            if _PLAIN_LENGTH_RE.match(clauses[i]):
                clauses[i] = clause; groups.add(g)
            continue
        clauses.append(clause); groups.add(g)
        if len(clauses) >= k:
            break
    if c.bid_suit and c.bid_suit in c.opp_suits and (c.gt.get("bid_is_michaels_cuebid") or c.cat == "overcall_cuebid"):
        long2 = sorted("SHDC", key=lambda x: (-c.h.lengths[x], "SHDC".index(x)))[:2]
        clauses.insert(0, f"{c.seat} has {holding(c.h, long2[0])} and {holding(c.h, long2[1])}")
    natural = c.bid_suit and not _is_artificial(c) and c.conv not in (
        "stayman_after_2c", "transfer_after_2c", "nt_minor_puppet", "nt_minor_puppet_accept",
        "nt_overcall_stayman", "nt_overcall_transfer", "nt_opener_control")
    cite = c.bid_suit if natural else _target_suit(c)
    if cite and clauses and SUIT_NAME[cite] not in " ".join(clauses) and SUIT_NAME[cite][:-1] not in " ".join(clauses):
        clauses.insert(0, f"{c.seat} holds {holding(c.h, cite)}")
    if _natural_nt(c) and clauses and len(clauses) < 3:
        # complete a notrump call's picture: shape / no major to look for
        for x in _nt_evidence(c):
            have = " ".join(clauses)
            if len(clauses) >= 3 or x in clauses or (" HCP" in x and " HCP" in have) \
                    or ("balanced" in x and ("shape" in have or "balanced" in have)) \
                    or ("stop" in x and "stop" in have):
                continue
            clauses.append(x)
    if clauses and c.bid not in NB and not any(w in " ".join(clauses) for w in _STRENGTH_IN):
        # append to "X holds <holding>" only when that clause is a plain list
        # (not the ungrammatical "...so a 4-4 major fit is possible and 8 HCP")
        if clauses[0].startswith(f"{c.seat} holds ") and " HCP" not in clauses[0] and "," not in clauses[0]:
            clauses[0] += f" and {c.h.hcp} HCP"
        else:
            clauses.append(c.pick(f"the hand has {c.h.hcp} HCP", f"{c.seat} has {c.h.hcp} HCP"))
    if vul_cl in clauses and clauses[-1] != vul_cl:
        clauses.remove(vul_cl)
        clauses.append(vul_cl)
    hh = [i for i, x in enumerate(clauses) if x.startswith(("the hand has ", "The hand has "))]
    if len(hh) >= 2:  # "the hand has 5 spades (...), the hand has 2 quick tricks, and the hand has 11 HCP"
        parts = [clauses[i][len("the hand has "):] for i in hh]
        parts = [p_.replace(" and ", ", ") for p_ in parts[:-1]] + parts[-1:]  # no "A and B and C"
        merged = "the hand has " + (", ".join(parts[:-1]) + " and " + parts[-1])
        clauses = [merged if i == hh[0] else x for i, x in enumerate(clauses) if i == hh[0] or i not in hh]
    return clauses


def with_context(c: Ctx, clauses: List[str]) -> List[str]:
    """The reasons first, then what partner's last call showed against the hand
    (_pass_game_context), which would otherwise crowd out the concept's own clause
    ("one strength statement is enough" in evidence)."""
    ctx = _pass_game_context(c)
    # not when a clause already weighs partner's range ("no game opposite partner's 15 to 17"): the
    # same point twice
    if ctx and ctx not in clauses and not any("opposite partner's" in x or "the partnership has at most" in x for x in clauses):
        clauses = list(clauses) + [ctx]
    return clauses


def reason_clauses(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each of the explanation's reason concepts with the clause of hand evidence the
    teacher writes for it (None when the concept has no hand clause): the link from
    the model's concepts to the text, shown concept by concept in the interface."""
    from nlg.explanation_inputs import hand_dependent_concepts
    c = Ctx(frame, random.Random(f"{frame.get('deal_id')}:{frame.get('step')}"))
    dep = hand_dependent_concepts(frame)
    out = []
    for concept in c.reasons:
        fn = _EVIDENCE.get(concept)
        try:
            clause = fn(c) if fn else None
            if not clause and concept == "bid_pass_lacks_values_to_compete" and not c.partner_bids:
                clause = _ev_pass_no_suit(c)  # stated as a situation fact (_forced) in the explanation
        except Exception:
            clause = None
        out.append({"name": concept, "gloss": gloss_for_concept(concept), "clause": _cap(clause) if clause else None,
                    "hand_based": bool(dep.get(concept))})
    return out


def _target_suit(c: Ctx) -> Optional[str]:
    """The suit an artificial call is really about (a transfer's major)."""
    if (c.cat == "response_1nt_transfer" or c.gt.get("bid_is_jacoby_transfer")
            or c.conv == "nt_overcall_transfer") and c.bid_suit in ("D", "H"):
        return "H" if c.bid_suit == "D" else "S"
    if c.conv == "nt_overcall_transfer_accept":
        return c.bid_suit
    return None


def _partner_nt_range(c: Ctx):
    """(lo, hi) of partner's notrump OPENING when I am answering it, else None."""
    opening = (c.opener or {}).get("bid") if (c.opener or {}).get("seat") == c.partner_seat else None
    rng_ = {"1NT": (15, 17), "1N": (15, 17), "2NT": (20, 21), "2N": (20, 21)}.get(opening or "")
    if not rng_ or c.opp_bids:
        return None
    stayman = "2C" if opening in ("1NT", "1N") else "3C"
    # answering partner's NT opening directly, or after my Stayman and partner's answer
    return rng_ if not c.my_bids or c.my_bids == [stayman] else None


def _nt_points(c: Ctx) -> Optional[str]:
    rng_ = _partner_nt_range(c)
    if not rng_:
        return None
    lo, hi = c.h.hcp + rng_[0], c.h.hcp + rng_[1]
    head = f"with {c.h.hcp} HCP opposite partner's {rng_[0]} to {rng_[1]}, the partnership has {lo} to {hi}"
    if lo >= 25:
        return head + ", enough for game"
    if c.bid in ("3NT", "3N") and hi >= 24:
        long_s = max("SHDC", key=lambda x: (c.h.lengths[x], -"SHDC".index(x)))
        if c.h.lengths[long_s] >= 5:  # WBridge5 bids game on 9-10 HCP with a long running suit
            return head + f", borderline for game, and {holding(c.h, long_s)} should add tricks"
        return head + ", borderline for game"
    return head + (", enough to invite game" if hi >= 25 else "")


def _nt_stoppers(c: Ctx) -> Optional[str]:
    """The stoppers in the opponents' suits -- and the suit that has none, so a 3NT with 8 3 in their
    hearts is not explained by "the A 8 5 4 of diamonds stops the opponents' suit"."""
    stops = [x for x in c.opp_suits if is_stopper(c.h.cards[x])]
    open_ = [x for x in c.opp_suits if not is_stopper(c.h.cards[x])]
    if not stops:
        return None
    parts = [f"{' '.join(c.h.cards[x])} of {SUIT_NAME[x]}" for x in stops[:2]]
    out = (f"the {' and the '.join(parts)} stop{'s' if len(parts) == 1 else ''} the opponents' "
           f"{'suit' if len(parts) == 1 and not open_ else SUIT_NAME[stops[0]] if len(parts) == 1 else 'suits'}")
    if open_:
        x = open_[0]
        out += f", but there is no stopper in {SUIT_NAME[x]} ({' '.join(c.h.cards[x]) or 'void'})"
    return out


def _natural_nt(c: Ctx) -> bool:
    return (c.bid[:1] in "123" and c.bid.endswith(("NT", "N")) and not c.gt.get("bid_is_unusual_2nt")
            and not c.gt.get("bid_is_conventional") and c.conv != "michaels_ask")


def _nt_evidence(c: Ctx) -> List[str]:
    """A natural notrump call: combined values opposite partner's notrump
    range, stoppers in the opponents' suits, the balanced shape, and (over
    partner's 1NT) no four-card major to look for, rather than only "South has 12 HCP with a 2-3-4-4
    shape"."""
    h = c.h
    rng_ = _partner_nt_range(c)
    out = [x for x in (_nt_points(c), _nt_stoppers(c)) if x]
    if h.balanced or is_semi_balanced(h):
        out.append(f"the hand is {'balanced' if h.balanced else 'semi-balanced'} ({shape(h)})")
    if rng_ and not c.opp_suits and max(h.lengths["S"], h.lengths["H"]) < 4 and len(out) < 3:
        out.append(f"there is no four-card major to look for ({count_phrase(h, 'S')}, {count_phrase(h, 'H')})")
    if not any(" HCP" in x for x in out):
        out.append(f"{c.seat} has {h.hcp} HCP")
    return out[:3]


def _fallback_evidence(c: Ctx) -> List[str]:
    """When the model's reasons give no clause: a plain, true description."""
    if c.bid in ("P", "Pass"):
        real = [b for b in c.prior if b["bid"] not in NB]
        if real and real[-1]["seat"] == c.partner_seat and (
                real[-1]["bid"] in ("3NT", "3N", "4H", "4S", "5C", "5D") or real[-1]["bid"][0] in "67"):
            if c.gt.get("bid_pass_accepts_partners_contract"):  # the headline already says "leaving partner in 4S"
                return [f"with {c.h.hcp} HCP {c.seat} has nothing more to add to partner's game"
                        if real[-1]["bid"][0] not in "67" else f"with {c.h.hcp} HCP {c.seat} has nothing more to add"]
            return [f"partner has placed the contract in {real[-1]['bid']}, and {c.seat} has nothing more to add"]
        if c.my_bids:  # the actor has already bid: what that showed (not "nothing more to add")
            mine = _ev_pass_no_extra(c)
            if mine:
                return [mine]
            first = c.my_bids[0]
            fs = _suit_of(first)
            if fs and fs in natural_suits_of(auction_from_frame(c.f), (auction_from_frame(c.f).actor_seat(),)):
                return [f"{c.seat} has {c.h.hcp} HCP and has already shown the {SUIT_NAME[fs]} with {first}"]
            return [f"{c.seat} has {c.h.hcp} HCP and has already described the hand with {first}"]
        if c.cat in ("response_pass", "responder_pass_competitive", "response_1nt_pass"):
            return [_ev_pass_too_weak(c)]
        if c.partner_suit:
            if strong_pass_note(c.f, c.h):  # the unusual note says what WBridge5 passes with; no reason to invent
                over = f" over {c.last_opp_bid}" if c.last_opp_bid else ""
                return [f"with {c.h.hcp} HCP and {holding(c.h, c.partner_suit)} in partner's suit, {c.seat} passes{over} "
                        f"rather than raise"]
            return [f"with {c.h.hcp} HCP and {holding(c.h, c.partner_suit)} in partner's suit, {c.seat} has no reason to bid"]
        if c.cat == "opening_pass" and rule_of_20_satisfied(c.h):
            return [f"with {c.h.hcp} HCP the Rule of 20 is just met ({rule_of_20(c.h)}), a borderline opening, "
                    f"and {c.seat} chooses to pass"]
        return [_ev_pass_weak(c) if c.h.hcp < 12 else _ev_pass_no_good_call(c) or f"{c.seat} has {c.h.hcp} HCP but no convenient call"]
    t = _target_suit(c)
    if t:
        return [f"{c.seat} holds {holding(c.h, t)} and {c.h.hcp} HCP"]
    if c.bid_suit and not _is_artificial(c):
        return [f"{c.seat} has {holding(c.h, c.bid_suit)} and {c.h.hcp} HCP"]
    if c.cat == "overcall_cuebid" or (c.bid_suit and c.bid_suit in c.opp_suits) or c.gt.get("bid_is_unusual_2nt"):
        long2 = sorted("SHDC", key=lambda x: (-c.h.lengths[x], "SHDC".index(x)))[:2]
        return [f"{c.seat} has {holding(c.h, long2[0])} and {holding(c.h, long2[1])}, with {c.h.hcp} HCP"]
    if c.bid[:1] in "123" and c.bid.endswith(("NT", "N")):
        return _nt_evidence(c)
    if c.bid in ("X", "Dbl") and c.last_opp_suit:
        return [f"{c.seat} holds {holding(c.h, c.last_opp_suit)} {_where(c, c.last_opp_suit, 'in the opponents suit')} "
                f"and {c.h.hcp} HCP".replace("opponents suit", "opponents' suit")]
    if c.bid in ("X", "Dbl") and c.last_opp_bid and c.last_opp_bid.endswith(("NT", "N")) and c.partner_suit:
        return [f"{c.seat} has {c.h.hcp} HCP and {holding(c.h, c.partner_suit)} in partner's suit"]
    return [f"{c.seat} has {c.h.hcp} HCP with a {shape(c.h)} shape"]


# ---------------------------------------------------------------------------
# 3. contrast with the runner-up call
# ---------------------------------------------------------------------------

def contrast(c: Ctx) -> Optional[str]:
    """Why this call and not the runner-up -- only when the comparison can
    be stated with the hand's own lengths/points (never a bare "undersells")."""
    ru = (c.f.get("runner_up") or {})
    rb, p = ru.get("bid"), ru.get("p", 0)
    if not rb or rb == c.bid or p < 0.05:
        return None
    b, h = c.bid, c.h
    preemptive = any(c.gt.get(x) for x in _PREEMPTIVE)
    if rb in ("P", "Pass") and b not in NB and b[:1] in "123" and b not in ("3NT", "3N"):
        if c.bid_suit and h.lengths[c.bid_suit] >= 6:
            # the holding is already in the evidence (not "6 clubs (K J 8 7 5 2) is worth showing" again)
            return (f"Passing was the alternative, but a {_WORDS.get(h.lengths[c.bid_suit], h.lengths[c.bid_suit])}"
                    f"-card {SUIT_NAME[c.bid_suit][:-1]} suit is worth showing.")
        if h.hcp >= 10 and not preemptive:
            return f"Passing was the alternative, but {h.hcp} HCP are worth a bid."
        return None
    if b in ("P", "Pass") and rb not in NB:
        if h.hcp < 10:
            return f"{rb} was the alternative, but {h.hcp} HCP is thin for it."
        return None
    rs = _suit_of(rb)
    if c.bid_suit and rs and rs != c.bid_suit and rb[:1] == b[:1] and not _is_artificial(c):
        lb, lr = h.lengths[c.bid_suit], h.lengths[rs]
        if lb > lr >= 3 and not c.conv:
            return f"{SUIT_NAME[c.bid_suit].capitalize()} rather than {SUIT_NAME[rs]}: {lb} cards against {lr}."
        return None
    if (rb[:1].isdigit() and b[:1].isdigit() and not preemptive and c.cat != "slam_bid"
            and not _is_artificial(c)
            and (rs == c.bid_suit or (rs is None and _suit_of(b) is None))):
        # one level apart only (not "6S would promise more than the 9 HCP here" for 4S)
        if int(rb[0]) == int(b[0]) + 1 and h.hcp <= 12:
            return f"{rb} would promise more than the {h.hcp} HCP here."
        if int(rb[0]) == int(b[0]) - 1 and h.hcp >= 13:
            return f"{rb} would undersell {h.hcp} HCP."
    return None


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------

def situation(frame: Dict[str, Any]) -> Dict[str, Any]:
    """The situation analysis the explainer's PROMPT carries
    (nlg.prompt.build_llm_prompt): the meaning of the call and the
    facts that define the situation (_forced), i.e. everything the teacher's
    explanation states beyond the hand facts, the reasons and the runner-up, so the explainer is
    told the convention (Stayman, not just "an artificial (conventional) bid") instead of guessing."""
    c = Ctx(frame, random.Random(f"{frame.get('deal_id')}:{frame.get('step')}"))
    facts = [_cap(cl) for _, cl in _forced(c)]
    ctx = _pass_game_context(c)
    if ctx and ctx in with_context(c, evidence(c) or _fallback_evidence(c)):  # only when the text states it
        facts.append(_cap(ctx))
    return {"meaning": meaning_phrase(c), "facts": facts}


def explain(frame: Dict[str, Any], rng: Optional[random.Random] = None) -> str:
    rng = rng or random.Random(f"{frame.get('deal_id')}:{frame.get('step')}")
    c = Ctx(frame, rng)
    sentences = [headline(c)]
    clauses = with_context(c, evidence(c) or _fallback_evidence(c))
    if len(clauses) <= 2:
        body = c.pick(", and ", "; ").join(clauses)
    else:  # "A, B, and C" rather than "A, and B, and C"
        body = ", ".join(clauses[:-1]) + ", and " + clauses[-1]
    sentences.append(_cap(body) + ".")
    con = contrast(c)
    if con:
        sentences.append(con)
    notes = deviation_notes(frame)
    if notes:
        sentences.append("This is unusual: " + " ".join(notes))
    return _dedupe_holdings(" ".join(sentences))


def _dedupe_holdings(text: str) -> str:
    """A holding stated twice ("The long suit, 6 spades (K J 8 7 6 4), is the
    opponents' suit ... playing for penalties with 6 spades (K J 8 7 6 4) in
    their suit"): the second mention drops the cards."""
    seen = set()

    def short(m):
        full = m.group(0)
        if full not in seen:
            seen.add(full)
            return full
        return "the " + full.split(" (")[0] if full[0].isdigit() else full.split(" (")[0]
    return _HOLDING_RE.sub(short, text)
