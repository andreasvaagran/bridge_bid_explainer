"""Predicates that need only the auction (`AuctionContext`): who opened, responses, overcalls and
reverses, which conventional sequence is in progress, and how long a bid promises its suit to be."""
from __future__ import annotations
from typing import Optional

from .vocabulary import SUITS, NON_BIDS
from .types import AuctionContext, seat_of_bid


# ============================================================
# Numeric helpers
# ============================================================

def bid_level_safe(b: Optional[str]) -> Optional[int]:
    if not b or len(b) < 2 or not b[0].isdigit():
        return None
    return int(b[0])


def suit_rank(suit: str) -> int:
    """Bidding rank: C=0, D=1, H=2, S=3."""
    return {"C": 0, "D": 1, "H": 2, "S": 3}.get(suit, -1)


# ============================================================
# Partnership / Opposition helpers
# ============================================================

def partnership_has_bid(a: AuctionContext) -> bool:
    s = a.actor_seat()
    p = (s + 2) % 4
    for i, b in enumerate(a.bids):
        if b not in NON_BIDS and seat_of_bid(a, i) in (s, p):
            return True
    return False


def opponents_have_bid(a: AuctionContext) -> bool:
    s = a.actor_seat()
    p = (s + 2) % 4
    for i, b in enumerate(a.bids):
        if b not in NON_BIDS and seat_of_bid(a, i) not in (s, p):
            return True
    return False


def call_is_artificial(a: AuctionContext, idx: int) -> bool:
    """The real bid a.bids[idx] is a convention, not a natural suit: the
    strong 2C opening, Stayman / Jacoby transfers over the bidder's
    partner's notrump, the 2D waiting response to partner's 2C, a Blackwood
    reply to partner's 4NT, or a cuebid of the bidder's opponents' suit."""
    pb = a.bids[idx]
    if pb in NON_BIDS or len(pb) < 2:
        return False
    bidder = seat_of_bid(a, idx)
    their_partner = (bidder + 2) % 4
    before = [(i, b) for i, b in enumerate(a.bids[:idx]) if b not in NON_BIDS]
    if not before and pb == "2C":
        return True  # strong 2C opening
    partner_before = [b for i, b in before if seat_of_bid(a, i) == their_partner]
    if partner_before:
        pm = partner_before[-1]
        if pm in ("1N", "1NT") and pb in ("2C", "2D", "2H"):
            return True
        if pm in ("2N", "2NT") and pb in ("3C", "3D", "3H"):
            return True
        if pm == "2C" and pb == "2D":
            return True
        if pm in ("4N", "4NT") and pb[0] == "5":
            return True
    opp_suits = {b[-1] for i, b in before if seat_of_bid(a, i) not in (bidder, their_partner) and b[-1] in SUITS}
    return pb[-1] in opp_suits


def natural_suits_of(a: AuctionContext, seats) -> set:
    """Suits the given seats have bid naturally (call_is_artificial excluded)."""
    return {b[-1] for i, b in enumerate(a.bids)
            if b not in NON_BIDS and len(b) >= 2 and b[-1] in SUITS and seat_of_bid(a, i) in seats
            and not call_is_artificial(a, i)}


def opponent_natural_suits(a: AuctionContext) -> set:
    s = a.actor_seat()
    return natural_suits_of(a, ((s + 1) % 4, (s + 3) % 4))


def partner_bid_is_artificial(a: AuctionContext) -> bool:
    """Partner's last call is a convention, not a natural suit (see
    call_is_artificial). Rules that treat partner's call as a suit to raise
    or support must exclude these (4C over partner's strong 2C is not a club raise)."""
    pb = a.partner_last_bid()
    if pb is None or pb in NON_BIDS or len(pb) < 2:
        return False
    p = a.partner
    pidx = max(i for i, b in enumerate(a.bids) if b not in NON_BIDS and seat_of_bid(a, i) == p and b == pb)
    return call_is_artificial(a, pidx)


def partner_has_suit(a: AuctionContext) -> bool:
    b = a.partner_last_bid()
    return b is not None and len(b) >= 2 and b[-1] in SUITS and not partner_bid_is_artificial(a)


def uncontested(a: AuctionContext) -> bool:
    return not opponents_have_bid(a)


# ============================================================
# Auction-type predicates
# ============================================================

def in_opening_position(a: AuctionContext) -> bool:
    """Nobody has bid yet (the actor may still pass)."""
    return all(b in NON_BIDS for b in a.bids)


def is_opening(a: AuctionContext) -> bool:
    """The actor makes the first real bid of the auction (a pass in opening
    position is not an opening, so concepts gated on is_opening never fire
    on passes)."""
    return in_opening_position(a) and a.bid not in NON_BIDS


def _opener_seat(a: AuctionContext) -> Optional[int]:
    for i, b in enumerate(a.bids):
        if b not in NON_BIDS:
            return seat_of_bid(a, i)
    return None


def _actor_has_bid(a: AuctionContext) -> bool:
    s = a.actor_seat()
    return any(b not in NON_BIDS and seat_of_bid(a, i) == s for i, b in enumerate(a.bids))


def is_response(a: AuctionContext) -> bool:
    """The actor's first real action after PARTNER opened the auction (the
    SAYC sense of "response"). Advancing partner's overcall is is_advance(),
    not a response."""
    return _opener_seat(a) == a.partner and not _actor_has_bid(a)


def is_advance(a: AuctionContext) -> bool:
    """The actor's first real action after the OPPONENTS opened and partner
    made a real bid (overcall / notrump overcall); partner's takeout double
    is not a real bid here, matching partnership_has_bid."""
    opener = _opener_seat(a)
    return (opener is not None and opener not in (a.actor_seat(), a.partner)
            and partnership_has_bid(a) and not _actor_has_bid(a))


def i_opened_auction(a: AuctionContext) -> bool:
    """The actor made the first real bid of the whole auction (a true
    opener's rebid), unlike is_partnership_opener, which is also true for
    an overcaller who made the partnership's first bid."""
    return _opener_seat(a) == a.actor_seat()


def opening_seat_position(a: AuctionContext) -> Optional[int]:
    """1-4: how many calls, including the opening bid itself, the opening
    took (1 = first seat). Only meaningful when is_opening(a)."""
    return len(a.bids) + 1 if in_opening_position(a) else None


def is_preemptive_opening(a: AuctionContext) -> bool:
    return (
        is_opening(a)
        and len(a.bid) >= 2
        and a.bid[0] in "234567"
        and a.bid[-1] in SUITS
        and a.bid != "2C"
    )


def is_preemptive_opening_continuation(a: AuctionContext) -> bool:
    """A later bid by a player whose OPENING bid was a preempt (2-over-1
    responses and overcalls at the 2+ level do not count)."""
    if in_opening_position(a) or not i_opened_auction(a):
        return False
    first = a.my_first_bid()
    if first is None or first in NON_BIDS:
        return False
    lvl = bid_level_safe(first)
    if lvl is None:
        return False
    return lvl >= 2 and len(first) >= 2 and first[-1] in SUITS and first != "2C"


def is_partnership_opener(a: AuctionContext) -> bool:
    actor = a.actor_seat()
    partner = (actor + 2) % 4
    for i, b in enumerate(a.bids):
        if b not in NON_BIDS and seat_of_bid(a, i) in (actor, partner):
            return seat_of_bid(a, i) == actor
    return False


def is_overcall(a: AuctionContext) -> bool:
    opener_seat = None
    for i, b in enumerate(a.bids):
        if b not in NON_BIDS:
            opener_seat = seat_of_bid(a, i)
            break
    if opener_seat is None:
        return False
    s = a.actor_seat()
    p = (s + 2) % 4
    return (
        opener_seat not in (s, p)
        and not any(
            seat_of_bid(a, i) in (s, p) and b not in NON_BIDS
            for i, b in enumerate(a.bids)
        )
        # answering partner's takeout double is advancing, not overcalling
        and not any(seat_of_bid(a, i) == p and b in ("X", "Dbl") for i, b in enumerate(a.bids))
        and a.bid not in NON_BIDS
        and a.bid not in ("Dbl", "X")
    )


def is_jump_overcall(a: AuctionContext) -> bool:
    if not is_overcall(a):
        return False
    bid_level = bid_level_safe(a.bid)
    if bid_level is None:
        return False
    bid_suit_rk = suit_rank(a.bid[-1]) if len(a.bid) >= 2 and a.bid[-1] in SUITS else -1
    for i, b in enumerate(a.bids):
        if b not in NON_BIDS:
            opp_level = bid_level_safe(b) or 0
            if b.endswith(("N", "NT")):
                min_level = opp_level + 1
            else:
                opp_suit_rk = suit_rank(b[-1]) if len(b) >= 2 and b[-1] in SUITS else -1
                min_level = opp_level if bid_suit_rk > opp_suit_rk else opp_level + 1
            return bid_level > min_level
    return False


# ============================================================
# Conventional / structural sequence detectors
# ============================================================

def is_response_to_weak_two(a: AuctionContext) -> bool:
    if not is_response(a):
        return False
    pb = a.partner_last_bid()
    return pb is not None and bid_level_safe(pb) == 2 and pb[-1] in ("D", "H", "S")


def is_stayman_sequence(a: AuctionContext) -> bool:
    pl = a.partner_last_bid()
    first_bid = a.my_first_bid()
    if pl == "2C":
        return first_bid in ("1N", "1NT")
    if pl == "3C":
        return first_bid in ("2N", "2NT")
    return False


def is_jacoby_transfer_sequence(a: AuctionContext) -> bool:
    pl = a.partner_last_bid()
    if pl in ("2D", "2H"):
        first_bid = a.my_first_bid()
        if first_bid not in ("1N", "1NT"):
            return False
        if not is_partnership_opener(a):
            return False
        return (pl == "2D" and a.bid == "2H") or (pl == "2H" and a.bid == "2S")
    if pl in ("3D", "3H"):
        first_bid = a.my_first_bid()
        if first_bid not in ("2N", "2NT"):
            return False
        return (pl == "3D" and a.bid == "3H") or (pl == "3H" and a.bid == "3S")
    return False


def is_minor_suit_transfer_completion(a: AuctionContext) -> bool:
    if a.partner_last_bid() != "2S":
        return False
    if a.bid != "3C":
        return False
    first_bid = a.my_first_bid()
    return first_bid in ("1N", "1NT")


def is_strong_2c_waiting_sequence(a: AuctionContext) -> bool:
    if a.partner_last_bid() != "2D":
        return False
    first_bid = a.my_first_bid()
    return first_bid == "2C"


def is_new_minor_forcing_sequence(a: AuctionContext) -> bool:
    if a.partner_last_bid() not in ("1N", "1NT"):
        return False
    if a.bid not in ("2C", "2D"):
        return False
    our_first = a.my_first_bid()
    if our_first is None or our_first[0] != "1" or our_first[-1] not in ("H", "S"):
        return False
    partner_first = a.partner_first_bid()
    if partner_first is None or partner_first[-1] not in ("C", "D"):
        return False
    return a.bid[-1] != partner_first[-1]


def is_jump_rebid_same_suit(a: AuctionContext) -> bool:
    prev = a.my_previous_bid()
    if prev is None:
        return False
    lvl = bid_level_safe(a.bid)
    prev_lvl = bid_level_safe(prev)
    return (
        lvl is not None
        and prev_lvl is not None
        and len(a.bid) >= 2 and len(prev) >= 2
        and a.bid[-1] in SUITS and a.bid[-1] == prev[-1]  # a suit, not 1NT-3NT
        and lvl > prev_lvl + 1
    )


def is_jump_raise_after_nt(a: AuctionContext) -> bool:
    """A direct jump to the 3-level or higher in a suit, made as the
    immediate reply to partner's 1NT -- e.g. 1NT-3H, a natural
    slam-invitational (or game-forcing) raise typically showing a
    6-card suit. Responder to a 1NT opening has no earlier bid of their
    own, so the jump is measured from the level, not from a previous bid.
    """
    if a.partner_last_bid() not in ("1N", "1NT"):
        return False
    if len(a.bid) < 2 or a.bid[-1] not in SUITS:
        return False
    lvl = bid_level_safe(a.bid)
    return lvl is not None and lvl >= 3


def is_reverse_bid(a: AuctionContext) -> bool:
    """Opener rebids a NEW suit, ranked higher than the first suit,
    without a jump, at the 2-level or higher — forces responder to the
    3-level to return to opener's first suit, so promises extra strength.
    """
    if is_opening(a) or is_response(a):
        return False
    prev = a.my_previous_bid()
    if prev is None or len(prev) < 2:
        return False
    if a.bid[0] != "2" or a.bid[-1] not in SUITS or len(a.bid) < 2:
        return False
    if prev[0] != "1" or prev[-1] not in SUITS:
        return False
    if prev[-1] == a.bid[-1]:
        return False
    plb = a.partner_last_bid()
    if plb and len(plb) >= 2 and a.bid[-1] == plb[-1]:
        return False
    return suit_rank(a.bid[-1]) > suit_rank(prev[-1])


def promised_length(a: AuctionContext) -> Optional[int]:
    """How many cards in its strain `a.bid` conventionally promises,
    purely from auction structure (level/position), independent of what
    the actual hand holds."""
    if is_preemptive_opening(a):
        # WBridge5 (the system the SL data was played in; see
        # notes/wbridge5_system.md): 3-level preempts from 6 cards, weak twos
        # from 6 cards but 5 in third seat. Verified on the data: 100% of
        # 3-level preempts have 6+, 67% of third-seat weak twos have 5.
        if bid_level_safe(a.bid) >= 4:
            return 7  # WBridge5 4M: 7+ (4C/4D openings in the data: 8)
        if bid_level_safe(a.bid) == 3:
            return 6
        if bid_level_safe(a.bid) == 2:
            return 5 if opening_seat_position(a) == 3 else 6

    if is_strong_2c_waiting_sequence(a) and a.bid[-1] in ("H", "S"):
        return 5

    if is_jump_rebid_same_suit(a):
        return 6

    if a.bid[0] == "1" and a.bid[-1] in ("H", "S"):
        if is_opening(a) or is_overcall(a):
            return 5

    if is_opening(a) and a.bid == "1C":
        return 3  # SAYC/WBridge5 1C opening: 3+ clubs
    if bid_level_safe(a.bid) == 1 and a.bid[-1] in SUITS:
        return 4

    if (
        not is_opening(a) and not is_response(a)
        and is_partnership_opener(a)
        and bid_level_safe(a.bid) == 2
        and a.bid[-1] in SUITS
        and not a.bid.endswith(("N", "NT"))
    ):
        prev = a.my_previous_bid()
        if prev is not None and len(prev) >= 2 and prev[-1] in SUITS and a.bid[-1] != prev[-1]:
            return 4

    return None
