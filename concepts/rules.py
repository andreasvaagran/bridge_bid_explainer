"""The rule engine. `build_sayc_state_rules` gives one `StateRule` (an AND of predicates) per name in
`HARD_STATE_CONCEPTS`, and `SaycStateAuthority` labels a call in its auction with all 146 concepts. The rules are
calibrated to WBridge5 in SAYC mode (notes/wbridge5_system.md). The engine is deterministic and needs no
checkpoint. It supplies every training label and grounds every explanation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List

from .vocabulary import HARD_STATE_CONCEPTS, SUITS, NON_BIDS
from .types import AuctionContext, HandFeatures, seat_of_bid
from .predicates_auction import (
    bid_level_safe, suit_rank, partner_bid_is_artificial, call_is_artificial, natural_suits_of,
    opponent_natural_suits,
    partnership_has_bid, opponents_have_bid, partner_has_suit, uncontested,
    is_opening, in_opening_position, is_response, is_advance, i_opened_auction, opening_seat_position,
    is_preemptive_opening, is_preemptive_opening_continuation,
    is_partnership_opener, is_overcall, is_jump_overcall,
    is_response_to_weak_two, is_stayman_sequence, is_jacoby_transfer_sequence,
    is_minor_suit_transfer_completion, is_strong_2c_waiting_sequence,
    is_new_minor_forcing_sequence, is_jump_rebid_same_suit, is_jump_raise_after_nt,
    is_reverse_bid, promised_length,
)
from .predicates_hand import (
    opens_longer_minor, opens_longer_major, prefers_notrump_over_major,
    has_singleton, has_void, is_semi_balanced,
    denies_major_fit_after_1nt_response,
    losing_trick_count, quick_tricks, rule_of_20_satisfied, support_points,
    preempt_suit_quality_score, wb5_points,
)


# ---------------------------------------------------------------------------
# Shared sequence helpers (used by several rules; defined once so e.g. "is
# this a Blackwood reply" means the same thing everywhere it is excluded)
# ---------------------------------------------------------------------------

def _last_real_bid(a: AuctionContext):
    real = [b for b in a.bids if b not in NON_BIDS]
    return real[-1] if real else None


def _last_real_bid_is_theirs(a: AuctionContext) -> bool:
    """The current contract was bid by an opponent (not by our side)."""
    real = [i for i, b in enumerate(a.bids) if b not in NON_BIDS]
    return bool(real) and seat_of_bid(a, real[-1]) not in (a.actor_seat(), a.partner)


def _opener_reopening_double(a: AuctionContext) -> bool:
    """I opened, partner has only passed, and I double the opponents' natural
    suit bid at the 1-3 level (not my own suit): 1C P P 1D X, 1S 2C P P X.
    In WBridge5's data this is a TAKEOUT double showing extra values: over
    300k deals, the doubler held 0-2 cards in the doubled suit 80% of the time
    at the 1-2 level and partner took it out 81% (1-level) / 70% (2-level)
    of the time."""
    if not i_opened_auction(a) or a.bid not in ("X", "Dbl"):
        return False
    s, p = a.actor_seat(), a.partner
    real = [(i, b) for i, b in enumerate(a.bids) if b not in NON_BIDS]
    if any(seat_of_bid(a, i) == p for i, _ in real):
        return False
    mine = [b for i, b in real if seat_of_bid(a, i) == s]
    li, lb = real[-1]
    return (len(mine) == 1 and seat_of_bid(a, li) not in (s, p) and len(lb) >= 2 and lb[-1] in SUITS
            and int(lb[0]) <= 3 and lb[-1] != mine[0][-1] and not call_is_artificial(a, li))


def _is_blackwood_ask(a: AuctionContext) -> bool:
    return (a.bid in ("4N", "4NT") and partnership_has_bid(a) and partner_has_suit(a)
            and a.partner_last_bid() is not None
            and not a.partner_last_bid().endswith(("N", "NT")) and not opponents_have_bid(a))


def _is_blackwood_reply(a: AuctionContext) -> bool:
    """A 5-level reply to partner's 4NT when that 4NT was Blackwood, i.e. our
    side had bid a suit before it. Partner's 4NT with no earlier suit bid by
    our side (e.g. over the opponents' 1S-2C) is the Unusual 4NT for the
    minors, and 5D there is a natural choice of minor."""
    if a.partner_last_bid() not in ("4N", "4NT") or a.bid not in ("5C", "5D", "5H", "5S"):
        return False
    idx = max(i for i, b in enumerate(a.bids) if b in ("4N", "4NT") and seat_of_bid(a, i) == a.partner)
    return any(b not in NON_BIDS and len(b) >= 2 and b[-1] in SUITS
               and seat_of_bid(a, i) in (a.actor_seat(), a.partner) for i, b in enumerate(a.bids[:idx]))


def _is_jump_over_last(a: AuctionContext) -> bool:
    """The bid skips at least one level: it is higher than the cheapest legal
    bid in its strain over the last real bid at the table."""
    last = _last_real_bid(a)
    if last is None or len(a.bid) < 2 or not a.bid[0].isdigit():
        return False
    order = "CDHSN"
    strain = "N" if a.bid.endswith(("N", "NT")) else a.bid[-1]
    last_strain = "N" if last.endswith(("N", "NT")) else last[-1]
    cheapest = int(last[0]) + (0 if order.index(strain) > order.index(last_strain) else 1)
    return int(a.bid[0]) > cheapest


def _same_suit(x, y) -> bool:
    """Both calls are SUIT bids in the same suit. Comparing x[-1] == y[-1]
    alone would also match two notrump bids ('T' == 'T') and make 1NT-3NT
    a raise."""
    return (bool(x) and bool(y) and len(x) >= 2 and len(y) >= 2 and x[0].isdigit() and y[0].isdigit()
            and x[-1] in SUITS and y[-1] in SUITS and x[-1] == y[-1])


def _same_suit_as_partner(a: AuctionContext) -> bool:
    """The call is in the suit of partner's last call, and that call was a
    natural suit (not 2C strong, Stayman, a transfer, a Blackwood reply, a
    cuebid...)."""
    return _same_suit(a.bid, a.partner_last_bid()) and not partner_bid_is_artificial(a)


def _suit_len(a: AuctionContext, h: HandFeatures) -> int:
    return h.lengths.get(a.bid[-1], 0) if len(a.bid) >= 2 and a.bid[-1] in SUITS else 0


def _shown_suit(a: AuctionContext, i: int):
    """The suit the real call a.bids[i] shows: its own suit if natural, the
    target major of a Jacoby transfer, None for any other artificial call
    (2C strong / waiting 2D / Stayman / Blackwood replies / cuebids)."""
    b = a.bids[i]
    if b in NON_BIDS or len(b) < 2 or b[-1] not in SUITS:
        return None
    if not call_is_artificial(a, i):
        return b[-1]
    before = [x for j, x in enumerate(a.bids[:i]) if x not in NON_BIDS and seat_of_bid(a, j) == (seat_of_bid(a, i) + 2) % 4]
    if before and before[-1] in ("1N", "1NT", "2N", "2NT") and b in ("2D", "2H", "3D", "3H"):
        return "H" if b[-1] == "D" else "S"
    return None


def _agreed_trump(a: AuctionContext):
    """A suit both partners have SHOWN (the partnership's agreed trump suit),
    or None. Artificial calls do not count (2C-2D-3D does not agree
    diamonds); a transfer shows its major."""
    mine, theirs = set(), set()
    for i, b in enumerate(a.bids):
        suit = _shown_suit(a, i)
        if suit is None:
            continue
        if seat_of_bid(a, i) == a.actor_seat():
            mine.add(suit)
        elif seat_of_bid(a, i) == a.partner:
            theirs.add(suit)
    both = mine & theirs
    real = [(i, b) for i, b in enumerate(a.bids) if b not in NON_BIDS]
    for (i1, b1), (i2, b2) in zip(real, real[1:]):
        # partner's Jacoby 2NT over my 1H/1S opening agrees my major
        if (seat_of_bid(a, i1) == a.actor_seat() and b1 in ("1H", "1S") and i1 == real[0][0]
                and seat_of_bid(a, i2) == a.partner and b2 in ("2N", "2NT")):
            both = both | {b1[-1]}
    if not both:
        return None
    # the most recently shown of the shared suits
    for i in reversed([i for i, b in enumerate(a.bids) if b not in NON_BIDS and seat_of_bid(a, i) in (a.actor_seat(), a.partner)]):
        suit = _shown_suit(a, i) or (a.bids[i][-1] if a.bids[i] in ("2N", "2NT") else None)
        if suit in both:
            return suit
    return next(iter(both))


def _control_bid(a: AuctionContext) -> bool:
    """A new-suit bid at the 3+ level, after partner's 3+ level bid, once the
    partnership has AGREED a trump suit (both have bid it) -- the shape of a
    slam-try control cuebid. Without the agreed-suit condition every natural
    new suit in a competitive auction would count. Structural only;
    bid_shows_*_control add the hand."""
    plb = a.partner_last_bid()
    trump = _agreed_trump(a)
    return (trump is not None and len(a.bid) >= 2 and a.bid[-1] in SUITS and a.bid[-1] != trump
            and partnership_has_bid(a) and not is_overcall(a) and a.my_previous_bid() is not None
            and plb is not None and partner_has_suit(a)
            and len(a.bid) >= 2 and a.bid[-1] in SUITS
            and (bid_level_safe(a.bid) or 0) >= 3 and (bid_level_safe(plb) or 0) >= 3
            and a.bid[-1] != plb[-1]
            and not is_jacoby_transfer_sequence(a) and not is_stayman_sequence(a)
            and not _is_blackwood_reply(a)
            and (a.my_first_bid() is None or len(a.my_first_bid()) < 2 or a.bid[-1] != a.my_first_bid()[-1]))


@dataclass(frozen=True)
class StateRule:
    name: str
    holds_if: List[Callable[[AuctionContext, HandFeatures], bool]]


class SaycStateAuthority:
    def __init__(self, rules: List[StateRule]):
        self.rules = rules

    def classify(self, auction: AuctionContext, hand: HandFeatures) -> Dict[str, bool]:
        result = {r.name: all(fn(auction, hand) for fn in r.holds_if) for r in self.rules}
        # bid_is_splinter is defined as the union of its two sub-types (see
        # the bid_is_splinter StateRule's comment) -- computed here instead
        # of duplicating the shortness-check skeleton a third time.
        result["bid_is_splinter"] = result["bid_is_void_splinter"] or result["bid_is_singleton_splinter"]
        return result


def create_sayc_state_authority() -> SaycStateAuthority:
    return SaycStateAuthority(build_sayc_state_rules())


def build_sayc_state_rules() -> List[StateRule]:
    R: List[StateRule] = []

    # =================================================================
    # 1.  OPENING / RESPONSE / BASIC STRUCTURAL BIDS
    # =================================================================
    R += [
        StateRule("bid_is_1_level_opening", [
            lambda a, h: is_opening(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[0] == "1",
        ]),
        StateRule("bid_is_2_level_opening", [
            lambda a, h: is_opening(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[0] == "2",
        ]),
        StateRule("bid_is_3_level_opening", [
            lambda a, h: is_opening(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[0] == "3",
        ]),
        StateRule("bid_is_4plus_level_opening", [
            lambda a, h: is_opening(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[0] in "4567",
        ]),
        StateRule("bid_is_1nt_opening", [
            lambda a, h: is_opening(a),
            lambda a, h: a.bid in ("1N", "1NT"),
            lambda a, h: h.balanced or is_semi_balanced(h),
            lambda a, h: 15 <= h.hcp <= 17,
        ]),
        StateRule("bid_is_2nt_opening", [
            lambda a, h: is_opening(a),
            lambda a, h: a.bid in ("2N", "2NT"),
            lambda a, h: h.balanced or is_semi_balanced(h),
            # SAYC's 20-21 HCP; 63 of the 68 2NT openings in the SL data are
            # 20-21, with 19 and 22 as rare judgment calls.
            lambda a, h: 20 <= h.hcp <= 21,
        ]),
        StateRule("bid_is_strong_2cc_opening", [
            lambda a, h: is_opening(a),
            lambda a, h: a.bid == "2C",
            # WBridge5's 2C (notes/wbridge5_system.md): 21+ points or a
            # 4-loser hand. SAYC's "22+ HCP" matched only 30% of the 2C
            # openings in the SL data; this matches 99.6%.
            lambda a, h: h.hcp >= 22 or wb5_points(h) >= 21 or losing_trick_count(h) <= 4,
        ]),
        StateRule("bid_is_1_level_response", [
            lambda a, h: is_response(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[0] == "1",
            lambda a, h: uncontested(a),
        ]),
        StateRule("bid_is_2_level_response", [
            lambda a, h: is_response(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[0] == "2",
            lambda a, h: uncontested(a),
        ]),
        StateRule("bid_is_raise_response", [
            lambda a, h: is_response(a),
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.bid) >= 2 and len(a.partner_last_bid()) >= 2,
            lambda a, h: a.bid[-1] in SUITS and _same_suit_as_partner(a),
            lambda a, h: not is_jacoby_transfer_sequence(a),
        ]),
        StateRule("bid_is_game_bid", [
            # 3NT, 4H/4S, 5C/5D -- the game-level contracts (not 4NT asks or
            # 5C/5D Blackwood replies).
            lambda a, h: a.bid in ("3N", "3NT", "4H", "4S", "5C", "5D"),
            lambda a, h: not _is_blackwood_reply(a),
        ]),
        StateRule("bid_is_opener_rebid", [
            lambda a, h: not is_opening(a),
            lambda a, h: not is_response(a),
            lambda a, h: partnership_has_bid(a),
            lambda a, h: a.my_previous_bid() is not None,
            lambda a, h: len(a.bid) >= 2,
            lambda a, h: a.bid[-1] in SUITS,
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_stayman_sequence(a),
            lambda a, h: i_opened_auction(a),  # not is_partnership_opener, which includes overcallers
            lambda a, h: (
                a.my_first_bid() is not None
                and len(a.my_first_bid()) >= 2
                and _same_suit(a.bid, a.my_first_bid())
            ),
        ]),
        # A reverse promises extra strength by forcing partner to the 3-level;
        # its own concept gives a "why this rebid" answer.
        StateRule("bid_is_reverse", [lambda a, h: is_reverse_bid(a)]),
        StateRule("bid_is_overcall", [
            lambda a, h: is_overcall(a),
        ]),
        StateRule("bid_is_double", [
            lambda a, h: a.bid in ("Dbl", "X"),
        ]),
        StateRule("bid_is_pass", [
            lambda a, h: a.bid in ("Pass", "P"),
        ]),
        StateRule("bid_is_cuebid", [
            lambda a, h: a.bid not in NON_BIDS,
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
            # up to the 4-level: advancer's / responder's cuebids are often
            # 3- or 4-level (1H 1S 2H 3H, 1C 1S P 4C)
            lambda a, h: a.bid[0] in "1234",
            lambda a, h: a.bid[-1] in opponent_natural_suits(a),
            # a suit our side has not bid naturally (a cuebid that supports
            # partner, the commonest kind, still counts)
            lambda a, h: a.bid[-1] not in natural_suits_of(a, (a.actor_seat(), a.partner)),
            lambda a, h: not is_overcall(a),
        ]),
    ]

    # =================================================================
    # 2.  LENGTH / EXACT-LENGTH
    # =================================================================
    R += [
        # Length family: the bid is a natural length-showing bid (its
        # promised_length band) AND the hand really holds that length, so an
        # explanation never claims a length the hand does not have (1C on 3
        # clubs, 5-card third-seat weak twos).
        StateRule("bid_shows_4_plus_cards", [
            lambda a, h: promised_length(a) == 4,
            lambda a, h: _suit_len(a, h) >= 4,
        ]),
        StateRule("bid_shows_5_plus_cards", [
            lambda a, h: promised_length(a) == 5,
            lambda a, h: _suit_len(a, h) >= 5,
        ]),
        StateRule("bid_shows_6_plus_cards", [
            lambda a, h: promised_length(a) in (5, 6, 7) and not (
                promised_length(a) == 5 and not is_preemptive_opening(a)),
            lambda a, h: _suit_len(a, h) >= 6,
        ]),
        StateRule("bid_shows_7_plus_cards", [
            lambda a, h: _suit_len(a, h) >= 7,
            lambda a, h: (
                (is_preemptive_opening(a) and bid_level_safe(a.bid) is not None and bid_level_safe(a.bid) >= 3)
                or (
                    is_overcall(a)
                    and bid_level_safe(a.bid) is not None
                    and bid_level_safe(a.bid) >= 3
                    and len(a.bid) >= 2 and a.bid[-1] in SUITS
                    and h.lengths.get(a.bid[-1], 0) >= 7
                    and h.hcp <= 12
                )
            ),
        ]),
        StateRule("bid_shows_exactly_5_cards", [
            lambda a, h: promised_length(a) == 5,
            lambda a, h: not is_preemptive_opening(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
            lambda a, h: h.lengths.get(a.bid[-1], 0) == 5,
        ]),
        StateRule("bid_shows_exactly_6_cards", [
            lambda a, h: promised_length(a) == 6,
            lambda a, h: not is_preemptive_opening(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
            lambda a, h: h.lengths.get(a.bid[-1], 0) == 6,
        ]),
        StateRule("bid_shows_extra_length_on_rebid", [
            lambda a, h: is_jump_rebid_same_suit(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
            lambda a, h: h.lengths.get(a.bid[-1], 0) >= 6,
        ]),
        StateRule("bid_shows_extra_length_on_jump_raise_after_nt", [
            lambda a, h: is_jump_raise_after_nt(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
            lambda a, h: h.lengths.get(a.bid[-1], 0) >= 6,
        ]),
        StateRule("bid_is_weak_jump_overcall", [
            lambda a, h: is_jump_overcall(a),
            lambda a, h: h.hcp <= 10,
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
            lambda a, h: h.lengths.get(a.bid[-1], 0) >= 6,
        ]),
        StateRule("bid_shows_5_card_major_after_strong_2c", [
            lambda a, h: is_strong_2c_waiting_sequence(a),
            lambda a, h: a.bid in ("2H", "2S"),
            lambda a, h: h.lengths.get(a.bid[-1], 0) >= 5,
        ]),
    ]

    # =================================================================
    # 3.  SUIT / NT / SHAPE
    # =================================================================
    R += [
        StateRule("bid_shows_major_suit", [
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in ("H", "S"),
            lambda a, h: not a.bid.endswith(("N", "NT")),
            lambda a, h: not (is_opening(a) and a.bid == "2C"),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_strong_2c_waiting_sequence(a),
            lambda a, h: not (
                is_response(a)
                and a.bid in ("2D", "2H")
                and a.partner_last_bid() in ("1N", "1NT")
            ),
            lambda a, h: not (
                is_response(a)
                and a.bid == "2S"
                and a.partner_last_bid() in ("1N", "1NT")
            ),
        ]),
        StateRule("bid_shows_minor_suit", [
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in ("C", "D"),
            lambda a, h: not (is_opening(a) and a.bid == "2C"),
            lambda a, h: not (
                a.bid == "2C" and is_response(a)
                and a.partner_last_bid() in ("1N", "1NT")
            ),
            lambda a, h: not (a.bid == "2D" and a.partner_last_bid() == "2C"),
            lambda a, h: not (
                a.bid == "2D" and is_response(a)
                and a.partner_last_bid() in ("1N", "1NT")
            ),
            lambda a, h: not is_new_minor_forcing_sequence(a),
            lambda a, h: not is_minor_suit_transfer_completion(a),
        ]),
        StateRule("bid_shows_notrump", [
            lambda a, h: a.bid.endswith(("N", "NT")),
            # 4NT Blackwood / 5NT king ask are artificial, not notrump bids
            lambda a, h: not _is_blackwood_ask(a) and not (a.bid in ("5N", "5NT") and partner_has_suit(a)),
            lambda a, h: h.balanced or is_semi_balanced(h),
        ]),
        # Suit choice among alternatives; see predicates_hand.py.
        StateRule("bid_shows_longer_minor", [opens_longer_minor]),
        StateRule("bid_shows_longer_major", [opens_longer_major]),
        StateRule("bid_prefers_notrump_over_major", [prefers_notrump_over_major]),
        StateRule("bid_limits_hand", [
            lambda a, h: (
                (a.bid.endswith(("N", "NT")) and not is_opening(a))
                or (
                    is_response(a)
                    and bid_level_safe(a.bid) == 2
                    and a.partner_last_bid() is not None
                    and len(a.partner_last_bid()) >= 2
                    and _same_suit_as_partner(a)
                )
            ),
            lambda a, h: not is_stayman_sequence(a),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not (a.bid in ("2N", "2NT") and is_response_to_weak_two(a)),
            # Blackwood / king-ask are artificial asks, not limit bids
            lambda a, h: not _is_blackwood_ask(a) and not (a.bid in ("5N", "5NT") and partner_has_suit(a)),
        ]),
    ]

    # =================================================================
    # 4.  STRENGTH SEMANTICS
    # =================================================================
    R += [
        StateRule("bid_shows_weak_values", [
            lambda a, h: (
                (is_opening(a) and a.bid[0] in "234" and a.bid[-1] in SUITS and a.bid != "2C")
                or (is_response(a) and (
                    (a.bid in ("1N", "1NT") and h.hcp <= 10)
                    or (
                        a.partner_last_bid() is not None
                        and len(a.partner_last_bid()) >= 2
                        and a.bid[-1] in SUITS
                        and _same_suit_as_partner(a)
                        and a.partner_last_bid()[0] == "1"
                        and a.bid[0] == "2"
                    )
                ))
            ),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_stayman_sequence(a),
        ]),
        StateRule("bid_shows_minimum_values", [
            lambda a, h: (
                not is_opening(a) and not is_response(a)
                and partnership_has_bid(a)
                and len(a.bid) >= 2
                and a.bid[0] in "12"
                and not a.bid.startswith("2N")
            ),
            lambda a, h: h.hcp <= 14,
            lambda a, h: not is_stayman_sequence(a),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_strong_2c_waiting_sequence(a),
            lambda a, h: not is_reverse_bid(a),
        ]),
        StateRule("bid_shows_invitational_values", [
            lambda a, h: (
                (
                    is_response(a)
                    and uncontested(a)  # over an overcall the jump raise is preemptive
                    and a.partner_last_bid() is not None
                    and len(a.bid) >= 2 and len(a.partner_last_bid()) >= 2
                    and a.partner_last_bid()[0] == "1"
                    and a.bid[0] == "3"
                    and a.bid[-1] in SUITS
                    and _same_suit_as_partner(a)
                )
                or (
                    is_response(a)
                    and a.bid in ("2N", "2NT")
                    and a.partner_last_bid() is not None
                    and bid_level_safe(a.partner_last_bid()) == 1
                    and a.partner_last_bid() not in ("1H", "1S")
                )
            ),
            # hand-grounded: invitational strength (HCP, or support points
            # for a limit raise)
            lambda a, h: 10 <= h.hcp <= 12 or (
                a.bid[-1] in SUITS and 10 <= support_points(h, a.bid[-1]) <= 12),
        ]),
        StateRule("bid_shows_game_forcing_values", [
            # Only bids that really create a game force, and only with the
            # strength for it. Not 2NT openings (not forcing), 2-over-1
            # responses (SAYC: 10+, forcing one round -- WBridge5 data: 9-17
            # HCP, median 12) or responses at the 4+ level (mostly preemptive).
            lambda a, h: (
                (is_opening(a) and a.bid == "2C")
                or (  # Jacoby 2NT: 4+ support for partner's major, game force
                    is_response(a) and a.bid in ("2N", "2NT")
                    and a.partner_last_bid() in ("1H", "1S")
                    and h.lengths.get(a.partner_last_bid()[-1], 0) >= 4
                )
                or (  # jump shift: a jump in a new suit by responder
                    is_response(a) and len(a.bid) >= 2 and a.bid[-1] in SUITS
                    and a.partner_last_bid() is not None and len(a.partner_last_bid()) >= 2
                    and a.bid[-1] != a.partner_last_bid()[-1]
                    and _is_jump_over_last(a)  # over the last bid, so 1D-2S is a jump
                    and (bid_level_safe(a.bid) or 0) <= 3
                    and uncontested(a)
                )
            ),
            lambda a, h: (a.bid == "2C" and is_opening(a)) or h.hcp >= 12,
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_stayman_sequence(a),
            lambda a, h: not is_new_minor_forcing_sequence(a),
        ]),
        StateRule("bid_shows_extra_values", [
            # hand-grounded: above the minimum for the actor's role (opener
            # 15+, anyone else 13+)
            lambda a, h: h.hcp >= (15 if (is_opening(a) or i_opened_auction(a)) else 13),
            lambda a, h: not _is_blackwood_reply(a),
            lambda a, h: (
                (is_opening(a) and a.bid in ("2C", "2N", "2NT", "3N", "3NT"))
                or (
                    not is_opening(a) and not is_response(a)
                    and partnership_has_bid(a)
                    and len(a.bid) >= 2
                    and a.bid[0] in "34"
                )
                or (is_response(a) and len(a.bid) >= 2 and a.bid[0] in "34567")
                or is_reverse_bid(a)
            ),
            lambda a, h: not is_preemptive_opening(a),
            lambda a, h: not is_stayman_sequence(a),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not (
                opponents_have_bid(a)
                and not is_partnership_opener(a)
                and h.hcp <= 10
                and bid_level_safe(a.bid) is not None
                and bid_level_safe(a.bid) >= 3
            ),
            lambda a, h: not (
                is_preemptive_opening_continuation(a)
                and h.hcp <= 12
            ),
            lambda a, h: not (
                not is_partnership_opener(a)
                and a.my_previous_bid() is not None
                and len(a.my_previous_bid()) >= 2
                and a.bid[-1] in SUITS
                and _same_suit(a.bid, a.my_previous_bid())
                and h.hcp <= 12
                and bid_level_safe(a.bid) is not None
                and bid_level_safe(a.bid) >= 3
            ),
            lambda a, h: not (
                is_response(a)
                and a.bid in ("3N", "3NT")
                and a.partner_last_bid() in ("2N", "2NT")
            ),
            lambda a, h: not (
                is_partnership_opener(a)
                and opponents_have_bid(a)
                and a.my_first_bid() is not None
                and len(a.my_first_bid()) >= 2
                and a.bid[-1] in SUITS
                and _same_suit(a.bid, a.my_first_bid())
                and bid_level_safe(a.my_first_bid()) == 1
                and h.hcp <= 12
                and bid_level_safe(a.bid) is not None
                and bid_level_safe(a.bid) == 3
            ),
        ]),
    ]

    # =================================================================
    # 5.  FIT / SUPPORT
    # =================================================================
    R += [
        StateRule("bid_shows_support", [
            lambda a, h: a.partner_last_bid() is not None,
            # a raise needs cards, not a 2-card preference or a cuebid
            lambda a, h: len(a.partner_last_bid()) >= 2 and h.lengths.get(a.partner_last_bid()[-1], 0) >= 3,
            lambda a, h: len(a.bid) >= 2 and len(a.partner_last_bid()) >= 2,
            lambda a, h: a.bid[-1] in SUITS,
            lambda a, h: _same_suit_as_partner(a),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_stayman_sequence(a),
        ]),
        StateRule("bid_shows_3_card_support", [
            lambda a, h: partnership_has_bid(a),
            lambda a, h: not is_opening(a),
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.partner_last_bid()) >= 2 and len(a.bid) >= 2,
            lambda a, h: a.bid[-1] in SUITS,
            lambda a, h: _same_suit_as_partner(a),
            lambda a, h: bid_level_safe(a.bid) is not None,
            lambda a, h: bid_level_safe(a.partner_last_bid()) is not None,
            lambda a, h: bid_level_safe(a.bid) == bid_level_safe(a.partner_last_bid()) + 1,
            lambda a, h: h.lengths.get(a.bid[-1], 0) >= 3,
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_stayman_sequence(a),
        ]),
        StateRule("bid_shows_4_plus_support", [
            lambda a, h: is_response(a),
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.partner_last_bid()) >= 2,
            lambda a, h: _same_suit_as_partner(a),
            lambda a, h: (
                (a.partner_last_bid()[0] == "1" and a.bid[0] == "3")
                or (a.partner_last_bid()[0] == "1" and a.bid[0] == "4")
                or (a.partner_last_bid() in ("1H", "1S") and a.bid in ("2N", "2NT"))
            ),
            lambda a, h: h.lengths.get(a.bid[-1], 0) >= 4,
        ]),
        StateRule("bid_promises_8_card_fit", [
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.partner_last_bid()) >= 2,
            lambda a, h: a.partner_last_bid()[-1] in SUITS,
            lambda a, h: len(a.bid) >= 2 and _same_suit_as_partner(a),
            lambda a, h: h.lengths.get(a.bid[-1], 0) >= 4,
            lambda a, h: not is_jacoby_transfer_sequence(a),
        ]),
        StateRule("bid_denies_fit", [
            lambda a, h: a.bid in ("1N", "1NT", "2N", "2NT"),
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.partner_last_bid()) >= 2,
            lambda a, h: a.partner_last_bid()[-1] in ("H", "S"),
            lambda a, h: h.lengths.get(a.partner_last_bid()[-1], 0) < 3,
        ]),
        # Same underlying signal as bid_denies_3_card_support; kept as two
        # names because downstream concept-name vocabularies treat them as
        # distinct, and merging would shrink hard_state_head's output width.
        StateRule("bid_denies_major_fit", [denies_major_fit_after_1nt_response]),
        StateRule("bid_denies_major_fit_after_5_card_major", [
            lambda a, h: a.bid in ("1N", "1NT"),
            lambda a, h: is_response(a),
            lambda a, h: a.partner_last_bid() in ("1H", "1S"),
            lambda a, h: h.lengths.get(a.partner_last_bid()[-1], 0) < 3,
            lambda a, h: (
                h.lengths.get("H", 0) < 4 if a.partner_last_bid() == "1S"
                else h.lengths.get("S", 0) < 4
            ),
        ]),
        # Same underlying signal as bid_denies_major_fit; see comment there.
        StateRule("bid_denies_3_card_support", [denies_major_fit_after_1nt_response]),
        StateRule("bid_completes_jacoby_transfer", [
            lambda a, h: is_jacoby_transfer_sequence(a),
        ]),
        StateRule("bid_shows_minor_suit_only", [
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in ("C", "D"),
            lambda a, h: h.lengths.get("H", 0) < 4,
            lambda a, h: h.lengths.get("S", 0) < 4,
            lambda a, h: is_opening(a) or is_response(a),
            lambda a, h: not (is_opening(a) and a.bid == "2C"),
            lambda a, h: not (
                a.bid == "2C" and is_response(a)
                and a.partner_last_bid() in ("1N", "1NT")
            ),
        ]),
    ]

    # =================================================================
    # 6.  LIMITING CONCEPTS
    # =================================================================
    R += [
        StateRule("bid_limits_to_minimum", [
            lambda a, h: (
                (
                    is_response(a)
                    and bid_level_safe(a.bid) == 2
                    and a.partner_last_bid() is not None
                    and len(a.partner_last_bid()) >= 2
                    and bid_level_safe(a.partner_last_bid()) == 1
                    and _same_suit_as_partner(a)
                )
                or (
                    is_response(a)
                    and a.bid in ("1N", "1NT")
                    and a.partner_last_bid() not in ("1H", "1S")
                )
            ),
            lambda a, h: h.hcp <= 10,
        ]),
        StateRule("bid_limits_to_invitational", [
            lambda a, h: (
                (
                    is_response(a)
                    and a.bid in ("2N", "2NT")
                    and a.partner_last_bid() is not None
                    and bid_level_safe(a.partner_last_bid()) == 1
                    and a.partner_last_bid() not in ("1H", "1S")
                )
                or (
                    is_response(a)
                    and bid_level_safe(a.bid) == 3
                    and a.partner_last_bid() is not None
                    and len(a.partner_last_bid()) >= 2
                    and bid_level_safe(a.partner_last_bid()) == 1
                    and _same_suit_as_partner(a)
                )
            ),
            lambda a, h: h.hcp <= 12,
        ]),
        StateRule("bid_limits_to_game_values", [
            lambda a, h: bid_level_safe(a.bid) is not None,
            lambda a, h: bid_level_safe(a.bid) >= 3,
            # not a slam action (slam bids, Blackwood asks/replies, 5NT asks)
            lambda a, h: bid_level_safe(a.bid) <= 5 and not _is_blackwood_ask(a)
                         and not _is_blackwood_reply(a) and a.bid not in ("5N", "5NT"),
            lambda a, h: not is_opening(a),
            lambda a, h: partnership_has_bid(a),
            lambda a, h: h.hcp <= 17,
            lambda a, h: h.hcp >= 10 or not opponents_have_bid(a),
            lambda a, h: not is_stayman_sequence(a),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_strong_2c_waiting_sequence(a),
        ]),
    ]

    # =================================================================
    # 7.  SLAM INTEREST / INTENT / PREEMPTIVE
    # =================================================================
    R += [
        StateRule("bid_shows_slam_interest", [
            # Slam TRIES: Blackwood, a 5NT king ask, a control cuebid, or a
            # 5-level raise of an agreed major (a slam invitation); not
            # Blackwood replies or slam bids themselves.
            lambda a, h: (
                _is_blackwood_ask(a)
                or (a.bid in ("5N", "5NT") and partner_has_suit(a))
                or _control_bid(a)
                or (a.bid in ("5H", "5S") and a.partner_last_bid() is not None
                    and a.partner_last_bid()[-1:] == a.bid[-1] and not opponents_have_bid(a))
            ),
            lambda a, h: not _is_blackwood_reply(a),
            lambda a, h: not is_strong_2c_waiting_sequence(a),
        ]),
        StateRule("bid_is_sacrifice_candidate", [
            lambda a, h: opponents_have_bid(a),
            # bidding over the OPPONENTS' game (or higher): answering partner's
            # takeout double of 3H with 4D is not a sacrifice
            lambda a, h: _last_real_bid_is_theirs(a)
                         and ((bid_level_safe(_last_real_bid(a)) or 0) >= 4 or _last_real_bid(a) in ("3N", "3NT")),
            lambda a, h: not _is_blackwood_reply(a),
            lambda a, h: h.hcp <= 12,  # a sacrifice is a weak-hand action
            lambda a, h: len(a.bid) >= 2,
            lambda a, h: a.bid[0] in "4567",
            lambda a, h: a.bid[-1] in SUITS,
            lambda a, h: not partner_has_suit(a),
        ]),
        StateRule("bid_shows_preemptive_intent", [
            lambda a, h: (
                is_preemptive_opening(a)
                or (
                    is_jump_overcall(a)
                    and h.hcp <= 10
                    and len(a.bid) >= 2 and a.bid[-1] in SUITS
                    and h.lengths.get(a.bid[-1], 0) >= 6
                )
                or (
                    is_response(a)
                    and opponents_have_bid(a)
                    and h.hcp <= 11
                    and bid_level_safe(a.bid) is not None
                    and bid_level_safe(a.bid) >= 3
                    and len(a.bid) >= 2 and a.bid[-1] in SUITS
                    and h.lengths.get(a.bid[-1], 0) >= 5
                )
            ),
        ]),
        StateRule("bid_shows_competitive_intent", [
            lambda a, h: is_overcall(a),
            lambda a, h: len(a.bid) >= 2 and not a.bid.endswith(("N", "NT")),
            lambda a, h: a.bid[0] in "1234",
        ]),
        StateRule("bid_shows_constructive_intent", [
            lambda a, h: (
                not is_opening(a) and not is_response(a)
                and partnership_has_bid(a)
                and len(a.bid) >= 2
                and a.bid[0] in "23"
                and not a.bid.endswith(("N", "NT"))
            ),
            lambda a, h: not is_stayman_sequence(a),
            lambda a, h: not is_strong_2c_waiting_sequence(a),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_new_minor_forcing_sequence(a),
        ]),
    ]

    # =================================================================
    # 8.  JUMP / REBID FAMILY
    # =================================================================
    R += [
        StateRule("bid_is_jump_bid", [
            lambda a, h: (
                bid_level_safe(a.bid) is not None
                and bid_level_safe(a.my_previous_bid()) is not None
                and bid_level_safe(a.bid) > bid_level_safe(a.my_previous_bid()) + 1
            ),
            lambda a, h: a.bid not in NON_BIDS,
            lambda a, h: a.my_previous_bid() is not None,
            lambda a, h: not is_stayman_sequence(a),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_strong_2c_waiting_sequence(a),
            lambda a, h: not (
                len(a.bid) >= 2 and a.bid[-1] in SUITS
                and a.my_previous_bid() is not None
                and len(a.my_previous_bid()) >= 2
                and _same_suit(a.bid, a.my_previous_bid())
                and (
                    (bid_level_safe(a.bid) == 4 and a.bid[-1] in ("H", "S"))
                    or (bid_level_safe(a.bid) == 5 and a.bid[-1] in ("C", "D"))
                )
            ),
            lambda a, h: not (
                len(a.bid) >= 2 and a.bid[-1] in SUITS
                and a.partner_last_bid() is not None
                and len(a.partner_last_bid()) >= 2
                and _same_suit_as_partner(a)
                and (
                    (bid_level_safe(a.bid) == 4 and a.bid[-1] in ("H", "S"))
                    or (bid_level_safe(a.bid) == 5 and a.bid[-1] in ("C", "D"))
                )
            ),
        ]),
        StateRule("bid_is_rebid", [
            lambda a, h: a.bid not in ("Pass", "P"),
            lambda a, h: a.my_previous_bid() is not None,
            lambda a, h: len(a.bid) >= 2,
            lambda a, h: a.bid[-1] in SUITS,
            lambda a, h: len(a.my_previous_bid()) >= 2,
            lambda a, h: _same_suit(a.bid, a.my_previous_bid()),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_stayman_sequence(a),
        ]),
        StateRule("bid_is_competitive_jump", [
            lambda a, h: opponents_have_bid(a),
            lambda a, h: len(a.bid) >= 2,
            lambda a, h: a.bid[0] in "34",
            lambda a, h: a.bid[-1] in SUITS,
            lambda a, h: bid_level_safe(a.bid) is not None and bid_level_safe(a.bid) > 1 + max(
                (bid_level_safe(b) or 0) for b in a.bids if b not in NON_BIDS
            ),
            lambda a, h: not (
                is_overcall(a) and h.hcp <= 10
                and h.lengths.get(a.bid[-1], 0) >= 6
            ),
            lambda a, h: not is_preemptive_opening(a),
        ]),
        StateRule("bid_shows_jump_shift_strength", [
            lambda a, h: is_response(a),
            # a new natural suit with a jump: not a cuebid of their suit and
            # not a reply to partner's notrump
            lambda a, h: partner_has_suit(a) and _is_jump_over_last(a),
            lambda a, h: not any(b[-1:] == a.bid[-1:] for i, b in enumerate(a.bids)
                                 if b not in NON_BIDS and seat_of_bid(a, i) not in (a.actor_seat(), a.partner)),
            lambda a, h: h.hcp >= 14,
            lambda a, h: bid_level_safe(a.bid) is not None,
            lambda a, h: bid_level_safe(a.partner_last_bid()) is not None,
            lambda a, h: bid_level_safe(a.bid) > bid_level_safe(a.partner_last_bid()) + 1,
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
            lambda a, h: not is_stayman_sequence(a),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_strong_2c_waiting_sequence(a),
        ]),
    ]

    # =================================================================
    # 9.  CONVENTIONAL BIDS
    # =================================================================
    R += [
        StateRule("bid_is_conventional", [
            lambda a, h: (
                (is_opening(a) and a.bid == "2C")
                or (is_response(a) and a.bid in ("2N", "2NT") and
                    a.partner_last_bid() in ("1H", "1S"))
                or (a.bid in ("4N", "4NT"))
                or (is_overcall(a) and a.bid in ("2N", "2NT"))
                or (is_response(a) and a.bid == "2C" and
                    a.partner_last_bid() in ("1N", "1NT"))
                or (is_response(a) and a.bid in ("2D", "2H") and
                    a.partner_last_bid() in ("1N", "1NT"))
                or (is_response(a) and a.bid == "3C" and
                    a.partner_last_bid() in ("2N", "2NT"))
                or (is_response(a) and a.bid in ("3D", "3H") and
                    a.partner_last_bid() in ("2N", "2NT"))
                or is_new_minor_forcing_sequence(a)
                or (is_response(a) and a.bid == "2D" and a.partner_last_bid() == "2C")  # waiting; not 1S P 2C P 2D
                or (is_response(a) and a.bid in ("2N", "2NT") and is_response_to_weak_two(a))
            ),
            lambda a, h: a.bid not in ("1C", "1D", "1H", "1S"),
        ]),
        StateRule("bid_is_stayman", [
            lambda a, h: a.bid in ("2C", "3C"),
            lambda a, h: is_response(a),
            lambda a, h: (
                (a.bid == "2C" and a.partner_last_bid() in ("1N", "1NT"))
                or (a.bid == "3C" and a.partner_last_bid() in ("2N", "2NT"))
            ),
            lambda a, h: not opponents_have_bid(a),
        ]),
        StateRule("bid_is_jacoby_transfer", [
            lambda a, h: is_response(a),
            lambda a, h: (
                (a.bid in ("2D", "2H") and a.partner_last_bid() in ("1N", "1NT"))
                or (a.bid in ("3D", "3H") and a.partner_last_bid() in ("2N", "2NT"))
            ),
            lambda a, h: not opponents_have_bid(a),
        ]),
        StateRule("bid_is_fourth_suit_forcing", [
            lambda a, h: not is_opening(a),
            lambda a, h: not is_response(a),
            lambda a, h: partnership_has_bid(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
            lambda a, h: a.my_previous_bid() is not None,
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.my_previous_bid()) >= 2 and a.my_previous_bid()[-1] in SUITS,
            lambda a, h: len(a.partner_last_bid()) >= 2 and a.partner_last_bid()[-1] in SUITS,
            lambda a, h: len({
                a.bid[-1],
                a.my_previous_bid()[-1],
                a.partner_last_bid()[-1],
            }) == 3,
            lambda a, h: not is_stayman_sequence(a),
            lambda a, h: not is_strong_2c_waiting_sequence(a),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_new_minor_forcing_sequence(a),
            lambda a, h: not a.bid.endswith(("N", "NT")),
            lambda a, h: not is_partnership_opener(a),
            lambda a, h: not (
                bid_level_safe(a.bid) is not None
                and bid_level_safe(a.bid) >= 3
                and bid_level_safe(a.partner_last_bid()) is not None
                and bid_level_safe(a.partner_last_bid()) >= 3
            ),
        ]),
        StateRule("bid_is_splinter", [
            # Value is overridden in SaycStateAuthority.classify() as
            # OR(bid_is_void_splinter, bid_is_singleton_splinter) -- a
            # splinter is by definition exactly the union of those two
            # shortness types, so this predicate list is unused.
            lambda a, h: False,
        ]),
        StateRule("bid_is_void_splinter", [
            lambda a, h: is_response(a),
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.partner_last_bid()) >= 2,
            lambda a, h: len(a.bid) >= 2 and a.bid[0] in "34",
            lambda a, h: a.bid[-1] in SUITS,
            lambda a, h: a.bid[-1] != a.partner_last_bid()[-1],
            lambda a, h: has_void(h),
            lambda a, h: h.lengths.get(a.bid[-1], 0) == 0,
            lambda a, h: h.hcp >= 9,
            lambda a, h: (
                h.lengths.get(a.partner_last_bid()[-1], 0) >= 3
                if a.partner_last_bid()[-1] in ("H", "S")
                else h.lengths.get(a.partner_last_bid()[-1], 0) >= 4
            ),
        ]),
        StateRule("bid_is_singleton_splinter", [
            lambda a, h: is_response(a),
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.partner_last_bid()) >= 2,
            lambda a, h: len(a.bid) >= 2 and a.bid[0] in "34",
            lambda a, h: a.bid[-1] in SUITS,
            lambda a, h: a.bid[-1] != a.partner_last_bid()[-1],
            lambda a, h: has_singleton(h),
            lambda a, h: h.lengths.get(a.bid[-1], 0) == 1,
            lambda a, h: h.hcp >= 9,
            lambda a, h: (
                h.lengths.get(a.partner_last_bid()[-1], 0) >= 3
                if a.partner_last_bid()[-1] in ("H", "S")
                else h.lengths.get(a.partner_last_bid()[-1], 0) >= 4
            ),
        ]),
    ]

    # =================================================================
    # 10. SLAM CONVENTIONS
    # =================================================================
    R += [
        # Gerber: 4C as responder's FIRST bid directly over partner's 1NT/2NT
        # opening, not a natural 4C rebid. The WBridge5 data has no Gerber
        # sequences (0 in 20k deals), so this is almost never true.
        StateRule("bid_is_gerber", [
            lambda a, h: a.bid == "4C",
            lambda a, h: is_response(a),
            lambda a, h: a.partner_last_bid() in ("1N", "1NT", "2N", "2NT"),
            lambda a, h: not opponents_have_bid(a),
        ]),
        StateRule("bid_is_gerber_response", [
            lambda a, h: a.partner_last_bid() == "4C",
            lambda a, h: a.my_first_bid() in ("1N", "1NT", "2N", "2NT"),
            lambda a, h: sum(1 for i, b in enumerate(a.bids)
                             if b not in NON_BIDS and seat_of_bid(a, i) == a.actor_seat()) == 1,
            lambda a, h: a.bid in ("4D", "4H", "4S", "4N", "4NT"),
            lambda a, h: not opponents_have_bid(a),
        ]),
        # Roman Keycard is NOT played in this system: WBridge5 answers 4NT
        # by plain ace count (5C = 0/3, 5D = 1/4, 5H = 2). Kept in the
        # vocabulary (order is load-bearing), never true.
        StateRule("bid_is_roman_keycard", [lambda a, h: False]),
        StateRule("bid_is_keycard_response", [lambda a, h: False]),
        StateRule("bid_is_blackwood", [
            lambda a, h: a.bid in ("4N", "4NT"),
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: not a.partner_last_bid().endswith(("N", "NT")),
            lambda a, h: partner_has_suit(a),
            lambda a, h: not opponents_have_bid(a),
            lambda a, h: partnership_has_bid(a),
        ]),
        StateRule("bid_is_blackwood_response", [
            lambda a, h: _is_blackwood_reply(a),
            lambda a, h: not opponents_have_bid(a),
        ]),
        # Standard "5C = 0 or 4, 5D = 1" is not WBridge5's scheme (5C = 0/3,
        # 5D = 1/4 -- 179 of 393 real 5C replies hold 3 aces), so these two
        # are never true; bid_shows_aces_0_or_3 / bid_shows_aces_1_or_4
        # (appended) replace them.
        StateRule("bid_shows_aces_0_or_4", [lambda a, h: False]),
        StateRule("bid_shows_aces_1", [lambda a, h: False]),
        StateRule("bid_shows_aces_2", [
            lambda a, h: a.bid == "5H",
            lambda a, h: sum("A" in h.cards.get(s_, []) for s_ in SUITS) == 2,
            lambda a, h: _is_blackwood_reply(a),
            lambda a, h: a.bids[-1:] in (["P"], ["Pass"]),  # nothing over the 4NT (earlier bids do not matter)
        ]),
        StateRule("bid_shows_aces_3", [
            lambda a, h: a.bid == "5S",
            lambda a, h: sum("A" in h.cards.get(s_, []) for s_ in SUITS) == 3,
            lambda a, h: _is_blackwood_reply(a),
            lambda a, h: a.bids[-1:] in (["P"], ["Pass"]),  # nothing over the 4NT (earlier bids do not matter)
        ]),
        StateRule("bid_shows_stayman_major_structure", [
            lambda a, h: is_stayman_sequence(a),
            lambda a, h: a.bid in ("2D", "2H", "2S", "3D", "3H", "3S"),
            lambda a, h: (
                (a.bid == "2D" and h.lengths.get("H", 0) < 4 and h.lengths.get("S", 0) < 4)
                or (a.bid == "2H" and h.lengths.get("H", 0) >= 4)
                or (a.bid == "2S" and h.lengths.get("S", 0) >= 4 and h.lengths.get("H", 0) < 4)
                or (a.bid == "3D" and h.lengths.get("H", 0) < 4 and h.lengths.get("S", 0) < 4)
                or (a.bid == "3H" and h.lengths.get("H", 0) >= 4)
                or (a.bid == "3S" and h.lengths.get("S", 0) >= 4 and h.lengths.get("H", 0) < 4)
            ),
        ]),
    ]

    # =================================================================
    # 11. ARTIFICIAL / FORCING RESPONSES
    # =================================================================
    R += [
        StateRule("bid_is_waiting_response", [
            lambda a, h: a.bid == "2D",
            lambda a, h: a.partner_last_bid() == "2C",
            lambda a, h: is_response(a),
        ]),
        StateRule("bid_is_check_back", [
            lambda a, h: is_new_minor_forcing_sequence(a),
        ]),
        StateRule("bid_is_weak_two", [
            # WBridge5 practice (notes/wbridge5_system.md), not textbook
            # SAYC's 6 cards + 5-10 HCP (which matched 75% of the data's own
            # weak twos): 6+ cards, or 5 in third seat, or 5 headed by AK /
            # AQJ; under 12 HCP; at least 6 points (HCP + shape).
            lambda a, h: is_preemptive_opening(a),
            lambda a, h: bid_level_safe(a.bid) == 2,
            lambda a, h: a.bid[-1] in ("D", "H", "S"),
            lambda a, h: (
                _suit_len(a, h) >= 6
                or (_suit_len(a, h) == 5 and (
                    opening_seat_position(a) == 3
                    or {"A", "K"} <= set(h.cards.get(a.bid[-1], []))
                    or {"A", "Q", "J"} <= set(h.cards.get(a.bid[-1], []))))
            ),
            lambda a, h: h.hcp <= 11,
            lambda a, h: wb5_points(h) >= 6,
        ]),
    ]

    # =================================================================
    # 12. INVITATIONAL / QUANTITATIVE NT
    # =================================================================
    R += [
        StateRule("bid_is_pass_or_cuebid", [
            lambda a, h: a.bid in ("P", "Pass"),
            lambda a, h: a.partner_last_bid() in ("1N", "1NT"),
            lambda a, h: any(
                len(b) >= 2 and b[-1] in SUITS
                and seat_of_bid(a, i) not in (a.actor_seat(), a.partner)
                and h.stoppers.get(b[-1], False)
                for i, b in enumerate(a.bids)
                if b not in NON_BIDS
            ),
        ]),
        StateRule("bid_is_quantitative_invite", [
            # 4NT raising partner's notrump (no suit agreed): invites 6NT on
            # high cards. A 4-level suit bid over 3NT is not one.
            lambda a, h: a.bid in ("4N", "4NT"),
            lambda a, h: a.partner_last_bid() in ("1N", "1NT", "2N", "2NT", "3N", "3NT"),
            lambda a, h: not opponents_have_bid(a),
        ]),
        StateRule("bid_is_qualitative_invite", [
            lambda a, h: len(a.bid) >= 2 and a.bid[0] == "3",
            lambda a, h: a.partner_last_bid() in ("2N", "2NT"),
            lambda a, h: 10 <= h.hcp <= 12,
        ]),
        StateRule("bid_is_negative_invite", [
            lambda a, h: a.bid == "2C",
            lambda a, h: a.partner_last_bid() in ("1H", "1S"),
            lambda a, h: h.hcp < 8,
            lambda a, h: max(h.lengths.get("H", 0), h.lengths.get("S", 0)) >= 4,
        ]),
    ]

    # =================================================================
    # 13. COMPETITIVE CONVENTIONS
    # =================================================================
    R += [
        StateRule("bid_is_michaels_cuebid", [
            lambda a, h: is_overcall(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[0] == "2" and a.bid[-1] in SUITS,  # not 2NT
            # directly over a 1-level suit opening (the only real bid so far)
            lambda a, h: [b for b in a.bids if b not in NON_BIDS][:1] != []
                         and len([b for b in a.bids if b not in NON_BIDS]) == 1
                         and bid_level_safe(_last_real_bid(a)) == 1,
            lambda a, h: any(
                b[-1] == a.bid[-1] and b not in NON_BIDS
                for i, b in enumerate(a.bids)
                if seat_of_bid(a, i) not in (a.actor_seat(), a.partner) and len(b) >= 2
            ),
        ]),
        StateRule("bid_is_michaels_response", [
            lambda a, h: is_response(a),
            lambda a, h: opponents_have_bid(a),
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: any(
                b not in NON_BIDS and len(b) >= 2 and b[-1] in SUITS
                and a.partner_last_bid() is not None
                and len(a.partner_last_bid()) >= 2
                and a.partner_last_bid()[-1] == b[-1]
                and a.partner_last_bid()[0] == "2"
                for i, b in enumerate(a.bids)
                if seat_of_bid(a, i) not in (a.actor_seat(), a.partner)
                and b not in NON_BIDS
            ),
            lambda a, h: a.bid in ("2H", "2S"),
        ]),
        StateRule("bid_is_unusual_2nt", [
            # A JUMP to 2NT directly over a 1-level suit opening (137/146 real
            # ones hold 5-5 in the two lowest unbid suits). 2NT over a weak
            # two is natural (median 17 HCP) -> bid_is_natural_2nt_overcall.
            lambda a, h: is_overcall(a),
            lambda a, h: a.bid in ("2N", "2NT"),
            lambda a, h: len([b for b in a.bids if b not in NON_BIDS]) == 1,
            lambda a, h: bid_level_safe(_last_real_bid(a)) == 1 and _last_real_bid(a)[-1] in SUITS,
        ]),
        StateRule("bid_is_negative_double", [
            lambda a, h: a.bid in ("Dbl", "X"),
            lambda a, h: is_response(a),  # partner OPENED (advancer doubles are not negative doubles)
            lambda a, h: h.hcp >= 6,
            lambda a, h: (_last_real_bid(a) or "")[-1:] in SUITS,
            lambda a, h: opponents_have_bid(a),
            lambda a, h: a.partner_last_bid() is not None,
            # SAYC negative doubles apply through about the 4-level; a
            # double of a higher contract is virtually always penalty.
            lambda a, h: max((bid_level_safe(b) or 0) for b in a.bids if b not in NON_BIDS) <= 4,
        ]),
        StateRule("bid_is_penalty_double", [
            lambda a, h: a.bid in ("Dbl", "X"),
            lambda a, h: not is_response(a),
            lambda a, h: opponents_have_bid(a),
            lambda a, h: partnership_has_bid(a),
            lambda a, h: not _opener_reopening_double(a),  # takeout, see its docstring
        ]),
        StateRule("bid_is_takeout_double", [
            lambda a, h: a.bid in ("Dbl", "X"),
            lambda a, h: any(
                len(b) >= 2 and b[-1] in SUITS
                for i, b in enumerate(a.bids)
                if seat_of_bid(a, i) not in (a.actor_seat(), a.partner)
                and b not in NON_BIDS
            ),
            # our side silent, or opener's reopening double (partner silent)
            lambda a, h: not partnership_has_bid(a) or _opener_reopening_double(a),
            # the opponents have not bid notrump (my own 1NT opening is fine)
            lambda a, h: not any(b.endswith(("N", "NT")) and seat_of_bid(a, i) not in (a.actor_seat(), a.partner)
                                 for i, b in enumerate(a.bids)),
            # Doubling a suit bid below game, of which the doubler holds at
            # most 3 cards; doubles of 4H/4S are penalty in practice.
            lambda a, h: (bid_level_safe(_last_real_bid(a)) or 9) <= 3,
            lambda a, h: h.lengths.get((_last_real_bid(a) or "?")[-1], 9) <= 3,
        ]),
    ]

    # =================================================================
    # 14. CONTROL / CUE-BID FAMILY
    # =================================================================
    R += [
        StateRule("bid_is_control_bid", [
            lambda a, h: _control_bid(a),
            lambda a, h: not _is_blackwood_reply(a),
            lambda a, h: partnership_has_bid(a),
            lambda a, h: not is_overcall(a),
            lambda a, h: not is_opening(a),
            lambda a, h: a.my_previous_bid() is not None,
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
            lambda a, h: bid_level_safe(a.bid) is not None and bid_level_safe(a.bid) >= 3,
            lambda a, h: not a.bid.endswith(("N", "NT")),
            lambda a, h: not is_jacoby_transfer_sequence(a),
            lambda a, h: not is_stayman_sequence(a),
            lambda a, h: partner_has_suit(a),
            lambda a, h: bid_level_safe(a.partner_last_bid()) is not None,
            lambda a, h: bid_level_safe(a.partner_last_bid()) >= 3,
            lambda a, h: a.bid[-1] != a.partner_last_bid()[-1],
            lambda a, h: (
                a.my_first_bid() is None
                or len(a.my_first_bid()) < 2
                or a.bid[-1] != a.my_first_bid()[-1]
            ),
        ]),
        StateRule("bid_shows_control", [
            # hand-grounded: first- or second-round control of the bid suit
            lambda a, h: _control_bid(a),
            lambda a, h: bool({"A", "K"} & set(h.cards.get(a.bid[-1], []))) or _suit_len(a, h) <= 1,
            lambda a, h: bid_level_safe(a.bid) is not None,
            lambda a, h: bid_level_safe(a.bid) >= 3,
            lambda a, h: a.bid[-1] in SUITS,
            lambda a, h: partner_has_suit(a),
            lambda a, h: not is_opening(a),
            lambda a, h: not is_stayman_sequence(a),
            lambda a, h: not is_strong_2c_waiting_sequence(a),
            lambda a, h: a.partner_last_bid() is None or a.bid[-1] != a.partner_last_bid()[-1],
            lambda a, h: a.my_previous_bid() is not None,
            lambda a, h: (
                a.my_first_bid() is None
                or len(a.my_first_bid()) < 2
                or a.bid[-1] != a.my_first_bid()[-1]
            ),
        ]),
        # A control cuebid holding the ace or a void (first) / the king or a
        # singleton (second) in the bid suit.
        StateRule("bid_shows_first_control", [
            lambda a, h: _control_bid(a),
            lambda a, h: "A" in h.cards.get(a.bid[-1], []) or _suit_len(a, h) == 0,
        ]),
        StateRule("bid_shows_second_control", [
            lambda a, h: _control_bid(a),
            lambda a, h: not ("A" in h.cards.get(a.bid[-1], []) or _suit_len(a, h) == 0),
            lambda a, h: "K" in h.cards.get(a.bid[-1], []) or _suit_len(a, h) == 1,
        ]),
    ]

    # =================================================================
    # 15. DEFENSIVE / STOPPER / COMPETITIVE
    # =================================================================
    R += [
        StateRule("bid_shows_stopper_in_their_suit", [
            lambda a, h: a.bid.endswith(("N", "NT")),
            lambda a, h: opponents_have_bid(a),
            lambda a, h: not _is_blackwood_ask(a),
            lambda a, h: any(len(b) >= 2 and b[-1] in SUITS for i, b in enumerate(a.bids)
                             if b not in NON_BIDS and seat_of_bid(a, i) not in (a.actor_seat(), a.partner)),
            # ALL of the opponents' suits stopped, not just one
            lambda a, h: all(
                h.stoppers.get(b[-1], False)
                for i, b in enumerate(a.bids)
                if b not in NON_BIDS and len(b) >= 2 and b[-1] in SUITS
                and seat_of_bid(a, i) not in (a.actor_seat(), a.partner)
            ),
        ]),
        StateRule("bid_competes_for_partscore", [
            lambda a, h: opponents_have_bid(a),
            lambda a, h: bid_level_safe(a.bid) is not None,
            lambda a, h: bid_level_safe(a.bid) <= 3,
            lambda a, h: h.hcp <= 12,
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
        ]),
        StateRule("bid_shows_defensive_values", [
            lambda a, h: opponents_have_bid(a),
            lambda a, h: (
                a.bid in ("Dbl", "X")
                or (
                    a.bid in ("P", "Pass")
                    # passing partner's contract is not defending
                    and _last_real_bid_is_theirs(a)
                    and any(
                        bid_level_safe(b) is not None and bid_level_safe(b) >= 3
                        for b in a.bids
                        if b not in NON_BIDS
                    )
                )
            ),
            lambda a, h: h.hcp >= 10,
        ]),
    ]

    # =================================================================
    # 16. SHAPE DESCRIPTION
    # =================================================================
    R += [
        StateRule("bid_shows_single_suited", [
            lambda a, h: promised_length(a) in (6, 7),
            lambda a, h: is_preemptive_opening(a) or is_jump_overcall(a),
        ]),
        StateRule("bid_denies_shortness", [
            lambda a, h: a.bid.endswith(("N", "NT")),
            lambda a, h: not _is_blackwood_ask(a) and not (a.bid in ("5N", "5NT") and partner_has_suit(a)),
            lambda a, h: min(h.lengths.values()) >= 2,
        ]),
    ]

    # =================================================================
    # 17. TARGETED CONCEPTS
    # =================================================================
    R += [
        StateRule("bid_shows_5_card_major_in_1nt", [
            lambda a, h: is_opening(a),
            lambda a, h: a.bid in ("1N", "1NT"),
            lambda a, h: h.balanced or is_semi_balanced(h),
            lambda a, h: h.lengths.get("H", 0) >= 5 or h.lengths.get("S", 0) >= 5,
        ]),
        StateRule("bid_is_simple_game_raise", [
            lambda a, h: not is_opening(a),
            lambda a, h: not is_response(a),
            lambda a, h: partnership_has_bid(a),
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.bid) >= 2 and len(a.partner_last_bid()) >= 2,
            lambda a, h: _same_suit_as_partner(a),
            lambda a, h: a.bid[-1] in SUITS,
            lambda a, h: (
                (bid_level_safe(a.bid) == 4 and a.bid[-1] in ("H", "S"))
                or (bid_level_safe(a.bid) == 5 and a.bid[-1] in ("C", "D"))
            ),
            lambda a, h: bid_level_safe(a.partner_last_bid()) is not None,
            lambda a, h: bid_level_safe(a.partner_last_bid()) < bid_level_safe(a.bid),
            lambda a, h: not opponents_have_bid(a),
        ]),
        StateRule("bid_is_opener_support_raise", [
            lambda a, h: not is_opening(a),
            lambda a, h: not is_response(a),
            lambda a, h: i_opened_auction(a),  # not is_partnership_opener, which includes overcallers
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.partner_last_bid()) >= 2 and len(a.bid) >= 2,
            lambda a, h: _same_suit_as_partner(a),
            lambda a, h: a.bid[-1] in SUITS,
            lambda a, h: a.my_first_bid() is not None,
            lambda a, h: len(a.my_first_bid()) >= 2,
            lambda a, h: a.bid[-1] != a.my_first_bid()[-1],
            lambda a, h: h.lengths.get(a.bid[-1], 0) >= 3,
            lambda a, h: bid_level_safe(a.bid) is not None,
            lambda a, h: bid_level_safe(a.partner_last_bid()) is not None,
            lambda a, h: bid_level_safe(a.bid) == bid_level_safe(a.partner_last_bid()) + 1,
        ]),
        StateRule("bid_is_redouble", [
            # "RDbl" in pyspiel's notation, "XX" in pgx's
            lambda a, h: a.bid in ("RDbl", "XX"),
            lambda a, h: partnership_has_bid(a),
            lambda a, h: h.hcp >= 7,  # WBridge5: 7+ (p5 of 306 real redoubles; median 12)
        ]),
    ]

    # =================================================================
    # 18. 2NT OPENER SYSTEM BIDS
    # =================================================================
    R += [
        StateRule("bid_is_2nt_stayman", [
            lambda a, h: a.bid == "3C",
            lambda a, h: is_response(a),
            lambda a, h: a.partner_last_bid() in ("2N", "2NT"),
            lambda a, h: not opponents_have_bid(a),
        ]),
        StateRule("bid_completes_2nt_jacoby_transfer", [
            lambda a, h: is_jacoby_transfer_sequence(a),
            lambda a, h: a.my_first_bid() is not None,
            lambda a, h: a.my_first_bid() in ("2N", "2NT"),
            lambda a, h: a.bid in ("3H", "3S"),
        ]),
    ]

    # =================================================================
    # 19. MINOR SUIT FIT RAISE
    # =================================================================
    R += [
        StateRule("bid_shows_minor_fit_raise", [
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in ("C", "D"),
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.partner_last_bid()) >= 2,
            lambda a, h: a.partner_last_bid()[-1] in ("C", "D"),
            lambda a, h: _same_suit_as_partner(a),
            lambda a, h: h.lengths.get(a.bid[-1], 0) >= 4,
            lambda a, h: partnership_has_bid(a),
            lambda a, h: not is_opening(a),
            lambda a, h: not is_stayman_sequence(a),
        ]),
    ]

    # =================================================================
    # 20. STAYMAN GAME FIT RAISE
    # =================================================================
    R += [
        StateRule("bid_is_stayman_game_fit_raise", [
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in ("H", "S"),
            lambda a, h: bid_level_safe(a.bid) == 4,
            lambda a, h: a.partner_last_bid() in ("2H", "2S", "3H", "3S"),
            lambda a, h: _same_suit_as_partner(a),
            lambda a, h: a.my_first_bid() in ("2C", "3C"),
            lambda a, h: not opponents_have_bid(a),
            lambda a, h: h.lengths.get(a.bid[-1], 0) >= 4,
        ]),
    ]

    # =================================================================
    # 21. HAND EVALUATION HEURISTICS
    #
    # Classic hand-evaluation heuristics (as opposed to the SAYC
    # bid-meaning rules above, which describe what a bid PROMISES). See
    # predicates_hand.py's module docstring for the calibration evidence
    # behind each threshold.
    # =================================================================
    R += [
        StateRule("bid_opening_meets_rule_of_20", [
            lambda a, h: is_opening(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[0] == "1" and a.bid[-1] in SUITS,
            lambda a, h: rule_of_20_satisfied(h),
        ]),
        StateRule("bid_opening_meets_ltc_standard", [
            lambda a, h: is_opening(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[0] == "1",
            lambda a, h: losing_trick_count(h) <= 7,
        ]),
        StateRule("bid_shows_quick_tricks_2plus", [
            lambda a, h: is_opening(a) or is_overcall(a),
            lambda a, h: quick_tricks(h) >= 2.0,
        ]),
        StateRule("bid_shows_support_points_invite_plus", [
            lambda a, h: a.partner_last_bid() is not None,
            lambda a, h: len(a.bid) >= 2 and len(a.partner_last_bid()) >= 2,
            lambda a, h: a.bid[-1] in SUITS and _same_suit_as_partner(a),
            lambda a, h: h.lengths.get(a.bid[-1], 0) >= 3,
            lambda a, h: support_points(h, a.bid[-1]) >= 10,
        ]),
        StateRule("bid_is_preempt_quality_good", [
            lambda a, h: is_preemptive_opening(a) or is_preemptive_opening_continuation(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
            lambda a, h: preempt_suit_quality_score(h, a.bid[-1]) >= 2,
        ]),
        StateRule("bid_is_preempt_quality_marginal", [
            lambda a, h: is_preemptive_opening(a) or is_preemptive_opening_continuation(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
            lambda a, h: preempt_suit_quality_score(h, a.bid[-1]) < 2,
        ]),
    ]

    # =================================================================
    # 22. WBRIDGE5 SYSTEM + PASS REASONS (see the vocabulary comment and
    #     scripts/audit_concepts.py)
    # =================================================================
    def _wb5_can_open(h: HandFeatures) -> bool:
        # WBridge5: 12+ HCP, or 11 HCP with 13+ points; SAYC's Rule of 20 also
        return h.hcp >= 12 or (h.hcp >= 11 and wb5_points(h) >= 13) or rule_of_20_satisfied(h)

    def _opponents_opened(a: AuctionContext) -> bool:
        real = [(i, b) for i, b in enumerate(a.bids) if b not in NON_BIDS]
        return bool(real) and seat_of_bid(a, real[0][0]) not in (a.actor_seat(), a.partner)

    def _has_overcall_suit(a: AuctionContext, h: HandFeatures) -> bool:
        """A 5+ card suit the opponents have not bid, headed by 2+ of AKQJ
        (WBridge5 passes 8-11 HCP hands whose only long suit is poor or theirs)."""
        theirs = {b[-1] for i, b in enumerate(a.bids) if b not in NON_BIDS and b[-1] in SUITS
                  and seat_of_bid(a, i) not in (a.actor_seat(), a.partner)}
        return any(h.lengths.get(s_, 0) >= 5 and sum(r in h.cards.get(s_, []) for r in "AKQJ") >= 2
                   for s_ in SUITS if s_ not in theirs)

    def _agreed_suit(a: AuctionContext):
        """Our side's trump suit: the last suit both partners have bid, else partner's last suit."""
        ours = [(seat_of_bid(a, i), b[-1]) for i, b in enumerate(a.bids)
                if b not in NON_BIDS and len(b) >= 2 and b[-1] in SUITS
                and seat_of_bid(a, i) in (a.actor_seat(), a.partner)]
        mine = {s_ for seat, s_ in ours if seat == a.actor_seat()}
        theirs_p = [s_ for seat, s_ in ours if seat == a.partner]
        for s_ in reversed(theirs_p):
            if s_ in mine:
                return s_
        return theirs_p[-1] if theirs_p else None

    def _partner_aces_from_reply(a: AuctionContext, my_aces: int):
        """Partner's aces from a Blackwood reply to MY 4NT (5C = 0/3, 5D = 1/4,
        5H = 2, 5S = 3), resolving 0/3 and 1/4 against the aces I hold."""
        reply = a.partner_last_bid()
        options = {"5C": (0, 3), "5D": (1, 4), "5H": (2,), "5S": (3,)}.get(reply)
        if not options or a.my_previous_bid() not in ("4N", "4NT"):
            return None
        fits = [o for o in options if o + my_aces <= 4]
        return max(fits) if fits else None

    def _fit_with_partner(a: AuctionContext, h: HandFeatures) -> int:
        pl = a.partner_last_bid()
        return h.lengths.get(pl[-1], 0) if pl and len(pl) >= 2 and pl[-1] in SUITS else 0

    def _actor_has_real_bid(a: AuctionContext) -> bool:
        return any(b not in NON_BIDS and seat_of_bid(a, i) == a.actor_seat() for i, b in enumerate(a.bids))

    def _actor_only_passed(a: AuctionContext) -> bool:
        return all(b in ("P", "Pass") for i, b in enumerate(a.bids) if seat_of_bid(a, i) == a.actor_seat())

    R += [
        StateRule("bid_shows_aces_0_or_3", [
            lambda a, h: a.bid == "5C",
            lambda a, h: sum("A" in h.cards.get(s_, []) for s_ in SUITS) in (0, 3),
            lambda a, h: _is_blackwood_reply(a),
            lambda a, h: a.bids[-1:] in (["P"], ["Pass"]),  # nothing over the 4NT (earlier bids do not matter)
        ]),
        StateRule("bid_shows_aces_1_or_4", [
            lambda a, h: a.bid == "5D",
            lambda a, h: sum("A" in h.cards.get(s_, []) for s_ in SUITS) in (1, 4),
            lambda a, h: _is_blackwood_reply(a),
            lambda a, h: a.bids[-1:] in (["P"], ["Pass"]),  # nothing over the 4NT (earlier bids do not matter)
        ]),
        StateRule("bid_is_light_third_seat_weak_two", [
            lambda a, h: is_preemptive_opening(a) and bid_level_safe(a.bid) == 2 and a.bid[-1] in ("D", "H", "S"),
            lambda a, h: opening_seat_position(a) == 3,
            lambda a, h: _suit_len(a, h) == 5,
            lambda a, h: h.hcp <= 11,
        ]),
        StateRule("bid_is_unusual_4nt_opening", [
            lambda a, h: is_opening(a) and a.bid in ("4N", "4NT"),
            lambda a, h: h.lengths.get("C", 0) >= 5 and h.lengths.get("D", 0) >= 5,
            lambda a, h: h.lengths.get("C", 0) + h.lengths.get("D", 0) >= 11,
        ]),
        StateRule("bid_is_natural_2nt_overcall", [
            lambda a, h: is_overcall(a) and a.bid in ("2N", "2NT"),
            lambda a, h: not (len([b for b in a.bids if b not in NON_BIDS]) == 1
                              and bid_level_safe(_last_real_bid(a)) == 1),  # not the unusual jump
            lambda a, h: 15 <= h.hcp <= 19,
            lambda a, h: all(h.stoppers.get(b[-1], False) for i, b in enumerate(a.bids)
                             if b not in NON_BIDS and b[-1] in SUITS
                             and seat_of_bid(a, i) not in (a.actor_seat(), a.partner)),
        ]),
        StateRule("bid_is_two_over_one_response", [
            lambda a, h: is_response(a) and uncontested(a),
            lambda a, h: a.partner_last_bid() in ("1C", "1D", "1H", "1S"),
            lambda a, h: bid_level_safe(a.bid) == 2 and a.bid[-1] in SUITS,
            lambda a, h: a.bid[-1] != a.partner_last_bid()[-1],
            lambda a, h: suit_rank(a.bid[-1]) < suit_rank(a.partner_last_bid()[-1]),  # not a jump shift
            lambda a, h: h.hcp >= 10,
        ]),
        StateRule("bid_raises_partners_overcall", [
            lambda a, h: is_advance(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS,
            lambda a, h: partner_has_suit(a) and _same_suit_as_partner(a),
            lambda a, h: _suit_len(a, h) >= 3,
        ]),
        StateRule("bid_pass_lacks_opening_values", [
            lambda a, h: a.bid in ("P", "Pass"),
            lambda a, h: in_opening_position(a),
            lambda a, h: not _wb5_can_open(h),
        ]),
        StateRule("bid_pass_lacks_values_to_compete", [
            lambda a, h: a.bid in ("P", "Pass"),
            lambda a, h: _opponents_opened(a),
            lambda a, h: _actor_only_passed(a),
            # no real bid by our side yet: too weak / no long suit to enter;
            # partner already acted (I am the advancer, WBridge5 data: median
            # 7 HCP, 2 cards in partner's suit): too weak, or no fit
            lambda a, h: (
                (h.hcp < 8 or (h.hcp < 12 and not _has_overcall_suit(a, h)))
                if not partnership_has_bid(a) else
                (h.hcp < 8 or (_fit_with_partner(a, h) < 3 and h.hcp < 11))
            ),
        ]),
        StateRule("bid_pass_accepts_partners_contract", [
            lambda a, h: a.bid in ("P", "Pass"),
            lambda a, h: _last_real_bid(a) is not None,
            # the contract on the table is partner's own natural bid (the last
            # real bid was partner's and nobody has doubled it since)
            lambda a, h: any(b == _last_real_bid(a) and seat_of_bid(a, i) == a.partner
                             for i, b in enumerate(a.bids)),
            lambda a, h: a.bids[-1] in ("P", "Pass") or seat_of_bid(a, len(a.bids) - 1) == a.partner,
            lambda a, h: not any(b in ("X", "Dbl", "XX", "RDbl") for b in
                                 a.bids[max(i for i, b in enumerate(a.bids) if b == _last_real_bid(a)):]),
            lambda a, h: not _is_blackwood_ask(AuctionContext(bids=a.bids, bid=_last_real_bid(a), dealer=a.dealer))
                         and _last_real_bid(a) not in ("4N", "4NT", "5N", "5NT"),
        ]),
    ]

    def _role_minimum(a: AuctionContext) -> int:
        """Upper HCP bound of a minimum hand for the actor's role so far."""
        if i_opened_auction(a):
            return 14                     # minimum opening
        opener = next((i for i, b in enumerate(a.bids) if b not in NON_BIDS), None)
        if opener is not None and seat_of_bid(a, opener) == a.partner:
            return 9                      # responder: a minimum response
        return 11                         # overcaller / advancer / competitor

    R += [
        StateRule("bid_pass_too_weak_to_respond", [
            lambda a, h: a.bid in ("P", "Pass"),
            lambda a, h: is_response(a),  # partner opened, I have not bid
            lambda a, h: (
                # over an overcall (WBridge5 data: median 5 HCP, 2-card fit)
                (h.hcp < 8 or (_fit_with_partner(a, h) < 3 and h.hcp < 10))
                if opponents_have_bid(a) else
                h.hcp < (6 if bid_level_safe(a.partner_last_bid()) == 1 and not a.partner_last_bid().endswith(("N", "NT"))
                         else 8 if a.partner_last_bid() in ("1N", "1NT")
                         else 14)  # partner's 2-level+ opening: no game interest
            ),
        ]),
        StateRule("bid_double_with_length_in_their_suit", [
            lambda a, h: a.bid in ("X", "Dbl"),
            lambda a, h: (_last_real_bid(a) or "")[-1:] in SUITS,
            lambda a, h: any(b == _last_real_bid(a) and seat_of_bid(a, i) not in (a.actor_seat(), a.partner)
                             for i, b in enumerate(a.bids)),
            lambda a, h: h.lengths.get(_last_real_bid(a)[-1], 0) >= 3,
            lambda a, h: not is_response(a),  # responder's double after an overcall is negative, not length
            # not a takeout-shaped double (our side silent, a low-level suit
            # bid): there 3 cards in their suit is not "length", and saying so
            # contradicts the takeout message
            # (opener's reopening double, 1C P P 1S X, is takeout-shaped too)
            lambda a, h: (partnership_has_bid(a) and not _opener_reopening_double(a))
                         or (bid_level_safe(_last_real_bid(a)) or 0) >= 4
                         or h.lengths.get(_last_real_bid(a)[-1], 0) >= 4,
        ]),
        StateRule("bid_pass_no_extra_values", [
            lambda a, h: a.bid in ("P", "Pass"),
            lambda a, h: partnership_has_bid(a) and _actor_has_real_bid(a),
            lambda a, h: h.hcp <= _role_minimum(a),
        ]),
        StateRule("bid_stayman_with_4_card_major", [
            lambda a, h: a.bid in ("2C", "3C") and is_response(a) and not opponents_have_bid(a),
            lambda a, h: a.partner_last_bid() in (("1N", "1NT") if a.bid == "2C" else ("2N", "2NT")),
            # a 4-card major, including 5-4 in the majors (16% of WBridge5's
            # Stayman bids are 5-4 hands)
            lambda a, h: 4 in (h.lengths.get("H", 0), h.lengths.get("S", 0)),
            lambda a, h: a.bids[-1:] in (["P"], ["Pass"]),  # after 1NT-X the system is off
        ]),
        StateRule("bid_transfer_with_5_card_major", [
            lambda a, h: is_response(a) and not opponents_have_bid(a),
            lambda a, h: (a.bid in ("2D", "2H") and a.partner_last_bid() in ("1N", "1NT"))
                         or (a.bid in ("3D", "3H") and a.partner_last_bid() in ("2N", "2NT")),
            lambda a, h: h.lengths.get("H" if a.bid[-1] == "D" else "S", 0) >= 5,
            lambda a, h: a.bids[-1:] in (["P"], ["Pass"]),  # after 1NT-X the system is off
        ]),
    ]

    R += [
        StateRule("bid_is_in_clubs", [lambda a, h: len(a.bid) >= 2 and a.bid[0].isdigit() and a.bid[-1] == "C"]),
        StateRule("bid_is_in_diamonds", [lambda a, h: len(a.bid) >= 2 and a.bid[0].isdigit() and a.bid[-1] == "D"]),
        StateRule("bid_is_in_hearts", [lambda a, h: len(a.bid) >= 2 and a.bid[0].isdigit() and a.bid[-1] == "H"]),
        StateRule("bid_is_in_spades", [lambda a, h: len(a.bid) >= 2 and a.bid[0].isdigit() and a.bid[-1] == "S"]),
        StateRule("bid_is_in_notrump", [lambda a, h: len(a.bid) >= 2 and a.bid[0].isdigit() and a.bid.endswith(("N", "NT"))]),
    ]

    ACES = lambda h: sum("A" in h.cards.get(s_, []) for s_ in SUITS)  # noqa: E731
    KINGS = lambda h: sum("K" in h.cards.get(s_, []) for s_ in SUITS)  # noqa: E731
    R += [
        # rare-situation reasons
        StateRule("bid_pass_content_with_final_contract", [
            lambda a, h: a.bid in ("P", "Pass"),
            lambda a, h: a.bids[-1:] in (["P"], ["Pass"]) or (len(a.bids) and seat_of_bid(a, len(a.bids) - 1) == a.partner),
            lambda a, h: any(b == _last_real_bid(a) and seat_of_bid(a, i) == a.partner for i, b in enumerate(a.bids)),
            lambda a, h: _last_real_bid(a) in ("3N", "3NT", "4H", "4S", "5C", "5D")
                         or (bid_level_safe(_last_real_bid(a)) or 0) >= 6,
            lambda a, h: not any(b in ("X", "Dbl") for b in a.bids[-3:]),
            # not strong enough to go on: game -> no slam drive, slam -> no grand
            # game: up to 19 HCP is still short of slam opposite a game bid
            # (WBridge5 openers pass 3NT with 18-19); slam: no grand-slam drive
            lambda a, h: h.hcp <= (19 if (bid_level_safe(_last_real_bid(a)) or 0) < 6 else 21),
        ]),
        StateRule("bid_slam_with_enough_aces", [
            lambda a, h: (bid_level_safe(a.bid) or 0) >= 6,
            lambda a, h: _partner_aces_from_reply(a, ACES(h)) is not None,
            lambda a, h: ACES(h) + _partner_aces_from_reply(a, ACES(h)) >= 3,
        ]),
        StateRule("bid_answers_king_ask", [
            lambda a, h: a.partner_last_bid() in ("5N", "5NT"),
            lambda a, h: any(b in ("4N", "4NT") and seat_of_bid(a, i) == a.partner for i, b in enumerate(a.bids)),
            lambda a, h: {"6C": 0, "6D": 1, "6H": 2, "6S": 3}.get(a.bid) == KINGS(h),
        ]),
        StateRule("bid_is_preemptive_jump_response", [
            lambda a, h: is_response(a),
            lambda a, h: opponents_have_bid(a) or any(b in ("X", "Dbl") and seat_of_bid(a, i) != a.partner
                                                      for i, b in enumerate(a.bids)),
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS and _is_jump_over_last(a),
            lambda a, h: a.partner_last_bid() is not None and a.bid[-1] != a.partner_last_bid()[-1:],
            lambda a, h: h.hcp <= 9 and _suit_len(a, h) >= 6,
        ]),
        StateRule("bid_blackwood_with_fit", [
            lambda a, h: _is_blackwood_ask(a),
            lambda a, h: _agreed_suit(a) is not None and h.lengths.get(_agreed_suit(a), 0) >= 3,
        ]),
        StateRule("bid_waiting_response_without_positive", [
            lambda a, h: a.bid == "2D" and a.partner_last_bid() == "2C" and is_response(a),
            # no positive response available (WBridge5 data: waiting 2-9 HCP,
            # positive suit responses 7+ HCP with a 5+ card suit)
            lambda a, h: h.hcp < 7 or not any(h.lengths.get(s_, 0) >= 5 for s_ in ("H", "S", "C", "D")),
        ]),
        StateRule("bid_positive_response_to_2c", [
            lambda a, h: is_response(a) and a.partner_last_bid() == "2C",
            lambda a, h: (a.bid in ("2H", "2S", "3C", "3D") and _suit_len(a, h) >= 5 and h.hcp >= 7)
                         or (a.bid in ("2N", "2NT") and h.hcp >= 10),
        ]),
    ]

    R += [
        StateRule("bid_completes_transfer_without_super_accept", [
            lambda a, h: is_jacoby_transfer_sequence(a),
            # a super-accept (jump) needs 4-card support and a maximum
            lambda a, h: _suit_len(a, h) < 4 or h.hcp <= 16,
        ]),
        StateRule("bid_king_ask_with_all_aces", [
            lambda a, h: a.bid in ("5N", "5NT") and a.my_previous_bid() in ("4N", "4NT"),
            lambda a, h: _partner_aces_from_reply(a, ACES(h)) is not None,
            lambda a, h: ACES(h) + _partner_aces_from_reply(a, ACES(h)) == 4,
        ]),
        StateRule("bid_small_slam_missing_a_king", [
            lambda a, h: (bid_level_safe(a.bid) or 0) == 6 and a.my_previous_bid() in ("5N", "5NT"),
            lambda a, h: a.partner_last_bid() in ("6C", "6D", "6H", "6S"),
            lambda a, h: KINGS(h) + {"6C": 0, "6D": 1, "6H": 2, "6S": 3}[a.partner_last_bid()] < 4,
        ]),
        StateRule("bid_is_preemptive_raise", [
            lambda a, h: opponents_have_bid(a),
            lambda a, h: len(a.bid) >= 2 and a.bid[-1] in SUITS and (bid_level_safe(a.bid) or 0) >= 3,
            lambda a, h: _agreed_suit(a) == a.bid[-1] or (partner_has_suit(a) and a.partner_last_bid()[-1] == a.bid[-1]),
            lambda a, h: h.hcp <= 9 and _suit_len(a, h) >= 4,
        ]),
    ]

    # =================================================================
    # VALIDATION
    # =================================================================
    missing = set(HARD_STATE_CONCEPTS) - {r.name for r in R}
    extra   = {r.name for r in R} - set(HARD_STATE_CONCEPTS)
    if missing or extra:
        print("MISSING concepts :", missing)
        print("EXTRA concepts   :", extra)
        assert not missing and not extra, "Concept mismatch!"

    return R
