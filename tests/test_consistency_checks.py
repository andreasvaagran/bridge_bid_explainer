"""Regression tests for nlg/consistency_checks.py and the fallback concept selection: every false positive and bug
found so far."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concepts import HandFeatures
from nlg.consistency_checks import (
    check_explanation,
    check_partner_opponent_attribution,
    render_fallback_explanation,
    strip_tautological_clauses,
)


def _hand(hcp=13, lengths=None, cards=None):
    lengths = lengths or {"S": 4, "H": 3, "D": 4, "C": 2}
    cards = cards or {"S": ["K", "J", "8", "4"], "H": ["3"], "D": ["K", "8", "7"], "C": ["A", "Q"]}
    return HandFeatures(hcp=hcp, lengths=lengths, balanced=True,
                         stoppers={"S": True, "H": False, "D": True, "C": False},
                         passed_hand=False, cards=cards)


def test_fallback_explanation_passes_its_own_check_for_every_bid_category():
    """Regression test for the fallback template's headline verb needing
    to be bid-aware -- it originally always said "bids {bid}", which
    never contains "double"/"redouble" that check_bid_consistency
    requires for X/XX, so the "guaranteed-safe" last resort failed its
    own safety check for every Double/Redouble decision."""
    hand = _hand()
    base_frame = {"table": {"you": "S"}, "hcp": 13, "hand_str": "S:KJ84 H:3 D:K87 C:AQ"}
    for bid in ("P", "X", "XX", "1NT", "3D"):
        frame = dict(base_frame, bid=bid, auction_before=(
            [{"seat": "W", "bid": "1D"}, {"seat": "N", "bid": "P"}, {"seat": "E", "bid": "P"}]
            if bid in ("X", "XX") else []
        ))
        explanation = render_fallback_explanation(frame, hand)
        warnings = check_explanation(frame, explanation)
        assert warnings == [], f"bid={bid!r}: {explanation!r} -> {warnings}"


def test_tautological_clause_is_stripped_not_just_flagged():
    text = "East passes, holding 12 HCP. Given that it is a pass. It also shows minimum values."
    stripped = strip_tautological_clauses(text)
    assert "Given that it is a pass" not in stripped
    assert "It also shows minimum values." in stripped

    frame = {"bid": "P", "hcp": 12, "hand_str": "S:AKQ2 H:432 D:432 C:432"}
    # The un-stripped text would fail (tautological clause present);
    # after stripping it should pass cleanly.
    assert check_explanation(frame, text) != []
    assert check_explanation(frame, stripped) == []


def test_tautological_clause_with_real_content_is_left_alone():
    """Anchored ($) so a sentence with real content after "pass" is not
    touched -- only a sentence that is JUST the restatement."""
    text = "As it is a pass in first seat with no opening values."
    assert strip_tautological_clauses(text) == text


def test_suit_length_claim_known_bad_and_known_good():
    frame = {"hand_str": "S:AKQ2 H:43 D:432 C:432"}  # 4 spades, 2 hearts
    bad = "This bid shows at least five spades."
    good = "This bid shows at least four spades."
    assert check_explanation({**frame, "bid": "1S"}, bad) != []
    assert check_explanation({**frame, "bid": "1S"}, good) == []


def test_hcp_threshold_claim_known_bad_and_known_good():
    bad = "South passes, holding 14 HCP -- below the 12-point minimum."
    good = "South passes, holding 9 HCP -- below the 12-point minimum."
    frame_bad = {"bid": "P", "hcp": 14, "hand_str": "S:AKQ2 H:43 D:432 C:432"}
    frame_good = {"bid": "P", "hcp": 9, "hand_str": "S:AKQ2 H:43 D:432 C:432"}
    assert check_explanation(frame_bad, bad) != []
    assert check_explanation(frame_good, good) == []


def test_partner_attribution_known_bad_and_known_good():
    # Opponent bid 1S most recently; a "partner's 1S" claim is wrong.
    frame_bad = {
        "bid": "2S", "auction_before": [{"seat": "N", "bid": "1C"}, {"seat": "E", "bid": "1S"}],
    }
    bad = "raise partner's 1S raise"
    assert check_partner_opponent_attribution(frame_bad, bad) is not None

    # No prior auction at all -- the check has nothing to contradict a
    # "partner's X" phrase with, so it's always safe (this is also the
    # only case where the check CAN pass cleanly: in a strict 4-seat
    # rotation, partner's own last bid can never equal the auction's
    # overall most recent call once any auction has actually happened --
    # there's always at least one intervening opponent call between any
    # two of the same partnership's turns -- so a non-empty auction_before
    # always makes this check fire on any "partner's X" phrase unless the
    # referenced value happens to coincidentally match the true last bid).
    frame_good = {"bid": "1NT", "auction_before": []}
    good = "in response to partner's 1C"
    assert check_partner_opponent_attribution(frame_good, good) is None

    # A real Qwen3-8B case: partner (E) bid 2S two calls earlier; comparing
    # against the most recent call (South's pass) would reject every correct
    # "partner's <bid>".
    frame_cue = {"bid": "2NT", "auction_before": [
        {"seat": "S", "bid": "1C"}, {"seat": "W", "bid": "X"}, {"seat": "N", "bid": "1S"},
        {"seat": "E", "bid": "2S"}, {"seat": "S", "bid": "P"}]}
    assert check_partner_opponent_attribution(frame_cue, "Partner's 2S bid was a cuebid.") is None
    assert check_partner_opponent_attribution(frame_cue, "After partner's 1S, ...") is not None


def test_partner_attribution_does_not_false_flag_fallback_template_phrasing():
    """Regression test for a real false positive in the predecessor
    codebase: an early, broader partner-attribution regex matched the
    fallback template's own generic rule-engine gloss text (e.g. "...is a
    response to partner's bid...", a category name, not a claim about who
    made the most recent auction bid)."""
    hand = _hand()
    frame = {
        "table": {"you": "S"}, "hcp": 13, "hand_str": "S:KJ84 H:3 D:K87 C:AQ", "bid": "1NT",
        "auction_before": [],
    }
    explanation = render_fallback_explanation(frame, hand)
    assert check_explanation(frame, explanation) == []


def test_select_concepts_drops_structurally_impossible_pass_label():
    """Regression test found while smoke-testing an early checkpoint:
    the concept head can predict bid_is_pass with high confidence for a
    decision whose real bid was an opening (the two heads haven't
    converged to agree yet) -- selected_concepts must never carry that
    label through to training data."""
    from nlg.explanation_inputs import _select_concepts
    frame = {"bid": "1D", "top_concepts": [["bid_is_pass", 0.9]]}
    assert _select_concepts(frame) == []

    frame_ok = {"bid": "P", "top_concepts": [["bid_is_pass", 0.9]]}
    assert _select_concepts(frame_ok) == ["bid_is_pass"]


def _auction_frame(you, partner, prior, bid, hand_str="S:AK83 H:JT74 D:J86 C:AT"):
    return {"table": {"you": you, "partner": partner,
                      "opponents": [s for s in "NESW" if s not in (you, partner)]},
            "auction_before": [{"seat": s, "bid": b} for s, b in prior], "bid": bid,
            "hand_str": hand_str, "hcp": 13, "dealer": prior[0][0] if prior else you}


def test_role_bid_claims_reject_misattributed_openings_and_bids():
    """Real Qwen3-8B errors: "Partner opened 1S" when partner had responded
    1S to our 1C, and "Partner opened 1C" when an opponent opened 1C.
    check_partner_opponent_attribution covers only the "partner's <bid>"
    phrasing."""
    from nlg.consistency_checks import check_role_bid_claims
    raise_2s = _auction_frame("N", "S", [("S", "P"), ("W", "P"), ("N", "1C"), ("E", "P"),
                                         ("S", "1S"), ("W", "P")], "2S")
    assert check_role_bid_claims(raise_2s, "Partner opened 1S, and you hold 4 spades.")
    assert check_role_bid_claims(raise_2s, "You opened 1C and partner responded 1S.") is None
    assert check_role_bid_claims(raise_2s, "After you opened 1C, partner bid 1S.") is None

    takeout = _auction_frame("W", "E", [("S", "1C")], "X")
    assert check_role_bid_claims(takeout, "Partner opened 1C, so this is a takeout double.")
    assert check_role_bid_claims(takeout, "South opened 1C, so West doubles for takeout.") is None
    assert check_role_bid_claims(takeout, "The opponents opened 1C.") is None

    cue = _auction_frame("W", "E", [("S", "1C"), ("W", "X"), ("N", "1S"), ("E", "2S"), ("S", "P")], "2NT")
    assert check_role_bid_claims(cue, "Partner bid 2S, a cuebid of North's 1S.") is None
    assert check_role_bid_claims(cue, "The opponents bid 2S.")


def test_role_bid_claims_is_part_of_check_explanation():
    from nlg.consistency_checks import check_explanation as ce
    takeout = _auction_frame("W", "E", [("S", "1C")], "X")
    assert any("opened" in w or "no such call" in w
               for w in ce(takeout, "Partner opened 1C, so West doubles for takeout."))


def test_attribution_checks_ignore_actor_headline_and_point_ranges():
    """False positives found validating the hand-written gold set
    (nlg/gold_examples.jsonl): "North bids 3H" when North is the bidder,
    and "partner's 15 to 17 HCP" read as a call."""
    from nlg.consistency_checks import check_role_bid_claims
    f = _auction_frame("N", "S", [("E", "1H"), ("S", "1S"), ("W", "P")], "3H")
    assert check_role_bid_claims(f, "North bids 3H, a cuebid raise of partner's spades.") is None
    nt = {"bid": "3NT", "auction_before": [{"seat": "N", "bid": "P"}, {"seat": "E", "bid": "1NT"},
                                           {"seat": "S", "bid": "P"}]}
    assert check_partner_opponent_attribution(nt, "game opposite partner's 15 to 17 HCP") is None
    assert check_partner_opponent_attribution(nt, "West bids 3NT over partner's 1NT.") is None


def test_hand_claims_catch_real_qwen_errors():
    """Real Qwen3-8B few-shot outputs: wrong support length, wrong ace count, "no 4-card major" while
    holding two, plus void/singleton and "N suit (" count claims."""
    from nlg.consistency_checks import check_hand_claims
    raise_f = _auction_frame("N", "S", [("N", "P"), ("E", "1S"), ("S", "2H"), ("W", "3C")], "3H",
                             hand_str="S:Q965 H:87642 D:A8 C:J2")
    assert check_hand_claims(raise_f, "North bids 3H, showing 3-card support for hearts.")
    assert check_hand_claims(raise_f, "North bids 3H with 5 hearts (8 7 6 4 2) and 5-card support.") is None
    aces = {"hand_str": "S:AQ6 H:754 D:AT93 C:KQ6", "bid": "5NT"}
    assert check_hand_claims(aces, "North holds only one ace.")
    assert check_hand_claims(aces, "North holds two aces.") is None
    assert check_hand_claims({"hand_str": "S:KJ97 H:A986 D:K2 C:KQ8", "bid": "X"},
                             "") is None
    assert check_explanation({"hand_str": "S:KJ97 H:A986 D:K2 C:KQ8", "bid": "X", "hcp": 16},
                             "South doubles for takeout, with no 4-card major.") != []
    shape = {"hand_str": "S:AKJT643 H:QJ643 D:- C:7", "bid": "3H"}
    assert check_hand_claims(shape, "South bids 3H, with a void in diamonds and a singleton club.") is None
    assert check_hand_claims(shape, "South bids 3H, with a void in clubs.")
    assert check_hand_claims(shape, "South bids 3H, with 6 spades (A K J T 6 4 3).")


def test_at_least_claim_about_partners_call_is_not_a_hand_claim():
    from nlg.consistency_checks import check_suit_length_claims
    f = {"hand_str": "S:J2 H:KT D:AJ63 C:AQT98"}
    assert check_suit_length_claims(f, "Partner's 3S promised at least six spades, so J 2 is enough.") is None
    assert check_suit_length_claims(f, "West has at least six spades.") is not None


def test_partners_double_may_be_an_earlier_call():
    from nlg.consistency_checks import check_partner_opponent_attribution
    f = {"auction_before": [{"seat": "E", "bid": "1D"}, {"seat": "S", "bid": "1H"}, {"seat": "W", "bid": "X"},
                            {"seat": "N", "bid": "XX"}, {"seat": "E", "bid": "2S"}, {"seat": "S", "bid": "P"},
                            {"seat": "W", "bid": "P"}, {"seat": "N", "bid": "3H"}]}
    assert check_partner_opponent_attribution(f, "Partner's double promised four spades.") is None
    assert check_partner_opponent_attribution(f, "Partner's redouble showed values.") is not None


def test_headline_convention_must_fit_the_call():
    from nlg.consistency_checks import check_convention_claims
    f = {"bid": "2C"}
    assert check_convention_claims(f, "East bids 2C, a transfer to hearts. East holds 4 hearts.") is not None
    assert check_convention_claims(f, "East bids 2C, Stayman, asking partner for a four-card major.") is None
    assert check_convention_claims({"bid": "2D"}, "North bids 2D, answering partner's Stayman.") is None
    assert check_convention_claims({"bid": "5H"}, "North bids 5H, showing 1 or 4 aces.") is not None


def test_length_comparison_must_hold():
    from nlg.consistency_checks import check_length_comparisons
    f = {"hand_str": "S:AT4 H:A8 D:KT86 C:JT63"}
    assert check_length_comparisons(f, "North opens 1D; diamonds are longer than clubs (4 against 4).") is not None
    g = {"hand_str": "S:AT H:KQ3 D:AQ3 C:KJ752"}
    assert check_length_comparisons(g, "Clubs are longer than diamonds (5 against 3).") is None


def test_the_false_shape_clause_is_repaired_by_cutting_it():
    from nlg.eval_qwen import repair
    f = {"deal_id": 1, "step": 2, "table": {"you": "N", "partner": "S", "opponents": ["E", "W"]},
         "auction_before": [{"seat": "N", "bid": "P"}, {"seat": "E", "bid": "P"}, {"seat": "S", "bid": "1NT"},
                            {"seat": "W", "bid": "P"}],
         "bid": "2NT", "hand_str": "S:7 H:95 D:QJT92 C:AT953", "dealer": "N"}
    bad = ("North bids 2NT, inviting partner to bid 3NT with a maximum. With 7 HCP opposite partner's 15 to 17, "
           "the partnership has 22 to 24, there is no four-card major to look for (1 spade, 2 hearts), and the hand "
           "is semi-balanced (1-2-5-5).")
    fixed = repair(f, bad)
    assert fixed and "semi-balanced" not in fixed and fixed.endswith("(1 spade, 2 hearts).")


def test_strength_claims_catch_the_cuebid_contradiction_and_false_numbers():
    """An advancer's cuebid "showing a limit raise or better" whose next clause
    says "not to invite game" is caught, and HCP figures, support points,
    Rule-of-20 sums, losers, quick tricks, partnership sums, shape strings and
    stoppers are compared with the hand. Each case below is a real teacher or
    Qwen3-8B output or a minimal variant of one."""
    from nlg.consistency_checks import check_strength_claims
    f = {"bid": "3D", "hcp": 11, "hand_str": "S:K543 H:Q863 D:Q C:A974",
         "table": {"you": "E", "partner": "W", "opponents": ["N", "S"]},
         "auction_before": [{"seat": "N", "bid": "P"}, {"seat": "E", "bid": "P"}, {"seat": "S", "bid": "1D"},
                            {"seat": "W", "bid": "2H"}, {"seat": "N", "bid": "X"}]}
    bad = ("East bids 3D, a cuebid of the opponents' diamonds, showing a limit raise or better in partner's hearts. "
           "East holds 4 hearts (Q 8 6 3) in partner's suit and 14 support points; the 11 HCP are enough to compete "
           "for the partscore but not to invite game.")
    assert check_strength_claims(f, bad)
    good = ("East bids 3D, a cuebid of the opponents' diamonds, showing a limit raise or better in partner's hearts. "
            "East holds 4 hearts (Q 8 6 3) in partner's suit and 14 support points.")
    assert check_strength_claims(f, good) is None
    assert check_strength_claims(f, good + " This is unusual: the hand has only 11 HCP. "
                                 "The 11 HCP are not enough to invite game.") is None
    assert check_strength_claims(f, good.replace("14 support points", "11 support points"))
    assert check_strength_claims(f, "East bids 3D. The 12 HCP are enough.")
    h = {"bid": "P", "hcp": 7, "hand_str": "S:J98 H:K76 D:Q9876 C:J2"}
    assert check_strength_claims(h, "South passes. With 7 HCP the hand is short of an opening "
                                    "(Rule of 20: 7 + 5 + 3 = 15).") is None
    assert check_strength_claims(h, "South passes (Rule of 20: 7 + 5 + 4 = 16).")
    nt = {"bid": "P", "hcp": 8, "hand_str": "S:J973 H:Q976 D:54 C:AJ3"}
    assert check_strength_claims(nt, "South passes. With only 8 HCP there is no game opposite partner's 15 to 17.")
    assert check_strength_claims(nt, "South passes. With only 8 HCP the partnership has at most 25 opposite "
                                     "partner's 15 to 17, so game is unlikely.") is None
    q = {"bid": "1C", "hcp": 15, "hand_str": "S:AQ H:K843 D:Q72 C:KJ65"}
    assert check_strength_claims(q, "North opens 1C. The hand has 2 quick tricks.")
    assert check_strength_claims(q, "North opens 1C. The hand has 2.5 quick tricks.") is None
    assert check_strength_claims(q, "a 2-3-4-4 distribution")
    assert check_strength_claims(q, "the hand is balanced (2-4-3-4)") is None
    assert check_strength_claims(q, "with a flat 4-4-3-2 shape") is None  # the pattern, longest first
    st = {"bid": "X", "hcp": 9, "hand_str": "S:QJ7 H:K8 D:T9654 C:954"}
    assert check_strength_claims(st, "The hand has only a stopper in clubs.")
    assert check_strength_claims(st, "The hand has no stopper in clubs.") is None
    assert check_strength_claims(st, "the Q J 7 of spades stops the opponents' suit") is None
    assert check_strength_claims(st, "the K 8 of hearts stops the opponents' suit") is None
    assert check_strength_claims({"bid": "4NT", "hcp": 9, "hand_str": "S:QJ7 H:8 D:T9654 C:K954"},
                                 "the K 8 of hearts stops the opponents' suit")


def test_number_coverage_counts_hand_numbers_not_call_meanings():
    from nlg.consistency_checks import number_coverage
    f = {"bid": "2NT", "hcp": 20, "hand_str": "S:AK3 H:KQ5 D:AJ72 C:Q93"}
    c = number_coverage(f, "South opens 2NT, showing a balanced 20 to 21 HCP. South has 20 HCP and "
                           "3 spades (A K 3); the hand is balanced (3-3-4-3).")
    assert c == {"numbers": 6, "verified": 6, "failed": 0}  # 20 HCP, 3 spades, the four shape digits
    c = number_coverage(f, "South opens 2NT. South has 21 HCP.")
    assert c["failed"] == 1
