"""The 146 concept names (one sigmoid each in a checkpoint's concept head) and their
English glosses. Order is load-bearing: new concepts are only ever appended."""
from __future__ import annotations

SUITS = ("S", "H", "D", "C")
NON_BIDS = ("Pass", "P", "Dbl", "X", "RDbl", "XX")

HARD_STATE_CONCEPTS = [
    # Opening or structural
    "bid_is_1_level_opening",
    "bid_is_2_level_opening",
    "bid_is_3_level_opening",
    "bid_is_4plus_level_opening",
    "bid_is_1nt_opening",
    "bid_is_2nt_opening",
    "bid_is_strong_2cc_opening",
    "bid_denies_major_fit",
    # Bid structure
    "bid_is_1_level_response",
    "bid_is_2_level_response",
    "bid_is_raise_response",
    "bid_is_game_bid",
    "bid_is_opener_rebid",
    "bid_denies_3_card_support",
    # Competitive actions
    "bid_is_overcall",
    "bid_is_double",
    "bid_is_pass",
    "bid_is_cuebid",
    "bid_is_weak_jump_overcall",
    # Length
    "bid_shows_4_plus_cards",
    "bid_shows_5_plus_cards",
    "bid_shows_6_plus_cards",
    "bid_shows_7_plus_cards",
    "bid_shows_exactly_5_cards",
    "bid_shows_exactly_6_cards",
    "bid_shows_extra_length_on_rebid",
    "bid_shows_extra_length_on_jump_raise_after_nt",
    # Suit / NT
    "bid_shows_major_suit",
    "bid_shows_minor_suit",
    "bid_shows_notrump",
    # Strength
    "bid_shows_weak_values",
    "bid_shows_minimum_values",
    "bid_shows_invitational_values",
    "bid_shows_game_forcing_values",
    "bid_shows_extra_values",
    # Fit checks
    "bid_shows_support",
    "bid_shows_3_card_support",
    "bid_shows_4_plus_support",
    "bid_promises_8_card_fit",
    "bid_denies_fit",
    "bid_shows_minor_suit_only",
    "bid_completes_jacoby_transfer",
    # Limiting
    "bid_limits_to_minimum",
    "bid_limits_to_invitational",
    "bid_limits_to_game_values",
    "bid_limits_hand",
    # Invitationals
    "bid_shows_slam_interest",
    "bid_is_sacrifice_candidate",
    "bid_shows_preemptive_intent",
    "bid_shows_competitive_intent",
    "bid_shows_constructive_intent",
    # Rebids or jumps
    "bid_is_jump_bid",
    "bid_is_rebid",
    "bid_is_competitive_jump",
    "bid_shows_jump_shift_strength",
    # Very artificial
    "bid_is_conventional",
    "bid_is_stayman",
    "bid_is_jacoby_transfer",
    "bid_is_fourth_suit_forcing",
    "bid_is_splinter",
    "bid_is_void_splinter",
    "bid_is_singleton_splinter",
    # Slams
    "bid_is_gerber",
    "bid_is_gerber_response",
    "bid_is_roman_keycard",
    "bid_is_keycard_response",
    "bid_is_blackwood",
    "bid_is_blackwood_response",
    "bid_shows_aces_0_or_4",
    "bid_shows_aces_1",
    "bid_shows_aces_2",
    "bid_shows_aces_3",
    "bid_shows_stayman_major_structure",
    # Artificial / forcing
    "bid_is_waiting_response",
    "bid_is_check_back",
    "bid_is_weak_two",
    # Invitational
    "bid_is_pass_or_cuebid",
    "bid_is_quantitative_invite",
    "bid_is_qualitative_invite",
    "bid_is_negative_invite",
    # Competitive conventions
    "bid_is_michaels_cuebid",
    "bid_is_michaels_response",
    "bid_is_unusual_2nt",
    "bid_is_negative_double",
    "bid_is_penalty_double",
    "bid_is_takeout_double",
    # Control or some cue-bidding
    "bid_is_control_bid",
    "bid_shows_control",
    "bid_shows_first_control",
    "bid_shows_second_control",
    # Defensive
    "bid_shows_stopper_in_their_suit",
    "bid_competes_for_partscore",
    "bid_shows_defensive_values",
    "bid_shows_5_card_major_after_strong_2c",
    # Shape
    "bid_denies_major_fit_after_5_card_major",
    "bid_shows_single_suited",
    "bid_denies_shortness",
    # 1NT / 5-card major
    "bid_shows_5_card_major_in_1nt",
    # Raise
    "bid_is_simple_game_raise",
    "bid_is_opener_support_raise",
    # Redouble
    "bid_is_redouble",
    # 2NT opening system
    "bid_is_2nt_stayman",
    "bid_completes_2nt_jacoby_transfer",
    # Minor suit fit
    "bid_shows_minor_fit_raise",
    # Stayman game fit raise
    "bid_is_stayman_game_fit_raise",
    # Hand evaluation metrics
    "bid_opening_meets_rule_of_20",
    "bid_opening_meets_ltc_standard",
    "bid_shows_quick_tricks_2plus",
    "bid_shows_support_points_invite_plus",
    "bid_is_preempt_quality_good",
    "bid_is_preempt_quality_marginal",
    # Suit choice among alternatives: why THIS strain and not another that
    # qualified (e.g. 1D and not 1C); see predicates_hand.py.
    "bid_shows_longer_minor",
    "bid_shows_longer_major",
    "bid_prefers_notrump_over_major",
    "bid_is_reverse",
    # WBridge5 system concepts (scripts/audit_concepts.py,
    # notes/wbridge5_system.md): its Blackwood replies (5C = 0/3 aces,
    # 5D = 1/4), WBridge5-specific openings/overcalls, 2-over-1 responses,
    # raises of partner's overcall, and hand-dependent reasons for passes
    # (57% of all decisions).
    "bid_shows_aces_0_or_3",
    "bid_shows_aces_1_or_4",
    "bid_is_light_third_seat_weak_two",
    "bid_is_unusual_4nt_opening",
    "bid_is_natural_2nt_overcall",
    "bid_is_two_over_one_response",
    "bid_raises_partners_overcall",
    "bid_pass_lacks_opening_values",
    "bid_pass_lacks_values_to_compete",
    "bid_pass_accepts_partners_contract",
    "bid_pass_too_weak_to_respond",
    "bid_pass_no_extra_values",
    "bid_stayman_with_4_card_major",
    "bid_transfer_with_5_card_major",
    "bid_double_with_length_in_their_suit",
    # Strain of the call. Without these no concept separates 1H from 1S, so
    # the linear concepts->bid head of the true-bottleneck modes cannot
    # choose the suit. Decided by the call itself, so the explanation layer
    # never offers them as reasons (nlg.explanation_inputs filters
    # hand-independent concepts).
    "bid_is_in_clubs",
    "bid_is_in_diamonds",
    "bid_is_in_hearts",
    "bid_is_in_spades",
    "bid_is_in_notrump",
    # Rare-situation reasons: slam decisions, passes of partner's game/slam,
    # 2C responses, weak jump responses
    "bid_pass_content_with_final_contract",
    "bid_slam_with_enough_aces",
    "bid_answers_king_ask",
    "bid_is_preemptive_jump_response",
    "bid_blackwood_with_fit",
    "bid_waiting_response_without_positive",
    "bid_positive_response_to_2c",
    "bid_completes_transfer_without_super_accept",
    "bid_king_ask_with_all_aces",
    "bid_small_slam_missing_a_king",
    "bid_is_preemptive_raise",
]

SAYC_CONCEPTS = {
    "hard_state": HARD_STATE_CONCEPTS,
}


# ---------------------------------------------------------------- glosses
# Each concept as a verb phrase ("is a 1-level opening"); names without an entry are derived from the name.
_CONCEPT_GLOSSES = {
    "bid_is_strong_2cc_opening": "is a strong, artificial 2C opening (about 21+ points or a 4-loser hand)",
    "bid_is_cuebid": "is a cuebid of the opponents' suit",
    "bid_is_weak_jump_overcall": "is a weak jump overcall showing a long suit and a weak hand",
    "bid_completes_jacoby_transfer": "completes a Jacoby transfer, accepting the transfer suit as trumps",
    "bid_is_conventional": "is an artificial (conventional) bid, not a natural suit/NT bid",
    "bid_is_stayman": "is Stayman, asking opener for a 4-card major",
    "bid_is_jacoby_transfer": "is a Jacoby transfer, asking opener to bid the next suit up as trumps",
    "bid_is_fourth_suit_forcing": "is fourth-suit-forcing, an artificial game-forcing relay",
    "bid_is_splinter": "is a splinter, a jump showing a singleton or void with support and slam interest",
    "bid_is_void_splinter": "is a splinter specifically showing a void",
    "bid_is_singleton_splinter": "is a splinter specifically showing a singleton",
    "bid_is_gerber": "is Gerber, a 4C ace ask directly over partner's notrump opening",
    "bid_is_gerber_response": "is a Gerber ace-count response",
    "bid_is_roman_keycard": "is Roman Keycard Blackwood (not played in this system; never true)",
    "bid_is_keycard_response": "is a Roman Keycard reply (not played in this system; never true)",
    "bid_is_blackwood": "is Blackwood, an artificial ace-asking convention",
    "bid_is_blackwood_response": "is a reply to partner's Blackwood 4NT, showing the number of aces",
    "bid_shows_aces_0_or_4": "shows 0 or 4 aces (standard scheme, not this system's; never true)",
    "bid_shows_aces_1": "shows exactly 1 ace (standard scheme, not this system's; never true)",
    "bid_shows_aces_2": "shows 2 aces (the 5H reply to Blackwood)",
    "bid_shows_aces_3": "shows 3 aces (the 5S reply to Blackwood)",
    "bid_shows_stayman_major_structure": "shows opener's major-suit holding after a Stayman inquiry",
    "bid_is_waiting_response": "is an artificial waiting response with no specific shape/strength shown",
    "bid_is_check_back": "is a check-back (new minor forcing) bid asking opener to describe further",
    "bid_is_weak_two": "is a weak two opening: a long suit (6 cards, or 5 in third seat) with under 12 HCP",
    "bid_is_pass_or_cuebid": "passes over an opponent's interference of partner's 1NT while holding a stopper in their suit",
    "bid_is_quantitative_invite": "is a quantitative 4NT raise of partner's notrump, inviting slam on high cards",
    "bid_is_qualitative_invite": "is a qualitative (shape-based) invitational raise",
    "bid_is_negative_invite": "is an invitational bid showing a weaker-than-expected hand",
    "bid_is_michaels_cuebid": "is a Michaels cuebid of the opening suit, showing a two-suited hand",
    "bid_is_michaels_response": "is a response to a Michaels cuebid",
    "bid_is_unusual_2nt": "is the Unusual 2NT, a jump over a one-level opening showing the two lowest unbid suits",
    "bid_is_negative_double": "is a negative double after partner opened and an opponent overcalled, showing values and the unbid suits",
    "bid_is_penalty_double": "is a penalty double, expecting to defeat the doubled contract",
    "bid_is_takeout_double": "is a takeout double of a low-level suit bid, asking partner to bid an unbid suit",
    "bid_is_control_bid": "is a control-showing cuebid on the way to a slam decision",
    "bid_shows_control": "shows first- or second-round control of the suit bid, as a slam try",
    "bid_shows_first_control": "shows first-round control (ace or void) of the suit bid, as a slam try",
    "bid_shows_second_control": "shows second-round control (king or singleton) of the suit bid, as a slam try",
    "bid_shows_stopper_in_their_suit": "bids notrump with a stopper in every suit the opponents have bid",
    "bid_competes_for_partscore": "competes for the partscore in a contested auction",
    "bid_shows_defensive_values": "shows defensive values rather than offensive shape",
    "bid_shows_5_card_major_after_strong_2c": "shows a 5-card major after a strong 2C opening",
    "bid_denies_major_fit_after_5_card_major": "denies a fit for partner's 5-card major",
    "bid_shows_single_suited": "shows a single long suit",
    "bid_denies_shortness": "denies a singleton or void",
    "bid_shows_5_card_major_in_1nt": "shows a 5-card major within a 1NT-opening structure",
    "bid_is_simple_game_raise": "is a simple raise straight to game",
    "bid_is_opener_support_raise": "is opener's raise showing support for responder's suit",
    "bid_is_redouble": "redoubles to show useful values (7+ HCP) after the opponents' double",
    "bid_is_2nt_stayman": "is Stayman over a 2NT opening",
    "bid_completes_2nt_jacoby_transfer": "completes a Jacoby transfer over a 2NT opening",
    "bid_shows_minor_fit_raise": "raises showing a minor-suit fit",
    "bid_is_stayman_game_fit_raise": "is a game-forcing raise found via Stayman",
    "bid_promises_8_card_fit": "promises at least an 8-card combined trump fit",
    "bid_shows_minor_suit_only": "shows a minor suit only, denying a major",
    "bid_shows_jump_shift_strength": "shows extra strength via a jump shift",
    "bid_is_pass": "passes, declining to bid further",
    "bid_is_double": "doubles the opponents' bid or contract",
    "bid_opening_meets_rule_of_20": "is a 1-level opening that satisfies the Rule of 20",
    "bid_opening_meets_ltc_standard": "is a 1-level opening with a standard (<=7) Losing Trick Count",
    "bid_shows_quick_tricks_2plus": "opens or overcalls with at least 2 quick tricks",
    "bid_shows_support_points_invite_plus": "shows at least invitational support points (HCP plus shortness) for partner's suit",
    "bid_is_preempt_quality_good": "is a preempt with good suit quality (2+ top honors)",
    "bid_is_preempt_quality_marginal": "is a preempt with marginal suit quality (fewer than 2 top honors)",
    "bid_shows_longer_minor": "opens the longer minor suit (or, when tied, follows the standard 3-3-opens-clubs / 4-4-or-more-opens-diamonds convention)",
    "bid_shows_longer_major": "opens the longer major suit (or, when tied, opens spades)",
    "bid_prefers_notrump_over_major": "opens 1NT on a balanced 15-17 hand even though a 5-card major also qualified, by standard NT-priority convention",
    "bid_is_reverse": "is a reverse -- a rebid in a new, higher-ranking suit that forces partner to the 3-level to return to the first suit, promising extra strength",
    "bid_shows_3_card_support": "raises partner's suit one level with at least 3-card support",
    "bid_shows_4_plus_support": "raises partner's suit with at least 4-card support",
    "bid_shows_4_plus_cards": "bids a natural suit of 4 or more cards",
    "bid_shows_5_plus_cards": "bids a natural suit of 5 or more cards",
    "bid_shows_6_plus_cards": "bids a suit of 6 or more cards",
    "bid_shows_7_plus_cards": "bids a suit of 7 or more cards",
    "bid_shows_exactly_5_cards": "bids a suit of exactly 5 cards",
    "bid_shows_exactly_6_cards": "bids a suit of exactly 6 cards",
    "bid_shows_game_forcing_values": "shows game-forcing strength (about 12+ HCP) with a bid that commits to game",
    "bid_shows_extra_values": "shows extra strength beyond the minimum for this position",
    "bid_shows_invitational_values": "shows invitational strength (about 10-12 points)",
    "bid_is_game_bid": "bids a game contract (3NT, 4H, 4S, 5C or 5D)",
    "bid_is_opener_rebid": "is opener's rebid of the suit opened",
    "bid_shows_slam_interest": "makes a slam try (ace or king ask, control cuebid, or a five-level invitation)",
    "bid_is_sacrifice_candidate": "is a high-level competitive bid with a weak hand, possibly a sacrifice",
    "bid_shows_aces_0_or_3": "shows 0 or 3 aces (the 5C reply to Blackwood in this system)",
    "bid_shows_aces_1_or_4": "shows 1 or 4 aces (the 5D reply to Blackwood in this system)",
    "bid_is_light_third_seat_weak_two": "is a light third-seat weak two on a 5-card suit",
    "bid_is_unusual_4nt_opening": "is the Unusual 4NT opening, showing at least 6-6 in the minors",
    "bid_is_natural_2nt_overcall": "is a natural 2NT overcall: about 15-18 HCP with the opponents' suit stopped",
    "bid_is_two_over_one_response": "is a two-over-one response: a new suit at the two level showing 10+ HCP",
    "bid_raises_partners_overcall": "raises partner's overcall with support for the suit",
    "bid_pass_lacks_opening_values": "passes in opening position without the strength to open (under 12 HCP and the Rule of 20 not met)",
    "bid_pass_lacks_values_to_compete": "passes over the opponents' bidding without the strength, a long suit, or a fit with partner to compete",
    "bid_pass_accepts_partners_contract": "passes, leaving partner in the contract partner chose",
    "bid_pass_too_weak_to_respond": "passes partner's opening without the values (or, over an overcall, the fit) to respond",
    "bid_pass_no_extra_values": "passes with nothing extra beyond what the earlier bids already showed",
    "bid_stayman_with_4_card_major": "uses Stayman holding a 4-card major, looking for a 4-4 major fit",
    "bid_transfer_with_5_card_major": "transfers holding 5 or more cards in the major above the transfer suit",
    "bid_double_with_length_in_their_suit": "doubles holding length (3+ cards) in the suit the opponents just bid",
    "bid_is_in_clubs": "is a bid in clubs",
    "bid_is_in_diamonds": "is a bid in diamonds",
    "bid_is_in_hearts": "is a bid in hearts",
    "bid_is_in_spades": "is a bid in spades",
    "bid_is_in_notrump": "is a bid in notrump",
    "bid_pass_content_with_final_contract": "passes partner's game or slam, without the strength to go further",
    "bid_slam_with_enough_aces": "bids slam after Blackwood, the partnership holding at least three aces",
    "bid_answers_king_ask": "answers partner's 5NT king ask with the number of kings held (6C = 0, 6D = 1, 6H = 2, 6S = 3)",
    "bid_is_preemptive_jump_response": "is a weak jump response in a long suit over the opponents' interference",
    "bid_blackwood_with_fit": "asks for aces with Blackwood holding support for the agreed suit",
    "bid_waiting_response_without_positive": "makes the 2D waiting response to 2C without the values or suit for a positive response",
    "bid_positive_response_to_2c": "makes a positive response to partner's 2C: a good 5-card suit (or 2NT) and 7+ HCP",
    "bid_completes_transfer_without_super_accept": "completes partner's transfer at the cheapest level, without the 4-card support and maximum needed to jump",
    "bid_king_ask_with_all_aces": "asks for kings with 5NT, the partnership holding all four aces",
    "bid_small_slam_missing_a_king": "settles in a small slam because the partnership is missing a king",
    "bid_is_preemptive_raise": "raises partner's suit high in competition with a weak hand and long support, to take away bidding room",
}


def gloss_for_concept(concept: str) -> str:
    """The concept as a short verb phrase ("shows constructive intent",
    "is a 1-level opening") so it reads as part of a sentence."""
    if concept in _CONCEPT_GLOSSES:
        return _CONCEPT_GLOSSES[concept]
    if concept.startswith("bid_shows_"):
        return "shows " + concept[len("bid_shows_"):].replace("_", " ")
    if concept.startswith("bid_is_"):
        # Derived noun phrases start with a letter, never a digit, so a
        # first-letter vowel check gives the right "a"/"an".
        noun_phrase = concept[len("bid_is_"):].replace("_", " ")
        article = "an" if noun_phrase[:1].lower() in "aeiou" else "a"
        return f"is {article} " + noun_phrase
    if concept.startswith("bid_denies_"):
        return "denies " + concept[len("bid_denies_"):].replace("_", " ")
    if concept.startswith("bid_limits_"):
        return "limits " + concept[len("bid_limits_"):].replace("_", " ")
    if concept.startswith("bid_completes_"):
        return "completes " + concept[len("bid_completes_"):].replace("_", " ")
    if concept.startswith("bid_promises_"):
        return "promises " + concept[len("bid_promises_"):].replace("_", " ")
    return concept.replace("bid_", "").replace("_", " ")


# The named predicates of predicates_auction.py / predicates_hand.py as plain present-tense clauses, for
# explain/rule_justification.py; each must also read after "it is not the case that". Unlisted predicates render as source.
PREDICATE_GLOSSES = {
    "partnership_has_bid": "the partnership has already made a bid this auction",
    "opponents_have_bid": "the opponents have already bid",
    "partner_has_suit": "partner's last bid named a suit",
    "uncontested": "the opponents have not bid at all",
    "is_opening": "this is the opening bid",
    "is_response": "this is the bidder's first bid after partner opened (a response)",
    "in_opening_position": "nobody has bid yet",
    "is_advance": "this is the bidder's first bid after partner overcalled the opponents' opening",
    "i_opened_auction": "the bidder opened the auction",
    "_opponents_opened": "the opponents opened the auction",
    "_actor_only_passed": "the bidder has only passed so far",
    "_actor_has_real_bid": "the bidder has already made a bid",
    "_is_blackwood_ask": "this is a Blackwood 4NT ace ask",
    "_is_blackwood_reply": "this is a reply to partner's Blackwood 4NT",
    "_control_bid": "this is a new-suit bid at the 3+ level after partner's suit bid (a control cuebid)",
    "_wb5_can_open": "the hand has opening strength (12+ HCP, 11 HCP with 13+ points, or the Rule of 20)",
    "is_preemptive_opening": "this is a preemptive opening bid",
    "is_preemptive_opening_continuation": "this continues an earlier preemptive opening",
    "is_partnership_opener": "the bidder opened for the partnership",
    "is_overcall": "this is an overcall of the opponents' opening",
    "is_jump_overcall": "this is a jump overcall",
    "is_response_to_weak_two": "this responds to partner's weak two-bid",
    "is_stayman_sequence": "this is a Stayman sequence",
    "is_jacoby_transfer_sequence": "this is a Jacoby transfer sequence",
    "is_minor_suit_transfer_completion": "this completes a minor-suit transfer",
    "is_strong_2c_waiting_sequence": "this is a waiting bid after a strong 2C opening",
    "is_new_minor_forcing_sequence": "this is a new-minor-forcing sequence",
    "is_jump_rebid_same_suit": "this is a jump rebid in the same suit",
    "is_jump_raise_after_nt": "this is a jump raise made directly over partner's notrump",
    "is_reverse_bid": "this is a reverse bid",
    "has_singleton": "the hand has a singleton",
    "has_void": "the hand has a void",
    "is_semi_balanced": "the hand is semi-balanced (5-4-2-2, 6-3-2-2, or 5-3-3-2)",
    "denies_major_fit_after_1nt_response": "this denies a major-suit fit by rebidding notrump",
    "rule_of_20_satisfied": "the hand satisfies the Rule of 20 (HCP plus the two longest suits is at least 20)",
    "opens_longer_minor": "the bid opens the hand's longer minor (1C with 3-3 in the minors, 1D with 4-4 or 5-5)",
    "opens_longer_major": "the bid opens the longer of two 5-card majors (1S when they are equally long)",
    "prefers_notrump_over_major": "the bid opens 1NT with a balanced or semi-balanced 15-17 HCP hand that also "
                                  "holds a 5-card major",
}
