# The bidding system in the SL data: WBridge5 in SAYC mode

The supervised data (`data/wbridge5/official_*.txt`, the OpenSpiel WBridge5
dataset) was played by WBridge5 with its **SAYC** setting. The concept rule
engine (`concepts/rules.py`) started as textbook SAYC; an audit
(`scripts/audit_concepts.py`) measured where WBridge5's real bids differ and
the rules were changed to match the data. This file records what was
measured, so each rule choice can be traced.

WBridge5 is closed source; there is no official rule base. The only usable
description is a third-party collation of its *native* 2/1 system:
[jdh8's WBridge5 book](https://jdh8.gitbooks.io/wbridge5/)
([source](https://github.com/jdh8/wbridge5.book), CC BY-SA 4.0, 2017). Only
its claims that the data confirms were adopted.

## Measured facts (150k-400k deals of the training data)

The headline rows were re-measured on the whole training split by
`scripts/verify_wbridge5_system.py`.

"Textbook SAYC" below is the common summary of the system that the first rule
engine encoded, **not** the ACBL booklet (`acbl_sayc`), which differs in places:
weak twos 5–11 HCP, a strong 2♣ with "22+ points, or the playing equivalent",
pre-empts defined by playing tricks, no minimum for openings, and a non-forcing
new suit after opener's 1NT rebid (checked against the booklet's text).

| Topic | Data | Textbook SAYC | Rule engine now |
|---|---|---|---|
| Vulnerability | 100% of deals non-vulnerable (OpenSpiel's bridge game has no vulnerability) | varies | frames/walkthrough force non-vulnerable deals for SL checkpoints |
| Weak two length | 6+ cards; **67% of third-seat weak twos have 5** (majors: 22-23% of 2H/2S) | 6 cards | `bid_is_weak_two`: 6+, or 5 in third seat, or 5 with AK/AQJ; `bid_is_light_third_seat_weak_two` |
| Weak two strength | 99.98% have >= 7 points (HCP + shortness + length), 0% have 12+ HCP, 91% have 5-10 HCP (94% 5-11); the textbook definition (a six-card suit and 5-10 HCP) fits only 73%, the booklet's (six cards, 5-11 HCP) 74% (full training split, 91,574 weak twos) | 5-10 HCP (booklet: 5-11) | < 12 HCP and >= 6 points (`wb5_points`) |
| Weak two, 1st/2nd seat | never with 4+ cards in the other major (0 of 6,569) | not specified | (description only) |
| 3-level preempts | 100% have 6+ cards, 55% have 7+ | 7 cards | promise 6+ (`promised_length`) |
| 2C opening | **99.6% have 21+ points or <= 4 losers**; only 30% have 22+ HCP | 22+ HCP, game forcing | `bid_is_strong_2cc_opening`: 22+ HCP or 21+ points or <= 4 losers |
| 1NT / 2NT openings | 95.8% 15-17 / 92% 20-21 HCP | same | unchanged |
| 1C opening | can be 3 cards; 4=4=3=2 opens 1D (100% of 1,488) | 3+ clubs | `promised_length(1C) = 3` |
| 2-over-1 response | 9-17 HCP (median 12) -- not game forcing | 10+, forcing one round | `bid_is_two_over_one_response` (10+ HCP); no longer "game forcing" |
| 4NT reply scheme | **5C = 0 or 3 aces, 5D = 1 or 4, 5H = 2**, 5S essentially unused | 5C = 0/4, 5D = 1, 5H = 2, 5S = 3 | `bid_shows_aces_0_or_3`, `bid_shows_aces_1_or_4`; the standard-scheme names are retired (never true) |
| Roman Keycard | not played (key-card counting does not fit the replies) | optional | `bid_is_roman_keycard` / `bid_is_keycard_response` never true |
| Gerber | no 4C ace ask over 1NT/2NT in 20k deals | 4C over NT | defined strictly (almost never true) |
| Unusual 2NT | jump to 2NT over a 1-level opening: 137/146 hold 5-5 in the two lowest unbid suits | same | only that sequence; 2NT over a weak two is natural (`bid_is_natural_2nt_overcall`, median 17 HCP) |
| Unusual 4NT opening | 90/90 hold 6+ in both minors | not in SAYC | `bid_is_unusual_4nt_opening` |
| Michaels | 217/225 hold two 5-card suits | same | only a direct 2-level cuebid of a 1-level opening |
| 4C/4D openings | natural 8-card preempts (the book's Namyats is not played in SAYC mode) | natural | natural |
| Takeout double of a 1-level opening | 10-19 HCP, median 2 cards in the doubled suit | shortness | doubled bid <= 3 level, doubler holds <= 3 cards there |
| Negative double | 6-13 HCP | 6+ | partner opened, 6+ HCP |
| Opener's reopening double (1C P P 1D X, partner silent, their suit at the 1-3 level) | doubler holds 0-2 cards in their suit 86% of the time at the 1-2 level (8,033 doubles); partner takes it out 80% (1-level) / 71% (2-level) (full training split; a first sample said 80% / 81% / 70%) | takeout, extra values | `bid_is_takeout_double` (was `bid_is_penalty_double`) |
| Cuebids by responder / advancer | 1H 1S 2H 3H, 1C 1S P 4C, 2D 2S 3S: support for partner's suit, 11+ HCP | limit raise or better | `bid_is_cuebid`: up to the 4-level, in the opponents' natural suit (was 1-2 level only, and never when partner had a suit) |
| 5NT king ask replies | 6C = 0, 6D = 1, 6H = 2, 6S = 3, **6NT = 4** kings (66/66) | 6C = 0 or 4 | `bid_answers_king_ask` covers 6C-6S; the 6NT reply is handled in `nlg/explanation_inputs.py` (`_KING_REPLY`); situation `slam_blackwood_answer` (was `slam_bid`) |
| Stopper | -- | A, Kx, Qxx, Jxxx | `concepts.hand.is_stopper` (was "any A/K/Q") |
| Support double: 1x (P or X) 1y/2y (RHO's suit bid) X | opener holds **exactly three** of partner's suit: after LHO's pass, RHO at the 1/2/3 level 86%/97%/80% (of 4,078/4,598/341); after LHO's takeout double, RHO at the 1/2 level 88%/96% (of 679/1,391); any length in RHO's suit, 9-20 HCP; the direct raise holds four. Not after an LHO overcall (60%/19%) | not in SAYC | rules still say `bid_is_penalty_double`; the explanation layer reads it as a support double (`nlg.explanation_inputs.support_double_suit`, teacher meaning, deviation note when not three) |
| 1M (overcall) 3M | preemptive: 5/8/10 HCP (10th/50th/90th percentile), three-card support 38% of 9,073; 1M (P) 3M is a limit raise: 9/10/11 HCP, four-card support 87% | preemptive over interference | teacher: `preemptive_jump_raise` |
| 1x (2z) 3y, a non-jump new suit | forcing: 9/11/15 HCP, 5+ cards 98%, opener passed 7% of 3,911 | forcing | rules: "preemptive intent"; teacher: `suit_forcing_new_suit` |
| 1x (X) 2y | non-jump: 9/11/13 HCP, opener passed 3% (the 2-level new suit shows values); jump: 4/7/9 HCP, 6+ cards 87%, passed 59% -- a weak jump shift | -- | teacher: "a weak jump shift over the double" |
| Advancer's cuebid of the suit partner doubled: (1x) X (…) 2x | after RHO's 1NT/XX: 5/7/10 HCP, two 4+ card unbid suits 95% of 336 -- asks partner to choose; after a pass or a bid: median 10-11 HCP, the doubler names a strain | strong, forcing | teacher: `_takeout_cuebid` ("choose between the unbid suits" / "showing values") |
| Support redouble: 1x (X or P) 1y (X) XX | opener held **exactly three** of partner's suit 290 of 290 times after LHO's double, 3,090 of 3,140 after LHO's pass | optional | teacher: `support_redouble_suit` (the rules: "useful values") |
| Responsive double: (1x) X (2x) X | advancer held four+ cards in two unbid suits 1,604 of 1,604 times, 5/8/10 HCP | responsive | teacher: `responsive_double_suits` |
| High doubles without shape data | a double of a 4-level+ suit contract is left in 76-99% in every shape measured | -- | teacher: penalty reading when `double_stats.json` has no entry |
| After partner's 1NT overcall: (1x) 1NT (P) 2C/2D/2H/2S | **systems on**: 2D held 5+ hearts every time, the overcaller bid 2H 76-85%; 2H likewise for spades; 2C a four-card major; 2S -> 3C 92% -- even when 2D/2H is the opened suit | systems on (optional) | rules: a cuebid / natural; teacher: `nt_overcall_stayman`, `nt_overcall_transfer` (also over a natural 2NT overcall of a weak two: 3C Stayman, 3D/3H transfers -- the rules already read the completion as a transfer) |
| 1NT 2D/2H 2M 3NT | responder held exactly five in the major 6,051 of 6,051 times (9-13 HCP); opener passed every time, even with four-card support | choice of games | teacher: `transfer_3nt` |
| 1NT (P) 2S | a relay: opener bids 3C 1,567 of 1,714 times; responder holds 6+ in a minor, 0-3 spades 91% | minor-suit transfer | teacher: `nt_minor_puppet` (the rules read 2S as natural) |
| 1H/1S (P) 2NT | 13/14/15 HCP at the 10th/50th/90th percentile; three of opener's major 67%, four 25% -- not Jacoby 2NT | Jacoby 2NT (4+ support) | teacher: `major_2nt_response`, balanced 13-15, forcing to game |
| 1NT (P) 4D/4H | never bid (no Texas transfers) | optional | -- |
| 1x (P) 1M (P) 1NT (P) 2m | **natural, not new minor forcing**: responder held 4+ in the bid minor 2,747 of 2,747 times (5+ in the major 97%); opener passed 11-28% | new minor forcing (optional) | the rules' `bid_is_check_back` still fires; the teacher reads it as a natural second suit |
| Fourth suit (1m 1M 1M' 2 of the fourth) | four or more cards only 55% of 5,319 | forcing, artificial | teacher: fourth-suit forcing when the rules say so |
| Doubles in general | per auction shape, how often partner leaves the double in when the next opponent passes (`nlg/double_stats.json`, 716 shapes): double of a 1NT overcall of partner's opening 81% of 1,341; of a 1NT opening 69% of 2,286; of a 4-level jump over partner's opening 86%; of their transfer after 1NT 100% of 136; a takeout double of a 1-level opening 5% of 21,649; negative double 0% of 9,127 | -- | the explanation layer names a double penalty (>= 60%) or competitive (<= 30%) where the rules' category disagrees (`nlg.double_meaning`) |
| Opener's double after both opponents bid: 1x (overcall) 1y/2y (RHO at the 2-3 level) X | partner bids on 94% (passed 47 of 792 at the 2 level, 44 of 383 at the 3); opener 13-14 HCP median, any length in RHO's suit | -- | rules: `bid_is_penalty_double`; explanation layer: a competitive double (`nlg.explanation_inputs.competitive_double_suit`) |
| New suit over an overcall of partner's 1NT: 1NT (2x) 3y | **forcing**: opener passed 7 of 566 times; responder 5+ cards, 10 HCP median (8 at the 10th percentile) | forcing | rules say "preemptive intent" / "competes for the partscore"; the teacher reads it as forcing (`nt_forcing_new_suit`), with a note below ~7 HCP |
| 1NT (2x) 2y | competitive: opener passed 282 of 407 (raised the rest); responder 5+ cards, at most 9 HCP (median 6) | to play | teacher: "a natural, non-forcing bid to compete" |

## Where the policy departs from all of this

`nlg.explanation_inputs.deviation_notes` flags bids that break their situation's own
requirements (after the WBridge5 variants above are accounted for). Measured
false-alarm rate on WBridge5's own bids: 0.26%. On frames from an early
independent checkpoint: 3.7% of decisions, most of them wrong-major openings
that the concepts did not yet cover (see `concepts/vocabulary.py`).
