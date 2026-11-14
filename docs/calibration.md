# Calibrating the rules to WBridge5

The rule engine (`concepts/rules.py`) describes the system that WBridge5 plays in the supervised data, its SAYC
setting. WBridge5 is closed source, so the first rules were written from the
[ACBL SAYC booklet](https://web2.acbl.org/documentlibrary/play/SP3(SAYC).pdf) and then corrected against
WBridge5's own bids, which the data show together with all four hands. This file sums up what the corrections
were. The measurements behind each rule are in [`notes/wbridge5_system.md`](../notes/wbridge5_system.md).

## What changed

The first rules often did not describe WBridge5. For example, the 2♣ opening rule (22+ HCP) fitted only 31% of
WBridge5's 2♣ openings, a pass was counted as an opening, and a bare king was counted as a stopper. The changes
fall into seven kinds. After them, the audit (`scripts/audit_concepts.py`) finds no hand that contradicts any
length, support, stopper, ace-count or control concept.

| Kind of difference | Examples |
|---|---|
| Same convention, other limits | weak twos; the strong 2♣; three-level pre-empts from six cards |
| Convention not played | Roman Keycard; Gerber |
| Different scheme | Blackwood replies: 5♣ = 0 or 3 aces, 5♦ = 1 or 4 |
| Right idea, wrong auctions | a pass is not an opening; a reply to partner's overcall is not a response |
| The hand must hold what the bid promises | suit lengths; stoppers (A, Kx, Qxx, Jxxx) |
| A missing reason | reasons for passes; raising partner's overcall |
| Meaning set by partner's answer | doubles named by how often partner leaves them in; support doubles |

The last kind is handled by the explanation layer, not the rules. Renaming a concept, for example from "penalty
double" to "support double", would mean labelling the 12.6 million training decisions again and retraining every
model. The explanation layer instead recognises the sequence, states WBridge5's meaning, and adds a note to the
misleading concept name (`nlg/double_meaning.py`).

The calibrated rules describe WBridge5 closely. Only 0.43% of WBridge5's own non-pass test bids get a deviation
note, and the concept layers learn the rules to a macro-F1 of 0.92–0.94 on WBridge5's held-out bids.

## Measured practices

The practices that changed the rules most, measured on the whole training split (1,138,441 deals;
`scripts/verify_wbridge5_system.py`). *n* is the number of cases.

| Practice | Booklet | WBridge5 |
|---|---|---|
| Weak two | six cards, 5–11 HCP | 5–11 HCP in 94%; in third seat 65% hold only five cards (*n* = 17,296) |
| Strong 2♣ | 22+ points | only 31% have 22+ HCP; 99% have 22+ HCP, 21+ points or at most four losers (*n* = 48,080) |
| Three-level pre-empt | within two to four tricks of the bid | always six or more cards, seven or more in 56% (*n* = 37,207) |
| Two-over-one | 10+ points, forcing one round | usually 9–18 HCP, not game forcing (*n* = 63,238) |
| Blackwood replies | 5♣ = 0 or 4 aces, 5♦ = 1 | 5♣ = 0 or 3 (7,815 with three), 5♦ = 1 or 4 (898 with four) |
| Gerber | 4♣ over 1NT or 2NT | responder never bids 4♣ directly over 1NT or 2NT (0 of 132,088) |
| Support double | – | opener holds exactly three of partner's suit in 81–97% (*n* = 10,266) |

Some of these limits count *points* rather than HCP: the HCP plus three for a void, two for a singleton, one for
a doubleton, and one for every card beyond the fifth in a suit. This combination was found in an unofficial
collation of WBridge5's native system ([jdh8's WBridge5 book](https://jdh8.gitbooks.io/wbridge5/)) and kept
because the data confirm it: with points, WBridge5's strong 2♣ and its weak twos are described almost exactly
(99.45% and 99.98%), which HCP alone do not do.
