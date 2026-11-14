"""Rule-based checks of an explanation against its frame, without a model: the call, the action type,
suit lengths and holdings, HCP and other numbers about the hand, roles, conventions, vulnerability, shapes, and
strength against what the call promises. `check_explanation` runs them all; `claim_report` and
`number_coverage` measure what is covered. Deliberately conservative: narrowed after real false positives."""
from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional

from concepts import AuctionContext, HandFeatures, create_sayc_state_authority
from concepts.predicates_auction import is_opening
from explain.rule_justification import render_concept_justification

_OPENS_RE = re.compile(r"\bopen(?:s|ed|ing)?\s+\d")
_DOUBLES_RE = re.compile(r"\bdoubl(?:e|es|ed|ing)\b")
_REDOUBLES_RE = re.compile(r"\bredoubl(?:e|es|ed|ing)\b")
_PASSES_RE = re.compile(r"\bpass(?:es|ed|ing)?\b")
_BIDS_RE = re.compile(r"\bbid(?:s|ding)?\s+\d")


def _headline_clause(text: str) -> str:
    """Every explanation opens with a "<Seat> <verb> ..." headline clause
    stating the actual action, before any causal reasoning that may
    mention OTHER, rejected actions. Checking only this first clause
    avoids false-flagging those mentions."""
    m = re.search(r"[.,;]", text)
    return text[:m.start()] if m else text


def check_bid_consistency(bid: str, explanation: str) -> Optional[str]:
    text = explanation.lower()
    headline = _headline_clause(text)
    opens_claim = bool(_OPENS_RE.search(headline))
    doubles_claim = bool(_DOUBLES_RE.search(headline)) and not _REDOUBLES_RE.search(headline)
    redoubles_claim = bool(_REDOUBLES_RE.search(headline))
    passes_claim = bool(_PASSES_RE.search(headline))
    bids_claim = bool(_BIDS_RE.search(headline))

    if bid in ("P", "Pass"):
        if opens_claim or doubles_claim or redoubles_claim:
            return (f"explanation claims an opening/double/redouble action, "
                    f"but the actual bid was Pass: {explanation!r}")
    elif bid in ("X", "Dbl"):
        if not doubles_claim:
            return (f"explanation doesn't mention doubling, but the actual "
                    f"bid was {bid}: {explanation!r}")
    elif bid in ("XX", "Rdbl", "RDbl"):
        if not redoubles_claim:
            return (f"explanation doesn't mention redoubling, but the actual "
                    f"bid was {bid}: {explanation!r}")
    else:
        if passes_claim and not (opens_claim or bids_claim):
            return (f"explanation describes passing, but the actual bid was "
                    f"{bid}: {explanation!r}")
    return None


def _reconstruct_auction(frame: dict) -> AuctionContext:
    bids = [b.get("bid", "?") for b in (frame.get("auction_before") or [])]
    return AuctionContext(bids=bids, bid=frame.get("bid", "?"), dealer=0)


_CLAIMS_OPENS_RE = re.compile(r"\bopen(?:s|ed|ing)?\b")
_CLAIMS_OVERCALLS_RE = re.compile(r"\bovercall(?:s|ed|ing)?\b")


def check_action_type_consistency(frame: dict, explanation: str) -> Optional[str]:
    bid = frame.get("bid", "?")
    if bid in ("P", "Pass", "X", "Dbl", "XX", "Rdbl", "RDbl"):
        return None
    try:
        a = _reconstruct_auction(frame)
        real_is_opening = is_opening(a)
    except Exception:
        return None

    headline = _headline_clause(explanation.lower())
    claims_opens = bool(_CLAIMS_OPENS_RE.search(headline))
    claims_overcalls = bool(_CLAIMS_OVERCALLS_RE.search(headline))

    if claims_opens and not real_is_opening:
        return (f"explanation's headline claims an opening bid, but this bid "
                f"is not structurally the opening bid of the auction "
                f"(auction_before={frame.get('auction_before')!r}): {explanation!r}")
    if claims_overcalls and real_is_opening:
        return (f"explanation's headline claims an overcall, but this bid IS "
                f"the opening bid of the auction: {explanation!r}")
    return None


_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
}


def _word_or_digit(s: str) -> Optional[int]:
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    return _NUM_WORDS.get(s)


_SUIT_CHAR = {"spade": "S", "heart": "H", "diamond": "D", "club": "C"}
_AT_LEAST_RE = re.compile(r"\bat least (\w+)[- ](?:cards? )?(?:in )?(spade|heart|diamond|club)s?\b")
_NCARD_RE = re.compile(r"\b(\w+)[- ]card (spade|heart|diamond|club)\b")
_LACKS_MAJOR_MINOR_RE = re.compile(r"\b(?:lacks|denies|no|without)(?: an?)? (\w+)[- ]card (major|minor)\b")


def _parse_hand_lengths(hand_str: str) -> Dict[str, int]:
    """Suit lengths from a letter-coded hand_str ("S:QJ2 H:J4 D:6 C:AQJ9742"), the only form
    nlg/explanation_frames.py writes."""
    lengths: Dict[str, int] = {}
    for letter in ("S", "H", "D", "C"):
        m = re.search(letter + r":([-A-Za-z0-9]*)", hand_str)
        if m:
            lengths[letter] = len(m.group(1).replace("-", ""))
    return lengths


def _about_partner(text: str, start: int) -> bool:
    """The claim at text[start:] is in a clause that describes PARTNER's
    call ("partner's 3S promised at least six spades"), not the hand."""
    clause = re.split(r"[.;,]", text[:start])[-1]
    return bool(re.search(r"\bpartner(?:'s)?\b[^.;,]*\b(?:promised|promises|shows|showed|showing)\b", clause))


def check_suit_length_claims(frame: dict, explanation: str) -> Optional[str]:
    hand_str = frame.get("hand_str", "")
    if not hand_str:
        return None
    lengths = _parse_hand_lengths(hand_str)
    if not lengths:
        return None
    text = explanation.lower()

    for m in _AT_LEAST_RE.finditer(text):
        n = _word_or_digit(m.group(1))
        suit = _SUIT_CHAR.get(m.group(2))
        if n is None or suit is None or suit not in lengths or _about_partner(text, m.start()):
            continue
        if lengths[suit] < n:
            return (f"explanation claims 'at least {m.group(1)} {m.group(2)}s' but the "
                    f"real hand has only {lengths[suit]} {m.group(2)}s: {explanation!r}")

    for m in _NCARD_RE.finditer(text):
        n = _word_or_digit(m.group(1))
        suit = _SUIT_CHAR.get(m.group(2))
        if n is None or suit is None or suit not in lengths:
            continue
        if lengths[suit] != n:
            return (f"explanation claims a {m.group(1)}-card {m.group(2)} suit but the "
                    f"real hand has {lengths[suit]}: {explanation!r}")

    for m in _LACKS_MAJOR_MINOR_RE.finditer(text):
        n = _word_or_digit(m.group(1))
        if n is None:
            continue
        suits = ("S", "H") if m.group(2) == "major" else ("D", "C")
        if any(lengths.get(s, 0) >= n for s in suits):
            return (f"explanation claims the hand lacks a {m.group(1)}-card "
                    f"{m.group(2)}, but it has one: {explanation!r}")
    return None


_HOLDING_COUNT_RE = re.compile(r"\b(\w+) (spade|heart|diamond|club)s? \(")
_VOID_RE = re.compile(r"\bvoid in (spade|heart|diamond|club)s?\b|\b(spade|heart|diamond|club) void\b")
_SINGLETON_RE = re.compile(r"\bsingleton (?:in )?(spade|heart|diamond|club)s?\b|\b(spade|heart|diamond|club) singleton\b")
_CARD_SUPPORT_RE = re.compile(r"\b(\w+)[- ]card support\b")
_ACES_RE = re.compile(r"\b(?:holds?|has|holding|with) (?:exactly |only )?(\w+) aces?\b")


def check_hand_claims(frame: dict, explanation: str) -> Optional[str]:
    """Hand-shape claims seen in Qwen3-8B output, all checkable against
    hand_str: "5 hearts (A Q ...)" counts, voids and singletons, "N-card
    support" for partner's last natural suit, and "holds N aces"."""
    hand_str = frame.get("hand_str", "")
    lengths = _parse_hand_lengths(hand_str) if hand_str else {}
    if not lengths:
        return None
    text = explanation.lower()
    for m in _HOLDING_COUNT_RE.finditer(text):
        n, suit = _word_or_digit(m.group(1)), _SUIT_CHAR[m.group(2)]
        if n is not None and lengths.get(suit) != n:
            return f"explanation says {m.group(0)}... but the hand has {lengths.get(suit)}: {explanation!r}"
    for rx, want in ((_VOID_RE, 0), (_SINGLETON_RE, 1)):
        for m in rx.finditer(text):
            suit = _SUIT_CHAR[m.group(1) or m.group(2)]
            if lengths.get(suit) != want:
                return f"explanation claims {m.group(0)!r} but the hand has {lengths.get(suit)}: {explanation!r}"
    table = frame.get("table") or {}
    partner = table.get("partner")
    p_suits = [b["bid"][-1] for b in frame.get("auction_before") or []
               if b.get("seat") == partner and b.get("bid", "")[:1].isdigit() and b["bid"][-1] in "SHDC"]
    for m in _CARD_SUPPORT_RE.finditer(text):
        n = _word_or_digit(m.group(1))
        if n is not None and p_suits and lengths.get(p_suits[-1]) != n:
            return (f"explanation claims {m.group(0)!r} but the hand has {lengths.get(p_suits[-1])} "
                    f"cards in partner's suit: {explanation!r}")
    aces = sum(1 for part in hand_str.split() if "A" in part.split(":")[1])
    for m in _ACES_RE.finditer(text):
        n = _word_or_digit(m.group(1))
        if n is not None and n != aces:
            return f"explanation says the hand {m.group(0)!r} but it holds {aces}: {explanation!r}"
    return None


_STATED_BELOW_RE = re.compile(r"(\d+)\s*hcp[^.]{0,40}?below the (\d+)[- ]point")
_STATED_ATLEAST_RE = re.compile(r"(\d+)\s*hcp[^.]{0,40}?(?:at least|above) the (\d+)[- ]point")
_RANGE_RE = re.compile(r"(?:roughly|about)\s*(\d+)[–‑-](\d+)\s*hcp")


def check_hcp_threshold_claims(hcp, explanation: str) -> Optional[str]:
    if hcp is None:
        return None
    text = explanation.lower()

    m = _STATED_BELOW_RE.search(text)
    if m:
        stated_hcp, threshold = int(m.group(1)), int(m.group(2))
        if stated_hcp >= threshold:
            return (f"explanation says {stated_hcp} HCP is below a {threshold}-point "
                    f"threshold, which is arithmetically false: {explanation!r}")

    m = _STATED_ATLEAST_RE.search(text)
    if m:
        stated_hcp, threshold = int(m.group(1)), int(m.group(2))
        if stated_hcp < threshold:
            return (f"explanation says {stated_hcp} HCP meets/exceeds a {threshold}-point "
                    f"threshold, which is arithmetically false: {explanation!r}")

    m = _RANGE_RE.search(text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if not (lo - 1 <= hcp <= hi + 1):
            return (f"explanation claims a 'roughly {lo}-{hi} HCP' range but the real "
                    f"hand has {hcp} HCP: {explanation!r}")
    return None


_PARTNERS_BID_RE = re.compile(
    r"partner'?s?\s+(?:\d[a-zA-Z]*|pass(?:es)?\b|double\b|redouble\b)"
)
# Deliberately narrow to a concrete bid-string or an unambiguous named call
# right after "partner's": generic words like "bid"/"contract" would
# false-flag the fallback template's rule-engine gloss text.


def check_partner_opponent_attribution(frame: dict, explanation: str) -> Optional[str]:
    text = explanation.lower()
    if not _PARTNERS_BID_RE.search(text):
        return None
    auction_before = frame.get("auction_before") or []
    if not auction_before:
        return None
    # Partner's calls are the ones made 2, 6, 10, ... calls before ours --
    # seat-label-free (the auction's most recent call is never partner's).
    n = len(auction_before)
    partner_calls = [b.get("bid") for i, b in enumerate(auction_before) if (n - i) % 4 == 2]
    norm = lambda b: (b or "").upper().replace(" ", "").replace("NT", "N")  # noqa: E731
    named = {"pass": "P", "passes": "P", "double": "X", "redouble": "XX"}
    for m in _PARTNERS_BID_RE.finditer(text):
        token = m.group(0).split()[-1]
        if token not in named and not re.fullmatch(r"[1-7](?:nt|n|c|d|h|s)", token):
            continue  # "partner's 15 to 17 HCP" is a point range, not a call
        claimed = named.get(token, token)
        if claimed == "P":
            ok = bool(partner_calls) and norm(partner_calls[-1]) == claimed
        elif claimed in ("X", "XX"):
            # "partner's double promised four spades" after partner's X and a
            # later pass is still a true reference
            ok = any(norm(c) == claimed for c in partner_calls)
        else:
            ok = any(norm(c) == norm(claimed) for c in partner_calls)
        if not ok:
            return (f"explanation attributes {token.upper()!r} to partner via a 'partner's ...' "
                    f"phrase, but partner's calls were {partner_calls!r}: {explanation!r}")
    return None


_ROLE_BID_RE = re.compile(
    r"\b(partner|opponents?|lho|rho|you|north|south|east|west)\b(?:'s)?"
    r"(?:\s+(?:has|had|have|just|then|first|already|originally|also))*\s+"
    r"(opened|opens|bid|bids|responded|responds|overcalled|overcalls|rebid|rebids|raised|raises)"
    r"(?:\s+(?:the bidding|with|to|at))*\s+(\d\s*(?:nt|n|c|d|h|s))\b"
)
_SEAT_WORDS = {"north": "N", "south": "S", "east": "E", "west": "W"}


def check_role_bid_claims(frame: dict, explanation: str) -> Optional[str]:
    """Rejects '<who> opened/bid/responded <bid>' claims that don't match the
    real auction -- e.g. "Partner opened 1C" when an opponent opened 1C
    (check_partner_opponent_attribution only covers the "partner's <bid>"
    phrasing). "opened" additionally
    requires <who> to have made the first non-pass call of the auction."""
    table = frame.get("table") or {}
    you, partner = table.get("you"), table.get("partner")
    prior = frame.get("auction_before") or []
    if not you:
        return None
    norm = lambda b: b.upper().replace(" ", "").replace("NT", "N")  # noqa: E731
    first_call = next((b for b in prior if b.get("bid") not in ("P", "Pass")), None)
    for m in _ROLE_BID_RE.finditer(explanation.lower()):
        who, verb, bid = m.group(1), m.group(2), norm(m.group(3))
        if who in ("partner",):
            seats = {partner}
        elif who in ("you",):
            seats = {you}
        elif who in _SEAT_WORDS:
            seats = {_SEAT_WORDS[who]}
        else:  # opponents / lho / rho
            seats = {s for s in "NESW" if s not in (you, partner)}
        made = [b for b in prior if b.get("seat") in seats and norm(b.get("bid", "")) == bid]
        # "<actor> bids X" / "you bid X" may describe the call being explained
        if seats == {you} and norm(frame.get("bid", "")) == bid:
            continue
        if not made:
            return (f"explanation says {who} {verb} {m.group(3).upper()}, but no such call by "
                    f"{who} is in the auction: {explanation!r}")
        if verb.startswith("open") and (first_call is None or first_call.get("seat") not in seats
                                        or norm(first_call.get("bid", "")) != bid):
            opener = f"{first_call.get('seat')} with {first_call.get('bid')}" if first_call else "nobody yet"
            return (f"explanation says {who} opened {m.group(3).upper()}, but the auction was "
                    f"opened by {opener}: {explanation!r}")
    return None


_TAUTOLOGICAL_CLAUSE_RE = re.compile(
    r"^(?:as|since|given that|because)?\s*it(?:'s| is)\s+(?:also\s+)?an?\s+"
    r"(pass|double|redouble)\.?$"
)
# A generated clause like "As it is a pass." restates the headline with
# zero new information. It is common, and redundant rather than wrong, so
# it is stripped instead of rejecting the candidate. Anchored ($) so it only
# fires on a sentence that is JUST the restatement.


def strip_tautological_clauses(explanation: str) -> str:
    """Removes any standalone sentence matching `_TAUTOLOGICAL_CLAUSE_RE`.
    Provably safe to delete outright (adds no information beyond the
    headline), so stripping is strictly better than rejecting the whole
    candidate. Call this on each candidate BEFORE `check_explanation()`
    (see nlg.eval_qwen.explain_checked); `check_tautological_clause` stays in
    `check_explanation`'s aggregator as a defensive backstop for callers
    that don't strip first."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", explanation) if s.strip()]
    kept = [s for s in sentences if not _TAUTOLOGICAL_CLAUSE_RE.match(s.lower())]
    return " ".join(kept)


def check_tautological_clause(bid: str, explanation: str) -> Optional[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", explanation) if s.strip()]
    for s in sentences:
        m = _TAUTOLOGICAL_CLAUSE_RE.match(s.lower())
        if m:
            return (f"explanation contains a tautological clause that just restates "
                    f"the bid type ({m.group(1)!r}) with no new information: {s!r}")
    return None


def check_repeated_sentences(explanation: str) -> Optional[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", explanation) if s.strip()]
    for i in range(len(sentences) - 1):
        ratio = difflib.SequenceMatcher(None, sentences[i].lower(), sentences[i + 1].lower()).ratio()
        if ratio > 0.85:
            return (f"explanation has near-duplicate consecutive sentences "
                    f"(similarity {ratio:.2f}): {sentences[i]!r} / {sentences[i + 1]!r}")
    return None


def render_fallback_explanation(frame: dict, hand: HandFeatures) -> str:
    """A deterministic explanation, consistent by construction: raw facts from the frame plus a
    symbolic justification from explain.rule_justification.render_concept_justification(). Not used to
    explain (that is nlg.teacher and the Qwen explainer); its rule-engine gloss text exercises
    check_explanation's false-positive regressions in tests/test_consistency_checks.py.

    The headline verb is bid-aware (doubles/redoubles/passes/bids): check_bid_consistency requires the
    word "double"/"redouble" for X/XX.
    """
    bid = frame.get("bid", "?")
    seat = (frame.get("table") or {}).get("you", "?")
    hcp = frame.get("hcp", "?")
    hand_str = frame.get("hand_str", "")
    auction_before = frame.get("auction_before") or []

    justification = ""
    try:
        auction = AuctionContext(bids=[b.get("bid", "?") for b in auction_before], bid=bid, dealer=0)
        state_auth = create_sayc_state_authority()
        true_concepts = [n for n, v in state_auth.classify(auction, hand).items() if v]
        if true_concepts:
            justification = " " + render_concept_justification(true_concepts[0], auction, hand)
    except Exception:
        justification = ""

    if bid in ("P", "Pass"):
        verb_clause = f"{seat} passes"
    elif bid in ("X", "Dbl"):
        verb_clause = f"{seat} doubles"
    elif bid in ("XX", "Rdbl", "RDbl"):
        verb_clause = f"{seat} redoubles"
    else:
        verb_clause = f"{seat} bids {bid}"

    return f"{verb_clause}, holding {hcp} HCP ({hand_str}).{justification}"


# what the call being explained is said to BE, in the headline (the first
# sentence), and the calls that can be that ("East bids 2C, a transfer to
# hearts" is wrong: 2C is Stayman).
_NAMED_CALLS = (
    # all as an apposition right after the call ("bids 2C, Stayman, ..."), so
    # mentions of someone else's call ("answering partner's 4NT Blackwood") pass
    (re.compile(r", a transfer to hearts\b"), ("2D", "3D")),
    (re.compile(r", a transfer to spades\b"), ("2H", "3H")),
    (re.compile(r", stayman\b"), ("2C", "3C")),
    (re.compile(r", blackwood\b"), ("4NT", "4N")),
    (re.compile(r", asking partner for kings\b"), ("5NT", "5N")),
    (re.compile(r", an? (?:light third-seat )?weak two\b"), ("2D", "2H", "2S")),
    (re.compile(r", an? (?:reopening )?(?:negative|takeout|penalty) double\b"), ("X", "Dbl")),
    (re.compile(r", showing 0 or 3 aces\b"), ("5C",)),
    (re.compile(r", showing 1 or 4 aces\b"), ("5D",)),
    (re.compile(r", showing 2 aces\b"), ("5H",)),
    (re.compile(r", showing no king\b"), ("6C",)),
    (re.compile(r", showing one king\b"), ("6D",)),
    (re.compile(r", showing two kings\b"), ("6H",)),
    (re.compile(r", showing three kings\b"), ("6S",)),
    (re.compile(r", showing all four kings\b"), ("6NT", "6N")),
)


def check_convention_claims(frame: dict, explanation: str) -> Optional[str]:
    """The headline names a convention the call cannot be (2C called a
    transfer, 5H called "1 or 4 aces", ...)."""
    bid = (frame.get("bid") or "").upper()
    head = explanation.split(". ")[0].lower()
    for rx, calls in _NAMED_CALLS:
        for m in rx.finditer(head):
            if bid not in calls:
                return f"headline calls {bid} '{m.group(0)}', which only {'/'.join(calls)} can be: {explanation!r}"
    return None


_LONGER_RE = re.compile(r"\b(spade|heart|diamond|club)s (?:are|is) longer than (spade|heart|diamond|club)s "
                        r"\((\d+) against (\d+)\)")
_RATHER_RE = re.compile(r"\b(spade|heart|diamond|club)s rather than (spade|heart|diamond|club)s: "
                        r"(\d+) cards? against (\d+)")


def check_length_comparisons(frame: dict, explanation: str) -> Optional[str]:
    """ "clubs are longer than diamonds (5 against 3)" / "Spades rather than
    hearts: 5 cards against 1" -- both counts must match the hand and the
    first must really be longer ("longer (4 against 4)" was generated)."""
    lengths = _parse_hand_lengths(frame.get("hand_str", ""))
    if not lengths:
        return None
    text = explanation.lower()
    for rx in (_LONGER_RE, _RATHER_RE):
        for m in rx.finditer(text):
            a, b = _SUIT_CHAR[m.group(1)], _SUIT_CHAR[m.group(2)]
            na, nb = int(m.group(3)), int(m.group(4))
            if na <= nb or lengths.get(a) != na or lengths.get(b) != nb:
                return (f"length comparison {m.group(0)!r} does not hold for this hand "
                        f"({lengths.get(a)} {m.group(1)}s, {lengths.get(b)} {m.group(2)}s): {explanation!r}")
    return None


def check_vulnerability_claims(frame: dict, explanation: str) -> Optional[str]:
    """"favourable/unfavourable vulnerability", "the opponents are vulnerable",
    "not vulnerable" must match the frame's vulnerability, seen from the bidder."""
    from nlg.explanation_inputs import vulnerability_of
    vul = vulnerability_of(frame)
    text = explanation.lower()
    if "vulnerab" not in text:
        return None
    we, they = vul if vul else (False, False)  # frames without the field are the non-vulnerable SL data
    wrong = ((re.search(r"(?<!un)favou?rable vulnerability", text) and not (not we and they))
             or (re.search(r"\bunfavou?rable vulnerability", text) and not (we and not they))
             or (re.search(r"\bthe opponents are vulnerable", text) and not they)
             or (re.search(r"\bboth sides are vulnerable", text) and not (we and they))
             or (re.search(r"\b(?:being|even though|while|when) vulnerable", text) and not we)
             or (re.search(r"\bnot vulnerable\b", text) and we
                 and not re.search(r"\bthe opponents are not vulnerable", text)))
    if wrong:
        return f"vulnerability claim does not match the deal (we {we}, they {they}): {explanation!r}"
    return None


_BALANCED_CLAIM_RE = re.compile(r"\b(?:the hand is (?:semi-)?balanced|describe a (?:semi-)?balanced hand"
                                r"|with a (?:semi-)?balanced hand)\b")


def check_shape_claims(frame: dict, explanation: str) -> Optional[str]:
    """A hand with a singleton or void is never balanced, not even
    semi-balanced ("rebidding notrump to describe a balanced hand" on a
    4-4-1-4). Statements about what a call SHOWS ("1NT, showing
    a balanced 15 to 17") are not claims about this hand and are not matched."""
    lengths = _parse_hand_lengths(frame.get("hand_str", ""))
    if not lengths:
        return None
    m = _BALANCED_CLAIM_RE.search(explanation.lower())
    if m and min(lengths.values()) <= 1:
        return f"explanation calls a hand with a singleton/void {m.group(0)!r}: {explanation!r}"
    return None


# ---------------------------------------------------------------------------
# Strength, arithmetic and stopper claims: "the 11 HCP", a Rule-of-20 sum,
# "14 support points", and strength that contradicts the call (a 3D cuebid
# "a limit raise or better" with "the 11 HCP are enough to compete for the
# partscore but not to invite game"). Every claim below is
# recomputed from hand_str with the same definitions the rule engine uses
# (concepts/predicates_hand.py), and `claim_report` lists each claim found,
# so the checks' coverage of what an explanation states can be measured.
# ---------------------------------------------------------------------------

_SUIT_W = r"(spade|heart|diamond|club)"
# "<who has/with/the> 11 HCP": the hand's own HCP. Not "showing 20 to 21 HCP",
# "partner's 15 to 17 HCP", "a 15 HCP hand" (a counterfactual) -- the word
# before the number decides.
_HAND_HCP_RE = re.compile(r"\b(with|has|holds|having|holding|and|the|undersell|undersells|of|but|so|since)\s+"
                          r"(?:only\s+|just\s+)?(\d+)\s*hcp\b")
_HCP_POINTS_RE = re.compile(r"\b(\d+)\s*hcp\s*\((\d+) points\)")
_SUPPORT_POINTS_RE = re.compile(r"\b(\d+) support points\b|\bworth (\d+) (?:support )?points counting shortness\b")
_RULE20_RE = re.compile(r"rule of 20[^.;]{0,25}?(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)")
_LOSERS_RE = re.compile(r"\b(?:only )?(\d+) losers\b|\blosing trick count is (\d+)\b")
_QUICK_TRICKS_RE = re.compile(r"\b(\d+(?:\.\d+)?) quick tricks?\b")
_PARTNERSHIP_RE = re.compile(r"\bwith (\d+) hcp opposite partner's (\d+) to (\d+),? the partnership has (\d+) to (\d+)"
                             r"(,? enough for game)?")
_NO_GAME_RE = re.compile(r"\bwith (?:only )?(\d+) hcp there is no game opposite partner's (\d+) to (\d+)")
_AT_MOST_RE = re.compile(r"\bwith (?:only )?(\d+) hcp the partnership has at most (\d+) opposite partner's (\d+) to (\d+)")
_LENGTH_LIST_RE = re.compile(r"(?<![\w-])(\d+|no) (spade|heart|diamond|club)s?\b(?! \()(?!\s+(?:fit|suit|stopper))")
_MAJOR_FIT_RE = re.compile(r"\ba 4-4 major fit is possible\b")
_BW_MATH_RE = re.compile(r"\bwith (\w+)'s (\d+), (?:partner has (\d+), )?(?:so )?the partnership holds (?:only )?(\d+) "
                         r"of the 4 (ace|king)s")
_SHAPE_RE = re.compile(r"(?<![\d-])(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})(?![\d-])")
_HOLDING_OF_RE = re.compile(r"\bthe ((?:[akqjt2-9] )*[akqjt2-9]) of " + _SUIT_W + r"s\b")
# bid_pass_no_extra_values (nlg.teacher._ev_pass_no_extra): the hand against the rule's role minimum
_ROLE_MINIMUM_RE = re.compile(r"\bwith (\d+) hcp \w+ has no more than a minimum (opening|response|overcall|advance|bid) "
                              r"\(at most (\d+)\)")
_ROLE_MAX = {"opening": 14, "response": 9, "overcall": 11, "advance": 11, "bid": 11}
# nlg.teacher._pass_game_context: partner's measured range plus the hand
_PARTNER_RANGE_RE = re.compile(r"\b(?:partner's ([1-7](?:nt|n|c|d|h|s))|the preempt) showed (\d+) to (\d+) hcp, so the partnership has "
                               r"at most about (\d+)(, short of (?:game at the five level|game|slam))?")
# nlg.explanation_inputs.strong_pass_note: WBridge5's measured range for a pass in this exact auction
_STRONG_PASS_RE = re.compile(r"wbridge5 passes here with (\d+) to (\d+) hcp \(the middle 90% of its ([\d,]+) passes "
                             r"in this exact auction\); this hand has (\d+) hcp")
_SHORT_OF = {", short of game": 25, ", short of game at the five level": 29, ", short of slam": 33}
_MAJORS_MINORS_RE = re.compile(r"\b(\d)-(\d) in the majors and (\d)-(\d) in the minors\b")
_HCP_AT_START_RE = re.compile(r"(?:^|\. )(\d+) hcp\b")
_STOPPER_IN_RE = re.compile(r"\b(no|without an?|lacks? an?|a|the|one)?\s*stoppers? in (?:the )?" + _SUIT_W + r"s?\b")
_SUIT_STOPPED_RE = re.compile(r"\b" + _SUIT_W + r"s? (?:are |is )?(?:well |safely )?stopped\b")

# what the CALL promises (headline) against what the HAND is said to be worth.
_MEANS_INVITE_PLUS = re.compile(r"limit raise or better|invitational or better|game[- ]forcing|forcing to game"
                                r"|\bgame force\b|\ba strong\b|\bstrong,|\bjump shift\b|\bslam try\b")
_MEANS_WEAK = re.compile(r"\bweak two\b|\bpreempt|\bweak jump|\bsign-?off\b")
_HAND_BELOW_INVITE = re.compile(r"not (?:enough )?to invite game|\bno game\b|too weak to invite|short of an invitation")
_HAND_GAME_VALUES = re.compile(r"enough to force to game|enough for game|strong enough for a jump shift"
                               r"|game-forcing values")
_HAND_AT_MOST_INVITE = re.compile(r"invitational at most|a minimum response|a weak hand for this call"
                                  r"|enough to invite game but not to insist")
_MEANS_GAME_FORCE = re.compile(r"game[- ]forcing|forcing to game|\bgame force\b|\ba strong\b|\bstrong,")


def _hand(frame: dict) -> Optional[HandFeatures]:
    hs = frame.get("hand_str") or ""
    if not hs:
        return None
    from nlg.explanation_inputs import hand_from_str
    return hand_from_str(hs)


def _clause_before(text: str, start: int) -> str:
    return re.split(r"[.;,]", text[:start])[-1]


def _sentence_at(text: str, start: int) -> str:
    a = max(text.rfind(". ", 0, start), text.rfind("; ", 0, start))
    b = min([i for i in (text.find(". ", start), text.find("; ", start)) if i >= 0] or [len(text)])
    return text[a + 1:b]


def claim_report(frame: dict, explanation: str) -> List[Dict]:
    """Every strength / arithmetic / stopper / shape claim about the hand in
    `explanation`, recomputed from the frame: [{"type", "text", "span", "ok",
    "why"}]. `check_strength_claims` fails on the first claim with ok=False;
    this list is what the coverage measurement (evaluation) counts."""
    from concepts.predicates_hand import losing_trick_count, quick_tricks, support_points, wb5_points
    h = _hand(frame)
    if h is None:
        return []
    if isinstance(frame.get("hcp"), int) and frame["hcp"] != h.hcp:
        h = HandFeatures(hcp=frame["hcp"], lengths=h.lengths, balanced=h.balanced, stoppers=h.stoppers,
                         passed_hand=h.passed_hand, cards=h.cards)  # the frame's own HCP is authoritative
    text = explanation.lower()
    out: List[Dict] = []

    def add(kind, m, ok, why=""):
        out.append({"type": kind, "text": m.group(0), "span": m.span(), "ok": bool(ok), "why": why})

    for m in _HAND_HCP_RE.finditer(text):
        clause = _clause_before(text, m.start())
        if re.search(r"\bpartner\b(?!'s)|\bopener\b|\bresponder\b|\bopponents? (?:has|have|show)", clause):
            continue  # a claim about someone else's hand
        n = int(m.group(2))
        add("hcp", m, n == h.hcp, f"the hand has {h.hcp} HCP")
    for m in _HCP_POINTS_RE.finditer(text):
        add("points", m, int(m.group(2)) == wb5_points(h), f"{wb5_points(h)} points (HCP + distribution + length)")
    for m in _SUPPORT_POINTS_RE.finditer(text):
        n = int(m.group(1) or m.group(2))
        sent = _sentence_at(text, m.start())
        partner = (frame.get("table") or {}).get("partner")
        trumps = {b["bid"][-1] for b in frame.get("auction_before") or []
                  if b.get("seat") == partner and b.get("bid", "")[:1].isdigit() and b["bid"][-1] in "SHDC"}
        # suits named in the sentence, but not "the opponents' diamonds" (a cuebid is no trump suit)
        trumps |= {_SUIT_CHAR[m2.group(1)] for m2 in re.finditer(_SUIT_W, sent)
                   if not re.search(r"(?:opponents'|their)\s+$", sent[:m2.start()])}
        opp_suits = {b["bid"][-1] for b in frame.get("auction_before") or []
                     if b.get("seat") not in (partner, (frame.get("table") or {}).get("you"))
                     and b.get("bid", "")[:1].isdigit()}
        if (frame.get("bid") or "")[-1:] in ("S", "H", "D", "C") and frame["bid"][-1] not in opp_suits:
            trumps.add(frame["bid"][-1])
        valid = {support_points(h, s) for s in (trumps or set("SHDC"))}
        add("support_points", m, n in valid, f"support points {sorted(valid)}")
    for m in _RULE20_RE.finditer(text):
        a, b, c, s = (int(x) for x in m.groups())
        l1, l2 = sorted(h.lengths.values(), reverse=True)[:2]
        ok = a == h.hcp and (b, c) == (l1, l2) and a + b + c == s
        sent = _sentence_at(text, m.start())
        if ok and re.search(r"short of an opening|not met|fails", sent):
            ok = s < 20
        elif ok and re.search(r"\bis (?:just )?met\b|meets", sent):
            ok = s >= 20
        add("rule_of_20", m, ok, f"{h.hcp} + {l1} + {l2} = {h.hcp + l1 + l2}")
    for m in _LOSERS_RE.finditer(text):
        n = int(m.group(1) or m.group(2))
        add("losers", m, n == losing_trick_count(h), f"{losing_trick_count(h)} losers")
    for m in _QUICK_TRICKS_RE.finditer(text):
        add("quick_tricks", m, abs(float(m.group(1)) - quick_tricks(h)) < 1e-6, f"{quick_tricks(h):g} quick tricks")
    for m in _PARTNERSHIP_RE.finditer(text):
        n, a, b, lo, hi = (int(x) for x in m.groups()[:5])
        ok = n == h.hcp and lo == n + a and hi == n + b and (not m.group(6) or lo >= 25)
        add("partnership_sum", m, ok, f"{h.hcp} + {a}..{b} = {h.hcp + a}..{h.hcp + b}")
    for m in _NO_GAME_RE.finditer(text):
        n, a, b = (int(x) for x in m.groups())
        add("no_game", m, n == h.hcp and n + b < 25, f"{h.hcp} + {b} = {h.hcp + b} (game needs 25)")
    for m in _AT_MOST_RE.finditer(text):
        n, tot, a, b = (int(x) for x in m.groups())
        add("partnership_sum", m, n == h.hcp and tot == n + b, f"{h.hcp} + {b} = {h.hcp + b}")
    for m in _SHAPE_RE.finditer(text):
        ls = tuple(int(x) for x in m.groups())
        if sum(ls) != 13:
            continue
        shdc = tuple(h.lengths[s] for s in "SHDC")
        # "a flat 4-3-3-3 shape" is the pattern (longest first), not S-H-D-C
        pattern = ls == tuple(sorted(ls, reverse=True)) and ls == tuple(sorted(shdc, reverse=True))
        add("shape", m, ls == shdc or pattern, "shape " + "-".join(str(x) for x in shdc))
    for m in _LENGTH_LIST_RE.finditer(text):
        clause = _clause_before(text, m.start())
        if re.search(r"\b(?:showing|shows|promis\w*|partner\b(?!'s)|opener\b|at least|or more|they|opponents?\b)",
                     clause) or re.match(r"\s*(?:or more|\+|-card|-level)", text[m.end():]):
            continue
        n = 0 if m.group(1) == "no" else int(m.group(1))
        if n > 13:
            continue
        suit = _SUIT_CHAR[m.group(2)]
        add("length", m, h.lengths[suit] == n, f"{h.lengths[suit]} {SUIT_WORD[suit]}")
    for m in _MAJOR_FIT_RE.finditer(text):
        add("length", m, max(h.lengths["S"], h.lengths["H"]) >= 4, "a four-card major")
    for m in _BW_MATH_RE.finditer(text):
        own, partner_n, total, what = int(m.group(2)), m.group(3), int(m.group(4)), m.group(5)
        mine = sum(1 for s in "SHDC" if ("A" if what == "ace" else "K") in h.cards[s])
        ok = own == mine and (partner_n is None or total == own + int(partner_n)) and total <= 4
        add("blackwood_sum", m, ok, f"the hand holds {mine} {what}s")
    for m in _HOLDING_OF_RE.finditer(text):  # "the K T 5 4 3 of clubs (and ...) stop(s) ..."
        suit = _SUIT_CHAR[m.group(2)]
        cards = m.group(1).upper().split()
        ok = set(cards) <= set(h.cards[suit]) and len(set(cards)) == len(cards)  # "the A K of clubs" (top honours)
        kind = "holding"
        if re.search(r"\bstops?\b", _sentence_at(text, m.start())):
            ok, kind = ok and is_stopper_ranks(h.cards[suit]), "stopper"
        add(kind, m, ok, f"{SUIT_WORD[suit]}: {' '.join(h.cards[suit]) or 'void'}")
    for m in _MAJORS_MINORS_RE.finditer(text):
        a, b, c_, d = (int(x) for x in m.groups())
        ok = sorted((a, b)) == sorted((h.lengths["S"], h.lengths["H"])) and \
            sorted((c_, d)) == sorted((h.lengths["D"], h.lengths["C"]))
        add("length", m, ok, "shape " + "-".join(str(h.lengths[s]) for s in "SHDC"))
    for m in _ROLE_MINIMUM_RE.finditer(text):
        n, role, mx = int(m.group(1)), m.group(2), int(m.group(3))
        add("hcp", m, n == h.hcp and n <= mx and _ROLE_MAX[role] == mx, f"the hand has {h.hcp} HCP; a minimum {role} is at most {_ROLE_MAX[role]}")
    for m in _PARTNER_RANGE_RE.finditer(text):
        prior = frame.get("auction_before") or []
        partner = (frame.get("table") or {}).get("partner")
        call = m.group(1) or next((b.get("bid", "") for b in reversed(prior) if b.get("seat") == partner
                                   and b.get("bid") not in ("P", "X", "XX", "Pass", "Dbl", "RDbl")), "")
        call, lo, hi, tot = call.upper().replace("N", "NT").replace("NTT", "NT"), int(m.group(2)), int(m.group(3)), int(m.group(4))
        ok = tot == h.hcp + hi and (not m.group(5) or tot < _SHORT_OF[m.group(5)])
        idx = [i for i, b in enumerate(prior) if b.get("seat") == partner and b.get("bid", "").upper().replace("N", "NT").replace("NTT", "NT") == call]
        if ok and idx:  # the range is the measured one for partner's call in this auction
            from nlg.call_meaning import wb5_entry
            e = wb5_entry([b["bid"] for b in prior[:idx[-1]]], call)
            ok = bool(e) and (int(e["hcp"][0]), int(e["hcp"][2])) == (lo, hi)
        add("partnership_sum", m, ok and bool(idx), f"{h.hcp} + {hi} = {h.hcp + hi}; partner's {call} as measured")
    for m in _STRONG_PASS_RE.finditer(text):
        from nlg.call_meaning import wb5_entry
        e = wb5_entry([b["bid"] for b in frame.get("auction_before") or []], "P")
        ok = bool(e) and (int(e["hcp"][0]), int(e["hcp"][2]), e["n"]) == (int(m.group(1)), int(m.group(2)),
                                                                         int(m.group(3).replace(",", ""))) \
            and int(m.group(4)) == h.hcp
        add("measured_range", m, ok, "WBridge5's passes in this auction" + (f": {e['hcp'][0]}-{e['hcp'][2]} HCP, n {e['n']}" if e else ": none"))
    for m in _HCP_AT_START_RE.finditer(text):
        add("hcp", m, int(m.group(1)) == h.hcp, f"the hand has {h.hcp} HCP")
    for m in _STOPPER_IN_RE.finditer(text):
        neg = (m.group(1) or "").strip()
        if neg in ("a", "the", "one", "") and re.search(r"\b(?:asks?|asking|looking|look) for\b|\bpartner\b(?!'s)",
                                                        _clause_before(text, m.start())):
            continue  # "asking partner for a stopper in hearts": not a claim about this hand
        suit = _SUIT_CHAR[m.group(2)]
        has = is_stopper_ranks(h.cards[suit])
        ok = (not has) if neg.startswith(("no", "without", "lack")) else has
        add("stopper", m, ok, f"{SUIT_WORD[suit]}: {' '.join(h.cards[suit]) or 'void'}")
    for m in _SUIT_STOPPED_RE.finditer(text):
        if re.search(r"\bpartner\b(?!'s)", _clause_before(text, m.start())):
            continue
        suit = _SUIT_CHAR[m.group(1)]
        add("stopper", m, is_stopper_ranks(h.cards[suit]), f"{SUIT_WORD[suit]}: {' '.join(h.cards[suit]) or 'void'}")

    # the call's promised strength against what the hand is said to be worth
    if "this is unusual" not in text:
        head, rest = (text.split(". ", 1) + [""])[:2]
        for means, hand_rx, label in ((_MEANS_INVITE_PLUS, _HAND_BELOW_INVITE, "invitational or better"),
                                      (_MEANS_GAME_FORCE, _HAND_AT_MOST_INVITE, "game-forcing"),
                                      (_MEANS_WEAK, _HAND_GAME_VALUES, "weak")):
            mm = means.search(head)
            if not mm:
                continue
            for m in hand_rx.finditer(rest):
                add("strength_vs_meaning", m, False,
                    f"the headline says the call is {label} ({mm.group(0)!r}), the hand is called {m.group(0)!r}")
    return out


SUIT_WORD = {"S": "spades", "H": "hearts", "D": "diamonds", "C": "clubs"}


def is_stopper_ranks(ranks) -> bool:
    from concepts.hand import is_stopper
    return is_stopper(list(ranks))


def check_strength_claims(frame: dict, explanation: str) -> Optional[str]:
    """HCP figures, points, support points, Rule-of-20 sums, losers, quick
    tricks, partnership sums, shape strings and stoppers must match the hand,
    and the hand's strength must not contradict what the headline says the
    call promises (unless the explanation flags the call as unusual)."""
    for c in claim_report(frame, explanation):
        if not c["ok"]:
            return f"{c['type']} claim {c['text']!r} does not hold ({c['why']}): {explanation!r}"
    return None


_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")
_HOLDING_PAREN_RE = re.compile(r"\((?:[akqjt2-9]\s?)+\)")
_CALL_TOKEN_RE = re.compile(r"\b[1-7](?:nt|n|c|d|h|s)\b")


def number_coverage(frame: dict, explanation: str, details: bool = False) -> Dict:
    """How many of the numbers an explanation states about the hand are
    verified by some check: {"numbers": n, "verified": k, "failed": f}.
    Card ranks inside holdings "(A K 8 7 5)", call names (2NT) and the
    numbers of a call's promised range ("showing 15 to 17 HCP", "partner's
    15 to 17") are not claims about this hand and are not counted."""
    text = explanation.lower()
    skip = [m.span() for m in _HOLDING_PAREN_RE.finditer(text)] + [m.span() for m in _CALL_TOKEN_RE.finditer(text)]
    skip += [m.span() for m in re.finditer(r"(?:showing|promis\w+|shows|showed|partner's|opener's|of|to)\s+(?:an? |a balanced |"
                                           r"balanced |at least |more than )?\d+(?:\s*(?:to|-|–)\s*\d+)?(?:\+)?\s*"
                                           r"(?:hcp|points|cards?)?", text)
             if not re.search(r"\bthe \d", m.group(0))]
    skip += [m.span() for m in re.finditer(r"rule of 20\b|\b25\b(?= for game)|\beight-card|\b\d+-level|\bthe \d level\b"
                                           r"|\b\d+-\d+ range\b|\b(?:showing|shows) \d+ (?:or \d+ )?(?:aces?|kings?)\b"
                                           r"|\bof the 4 (?:aces|kings)\b|\babout \d+ to \d+\b|\bunder \d+\b|\b\d+\+"
                                           r"|\(at most \d+\)", text)]
    verified = []
    for c in claim_report(frame, explanation):
        verified.append((c["span"], c["ok"]))
    for rx in (_HOLDING_COUNT_RE, _AT_LEAST_RE, _NCARD_RE, _CARD_SUPPORT_RE, _LONGER_RE, _RATHER_RE, _ACES_RE):
        for m in rx.finditer(text):
            verified.append((m.span(), True))
    n = k = f = 0
    unverified = []
    for m in _NUMBER_RE.finditer(text):
        s = m.start()
        if any(a <= s < b for a, b in skip) and not any(a <= s < b for (a, b), _ in verified):
            continue
        n += 1
        hits = [ok for (a, b), ok in verified if a <= s < b]
        if hits:
            k += 1
            f += not all(hits)
        else:
            unverified.append(text[max(0, s - 40):m.end() + 25])
    out = {"numbers": n, "verified": k, "failed": f}
    if details:
        out["unverified"] = unverified
    return out


def check_explanation(frame: dict, explanation: str) -> List[str]:
    """The one entry point callers should use. Each check is wrapped
    defensively -- a malformed/missing frame field reduces detection
    power, not crashes generation."""
    bid = frame.get("bid", "?")
    checks = (
        lambda: check_bid_consistency(bid, explanation),
        lambda: check_action_type_consistency(frame, explanation),
        lambda: check_suit_length_claims(frame, explanation),
        lambda: check_hcp_threshold_claims(frame.get("hcp"), explanation),
        lambda: check_partner_opponent_attribution(frame, explanation),
        lambda: check_role_bid_claims(frame, explanation),
        lambda: check_hand_claims(frame, explanation),
        lambda: check_repeated_sentences(explanation),
        lambda: check_tautological_clause(bid, explanation),
        lambda: check_convention_claims(frame, explanation),
        lambda: check_length_comparisons(frame, explanation),
        lambda: check_vulnerability_claims(frame, explanation),
        lambda: check_shape_claims(frame, explanation),
        lambda: check_strength_claims(frame, explanation),
    )
    warnings: List[str] = []
    for check in checks:
        try:
            w = check()
        except Exception:
            w = None
        if w:
            warnings.append(w)
    return warnings
