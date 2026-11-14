"""Why a concept holds, in English, generated from the rule engine's own predicate source: each
predicate lambda's AST is re-evaluated and rendered with templates. Shapes without a template fall back to the
literal source and its value, so the text is never invented. `rule_conditions` renders every condition of a
concept's rule, holding or not, for the interface."""
from __future__ import annotations

import ast
import inspect
import linecache
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

from concepts.vocabulary import PREDICATE_GLOSSES, gloss_for_concept as _gloss_for_concept
from concepts.rules import StateRule, build_sayc_state_rules
from concepts.types import AuctionContext, HandFeatures

_OP_PHRASES = {
    ast.Lt: "is less than",
    ast.LtE: "is at most",
    ast.Gt: "is more than",
    ast.GtE: "is at least",
    ast.Eq: "is exactly",
    ast.NotEq: "is not",
    ast.In: "is one of",
    ast.NotIn: "is not one of",
    ast.Is: "is",
    ast.IsNot: "is not",
}

# sub-expressions as noun phrases, keyed by their source (canonical form: _code)
_EXPR_LABELS = {
    "_suit_len(a, h)": "the hand's length in the bid suit",
    "h.lengths.get(a.bid[-1], 0)": "the hand's length in the bid suit",
    "h.cards.get(a.bid[-1], [])": "the hand's cards in the bid suit",
    "set(h.cards.get(a.bid[-1], []))": "the hand's cards in the bid suit",
    "_fit_with_partner(a, h)": "the hand's length in partner's suit",
    "h.lengths.get(a.partner_last_bid()[-1], 0)": "the hand's length in partner's suit",
    "h.lengths.get(_last_real_bid(a)[-1], 0)": "the hand's length in the suit of the last bid at the table",
    'h.lengths.get((_last_real_bid(a) or "?")[-1], 9)': "the hand's length in the suit of the last bid at the table",
    "h.lengths.get(_agreed_suit(a), 0)": "the hand's length in the agreed suit",
    'max(h.lengths.get("H", 0), h.lengths.get("S", 0))': "the hand's length in its longer major",
    'h.lengths.get("C", 0) + h.lengths.get("D", 0)': "the hand's clubs and diamonds together",
    "min(h.lengths.values())": "the hand's shortest suit",
    "wb5_points(h)": "the hand's points (HCP plus shape)",
    "opening_seat_position(a)": "the opening seat (1 = first seat)",
    "_last_real_bid(a)": "the last bid made at the table",
    "_role_minimum(a)": "the upper limit of a minimum hand for the bidder's role",
    'sum("A" in h.cards.get(s_, []) for s_ in SUITS)': "the number of aces held",
    "ACES(h)": "the number of aces held",
    "KINGS(h)": "the number of kings held",
    "_partner_aces_from_reply(a, ACES(h))": "partner's aces, from partner's Blackwood reply",
    "ACES(h) + _partner_aces_from_reply(a, ACES(h))": "the number of aces the partnership holds",
    '{"6C": 0, "6D": 1, "6H": 2, "6S": 3}.get(a.bid)': "the kings the bid shows (6C 0, 6D 1, 6H 2, 6S 3)",
    'KINGS(h) + {"6C": 0, "6D": 1, "6H": 2, "6S": 3}[a.partner_last_bid()]':
        "the number of kings the partnership holds",
    'h.lengths.get("H" if a.bid[-1] == "D" else "S", 0)': "the hand's length in the major the transfer shows",
    "h.hcp": "the hand's HCP",
    "h.balanced": "the hand's balanced-shape flag",
    "a.bid": "the bid",
    "a.bid[0]": "the bid's level",
    "a.bid[-1]": "the bid's suit",
    "a.bid[-1:]": "the bid's suit",
    "len(a.bid)": "the length of the bid's name",
    "bid_level_safe(a.bid)": "the bid's level",
    "bid_level_safe(a.bid) or 0": "the bid's level",
    "suit_rank(a.bid[-1])": "the rank of the bid's suit",
    "promised_length(a)": "the suit length the bid promises",
    "a.partner_last_bid()": "partner's last bid",
    "a.partner_last_bid()[0]": "the level of partner's last bid",
    "a.partner_last_bid()[-1]": "the suit of partner's last bid",
    "a.partner_last_bid()[-1:]": "the suit of partner's last bid",
    "bid_level_safe(a.partner_last_bid())": "the level of partner's last bid",
    "bid_level_safe(a.partner_last_bid()) + 1": "one above the level of partner's last bid",
    "suit_rank(a.partner_last_bid()[-1])": "the rank of partner's suit",
    "a.partner_first_bid()": "partner's first bid this auction",
    "a.my_first_bid()": "the bidder's first bid this auction",
    "a.my_first_bid()[-1]": "the suit of the bidder's first bid",
    "bid_level_safe(a.my_first_bid())": "the level of the bidder's first bid",
    "a.my_previous_bid()": "the bidder's previous bid",
    "a.my_previous_bid()[-1]": "the suit of the bidder's previous bid",
    "bid_level_safe(a.my_previous_bid())": "the level of the bidder's previous bid",
    "bid_level_safe(a.my_previous_bid()) + 1": "one above the level of the bidder's previous bid",
    "bid_level_safe(_last_real_bid(a))": "the level of the last bid at the table",
    "bid_level_safe(_last_real_bid(a)) or 0": "the level of the last bid at the table",
    "bid_level_safe(_last_real_bid(a)) or 9": "the level of the last bid at the table",
    "_last_real_bid(a)[-1]": "the suit of the last bid at the table",
    '(_last_real_bid(a) or "")[-1:]': "the suit of the last bid at the table",
    "_agreed_suit(a)": "the agreed suit",
    "opponent_natural_suits(a)": "the opponents' natural suits",
    "natural_suits_of(a, (a.actor_seat(), a.partner))": "our side's natural suits",
    "a.bids[-1]": "the bid just before this one",
    "a.bids[-1:]": "the bid just before this one",
    "len(a.bids)": "the number of bids so far",
    "len([b for b in a.bids if b not in NON_BIDS])": "the number of bids so far that name a level",
    "[b for b in a.bids if b not in NON_BIDS][:1]": "the first bid that names a level",
    "max((bid_level_safe(b) or 0) for b in a.bids if b not in NON_BIDS)": "the highest level bid so far",
    "1 + max((bid_level_safe(b) or 0) for b in a.bids if b not in NON_BIDS)": "one above the highest level bid so far",
    "losing_trick_count(h)": "the hand's Losing Trick Count",
    "quick_tricks(h)": "the hand's quick tricks",
    "support_points(h, a.bid[-1])": "the hand's support points (HCP plus shortness bonus)",
    "preempt_suit_quality_score(h, a.bid[-1])": "the preempt suit's top-honor (A/K/Q) count",
}

# whole conditions as statements, keyed like _EXPR_LABELS; whether one holds is shown next to it
_COND_LABELS = {
    "len(a.bid) >= 2": "the bid names a level (it is not Pass, Double or Redouble)",
    "a.bid not in NON_BIDS": "the bid names a level (it is not Pass, Double or Redouble)",
    "a.bid[0].isdigit()": "the bid starts with its level",
    "len(a.bids)": "someone has bid before",
    '4 in (h.lengths.get("H", 0), h.lengths.get("S", 0))': "the hand has exactly 4 cards in a major",
    "len({a.bid[-1], a.my_previous_bid()[-1], a.partner_last_bid()[-1]}) == 3":
        "the bid, the bidder's previous bid and partner's last bid are in three different suits",
    "a.bid[-1] in SUITS": "the bid is in a suit (not notrump)",
    'a.bid.endswith(("N", "NT"))': "the bid is in notrump",
    "len(a.partner_last_bid()) >= 2": "partner's last bid names a level",
    "a.partner_last_bid() is not None": "partner has bid",
    "a.partner_last_bid() is None": "partner has not bid yet",
    "a.my_first_bid() is not None": "the bidder has bid before",
    "a.my_first_bid() is None": "the bidder has not bid before",
    "a.my_previous_bid() is not None": "the bidder has bid before",
    "len(a.my_first_bid()) >= 2": "the bidder's first bid names a level",
    "len(a.my_first_bid()) < 2": "the bidder's first bid does not name a level",
    "len(a.my_previous_bid()) >= 2": "the bidder's previous bid names a level",
    "a.partner_last_bid()[-1] in SUITS": "partner's last bid is in a suit",
    "a.my_previous_bid()[-1] in SUITS": "the bidder's previous bid is in a suit",
    "bid_level_safe(a.bid) is not None": "the bid names a level",
    "bid_level_safe(a.partner_last_bid()) is not None": "partner's last bid names a level",
    "bid_level_safe(a.my_previous_bid()) is not None": "the bidder's previous bid names a level",
    "h.balanced": "the hand is balanced",
    "h.passed_hand": "the bidder passed earlier",
    "_same_suit_as_partner(a)": "the bid is in partner's suit",
    "_same_suit(a.bid, a.my_previous_bid())": "the bid is in the suit of the bidder's previous bid",
    "_same_suit(a.bid, a.my_first_bid())": "the bid is in the suit of the bidder's first bid",
    "_is_jump_over_last(a)": "the bid jumps a level over the last bid at the table",
    "_last_real_bid_is_theirs(a)": "the opponents made the last bid at the table",
    "_opener_reopening_double(a)": "this is the opener's reopening double (for takeout)",
    "_has_overcall_suit(a, h)": "the hand has a suit to overcall in",
    "_partner_aces_from_reply(a, ACES(h)) is not None": "partner's Blackwood reply shows how many aces partner has",
    'a.bids[-1:] in (["P"], ["Pass"])': "the bid just before this one is Pass",
    'a.bids[-1] in ("P", "Pass")': "the bid just before this one is Pass",
    "seat_of_bid(a, len(a.bids) - 1) == a.partner": "partner made the bid just before this one",
    'bool({"A", "K"} & set(h.cards.get(a.bid[-1], [])))': "the hand holds the ace or king of the bid suit",
    "any(b[-1:] == a.bid[-1:] for i, b in enumerate(a.bids) if b not in NON_BIDS and seat_of_bid(a, i) not in "
    "(a.actor_seat(), a.partner))": "an opponent has bid the bid's suit",
    "any(b[-1] == a.bid[-1] and b not in NON_BIDS for i, b in enumerate(a.bids) if seat_of_bid(a, i) not in "
    "(a.actor_seat(), a.partner) and len(b) >= 2)": "an opponent has bid the bid's suit",
    "sum(1 for i, b in enumerate(a.bids) if b not in NON_BIDS and seat_of_bid(a, i) == a.actor_seat()) == 1":
        "the bidder has named a level exactly once before",
    "any(len(b) >= 2 and b[-1] in SUITS and seat_of_bid(a, i) not in (a.actor_seat(), a.partner) and "
    "h.stoppers.get(b[-1], False) for i, b in enumerate(a.bids) if b not in NON_BIDS)":
        "the hand has a stopper in a suit the opponents bid",
    "any(b not in NON_BIDS and len(b) >= 2 and b[-1] in SUITS and a.partner_last_bid() is not None and "
    "len(a.partner_last_bid()) >= 2 and a.partner_last_bid()[-1] == b[-1] and a.partner_last_bid()[0] == \"2\" "
    "for i, b in enumerate(a.bids) if seat_of_bid(a, i) not in (a.actor_seat(), a.partner) and b not in NON_BIDS)":
        "partner's last bid is a 2-level bid in a suit the opponents bid (a Michaels cuebid)",
    "any(len(b) >= 2 and b[-1] in SUITS for i, b in enumerate(a.bids) if seat_of_bid(a, i) not in "
    "(a.actor_seat(), a.partner) and b not in NON_BIDS)": "the opponents have bid a suit",
    "any(len(b) >= 2 and b[-1] in SUITS for i, b in enumerate(a.bids) if b not in NON_BIDS and "
    "seat_of_bid(a, i) not in (a.actor_seat(), a.partner))": "the opponents have bid a suit",
    'any(b.endswith(("N", "NT")) and seat_of_bid(a, i) not in (a.actor_seat(), a.partner) for i, b in '
    "enumerate(a.bids))": "the opponents have bid notrump",
    "all(h.stoppers.get(b[-1], False) for i, b in enumerate(a.bids) if b not in NON_BIDS and len(b) >= 2 and "
    "b[-1] in SUITS and seat_of_bid(a, i) not in (a.actor_seat(), a.partner))":
        "the hand has a stopper in every suit the opponents bid",
    "all(h.stoppers.get(b[-1], False) for i, b in enumerate(a.bids) if b not in NON_BIDS and b[-1] in SUITS and "
    "seat_of_bid(a, i) not in (a.actor_seat(), a.partner))": "the hand has a stopper in every suit the opponents bid",
    "any(bid_level_safe(b) is not None and bid_level_safe(b) >= 3 for b in a.bids if b not in NON_BIDS)":
        "someone has bid at the 3-level or higher",
    "any(b == _last_real_bid(a) and seat_of_bid(a, i) == a.partner for i, b in enumerate(a.bids))":
        "partner made the last bid at the table",
    'any(b in ("X", "Dbl", "XX", "RDbl") for b in a.bids[max(i for i, b in enumerate(a.bids) if '
    "b == _last_real_bid(a)):])": "the last bid at the table has been doubled or redoubled",
    "any(b == _last_real_bid(a) and seat_of_bid(a, i) not in (a.actor_seat(), a.partner) for i, b in "
    "enumerate(a.bids))": "an opponent made the last bid at the table",
    'any(b in ("X", "Dbl") for b in a.bids[-3:])': "one of the last three bids is Double",
    'any(b in ("4N", "4NT") and seat_of_bid(a, i) == a.partner for i, b in enumerate(a.bids))':
        "partner has bid 4NT",
    'any(b in ("X", "Dbl") and seat_of_bid(a, i) != a.partner for i, b in enumerate(a.bids))':
        "someone other than partner has doubled",
    'any(h.lengths.get(s_, 0) >= 5 for s_ in ("H", "S", "C", "D"))': "the hand has a suit of 5 or more cards",
}

_SUIT_WORDS = {"S": "spades", "H": "hearts", "D": "diamonds", "C": "clubs", "N": "notrump", "NT": "notrump"}
_CALL_WORDS = {"P": "Pass", "Pass": "Pass", "X": "Double", "Dbl": "Double", "XX": "Redouble", "RDbl": "Redouble"}


def _code(node: ast.AST) -> str:
    """A node's source in canonical form (no comments, one line), as the label tables are keyed."""
    return ast.unparse(node)


_EXPR_LABELS = {_code(ast.parse(k, mode="eval").body): v for k, v in _EXPR_LABELS.items()}
_COND_LABELS = {_code(ast.parse(k, mode="eval").body): v for k, v in _COND_LABELS.items()}
_SUIT_LENGTH = re.compile(r"^h\.lengths\.get\('([SHDC])', 0\)$")


def _fmt(v: Any) -> str:
    """A value in words: suits by name, Pass/Double/Redouble by name, collections as 'a, b or c'."""
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        if re.fullmatch(r"[1-7]N", v):  # the rules accept both spellings, 1N and 1NT
            return v + "T"
        return _SUIT_WORDS.get(v, _CALL_WORDS.get(v, v))
    if isinstance(v, (tuple, list, set, frozenset)):
        items = list(dict.fromkeys(_fmt(x) for x in (sorted(v, key=str) if isinstance(v, (set, frozenset)) else v)))
        if not items:
            return "none"
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]
    return repr(v)


def _parse_lambda_body(src: str) -> Optional[tuple]:
    """(body_node, parsed_source) of one `lambda a, h: <body>,` list entry as inspect.getsource returns
    it, parsed as-is and then without its trailing comma (which would make it a 1-tuple); None if neither
    parses to a Lambda. The parsed string is returned because column offsets refer to it, not to `src`."""
    stripped = src.strip()
    for candidate in (stripped, stripped.rstrip(",")):
        try:
            tree = ast.parse(candidate, mode="eval")
        except SyntaxError:
            continue
        if isinstance(tree.body, ast.Lambda):
            return tree.body.body, candidate
    return None


@lru_cache(maxsize=None)
def _module_lambdas(filename: str):
    """A module's lambdas by first line (parsed once per file)."""
    by_line = defaultdict(list)
    for n in ast.walk(ast.parse("".join(linecache.getlines(filename)))):
        if isinstance(n, ast.Lambda):
            by_line[n.lineno].append(n)
    return by_line


@lru_cache(maxsize=None)
def _lambda_body(fn: Callable) -> Optional[ast.AST]:
    """The body node of a lambda, found in its module's parsed source by line, so trailing comments and one-line
    `StateRule("x", [lambda ...])` entries parse; inspect.getsource's line is the fallback."""
    code = fn.__code__
    try:
        args = list(code.co_varnames[:code.co_argcount])
        found = [n for n in _module_lambdas(code.co_filename).get(code.co_firstlineno, [])
                 if [x.arg for x in n.args.args] == args]
        if len(found) == 1:
            return found[0].body
    except (OSError, SyntaxError, ValueError):
        pass
    try:
        parsed = _parse_lambda_body(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    return parsed[0] if parsed else None


def _namespace(fn: Callable, auction: AuctionContext, hand: HandFeatures) -> dict:
    ns = dict(fn.__globals__)
    if fn.__closure__:  # names local to build_sayc_state_rules (ACES, KINGS)
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                ns[name] = cell.cell_contents
            except ValueError:
                pass
    ns["a"] = auction
    ns["h"] = hand
    return ns


def _eval_node(node: ast.AST, ns: dict):
    """Returns (value_or_None, canonical source)."""
    seg = _code(node)
    try:
        code = compile(ast.Expression(body=node), "<explain.rule_justification>", "eval")
        return eval(code, ns), seg
    except Exception:
        return None, seg


def _label(seg: str) -> str:
    m = _SUIT_LENGTH.match(seg)
    if m:
        return f"the hand's length in {_SUIT_WORDS[m.group(1)]}"
    return _EXPR_LABELS.get(seg, seg)


def _is_constant(node: ast.AST) -> bool:
    """A literal, or a module constant such as SUITS / NON_BIDS: shown by its value."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_constant(e) for e in node.elts)
    return isinstance(node, ast.Name) and node.id.isupper()


def _branch(node: ast.AST, ns: dict, since: List[str]) -> ast.AST:
    """The branch of a conditional expression that applies here; why goes into `since`."""
    while isinstance(node, ast.IfExp):
        test, _ = _eval_node(node.test, ns)
        cond = _render_node(node.test, ns)
        since.append(cond if test else f"it is not the case that {cond}")
        node = node.body if test else node.orelse
    return node


def _render_compare(node: ast.Compare, ns: dict) -> str:
    since: List[str] = []
    parts = [_branch(p, ns, since) for p in [node.left] + list(node.comparators)]
    evaluated = [_eval_node(p, ns) for p in parts]
    tail = "".join(f", since {c}" for c in since)

    if (
        len(node.ops) == 2
        and isinstance(node.ops[0], (ast.Lt, ast.LtE))
        and isinstance(node.ops[1], (ast.Lt, ast.LtE))
    ):
        (lo_val, _), (mid_val, mid_seg), (hi_val, _) = evaluated
        return f"{_label(mid_seg)} is between {_fmt(lo_val)} and {_fmt(hi_val)} (currently {_fmt(mid_val)}){tail}"

    clauses = []
    for i, op in enumerate(node.ops):
        lv, lseg = evaluated[i]
        rv, rseg = evaluated[i + 1]
        phrase = _OP_PHRASES.get(type(op), "compares to")
        if isinstance(op, (ast.In, ast.NotIn)) and isinstance(rv, str):
            rv = tuple(rv)  # a.bid[0] in "1234": the characters
        if isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) and not isinstance(rv, (int, float)):
            single = isinstance(rv, (tuple, list, set, frozenset)) and len({_fmt(x) for x in rv}) == 1
            phrase = {ast.Eq: "is", ast.NotEq: "is not", ast.In: "is" if single else "is one of",
                      ast.NotIn: "is not" if single else "is not one of"}[type(op)]
        if _is_constant(parts[i + 1]):
            clauses.append(f"{_label(lseg)} {phrase} {_fmt(rv)} (currently {_fmt(lv)})")
        else:  # two things from the auction or hand: name both, give both values
            clauses.append(f"{_label(lseg)} {phrase} {_label(rseg)} (currently {_fmt(lv)} against {_fmt(rv)})")
    return " and ".join(clauses) + tail


_ITER_LABELS = {
    "a.bids": "other bid in the auction",
}


def _render_method_call(node: ast.Call, ns: dict) -> Optional[str]:
    """.endswith(...)/.startswith(...) on an expression, e.g. `a.bid.endswith(("N", "NT"))`, the most common
    method-call shape in the rules. None for any other call (the caller falls through to the next template)."""
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr in ("endswith", "startswith")
            and len(node.args) == 1):
        return None
    base_val, base_seg = _eval_node(func.value, ns)
    args_val, args_seg = _eval_node(node.args[0], ns)
    verb = "ends with" if func.attr == "endswith" else "starts with"
    args_desc = _fmt(args_val) if args_val is not None else args_seg
    return f"{_label(base_seg)} {verb} {args_desc} (currently {_fmt(base_val)})"


def _render_any_all(node: ast.Call, ns: dict) -> Optional[str]:
    """any(...)/all(...) over a single-generator GeneratorExp with zero or more `if`s (e.g.
    `any(b.endswith(("N","NT")) for b in a.bids)`). The loop variable has no single current value, so the
    condition is given as its source text. None for any other generator shape."""
    if not (isinstance(node.func, ast.Name) and node.func.id in ("any", "all")
            and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp)):
        return None
    gen = node.args[0]
    if len(gen.generators) != 1:
        return None
    comp = gen.generators[0]
    if comp.is_async or not isinstance(comp.target, ast.Name):
        return None
    iter_seg = _code(comp.iter)
    iter_label = _ITER_LABELS.get(iter_seg, f"item in {iter_seg}")
    quantifier = "at least one" if node.func.id == "any" else "every"
    clauses = [_code(gen.elt)] + [_code(if_clause) for if_clause in comp.ifs]
    return f"{quantifier} {iter_label} satisfies: " + " and ".join(clauses)


def _render_node(node: ast.AST, ns: dict) -> str:
    seg = _code(node)
    if seg in _COND_LABELS:
        return _COND_LABELS[seg]

    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return "always" if node.value else "never (a retired concept)"

    if isinstance(node, ast.BoolOp):
        joiner = " and " if isinstance(node.op, ast.And) else " or "
        # a nested and/or of the other kind is bracketed, so "A or (B and C)" keeps its grouping
        return joiner.join(f"({_render_node(v, ns)})" if isinstance(v, ast.BoolOp) else _render_node(v, ns)
                           for v in node.values)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = _render_node(node.operand, ns)
        return f"it is not the case that {f'({inner})' if isinstance(node.operand, ast.BoolOp) else inner}"

    if isinstance(node, ast.IfExp):  # only the branch that applies here
        since: List[str] = []
        branch = _branch(node, ns, since)
        return _render_node(branch, ns) + "".join(f", since {c}" for c in since)

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in PREDICATE_GLOSSES:
        return PREDICATE_GLOSSES[node.func.id]

    if isinstance(node, ast.Call):
        rendered = _render_method_call(node, ns) or _render_any_all(node, ns)
        if rendered is not None:
            return rendered

    if isinstance(node, ast.Compare):
        return _render_compare(node, ns)

    if isinstance(node, (ast.Attribute, ast.Name)):
        val, seg = _eval_node(node, ns)
        if val is not None:
            return f"{_label(seg)} is {'true' if val else 'false'}"

    # fallback: the literal source and its evaluated value, never an invented reading
    val, seg = _eval_node(node, ns)
    if val is None:
        return f"an internal rule condition (`{seg}`) holds for this hand/auction"
    return f"the condition `{seg}` holds (value: {val!r})"


def render_condition(fn: Callable, auction: AuctionContext, hand: HandFeatures) -> str:
    """One predicate (a StateRule.holds_if entry) in English for this (auction, hand), whether or not it holds;
    the values it compares are given as 'currently ...'."""
    if fn.__name__ in PREDICATE_GLOSSES:  # a named predicate passed as itself (opens_longer_minor)
        return PREDICATE_GLOSSES[fn.__name__]
    body = _lambda_body(fn)
    if body is None:
        return "an internal rule condition"
    try:
        return _render_node(body, _namespace(fn, auction, hand))
    except Exception:
        return f"an internal rule condition (`{_code(body)}`)"


def _condition_source(fn: Callable) -> str:
    body = _lambda_body(fn) if fn.__name__ == "<lambda>" else None
    return f"{fn.__name__}(a, h)" if body is None else _code(body)


def _reads_hand(fn: Callable) -> bool:
    """Whether the predicate looks at the hand (`h`), not only at the bid and the auction."""
    body = _lambda_body(fn) if fn.__name__ == "<lambda>" else None
    if body is None:
        return "h" in inspect.signature(fn).parameters
    return any(isinstance(n, ast.Name) and n.id == "h" for n in ast.walk(body))


def explain_predicate(fn: Callable, auction: AuctionContext, hand: HandFeatures) -> str:
    """Why one predicate (a StateRule.holds_if entry) is True for this (auction, hand). Raises ValueError
    if it does not hold."""
    if not fn(auction, hand):
        raise ValueError("explain_predicate called on a predicate that does not hold")
    text = render_condition(fn, auction, hand)
    return text if text != "an internal rule condition" else "an internal rule condition holds for this hand/auction"


_RULES_BY_NAME: Optional[Dict[str, StateRule]] = None


def _rules_by_name() -> Dict[str, StateRule]:
    global _RULES_BY_NAME
    if _RULES_BY_NAME is None:
        _RULES_BY_NAME = {r.name: r for r in build_sayc_state_rules()}
    return _RULES_BY_NAME


def _concept_gloss(name: str) -> str:
    try:
        return _gloss_for_concept(name)
    except Exception:
        return name.replace("bid_", "").replace("_", " ")


def render_concept_justification(name: str, auction: AuctionContext, hand: HandFeatures) -> str:
    """The sentence "why this concept is true" for one HARD_STATE_CONCEPTS name and (auction, hand).
    bid_is_splinter has no StateRule of its own (as in SaycStateAuthority.classify it is the union of
    bid_is_void_splinter and bid_is_singleton_splinter), so the sub-rule that holds is justified."""
    rules = _rules_by_name()

    if name == "bid_is_splinter":
        for sub in ("bid_is_void_splinter", "bid_is_singleton_splinter"):
            if rules[sub].holds_if and all(f(auction, hand) for f in rules[sub].holds_if):
                return render_concept_justification(sub, auction, hand)
        raise ValueError("bid_is_splinter does not hold (neither sub-rule fires)")

    rule = rules[name]
    reasons = [explain_predicate(fn, auction, hand) for fn in rule.holds_if]
    gloss = _concept_gloss(name)
    return f"'{name}' holds ({gloss}) because " + "; and ".join(reasons) + "."


def rule_conditions(name: str, auction: AuctionContext, hand: HandFeatures) -> Dict[str, Any]:
    """Concept `name`'s rule for this bid with this hand, condition by condition (the interface's rule view):
    {"holds", "retired", "groups": [[{"text", "holds", "hand", "code"}, ...], ...]}. The rule holds when every
    condition of one group holds; only bid_is_splinter has two groups, its void and singleton sub-rules. `hand`
    marks a condition that looks at the hand rather than only at the bid and the auction."""
    rules = _rules_by_name()
    subs = ["bid_is_void_splinter", "bid_is_singleton_splinter"] if name == "bid_is_splinter" else [name]
    groups: List[List[Dict[str, Any]]] = []
    for sub in subs:
        group = []
        for fn in rules[sub].holds_if:
            try:
                holds = bool(fn(auction, hand))
            except Exception:
                holds = False
            group.append({"text": render_condition(fn, auction, hand), "holds": holds,
                          "hand": _reads_hand(fn), "code": _condition_source(fn)})
        groups.append(group)
    retired = all(len(g) == 1 and g[0]["code"] == "False" for g in groups)
    return {"holds": any(all(c["holds"] for c in g) for g in groups), "retired": retired, "groups": groups}
