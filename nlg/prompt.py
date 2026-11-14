"""The explainer's prompt (`build_llm_prompt`, identical at training and inference) and the style gate
(`check_style`). The LLM only writes prose: the reasons come from nlg.explanation_inputs, and every output is checked."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from concepts.vocabulary import gloss_for_concept
from nlg.explanation_inputs import (SEAT_NAME, bid_meaning_concepts, deviation_notes, hand_facts, narrate_auction,
                           vulnerability_line)

# The fine-tuned explainer learns the style from its training data, so the system prompt is short: a long
# instruction block would push most teacher-dataset rows past train_qwen's 768-token limit and truncate
# the target away. Training and inference use this same prompt.
SYSTEM_PROMPT = (
    "You are a bridge teacher. Explain the given SAYC bid in 2-4 sentences: start with the seat, the "
    "action and what the bid means here, then connect concrete holdings and counts from the hand facts to "
    "the auction and the call. Use only the facts given; if the bid is marked unusual, say so plainly.")


def _fmt_reasons(frame: Dict[str, Any], selected_concepts: List[str]) -> str:
    if not selected_concepts:
        return "(no hand-specific reason stood out; explain from the hand facts and the auction)"
    return "\n".join(f"- {gloss_for_concept(c)}" for c in selected_concepts)


def _fmt_deviation(frame: Dict[str, Any]) -> str:
    notes = deviation_notes(frame)
    if not notes:
        return ""
    return ("\nUnusual bid (say so plainly in the explanation instead of pretending it is standard):\n"
            + "\n".join(f"- {n}" for n in notes))


def build_llm_prompt(frame: Dict[str, Any], selected_concepts: List[str]) -> List[Dict[str, str]]:
    """Chat-formatted (system, user) message list, the same at training (nlg/train_qwen.py) and inference.
    `selected_concepts` comes from `nlg.explanation_inputs.select_reasoned_concepts()`. The prompt holds
    neither the rule-engine justification nor concept identifiers (the LLM parrots both); the hand is
    given as exact facts in the auction's context (nlg.explanation_inputs.hand_facts)."""
    table = frame.get("table") or {}
    you = SEAT_NAME.get(table.get("you"), table.get("you", "?"))
    ru = frame.get("runner_up") or {}
    alt = f"\nMain alternative the bidder considered: {ru['bid']}" if ru.get("bid") else ""
    facts = "\n".join(f"- {f}" for f in hand_facts(frame))
    from nlg.teacher import situation  # lazy: nlg.teacher imports nlg.explanation_inputs, not this module
    sit = situation(frame)
    if sit["meaning"]:
        meaning_line = f"\nWhat this bid means here: {sit['meaning']}"
    else:
        meaning = bid_meaning_concepts(frame)
        meaning_line = ("\nWhat this bid means here: " + "; ".join(gloss_for_concept(c) for c in meaning)
                        if meaning else "")
    vl = vulnerability_line(frame)
    vul_line = f"\nVulnerability: {vl}." if vl else ""
    key_facts = ("\n\nKey facts for this situation:\n" + "\n".join(f"- {f}" for f in sit["facts"])
                 if sit["facts"] else "")

    user = f"""Bid to explain: {you} bids {frame.get('bid', '?')}.{meaning_line}{alt}
Seats: you are {you}, partner is {SEAT_NAME.get(table.get('partner'), '?')}, opponents are {' and '.join(SEAT_NAME.get(o, o) for o in table.get('opponents') or [])}.{vul_line}
Auction so far: {narrate_auction(frame)}

Hand facts (exact):
{facts}{key_facts}

Reasons that most pushed toward {frame.get('bid', '?')} (verified true for this hand):
{_fmt_reasons(frame, selected_concepts)}{_fmt_deviation(frame)}

Write the explanation now. Output ONLY the explanation sentences, nothing else."""

    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# Style gate on top of nlg.consistency_checks's factual checks: no identifiers ("'bid_is_double' holds")
# or filler ("the hand's characteristics ... support this doubling action").
_STYLE_BANNED = re.compile(
    r"_|\bconcepts?\b|\brule[- ]engine\b|\bthe model\b|\bconfiden|characteristics|\balign(?:s|ing|ed)? with|as confirmed by",
    re.IGNORECASE)


def check_style(explanation: str) -> Optional[str]:
    m = _STYLE_BANNED.search(explanation)
    return f"style: contains {m.group(0)!r}" if m else None
