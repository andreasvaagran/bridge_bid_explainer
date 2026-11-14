"""The hand-written gold explanations (gold_examples.jsonl): one per bidding situation plus WBridge5-specific
and unusual-bid variants, each with its frame; train_qwen.py --gold adds them to the training rows."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

GOLD_PATH = Path(__file__).resolve().parent / "gold_examples.jsonl"


@lru_cache(maxsize=4)
def load_gold(path: str = str(GOLD_PATH)) -> Tuple[Dict[str, Any], ...]:
    p = Path(path)
    if not p.exists():
        return ()
    return tuple(json.loads(l) for l in p.read_text().splitlines() if l.strip())


def decision_key(frame: Dict[str, Any]) -> tuple:
    return (frame.get("hand_str"), tuple(b.get("bid") for b in frame.get("auction_before") or []),
            frame.get("bid"))
