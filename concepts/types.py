"""`AuctionContext` (the auction so far and the call being labelled) and `HandFeatures` (one hand's shape
and strength): the inputs of every predicate and rule."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

from .vocabulary import NON_BIDS


@dataclass(frozen=True)
class HandFeatures:
    hcp: int
    lengths: Dict[str, int]       # {"S":2,"H":5,"D":3,"C":3}
    balanced: bool
    stoppers: Dict[str, bool]     # {"S":True,"H":False,...}
    passed_hand: bool
    cards: Dict[str, List[str]]   # {"S":["A","K"],"H":["Q"]}


def seat_of_bid(a: "AuctionContext", i: int) -> int:
    """Absolute seat that made bids[i]."""
    return (a.dealer + i) % 4


@dataclass(frozen=True)
class AuctionContext:
    bids: List[str]   # full auction up to but NOT INCLUDING current bid
    bid: str          # the *current* bid being classified
    dealer: int       # absolute seat of dealer (0-3)

    def actor_seat(self) -> int:
        return (self.dealer + len(self.bids)) % 4

    @property
    def partner(self) -> int:
        return (self.actor_seat() + 2) % 4

    def last_bid_by(self, rel_seat: int) -> Optional[str]:
        target = (self.actor_seat() + rel_seat) % 4
        for i in range(len(self.bids) - 1, -1, -1):
            if seat_of_bid(self, i) == target and self.bids[i] not in NON_BIDS:
                return self.bids[i]
        return None

    def partner_last_bid(self) -> Optional[str]:
        return self.last_bid_by(2)

    def my_previous_bid(self) -> Optional[str]:
        return self.last_bid_by(0)

    def my_first_bid(self) -> Optional[str]:
        actor = self.actor_seat()
        for i, b in enumerate(self.bids):
            if seat_of_bid(self, i) == actor and b not in NON_BIDS:
                return b
        return None

    def partner_first_bid(self) -> Optional[str]:
        p = self.partner
        for i, b in enumerate(self.bids):
            if seat_of_bid(self, i) == p and b not in NON_BIDS:
                return b
        return None
