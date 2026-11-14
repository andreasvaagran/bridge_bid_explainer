"""The table behind the web interface: the model registry (our checkpoints and brl's networks) and a pgx game
where each seat is a model or a human. Deals come from the held-out double-dummy table or are entered as PBN. Every
call is stored with its explained record (a human call with the advisor's), and the contract is scored double-dummy."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from pgx.bridge_bidding import BridgeBidding, _contract, _observe, _value_to_dds_tricks

from concepts.hand import decode_hand
from evaluation.common import RUNS, PPO_RUN, converged_checkpoint
from nlg.explanation_frames import BIDS, hand_str_letter_coded
from nlg.decision_record import Bidder, analyze, score_breakdown

REPO = Path(__file__).resolve().parent.parent
SEATS = "NESW"
STRAINS = ["C", "D", "H", "S", "NT"]  # pgx denomination order
VUL = {"None": (False, False), "NS": (True, False), "EW": (False, True), "Both": (True, True)}


def final_ppo_runs(runs_dir: Path, mode: str) -> Dict[str, Any]:
    """The converged checkpoints of the thesis's PPO runs of `mode` (runs/ppo/<run>/converged.json): {"": (label,
    path)} with the anchor, {"-unanchored": ...} without it; {} when they are not there."""
    out = {}
    root = runs_dir / RUNS["ppo_root"].removeprefix("runs/")
    for suffix, label, eta in (("", "", RUNS["eta"]), ("-unanchored", ", no anchor", "0")):
        path = converged_checkpoint(root / PPO_RUN[mode].format(eta))
        if path:
            out[suffix] = (label, path)
    return out


class Registry:
    """Every model the interface can seat, loaded lazily and kept."""

    def __init__(self, runs_dir: Path = REPO / "runs", brl_dir: Optional[Path] = None):
        brl_dir = brl_dir or REPO / "bridge_models"  # brl's published networks
        self.specs: Dict[str, Dict[str, str]] = {}
        for mode in ("independent", "joint", "sequential"):
            sl = runs_dir / RUNS["sl"].removeprefix("runs/").format(mode=mode) / "params-best.pkl"
            if sl.exists():
                self.specs[f"{mode}-sl"] = {"label": f"{mode} SL (concept bottleneck)", "short": f"{mode} SL",
                                            "path": str(sl), "kind": "concept"}
            for suffix, (label, path) in final_ppo_runs(runs_dir, mode).items():  # with and without the anchor
                self.specs[f"{mode}-ppo{suffix}"] = {"label": f"{mode} PPO{label} (concept bottleneck)",
                                                     "short": f"{mode} PPO{label}", "path": str(path),
                                                     "kind": "concept"}
        if brl_dir.is_dir():
            for p in sorted(brl_dir.glob("model-*.pkl")):
                short = {"model-sl": "brl SL", "model-pretrained-rl": "brl RL", "model-pretrained-rl-with-fsp": "brl RL+FSP",
                         "model-pretrained-rl-with-pfsp": "brl RL+PFSP", "model-from-scratch-rl": "brl RL (from scratch)"
                         }.get(p.stem, "brl " + p.stem)
                self.specs[f"brl-{p.stem}"] = {"label": f"original brl: {p.stem}", "short": short, "path": str(p),
                                               "kind": "upstream"}
        self._models: Dict[str, Bidder] = {}
        self._lock = threading.Lock()

    @classmethod
    def of(cls, paths: Dict[str, str]) -> "Registry":
        """A registry of just these checkpoints ({id: path}), for scripts."""
        reg = cls.__new__(cls)
        reg.specs, reg._models, reg._lock = {}, {}, threading.Lock()
        for mid, path in paths.items():
            b = Bidder(path)
            reg.specs[mid] = {"label": b.label, "short": b.label, "path": str(path), "kind": b.kind}
            reg._models[mid] = b
        return reg

    def list(self) -> List[Dict[str, str]]:
        return [{"id": k, **v} for k, v in self.specs.items()]

    def default(self) -> str:
        return next(iter(self.specs))

    def get(self, model_id: str) -> Bidder:
        with self._lock:
            if model_id not in self._models:
                spec = self.specs[model_id]
                self._models[model_id] = Bidder(spec["path"], label=spec["label"])
            return self._models[model_id]


class _Env:
    """pgx functions, jitted once per DDS table."""
    _cache: Dict[str, "_Env"] = {}

    def __init__(self, dds_path: str):
        env = BridgeBidding(dds_results_table_path=dds_path)
        self.init = jax.jit(env.init)
        self.step = jax.jit(env.step)
        self.observe = jax.jit(_observe)

    @classmethod
    def get(cls, dds_path: str) -> "_Env":
        if dds_path not in cls._cache:
            cls._cache[dds_path] = _Env(dds_path)
        return cls._cache[dds_path]


_PBN_RANK = {"A": 0, "K": 12, "Q": 11, "J": 10, "T": 9}


def hand_array_from_pbn(pbn: str) -> np.ndarray:
    """A PBN deal ("N:Q6.A76.JT32.KT64 AT74.K9.KQ96.953 ...", hands clockwise from the
    seat named first, suits S.H.D.C) -> pgx's 52-card `_hand` array. Raises ValueError
    unless it is 4 hands of 13 different cards."""
    from pgx.bridge_bidding import _key_to_hand, _to_binary
    pbn = pbn.strip().replace("10", "T")
    first, _, rest = pbn.partition(":")
    if first.upper() not in SEATS or not rest:
        raise ValueError("a PBN deal starts with the first hand's seat, e.g. 'N:AKQ.xxx...'")
    hands = rest.split()
    if len(hands) != 4:
        raise ValueError("a PBN deal has four hands")
    key = np.full(52, -1, np.int32)
    start = SEATS.index(first.upper())
    for k, h in enumerate(hands):
        player = (start + k) % 4
        suits = h.split(".")
        if len(suits) != 4 or sum(len(x) for x in suits) != 13:
            raise ValueError(f"hand {k + 1} is not 13 cards in four suits: {h!r}")
        for s, cards in enumerate(suits):
            for c in cards.upper():
                if c not in "AKQJT98765432":
                    raise ValueError(f"not a card: {c!r}")
                idx = s * 13 + (_PBN_RANK[c] if c in _PBN_RANK else int(c) - 1)
                if key[idx] >= 0:
                    raise ValueError(f"a card appears twice: {'SHDC'[s]}{c}")
                key[idx] = player
    return np.asarray(_key_to_hand(_to_binary(jnp.array(key.reshape(4, 13)))))


class Game:
    def __init__(self, registry: Registry, seats: Dict[str, str], seed: int = 0, dealer: Optional[str] = None,
                 vulnerability: str = "random", advisor: Optional[str] = None,
                 dds_path: str = str(REPO / "dds_results" / "dds_results_500K.npy"), pbn: Optional[str] = None):
        self.registry, self.seats, self.seed = registry, dict(seats), int(seed)
        self.advisor = advisor or next((m for m in seats.values() if m != "human"), registry.default())
        self.env = _Env.get(dds_path)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, k = jax.random.split(self.key)
        st = self.env.init(k)
        # an entered deal (a hand from another site): its double-dummy result is not in the table
        self.custom = bool(pbn)
        if pbn:
            st = st.replace(_hand=jnp.asarray(hand_array_from_pbn(pbn)))
        if dealer:
            d = SEATS.index(dealer)
            st = st.replace(_dealer=jnp.int32(d), current_player=st._shuffled_players[d])
        if vulnerability in VUL:
            ns, ew = VUL[vulnerability]
            st = st.replace(_vul_NS=jnp.bool_(ns), _vul_EW=jnp.bool_(ew))
        self.state = st.replace(observation=self.env.observe(st, st.current_player))
        self.dealer = int(self.state._dealer)
        self.vulnerability = {v: k for k, v in VUL.items()}[(bool(self.state._vul_NS), bool(self.state._vul_EW))]
        self.hands = {s: hand_str_letter_coded(decode_hand(np.asarray(self._obs(s)))) for s in SEATS}
        self.auction: List[Dict[str, Any]] = []
        self.records: List[Dict[str, Any]] = []

    # -- helpers ------------------------------------------------------------
    def _obs(self, seat: str):
        player = self.state._shuffled_players[SEATS.index(seat)]
        return np.asarray(self.env.observe(self.state, player), np.float32)[:480]

    def finished(self) -> bool:
        return bool(self.state.terminated)

    def to_act(self) -> Optional[str]:
        return None if self.finished() else SEATS[(self.dealer + len(self.auction)) % 4]

    def legal_calls(self) -> List[str]:
        return [] if self.finished() else [BIDS[i] for i in np.flatnonzero(np.asarray(self.state.legal_action_mask))]

    def auction_before(self) -> List[Dict[str, str]]:
        return [{"seat": a["seat"], "bid": a["call"]} for a in self.auction]

    def _analyze(self, model_id: str) -> Dict[str, Any]:
        seat = self.to_act()
        obs, legal = self._obs(seat), np.asarray(self.state.legal_action_mask, bool)
        rec = analyze(self.registry.get(model_id), obs, legal, self.auction_before(), seat, self.dealer,
                      self.vulnerability, deal_id=self.seed, step=len(self.auction))
        rec["_position"] = (obs, legal)  # kept for comparing models at this position (not sent to the browser)
        rec["_model_id"] = model_id      # the follow-up questions recompute its facts
        rec["_k"] = len(self.auction)    # the auction position it explains
        return rec

    # -- switching models mid-game ------------------------------------------------
    def set_seat(self, seat: str, player: str) -> None:
        """Who plays `seat` from now on (a model id or "human"); calls already made stay."""
        if seat not in SEATS or (player != "human" and player not in self.registry.specs):
            raise ValueError(f"unknown seat or player: {seat} {player}")
        self.seats[seat] = player

    def set_advisor(self, model_id: str) -> None:
        if model_id not in self.registry.specs:
            raise ValueError(f"unknown model {model_id}")
        self.advisor = model_id
        self.__dict__.pop("_advice", None)  # the hint at the current position is the new advisor's

    def explain_as(self, i: int, model_id: str) -> int:
        """Record i's position explained by another model (what IT would call there, and
        why): a new record, computed once per (position, model). Returns its index."""
        rec = self.records[i]
        if rec.get("_model_id") == model_id:
            return i
        cache = self.__dict__.setdefault("_views", {})
        key = (rec.get("_k"), model_id)
        if key in cache:
            return cache[key]
        obs, legal = rec["_position"]
        k = rec.get("_k", 0)
        new = analyze(self.registry.get(model_id), obs, legal, self.auction_before()[:k], rec["seat"], self.dealer,
                      self.vulnerability, deal_id=self.seed, step=k)
        new.update({"_position": (obs, legal), "_model_id": model_id, "_k": k, "view_of": i})
        for f in ("human_call", "advice"):
            if f in rec:
                new[f] = rec[f]
        if k < len(self.auction) and not self.auction[k]["human"]:
            new["actual_call"] = self.auction[k]["call"]  # the call the seated model made there
        self.records.append(new)
        cache[key] = len(self.records) - 1
        return cache[key]

    def compare(self, i: int) -> List[Dict[str, Any]]:
        """What every seatable model would call at the position of record i."""
        obs, legal = self.records[i]["_position"]
        out = []
        for spec in self.registry.list():
            b = self.registry.get(spec["id"])
            logits = b.outputs(obs)["policy_logits"]
            z = np.where(legal, logits, -1e9)
            p = np.exp(z - z.max()); p /= p.sum()
            a = int(np.argmax(p))
            out.append({"id": spec["id"], "label": spec["short"], "kind": spec["kind"], "call": BIDS[a],
                        "p": round(float(p[a]), 3)})
        return out

    def weights(self, i: int, vs: Optional[str] = None) -> Dict[str, Any]:
        """How record i's model scored every bid at its position, and its bid against `vs`, concept by concept
        (nlg.decision_record.score_breakdown)."""
        rec = self.records[i]
        obs, legal = rec["_position"]
        return score_breakdown(self.registry.get(rec.get("_model_id") or self.advisor), obs, legal, vs=vs,
                               frame=rec.get("frame"), top=20)

    def _apply(self, call: str):
        self.key, k = jax.random.split(self.key)
        self.state = self.env.step(self.state, jnp.int32(BIDS.index(call)), k)

    # -- moves ----------------------------------------------------------------
    def model_call(self) -> Dict[str, Any]:
        seat = self.to_act()
        if seat is None:
            raise ValueError("the auction is over")
        model_id = self.seats[seat]
        if model_id == "human":
            raise ValueError(f"{seat} is played by a human")
        rec = self._analyze(model_id)
        self.records.append(rec)
        self.auction.append({"seat": seat, "call": rec["call"], "by": model_id, "record": len(self.records) - 1,
                             "unusual": bool(rec["unusual"]), "human": False})
        self._apply(rec["call"])
        return rec

    def human_call(self, call: str) -> Dict[str, Any]:
        seat = self.to_act()
        if seat is None:
            raise ValueError("the auction is over")
        if call not in self.legal_calls():
            raise ValueError(f"{call} is not a legal call here")
        advice = self._analyze(self.advisor)  # what the advisor model would bid here, and why
        advice["human_call"] = call
        self.records.append(advice)
        self.auction.append({"seat": seat, "call": call, "by": "human", "record": len(self.records) - 1,
                             "unusual": False, "human": True, "advisor_call": advice["call"]})
        self._apply(call)
        return advice

    def advice(self) -> int:
        """The advisor's explained view of the CURRENT position, before anyone calls (a
        hint for the human to act): stored as a record, not as a call; computed once per
        position. Returns the record's index."""
        if self.finished():
            raise ValueError("the auction is over")
        cur = self.__dict__.get("_advice")
        if cur and cur[0] == len(self.auction):
            return cur[1]
        rec = self._analyze(self.advisor)
        rec["advice"] = True
        self.records.append(rec)
        self._advice = (len(self.auction), len(self.records) - 1)
        return self._advice[1]

    def auto(self) -> int:
        n = 0
        while not self.finished() and self.seats[self.to_act()] != "human":
            self.model_call()
            n += 1
        return n

    # -- result -----------------------------------------------------------------
    def dd_table(self) -> Optional[Dict[str, Dict[str, int]]]:
        if self.custom:
            return None
        t = np.asarray(_value_to_dds_tricks(self.state._dds_val)).reshape(4, 5)
        return {SEATS[p]: {STRAINS[d]: int(t[p, d]) for d in range(5)} for p in range(4)}

    def result(self) -> Optional[Dict[str, Any]]:
        if not self.finished():
            return None
        if int(self.state._last_bid) < 0:
            return {"contract": "passed out", "score_ns": 0}
        pos, denom, level, _ = (int(x) for x in _contract(self.state))
        dbl = "XX" if bool(self.state._call_xx) else "X" if bool(self.state._call_x) else ""
        if self.custom:
            return {"contract": f"{level}{STRAINS[denom]}{dbl}", "declarer": SEATS[pos], "tricks": None}
        tricks = self.dd_table()[SEATS[pos]][STRAINS[denom]]
        player_n = int(self.state._shuffled_players[0])
        over = tricks - (level + 6)
        return {"contract": f"{level}{STRAINS[denom]}{dbl}", "declarer": SEATS[pos], "tricks": tricks,
                "made": over >= 0, "overtricks" if over >= 0 else "down": abs(over),
                "score_ns": int(np.asarray(self.state.rewards)[player_n])}

    def to_json(self) -> Dict[str, Any]:
        return {"seed": self.seed, "dealer": SEATS[self.dealer], "vulnerability": self.vulnerability,
                "seats": {s: {"id": m, "label": "you" if m == "human" else self.registry.specs[m]["short"]}
                          for s, m in self.seats.items()},
                "advisor": {"id": self.advisor, "label": self.registry.specs[self.advisor]["label"]},
                "hands": self.hands, "auction": self.auction, "to_act": self.to_act(),
                "legal": self.legal_calls(), "finished": self.finished(), "result": self.result(),
                "dd_table": self.dd_table() if self.finished() else None, "custom": self.custom}

    # -- what each call means here ----------------------------------------------
    def _position_at(self, i: Optional[int]):
        """(seat, obs, legal, calls before) at auction position i (None: the current one)."""
        if i is None or i >= len(self.auction):
            if self.finished():
                raise ValueError("the auction is over")
            seat = self.to_act()
            return seat, self._obs(seat), np.asarray(self.state.legal_action_mask, bool), self.auction
        a = self.auction[i]
        obs, legal = self.records[a["record"]]["_position"]
        return a["seat"], obs, legal, self.auction[:i]

    def meanings(self, i: Optional[int] = None, model_id: Optional[str] = None, with_model: bool = True,
                 calls: Optional[List[str]] = None) -> Dict[str, Any]:
        """Funbridge-style meanings of the calls at position i: WBridge5's measured
        ranges in this exact auction (nlg.call_meaning), the teacher's phrase, the
        model's probability and -- with_model -- the hands THIS model makes each call
        with here, given its own earlier calls."""
        from nlg.call_meaning import describe, model_meanings, summarize
        key = (i if i is not None and i < len(self.auction) else len(self.auction), model_id, with_model,
               tuple(calls) if calls else None)
        cache = self.__dict__.setdefault("_meaning_cache", {})
        if key in cache:
            return cache[key]
        seat, obs, legal, prior = self._position_at(i)
        idx = SEATS.index(seat)
        frame_like = {"table": {"you": seat, "partner": SEATS[(idx + 2) % 4],
                                "opponents": [SEATS[(idx + 1) % 4], SEATS[(idx + 3) % 4]]},
                      "auction_before": [{"seat": a["seat"], "bid": a["call"]} for a in prior],
                      "dealer": SEATS[self.dealer], "vulnerability": self.vulnerability,
                      "deal_id": self.seed, "step": len(prior)}
        before = [a["call"] for a in prior]
        model_id = model_id or self.advisor
        bidder = self.registry.get(model_id)
        logits = bidder.outputs(obs)["policy_logits"]
        z = np.where(legal, logits, -1e9)
        p = np.exp(z - z.max())
        p /= p.sum()
        mm = {}
        if with_model:
            hist = [(self.records[a["record"]]["_position"][0], self.records[a["record"]]["_position"][1],
                     BIDS.index(a["call"])) for a in prior if a["seat"] == seat]
            mm = model_meanings(bidder.policy_logits_batch, hist, obs, legal, seed=self.seed + len(prior))
        wanted = calls or [BIDS[k] for k in np.flatnonzero(legal)]
        out = []
        for c in wanted:
            k = BIDS.index(c)
            d = describe(before, c, frame_like)
            d.update(legal=bool(legal[k]), p=round(float(p[k]), 4))
            if k in mm:
                d["model"] = summarize(mm[k], bidder.label)
                d["model"]["share"] = round(mm[k]["share"], 4)
            out.append(d)
        res = {"position": key[0], "seat": seat, "model": bidder.label, "calls": out}
        cache[key] = res
        return res
