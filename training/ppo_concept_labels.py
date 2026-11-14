"""Rule-engine concept labels for PPO's concept-head supervision (the agent's own calls), with the
cached, compiled fast path and the worker pool. Imports only numpy and concepts/: training/ppo.py forks the pool
before JAX is imported."""
from __future__ import annotations

import numpy as np

from concepts import AuctionContext, HARD_STATE_CONCEPTS, create_sayc_state_authority, decode_hand
from concepts.auction_obs import auction_from_obs

GAME_STATE_AUTH = create_sayc_state_authority()
HARD_STATE = HARD_STATE_CONCEPTS
NUM_HARD = len(HARD_STATE)
_BIDS = ["P", "X", "XX", "1C", "1D", "1H", "1S", "1NT", "2C", "2D", "2H", "2S", "2NT",
         "3C", "3D", "3H", "3S", "3NT", "4C", "4D", "4H", "4S", "4NT", "5C", "5D", "5H",
         "5S", "5NT", "6C", "6D", "6H", "6S", "6NT", "7C", "7D", "7H", "7S", "7NT"]


# Hot auction look-ups of the rule engine, cached per classification (see _install_auction_memo).
_MEMO_HELPERS = ("partnership_has_bid", "call_is_artificial", "partner_has_suit", "in_opening_position",
                 "is_opening", "_opener_seat", "is_response", "is_overcall", "is_response_to_weak_two",
                 "is_stayman_sequence", "is_jacoby_transfer_sequence", "promised_length")
_MEMO_RULES = ("_is_blackwood_reply", "_shown_suit", "_agreed_trump", "_control_bid")
_MEMO_METHODS = ("last_bid_by", "my_first_bid", "partner_first_bid")


def _install_auction_memo() -> None:
    """Speed only, for the PPO concept step: one classification evaluates 612 predicates that ask the
    same questions of the auction (who opened, is this a response, partner's last bid, ...) dozens of times. This
    wraps those auction-only helpers so that, while SaycStateAuthority.classify runs, each answer is computed once
    per auction; outside classify they run unchanged. Installed at run time instead of in concepts/, so the rule
    engine itself stays as it is, and the labels do not change: tests/test_ppo_concept_labels.py checks they are
    identical with and without it."""
    import functools
    import concepts.predicates_auction as pa
    import concepts.rules as ru
    import concepts.types as ty
    if getattr(ru.SaycStateAuthority, "_auction_memo", False):
        return

    missing = object()

    def per_auction(fn):
        @functools.wraps(fn)
        def wrapper(a, *args):
            memo = a.__dict__.get("_memo")
            if memo is None:
                return fn(a, *args)
            key = (fn, args) if args else fn  # most look-ups take the auction only: no key tuple to build
            v = memo.get(key, missing)
            if v is missing:
                v = memo[key] = fn(a, *args)
            return v
        return wrapper
    for name in _MEMO_HELPERS:
        wrapped = per_auction(getattr(pa, name))
        setattr(pa, name, wrapped)
        if hasattr(ru, name):  # rules.py imported it by name
            setattr(ru, name, wrapped)
    for name in _MEMO_RULES:
        setattr(ru, name, per_auction(getattr(ru, name)))
    for name in _MEMO_METHODS:
        setattr(ty.AuctionContext, name, per_auction(getattr(ty.AuctionContext, name)))
    classify = ru.SaycStateAuthority.classify

    def memo_classify(self, auction, hand):
        object.__setattr__(auction, "_memo", {})
        try:
            return classify(self, auction, hand)
        finally:
            auction.__dict__.pop("_memo", None)
    ru.SaycStateAuthority.classify = memo_classify
    ru.SaycStateAuthority._auction_memo = True


def build_concept_targets(obs_list, action_list) -> np.ndarray:
    """Ground-truth concept labels for one environment's (obs, action)
    sequence -- CPU-bound, pure Python, called once per rollout per env.

    Each step's auction is decoded from its OWN observation
    (concepts.auction_obs), not accumulated along the sequence: a rollout
    column spans several deals (pgx resets finished deals in place) and can
    start mid-auction."""
    _install_auction_memo()
    T = len(obs_list)
    targets = np.zeros((T, NUM_HARD), dtype=np.float32)
    for t in range(T):
        hand = decode_hand(np.asarray(obs_list[t])[-52:].tolist())  # Python floats: same hand, without 52 numpy scalars
        calls, dealer = auction_from_obs(obs_list[t])
        auction = AuctionContext(bids=calls, bid=_BIDS[int(action_list[t])], dealer=dealer)
        hard_dict = GAME_STATE_AUTH.classify(auction, hand)
        targets[t] = [hard_dict[n] for n in HARD_STATE]
    return targets


def build_concept_targets_chunk(obs_chunk, action_chunk) -> np.ndarray:
    """build_concept_targets over a (T, n_envs_chunk, ...) slice -- the unit
    of work handed to each worker process."""
    return np.stack([build_concept_targets(obs_chunk[:, e], action_chunk[:, e])
                     for e in range(obs_chunk.shape[1])], axis=1)


# ---------------------------------------------------------------------------------------------------------------
# Fast path for the PPO concept step. Same labels as build_concept_targets, bit for bit
# (tests/test_ppo_concept_labels.py), from three observations about the work:
#   1. Most of a classification is interpreter overhead, not rule logic: a generator and an all() per rule, a dict
#      of 146 results. The rules are compiled once into straight-line `and` chains (same functions, same order,
#      same short-circuiting).
#   2. The auction alone settles almost every rule: of the 146 rules, on average all but ~5 are decided before
#      their first look at the hand (is_opening(a) is False, the bid is not 1NT, ...). Running the rules once per
#      (auction, call) against a probe hand that raises on any use gives a *plan*: the hand-independent results
#      plus the few rule tails that need the hand. The plan is cached, so a repeated auction costs ~5 rule tails
#      instead of 146 rules; about half of a self-play rollout's decisions repeat an auction seen earlier in the
#      same or a recent rollout (opening seats, common responses), more as the cache warms.
#   3. The observation is packed to bits (53 bytes of auction, 7 of hand) before it leaves the GPU, which is also
#      the plan-cache key, and the hand is decoded with per-suit tables instead of a 52-bit Python loop.
# The plan is exact because the rule functions are pure functions of (auction, hand): if a function returned
# without touching the probe, it returns the same value for every hand.

AUCTION_BITS = slice(4, 428)   # what auction_from_obs reads; vulnerability (0:4) does not enter AuctionContext
HAND_BITS = slice(428, 480)


def pack_obs(obs) -> tuple:
    """(..., 480) observations -> ((N, 53) auction bits, (N, 7) hand bits) as uint8, thresholded exactly like
    auction_from_obs (> 0.5) and decode_hand (> 0). training/ppo.py does the same on the GPU (pack_obs_jax)."""
    o = np.asarray(obs)
    o = o.reshape(-1, o.shape[-1])
    return np.packbits(o[:, AUCTION_BITS] > 0.5, axis=1), np.packbits(o[:, HAND_BITS] > 0, axis=1)


class _NeedsHand(Exception):
    pass


class _ProbeHand:
    """Stands in for the hand while a plan is built: any use of it (attribute, truth value, comparison, iteration,
    hashing) raises, so a rule function that returns normally did not depend on the hand."""
    __slots__ = ()

    def __getattribute__(self, name):
        raise _NeedsHand

    def _raise(self, *args):
        raise _NeedsHand
    __bool__ = __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __hash__ = __len__ = __iter__ = _raise
    __contains__ = __getitem__ = __int__ = __index__ = __add__ = __radd__ = _raise


_PROBE = _ProbeHand()


def _compile_probe(rules, slot_of):
    """One generated function that runs every rule against the probe: for each rule, a nested `if` per condition
    (so the same conditions run, in the same order, with the same short-circuit as all()); a rule whose conditions
    all pass sets its slot in `base`; a rule whose k-th condition needs the hand appends (slot, conditions[k:])."""
    fns, tails, src = [], [], ["def probe(a, P, base, resid):"]
    for j, rule in enumerate(rules):
        conds = list(rule.holds_if)
        tails.append([tuple(conds[k:]) for k in range(len(conds) + 1)])
        slot = slot_of[rule.name]
        if not conds:
            src.append(f"    base[{slot}] = 1")
            continue
        src += ["    k = 0", "    try:"]
        ind = "        "
        for k, fn in enumerate(conds):
            fns.append(fn)
            src.append(f"{ind}if F{len(fns) - 1}(a, P):")
            ind += "    "
            src.append(f"{ind}k = {k + 1}")
        src.append(f"{ind}base[{slot}] = 1")
        src += ["    except Exception:", f"        resid.append(({slot}, T[{j}][k]))"]
    # the conditions are bound as default arguments: local-variable loads instead of 612 global look-ups + indexing
    src[0] = "def probe(a, P, base, resid, " + ", ".join(f"F{i}=F[{i}]" for i in range(len(fns))) + "):"
    ns = {"F": fns, "T": tails}
    exec(compile("\n".join(src), "<concept_targets.probe>", "exec"), ns)
    return ns["probe"]


def _suit_table():
    """13-bit holding of one suit (bit r = rank RANKS[r]) -> (ranks ascending as decode_hand lists them, hcp,
    is_stopper)."""
    from concepts.hand import HCP_VALUES, RANKS, is_stopper
    table = []
    for m in range(1 << 13):
        ranks = tuple(RANKS[r] for r in range(13) if m >> r & 1)
        table.append((ranks, sum(HCP_VALUES.get(r, 0) for r in ranks), is_stopper(list(ranks))))
    return table


class FastLabeller:
    """build_concept_targets for packed observations, with a plan cache that lives as long as the labeller (in
    training/ppo.py: one per worker process, for the whole run). `max_plans` bounds the cache (~1.1 kB a plan); when
    it is full the older half is dropped. keep_memo=True also keeps each auction's memoised look-ups with its plan:
    a hit then costs ~0.02 ms instead of ~0.03 ms, but a plan ~3.7 kB, so the same memory holds a third as many."""

    def __init__(self, max_plans: int = 200_000, keep_memo: bool = False):
        from concepts.hand import SUIT_ORDER, is_balanced_shape
        from concepts.types import HandFeatures
        _install_auction_memo()
        rules = GAME_STATE_AUTH.rules
        self.slot = {n: i for i, n in enumerate(HARD_STATE)}
        assert sorted(r.name for r in rules) == sorted(HARD_STATE), "rule names must be the HARD_STATE vocabulary"
        self._probe = _compile_probe(rules, self.slot)
        self._sp, self._vs, self._ss = (self.slot[n] for n in
                                        ("bid_is_splinter", "bid_is_void_splinter", "bid_is_singleton_splinter"))
        self._suits = _suit_table()
        # card i of the 52 hand bits is rank i // 4 of suit SUIT_ORDER[3 - i % 4] (concepts/hand.py)
        self._suit_col = [3 - SUIT_ORDER.index(s) for s in SUIT_ORDER]
        self._suit_order, self._balanced, self._HandFeatures = SUIT_ORDER, is_balanced_shape, HandFeatures
        self.plans: dict = {}
        self.max_plans, self.keep_memo = max_plans, keep_memo
        self.hits = self.misses = 0

    def _plan(self, auction_row: np.ndarray, action: int):
        key = auction_row.tobytes() + bytes((action,))
        plan = self.plans.get(key)
        if plan is not None:
            self.hits += 1
            return plan
        self.misses += 1
        o = np.zeros(480, np.uint8)
        o[AUCTION_BITS] = np.unpackbits(auction_row)[:AUCTION_BITS.stop - AUCTION_BITS.start]
        calls, dealer = auction_from_obs(o)
        a = AuctionContext(bids=calls, bid=_BIDS[action], dealer=dealer)
        object.__setattr__(a, "_memo", {})  # the memoised look-ups depend on the auction only
        base, resid = np.zeros(NUM_HARD, np.uint8), []
        self._probe(a, _PROBE, base, resid)
        if not self.keep_memo:
            del a.__dict__["_memo"]
        if len(self.plans) >= self.max_plans:
            for k in list(self.plans)[: self.max_plans // 2]:
                del self.plans[k]
        plan = self.plans[key] = (a, base.tobytes(), tuple(resid))
        return plan

    def hands(self, hand_packed: np.ndarray) -> list:
        """(N, 7) packed hand bits -> HandFeatures equal to decode_hand's (same dict orders, same rank orders)."""
        bits = np.unpackbits(hand_packed, axis=1)[:, :52].reshape(-1, 13, 4).astype(np.int64)
        masks = (bits << np.arange(13)[None, :, None]).sum(axis=1)  # (N, 4) by suit column
        cols, order, table, HF, bal = self._suit_col, self._suit_order, self._suits, self._HandFeatures, self._balanced
        out = []
        for row in masks.tolist():
            ent = [table[row[c]] for c in cols]
            lengths = {s: len(e[0]) for s, e in zip(order, ent)}
            out.append(HF(hcp=sum(e[1] for e in ent), lengths=lengths, balanced=bal(lengths),
                          stoppers={s: e[2] for s, e in zip(order, ent)}, passed_hand=False,
                          cards={s: list(e[0]) for s, e in zip(order, ent)}))
        return out

    def submit(self, auction_packed, hand_packed, actions) -> None:
        """LabelPool's interface, in this process (training/ppo.py without a pool): labels at collect()."""
        self._pending = (auction_packed, hand_packed, actions)

    def collect(self) -> np.ndarray:
        batch, self._pending = self._pending, None
        return self.label(*batch)

    @property
    def stats(self) -> dict:
        return {"plan_hits": self.hits, "plan_misses": self.misses, "plans": len(self.plans)}

    def label(self, auction_packed: np.ndarray, hand_packed: np.ndarray, actions) -> np.ndarray:
        """(N, 53), (N, 7), (N,) -> (N, NUM_HARD) uint8 labels, equal to build_concept_targets'."""
        n = len(auction_packed)
        plans = [self._plan(row, act) for row, act in zip(auction_packed, np.asarray(actions).tolist())]
        out = np.frombuffer(b"".join(p[1] for p in plans), np.uint8).reshape(n, NUM_HARD).copy()
        keep = self.keep_memo
        for t, ((a, _, resid), h) in enumerate(zip(plans, self.hands(hand_packed))):
            if not resid:
                continue
            if not keep:
                object.__setattr__(a, "_memo", {})
            for slot, fns in resid:
                for fn in fns:
                    if not fn(a, h):
                        break
                else:
                    out[t, slot] = 1
            if not keep:
                del a.__dict__["_memo"]
        out[:, self._sp] = out[:, self._vs] | out[:, self._ss]
        return out


def route(auction_packed: np.ndarray, actions, n: int) -> np.ndarray:
    """Worker index for each decision, a hash of its plan-cache key: the same (auction, call) always goes to the
    same worker, so the workers' caches partition the keys instead of each warming its own copy."""
    a = np.zeros((len(auction_packed), 56), np.uint8)
    a[:, :auction_packed.shape[1]] = auction_packed
    a[:, 55] = np.asarray(actions, np.uint8)
    h = np.full(len(a), 0xCBF29CE484222325, np.uint64)
    with np.errstate(over="ignore"):
        for col in a.view(np.uint64).T:
            h = (h ^ col) * np.uint64(0x100000001B3)
    return ((h >> np.uint64(29)) % np.uint64(n)).astype(np.int64)


def _worker_loop(conn, max_plans):
    import gc
    lab = FastLabeller(max_plans)
    while True:
        msg = conn.recv()
        if msg is None:
            return
        conn.send((np.packbits(lab.label(*msg), axis=1), lab.hits, lab.misses, len(lab.plans)))
        # the cache is ~10^5 long-lived containers; without this every full collection re-scans all of them (~12%)
        gc.freeze()


class LabelPool:
    """Worker processes for FastLabeller, each with its own long-lived plan cache, fed by route(). Unlike a
    multiprocessing.Pool, a decision's worker is chosen by its key, and submit() returns at once, so training/ppo.py
    can run the GPU's PPO update while the CPUs label. Must be created before JAX is imported (see the module
    docstring)."""

    def __init__(self, n: int, max_plans_total: int = 1_000_000):  # ~1.1 GB when full
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        self.n, self.conns, self.procs, self._pending = n, [], [], None
        for _ in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker_loop, args=(child, max(1, max_plans_total // n)), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)
        self.stats = {}

    def submit(self, auction_packed, hand_packed, actions) -> None:
        assert self._pending is None, "collect() the previous batch first"
        actions = np.asarray(actions)
        w = route(auction_packed, actions, self.n)
        idx = [np.flatnonzero(w == k) for k in range(self.n)]
        for conn, i in zip(self.conns, idx):
            conn.send((auction_packed[i], hand_packed[i], actions[i]))
        self._pending = (len(actions), idx)

    def collect(self) -> np.ndarray:
        n_rows, idx = self._pending
        self._pending = None
        out = np.empty((n_rows, NUM_HARD), np.uint8)
        hits = misses = plans = 0
        for conn, i in zip(self.conns, idx):
            packed, h, m, p = conn.recv()
            out[i] = np.unpackbits(packed, axis=1, count=NUM_HARD)
            hits, misses, plans = hits + h, misses + m, plans + p
        self.stats = {"plan_hits": hits, "plan_misses": misses, "plans": plans}
        return out

    def label(self, auction_packed, hand_packed, actions) -> np.ndarray:
        self.submit(auction_packed, hand_packed, actions)
        return self.collect()

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
