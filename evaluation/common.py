#!/usr/bin/env python3
"""Shared by every thesis evaluation: the position sets (WBridge5's held-out test split, or a model's own
self-play), the model registry `MODELS`, policies, checkpoint stamps for resuming, and statistics (deal-cluster
bootstrap, Wilson, AUROC, masked softmax and KL)."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from training.concept_bottleneck import NUM_ACTIONS, OBS_SIZE

REPO = Path(__file__).resolve().parent.parent
TEST_SPLIT = REPO / "data" / "wbridge5" / "prepared" / "test"
HELD_OUT_DDS = REPO / "dds_results" / "dds_results_500K.npy"
SEATS = "NESW"
BIDS = ["P", "X", "XX"] + [f"{lvl}{s}" for lvl in range(1, 8) for s in ("C", "D", "H", "S", "NT")]
VUL_NAMES = {(False, False): "None", (True, False): "NS", (False, True): "EW", (True, True): "Both"}

def checkpoint_stamp(path, key: str = "path") -> dict:
    """What a cached result records about the checkpoint it was computed from."""
    p = Path(path)
    return {key: str(p), f"{key}_mtime": p.stat().st_mtime if p.exists() else None}


def same_checkpoint(entry: dict, path, key: str = "path") -> bool:
    """Was a cached result computed from this checkpoint file? The evaluations that resume from
    their output file cache by model name; an entry counts only if it records the same path and,
    when it recorded one, the same modification time (a checkpoint retrained in place under the
    same name is another model)."""
    p = Path(path)
    if str(entry.get(key)) != str(p):
        return False
    m = entry.get(f"{key}_mtime")
    return m is None or (p.exists() and abs(p.stat().st_mtime - m) < 1e-3)


def last_checkpoint(ppo_dir: Path) -> Path:
    """The highest-numbered checkpoint of a PPO run."""
    ckpts = sorted((ppo_dir / "rl_params").glob("params-*.pkl"), key=lambda q: int(q.stem.split("-")[1])) \
        if (ppo_dir / "rl_params").is_dir() else []
    return ckpts[-1] if ckpts else ppo_dir / "rl_params" / "params-missing.pkl"


def converged_checkpoint(run: Path) -> Optional[Path]:
    """The checkpoint pipeline/converge.py stopped a PPO run at (<run>/converged.json), not the highest file:
    checkpoints after convergence are moved to rl_params_after/, but a copy of the run can bring them back."""
    import json
    try:
        k = json.loads((run / "converged.json").read_text())["update"]
    except (OSError, ValueError, KeyError):
        return None
    path = run / "ppo" / "rl_params" / f"params-{k:08d}.pkl"
    return path if path.exists() else None


# The models the thesis evaluates, in the folders pipeline/train.sh writes: per mode the SL concept model, its two
# seed replicates, PPO with the KL anchor at eta 0.01 and without it (each until converged), and the mode's SL recipe
# distilled from brl RL+FSP's games; plus the plain network (no concept loss), its PPO run and brl's networks.
RUNS = {"sl": "runs/sl/{mode}", "seed": "runs/sl/{mode}_s{s}", "ppo_root": "runs/ppo", "eta": "0.01",
        "distilled": "runs/sl/distilled_{mode}"}
MODES = ("joint", "independent", "sequential")
PPO_RUN = {"joint": "joint_a{}_lr3e-5", "independent": "ind_a{}_lr1e-4", "sequential": "seq_a{}_lr1e-4",
           "plain": "plain_a{}_lr3e-5"}
# the concept models distilled from brl RL+FSP's games (scripts/distill_rl_fsp.py); registered by
# evaluation.strength_matches --distill and the scripts that ask for them, never in a default model list
DISTILLED = {f"distilled {m}": REPO / RUNS["distilled"].format(mode=m) / "params-best.pkl" for m in MODES}
BRL_NETWORKS = {
    "brl SL": REPO / "bridge_models/model-sl.pkl",
    "brl RL": REPO / "bridge_models/model-pretrained-rl.pkl",
    "brl RL+FSP": REPO / "bridge_models/model-pretrained-rl-with-fsp.pkl",  # the paper's best (+1.24 vs WBridge5)
    "brl RL+PFSP": REPO / "bridge_models/model-pretrained-rl-with-pfsp.pkl",
    "brl RL scratch": REPO / "bridge_models/model-from-scratch-rl.pkl",     # RL from scratch, never imitated WBridge5
}


def _registry():
    """{name: checkpoint} per group; a model whose SL run is incomplete or whose PPO run is not marked done is None."""
    import json as _json

    def sl(rel):
        d = REPO / rel
        try:
            s = _json.loads((d / "summary.json").read_text())
        except (OSError, ValueError):
            return None
        ok = s.get("completed") and s.get("config", {}).get("iterations") == 480_000
        return d / "params-best.pkl" if ok and (d / "params-best.pkl").exists() else None

    def ppo(run):
        d = REPO / RUNS["ppo_root"] / run
        if not (d / "done").exists():
            return None
        return converged_checkpoint(d) or last_checkpoint(d / "ppo")

    return {
        "main": {**{f"SL {m}": sl(RUNS["sl"].format(mode=m)) for m in MODES},
                 **{f"PPO {m}": ppo(PPO_RUN[m].format(RUNS["eta"])) for m in MODES},
                 **{f"PPO {m} unanchored": ppo(PPO_RUN[m].format("0")) for m in MODES}},
        "seeds": {f"SL {m} s{s}": sl(RUNS["seed"].format(mode=m, s=s)) for m in MODES for s in (1, 2)},
        "plain": {"SL plain": sl("runs/sl/plain"), "SL plain s1": sl("runs/sl/plain_s1"),
                  "SL plain s2": sl("runs/sl/plain_s2")},
        "plain_ppo": {"PPO plain unanchored": ppo(PPO_RUN["plain"].format("0"))},
        "brl": {n: (p if p.exists() else None) for n, p in BRL_NETWORKS.items()},
    }


_GROUPS = _registry()
MODELS = {n: p for g in _GROUPS.values() for n, p in g.items() if p is not None}
MISSING_CURRENT = [n for g in _GROUPS.values() for n, p in g.items() if p is None]   # checked by pipeline/evaluate.sh
_have = lambda g: [n for n in _GROUPS[g] if n in MODELS]  # noqa: E731
CONCEPT_MODELS = _have("main")                      # the main tables: 3 SL, 3 anchored PPO, 3 unanchored PPO
SEED_MODELS, PLAIN_MODELS, BRL_MODELS = _have("seeds"), _have("plain"), _have("brl")


@dataclass
class Positions:
    obs: np.ndarray                 # (n, 480) uint8 0/1
    legal: np.ndarray               # (n, 38) bool
    action: np.ndarray              # (n,) the call actually made: WBridge5's, or the self-play model's
    deal: np.ndarray                # (n,) deal index (cluster id for the bootstrap)
    step: np.ndarray                # (n,) decision number within the deal
    dealer: np.ndarray              # (n,) dealer seat, 0 = N
    vul: List[str]                  # (n,) "None" / "NS" / "EW" / "Both"
    targets: Optional[np.ndarray]   # (n, 146) rule-engine labels of `action` (WBridge5 split only)
    source: str
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.action)

    def obs_f32(self, rows=None) -> np.ndarray:
        o = self.obs if rows is None else self.obs[rows]
        return o.astype(np.float32)

    def auction(self, i: int) -> Tuple[List[Dict[str, str]], str, int]:
        """(auction_before as [{"seat", "bid"}], seat to act, dealer) for position i."""
        from concepts.auction_obs import auction_from_obs
        calls, _ = auction_from_obs(self.obs[i])
        dl = int(self.dealer[i])
        before = [{"seat": SEATS[(dl + j) % 4], "bid": c} for j, c in enumerate(calls)]
        return before, SEATS[(dl + len(calls)) % 4], dl


def load_test_split(split_dir: Path = TEST_SPLIT, max_deals: Optional[int] = None) -> Positions:
    """Every decision of a prepared WBridge5 split (default: the held-out test split)."""
    split_dir = Path(split_dir)
    offsets = np.load(split_dir / "deal_offsets.npy")
    n_deals = len(offsets) - 1 if max_deals is None else min(max_deals, len(offsets) - 1)
    n = int(offsets[n_deals])
    obs = np.unpackbits(np.load(split_dir / "obs_bits.npy", mmap_mode="r")[:n], axis=1)[:, :OBS_SIZE]
    legal = np.unpackbits(np.load(split_dir / "legal_bits.npy", mmap_mode="r")[:n], axis=1)[:, :NUM_ACTIONS] > 0
    tgt = np.unpackbits(np.load(split_dir / "target_bits.npy", mmap_mode="r")[:n], axis=1)
    import json
    num_concepts = json.loads((split_dir.parent / "meta.json").read_text())["num_concepts"]
    tgt = tgt[:, :num_concepts] > 0
    action = np.load(split_dir / "action.npy")[:n].astype(np.int64)
    deal = np.repeat(np.arange(n_deals), np.diff(offsets[:n_deals + 1]))
    step = np.concatenate([np.arange(offsets[g + 1] - offsets[g]) for g in range(n_deals)])
    return Positions(obs.astype(np.uint8), legal, action, deal, step, np.zeros(n, np.int64), ["None"] * n,
                     tgt, f"WBridge5 {split_dir.name} split", {"deals": n_deals})


def policy_fn(path) -> Tuple[Callable[[np.ndarray], np.ndarray], str]:
    """(obs_batch -> masked-free policy logits, kind) for a concept checkpoint or a brl network."""
    import jax
    from evaluation.duplicate_match import load_policy
    params, logits_fn, kind = load_policy(str(path))
    f = jax.jit(lambda o: logits_fn(params, o))
    return (lambda o: np.asarray(f(np.asarray(o, np.float32)))), kind


def selfplay(path, n_deals: int, seed: int = 0, random_vulnerability: bool = False,
             dds_path: Path = HELD_OUT_DDS, batch: int = 1000) -> Positions:
    """Every decision of `n_deals` pgx deals bid greedily by the model at `path` at all four seats.
    Deals come from the held-out DDS table (the one duplicate_match uses), not PPO's training table."""
    import jax
    import jax.numpy as jnp
    from pgx.bridge_bidding import BridgeBidding, _observe
    logits, kind = policy_fn(path)
    env = BridgeBidding(dds_results_table_path=str(dds_path))
    init, step_fn = jax.jit(jax.vmap(env.init)), jax.jit(jax.vmap(env.step))
    observe = jax.jit(jax.vmap(_observe))
    key = jax.random.PRNGKey(seed)
    rows_obs, rows_legal, rows_act, rows_deal, rows_step, rows_dealer, rows_vul = [], [], [], [], [], [], []
    for start in range(0, n_deals, batch):
        n = min(batch, n_deals - start)
        key, k_init = jax.random.split(key)
        state = init(jax.random.split(k_init, batch))
        if not random_vulnerability:
            state = state.replace(_vul_NS=jnp.zeros(batch, jnp.bool_), _vul_EW=jnp.zeros(batch, jnp.bool_))
            state = state.replace(observation=observe(state, state.current_player))
        dl = np.asarray(state._dealer)
        vul = [VUL_NAMES[(bool(a), bool(b))] for a, b in zip(np.asarray(state._vul_NS), np.asarray(state._vul_EW))]
        for t in range(60):
            term = np.asarray(state.terminated)
            if term[:n].all():
                break
            obs = np.asarray(state.observation)[:, :OBS_SIZE].astype(np.uint8)
            legal = np.asarray(state.legal_action_mask).astype(bool)
            act = np.where(legal, logits(obs), -np.inf).argmax(1)
            live = np.flatnonzero(~term[:n])
            rows_obs.append(obs[live]); rows_legal.append(legal[live]); rows_act.append(act[live])
            rows_deal.append(start + live); rows_step.append(np.full(len(live), t))
            rows_dealer.append(dl[live]); rows_vul += [vul[i] for i in live]
            key, k_step = jax.random.split(key)
            state = step_fn(state, jnp.asarray(act), jax.random.split(k_step, batch))
    order = np.lexsort((np.concatenate(rows_step), np.concatenate(rows_deal)))
    cat = lambda xs: np.concatenate(xs)[order]  # noqa: E731
    vul_all = [rows_vul[i] for i in order]
    return Positions(cat(rows_obs), cat(rows_legal), cat(rows_act).astype(np.int64), cat(rows_deal),
                     cat(rows_step), cat(rows_dealer).astype(np.int64), vul_all, None,
                     f"self-play of {Path(path).name} ({kind}), {n_deals:,} deals, "
                     f"{'random' if random_vulnerability else 'no'} vulnerability, seed {seed}",
                     {"deals": n_deals, "model": str(path)})


def sample_rows(pos: Positions, n: int, seed: int = 0) -> np.ndarray:
    """n decisions drawn uniformly without replacement (sorted, reproducible)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(pos), size=min(n, len(pos)), replace=False))


# ---------------------------------------------------------------- numerics

def log_softmax_masked(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """log pi over the legal calls (illegal calls get -inf), any leading shape."""
    z = np.where(legal, logits, -np.inf)
    m = z.max(-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(-1, keepdims=True))


def masked_argmax(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    return np.where(legal, logits, -np.inf).argmax(-1)


def kl_masked(logp: np.ndarray, logq: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """KL(p || q) over the legal calls (natural log, nats)."""
    lp = np.where(legal, logp, 0.0)
    return np.where(legal, np.exp(lp) * (lp - np.where(legal, logq, 0.0)), 0.0).sum(-1)


# ---------------------------------------------------------------- statistics

def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, stat: Callable = np.mean, n_boot: int = 1000,
                      seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """(estimate, lo, hi): percentile CI resampling whole clusters (deals) with replacement."""
    values = np.asarray(values, dtype=float)
    est = float(stat(values))
    uniq, inv = np.unique(clusters, return_inverse=True)
    if len(uniq) < 2:
        return est, float("nan"), float("nan")
    # per-cluster sums and counts make the mean bootstrap O(n_boot * n_clusters)
    if stat is np.mean:
        sums = np.bincount(inv, weights=values, minlength=len(uniq))
        cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
        boots = sums[draws].sum(1) / np.maximum(cnts[draws].sum(1), 1)
    else:
        rng = np.random.default_rng(seed)
        members = [np.flatnonzero(inv == c) for c in range(len(uniq))]
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            boots.append(stat(values[np.concatenate([members[c] for c in pick])]))
        boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return est, float(lo), float(hi)


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """(rate, lo, hi) Wilson score interval for k successes in n independent trials."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties)."""
    from scipy.stats import rankdata
    labels = np.asarray(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def concept_category(name: str) -> str:
    """A coarse grouping of the 146 concepts, for colouring figures and per-group sums. Keyword-based
    and deliberately coarse (the vocabulary's own section comments give ~30 finer groups)."""
    n = name
    if n.startswith("bid_is_in_"):
        return "strain"
    if n in ("bid_is_pass", "bid_is_double", "bid_is_redouble") or n.startswith("bid_pass_"):
        return "pass / double"
    if any(k in n for k in ("stayman", "transfer", "blackwood", "gerber", "keycard", "aces", "king",
                            "splinter", "fourth_suit", "conventional", "control", "check_back", "waiting",
                            "michaels", "unusual", "negative_double", "penalty_double", "takeout_double",
                            "positive_response", "puppet", "relay")):
        return "convention"
    if "opening" in n:
        return "opening"
    if any(k in n for k in ("overcall", "cuebid", "competitive", "sacrifice", "competes", "defensive",
                            "stopper", "length_in_their")):
        return "competitive"
    if any(k in n for k in ("support", "fit", "raise")):
        return "fit / support"
    if any(k in n for k in ("cards", "length", "suited", "shortness", "longer", "major_suit", "minor_suit",
                            "notrump", "prefers")):
        return "length / strain"
    if any(k in n for k in ("values", "limits", "invite", "slam", "game", "points", "quick_tricks",
                            "rule_of_20", "ltc", "quality", "jump", "preempt", "weak", "strength")):
        return "strength"
    return "auction structure"
