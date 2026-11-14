#!/usr/bin/env python3
"""Supervised concept-bottleneck training on the prepared WBridge5 data, in any bottleneck mode
(concept_bottleneck.py). The loss is cross-entropy on WBridge5's call plus per-concept reweighted BCE on the
rule-engine labels. In sequential mode the concept phase comes first
(concept_phase_iterations). Validates on `valid` with evaluation/split_accuracy.py's metrics.

Run by pipeline/train.sh (sl_run in pipeline/lib.sh), e.g.

    python -m training.sl prepared_path=data/wbridge5/prepared save_path=runs/sl/joint bottleneck_mode=joint"""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import multiprocessing as mp
import os
import pickle
from collections import deque
from typing import List, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyspiel
from omegaconf import OmegaConf
from pydantic import BaseModel

from concepts import AuctionContext, HARD_STATE_CONCEPTS, create_sayc_state_authority, decode_hand
from training.concept_bottleneck import (
    NUM_ACTIONS, OBS_SIZE, build_net, trunk_and_concept_param_mask,
)
from training.concept_prevalence import count_positives, sampled_rate
from training.prepare_sl_data import no_play_traj

GAME = pyspiel.load_game("bridge(use_double_dummy_result=false)")
MIN_ACTION = 52
STATE_AUTH = create_sayc_state_authority()
HARD_STATE = HARD_STATE_CONCEPTS
NUM_HARD = len(HARD_STATE)


class SLConfig(BaseModel):
    iterations: int = 15_000
    train_batch: int = 256
    learning_rate: float = 1e-4
    # "cosine": linear warmup over warmup_steps, then cosine decay to 0 at
    # `iterations`. weight_decay > 0 switches Adam -> AdamW.
    lr_schedule: Literal["constant", "cosine"] = "constant"
    warmup_steps: int = 0
    weight_decay: float = 0.0

    # Exactly one of these: `data_path` = directory with train.txt, sampled
    # live through pyspiel (~20k samples/s); or
    # `prepared_path` = output of training/prepare_sl_data.py (same decisions,
    # precomputed and bit-packed, GPU-bound instead of data-bound).
    data_path: str | None = None
    prepared_path: str | None = None
    save_path: str | None = None
    rng_seed: int = 42

    # Held-out validation (prepared_path/valid) every eval_every iterations
    # (0 = never). Writes save_path/metrics.jsonl, keeps params-best.pkl =
    # highest validation bid accuracy, and stops early after
    # early_stop_patience evals without improvement (0 = never). In
    # "sequential" mode neither best-tracking nor patience starts until
    # phase B -- the policy isn't trained during phase A.
    eval_every: int = 0
    early_stop_patience: int = 0

    actor_model_type: Literal["FAIR", "DeepMind"] = "DeepMind"
    actor_activation: str = "relu"

    bottleneck_mode: Literal["joint", "independent", "sequential"] = "joint"
    # "sequential" only: iterations 0..concept_phase_iterations train the
    # concept head exclusively (policy loss weight forced to 0); iterations
    # after that train policy_head exclusively (concept loss weight forced
    # to 0, trunk/concept-head gradients additionally zero-masked). Ignored
    # for "joint"/"independent".
    concept_phase_iterations: int = 0

    hard_coef: float = 4.2
    reweight_concepts: bool = True
    concept_freq_ema_decay: float = 0.99
    # p in the reweighting: "ema" = running average of the batch positive
    # rates from 0.05 (what the thesis models were trained with); "exact" =
    # fixed at the sampler's exact positive rate, counted from
    # prepared_path/train's labels at startup (training/concept_prevalence.py).
    concept_freq: Literal["ema", "exact"] = "ema"
    # "joint" mode only: in independent/sequential mode
    # concept_only_policy_head has policy_head's input and is exported as
    # policy_head (_export_params), so the coefficient is forced to 0.
    concept_only_coef: float = 0.1

    num_data_workers: int = 4
    data_queue_size: int = 8

    print_every: int = 200
    save_every: int = 5000  # periodic snapshots; 0 = only best/final


cfg = SLConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
if (cfg.data_path is None) == (cfg.prepared_path is None):
    raise ValueError("set exactly one of data_path / prepared_path")
if cfg.eval_every and not cfg.prepared_path:
    raise ValueError("eval_every needs prepared_path (validation split lives there)")
if cfg.concept_freq == "exact" and not cfg.prepared_path:
    raise ValueError("concept_freq=exact needs prepared_path (the rates are counted from its labels)")


# ============================================================
# Dataset: live sampling through pyspiel (data_path), or the prepared,
# bit-packed splits (prepared_path)
# ============================================================

def make_dataset(data_dir: str):
    trajs = [no_play_traj(l) for l in open(os.path.join(data_dir, "train.txt"))]

    while True:
        np.random.shuffle(trajs)
        for t in trajs:
            idx = np.random.randint(52, len(t))
            state = GAME.new_initial_state()
            bids: List[str] = []

            for a in t[:idx]:
                if not state.is_chance_node():
                    bids.append(state.action_to_string(a))
                state.apply_action(a)

            obs = np.asarray(state.observation_tensor()[4:484], np.float32)
            action = t[idx] - MIN_ACTION

            legal = np.zeros(NUM_ACTIONS, dtype=np.bool_)
            legal[np.array(state.legal_actions()) - MIN_ACTION] = True

            auction = AuctionContext(
                bids=bids,
                bid=state.action_to_string(action + MIN_ACTION),
                dealer=0,
            )

            hand = decode_hand(obs)
            hard_dict = STATE_AUTH.classify(auction, hand)
            hard_target = np.array([hard_dict[n] for n in HARD_STATE], np.float32)

            yield obs, action, legal, hard_target


def batch(ds, n):
    while True:
        xs = [next(ds) for _ in range(n)]
        yield tuple(np.stack(x) for x in zip(*xs))


def _batch_worker(data_dir: str, batch_size: int, seed: int, out_queue: "mp.Queue"):
    np.random.seed(seed)
    gen = batch(make_dataset(data_dir), batch_size)
    while True:
        out_queue.put(next(gen))


def make_parallel_batches(data_dir: str, batch_size: int, num_workers: int,
                           queue_size: int, base_seed: int):
    if num_workers <= 0:
        yield from batch(make_dataset(data_dir), batch_size)
        return
    ctx = mp.get_context("fork")
    q: "mp.Queue" = ctx.Queue(maxsize=queue_size)
    workers = [
        ctx.Process(target=_batch_worker, args=(data_dir, batch_size, base_seed + i, q), daemon=True)
        for i in range(num_workers)
    ]
    for w in workers:
        w.start()
    try:
        while True:
            yield q.get()
    finally:
        for w in workers:
            w.terminate()


class PackedSplit:
    """One split written by training/prepare_sl_data.py, held in RAM packed."""

    def __init__(self, split_dir: str):
        self.obs_bits = np.load(os.path.join(split_dir, "obs_bits.npy"))
        self.legal_bits = np.load(os.path.join(split_dir, "legal_bits.npy"))
        self.target_bits = np.load(os.path.join(split_dir, "target_bits.npy"))
        self.action = np.load(os.path.join(split_dir, "action.npy")).astype(np.int32)
        self.offsets = np.load(os.path.join(split_dir, "deal_offsets.npy"))
        self.counts = np.diff(self.offsets)
        self.num_deals = len(self.counts)

    def rows(self, rows):
        obs = np.unpackbits(self.obs_bits[rows], axis=1)[:, :OBS_SIZE].astype(np.float32)
        legal = np.unpackbits(self.legal_bits[rows], axis=1)[:, :NUM_ACTIONS].astype(np.bool_)
        tgt = np.unpackbits(self.target_bits[rows], axis=1)[:, :NUM_HARD].astype(np.float32)
        return obs, self.action[rows], legal, tgt


def make_packed_batches(split: PackedSplit, batch_size: int, seed: int, prefetch: int = 8):
    """Same sampling scheme as make_dataset(): deals visited in a fresh random
    order every epoch, one uniformly random decision per deal visit."""
    import queue
    import threading

    q: "queue.Queue" = queue.Queue(maxsize=prefetch)

    def producer():
        rng = np.random.RandomState(seed)
        buf = np.empty(0, dtype=np.int64)
        while True:
            while len(buf) < batch_size:
                buf = np.concatenate([buf, rng.permutation(split.num_deals)])
            deals, buf = buf[:batch_size], buf[batch_size:]
            k = (rng.random_sample(batch_size) * split.counts[deals]).astype(np.int64)
            q.put(split.rows(split.offsets[deals] + k))

    threading.Thread(target=producer, daemon=True).start()
    while True:
        yield q.get()


def evaluate_split(params, split: PackedSplit, chunk: int = 8192) -> dict:
    """Held-out metrics over EVERY decision of the split. Faithfulness
    metrics: concept macro/micro-F1 vs the rule engine (are the concepts
    the model states true?) and, for joint mode, how often the
    concept-only head picks the same bid as the full policy (does the
    concept account explain the actual decision?)."""
    n = len(split.action)
    correct = co_correct = agree = 0
    nll_sum = 0.0
    tp = np.zeros(NUM_HARD); fp = np.zeros(NUM_HARD); fn = np.zeros(NUM_HARD)
    for s in range(0, n, chunk):
        obs, action, legal, tgt = split.rows(np.arange(s, min(s + chunk, n)))
        out = _eval_forward(params, obs)
        pl = np.asarray(out["policy_logits"]); col = np.asarray(out["concept_only_logits"])
        cl = np.asarray(out["concept_logits"])
        pl_m = np.where(legal, pl, -np.inf); co_m = np.where(legal, col, -np.inf)
        pred, co_pred = pl_m.argmax(1), co_m.argmax(1)
        correct += int((pred == action).sum()); co_correct += int((co_pred == action).sum())
        agree += int((pred == co_pred).sum())
        z = pl - pl.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll_sum += float(-logp[np.arange(len(action)), action].sum())
        pb, gt = cl > 0.0, tgt > 0.5  # sigmoid(logit) >= 0.5  <=>  logit >= 0
        tp += (pb & gt).sum(0); fp += (pb & ~gt).sum(0); fn += (~pb & gt).sum(0)
    has_pos = (tp + fn) > 0
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return {
        "val_acc": correct / n, "val_nll": nll_sum / n,
        "concept_macro_f1": float(f1[has_pos].mean()),
        "concept_micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
        "concept_only_acc": co_correct / n, "concept_policy_agreement": agree / n,
        "n_decisions": n,
    }


# ============================================================
# Model
# ============================================================

net = build_net(cfg.bottleneck_mode, cfg.actor_model_type, cfg.actor_activation, NUM_HARD, has_value=False)
_eval_forward = jax.jit(net.apply)


def _make_optimizer():
    if cfg.lr_schedule == "cosine":
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=cfg.learning_rate,
            warmup_steps=max(cfg.warmup_steps, 1), decay_steps=max(cfg.iterations, cfg.warmup_steps + 1),
            end_value=0.0,
        )
    else:
        lr = cfg.learning_rate
    base = optax.adamw(lr, weight_decay=cfg.weight_decay) if cfg.weight_decay > 0 else optax.adam(lr)
    return optax.chain(optax.clip_by_global_norm(1.0), base)


opt = _make_optimizer()


def loss_fn(params, concept_freq_ema, batch_, concept_loss_weight, policy_loss_weight):
    obs, action, legal, hard_t = batch_
    out = net.apply(params, obs)
    policy_logits, hard_logits, concept_only_logits = (
        out["policy_logits"], out["concept_logits"], out["concept_only_logits"],
    )

    logp = jax.nn.log_softmax(policy_logits)
    nll = -jnp.mean(jnp.take_along_axis(logp, action[:, None], axis=1))

    masked_logits = jnp.where(legal, policy_logits, jnp.finfo(jnp.float32).min)
    acc = jnp.mean(jnp.argmax(masked_logits, axis=1) == action)

    log_probs_masked = jax.nn.log_softmax(masked_logits)
    probs_masked = jnp.exp(log_probs_masked)
    entropy = -jnp.sum(jnp.where(legal, probs_masked * log_probs_masked, 0.0), axis=-1).mean()

    per_concept_bce = optax.sigmoid_binary_cross_entropy(hard_logits, hard_t).mean(axis=0)
    if cfg.reweight_concepts:
        p = jnp.clip(concept_freq_ema, 1e-4, 1 - 1e-4)
        weight = 1.0 / jnp.sqrt(p * (1 - p))
        weight = weight / jnp.mean(weight)
    else:
        weight = jnp.ones_like(per_concept_bce)
    hard_loss = jnp.mean(per_concept_bce * weight)

    concept_only_coef = cfg.concept_only_coef if cfg.bottleneck_mode == "joint" else 0.0
    if cfg.bottleneck_mode != "joint":
        # the exported concept-only head IS policy_head in these modes
        # (_export_params); report that, not the untrained raw head
        concept_only_logits = policy_logits
    co_logp = jax.nn.log_softmax(concept_only_logits)
    concept_only_nll = -jnp.mean(jnp.take_along_axis(co_logp, action[:, None], axis=1))
    co_masked = jnp.where(legal, concept_only_logits, jnp.finfo(jnp.float32).min)
    concept_only_acc = jnp.mean(jnp.argmax(co_masked, axis=1) == action)

    total = (
        policy_loss_weight * nll
        + concept_loss_weight * cfg.hard_coef * hard_loss
        + concept_only_coef * concept_only_nll
    )

    batch_pos_rate = jnp.mean(hard_t, axis=0)
    return total, {
        "loss": total, "nll": nll, "acc": acc, "entropy": entropy,
        "hard": hard_loss, "concept_only_acc": concept_only_acc,
        "batch_pos_rate": batch_pos_rate,
    }


def make_step_fn(concept_loss_weight: float, policy_loss_weight: float, freeze_concepts: bool):
    """`freeze_concepts` zero-masks the parameter update into the
    trunk/hard_state_head (trunk_and_concept_param_mask). A zero gradient
    alone is not enough: Adam's moment state carries phase A's momentum
    forward and keeps moving the "frozen" params for several steps. Masking
    the update keeps them byte-identical (tests/test_concept_bottleneck.py)."""

    @jax.jit
    def step_fn(params, opt_state, concept_freq_ema, batch_):
        (loss, m), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, concept_freq_ema, batch_, concept_loss_weight, policy_loss_weight,
        )
        updates, opt_state = opt.update(grads, opt_state, params)  # params: AdamW (weight_decay > 0) needs them
        if freeze_concepts:
            mask = trunk_and_concept_param_mask(params)
            updates = jax.tree_util.tree_map(lambda u, mk: u * mk, updates, mask)
        params = optax.apply_updates(params, updates)
        new_ema = (cfg.concept_freq_ema_decay * concept_freq_ema
                   + (1 - cfg.concept_freq_ema_decay) * m["batch_pos_rate"]
                   if cfg.concept_freq == "ema" else concept_freq_ema)
        return params, opt_state, new_ema, m

    return step_fn


# ============================================================
# Train
# ============================================================

def _export_params(params):
    """In "independent"/"sequential" mode concept_only_policy_head is never
    trained (concept_only_coef is forced to 0 in loss_fn) and would stay at
    its random init -- yet explain/ (trust flag, concept-flip
    counterfactuals, walkthrough's per-concept contributions) and
    nlg/explanation_frames.py read ITS weights as the concepts-to-bid map. Both heads
    take the identical input (stop_gradient(concept_acts)), so the faithful
    concepts-to-bid map in these modes is policy_head itself: export it
    under both names."""
    if cfg.bottleneck_mode == "joint":
        return params
    return {**params, "concept_only_policy_head": params["policy_head"]}


def _save(params, name, extra=None):
    os.makedirs(cfg.save_path, exist_ok=True)
    params = _export_params(params)
    blob = {"params": params, "hard_concepts": HARD_STATE, "bottleneck_mode": cfg.bottleneck_mode,
            "model_type": cfg.actor_model_type, "activation": cfg.actor_activation}
    if extra:
        blob.update(extra)
    tmp = os.path.join(cfg.save_path, name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(blob, f)
    os.replace(tmp, os.path.join(cfg.save_path, name))  # never leave a half-written checkpoint


def _check_prepared_labels(prepared_path: str):
    """Refuse to train on prepared data whose concept labels were computed
    for a different vocabulary: the targets are precomputed, so their columns
    must be the code's HARD_STATE_CONCEPTS in the same order."""
    meta_path = os.path.join(prepared_path, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if meta.get("hard_concepts") != list(HARD_STATE):
        raise SystemExit(f"[sl] {prepared_path} is stale (vocabulary: prepared "
                         f"{len(meta.get('hard_concepts') or [])} concepts, code has {NUM_HARD}). Re-run "
                         f"`python -m training.prepare_sl_data --train ... --test ... --out {prepared_path}`.")


def main():
    if cfg.prepared_path:
        _check_prepared_labels(cfg.prepared_path)
        train_split = PackedSplit(os.path.join(cfg.prepared_path, "train"))
        valid_split = PackedSplit(os.path.join(cfg.prepared_path, "valid")) if cfg.eval_every else None
        ds = make_packed_batches(train_split, cfg.train_batch, cfg.rng_seed)
        print(f"[sl] prepared data: {train_split.num_deals:,} train deals / {len(train_split.action):,} decisions"
              + (f", {len(valid_split.action):,} validation decisions" if valid_split else ""), flush=True)
    else:
        valid_split = None
        ds = make_parallel_batches(cfg.data_path, cfg.train_batch, cfg.num_data_workers,
                                    cfg.data_queue_size, cfg.rng_seed)
    obs0, _, _, _ = next(ds)
    metrics_f = None
    if cfg.save_path and cfg.eval_every:
        os.makedirs(cfg.save_path, exist_ok=True)
        metrics_f = open(os.path.join(cfg.save_path, "metrics.jsonl"), "a")
    best_acc, best_eval, evals_since_best, stopped_early = -1.0, None, 0, False

    params = net.init(jax.random.PRNGKey(cfg.rng_seed), obs0)
    opt_state = opt.init(params)
    concept_freq_ema = jnp.full((NUM_HARD,), 0.05)
    if cfg.concept_freq == "exact":
        rate = sampled_rate(count_positives(train_split.target_bits, train_split.offsets, NUM_HARD))
        concept_freq_ema = jnp.asarray(rate, jnp.float32)
        print(f"[sl] concept p fixed at the exact sampled rates: {(rate < 0.01).sum()} of {NUM_HARD} "
              f"under 1%, {(rate == 0).sum()} never true", flush=True)

    if cfg.bottleneck_mode == "sequential":
        phase_a = make_step_fn(concept_loss_weight=1.0, policy_loss_weight=0.0, freeze_concepts=False)
        phase_b = make_step_fn(concept_loss_weight=0.0, policy_loss_weight=1.0, freeze_concepts=True)
        print(f"[sequential] phase A (concepts only): iterations 0-{cfg.concept_phase_iterations}")
        print(f"[sequential] phase B (policy only, concepts frozen): "
              f"iterations {cfg.concept_phase_iterations}-{cfg.iterations}")
    else:
        step_fn_always = make_step_fn(concept_loss_weight=1.0, policy_loss_weight=1.0, freeze_concepts=False)

    rolling = {k: deque(maxlen=50) for k in ("acc", "nll", "entropy", "hard", "concept_only_acc")}

    for it in range(cfg.iterations):
        if cfg.bottleneck_mode == "sequential":
            step_fn = phase_a if it < cfg.concept_phase_iterations else phase_b
        else:
            step_fn = step_fn_always

        params, opt_state, concept_freq_ema, m = step_fn(params, opt_state, concept_freq_ema, next(ds))

        for k in rolling:
            rolling[k].append(m[k])  # stays on device; converting every step would force a GPU sync

        if it % cfg.print_every == 0:
            avg = {k: float(sum(float(x) for x in v)) / len(v) for k, v in rolling.items()}
            phase_tag = ""
            if cfg.bottleneck_mode == "sequential":
                phase_tag = "A" if it < cfg.concept_phase_iterations else "B"
                phase_tag = f" phase={phase_tag}"
            print(
                f"[{it:06d}]{phase_tag} mode={cfg.bottleneck_mode} "
                f"acc={avg['acc']*100:5.1f}%  nll={avg['nll']:.3f}  ent={avg['entropy']:.3f}  "
                f"hard={avg['hard']:.3f}  concept_only_acc={avg['concept_only_acc']*100:5.1f}%"
            )

        if cfg.save_path and cfg.save_every > 0 and it > 0 and it % cfg.save_every == 0:
            _save(params, f"params-{it}.pkl")

        is_last = it == cfg.iterations - 1
        if valid_split is not None and it > 0 and (it % cfg.eval_every == 0 or is_last):
            ev = evaluate_split(_export_params(params), valid_split)
            ev["iteration"] = it
            in_policy_phase = cfg.bottleneck_mode != "sequential" or it >= cfg.concept_phase_iterations
            improved = in_policy_phase and ev["val_acc"] > best_acc
            if improved:
                best_acc, best_eval, evals_since_best = ev["val_acc"], ev, 0
                if cfg.save_path:
                    _save(params, "params-best.pkl", {"val_metrics": ev})
            elif in_policy_phase:
                evals_since_best += 1
            print(f"[eval {it:06d}] val_acc={ev['val_acc']*100:5.2f}%  val_nll={ev['val_nll']:.3f}  "
                  f"concept_F1 macro={ev['concept_macro_f1']:.3f} micro={ev['concept_micro_f1']:.3f}  "
                  f"concept_only_acc={ev['concept_only_acc']*100:5.1f}%  "
                  f"agree={ev['concept_policy_agreement']*100:5.1f}%{'  *best*' if improved else ''}", flush=True)
            if metrics_f:
                metrics_f.write(json.dumps(ev) + "\n"); metrics_f.flush()
            if cfg.early_stop_patience and evals_since_best >= cfg.early_stop_patience:
                print(f"[sl] early stop at iteration {it}: no val_acc gain in "
                      f"{cfg.early_stop_patience} evals", flush=True)
                stopped_early = True
                break

    if cfg.save_path:
        _save(params, "params-final.pkl")
        print(f"[sl] saved final checkpoint to {cfg.save_path}/params-final.pkl")
        if valid_split is not None:
            with open(os.path.join(cfg.save_path, "summary.json"), "w") as f:
                json.dump({"config": cfg.model_dump(), "best": best_eval, "last_iteration": it,
                           "stopped_early": stopped_early, "completed": True}, f, indent=2)


if __name__ == "__main__":
    main()
