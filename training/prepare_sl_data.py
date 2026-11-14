#!/usr/bin/env python3
"""Precomputes every decision of the OpenSpiel WBridge5 dataset once (observation, call, legal calls,
the rule engine's concept labels) into bit-packed arrays for training/sl.py. Rerun after any rule change."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

OBS_DIM, N_ACTIONS, MIN_ACTION = 480, 38, 52
_GAME = _AUTH = _HARD = None


def _init_worker():
    global _GAME, _AUTH, _HARD
    import pyspiel
    from concepts import HARD_STATE_CONCEPTS, create_sayc_state_authority
    _GAME = pyspiel.load_game("bridge(use_double_dummy_result=false)")
    _AUTH = create_sayc_state_authority()
    _HARD = HARD_STATE_CONCEPTS


def no_play_traj(line: str):  # also used by training/sl.py
    a = [int(x) for x in line.split()]
    return tuple(a[:-52]) if len(a) > 56 else tuple(a)


def deal_decisions(t):
    """All bidding decisions of one trajectory, each computed exactly as
    training/sl.py's make_dataset() computes the decision at index idx."""
    from concepts import AuctionContext, decode_hand
    state = _GAME.new_initial_state()
    bids = []
    out = []
    for idx, a in enumerate(t):
        if idx >= 52:
            obs = np.asarray(state.observation_tensor()[4:484], np.float32)
            action = a - MIN_ACTION
            legal = np.zeros(N_ACTIONS, dtype=np.bool_)
            legal[np.array(state.legal_actions()) - MIN_ACTION] = True
            auction = AuctionContext(bids=list(bids), bid=state.action_to_string(action + MIN_ACTION), dealer=0)
            hard = _AUTH.classify(auction, decode_hand(obs))
            out.append((obs, action, legal, np.array([hard[n] for n in _HARD], np.float32)))
        if not state.is_chance_node():
            bids.append(state.action_to_string(a))
        state.apply_action(a)
    return out


def _process_chunk(lines):
    obs_b, legal_b, tgt_b, act, counts = [], [], [], [], []
    for line in lines:
        dec = deal_decisions(no_play_traj(line))
        counts.append(len(dec))
        for obs, a, legal, tgt in dec:
            obs_b.append(np.packbits(obs.astype(np.uint8)))
            legal_b.append(np.packbits(legal.astype(np.uint8)))
            tgt_b.append(np.packbits(tgt.astype(np.uint8)))
            act.append(a)
    # target width follows the vocabulary
    from concepts import HARD_STATE_CONCEPTS
    tgt_bytes = (len(HARD_STATE_CONCEPTS) + 7) // 8
    return (np.array(obs_b, np.uint8).reshape(-1, 60), np.array(legal_b, np.uint8).reshape(-1, 5),
            np.array(tgt_b, np.uint8).reshape(-1, tgt_bytes), np.array(act, np.uint8), np.array(counts, np.int64))


def build_split(lines, out_dir, pool, chunk=2000):
    t0 = time.time()
    chunks = [lines[i:i + chunk] for i in range(0, len(lines), chunk)]
    parts = []
    for i, part in enumerate(pool.imap(_process_chunk, chunks)):
        parts.append(part)
        if i % 50 == 0 or i == len(chunks) - 1:
            done = min((i + 1) * chunk, len(lines))
            print(f"  [{out_dir}] {done:,}/{len(lines):,} deals ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    counts = np.concatenate([p[4] for p in parts])
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    for k, name in enumerate(("obs_bits", "legal_bits", "target_bits", "action")):
        np.save(os.path.join(out_dir, f"{name}.npy"), np.concatenate([p[k] for p in parts]))
    np.save(os.path.join(out_dir, "deal_offsets.npy"), offsets)
    return {"deals": len(lines), "decisions": int(offsets[-1])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--valid_deals", type=int, default=20_000)
    p.add_argument("--limit_train", type=int, default=None, help="(testing) only use the first N shuffled deals")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = p.parse_args()

    train_lines = [l for l in open(args.train) if l.strip()]
    test_lines = [l for l in open(args.test) if l.strip()]
    order = np.random.RandomState(0).permutation(len(train_lines))
    if args.limit_train:
        order = order[:args.limit_train]
    valid_idx, train_idx = order[-args.valid_deals:], order[:-args.valid_deals]

    from concepts import HARD_STATE_CONCEPTS
    meta = {"source_train": os.path.abspath(args.train), "source_test": os.path.abspath(args.test),
            "split_seed": 0, "num_concepts": len(HARD_STATE_CONCEPTS),
            "hard_concepts": list(HARD_STATE_CONCEPTS), "splits": {}}
    with mp.get_context("fork").Pool(args.workers, initializer=_init_worker) as pool:
        for name, lines in (("valid", [train_lines[i] for i in valid_idx]),
                            ("test", test_lines),
                            ("train", [train_lines[i] for i in train_idx])):
            meta["splits"][name] = build_split(lines, os.path.join(args.out, name), pool)
            print(f"[prepare] {name}: {meta['splits'][name]}", flush=True)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare] done -> {args.out}")


if __name__ == "__main__":
    main()
