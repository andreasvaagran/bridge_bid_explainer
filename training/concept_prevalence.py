#!/usr/bin/env python3
"""The exact positive rate of every concept label in a prepared SL split, counted from the stored
target bits, and the rates in the batches training actually draws."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import time

import numpy as np

CLIP = 1e-4  # sl.loss_fn clips p to [CLIP, 1 - CLIP]


def load_split(split_dir: str):
    """(target_bits, deal_offsets) of one prepared split; the bits stay on disk."""
    return (np.load(os.path.join(split_dir, "target_bits.npy"), mmap_mode="r"),
            np.load(os.path.join(split_dir, "deal_offsets.npy")))


def count_positives(target_bits, offsets, num_concepts: int, chunk_deals: int = 100_000) -> dict:
    """Positives per concept, grouped by deal length. A deal's decisions are
    contiguous rows, so per-deal sums are one reduceat (in uint8: a deal has
    at most a few dozen decisions), and a one-hot matmul sums the deals of
    each length."""
    offsets = np.asarray(offsets, np.int64)
    n_dec = np.diff(offsets)
    lengths, length_idx, deals_by_length = np.unique(n_dec, return_inverse=True, return_counts=True)
    if lengths.max() > 255:
        raise ValueError(f"deal with {lengths.max()} decisions overflows the uint8 per-deal sums")
    by_length = np.zeros((len(lengths), num_concepts), np.int64)
    deals_with_positive = np.zeros(num_concepts, np.int64)
    for a in range(0, len(n_dec), chunk_deals):
        b = min(a + chunk_deals, len(n_dec))
        r0, r1 = offsets[a], offsets[b]
        bits = np.unpackbits(np.asarray(target_bits[r0:r1]), axis=1)[:, :num_concepts]
        per_deal = np.add.reduceat(bits, offsets[a:b] - r0, axis=0, dtype=np.uint8)
        deals_with_positive += (per_deal > 0).sum(0)
        onehot = np.zeros((len(lengths), b - a))
        onehot[length_idx[a:b], np.arange(b - a)] = 1.0
        by_length += np.rint(onehot @ per_deal).astype(np.int64)  # float64 matmul: exact for these sizes
    return {"lengths": lengths, "deals_by_length": deals_by_length, "positives_by_length": by_length,
            "deals_with_positive": deals_with_positive}


def decision_rate(c: dict) -> np.ndarray:
    return c["positives_by_length"].sum(0) / (c["lengths"] * c["deals_by_length"]).sum()


def sampled_rate(c: dict) -> np.ndarray:
    return (c["positives_by_length"] / c["lengths"][:, None]).sum(0) / c["deals_by_length"].sum()


def loss_weights(p) -> np.ndarray:
    """sl.loss_fn's per-concept weights for rates p (last axis = concepts)."""
    q = np.clip(p, CLIP, 1 - CLIP)
    w = 1.0 / np.sqrt(q * (1 - q))
    return w / w.mean(axis=-1, keepdims=True)


def replay_ema(target_bits, offsets, num_concepts: int, batch: int, seed: int, iterations: int,
               decay: float = 0.99, init: float = 0.05, every: int = 10) -> np.ndarray:
    """The running p of a training run, every `every` iterations: the exact
    batch sequence of sl.make_packed_batches(seed), minus the first batch
    that sl.main uses for net.init, through sl's EMA update (float32 as on
    the device). Labels only, no network: ~40 s for 120k batches of 1024."""
    target_bits = np.asarray(target_bits)
    offsets = np.asarray(offsets, np.int64)
    counts = np.diff(offsets)
    rng = np.random.RandomState(seed)
    buf = np.empty(0, dtype=np.int64)

    def next_rate():
        nonlocal buf
        while len(buf) < batch:
            buf = np.concatenate([buf, rng.permutation(len(counts))])
        deals, buf = buf[:batch], buf[batch:]
        k = (rng.random_sample(batch) * counts[deals]).astype(np.int64)
        rows = offsets[deals] + k
        return (np.unpackbits(target_bits[rows], axis=1)[:, :num_concepts].sum(0) / batch).astype(np.float32)

    next_rate()
    ema = np.full(num_concepts, init, np.float32)
    out = np.empty((iterations // every, num_concepts), np.float32)
    for it in range(iterations):
        ema = np.float32(decay) * ema + np.float32(1 - decay) * next_rate()
        if (it + 1) % every == 0:
            out[(it + 1) // every - 1] = ema
    return out


def _replay_summary(traj: np.ndarray, p: np.ndarray, names: list[str], every: int) -> dict:
    w_exact = loss_weights(p)
    err = np.abs(loss_weights(traj.astype(np.float64)) / w_exact - 1)  # (snapshots, concepts)
    worst = err.max(1)
    its = (np.arange(len(traj)) + 1) * every
    warm = min(len(traj) - 1, 2000 // every)
    p95 = np.percentile(err[warm:], 95, axis=0)
    return {
        "median_weight_error_at": {int(i): float(np.median(err[i // every - 1]))
                                   for i in (100, 500, 1000, 2000, 5000) if i // every <= len(traj)},
        "median_weight_error_after_2000": float(np.median(err[warm:])),
        "median_concept_p95_weight_error_after_2000": float(np.median(p95)),
        "last_iteration_worst_error_over_50pct": int(its[worst > 0.5][-1]) if (worst > 0.5).any() else 0,
        "noisiest_after_2000": [{"concept": names[i], "sampled_rate": float(p[i]),
                                 "p95_weight_error": float(p95[i]), "max_weight_error": float(err[warm:, i].max())}
                                for i in np.argsort(-p95)[:10]],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prepared", help="output directory of training/prepare_sl_data.py")
    ap.add_argument("--out", default=None, help="JSON report (default <prepared>/concept_prevalence.json)")
    ap.add_argument("--replay_seed", type=int, default=None, help="replay this rng_seed's training batches")
    ap.add_argument("--iterations", type=int, default=120_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--decay", type=float, default=0.99)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.prepared, "meta.json")))
    names = meta["hard_concepts"]
    C = len(names)
    report = {"prepared": os.path.abspath(args.prepared), "splits": {}}

    for split in ("train", "valid", "test"):
        if not os.path.isdir(os.path.join(args.prepared, split)):
            continue
        t0 = time.time()
        c = count_positives(*load_split(os.path.join(args.prepared, split)), C)
        dr, sr = decision_rate(c), sampled_rate(c)
        pos = c["positives_by_length"].sum(0)
        n_deals, n_dec = int(c["deals_by_length"].sum()), int((c["lengths"] * c["deals_by_length"]).sum())
        print(f"[{split}] {n_deals:,} deals, {n_dec:,} decisions, counted in {time.time() - t0:.1f}s")
        report["splits"][split] = {
            "deals": n_deals, "decisions": n_dec,
            "concepts": {n: {"positives": int(pos[i]), "deals_with_positive": int(c["deals_with_positive"][i]),
                             "decision_rate": float(dr[i]), "sampled_rate": float(sr[i])}
                         for i, n in enumerate(names)},
        }
        if split == "train":
            p = sr

    tr = report["splits"]["train"]
    dr = np.array([v["decision_rate"] for v in tr["concepts"].values()])
    pos = np.array([v["positives"] for v in tr["concepts"].values()])
    w = loss_weights(p)
    at_clip = p < CLIP
    print(f"\n{'concept':<50} {'positives':>10} {'decisions':>10} {'sampled':>10} {'weight':>7}")
    for i in np.argsort(dr):
        print(f"{names[i]:<50} {pos[i]:>10,} {dr[i]:>10.4%} {p[i]:>10.4%} {w[i]:>7.2f}")
    print("\nconcepts true on fewer than ... of training decisions (decision rate / sampled rate):")
    for t in (0.01, 0.005, 0.001, 0.0001):
        print(f"  {t:>7.2%}: {(dr < t).sum():>3} / {(p < t).sum():>3} of {C}")
    nz = np.where(dr > 0)[0]
    ratio = p[nz] / dr[nz]
    print(f"sampled/decision rate ranges from {ratio.min():.2f}x ({names[nz[ratio.argmin()]]})"
          f" to {ratio.max():.2f}x ({names[nz[ratio.argmax()]]})")
    print(f"never true in training: {[names[i] for i in np.where(pos == 0)[0]]}")
    print(f"{at_clip.sum()} concepts are below the {CLIP} clip and share the top weight {w.max():.2f}; "
          f"they get {w[at_clip].sum() / w.sum():.1%} of the concept-loss weight "
          f"({w[pos == 0].sum() / w.sum():.1%} on the never-true ones)")
    report["train_loss_weights_exact"] = {n: float(w[i]) for i, n in enumerate(names)}

    if args.replay_seed is not None:
        t0 = time.time()
        traj = replay_ema(*load_split(os.path.join(args.prepared, "train")), C, args.batch,
                          args.replay_seed, args.iterations, args.decay)
        s = _replay_summary(traj, p, names, 10)
        s.update(seed=args.replay_seed, iterations=args.iterations, batch=args.batch, decay=args.decay)
        report["replay"] = s
        print(f"\n[replay seed {args.replay_seed}, {args.iterations:,} x {args.batch}, {time.time() - t0:.0f}s] "
              f"loss weights from the running p vs the exact p:")
        for it, e in s["median_weight_error_at"].items():
            print(f"  iteration {it:>5}: median concept off by {e:.1%}")
        print(f"  after iteration 2000: median {s['median_weight_error_after_2000']:.2%}; a typical concept's "
              f"95th-percentile error {s['median_concept_p95_weight_error_after_2000']:.1%}")
        for r in s["noisiest_after_2000"][:6]:
            print(f"    {r['concept']:<45} p={r['sampled_rate']:.1e}  95th pct {r['p95_weight_error']:.1%}"
                  f"  max {r['max_weight_error']:.1%}")

    out = args.out or os.path.join(args.prepared, "concept_prevalence.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
