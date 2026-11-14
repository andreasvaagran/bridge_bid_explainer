#!/usr/bin/env python3
"""Builds the Qwen explainer's training data. It takes positions from a prepared WBridge5 split (or a
checkpoint's own self-play), stratified per bidding situation, and lets the model decide. The teacher explains each
decision, and an explanation is kept only if it passes the checks. Output: train/val/test (--labels,
--explanations) JSONL pairs; --rerender rewrites the explanations from stored frames after a teacher change."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if "--rerender" in _sys.argv:
    # text only -- and pgx runs a JAX op at import time, so without this JAX preallocates
    # most of the GPU before main() starts
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import collections
import json
import random
import time
from pathlib import Path

import jax
import numpy as np

from concepts.auction_obs import auction_from_obs
from training.checkpoint import load_checkpoint
from training.concept_bottleneck import NUM_ACTIONS, OBS_SIZE

from nlg.consistency_checks import check_explanation
from nlg.explanation_frames import BIDS, Enricher, decision_frame
from nlg.prompt import check_style
from nlg.explanation_inputs import bid_category
from nlg.teacher import explain

SEATS = "NESW"  # OpenSpiel bridge: North deals every board of the WBridge5 data
VUL_NAMES = {(False, False): "None", (True, False): "NS", (False, True): "EW", (True, True): "Both"}


def _light_frame(calls, bid, obs, dealer: int = 0):
    """Just enough of a frame to categorise the decision (no model features)."""
    n = len(calls)
    you = SEATS[(dealer + n) % 4]
    return {"table": {"you": you, "partner": SEATS[(dealer + n + 2) % 4]},
            "auction_before": [{"seat": SEATS[(dealer + i) % 4], "bid": c} for i, c in enumerate(calls)],
            "bid": bid, "hand_str": "", "dealer": SEATS[dealer]}


def selfplay_positions(ck, n_deals: int, dds_path: str, random_vulnerability: bool, seed: int,
                       batch: int = 1024):
    """Every bidding decision of `n_deals` pgx deals bid by the checkpoint's own
    greedy policy at all four seats, in the prepared-split format
    (obs_bits, legal_bits, deal_offsets) plus each deal's real dealer and
    vulnerability. For PPO checkpoints, whose auctions drift from the WBridge5
    ones the prepared split holds (and which train on random vulnerability)."""
    from pgx.bridge_bidding import BridgeBidding, _observe
    import jax.numpy as jnp
    env = BridgeBidding(dds_results_table_path=dds_path)
    init, step = jax.jit(jax.vmap(env.init)), jax.jit(jax.vmap(env.step))
    observe = jax.jit(jax.vmap(_observe))
    fwd = jax.jit(lambda o: ck.net.apply(ck.params, o)["policy_logits"])
    key = jax.random.PRNGKey(seed)
    obs_rows, legal_rows, offsets, dealers, vuls = [], [], [0], [], []
    for start in range(0, n_deals, batch):
        n = min(batch, n_deals - start)
        key, k_init = jax.random.split(key)
        state = init(jax.random.split(k_init, batch))
        if not random_vulnerability:  # the SL data (and so SL checkpoints) never saw vulnerability
            state = state.replace(_vul_NS=jnp.zeros(batch, jnp.bool_), _vul_EW=jnp.zeros(batch, jnp.bool_))
            state = state.replace(observation=observe(state, state.current_player))
        per_obs = [[] for _ in range(n)]
        per_legal = [[] for _ in range(n)]
        for _ in range(60):  # an auction has at most ~40 calls in practice
            term = np.asarray(state.terminated)
            if term[:n].all():
                break
            obs = np.asarray(state.observation)[:, :OBS_SIZE].astype(bool)
            legal = np.asarray(state.legal_action_mask).astype(bool)
            logits = np.asarray(fwd(obs.astype(np.float32)))
            action = np.where(legal, logits, -np.inf).argmax(1)
            for i in np.flatnonzero(~term[:n]):
                per_obs[i].append(obs[i]); per_legal[i].append(legal[i])
            key, k_step = jax.random.split(key)
            state = step(state, jnp.asarray(action), jax.random.split(k_step, batch))
        dl, vns, vew = np.asarray(state._dealer), np.asarray(state._vul_NS), np.asarray(state._vul_EW)
        for i in range(n):
            if not per_obs[i]:
                continue
            obs_rows.append(np.packbits(np.stack(per_obs[i]), axis=1))
            legal_rows.append(np.packbits(np.stack(per_legal[i]), axis=1))
            offsets.append(offsets[-1] + len(per_obs[i]))
            dealers.append(int(dl[i])); vuls.append(VUL_NAMES[(bool(vns[i]), bool(vew[i]))])
        print(f"[selfplay] {start + n:,}/{n_deals:,} deals, {offsets[-1]:,} decisions", flush=True)
    return (np.concatenate(obs_rows), np.concatenate(legal_rows), np.asarray(offsets, np.int64),
            np.asarray(dealers), vuls)


def _render(frame_line: str):
    """teacher + checks for one stored frame -> (frame_line, label | None)."""
    from nlg.explanation_inputs import select_reasoned_concepts
    frame = json.loads(frame_line)
    try:
        text = explain(frame)
    except Exception as e:  # one teacher bug must not stop a long render
        print(f"[rerender] teacher failed on deal {frame.get('deal_id')} step {frame.get('step')}: {e!r}", flush=True)
        return frame_line, None
    if check_explanation(frame, text) or check_style(text):
        return frame_line, None
    return frame_line, {"deal_id": frame["deal_id"], "step": frame["step"], "category": bid_category(frame),
                        "selected_concepts": select_reasoned_concepts(frame), "explanation": text}


def _write_reports(out: Path, stats: dict, per_cat: dict, kept: collections.Counter, n_situations: int):
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    lines = [f"# Teacher dataset samples ({stats['kept']:,} explanations, {n_situations} situations)", ""]
    for c in sorted(per_cat):
        lines.append(f"## {c} ({kept[c]} kept)")
        for f, t in per_cat[c]:
            lines.append(f"- `{' '.join(b['bid'] for b in f['auction_before']) or '(first call)'} -> {f['bid']}` "
                         f"| {f['hand_str']}  \n  {t}")
        lines.append("")
    (out / "samples.md").write_text("\n".join(lines) + "\n")


def rerender(out: Path, workers: int):
    """Re-run only the teacher (and the checks) over the frames already in
    `out`: after a teacher-only change the positions, the model's decisions
    and its reasons are unchanged, so recomputing them (the slow part) is
    unnecessary. Situations are recomputed too (nlg.explanation_inputs.bid_category)."""
    import multiprocessing as mp
    stats = json.loads((out / "stats.json").read_text())
    kept, dropped, per_cat = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for sp in ("train", "val", "test"):
            frame_lines = (out / f"{sp}_frames.jsonl").read_text().splitlines()
            with open(out / f"{sp}_labels.jsonl.tmp", "w") as fl, open(out / f"{sp}_frames.jsonl.tmp", "w") as ff:
                for k, (frame_line, label) in enumerate(pool.imap(_render, frame_lines, chunksize=64)):
                    if label is None:
                        dropped[bid_category(json.loads(frame_line))] += 1
                        continue
                    fl.write(json.dumps(label, ensure_ascii=False) + "\n"); ff.write(frame_line + "\n")
                    cat = label["category"]; kept[cat] += 1
                    if len(per_cat[cat]) < 4:
                        per_cat[cat].append((json.loads(frame_line), label["explanation"]))
                    if k % 10000 == 0:
                        print(f"[rerender] {sp} {k:,}/{len(frame_lines):,} ({time.time() - t0:.0f}s)", flush=True)
            for kind in ("labels", "frames"):
                (out / f"{sp}_{kind}.jsonl.tmp").replace(out / f"{sp}_{kind}.jsonl")
    stats.update({"kept": sum(kept.values()), "dropped_by_checks": sum(dropped.values()), "rerendered": True})
    sits = stats.get("situations", {})
    for c in set(sits) | set(kept):
        sits.setdefault(c, {"seen": 0})
        sits[c].update({"kept": kept[c], "dropped": dropped[c]})
    stats["situations"] = dict(sorted(sits.items()))
    _write_reports(out, stats, per_cat, kept, len([c for c in sits if kept[c]]))
    print(f"[done] re-rendered: kept {stats['kept']:,}, dropped {stats['dropped_by_checks']:,} -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default=None, help="required unless --rerender")
    ap.add_argument("--prepared", default=str(Path(__file__).resolve().parent.parent / "data" / "wbridge5" / "prepared"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_category", type=int, default=1200)
    ap.add_argument("--max_deals", type=int, default=400_000)
    ap.add_argument("--val_frac", type=float, default=0.04)
    ap.add_argument("--test_frac", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_counterfactuals", action="store_true", help="faster; drops the what-if lines")
    ap.add_argument("--rerender", action="store_true",
                    help="only re-run the teacher over the frames already in --out_dir (after a teacher change)")
    ap.add_argument("--workers", type=int, default=max(1, (_os.cpu_count() or 2) - 1))
    ap.add_argument("--selfplay_deals", type=int, default=0,
                    help="take positions from this many self-play deals of --model_path instead of --prepared")
    ap.add_argument("--random_vulnerability", action="store_true",
                    help="self-play with pgx's random vulnerability (PPO checkpoints); default non-vulnerable")
    ap.add_argument("--dds_path", default=str(Path(__file__).resolve().parent.parent / "dds_results" / "dds_results_10M.npy"))
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.rerender:
        rerender(out, args.workers)
        return
    if not args.model_path:
        ap.error("--model_path is required unless --rerender")
    ck = load_checkpoint(args.model_path)
    if args.selfplay_deals:
        obs_bits, legal_bits, offsets, dealer_of, vul_of = selfplay_positions(
            ck, args.selfplay_deals, args.dds_path, args.random_vulnerability, args.seed)
        source = f"selfplay ({args.selfplay_deals:,} deals, {'random' if args.random_vulnerability else 'no'} vulnerability)"
    else:
        d = Path(args.prepared) / args.split
        obs_bits = np.load(d / "obs_bits.npy", mmap_mode="r")
        legal_bits = np.load(d / "legal_bits.npy", mmap_mode="r")
        offsets = np.load(d / "deal_offsets.npy")
        dealer_of, vul_of = None, None  # WBridge5/OpenSpiel: North deals, nobody vulnerable
        source = f"prepared WBridge5 split {d}"
    n_deals = len(offsets) - 1
    en = Enricher(ck)
    fwd = jax.jit(lambda o: ck.net.apply(ck.params, o))
    rng = random.Random(args.seed)
    order = list(range(n_deals)); rng.shuffle(order)
    order = order[:args.max_deals]

    # ---- pass 1: scan deals, model decides, categorise, fill quotas
    t0 = time.time()
    chosen = collections.defaultdict(list)   # category -> [(deal, step)]
    seen = collections.Counter()
    full = set()
    batch_deals = 2048
    for bstart in range(0, len(order), batch_deals):
        deals = order[bstart:bstart + batch_deals]
        rows = np.concatenate([np.arange(offsets[g], offsets[g + 1]) for g in deals])
        obs = np.unpackbits(obs_bits[rows], axis=1)[:, :OBS_SIZE].astype(np.float32)
        legal = np.unpackbits(legal_bits[rows], axis=1)[:, :NUM_ACTIONS].astype(bool)
        pl = np.asarray(fwd(obs)["policy_logits"])
        act = np.where(legal, pl, -np.inf).argmax(1)
        k = 0
        for g in deals:
            for step in range(offsets[g + 1] - offsets[g]):
                calls, _ = auction_from_obs(obs[k])
                cat = bid_category(_light_frame(calls, BIDS[act[k]], obs[k],
                                                0 if dealer_of is None else int(dealer_of[g])))
                seen[cat] += 1
                if cat not in full:
                    chosen[cat].append((g, step))
                    if len(chosen[cat]) >= args.per_category:
                        full.add(cat)
                k += 1
        if bstart // batch_deals % 10 == 0:
            print(f"[scan] {bstart + len(deals):,} deals, {sum(seen.values()):,} decisions, "
                  f"{sum(len(v) for v in chosen.values()):,} selected, {len(full)}/{len(seen)} situations full "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if len(full) == len(seen) and len(seen) >= 50:
            break

    # ---- pass 2: enrich + teacher explanation for every selected decision
    items = sorted((g, s, cat) for cat, v in chosen.items() for g, s in v)
    split_of = lambda g: ("test" if (g * 2654435761 % 1000) < args.test_frac * 1000  # noqa: E731
                          else "val" if (g * 2654435761 % 1000) < (args.test_frac + args.val_frac) * 1000
                          else "train")
    files = {sp: (open(out / f"{sp}_labels.jsonl", "w"), open(out / f"{sp}_frames.jsonl", "w"))
             for sp in ("train", "val", "test")}
    kept, dropped, per_cat = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    t1 = time.time()
    for i, (g, step, cat) in enumerate(items):
        r = offsets[g] + step
        o = np.unpackbits(obs_bits[r:r + 1], axis=1)[0, :OBS_SIZE].astype(np.float32)
        lg = np.unpackbits(legal_bits[r:r + 1], axis=1)[0, :NUM_ACTIONS].astype(bool)
        o_out = {k2: np.asarray(v[0]) for k2, v in fwd(o[None]).items()}
        calls, _ = auction_from_obs(o)
        dl = 0 if dealer_of is None else int(dealer_of[g])
        vul = "None" if vul_of is None else vul_of[g]
        seat = SEATS[(dl + len(calls)) % 4]
        before = [{"seat": SEATS[(dl + j) % 4], "bid": c} for j, c in enumerate(calls)]
        frame = decision_frame(en, o, lg, o_out, int(g), int(step), seat, SEATS[dl], dl, vul, before,
                               counterfactuals=not args.no_counterfactuals)
        try:
            text = explain(frame)
        except Exception as e:  # dropped, not fatal: the dataset is built unattended
            print(f"[teacher] failed on deal {int(g)} step {int(step)}: {e!r}", flush=True)
            dropped[cat] += 1
            continue
        warnings = check_explanation(frame, text)
        style = check_style(text)
        if warnings or style:
            dropped[cat] += 1
            continue
        from nlg.explanation_inputs import select_reasoned_concepts
        label = {"deal_id": int(g), "step": int(step), "category": cat,
                 "selected_concepts": select_reasoned_concepts(frame), "explanation": text}
        fl, ff = files[split_of(g)]
        fl.write(json.dumps(label, ensure_ascii=False) + "\n")
        ff.write(json.dumps(frame, ensure_ascii=False) + "\n")
        kept[cat] += 1
        if len(per_cat[cat]) < 4:
            per_cat[cat].append((frame, text))
        if i % 2000 == 0:
            print(f"[teacher] {i:,}/{len(items):,} ({time.time() - t1:.0f}s), kept {sum(kept.values()):,}, "
                  f"dropped {sum(dropped.values()):,}", flush=True)
    for fl, ff in files.values():
        fl.close(); ff.close()

    stats = {"model_path": args.model_path, "source": source, "split": args.split, "deals_scanned": min(len(order), bstart + batch_deals),
             "decisions_seen": sum(seen.values()), "per_category_cap": args.per_category,
             "kept": sum(kept.values()), "dropped_by_checks": sum(dropped.values()),
             "situations": {c: {"seen": seen[c], "kept": kept[c], "dropped": dropped[c]} for c in sorted(seen)}}
    _write_reports(out, stats, per_cat, kept, len(seen))
    print(f"[done] kept {stats['kept']:,}, dropped {stats['dropped_by_checks']:,} -> {out}", flush=True)


if __name__ == "__main__":
    main()
