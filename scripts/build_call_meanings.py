#!/usr/bin/env python3
"""What every call means in WBridge5's system, auction by auction, from the training split: per auction
and call the count, HCP and suit-length percentiles, and how often partner passed it. Writes
nlg/call_meanings.json.gz for nlg/call_meaning.py."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SPLIT = REPO / "data" / "wbridge5" / "prepared" / "train"
BIDS = ["P", "X", "XX"] + [f"{lvl}{s}" for lvl in range(1, 8) for s in ("C", "D", "H", "S", "NT")]
PASS = 0

# observation layout (concepts.hand): the last 52 of the 480 bits are the hand, bit i =
# rank i // 4 ("23456789TJQKA"), suit "SHDC"[3 - i % 4]
_RANK = np.arange(52) // 4
_SUIT = 3 - np.arange(52) % 4          # index into "SHDC"
_HCP = np.where(_RANK >= 9, _RANK - 8, 0)  # J=1 Q=2 K=3 A=4


def _hands(obs_rows: np.ndarray):
    """(n, 60) packed observation rows -> (hcp (n,), lengths (n, 4) in S H D C order)."""
    bits = np.unpackbits(obs_rows[:, 53:60], axis=1)[:, 4:56].astype(np.int16)  # bits 424..479 -> 428..479
    hcp = bits @ _HCP
    lengths = np.stack([bits[:, _SUIT == s].sum(1) for s in range(4)], axis=1)
    return hcp.astype(np.int16), lengths.astype(np.int16)


def _quantiles(hist: np.ndarray, qs=(0.05, 0.5, 0.95)):
    """Quantiles of an integer histogram (rows = keys)."""
    cdf = np.cumsum(hist, axis=1) / np.maximum(hist.sum(1, keepdims=True), 1)
    return [np.argmax(cdf >= q, axis=1) for q in qs]


def build(max_deals=None, min_n=20, log=print):
    t0 = time.time()
    offsets = np.load(SPLIT / "deal_offsets.npy")
    action = np.load(SPLIT / "action.npy")
    obs_bits = np.load(SPLIT / "obs_bits.npy", mmap_mode="r")
    n_deals = len(offsets) - 1 if max_deals is None else min(max_deals, len(offsets) - 1)
    n_dec = int(offsets[n_deals])
    action = action[:n_dec].astype(np.int64)
    starts = offsets[:n_deals]
    lens = np.diff(offsets[: n_deals + 1])
    # every seat's hand from its first decision (rows start+seat); a deal has >= 4 calls
    ok = lens >= 4
    hcp_seat = np.zeros((n_deals, 4), np.int16)
    len_seat = np.zeros((n_deals, 4, 4), np.int16)
    for seat in range(4):
        rows = np.asarray(obs_bits[(starts + seat)[ok]])
        h, L = _hands(rows)
        hcp_seat[ok, seat] = h
        len_seat[ok, seat] = L
    log(f"[meanings] hands decoded for {n_deals:,} deals ({time.time() - t0:.0f}s)")

    deal_of = np.repeat(np.arange(n_deals), lens)
    k_of = np.arange(n_dec) - np.repeat(starts, lens)
    seat_of = (k_of % 4).astype(np.int64)
    keep = ok[deal_of]
    dec_hcp = hcp_seat[deal_of, seat_of]
    dec_len = len_seat[deal_of, seat_of]

    # prefix ids: a trie over the call sequences (node -> parent, last call), exact: no hashing
    def make_trie():
        return {}, [-1], [-1]  # child map ((node * 38 + call) -> node), parent[], call[]

    def walk(parent, call_of, node):
        out = []
        while node > 0:
            out.append(call_of[node])
            node = parent[node]
        return tuple(reversed(out))

    kids, parent, call_of = make_trie()
    kids_np, parent_np, call_np = make_trie()
    prefix_id = np.zeros(n_dec, np.int64)
    prefix_np = np.zeros(n_dec, np.int64)
    act_list = action.tolist()
    off = offsets.tolist()
    for d in range(n_deals):
        a, b = off[d], off[d + 1]
        node, node_np, opened = 0, 0, False
        for i in range(a, b):
            prefix_id[i] = node
            prefix_np[i] = node_np
            c = act_list[i]
            k = node * 38 + c
            nxt_node = kids.get(k)
            if nxt_node is None:
                nxt_node = kids[k] = len(parent)
                parent.append(node)
                call_of.append(c)
            node = nxt_node
            if c != PASS or opened:
                opened = True
                k = node_np * 38 + c
                nxt_node = kids_np.get(k)
                if nxt_node is None:
                    nxt_node = kids_np[k] = len(parent_np)
                    parent_np.append(node_np)
                    call_np.append(c)
                node_np = nxt_node
        if d % 200_000 == 0:
            log(f"[meanings] {d:,} deals, {len(parent):,} prefixes ({time.time() - t0:.0f}s)")
    inv = lambda node: walk(parent, call_of, node)  # noqa: E731
    inv_np = lambda node: walk(parent_np, call_np, node)  # noqa: E731
    log(f"[meanings] {len(parent):,} exact prefixes, {len(parent_np):,} after leading passes ({time.time() - t0:.0f}s)")
    del kids, kids_np

    # "forcing": after the call, the next opponent passed -- did partner pass?
    nxt = np.arange(n_dec) + 1
    same1 = np.zeros(n_dec, bool)
    same2 = np.zeros(n_dec, bool)
    same1[:-1] = deal_of[1:] == deal_of[:-1]
    same2[:-2] = deal_of[2:] == deal_of[:-2]
    lho_pass = same1 & (action[np.minimum(nxt, n_dec - 1)] == PASS)
    pard_pass = same2 & (action[np.minimum(nxt + 1, n_dec - 1)] == PASS)
    measurable = lho_pass & same2 & (action != PASS)

    tables, totals = {}, {}
    for name, pid, inv_map in (("exact", prefix_id, inv), ("no_leading_passes", prefix_np, inv_np)):
        key = pid * 38 + action
        sel = np.flatnonzero(keep)
        uniq, first, ginv, counts = np.unique(key[sel], return_index=True, return_inverse=True, return_counts=True)
        big = counts >= min_n
        gmap = np.full(len(uniq), -1, np.int64)
        gmap[big] = np.arange(big.sum())
        g = gmap[ginv]
        m = g >= 0
        G = int(big.sum())
        gi = g[m]
        sel_m = sel[m]
        hcp_hist = np.bincount(gi * 41 + dec_hcp[sel_m], minlength=G * 41).reshape(G, 41)
        qs_h = _quantiles(hcp_hist)
        q_len = []
        for s in range(4):
            hist = np.bincount(gi * 14 + dec_len[sel_m, s], minlength=G * 14).reshape(G, 14)
            q_len.append(_quantiles(hist))
        meas = measurable[sel_m]
        n_meas = np.bincount(gi[meas], minlength=G)
        n_pp = np.bincount(gi[meas & pard_pass[sel_m]], minlength=G)
        per_prefix = np.bincount(pid[sel])  # every call made after the prefix, rare ones too
        out, tot = {}, {}
        uk = uniq[big]
        for j in range(G):
            p_id, call = divmod(int(uk[j]), 38)
            prefix = " ".join(BIDS[c] for c in inv_map(p_id))
            e = {"n": int(counts[big][j]),
                 "hcp": [int(qs_h[0][j]), int(qs_h[1][j]), int(qs_h[2][j])],
                 "len": [[int(q_len[s][0][j]), int(q_len[s][1][j]), int(q_len[s][2][j])] for s in range(4)]}
            if n_meas[j] >= 10:
                e["partner_passed"] = [int(n_pp[j]), int(n_meas[j])]
            out.setdefault(prefix, {})[BIDS[call]] = e
            tot[prefix] = int(per_prefix[p_id])
        tables[name] = out
        totals[name] = tot
        log(f"[meanings] {name}: {G:,} (auction, call) keys with n >= {min_n} over {len(out):,} auctions "
            f"({time.time() - t0:.0f}s)")
    return {"source": f"WBridge5 training split, {n_deals:,} deals, {n_dec:,} calls (non-vulnerable; "
                      f"auctions written from the dealer)",
            "built_by": "python scripts/build_call_meanings.py",
            "percentiles": [5, 50, 95], "suits": "SHDC", "min_n": min_n,
            "exact": tables["exact"], "no_leading_passes": tables["no_leading_passes"],
            "totals": totals}  # hands that reached each auction, so a call's n has a scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max_deals", type=int, default=None)
    ap.add_argument("--min_n", type=int, default=20)
    ap.add_argument("--out", default=str(REPO / "nlg" / "call_meanings.json.gz"))
    args = ap.parse_args()
    T = build(args.max_deals, args.min_n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump(T, f, separators=(",", ":"))
    print(f"[meanings] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
