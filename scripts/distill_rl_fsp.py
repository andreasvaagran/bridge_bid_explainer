#!/usr/bin/env python3
"""Distils brl RL+FSP into concept models. RL+FSP bids its own games and the rule engine labels them (stage data,
into runs/distill/prepared); pipeline/train.sh trains one SL concept model per mode on them (runs/sl/distilled_<mode>);
then agreement, duplicate matches and the explanations of the distilled models' own play. Stages resume.
Outputs go to runs/distill/."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "distill"
PY = _sys.executable
TEACHER = REPO / "bridge_models" / "model-pretrained-rl-with-fsp.pkl"
TRAIN_DDS = REPO / "dds_results" / "dds_results_10M.npy"
CHUNK = 25_000
EVAL_DIR = REPO / "runs" / "evaluation"   # --eval_dir: the thesis evaluation whose settings the reference rows share


def log(msg: str) -> None:
    print(f"[distill {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 1. data
def make_split(name: str, n_deals: int, seed0: int, pool) -> None:
    from evaluation.common import selfplay   # imports JAX: only after the label pool has forked
    from training.ppo_concept_labels import pack_obs
    d = OUT / "prepared" / name
    if (d / "deal_offsets.npy").exists():
        log(f"{name}: exists, skipped")
        return
    parts = {k: [] for k in ("obs_bits", "legal_bits", "target_bits", "action")}
    counts, t0 = [], time.time()
    for k, start in enumerate(range(0, n_deals, CHUNK)):
        n = min(CHUNK, n_deals - start)
        pos = selfplay(TEACHER, n, seed=seed0 + k, random_vulnerability=False, dds_path=TRAIN_DDS)
        order = np.lexsort((pos.step, pos.deal))          # rows of one deal contiguous, in bidding order
        obs, legal, act, deal = pos.obs[order], pos.legal[order], pos.action[order], pos.deal[order]
        auction, hand = pack_obs(obs)
        tgt = pool.label(auction, hand, act)
        parts["obs_bits"].append(np.packbits(obs.astype(np.uint8), axis=1))
        parts["legal_bits"].append(np.packbits(legal.astype(np.uint8), axis=1))
        parts["target_bits"].append(np.packbits(tgt.astype(np.uint8), axis=1))
        parts["action"].append(act.astype(np.uint8))
        counts.append(np.bincount(deal - deal.min(), minlength=n)[np.unique(deal - deal.min())])
        log(f"{name}: {start + n:,}/{n_deals:,} deals ({time.time() - t0:.0f}s)")
    d.mkdir(parents=True, exist_ok=True)
    for key, v in parts.items():
        np.save(d / f"{key}.npy", np.concatenate(v))
    c = np.concatenate(counts)
    np.save(d / "deal_offsets.npy", np.concatenate([[0], np.cumsum(c)]).astype(np.int64))
    log(f"{name}: {len(c):,} deals, {int(c.sum()):,} decisions -> {d}")


def stage_data(args) -> None:
    from training.ppo_concept_labels import HARD_STATE, LabelPool   # numpy + concepts only: fork before JAX
    pool = LabelPool(max(1, (os.cpu_count() or 2) - 1))
    try:
        make_split("train", args.train_deals, 10_000, pool)
        make_split("valid", args.eval_deals, 20_000, pool)
        make_split("test", args.eval_deals, 30_000, pool)
    finally:
        pool.close()
    meta = {"source": f"self-play of {TEACHER.name}, deals from {TRAIN_DDS.name}, nobody vulnerable",
            "split_seed": None, "num_concepts": len(HARD_STATE), "hard_concepts": list(HARD_STATE)}
    (OUT / "prepared" / "meta.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- 2. evaluate
def run(cmd, log_name: str) -> None:
    lp = OUT / "logs" / f"{log_name}.log"
    lp.parent.mkdir(parents=True, exist_ok=True)
    log(f"{log_name}: {' '.join(map(str, cmd))[:160]}")
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_ALLOCATOR": "platform"}
    f = open(lp, "w")
    proc = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=REPO, env=env)
    if proc.wait():
        raise SystemExit(f"{log_name} failed (rc {proc.returncode}) -- see {lp}")


def trained_models(modes) -> dict:
    """The distilled models pipeline/train.sh trained on the data of stage 1."""
    from evaluation.common import DISTILLED
    models = {mode: DISTILLED[f"distilled {mode}"] for mode in modes}
    missing = [str(p) for p in models.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}: run pipeline/train.sh first")
    return models


def stage_eval(models: dict) -> dict:
    """Agreement on RL+FSP's and WBridge5's test games, and duplicate strength. Matches with an RL model are played
    with random vulnerability, as in the thesis (Section 5.1); the supervised ones non-vulnerable, as their data."""
    from evaluation.common import MODELS
    ev = OUT / "eval"
    ev.mkdir(exist_ok=True)
    test, wb5 = OUT / "prepared" / "test", REPO / "data/wbridge5/prepared/test"
    ref = {"SL joint": MODELS["SL joint"], "SL independent": MODELS["SL independent"],
           "SL sequential": MODELS["SL sequential"]}
    jobs = {"agree_fsp_ref_sl_joint": [PY, "-m", "evaluation.split_accuracy", "--model_path", ref["SL joint"],
                                       "--split", test]}
    jobs["fsp_vs_brl_sl_rand"] = [PY, "-m", "evaluation.duplicate_match", "--a", TEACHER, "--b", MODELS["brl SL"],
                                  "--vulnerability", "random"]
    for mode, m in models.items():
        jobs[f"agree_fsp_{mode}"] = [PY, "-m", "evaluation.split_accuracy", "--model_path", m, "--split", test]
        jobs[f"agree_wb5_{mode}"] = [PY, "-m", "evaluation.split_accuracy", "--model_path", m, "--split", wb5]
        for vul in ("none", "random"):
            for k, b in (("brl_sl", MODELS["brl SL"]), ("fsp", TEACHER), ("sl_same", ref[f"SL {mode}"])):
                if vul == "none" and k == "fsp":
                    continue
                jobs[f"{mode}_vs_{k}_{vul}"] = [PY, "-m", "evaluation.duplicate_match", "--a", m, "--b", b,
                                                "--vulnerability", vul]
    res = {}
    for k, cmd in jobs.items():
        out = ev / f"{k}.json"
        if not out.exists():
            run(cmd + ["--out", out], k)
        res[k] = json.loads(out.read_text())
    return res


# ---------------------------------------------------------------- 3. explain
def stage_explain(models: dict) -> dict:
    """What the thesis's explanation pipeline makes of the distilled models, on their own play:
    explanation_faithfulness (teacher correctness, reason share, removal; as Section 5.4 for the thesis models),
    the deviation rate (convention_drift, 3,000 deals, seed 0, as Table drift) and convention cards from play
    (convention_cards_from_play, 50,000 deals, seed 3, as Section 5.5.2; WBridge5's and RL+FSP's cards are copied from
    the thesis evaluation --eval_dir, whose settings they match)."""
    import shutil
    ex = OUT / "explain"
    ex.mkdir(exist_ok=True)
    res = {}
    for mode, m in models.items():
        out = ex / f"faithfulness_{mode}.json"
        if not out.exists():
            run([PY, "-m", "evaluation.explanation_faithfulness", "--model", m, "--source", "selfplay",
                 "--random_vulnerability", "--out_dir", ex / f"faithfulness_{mode}"], f"faithfulness_{mode}")
            shutil.copy(next((ex / f"faithfulness_{mode}").glob("*__selfplay.json")), out)
        res[f"faithfulness_{mode}"] = json.loads(out.read_text())
    out = ex / "drift.json"
    if not out.exists():
        code = ("import json,sys; from evaluation.convention_drift import analyse; from evaluation.common import selfplay;"
                "r={n: analyse(n, selfplay(p, 3000, seed=0)) for n,p in zip(sys.argv[2::2], sys.argv[3::2])};"
                "open(sys.argv[1],'w').write(json.dumps(r, indent=1))")
        run([PY, "-c", code, out] + [x for mode, m in models.items() for x in (f"distilled {mode}", m)], "drift")
    res["drift"] = json.loads(out.read_text())
    thesis_drift = json.loads((EVAL_DIR / "drift/drift.json").read_text())
    res["drift_ref"] = {k: v["deviation_rate_non_pass"]["value"] for k, v in thesis_drift.items()}
    trees = OUT / "convention_trees"
    trees.mkdir(exist_ok=True)
    for f in ("wbridge5.json", "brl_rl_fsp.json"):
        if not (trees / f).exists():
            shutil.copy(EVAL_DIR / "convention_trees" / f, trees / f)
    if not all((trees / f"distilled_{mode}.json").exists() for mode in models):
        run([PY, "-m", "evaluation.convention_cards_from_play", "--deals", "50000", "--wb5_deals", "50000",
             "--agents", "brl RL+FSP", "--checkpoint", *[f"distilled {mode}={m}" for mode, m in models.items()],
             "--out_dir", trees], "trees")
    from evaluation.convention_card_figures import card_departures
    load = lambda n: json.loads((trees / f"{n}.json").read_text())  # noqa: E731
    wb5, fsp = load("wbridge5"), load("brl_rl_fsp")
    for n in ["brl_rl_fsp"] + [f"distilled_{mode}" for mode in models]:
        c = load(n)
        res[f"card_{n}"] = {"departures_vs_wbridge5": sum(bool(d) for d in card_departures(c, wb5)),
                            "departures_vs_rl_fsp": sum(bool(d) for d in card_departures(c, fsp)),
                            "entries": len(c["headline"]), "own_card": c["fidelity"],
                            "wbridge5_card": c["wbridge5_fidelity"], "auctions_differing": c["contexts_differing"]}
    return res


def concept_rates() -> dict:
    """How often each concept is true of the bids in RL+FSP's test games and in WBridge5's: the concepts that would
    claim SAYC's meaning of a bid are false where RL+FSP departs from it, so they fire less often."""
    from training.ppo_concept_labels import HARD_STATE
    names = [getattr(c, "name", c) for c in HARD_STATE]
    def load(d):
        t = np.unpackbits(np.load(d / "target_bits.npy"), axis=1)[:, :len(names)]
        return t.mean(0), t.sum(1)
    f, fn = load(OUT / "prepared" / "test")
    w, wn = load(REPO / "data/wbridge5/prepared/test")
    order = np.argsort(f - w)
    return {"per_decision": {"rl_fsp": float(fn.mean()), "wbridge5": float(wn.mean())},
            "none_true": {"rl_fsp": float((fn == 0).mean()), "wbridge5": float((wn == 0).mean())},
            "largest_drops": [(names[i], float(w[i]), float(f[i])) for i in order[:15]],
            "largest_rises": [(names[i], float(w[i]), float(f[i])) for i in order[::-1][:10]]}


# ---------------------------------------------------------------- 4. report
def stage_report(res: dict, modes) -> None:
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    lines = ["# Distilling brl RL+FSP into concept models", "",
             "Made by `scripts/distill_rl_fsp.py`.", "",
             "## Agreement", "", "| model | test games | agreement | concept F1 (macro / micro) | concept-only acc |",
             "|---|---|---|---|---|"]
    rows = [("SL joint (WBridge5)", "RL+FSP", "agree_fsp_ref_sl_joint")]
    for m in modes:
        rows += [(f"distilled {m}", "RL+FSP", f"agree_fsp_{m}"), (f"distilled {m}", "WBridge5", f"agree_wb5_{m}")]
    for lab, games, k in rows:
        r = res[k]
        lines.append(f"| {lab} | {games} | {r['acc']:.1%} | {r['concept_macro_f1']:.3f} / {r['concept_micro_f1']:.3f} "
                     f"| {r['concept_only_acc']:.1%} |")
    lines += ["", "## Strength (duplicate, 10,000 held-out boards; IMPs/board for the first, 95% CI)", "",
              "| match | nobody vulnerable | random vulnerability |", "|---|---|---|"]
    f = lambda r: f"{r['imps_per_board']:+.3f} ({r['ci95'][0]:+.3f} .. {r['ci95'][1]:+.3f})" if r else "--"
    lines.append(f"| RL+FSP vs brl SL (the gain to keep) | -- | {f(res['fsp_vs_brl_sl_rand'])} |")
    for m in modes:
        for k, lab in (("brl_sl", "brl SL"), ("fsp", "RL+FSP"), ("sl_same", f"SL {m} (WBridge5)")):
            lines.append(f"| distilled {m} vs {lab} | {f(res.get(f'{m}_vs_{k}_none'))} | {f(res[f'{m}_vs_{k}_random'])} |")
    g = res["fsp_vs_brl_sl_rand"]["imps_per_board"]
    lines += ["", "Share of RL+FSP's gain over brl SL kept (random vulnerability): " +
              ", ".join(f"{m} {res[f'{m}_vs_brl_sl_random']['imps_per_board'] / g:.0%}" for m in modes)]
    lines += ["", "## Explanations and convention, on each model's own play", "",
              "| model | explained | explanations pass checks | stated reasons' margin share, non-pass (random) | "
              "removal changes the bid, non-pass (random) | non-pass bids breaking WBridge5's system |", "|---|---|---|---|---|---|"]
    ref = EVAL_DIR / "explanations"
    rows = [(f"SL {m} (WBridge5, thesis)", json.loads((ref / f"SL_{m}__selfplay.json").read_text()),
             res["drift_ref"].get(f"SL {m}")) for m in modes if (ref / f"SL_{m}__selfplay.json").exists()]
    rows += [(f"distilled {m}", res[f"faithfulness_{m}"],
              res["drift"][f"distilled {m}"]["deviation_rate_non_pass"]["value"]) for m in modes]
    for lab, r, d in rows:
        v = lambda k: r[k]["value"]  # noqa: E731
        lines.append(f"| {lab} | {v('teacher_coverage'):.1%} | {v('passes_all_checks'):.2%} | "
                     f"{v('stated_share_nonpass'):.0%} ({v('random_stated_share_nonpass'):.0%}) | "
                     f"{v('stated_flip_nonpass'):.0%} ({v('random_stated_flip_nonpass'):.0%}) | "
                     f"{'--' if d is None else f'{d:.2%}'} |")
    lines += ["", "## Convention cards from play (24 headline entries)", "",
              "| model | entries departing from WBridge5's | from RL+FSP's | own card predicts | WBridge5's card predicts |",
              "|---|---|---|---|---|"]
    for n in ["brl_rl_fsp"] + [f"distilled_{m}" for m in modes]:
        c = res[f"card_{n}"]
        lines.append(f"| {n} | {c['departures_vs_wbridge5']} | {c['departures_vs_rl_fsp']} | {c['own_card']:.1%} | "
                     f"{c['wbridge5_card']:.1%} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    log(f"report -> {OUT / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_deals", type=int, default=1_100_000)
    ap.add_argument("--eval_deals", type=int, default=20_000)
    ap.add_argument("--modes", default="joint,independent,sequential")
    ap.add_argument("--stop_after", choices=("data", "eval"), default=None)
    ap.add_argument("--out", default=None, help="output folder (default runs/distill)")
    ap.add_argument("--eval_dir", default=None, help="the thesis evaluation the reference rows come from "
                    "(default runs/evaluation)")
    args = ap.parse_args()
    args.modes = args.modes.split(",")
    global OUT, EVAL_DIR
    if args.eval_dir:
        EVAL_DIR = (REPO / args.eval_dir).resolve()
    if args.out:
        OUT = (REPO / args.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_data(args)
    if args.stop_after == "data":
        return
    models = trained_models(args.modes)
    res = stage_eval(models)
    if args.stop_after == "eval":
        return
    res.update(stage_explain(models))
    res["concept_rates"] = concept_rates()
    stage_report(res, args.modes)


if __name__ == "__main__":
    main()
