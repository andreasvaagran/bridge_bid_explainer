"""Stopping rule for one PPO run (pipeline/lib.sh starts it next to the training process).

Every 1,000 updates, the checkpoint plays a 10,000-board duplicate match against the checkpoint 2,000 updates earlier
(the SL start at update 2,000) on the held-out deals, with random vulnerability and the same boards each time (seed 0).
A check is flat when the match's 95% CI contains 0 and the KL to the SL policy has levelled off: the mean `kl_to_sl` of
the last 1,000 updates differs from that of the 1,000 before by less than 5%, or by less than 0.001 (the anchored true
bottlenecks sit near 0.001, where 5% is below the sampling noise). After three flat checks in a row the training is
stopped, later checkpoints are moved to rl_params_after/, and converged.json records the update. If the training
reaches its cap first, params-final.pkl is renamed params-<cap>.pkl and converged.json records "cap". Every check is a
line in convergence.jsonl; a restarted watcher skips the updates already checked.

    python pipeline/converge.py --run_dir runs/ppo/joint_a0_lr3e-5 \
        --sl runs/sl/joint/params-best.pkl --pid 1234 --cap 30000
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.duplicate_match import evaluate  # noqa: E402

EVERY, BACK, WINDOW, NEED = 1000, 2000, 1000, 3
KL_REL, KL_ABS = 0.05, 0.001


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def ckpt(rp: Path, k: int) -> Path:
    return rp / f"params-{k:08d}.pkl"


def kl_windows(metrics: Path, k: int):
    rows = [json.loads(l) for l in metrics.open() if '"kl_to_sl"' in l] if metrics.exists() else []
    now = [r["kl_to_sl"] for r in rows if k - WINDOW < r["update"] <= k]
    prev = [r["kl_to_sl"] for r in rows if k - 2 * WINDOW < r["update"] <= k - WINDOW]
    if not now or not prev:
        return None, None
    return sum(now) / len(now), sum(prev) / len(prev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--sl", required=True, help="the SL checkpoint the run started from (the opponent at update 2,000)")
    ap.add_argument("--pid", type=int, required=True, help="the training process")
    ap.add_argument("--cap", type=int, required=True, help="the run's update cap: params-final.pkl is renamed to it")
    ap.add_argument("--num_deals", type=int, default=10_000)
    a = ap.parse_args()
    run = Path(a.run_dir)
    rp, log = run / "ppo" / "rl_params", run / "convergence.jsonl"
    rows = [json.loads(l) for l in log.open()] if log.exists() else []
    checked = {r["update"] for r in rows}
    grace_done = False
    while True:
        training = alive(a.pid)
        if not training and not grace_done:   # let the last checkpoint finish writing
            time.sleep(90)
            grace_done = True
            continue
        if not training and (rp / "params-final.pkl").exists():   # written after the last update: the cap
            (rp / "params-final.pkl").rename(ckpt(rp, a.cap))
        todo = sorted(k for k in (int(p.stem.split("-")[1]) for p in rp.glob("params-[0-9]*.pkl"))
                      if k % EVERY == 0 and k >= BACK and k not in checked)
        # a checkpoint is complete once its writer has moved on (mtime 30 s old) or the training has ended
        todo = [k for k in todo if not training or time.time() - ckpt(rp, k).stat().st_mtime > 30]
        for k in todo:
            opp = ckpt(rp, k - BACK) if k > BACK else Path(a.sl)
            if not opp.exists():
                checked.add(k)
                continue
            r = evaluate(str(ckpt(rp, k)), str(opp), a.num_deals, vulnerability="random", seed=0)
            kl_now, kl_prev = kl_windows(run / "ppo" / "metrics.jsonl", k)
            kl_stable = kl_now is not None and abs(kl_now - kl_prev) < max(KL_REL * kl_now, KL_ABS)
            row = {"update": k, "vs": k - BACK if k > BACK else "SL", "imps": r["imps_per_board"], "ci95": r["ci95"],
                   "ci_includes_0": r["ci95"][0] <= 0 <= r["ci95"][1], "kl_now": kl_now, "kl_prev": kl_prev,
                   "kl_stable": kl_stable, "time": time.strftime("%F %T")}
            rows.append(row)
            checked.add(k)
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            last = sorted(rows, key=lambda x: x["update"])[-NEED:]
            consecutive = len(last) == NEED and all(last[i + 1]["update"] - last[i]["update"] == EVERY
                                                    for i in range(NEED - 1))
            if consecutive and all(x["ci_includes_0"] and x["kl_stable"] for x in last):
                if alive(a.pid):
                    os.kill(a.pid, signal.SIGTERM)
                    for _ in range(120):
                        if not alive(a.pid):
                            break
                        time.sleep(1)
                after = run / "ppo" / "rl_params_after"
                for p in list(rp.glob("params-*.pkl")):
                    n = p.stem.split("-")[1]
                    if n == "final" or int(n) > k:
                        after.mkdir(exist_ok=True)
                        p.rename(after / p.name)
                (run / "converged.json").write_text(json.dumps(
                    {"reason": "converged", "update": k, "checks": last}, indent=2))
                print(f"converged at update {k}", flush=True)
                return 0
        if not training and not todo:
            last_k = max((int(p.stem.split("-")[1]) for p in rp.glob("params-[0-9]*.pkl")), default=0)
            (run / "converged.json").write_text(json.dumps(
                {"reason": "cap", "update": last_k, "checks": sorted(rows, key=lambda x: x["update"])[-NEED:]},
                indent=2))
            print(f"training ended without convergence (last checkpoint {last_k})", flush=True)
            return 0
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
