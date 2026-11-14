"""What the concepts cost in compute, one part per process (--part): static (parameters, multiply-adds),
infer, labels (the rule engine labelling the supervised data, once before training), sl (one step per mode), ppo
(roll-out, GAE, update with and without the anchor, the concept step), explain (the parts of one explained call),
and logs (wall-clock of the real GPU runs)."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PARTS = ("static", "infer", "labels", "sl", "ppo", "explain", "logs")
from evaluation.common import RUNS  # noqa: E402  the thesis's SL models
MODELS = {
    **{f"{m} SL": ROOT / RUNS["sl"].format(mode=m) / "params-best.pkl" for m in ("joint", "independent", "sequential")},
    "SL plain": ROOT / "runs/sl/plain/params-best.pkl",
    "brl SL": ROOT / "bridge_models/model-sl.pkl",
}
NO_CONCEPTS = "no concept head (brl architecture)"
SL_CONFIGS = {  # tab:sl_hparams; "plain" = the joint recipe with every concept loss off
    "plain": dict(bottleneck_mode="joint", hard_coef=0.0, concept_only_coef=0.0),
    "joint": dict(bottleneck_mode="joint", hard_coef=10.0, reweight_concepts=True),
    "independent": dict(bottleneck_mode="independent", hard_coef=4.2, reweight_concepts=False),
    "sequential A": dict(bottleneck_mode="sequential", hard_coef=4.2, reweight_concepts=True),
    "sequential B": dict(bottleneck_mode="sequential", hard_coef=4.2, reweight_concepts=True),
}


def machine() -> dict:
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")

    def read(p):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return None
    import jax
    return {"cpu": cpu, "cores": os.cpu_count(), "platform": jax.default_backend(), "jax": jax.__version__,
            "python": platform.python_version(), "host": platform.node(),
            "power_profile": read("/sys/firmware/acpi/platform_profile"),
            "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")}


def interleaved(fns: dict, rounds: int, reps: int, warmup: int = 2) -> dict:
    """Seconds per call of each fn in `fns`, measured in `rounds` rounds that visit every fn in turn (`reps` calls
    each), so that slow drifts of the machine (clock, temperature, other load) fall on all of them alike."""
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    ts = {k: [] for k in fns}
    for _ in range(rounds):
        for k, fn in fns.items():
            for _ in range(reps):
                t = time.perf_counter()
                fn()
                ts[k].append(time.perf_counter() - t)
    return {k: stats(v) for k, v in ts.items()}


def stats(xs) -> dict:
    import numpy as np
    xs = np.asarray(xs, float)
    return {"median": float(np.median(xs)), "mean": float(xs.mean()), "p10": float(np.percentile(xs, 10)),
            "p90": float(np.percentile(xs, 90)), "n": int(len(xs))}


def bench(fn, reps: int, warmup: int = 2) -> dict:
    """Seconds per call of fn() (which must block until its result is ready)."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return stats(ts)


def load_params(path: Path):
    import pickle
    obj = pickle.load(open(path, "rb"))
    return obj["params"] if isinstance(obj, dict) and "params" in obj else obj


# ----------------------------------------------------------------------------------------------- static
def part_static() -> dict:
    out = {}
    for name, path in MODELS.items():
        params = load_params(path)
        mods = {}
        for mod, leaves in params.items():
            n = int(sum(v.size for v in leaves.values()))
            key = mod.split("/")[-1]
            if key in ("linear", "linear_1", "linear_2", "linear_3"):
                group = "trunk"
            elif key == "linear_4":
                group = "policy_head"
            elif key == "linear_5":
                group = "value_head"
            else:
                group = key
            mods[group] = mods.get(group, 0) + n
        # multiply-adds of one decision: every weight on the path to the call, used once
        w = {mod.split("/")[-1]: leaves["w"].shape for mod, leaves in params.items()}
        path_mods = [m for m in w if m.startswith("linear") and m not in ("linear_4", "linear_5")]
        path_mods += ["linear_4"] if "linear_4" in w else ["hard_state_head", "policy_head"]
        macs = int(sum(w[m][0] * w[m][1] for m in path_mods))
        out[name] = {"parameters": mods, "total": int(sum(mods.values())), "macs_per_decision": macs}
    return out


# ------------------------------------------------------------------------------------------------ infer
def part_infer() -> dict:
    import jax
    import numpy as np
    from evaluation.duplicate_match import load_policy
    from evaluation.common import load_test_split
    test = load_test_split()
    obs = test.obs_f32(np.arange(4096))
    one = obs[:1]
    fns, kinds = {}, {}
    for name, path in MODELS.items():
        params, logits_fn, kinds[name] = load_policy(str(path))
        f = jax.jit(lambda o, params=params, logits_fn=logits_fn: logits_fn(params, o))
        fns[(name, 1)] = lambda f=f: f(one).block_until_ready()
        fns[(name, 4096)] = lambda f=f: f(obs).block_until_ready()
    b1 = interleaved({k: v for k, v in fns.items() if k[1] == 1}, rounds=10, reps=200, warmup=20)
    bb = interleaved({k: v for k, v in fns.items() if k[1] == 4096}, rounds=10, reps=5, warmup=3)
    out = {}
    for name in MODELS:
        s1, sb = b1[(name, 1)], bb[(name, 4096)]
        out[name] = {"kind": kinds[name], "batch1_ms": {k: (v * 1e3 if k != "n" else v) for k, v in s1.items()},
                     "batch4096_ms": sb["median"] * 1e3, "decisions_per_s": 4096 / sb["median"]}
        print(f"[infer] {name}: batch 1 {s1['median'] * 1e3:.3f} ms, 4096 in {sb['median'] * 1e3:.1f} ms",
              flush=True)
    return out


# ----------------------------------------------------------------------------------------------- labels
def part_labels(serial_deals: int = 2000) -> dict:
    """The one-off cost of the supervised concept labels: training/prepare_sl_data.py replays every deal and the rule
    engine labels each decision with the 146 concepts. Timed on the test split's deals: one core with the rule engine
    timed apart from the replay, then all cores as prepare_sl_data runs it, scaled to the decisions of all splits."""
    import multiprocessing as mp
    import numpy as np
    import training.prepare_sl_data as P
    prepared = ROOT / "data/wbridge5/prepared"
    meta = json.loads((prepared / "meta.json").read_text())
    lines = [l for l in open(ROOT / "data/wbridge5/official_test.txt") if l.strip()]
    P._init_worker()
    spent = {"rules": 0.0}
    classify = P._AUTH.classify

    def timed(*a, **k):
        t = time.perf_counter()
        r = classify(*a, **k)
        spent["rules"] += time.perf_counter() - t
        return r
    P._AUTH.classify = timed
    for line in lines[:50]:
        P.deal_decisions(P.no_play_traj(line))
    spent["rules"] = 0.0
    t = time.perf_counter()
    n = sum(len(P.deal_decisions(P.no_play_traj(line))) for line in lines[:serial_deals])
    total = time.perf_counter() - t
    P._AUTH.classify = classify
    workers = os.cpu_count() or 1
    chunks = [lines[i:i + 500] for i in range(0, len(lines), 500)]
    with mp.get_context("fork").Pool(workers, initializer=P._init_worker) as pool:
        pool.map(P._process_chunk, chunks[:workers])  # the workers' own start-up
        t = time.perf_counter()
        parts = pool.map(P._process_chunk, chunks)
        wall = time.perf_counter() - t
    done = int(sum(int(p[4].sum()) for p in parts))
    decisions = {k: v["decisions"] for k, v in meta["splits"].items()}
    all_dec = sum(decisions.values())
    size = {f: (prepared / "train" / f"{f}.npy").stat().st_size for f in ("target_bits", "obs_bits")}
    out = {"serial": {"deals": serial_deals, "decisions": n, "ms_per_decision": total / n * 1e3,
                      "rule_engine_ms_per_decision": spent["rules"] / n * 1e3,
                      "replay_ms_per_decision": (total - spent["rules"]) / n * 1e3},
           "pool": {"workers": workers, "deals": len(lines), "decisions": done, "seconds": wall,
                    "decisions_per_s": done / wall},
           "decisions": decisions, "all_decisions": all_dec,
           "all_minutes": all_dec / (done / wall) / 60,
           "all_minutes_rule_engine": all_dec / (done / wall) / 60 * spent["rules"] / total,
           "core_minutes_rule_engine": all_dec * spent["rules"] / n / 60,
           "train_label_bytes": size["target_bits"], "train_observation_bytes": size["obs_bits"],
           "concepts": meta["num_concepts"]}
    print(f"[labels] {out['serial']['rule_engine_ms_per_decision']:.3f} ms/decision in the rule engine, "
          f"{out['serial']['replay_ms_per_decision']:.3f} in the replay; {done / wall:,.0f} decisions/s on {workers} "
          f"workers; all {all_dec:,} decisions: {out['all_minutes']:.1f} min", flush=True)
    return out


# --------------------------------------------------------------------------------------------------- sl
def sl_times(config: str, reps: int) -> list:
    """Seconds of `reps` supervised steps (batch 1,024: forward, backward, AdamW) of one configuration, after
    compilation. Run in its own process: training/sl.py reads its settings from argv when imported."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from concepts import HARD_STATE_CONCEPTS
    from evaluation.common import load_test_split
    test = load_test_split()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test), 1024, replace=False)
    batch = (jnp.asarray(test.obs_f32(idx)), jnp.asarray(test.action[idx]), jnp.asarray(test.legal[idx]),
             jnp.asarray(rng.random((1024, len(HARD_STATE_CONCEPTS))) < 0.05, jnp.float32))
    if config == NO_CONCEPTS:
        import haiku as hk
        import optax

        def net_fn(obs):  # brl's architecture: the trunk and a policy head, no concept head
            x = obs
            for _ in range(4):
                x = jax.nn.relu(hk.Linear(1024)(x))
            return hk.Linear(38)(x)
        net = hk.without_apply_rng(hk.transform(net_fn))
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3, weight_decay=0.05))

        @jax.jit
        def step(p, o, obs, act):
            def loss(p):
                logp = jax.nn.log_softmax(net.apply(p, obs))
                return -jnp.mean(jnp.take_along_axis(logp, act[:, None], axis=1))
            g = jax.grad(loss)(p)
            u, o = opt.update(g, o, p)
            return optax.apply_updates(p, u), o
        params = net.init(jax.random.PRNGKey(0), batch[0])
        st = {"p": params, "o": opt.init(params)}

        def run():
            p, o = step(st["p"], st["o"], batch[0], batch[1])
            jax.block_until_ready(p)
            st.update(p=p, o=o)
    else:
        sys.argv = ([sys.argv[0]] + [f"{k}={v}" for k, v in SL_CONFIGS[config].items()]
                    + ["train_batch=1024", "prepared_path=data/wbridge5/prepared", "weight_decay=0.05"])
        import training.sl as sl
        params = sl.net.init(jax.random.PRNGKey(0), batch[0])
        cw, pw, freeze = {"sequential A": (1.0, 0.0, False),
                          "sequential B": (0.0, 1.0, True)}.get(config, (1.0, 1.0, False))
        step = sl.make_step_fn(cw, pw, freeze)
        st = {"p": params, "o": sl.opt.init(params), "e": jnp.full((sl.NUM_HARD,), 0.05)}

        def run():
            p, o, e, _ = step(st["p"], st["o"], st["e"], batch)
            jax.block_until_ready(p)
            st.update(p=p, o=o, e=e)
    for _ in range(3):
        run()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        run()
        ts.append(time.perf_counter() - t)
    return ts


def part_sl(rounds: int = 5, reps: int = 10) -> dict:
    """Every configuration once per round, each in a fresh process, so that drifts of the machine hit all alike."""
    ts = {c: [] for c in list(SL_CONFIGS) + [NO_CONCEPTS]}
    for r in range(rounds):
        for config in ts:
            out = subprocess.run([sys.executable, "-m", "evaluation.runtime_cost", "--sl_one", config,
                                  "--reps", str(reps)], cwd=ROOT, capture_output=True, text=True)
            line = [l for l in out.stdout.splitlines() if l.startswith("[")]
            if out.returncode or not line:
                raise RuntimeError(f"sl {config} failed:\n{out.stderr[-2000:]}")
            ts[config] += json.loads(line[-1])
        print(f"[sl] round {r + 1}/{rounds} done", flush=True)
    res = {k: {s: (v * 1e3 if s != "n" else v) for s, v in stats(t).items()} for k, t in ts.items()}
    for k, v in res.items():
        print(f"[sl] {k}: {v['median']:.1f} ms/step", flush=True)
    return res


# -------------------------------------------------------------------------------------------------- ppo
def part_ppo(pool, fast_pool=None) -> dict:
    """The parts of one PPO update on this CPU (the GPU runs' totals per update are in --part logs)."""
    import haiku as hk
    import jax
    import numpy as np
    import optax
    from training import ppo as P
    from training.ppo_concept_labels import build_concept_targets_chunk
    from training.ppo_anchor import make_anchored_update_step

    out = {"pool_workers": pool._processes}
    env = P.BridgeBidding()
    for mode in ("joint", "independent"):
        sl = str(ROOT / RUNS["sl"].format(mode=mode) / "params-best.pkl")
        cfg = P.PPOConfig(game_mode="competitive", bottleneck_mode=mode, load_initial_model=True,
                          initial_model_path=sl, lr=3e-5, num_envs=1024, num_steps=32, minibatch_size=32768,
                          reward_scale=7600, illegal_action_l2norm_coef=0)
        config = cfg.model_dump()
        config["num_updates"] = 1
        config["num_minibatches"] = config["num_envs"] * config["num_steps"] // config["minibatch_size"]
        net = P.build_net(mode, cfg.actor_model_type, cfg.actor_activation, P.NUM_HARD, has_value=True)
        fwd = P.PPOForward(net)
        rng = jax.random.PRNGKey(0)
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        params = hk.data_structures.merge(fwd.init(init_rng, jax.numpy.zeros((1, P.OBS_SIZE))), load_params(Path(sl)))
        optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5))
        opt_state = optimizer.init(params)
        from training.checkpoint import load_checkpoint
        ref = load_checkpoint(sl)
        ref_logits_fn = jax.jit(lambda o: ref.net.apply(ref.params, o.astype(jax.numpy.float32))["policy_logits"])
        roll_out = jax.jit(P.make_roll_out(config, env, fwd, fwd))
        calc_gae = jax.jit(P.make_calc_gae(config, fwd))
        update = jax.jit(P.make_update_step(config, fwd, optimizer))
        anchored = jax.jit(make_anchored_update_step(config, fwd, optimizer, float(RUNS["eta"])))
        concept_step = P.make_concept_loss_fn(net, 0.1 if mode == "joint" else 0.0)
        concept_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.lr, eps=1e-5))
        c_opt_state = concept_opt.init(params)
        env_state = jax.vmap(env.init)(jax.random.split(env_rng, cfg.num_envs))
        st = {"runner": (params, opt_state, env_state, env_state.observation, 0, rng)}

        def t_roll():
            r, traj = roll_out(st["runner"], st["runner"][0])
            jax.block_until_ready(traj)
            st.update(runner=r, traj=traj)
        res = {"rollout_s": bench(t_roll, reps=3, warmup=1)}

        def t_gae():
            adv, tgt = calc_gae(st["runner"], st["traj"])
            jax.block_until_ready(tgt)
            st.update(adv=adv, tgt=tgt)
        res["gae_s"] = bench(t_gae, reps=3, warmup=1)
        flat = st["traj"].obs.reshape(-1, st["traj"].obs.shape[-1])

        def t_ref():
            st["ref"] = ref_logits_fn(flat).reshape(st["traj"].obs.shape[:2] + (P.NUM_ACTIONS,))
            jax.block_until_ready(st["ref"])
        res["anchor_reference_logits_s"] = bench(t_ref, reps=3, warmup=1)

        def t_update():
            r, _ = update(st["runner"], st["traj"], st["adv"], st["tgt"])
            jax.block_until_ready(r[0])
        res["ppo_update_s"] = bench(t_update, reps=3, warmup=1)

        def t_anchored():
            r, _ = anchored(st["runner"], st["traj"], st["adv"], st["tgt"], st["ref"])
            jax.block_until_ready(r[0])
        res["ppo_update_anchored_s"] = bench(t_anchored, reps=3, warmup=1)

        obs_np = np.asarray(st["traj"].obs)
        act_np = np.asarray(st["traj"].action)
        legal_np = np.asarray(st["traj"].legal_action_mask)
        n_serial = 64  # environments labelled serially (32 calls each), to estimate the one-core rate

        def t_label_serial():
            build_concept_targets_chunk(obs_np[:, :n_serial], act_np[:, :n_serial])
        s = bench(t_label_serial, reps=3, warmup=1)
        res["label_serial_per_call_ms"] = s["median"] / (32 * n_serial) * 1e3

        def t_label_pool():
            chunks = np.array_split(np.arange(obs_np.shape[1]), pool._processes)
            st["targets"] = np.concatenate(pool.starmap(
                build_concept_targets_chunk, [(obs_np[:, i], act_np[:, i]) for i in chunks]), axis=1)
        res["label_pool_s"] = bench(t_label_pool, reps=3, warmup=1)
        if fast_pool is not None:  # training/ppo.py's labeller: packed bits, cached per-auction plans
            from training.ppo_concept_labels import pack_obs
            auc, hand = pack_obs(obs_np)
            t = time.perf_counter()
            fast = fast_pool.label(auc, hand, act_np.reshape(-1))
            res["label_fast_pool_cold_s"] = time.perf_counter() - t  # first sight of this rollout: no plans yet
            res["label_fast_cold_hit_rate"] = fast_pool.stats["plan_hits"] / len(fast)
            assert np.array_equal(fast.reshape(st["targets"].shape), st["targets"])
            fast_pool.close()
            fast_pool = None  # the second mode would hit the first mode's plans
        res["calls_per_update"] = int(obs_np.shape[0] * obs_np.shape[1])

        fo, fa = obs_np.reshape(-1, P.OBS_SIZE), act_np.reshape(-1)
        fl, ft = legal_np.reshape(-1, P.NUM_ACTIONS), st["targets"].reshape(-1, P.NUM_HARD)

        def t_cgrad():
            _, g, aux = concept_step(st["runner"][0], fo, ft, fa, fl, 0.5)
            u, o2 = concept_opt.update(g, c_opt_state)
            p2 = optax.apply_updates(st["runner"][0], u)
            jax.block_until_ready(p2)
        res["concept_gradient_step_s"] = bench(t_cgrad, reps=3, warmup=1)
        out[mode] = res
        print(f"[ppo] {mode}: " + ", ".join(f"{k} {v['median']:.2f}" if isinstance(v, dict) else f"{k} {v:.3f}"
                                           for k, v in res.items()), flush=True)
    return out


# ---------------------------------------------------------------------------------------------- explain
def part_explain(deals: int = 100, warmup_deals: int = 5) -> dict:
    """Times the parts of nlg.decision_record.analyze over the calls of `deals` deals each model bids at all four
    seats (after `warmup_deals` deals, so that compilation is excluded)."""
    import gui.game as G
    import nlg.explanation_frames as F
    import nlg.explanation_inputs as R
    import nlg.decision_record as X

    cur: dict = {}

    def wrap(mod, name, key):
        fn = getattr(mod, name)

        def w(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            cur[key] = cur.get(key, 0.0) + time.perf_counter() - t
            return r
        setattr(mod, name, w)
        return fn

    wrap(X.Bidder, "outputs", "forward pass")
    wrap(X.Bidder, "card_attribution", "integrated gradients")
    wrap(X, "decision_frame", "frame")
    wrap(X, "_upstream_frame", "frame")
    wrap(X, "teacher_explain", "teacher text")
    wrap(X, "check_explanation", "checker")
    wrap(X, "own_convention", "convention card lookup")
    wrap(F, "card_counterfactuals", "card counterfactuals")
    wrap(X, "card_counterfactuals", "card counterfactuals")
    wrap(F, "search_minimal_flip", "concept flips")
    wrap(F, "render_concept_justification", "rule justification")
    wrap(X, "render_concept_justification", "rule justification")
    wrap(F, "compute_trust_flag", "trust flag")
    analyze = G.analyze
    records = []

    def timed_analyze(bidder, *a, **k):
        cur.clear()
        if bidder.enricher is not None and not getattr(bidder.enricher, "_timed", False):
            sa = bidder.enricher.state_auth
            classify = sa.classify

            def c(*aa, **kk):
                t = time.perf_counter()
                r = classify(*aa, **kk)
                cur["rule engine (concept truth)"] = cur.get("rule engine (concept truth)", 0.0) + time.perf_counter() - t
                return r
            sa.classify = c
            bidder.enricher._timed = True
        t = time.perf_counter()
        r = analyze(bidder, *a, **k)
        rec = dict(cur, total=time.perf_counter() - t)
        records.append(rec)
        return r
    G.analyze = timed_analyze

    names = ("independent SL", "joint SL", "brl SL")
    regs = {n: G.Registry.of({"a": str(MODELS[n])}) for n in names}
    per = {n: [] for n in names}
    for i in range(deals + warmup_deals):  # the models take turns deal by deal, so that drifts hit all alike
        for n in names:
            records.clear()
            # the per-auction caches of the rule-engine look-ups start empty for every model and deal, so that
            # no model profits from the auctions another model has just bid
            R._probe_values.cache_clear()
            R._gt_cached.cache_clear()
            g = G.Game(regs[n], {s: "a" for s in "NESW"}, seed=1000 + i, vulnerability="None")
            g.auto()
            if i >= warmup_deals:
                per[n] += records
    out = {}
    for name in names:
        recs = per[name]
        keys = sorted({k for r in recs for k in r})
        out[name] = {"calls": len(recs), "deals": deals,
                     "ms": {k: {s: v * 1e3 for s, v in stats([r.get(k, 0.0) for r in recs]).items() if s != "n"}
                            for k in keys}}
        print(f"[explain] {name}: {len(recs)} calls, total median {out[name]['ms']['total']['median']:.1f} ms; "
              + ", ".join(f"{k} {v['mean']:.1f}" for k, v in out[name]["ms"].items() if k != "total"), flush=True)
    return out


# ------------------------------------------------------------------------------------------------- logs
def part_logs() -> dict:
    """Wall-clock costs from the training logs of the thesis's models: the PPO runs (runs/ppo, run side by side, so "concurrent"), the SL runs (the log of pipeline/train.sh, runs/train.log) and the Qwen evaluation (runs/evaluation/logs)."""
    import glob
    import numpy as np
    ppo = {}
    found = sorted(glob.glob(str(ROOT / RUNS["ppo_root"] / "*/ppo/metrics.jsonl")))
    for m in found:
        rows = [json.loads(l) for l in open(m)]
        rows = [r for r in rows if "time" in r]
        lp = Path(m).parent / "launches.jsonl"
        if len(rows) < 30 or not lp.exists():
            continue
        c = [json.loads(l) for l in open(lp)][-1]
        u = np.array([r["update"] for r in rows])
        t = np.array([r["time"] for r in rows])
        ok = np.diff(u) > 0
        d = (np.diff(t) / np.maximum(np.diff(u), 1))[ok][5:]
        run = str(Path(m).parent.parent.relative_to(ROOT / "runs"))
        ppo[run] = {"mode": c["bottleneck_mode"], "concept_coef": c["concept_coef"], "anchor": c["sl_anchor_coef"],
                    "started": c.get("started"), "updates": int(u[-1]), "s_per_update": stats(d), "concurrent": True}
    sl = {}
    # log lines of the form "end SL runs/sl/plain rc=0 after N min"
    names = {"sl/plain": "plain seed 44", "sl/plain_s1": "plain seed 1", "sl/plain_s2": "plain seed 2",
             **{RUNS["sl"].removeprefix("runs/").format(mode=m): m for m in ("joint", "independent", "sequential")}}
    for lf in (ROOT / "runs/train.log",):
        for line in (lf.read_text().splitlines() if lf.exists() else []):
            m = re.search(r"end (?:SL )?(?:runs/)?(\S+) rc=0 after (\d+) min", line)
            if m and m.group(1) in names:
                sl[names[m.group(1)]] = {"minutes": float(m.group(2)), "concurrent": True}
    qwen = {}
    for lp in sorted(glob.glob(str(ROOT / "runs/evaluation/logs/qwen_*_testsplit.log"))):
        last = None
        for line in open(lp, errors="replace"):
            m = re.match(r"\[eval_qwen\] (\d+)/(\d+) .*\((\d+)s\)", line)
            if m:
                last = (int(m.group(1)), int(m.group(3)))
        if last:
            qwen[Path(lp).stem.replace("qwen_", "eval_", 1)] = {"explanations": last[0], "seconds": last[1], "s_per_explanation": last[1] / last[0]}
    return {"ppo_gpu": ppo, "sl_gpu_minutes": sl, "qwen_gpu": qwen,
            "desktop": "Intel Core i5-12400F (12 threads), NVIDIA RTX 3060 Ti 8 GB"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all", choices=PARTS + ("all",))
    ap.add_argument("--out", default=str(ROOT / "runs/evaluation/runtime"))
    ap.add_argument("--ppo_inner", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--sl_one", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--reps", type=int, default=10, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.sl_one:
        print(json.dumps(sl_times(args.sl_one, args.reps)))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = ("ppo",) if args.ppo_inner else PARTS if args.part == "all" else (args.part,)
    for part in parts:
        if part == "ppo" and not args.ppo_inner:  # its own process: the labelling pool forks before JAX starts
            r = subprocess.run([sys.executable, "-m", "evaluation.runtime_cost", "--ppo_inner", "--out", str(out)],
                               cwd=ROOT)
            if r.returncode:
                raise SystemExit(f"ppo part failed ({r.returncode})")
            continue
        t = time.time()
        if part == "ppo":
            import multiprocessing as mp
            from training.ppo_concept_labels import LabelPool
            pool = mp.get_context("fork").Pool(os.cpu_count() or 1)  # a plain pool over build_concept_targets_chunk
            fast_pool = LabelPool(os.cpu_count() or 1)  # training/ppo.py's labelling pool
            try:
                res = part_ppo(pool, fast_pool)
            finally:
                pool.terminate()
                fast_pool.close()
        else:
            res = {"part_static": part_static, "part_infer": part_infer, "part_labels": part_labels, "part_sl": part_sl,
                   "part_explain": part_explain, "part_logs": part_logs}[f"part_{part}"]()
        doc = {"machine": machine(), "measured": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "seconds": round(time.time() - t, 1), "results": res}
        (out / f"{part}.json").write_text(json.dumps(doc, indent=1))
        print(f"[runtime] {part} -> {out / (part + '.json')} ({doc['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
