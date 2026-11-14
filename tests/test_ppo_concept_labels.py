"""training/ppo_concept_labels.py's fast labeller gives the rule engine's concept labels, bit for bit."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from conftest import requires

ROOT = Path(__file__).resolve().parents[1]

REFERENCE = r"""
import json, sys
import numpy as np
from concepts import AuctionContext, HARD_STATE_CONCEPTS, create_sayc_state_authority, decode_hand
from training.ppo_concept_labels import _BIDS
data = np.load(sys.argv[1])
auth = create_sayc_state_authority()

def auction(o):  # a plain decoder to compare with: a scan of every block of four bits
    from concepts.auction_obs import BID_NAMES
    o = o[:480] > 0.5
    rec = []
    for b in range(35):
        for k, call in ((0, BID_NAMES[b]), (4, "X"), (8, "XX")):
            s = np.flatnonzero(o[8 + b * 12 + k: 12 + b * 12 + k])
            if len(s):
                rec.append((int(s[0]), call))
    calls = ["P"] * int(o[4:8].sum())
    prev = None
    for seat, call in rec:
        if prev is not None:
            calls += ["P"] * ((seat - prev - 1) % 4)
        calls.append(call)
        prev = seat
    if rec:
        calls += ["P"] * ((0 - prev - 1) % 4)
    return calls, (-len(calls)) % 4

out = []
for o, a in zip(data["obs"], data["act"]):
    calls, dealer = auction(o)
    d = auth.classify(AuctionContext(bids=calls, bid=_BIDS[int(a)], dealer=dealer), decode_hand(o))
    out.append([int(d[n]) for n in HARD_STATE_CONCEPTS])
print(json.dumps(out))
"""


@requires("data/wbridge5/prepared/test")
def test_labels_identical_with_the_speedups(tmp_path):
    from evaluation.common import load_test_split
    from training.ppo_concept_labels import build_concept_targets
    test = load_test_split()
    idx = np.random.default_rng(3).choice(len(test), 1500, replace=False)
    obs, act = test.obs_f32(idx), test.action[idx]
    np.savez(tmp_path / "d.npz", obs=obs, act=act)
    r = subprocess.run([sys.executable, "-c", REFERENCE, str(tmp_path / "d.npz")], cwd=ROOT, capture_output=True,
                       text=True, env={"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr[-2000:]
    ref = np.array(json.loads(r.stdout.strip().splitlines()[-1]), np.float32)
    new = build_concept_targets(obs, act)
    assert ref.shape == new.shape and ref.sum() > 0
    assert np.array_equal(ref, new)

    # the PPO fast path: packed bits, compiled rules, per-auction plans -- cold cache, then all hits
    from training.ppo_concept_labels import FastLabeller, pack_obs
    lab = FastLabeller()
    auc, hand = pack_obs(obs)
    assert np.array_equal(ref, lab.label(auc, hand, act))
    assert lab.misses > 0
    hits = lab.hits
    assert np.array_equal(ref, lab.label(auc, hand, act))
    assert lab.hits == hits + len(act)

