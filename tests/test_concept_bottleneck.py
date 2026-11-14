"""Gradient isolation per mode: in independent and sequential mode policy gradients never reach the trunk
or concept head; in joint mode they reach the trunk."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import pytest

from conftest import REPO, requires
from training.concept_bottleneck import build_net, OBS_SIZE, NUM_ACTIONS


@pytest.mark.parametrize("mode", ["joint", "independent", "sequential"])
def test_net_builds_and_applies(mode):
    net = build_net(mode, "DeepMind", "relu", num_hard=10, has_value=False)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_SIZE)))
    out = net.apply(params, jnp.zeros((2, OBS_SIZE)))
    for key in ("policy_logits", "concept_logits", "concept_acts", "hidden", "concept_only_logits"):
        assert key in out, f"{mode}: missing {key!r} in net output"
    assert out["policy_logits"].shape == (2, NUM_ACTIONS)
    assert out["concept_acts"].shape == (2, 10)


@pytest.mark.parametrize("mode,expect_trunk_gradient", [
    ("joint", True),
    ("independent", False),
    ("sequential", False),
])
def test_policy_loss_gradient_isolation(mode, expect_trunk_gradient):
    """The core correctness property: does a gradient computed purely from
    policy_logits reach hard_state_head's weights (the concept head)?
    Should be True for "joint" (soft bottleneck: hidden feeds policy_head
    directly, and the concept head's own weights receive no policy
    gradient via any path other than through hidden, which doesn't
    connect back to hard_state_head's weights -- so this specifically
    checks the TRUNK, not hard_state_head itself, see below) and always
    False into hard_state_head specifically, in every mode, since
    concept_acts is stop-gradiented before it ever reaches policy_head."""
    net = build_net(mode, "DeepMind", "relu", num_hard=10, has_value=False)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_SIZE)))
    obs = jax.random.normal(jax.random.PRNGKey(1), (4, OBS_SIZE))

    def policy_only_loss(p):
        out = net.apply(p, obs)
        return jnp.sum(out["policy_logits"] ** 2)

    grads = jax.grad(policy_only_loss)(params)

    # hard_state_head's own weights must NEVER receive gradient from the
    # policy loss alone, in any mode -- concept_acts is always
    # stop-gradiented before reaching policy_head.
    hard_head_grad_norm = jnp.sum(grads["hard_state_head"]["w"] ** 2)
    assert float(hard_head_grad_norm) == 0.0, (
        f"{mode}: policy-only loss leaked a gradient into hard_state_head's "
        f"own weights -- concept_acts must be stop_gradiented before "
        f"policy_head in every mode"
    )

    # The trunk's own weights (e.g. the first linear layer, shared by
    # every mode's `_deepmind_trunk`) DO receive gradient in "joint" mode
    # (policy_head sees raw `hidden`) and must NOT in "independent"/
    # "sequential" (policy_head sees only stop_gradient(concept_acts)).
    trunk_leaf_name = next(iter(k for k in grads if k.startswith("linear") and "hard_state" not in k
                                 and "policy_head" not in k and "concept_only" not in k))
    trunk_grad_norm = jnp.sum(grads[trunk_leaf_name]["w"] ** 2)
    if expect_trunk_gradient:
        assert float(trunk_grad_norm) > 0.0, f"{mode}: expected policy loss to reach the trunk, it didn't"
    else:
        assert float(trunk_grad_norm) == 0.0, f"{mode}: policy loss leaked into the trunk, breaking the true-bottleneck property"


@requires("data/wbridge5/prepared")
def test_sequential_mode_phase_b_freezes_concept_head_exactly():
    """In "sequential" mode's phase B hard_state_head must not move at all. Masking only the gradient is not
    enough: Adam's momentum from phase A keeps producing small updates, so training/sl.py also masks the final
    updates. Runs training/sl.py across the phase A/B boundary and compares two checkpoints saved within
    phase B, which must be identical for hard_state_head."""
    import pickle
    import subprocess
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        # the parent process already holds JAX's preallocated GPU memory: without these the child's
        # CUDA/cuBLAS init runs out of memory
        child_env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                          XLA_PYTHON_CLIENT_MEM_FRACTION="0.15")
        result = subprocess.run(
            [sys.executable, "-m", "training.sl",
             f"prepared_path={os.path.join(REPO, 'data', 'wbridge5', 'prepared')}", f"save_path={tmpdir}",
             "iterations=40", "train_batch=64", "num_data_workers=0",
             "print_every=200", "save_every=10",
             "actor_model_type=DeepMind", "bottleneck_mode=sequential",
             "concept_phase_iterations=10"],
            cwd=REPO,
            env=child_env,
            capture_output=True, timeout=90,
        )
        assert result.returncode == 0, (
            f"training/sl.py subprocess failed (code {result.returncode})\n"
            f"STDOUT: {result.stdout.decode()[-2000:]}\n"
            f"STDERR: {result.stderr.decode()[-2000:]}"
        )

        with open(os.path.join(tmpdir, "params-10.pkl"), "rb") as f:
            p10 = pickle.load(f)["params"]
        with open(os.path.join(tmpdir, "params-30.pkl"), "rb") as f:
            p30 = pickle.load(f)["params"]

        hard_diff = float(np.abs(np.asarray(p10["hard_state_head"]["w"])
                                  - np.asarray(p30["hard_state_head"]["w"])).max())
        policy_diff = float(np.abs(np.asarray(p10["policy_head"]["w"])
                                    - np.asarray(p30["policy_head"]["w"])).max())

        assert hard_diff == 0.0, (
            f"hard_state_head drifted by {hard_diff} during phase B -- should be "
            f"byte-frozen once concept_phase_iterations is passed"
        )
        assert policy_diff > 0.0, (
            "policy_head did not change at all during phase B -- it should be "
            "the only thing training at this point"
        )


def test_trunk_and_concept_param_mask_zeroes_only_frozen_params():
    from training.concept_bottleneck import trunk_and_concept_param_mask

    net = build_net("sequential", "DeepMind", "relu", num_hard=10, has_value=False)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_SIZE)))
    mask = trunk_and_concept_param_mask(params)

    assert float(jnp.sum(mask["hard_state_head"]["w"])) == 0.0
    assert float(jnp.mean(mask["policy_head"]["w"])) == 1.0
    assert float(jnp.mean(mask["concept_only_policy_head"]["w"])) == 1.0


@pytest.mark.parametrize("mode", ["joint", "independent", "sequential"])
@pytest.mark.parametrize("model_type", ["DeepMind", "FAIR"])
def test_load_checkpoint_infers_model_type_when_the_field_is_missing(tmp_path, mode, model_type):
    """Checkpoints saved without `model_type` get it inferred. policy_head's input width minus the concept
    count cannot tell (it is 0 in the true-bottleneck modes): every independent/sequential DeepMind
    checkpoint would read as FAIR."""
    import pickle
    from concepts.vocabulary import HARD_STATE_CONCEPTS
    from training.checkpoint import load_checkpoint
    net = build_net(mode, model_type, "relu", num_hard=len(HARD_STATE_CONCEPTS), has_value=False)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_SIZE)))
    path = tmp_path / "ckpt.pkl"
    with open(path, "wb") as f:
        pickle.dump({"params": params, "bottleneck_mode": mode}, f)  # no model_type, no hard_concepts
    ck = load_checkpoint(str(path))
    assert ck.model_type == model_type
    assert ck.hard_concepts == list(HARD_STATE_CONCEPTS)
