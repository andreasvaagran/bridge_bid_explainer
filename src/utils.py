import jax
import jax.numpy as jnp
import distrax

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


def _vmap_step_with_reset(step_fn, state, action, rng):
    """`step_fn` here is always auto_reset(...)'s wrapped_step_fn, which
    needs one real, per-environment key for its internal reset -- splits
    `rng` into one key per row of the batch (the vmapped axis) so every
    environment gets its own independently-advancing reset key instead of
    all environments (and all resets within a rollout) sharing one fixed
    key. See auto_reset's docstring in this file for why this exists."""
    num_envs = state.terminated.shape[0]
    keys = jax.random.split(rng, num_envs)
    return jax.vmap(step_fn)(state, action, keys)


def auto_reset(step_fn, init_fn):
    """Auto reset wrapper.

    There are several concerns before staging this wrapper:

    1. Final state (observation)
    When auto restting happened, the termianl (or truncated) state/observation is replaced by initial state/observation,
    This is not problematic if it's termination.
    However, when truncation happened, value of truncated state/observation might be used by agent.
    So we have to preserve the final state (observation) somewhere.
    For example, in Gymnasium,

    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/autoreset.py#L59

    However, currently, truncation does *NOT* actually happens because
    all of Pgx environments (games) are finite-horizon and terminates in reasonable # of steps.
    (NOTE: Chess, Shogi, and Go have `max_termination_steps` option following AlphaZero approach)
    So, curren implementation is enough (so far), and I feel implementing `final_state/observation` is useless and not necessary.

    2. Performance:
    Might harm the performance as it always generates new state.
    Memory usage might be doubled. Need to check.
    """

    def wrapped_step_fn(state, action, rng):
        # NOTE: upstream harukaki/brl reset via `init_fn(state._rng_key)` --
        # the pgx version this repo is built against no longer exposes
        # `_rng_key` on State at all (confirmed: `dir(state)` has nothing
        # matching "rng"/"key"). A prior patch on this codebase (see brl's
        # uncommitted src/utils.py diff) worked around the AttributeError by
        # hardcoding `jax.random.PRNGKey(0)` for every reset -- its own
        # comment flagged this as a stopgap, and it is a real bug: every
        # auto-reset within a rollout would deal the SAME cards, collapsing
        # training-data diversity for every environment that terminates more
        # than once inside one `jax.lax.scan` window (i.e. constantly, since
        # a bidding auction is a handful of steps and rollouts span many).
        # Fixed properly here instead of ported: `rng` is a real, per-call,
        # per-environment key threaded in by the caller (see call sites in
        # single_play_step_two_policy_commpetitive/single_play_step_free_run
        # in this file, which now split a fresh key at every step and pass
        # it through `jax.vmap`'s batched axis -- one real key per
        # environment per reset, exactly matching what `_rng_key` used to
        # provide before this pgx version removed it).
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(  # type: ignore
                _step_count=jnp.int32(0),
                terminated=FALSE,
                truncated=FALSE,
                rewards=jnp.zeros_like(state.rewards),
            ),
            lambda: state,
        )
        state = step_fn(state, action)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            # state is replaced by initial state,
            # but preserve (terminated, truncated, reward)
            lambda: init_fn(rng).replace(  # type: ignore
                terminated=state.terminated,
                truncated=state.truncated,
                rewards=state.rewards,
            ),
            lambda: state,
        )
        return state

    return wrapped_step_fn


def single_play_step_two_policy_commpetitive(
    step_fn, actor_forward_pass, actor_params, opp_forward_pass, opp_params
):
    """
    assume bridge bidding
    """
    # teamB_model_params = teamB_param

    def wrapped_step_fn(state, action, rng):
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)
        rewards1 = state.rewards
        terminated1 = state.terminated
        # print(f"rewards: {state.rewards}")

        # sl model turn
        # print("===01==")
        # print(f"current player: {state.current_player}")
        rng, _rng = jax.random.split(rng)
        logits, _ = opp_forward_pass.apply(
            opp_params,
            state.observation.astype(jnp.float32),
        )
        legal = state.legal_action_mask.astype(jnp.bool_)
        logits = logits.astype(jnp.float32)
        logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=_rng)
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)  # step by left
        rewards2 = state.rewards
        terminated2 = state.terminated
        # print(f"sl model, action: {action}")
        # print(f"rewards: {state.rewards}")

        # actor teammate turn
        # print("===02==")
        # print(f"current player: {state.current_player}")
        rng, _rng = jax.random.split(rng)
        logits, _ = actor_forward_pass.apply(
            actor_params,
            state.observation.astype(jnp.float32),
        )
        legal = state.legal_action_mask.astype(jnp.bool_)
        logits = logits.astype(jnp.float32)
        logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=_rng)
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)  # step by pd
        rewards3 = state.rewards
        terminated3 = state.terminated
        # print(f"actor team, action: {action}")
        # print(f"rewards: {state.rewards}")

        # sl model turn
        # print("===03==")
        # print(f"current player: {state.current_player}")
        rng, _rng = jax.random.split(rng)
        logits, _ = opp_forward_pass.apply(
            opp_params,
            state.observation.astype(jnp.float32),
        )
        legal = state.legal_action_mask.astype(jnp.bool_)
        logits = logits.astype(jnp.float32)
        logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=_rng)
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)  # step by left
        rewards4 = state.rewards
        terminated4 = state.terminated
        # print(f"sl model, action: {action}")
        # print(f"rewards: {state.rewards}")

        rewards = rewards1 + rewards2 + rewards3 + rewards4
        terminated = terminated1 | terminated2 | terminated3 | terminated4
        return state.replace(rewards=rewards, terminated=terminated)

    return wrapped_step_fn


def single_play_step_two_policy_commpetitive_deterministic(
    step_fn, actor_forward_pass, actor_params, opp_forward_pass, opp_params
):
    """
    assume bridge bidding
    """
    # teamB_model_params = teamB_param

    def wrapped_step_fn(state, action, rng):
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)
        rewards1 = state.rewards
        terminated1 = state.terminated
        # print(f"rewards: {state.rewards}")

        # sl model turn
        # print("===01==")
        # print(f"current player: {state.current_player}")
        rng, _rng = jax.random.split(rng)
        logits, _ = opp_forward_pass.apply(
            opp_params,
            state.observation.astype(jnp.float32),
        )
        legal = state.legal_action_mask.astype(jnp.bool_)
        logits = logits.astype(jnp.float32)
        logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
        pi = distrax.Categorical(logits=logits)
        action = pi.mode()
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)  # step by left
        rewards2 = state.rewards
        terminated2 = state.terminated
        # print(f"sl model, action: {action}")
        # print(f"rewards: {state.rewards}")

        # actor teammate turn
        # print("===02==")
        # print(f"current player: {state.current_player}")
        rng, _rng = jax.random.split(rng)
        logits, _ = actor_forward_pass.apply(
            actor_params,
            state.observation.astype(jnp.float32),
        )
        legal = state.legal_action_mask.astype(jnp.bool_)
        logits = logits.astype(jnp.float32)
        logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
        pi = distrax.Categorical(logits=logits)
        action = pi.mode()
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)  # step by pd
        rewards3 = state.rewards
        terminated3 = state.terminated
        # print(f"actor team, action: {action}")
        # print(f"rewards: {state.rewards}")

        # sl model turn
        # print("===03==")
        # print(f"current player: {state.current_player}")
        rng, _rng = jax.random.split(rng)
        logits, _ = opp_forward_pass.apply(
            opp_params,
            state.observation.astype(jnp.float32),
        )
        legal = state.legal_action_mask.astype(jnp.bool_)
        logits = logits.astype(jnp.float32)
        logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
        pi = distrax.Categorical(logits=logits)
        action = pi.mode()
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)  # step by left
        rewards4 = state.rewards
        terminated4 = state.terminated
        # print(f"sl model, action: {action}")
        # print(f"rewards: {state.rewards}")

        rewards = rewards1 + rewards2 + rewards3 + rewards4
        terminated = terminated1 | terminated2 | terminated3 | terminated4
        return state.replace(rewards=rewards, terminated=terminated)

    return wrapped_step_fn


def single_play_step_free_run(
    step_fn, actor_forward_pass, actor_params, opp_forward_pass, opp_params
):
    """
    assume bridge bidding
    """

    def wrapped_step_fn(state, action, rng):
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)
        rewards1 = state.rewards
        terminated1 = state.terminated

        # opposite turn
        action = jnp.zeros_like(action)
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)  # step by left
        rewards2 = state.rewards
        terminated2 = state.terminated

        # actor teammate turn
        rng, _rng = jax.random.split(rng)
        logits, _ = actor_forward_pass.apply(
            actor_params,
            state.observation.astype(jnp.float32),
        )
        legal = state.legal_action_mask.astype(jnp.bool_)
        logits = logits.astype(jnp.float32)
        logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
        pi = distrax.Categorical(logits=logits)
        action = pi.mode()
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)  # step by pd
        rewards3 = state.rewards
        terminated3 = state.terminated

        # opposite turn
        action = jnp.zeros_like(action)
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)  # step by left
        rewards4 = state.rewards
        terminated4 = state.terminated

        rewards = rewards1 + rewards2 + rewards3 + rewards4
        terminated = terminated1 | terminated2 | terminated3 | terminated4
        return state.replace(rewards=rewards, terminated=terminated)

    return wrapped_step_fn


def normal_step(step_fn):
    def wrapped_step_fn(state, action, rng):
        rng, _rng_reset = jax.random.split(rng)
        state = _vmap_step_with_reset(step_fn, state, action, _rng_reset)
        return state

    return wrapped_step_fn
