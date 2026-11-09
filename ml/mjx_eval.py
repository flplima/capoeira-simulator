"""Evaluate a policy inside the MJX trainer: thousands of moves asked from the ginga, played without exploration
noise, counted the way check.py counts them (finished standing = still up KICK_SETTLE s into the ginga after)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO

import jax
import jax.numpy as jnp
import numpy as np

from env_consts import MOVES
from train_nets import apply_mlp, apply_norm, load_pt_checkpoint

EVAL_SECONDS = 10.0  # the wait for the entry (up to a ginga cycle, 4 s at 60 bpm), the move, and the settling after it


def evaluate(env, params: dict, norm, num_envs: int, rng: jax.Array, curriculum: jax.Array, fall_level: jax.Array, sigma: jax.Array) -> dict:
    """Per move and leg, [finished standing, asked]: the environments are shared evenly between the skill's moves and
    both legs, each starts in the ginga waiting for its move, and only its first move counts (afterwards the
    environment restarts at random, as in training)."""
    pairs = [(int(m), side) for m in np.unique(np.asarray(env.skill_moves)) for side in (0, 1)]
    assign = np.array([pairs[i % len(pairs)] for i in range(num_envs)])
    pending = (jnp.asarray(assign[:, 0], jnp.int32), jnp.asarray(assign[:, 1], jnp.int32))
    keys = jax.random.split(rng, num_envs)
    state = jax.jit(jax.vmap(env.reset, in_axes=(0, None, None, 0)))(keys, curriculum, fall_level, pending)
    v_step = jax.vmap(env.step, in_axes=(0, 0, None, None, None))

    @jax.jit
    def roll(state):
        def body(carry, _):
            st, counted, stood = carry
            action = apply_mlp(params["actor"], apply_norm(norm, st.obs))
            st, _, _, _, info = v_step(st, action, curriculum, fall_level, sigma)
            fresh = info["kick_attempt"] & ~counted
            return (st, counted | fresh, stood | (fresh & info["kick_ok"])), None

        empty = jnp.zeros(num_envs, bool)
        (_, counted, stood), _ = jax.lax.scan(body, (state, empty, empty), None, length=int(EVAL_SECONDS / env.control_dt))
        return counted, stood

    counted, stood = (np.asarray(x) for x in roll(state))
    report: dict = {}
    for k, (move, side) in enumerate(pairs):
        mine = np.arange(num_envs) % len(pairs) == k
        report.setdefault(MOVES[move], {})["r" if side == 1 else "l"] = [int(np.sum(stood & mine)), int(np.sum(counted & mine))]
    return report


def report(env, out: Path, log: IO[str], params: dict, norm, shape: tuple[int, int, float], num_envs: int, rng: jax.Array, curriculum, fall_level, sigma) -> None:
    """At the end of a run: evaluate the last policy and best.pt, print the counts and append them to log.jsonl."""
    best_params, best_norm, _ = load_pt_checkpoint(str(out / "best.pt"), *shape)
    line = {
        "eval": {
            name: evaluate(env, p, n, num_envs, rng, curriculum, fall_level, sigma)
            for name, (p, n) in (("last", (params, norm)), ("best", (best_params, best_norm)))
        }
    }
    print(json.dumps(line), flush=True)
    log.write(json.dumps(line) + "\n")
    log.flush()
