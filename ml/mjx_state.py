"""Per-environment state of MjxGingaEnv."""

from __future__ import annotations

from typing import NamedTuple

import jax
from mujoco import mjx


class EnvState(NamedTuple):
    data: mjx.Data
    obs: jax.Array
    rng: jax.Array
    phase: jax.Array
    bpm: jax.Array
    frame: jax.Array            # (3,): x, z, yaw
    # Active kick / esquiva: move_id (-1 = none, 0..6 = kicks, 7..9 = dodges)
    move_id: jax.Array          # int32 scalar
    side_id: jax.Array          # int32 scalar (0 = 'l', 1 = 'r')
    kick_t: jax.Array           # float32 scalar
    level: jax.Array            # int32 scalar (0..3)
    anchor: jax.Array           # (2,)
    # Scheduled move waiting for entry phase
    pending_move: jax.Array     # int32 scalar (-1 = none)
    pending_side: jax.Array     # int32 scalar
    pending_entry: jax.Array    # float32 scalar
    # Active recovery: recover_id (-1 = none, 0 = quatro, 1 = negativa, 2 = joelhos)
    recover_id: jax.Array       # int32 scalar
    recover_side: jax.Array     # int32 scalar (0 = 'l', 1 = 'r')
    recover_t: jax.Array        # float32 scalar
    recover_onset: jax.Array    # (36,)
    recover_place: jax.Array    # (3,)
    recover_sink: jax.Array     # float32 scalar
    # Episode bookkeeping & domain randomization
    steps: jax.Array            # int32 scalar
    kicked: jax.Array           # bool scalar
    from_ginga: jax.Array       # bool scalar
    after_kick: jax.Array       # float32 scalar (-1 = not finished kick yet)
    after_recover: jax.Array    # float32 scalar (-1 = not recovered yet)
    stuck_t: jax.Array          # float32 scalar
    from_fall: jax.Array        # bool scalar
    landing_seen: jax.Array     # int32 scalar (0 = none, 1..3)
    act_scale: jax.Array        # float32 scalar (per-episode PD/action authority randomization)
