"""Vectorised MJX (MuJoCo XLA) environment for Capoeira motion imitation.

Ports GingaEnv (env.py) to pure JAX + mujoco.mjx so thousands of environments
run in parallel on a single GPU (or Modal's free tier) while producing the
exact same 138-dim observation layout, PD actuator targets, and reward terms
used by train.py, check.py, export_web.py, and the browser simulation.
"""

from __future__ import annotations

import jax.numpy as jnp
from mujoco import mjx
import numpy as np

from env_consts import ALLOWED_CONTACT, EFFECTORS, EXTRA_MOVES, FRAME_SKIP, GROUND_SITES, KICK_MOVES, KNEE_CLEAR, MOVES, NEVER_CONTACT
from mjx_model import load_mjx_model
from mjx_reset import MjxReset
from mjx_step import MjxStep
from mjx_tables import build_ref_tables
from mjx_target import MjxTarget


class MjxGingaEnv(MjxTarget, MjxReset, MjxStep):
    """Vectorised MJX Capoeira environment with identical obs/act/reward to GingaEnv."""

    def __init__(
        self,
        xml: str = "humanoid.xml",
        skill: str = "ginga",
        max_seconds: float = 40.0,
        impl: str = "jax",
        focus: str = "",
        only: str = "",
    ):
        self.skill = skill
        self.m, self.mx = load_mjx_model(xml, impl=impl, strong_arms=(skill == "recover"))
        self.tables = build_ref_tables(self.m)
        self.control_dt = float(self.m.opt.timestep * FRAME_SKIP)
        self.max_steps = int(max_seconds / self.control_dt)

        self.joint_qpos = jnp.arange(7, self.m.nq)
        self.joint_qvel = jnp.arange(6, self.m.nv)
        self.ctrl_lo = jnp.asarray(self.m.actuator_ctrlrange[:, 0], jnp.float32)
        self.ctrl_hi = jnp.asarray(self.m.actuator_ctrlrange[:, 1], jnp.float32)
        self.effector_ids = jnp.asarray([self.m.body(n).id for n in EFFECTORS], jnp.int32)
        self.head_body_id = int(self.m.body("head").id)
        self.hips_body_id = int(self.m.body("hips").id)
        self.chest_body_id = int(self.m.body("chest").id)
        self.floor_geom_id = int(self.m.geom("floor").id)
        self.foot_l_geom_id = int(self.m.geom("foot_l").id)
        self.foot_r_geom_id = int(self.m.geom("foot_r").id)
        self.sole_sites = jnp.asarray(
            [[self.m.site(f"heel_{s}").id, self.m.site(f"toe_{s}").id] for s in "lr"], jnp.int32
        )
        self.ground_sites = jnp.asarray([self.m.site(n).id for n in GROUND_SITES], jnp.int32)
        self.ground_clear = jnp.asarray(
            [KNEE_CLEAR if n.startswith("knee") else 0.0 for n in GROUND_SITES], jnp.float32
        )
        self.arm_qpos_ids = jnp.asarray(
            [int(self.m.jnt_qposadr[j]) for j in range(self.m.njnt) if self.m.joint(j).name.startswith(("shoulder", "elbow"))],
            jnp.int32,
        )

        # Contact masks over static MJX contact pairs involving the floor
        dx0 = mjx.make_data(self.mx, impl=impl)
        dx0 = mjx.collision(self.mx, mjx.kinematics(self.mx, dx0))
        c = dx0._impl.contact
        g1, g2 = np.asarray(c.geom1), np.asarray(c.geom2)
        is_floor = (g1 == self.floor_geom_id) | (g2 == self.floor_geom_id)
        other = np.where(g1 == self.floor_geom_id, g2, np.where(g2 == self.floor_geom_id, g1, -1))
        allowed_ids = {self.m.geom(n).id for n in ALLOWED_CONTACT}
        never_ids = {self.m.geom(n).id for n in NEVER_CONTACT}

        self.contact_is_foot_l = jnp.asarray(is_floor & (other == self.foot_l_geom_id))
        self.contact_is_foot_r = jnp.asarray(is_floor & (other == self.foot_r_geom_id))
        self.contact_is_never = jnp.asarray(is_floor & np.isin(other, list(never_ids)))
        self.contact_is_disallowed = jnp.asarray(is_floor & (~np.isin(other, list(allowed_ids))))
        self.contact_is_floor = jnp.asarray(is_floor)

        # Which moves this skill practices: 0..len(KICK_MOVES)-1 = kicks, len(KICK_MOVES)..len(MOVES)-1 = dodges
        self.num_kicks = len(KICK_MOVES)
        self.esquiva_move_id = int(MOVES.index("esquiva"))
        self.extra_move_ids = jnp.asarray([MOVES.index(m) for m in EXTRA_MOVES], jnp.int32)
        if skill == "kicks":
            self.skill_moves = jnp.arange(len(KICK_MOVES), dtype=jnp.int32)
        elif skill == "esquiva":
            self.skill_moves = jnp.arange(len(KICK_MOVES), len(MOVES), dtype=jnp.int32)
        else:
            self.skill_moves = jnp.arange(len(MOVES), dtype=jnp.int32)
        if focus:  # half the draws go to this move, the rest are shared as before
            self.skill_moves = jnp.concatenate([self.skill_moves, jnp.full(len(self.skill_moves), MOVES.index(focus), jnp.int32)])
        if only:  # every draw goes to this move
            self.skill_moves = jnp.array([MOVES.index(only)], jnp.int32)

        self.obs_size = 138 + len(EXTRA_MOVES)
        self.act_size = int(self.m.nu)
        self._dx_template = dx0
