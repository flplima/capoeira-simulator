"""The left-right mirror of the body: which joint, observation and action channels swap and which flip sign when
the body is reflected through its own middle (x → -x; the model's left is +x).

The ginga reference is its own mirror half a cycle on, the triangle is symmetric and every kick's left clip
mirrors its right, so a mirrored state is as likely as the original and the policy ought to act mirrored in it.
train_mjx.py --mirror trains on every batch and its mirror (Abdolhosseini et al. 2019, "On learning symmetric
locomotion": duplicated data), so each move learns both legs at once.

    uv run python env_mirror.py    # checks that the observation of a mirrored state is the mirrored observation
"""

from __future__ import annotations

import re
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from env_consts import EFFECTORS, EXTRA_MOVES, SIGMA_INIT, SIGMA_ROWS, TERMS

REFLECT = np.array([-1.0, 1.0, 1.0])  # a point's coordinates in the mirror


class Mirror(NamedTuple):
    obs_perm: jax.Array
    obs_sign: jax.Array
    act_perm: jax.Array
    act_sign: jax.Array


def partner(name: str) -> str:
    """The name of a left thing's right twin and vice versa; a thing on the midline is its own."""
    return re.sub(r"_l(_|$)", r"_r\1", name) if re.search(r"_l(_|$)", name) else re.sub(r"_r(_|$)", r"_l\1", name)


def joint_mirror(m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Over the hinge joints (after the root, in qpos order): which joint each one's angle comes from in the
    mirrored body, and with which sign. A rotation about an axis mirrors into a rotation about the mirrored axis
    the other way round, so the sign depends on how the twin's axis is defined: the same as the mirrored axis
    flips the angle (the trunk's twist and side bend), the opposite keeps it (the limbs, as this model defines them)."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    names = [m.joint(j).name for j in range(1, m.njnt)]
    axes = [d.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_axis[j] for j in range(1, m.njnt)]
    perm = np.array([names.index(partner(n)) for n in names])
    sign = np.array([-np.sign((REFLECT * a) @ axes[p]) for a, p in zip(axes, perm)])
    return perm, sign


def obs_blocks(m: mujoco.MjModel) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The observation (MjxTarget._observe, GingaEnv._observe) block by block: its name, where each channel comes
    from in the mirrored body's observation and with which sign. Everything is in the heading frame, so x
    components flip, left and right twins swap, and the ginga phase moves on half a cycle."""
    jp, js = joint_mirror(m)
    ep = np.array([EFFECTORS.index(partner(e)) for e in EFFECTORS])
    side_feature = np.array([1.0, 1.0, -1.0, 1.0, 1.0])
    return [
        ("phase", np.arange(3), np.array([-1.0, -1.0, 1.0])),
        ("orientation", np.arange(6), np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0])),
        ("joints", jp, js),
        ("joint velocities", jp, js),
        ("root velocity", np.arange(3), REFLECT),
        ("angular velocity", np.arange(3), -REFLECT),
        ("effectors", np.repeat(3 * ep, 3) + np.tile(np.arange(3), len(ep)), np.tile(REFLECT, len(ep))),
        ("feet contacts", np.array([1, 0]), np.ones(2)),
        ("reference joints", jp, js),
        ("reference root", np.arange(3), REFLECT),
        ("reference feet", np.array([2, 3, 0, 1]), np.array([-1.0, 1.0, -1.0, 1.0])),
        ("tempo", np.arange(1), np.ones(1)),
        ("kick", np.arange(5), side_feature),
        ("recover", np.arange(5), side_feature),
        ("moves", np.arange(1 + len(EXTRA_MOVES)), np.ones(1 + len(EXTRA_MOVES))),
    ]


def mirror_maps(m: mujoco.MjModel) -> Mirror:
    """The observation's and the action's mirror as index and sign arrays: mirrored = x[perm] * sign."""
    actuated = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
    assert actuated == list(range(1, m.njnt)), "the actuators are expected to drive the hinge joints in their order"
    perm, sign, at = [], [], 0
    for _, p, s in obs_blocks(m):
        perm.append(at + p)
        sign.append(s)
        at += len(p)
    jp, js = joint_mirror(m)
    return Mirror(jnp.asarray(np.concatenate(perm)), jnp.asarray(np.concatenate(sign), jnp.float32), jnp.asarray(jp), jnp.asarray(js, jnp.float32))


def mirror_qpos(q: jax.Array, jp: jax.Array, js: jax.Array) -> jax.Array:
    """A pose in the mirror: x flips, the root quaternion keeps w and x and flips y and z, the joints swap."""
    return jnp.concatenate([-q[:1], q[1:3], q[3:5], -q[5:7], js * q[7:][jp]])


def mirror_qvel(v: jax.Array, jp: jax.Array, js: jax.Array) -> jax.Array:
    """A velocity in the mirror: the linear part like a point, the angular part like a pseudo-vector."""
    return jnp.concatenate([REFLECT * v[:3], -REFLECT * v[3:6], js * v[6:][jp]])


def main() -> None:
    """Play random actions through kick episodes on both legs and compare, at a few moments, the observation of
    each mirrored state with the mirrored observation, block by block. The video martelo's left clip is only
    nearly the mirror of its right (joints within 5°), so its reference blocks are reported apart."""
    from mujoco import mjx

    from env_mjx import MjxGingaEnv

    env = MjxGingaEnv(skill="kicks")
    mirror = mirror_maps(env.m)
    jp, js = (jnp.asarray(x) for x in joint_mirror(env.m))
    n, size, fall = 64, jnp.array(3, jnp.int32), jnp.array(0, jnp.int32)
    sigma = jnp.full((SIGMA_ROWS, len(TERMS)), SIGMA_INIT, jnp.float32)
    moves = np.asarray(env.skill_moves)
    pending = (jnp.asarray(moves[np.arange(n) % len(moves)], jnp.int32), jnp.asarray(np.arange(n) // len(moves) % 2, jnp.int32))
    rng = jax.random.PRNGKey(1)
    state = jax.jit(jax.vmap(env.reset, in_axes=(0, None, None, 0)))(jax.random.split(rng, n), size, fall, pending)
    v_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, None, None, None)))
    forward = jax.vmap(lambda d: mjx.forward(env.mx, d))
    flip_place = jnp.array([-1.0, 1.0, -1.0])

    @jax.jit
    def both(st):
        data = forward(st.data)
        twin = forward(st.data.replace(qpos=jax.vmap(mirror_qpos, (0, None, None))(data.qpos, jp, js), qvel=jax.vmap(mirror_qvel, (0, None, None))(data.qvel, jp, js)))
        mirrored = st._replace(
            phase=st.phase + 0.5,
            frame=st.frame * flip_place,
            side_id=1 - st.side_id,
            anchor=st.anchor * flip_place[:2],
            recover_side=1 - st.recover_side,
            recover_onset=jax.vmap(mirror_qpos, (0, None, None))(st.recover_onset, jp, js),
            recover_place=st.recover_place * flip_place,
        )

        def observe(s, d):
            return env._observe(d, s.phase, s.bpm, s.frame, s.move_id, s.side_id, s.kick_t, s.level, s.anchor, s.recover_id, s.recover_side, s.recover_t, s.recover_onset, s.recover_place, s.recover_sink)

        obs = jax.vmap(observe)(st, data)
        return jnp.abs(jax.vmap(observe)(mirrored, twin) - obs[:, mirror.obs_perm] * mirror.obs_sign), st.move_id == 0

    blocks = obs_blocks(env.m)
    worst = {name: [0.0, 0.0] for name, _, _ in blocks}  # per block: outside the martelo, in it
    for k in range(120):
        rng, key = jax.random.split(rng)
        state, *_ = v_step(state, 0.3 * jax.random.normal(key, (n, env.act_size)), size, fall, sigma)
        if k % 30 == 29:
            err, in_martelo = (np.asarray(x) for x in both(state))
            at = 0
            for name, p, _ in blocks:
                part = err[:, at : at + len(p)]
                worst[name][0] = max(worst[name][0], float(part[~in_martelo].max()) if (~in_martelo).any() else 0.0)
                worst[name][1] = max(worst[name][1], float(part[in_martelo].max()) if in_martelo.any() else 0.0)
                at += len(p)
    exact = {name: w for name, w in worst.items() if not name.startswith("reference")}
    ok = max(max(w) for w in exact.values()) < 1e-3 and max(w[0] for w in worst.values()) < 1e-3
    print(f"{'block':18s} {'other moves':>12s} {'video martelo':>14s}   (largest difference between the mirrored observation and the observation of the mirrored state)")
    for name, (other, martelo) in worst.items():
        print(f"{name:18s} {other:12.2e} {martelo:14.2e}")
    print("mirror ok" if ok else "mirror broken")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
