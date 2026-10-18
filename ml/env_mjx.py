"""Vectorised MJX (MuJoCo XLA) environment for Capoeira motion imitation.

Ports GingaEnv (env.py) to pure JAX + mujoco.mjx so thousands of environments
run in parallel on a single GPU (or Modal's free tier) while producing the
exact same 138-dim observation layout, PD actuator targets, and reward terms
used by train.py, check.py, export_web.py, and the browser simulation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from env import (
    ACTION_SCALE,
    ALLOWED_CONTACT,
    ANCHOR_LIMIT,
    BENT,
    BLEND_IN,
    BLEND_OUT,
    DIVE,
    DIVE_UNTIL,
    DODGES,
    EFFECTORS,
    EXTRA_MOVES,
    FALL_SPEEDS,
    FALL_TIME,
    FRAME_SKIP,
    GINGA_FIRST,
    GRAVITY,
    GROUND_SITES,
    HEAD_ABOVE,
    KICK_EVERY,
    KICK_MOVES,
    KNEE_CLEAR,
    LANDINGS,
    LOST,
    LOW_BELOW,
    LOW_HEAD,
    LOW_HIPS,
    MAX_BPM,
    MIN_BPM,
    MIN_PACE,
    MOVE_EPISODE_TAIL,
    MOVES,
    NEVER_CONTACT,
    PIVOT_SCALE,
    PIVOT_WEIGHT,
    RECOVER_TAIL,
    STRONG_ARMS,
    STRIKE_SCALE,
    STRIKE_WEIGHT,
    STUCK,
    TEMPO_CHANGE_EVERY,
    TEMPO_SKEW,
    TROUBLE_DROP,
    TROUBLE_TILT,
    WAIT_FROM,
    WAIT_SPAN,
    Floor,
    Kicks,
    Reference,
)

# Self-collision geom pairs kept in MJX (arms vs head/chest, left leg vs right leg).
# All 17 body geoms still collide with the floor plane.
MJX_SELF_PAIRS = [
    ("thigh_l", "thigh_r"),
    ("shin_l", "shin_r"),
    ("thigh_l", "shin_r"),
    ("thigh_r", "shin_l"),
    ("foot_l", "shin_r"),
    ("foot_r", "shin_l"),
    ("foot_l", "foot_r"),
    ("forearm_l", "head"),
    ("forearm_r", "head"),
    ("hand_l", "head"),
    ("hand_r", "head"),
    ("upperarm_l", "head"),
    ("upperarm_r", "head"),
    ("forearm_l", "chest"),
    ("forearm_r", "chest"),
    ("hand_l", "chest"),
    ("hand_r", "chest"),
]


def load_mjx_model(
    xml_path: str = "humanoid.xml", impl: str = "jax", strong_arms: bool = False
) -> tuple[mujoco.MjModel, mjx.Model]:
    """Load humanoid.xml with explicit self-collision pairs and MJX solver settings."""
    with open(xml_path) as f:
        xml = f.read()
    pair_xml = (
        "  <contact>\n"
        + "\n".join(f'    <pair geom1="{a}" geom2="{b}"/>' for a, b in MJX_SELF_PAIRS)
        + "\n  </contact>\n</mujoco>"
    )
    xml_mjx = xml.replace('contype="6" conaffinity="5"', 'contype="2" conaffinity="1"').replace("</mujoco>", pair_xml)
    m = mujoco.MjModel.from_xml_string(xml_mjx)
    if strong_arms:
        for i in range(m.nu):
            name = m.actuator(i).name
            joint = "shoulder" if name.startswith("shoulder") else "elbow" if name.startswith("elbow") else None
            if joint:
                kp, kv = m.actuator_gainprm[i, 0], m.actuator_biasprm[i, 2]
                strong_kp, strong_force = STRONG_ARMS[joint]
                m.actuator_gainprm[i, 0] = strong_kp
                m.actuator_biasprm[i, 1:3] = -strong_kp, kv * np.sqrt(strong_kp / kp)
                m.actuator_forcerange[i] = -strong_force, strong_force
    m.opt.iterations = 4
    m.opt.ls_iterations = 8
    mx = mjx.put_model(m, impl=impl)
    return m, mx


# ---- Pure JAX quaternion & frame helpers ------------------------------------


def quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_angle(a: jax.Array, b: jax.Array) -> jax.Array:
    dot = jnp.clip(jnp.abs(jnp.sum(a * b, axis=-1)), 0.0, 1.0 - 1e-7)
    return 2.0 * jnp.arccos(dot)


def nlerp(a: jax.Array, b: jax.Array, t: jax.Array) -> jax.Array:
    dot = jnp.sum(a * b, axis=-1, keepdims=True)
    b = jnp.where(dot < 0.0, -b, b)
    q = a + (b - a) * t
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


def yaw_of(q: jax.Array) -> jax.Array:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    fx = 2.0 * (x * z + w * y)
    fz = 1.0 - 2.0 * (x * x + y * y)
    return jnp.arctan2(fx, fz)


def yaw_matrix(yaw: jax.Array) -> jax.Array:
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    z = jnp.zeros_like(yaw)
    o = jnp.ones_like(yaw)
    return jnp.stack(
        [
            jnp.stack([c, z, -s], axis=-1),
            jnp.stack([z, o, z], axis=-1),
            jnp.stack([s, z, c], axis=-1),
        ],
        axis=-2,
    )


def quat_matrix(q: jax.Array) -> jax.Array:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack(
        [
            jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def place_xz(frame: jax.Array, xz: jax.Array) -> jax.Array:
    c, s = jnp.cos(frame[2]), jnp.sin(frame[2])
    x, z = xz[..., 0], xz[..., 1]
    return jnp.stack([c * x + s * z + frame[0], -s * x + c * z + frame[1]], axis=-1)


def unplace_xz(frame: jax.Array, xz: jax.Array) -> jax.Array:
    c, s = jnp.cos(frame[2]), jnp.sin(frame[2])
    x, z = xz[..., 0] - frame[0], xz[..., 1] - frame[1]
    return jnp.stack([c * x - s * z, s * x + c * z], axis=-1)


def place_pose(frame: jax.Array, q: jax.Array, v: jax.Array, feet: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    xz = place_xz(frame, jnp.stack([q[0], q[2]]))
    q = q.at[0].set(xz[0]).at[2].set(xz[1])
    yaw_q = jnp.array([jnp.cos(frame[2] / 2.0), 0.0, jnp.sin(frame[2] / 2.0), 0.0], dtype=q.dtype)
    q = q.at[3:7].set(quat_mul(yaw_q, q[3:7]))
    c, s = jnp.cos(frame[2]), jnp.sin(frame[2])
    vx, vz = c * v[0] + s * v[2], -s * v[0] + c * v[2]
    v = v.at[0].set(vx).at[2].set(vz)
    return q, v, place_xz(frame, feet)


def smoothstep(x: jax.Array) -> jax.Array:
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ---- Static reference tables on device --------------------------------------


class RefTables(NamedTuple):
    # Ginga reference: (T_tempos, N_samples, ...)
    ginga_tempos: jax.Array      # (5,)
    ginga_qpos: jax.Array        # (5, N, 36)
    ginga_qvel: jax.Array        # (5, N, 35)
    ginga_feet: jax.Array        # (5, N, 2, 2)
    triangle: jax.Array          # (3, 2)
    # Kicks & esquiva: move_id in {0: martelo, 1: bencao, 2: esquiva}, side_id in {0: l, 1: r}, level in 0..3
    move_durations: jax.Array    # (3,)
    move_fps: jax.Array          # (3,)
    move_lengths: jax.Array      # (3,)
    move_qpos: jax.Array         # (3, 2, 4, max_K, 36)
    move_qvel: jax.Array         # (3, 2, 4, max_K, 35)
    move_feet: jax.Array         # (3, 2, 4, max_K, 2, 2)
    move_mask: jax.Array         # (3, 2, 4, max_K, 2)
    # Floor recovery clips: landing_id in {0: quatro, 1: negativa, 2: joelhos}, side_id in {0: l, 1: r}
    floor_fps: float
    floor_durations: jax.Array   # (3, 2)
    floor_phases: jax.Array      # (3, 2)
    floor_lengths: jax.Array     # (3, 2)
    floor_qpos: jax.Array        # (3, 2, max_F, 36)
    floor_qvel: jax.Array        # (3, 2, max_F, 35)
    floor_feet: jax.Array        # (3, 2, max_F, 2, 2)
    floor_mask: jax.Array        # (3, 2, max_F, 2)
    # Recorded falls pool
    falls_qpos: jax.Array        # (P, 36)
    falls_qvel: jax.Array        # (P, 35)
    falls_phase: jax.Array       # (P,)
    falls_bpm: jax.Array         # (P,)
    falls_move: jax.Array        # (P, 4) int32
    falls_side: jax.Array        # (P, 4) int32
    falls_place: jax.Array       # (P, 4, 3) float32


def build_ref_tables(
    m: mujoco.MjModel,
    reference_path: str = "data/ginga_ref.npz",
    kicks_path: str = "data/kicks.npz",
    esquiva_path: str = "data/esquiva.npz",
    floor_path: str = "data/floor.npz",
    falls_path: str = "data/falls.npz",
) -> RefTables:
    ref = Reference(reference_path, m)
    kicks = Kicks([kicks_path, esquiva_path], m)
    floor = Floor(floor_path, m)
    falls = dict(np.load(falls_path))

    n_moves = len(MOVES)
    n_levels = len(kicks.amplitudes)
    max_k = max(len(kicks.clips[mv, s, lv]["qpos"]) for mv in MOVES for s in "lr" for lv in range(n_levels))
    move_durations = np.zeros(n_moves, np.float32)
    move_fps = np.zeros(n_moves, np.float32)
    move_lengths = np.zeros(n_moves, np.int32)
    move_qpos = np.zeros((n_moves, 2, n_levels, max_k, m.nq), np.float32)
    move_qvel = np.zeros((n_moves, 2, n_levels, max_k, m.nv), np.float32)
    move_feet = np.zeros((n_moves, 2, n_levels, max_k, 2, 2), np.float32)
    move_mask = np.zeros((n_moves, 2, n_levels, max_k, 2), np.float32)

    for mi, mv in enumerate(MOVES):
        for si, s in enumerate("lr"):
            for lv in range(n_levels):
                c = kicks.clips[mv, s, lv]
                k_len = len(c["qpos"])
                move_durations[mi] = c["duration"]
                move_fps[mi] = c["fps"]
                move_lengths[mi] = k_len
                move_qpos[mi, si, lv, :k_len] = c["qpos"]
                move_qpos[mi, si, lv, k_len:] = c["qpos"][-1]
                move_qvel[mi, si, lv, :k_len] = c["qvel"]
                move_feet[mi, si, lv, :k_len] = c["feet"]
                move_feet[mi, si, lv, k_len:] = c["feet"][-1]
                move_mask[mi, si, lv, :k_len] = c["mask"]
                move_mask[mi, si, lv, k_len:] = c["mask"][-1]

    n_land = len(LANDINGS)
    max_f = max(len(floor.clips[ld, s]["qpos"]) for ld in LANDINGS for s in "lr")
    floor_durations = np.zeros((n_land, 2), np.float32)
    floor_phases = np.zeros((n_land, 2), np.float32)
    floor_lengths = np.zeros((n_land, 2), np.int32)
    floor_qpos = np.zeros((n_land, 2, max_f, m.nq), np.float32)
    floor_qvel = np.zeros((n_land, 2, max_f, m.nv), np.float32)
    floor_feet = np.zeros((n_land, 2, max_f, 2, 2), np.float32)
    floor_mask = np.zeros((n_land, 2, max_f, 2), np.float32)

    for li, ld in enumerate(LANDINGS):
        for si, s in enumerate("lr"):
            c = floor.clips[ld, s]
            f_len = len(c["qpos"])
            floor_durations[li, si] = c["duration"]
            floor_phases[li, si] = c["phase"]
            floor_lengths[li, si] = f_len
            floor_qpos[li, si, :f_len] = c["qpos"]
            floor_qpos[li, si, f_len:] = c["qpos"][-1]
            floor_qvel[li, si, :f_len] = c["qvel"]
            floor_feet[li, si, :f_len] = c["feet"]
            floor_feet[li, si, f_len:] = c["feet"][-1]
            floor_mask[li, si, :f_len] = c["mask"]
            floor_mask[li, si, f_len:] = c["mask"][-1]

    # Precompute exact landing move, side, and placement for each recorded fall at each FALL_SPEEDS level
    n_falls = len(falls["qpos"])
    n_flevels = len(FALL_SPEEDS)
    falls_move = np.zeros((n_falls, n_flevels), np.int32)
    falls_side = np.zeros((n_falls, n_flevels), np.int32)
    falls_place = np.zeros((n_falls, n_flevels, 3), np.float32)
    scratch = mujoco.MjData(m)
    sole_ids = [[m.site(f"heel_{s}").id, m.site(f"toe_{s}").id] for s in "lr"]
    head_id = m.body("head").id
    for fi in range(n_falls):
        pq = falls["qpos"][fi]
        pv_raw = falls["qvel"][fi]
        p_phase = float(falls["phase"][fi])
        scratch.qpos[:] = pq
        mujoco.mj_kinematics(m, scratch)
        mujoco.mj_comPos(m, scratch)
        feet_pts = np.array([scratch.site_xpos[ids].mean(axis=0)[[0, 2]] for ids in sole_ids])
        com_offset = scratch.subtree_com[1][[0, 2]] - feet_pts.mean(axis=0)
        w, x, y, z = pq[3:7]
        yaw = float(np.arctan2(2.0 * (x * z + w * y), 1.0 - 2.0 * (x * x + y * y)))
        c_y, s_y = np.cos(yaw), np.sin(yaw)
        up = scratch.xpos[head_id] - pq[:3]
        bent_val = float(np.arctan2(s_y * up[0] + c_y * up[2], up[1]))
        def_side = 1 if (p_phase < 0.25 or p_phase >= 0.75) else 0
        for flv, spd in enumerate(FALL_SPEEDS):
            lean = com_offset + 0.3 * pv_raw[[0, 2]] * spd
            side_way = c_y * lean[0] - s_y * lean[1]
            forward = s_y * lean[0] + c_y * lean[1]
            if bent_val > BENT:
                mv_i, sd_i = 2, def_side
            elif abs(forward) >= abs(side_way):
                mv_i, sd_i = (2 if forward > 0 else 0), def_side
            else:
                stance = int(np.argmin(np.linalg.norm(feet_pts - (feet_pts.mean(axis=0) + lean), axis=1)))
                mv_i, sd_i = 1, (1 if stance == 0 else 0)
            land0 = floor_qpos[mv_i, sd_i, 0]
            norm_l = float(np.linalg.norm(lean))
            target_xz = pq[[0, 2]] + (0.25 * lean / norm_l if norm_l > 1e-6 else 0.0)
            placed_land = np.array([c_y * land0[0] + s_y * land0[2], -s_y * land0[0] + c_y * land0[2]])
            falls_move[fi, flv] = mv_i
            falls_side[fi, flv] = sd_i
            falls_place[fi, flv] = [target_xz[0] - placed_land[0], target_xz[1] - placed_land[1], yaw]

    return RefTables(
        ginga_tempos=jnp.asarray(ref.tempos, jnp.float32),
        ginga_qpos=jnp.asarray(ref.qpos, jnp.float32),
        ginga_qvel=jnp.asarray(ref.qvel, jnp.float32),
        ginga_feet=jnp.asarray(ref.feet, jnp.float32),
        triangle=jnp.asarray(ref.triangle, jnp.float32),
        move_durations=jnp.asarray(move_durations),
        move_fps=jnp.asarray(move_fps),
        move_lengths=jnp.asarray(move_lengths),
        move_qpos=jnp.asarray(move_qpos),
        move_qvel=jnp.asarray(move_qvel),
        move_feet=jnp.asarray(move_feet),
        move_mask=jnp.asarray(move_mask),
        floor_fps=float(floor.fps),
        floor_durations=jnp.asarray(floor_durations),
        floor_phases=jnp.asarray(floor_phases),
        floor_lengths=jnp.asarray(floor_lengths),
        floor_qpos=jnp.asarray(floor_qpos),
        floor_qvel=jnp.asarray(floor_qvel),
        floor_feet=jnp.asarray(floor_feet),
        floor_mask=jnp.asarray(floor_mask),
        falls_qpos=jnp.asarray(falls["qpos"], jnp.float32),
        falls_qvel=jnp.asarray(falls["qvel"], jnp.float32),
        falls_phase=jnp.asarray(falls["phase"], jnp.float32),
        falls_bpm=jnp.asarray(falls["bpm"], jnp.float32),
        falls_move=jnp.asarray(falls_move, jnp.int32),
        falls_side=jnp.asarray(falls_side, jnp.int32),
        falls_place=jnp.asarray(falls_place, jnp.float32),
    )


# ---- Pure JAX sampling functions --------------------------------------------


def sample_ginga(tables: RefTables, phase: jax.Array, bpm: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    tempos = tables.ginga_tempos
    bpm_c = jnp.clip(bpm, tempos[0], tempos[-1])
    j = jnp.clip(jnp.searchsorted(tempos, bpm_c), 0, len(tempos) - 1)
    i = jnp.maximum(j - 1, 0)
    denom = jnp.where(i == j, 1.0, tempos[j] - tempos[i])
    w = jnp.where(i == j, 0.0, (bpm_c - tempos[i]) / denom)

    n = tables.ginga_qpos.shape[1]
    x = (phase % 1.0) * n
    k0 = jnp.floor(x).astype(jnp.int32) % n
    k1 = (k0 + 1) % n
    a = x - jnp.floor(x)

    def _interp_tempo(t_idx):
        q0, q1 = tables.ginga_qpos[t_idx, k0], tables.ginga_qpos[t_idx, k1]
        q = q0 + (q1 - q0) * a
        q = q.at[3:7].set(nlerp(q0[3:7], q1[3:7], a))
        v0, v1 = tables.ginga_qvel[t_idx, k0], tables.ginga_qvel[t_idx, k1]
        v = v0 + (v1 - v0) * a
        f0, f1 = tables.ginga_feet[t_idx, k0], tables.ginga_feet[t_idx, k1]
        f = f0 + (f1 - f0) * a
        return q, v, f

    qa, va, fa = _interp_tempo(i)
    qb, vb, fb = _interp_tempo(j)
    q = qa + (qb - qa) * w
    q = q.at[3:7].set(nlerp(qa[3:7], qb[3:7], w))
    v = va + (vb - va) * w
    f = fa + (fb - fa) * w
    return q, v, f


def sample_move(
    tables: RefTables, move_id: jax.Array, side_id: jax.Array, level: jax.Array, t: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    dur = tables.move_durations[move_id]
    fps = tables.move_fps[move_id]
    max_idx = tables.move_lengths[move_id] - 2
    x = jnp.clip(t, 0.0, dur) * fps
    k = jnp.minimum(jnp.floor(x).astype(jnp.int32), max_idx)
    a = x - k.astype(jnp.float32)

    q0 = tables.move_qpos[move_id, side_id, level, k]
    q1 = tables.move_qpos[move_id, side_id, level, k + 1]
    q = q0 + (q1 - q0) * a
    q = q.at[3:7].set(nlerp(q0[3:7], q1[3:7], a))
    v0 = tables.move_qvel[move_id, side_id, level, k]
    v1 = tables.move_qvel[move_id, side_id, level, k + 1]
    v = v0 + (v1 - v0) * a
    f0 = tables.move_feet[move_id, side_id, level, k]
    f1 = tables.move_feet[move_id, side_id, level, k + 1]
    f = f0 + (f1 - f0) * a
    m0 = tables.move_mask[move_id, side_id, level, k]
    m1 = tables.move_mask[move_id, side_id, level, k + 1]
    mask = jnp.minimum(m0, m1)
    return q, v, f, mask


def sample_floor(
    tables: RefTables, landing_id: jax.Array, side_id: jax.Array, t: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    dur = tables.floor_durations[landing_id, side_id]
    max_idx = tables.floor_lengths[landing_id, side_id] - 2
    x = jnp.clip(t, 0.0, dur) * tables.floor_fps
    k = jnp.minimum(jnp.floor(x).astype(jnp.int32), max_idx)
    a = x - k.astype(jnp.float32)

    q0 = tables.floor_qpos[landing_id, side_id, k]
    q1 = tables.floor_qpos[landing_id, side_id, k + 1]
    q = q0 + (q1 - q0) * a
    q = q.at[3:7].set(nlerp(q0[3:7], q1[3:7], a))
    v0 = tables.floor_qvel[landing_id, side_id, k]
    v1 = tables.floor_qvel[landing_id, side_id, k + 1]
    v = v0 + (v1 - v0) * a
    f0 = tables.floor_feet[landing_id, side_id, k]
    f1 = tables.floor_feet[landing_id, side_id, k + 1]
    f = f0 + (f1 - f0) * a
    m0 = tables.floor_mask[landing_id, side_id, k]
    m1 = tables.floor_mask[landing_id, side_id, k + 1]
    mask = jnp.minimum(m0, m1)
    return q, v, f, mask


# ---- Per-environment state --------------------------------------------------


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


class MjxGingaEnv:
    """Vectorised MJX Capoeira environment with identical obs/act/reward to GingaEnv."""

    def __init__(
        self,
        xml: str = "humanoid.xml",
        skill: str = "ginga",
        max_seconds: float = 40.0,
        impl: str = "jax",
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

        self.obs_size = 138 + len(EXTRA_MOVES)
        self.act_size = int(self.m.nu)
        self._dx_template = dx0

    # ---- Target & kinematics helpers ----------------------------------------

    def _sinking(self, q: jax.Array) -> jax.Array:
        dx = self._dx_template.replace(qpos=q)
        dx = mjx.kinematics(self.mx, dx)
        heights = dx.site_xpos[self.ground_sites, 1] - self.ground_clear
        return jnp.maximum(-jnp.min(heights), 0.0)

    def _target(
        self,
        phase: jax.Array,
        bpm: jax.Array,
        frame: jax.Array,
        move_id: jax.Array,
        side_id: jax.Array,
        kick_t: jax.Array,
        level: jax.Array,
        anchor: jax.Array,
        recover_id: jax.Array,
        recover_side: jax.Array,
        recover_t: jax.Array,
        recover_onset: jax.Array,
        recover_place: jax.Array,
        recover_sink: jax.Array,
        ahead: float,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        period = 240.0 / bpm
        g_phase = phase + jnp.where(move_id >= 0, 0.0, ahead / period)
        g_q, g_v, g_feet = sample_ginga(self.tables, g_phase, bpm)

        if self.skill in ("kicks", "esquiva", "all"):
            # Kick / esquiva target in local frame
            safe_move = jnp.maximum(move_id, 0)
            kt = kick_t + ahead
            k_q, k_v, k_feet, k_mask = sample_move(self.tables, safe_move, side_id, level, kt)
            k_q = k_q.at[0].add(anchor[0]).at[2].add(anchor[1])
            k_feet = k_feet + anchor[None, :]
            dur = self.tables.move_durations[safe_move]
            b = smoothstep(kt / BLEND_IN) * smoothstep((dur - kt) / BLEND_OUT)
            mk_q = g_q + (k_q - g_q) * b
            mk_q = mk_q.at[3:7].set(nlerp(g_q[3:7], k_q[3:7], b))
            mk_v = g_v + (k_v - g_v) * b
            mk_feet = g_feet + (k_feet - g_feet) * b
            mk_mask = jnp.where(b < 1.0, jnp.ones(2, jnp.float32), k_mask)

            loc_q = jnp.where(move_id >= 0, mk_q, g_q)
            loc_v = jnp.where(move_id >= 0, mk_v, g_v)
            loc_feet = jnp.where(move_id >= 0, mk_feet, g_feet)
            loc_mask = jnp.where(move_id >= 0, mk_mask, jnp.ones(2, jnp.float32))
        else:
            loc_q, loc_v, loc_feet, loc_mask = g_q, g_v, g_feet, jnp.ones(2, jnp.float32)

        world_q, world_v, world_feet = place_pose(frame, loc_q, loc_v, loc_feet)
        if self.skill != "recover":
            return world_q, world_v, world_feet, loc_mask

        # Recovery target (going down for t < FALL_TIME, getting up for t >= FALL_TIME)
        safe_rec = jnp.maximum(recover_id, 0)
        rt = recover_t + ahead
        up_q, up_v, up_feet, up_mask = sample_floor(self.tables, safe_rec, recover_side, jnp.maximum(rt - FALL_TIME, 0.0))
        up_q, up_v, up_feet = place_pose(recover_place, up_q, up_v, up_feet)

        land0_q, _, land0_feet, _ = sample_floor(self.tables, safe_rec, recover_side, jnp.array(0.0, jnp.float32))
        land0_q, _, land0_feet = place_pose(recover_place, land0_q, jnp.zeros(self.m.nv, jnp.float32), land0_feet)
        x = jnp.clip(rt / FALL_TIME, 0.0, 1.0)
        ease = 1.0 - (1.0 - x) ** 2
        drop = recover_onset[1] - land0_q[1]
        fallen = jnp.maximum(-recover_sink * rt + 0.5 * GRAVITY * rt * rt, 0.0)
        b_root = jnp.where(drop > 0.01, jnp.minimum(ease, fallen / jnp.maximum(drop, 0.01)), ease)
        b_vec = jnp.full(self.m.nq, b_root, jnp.float32).at[self.arm_qpos_ids].set(ease)
        down_q = recover_onset + (land0_q - recover_onset) * b_vec
        down_q = down_q.at[3:7].set(nlerp(recover_onset[3:7], land0_q[3:7], b_root))
        down_q = down_q.at[1].add(self._sinking(down_q))
        down_v = jnp.zeros(self.m.nv, jnp.float32)
        down_mask = jnp.zeros(2, jnp.float32)

        rec_q = jnp.where(rt >= FALL_TIME, up_q, down_q)
        rec_v = jnp.where(rt >= FALL_TIME, up_v, down_v)
        rec_feet = jnp.where(rt >= FALL_TIME, up_feet, land0_feet)
        rec_mask = jnp.where(rt >= FALL_TIME, up_mask, down_mask)

        is_rec = recover_id >= 0
        return (
            jnp.where(is_rec, rec_q, world_q),
            jnp.where(is_rec, rec_v, world_v),
            jnp.where(is_rec, rec_feet, world_feet),
            jnp.where(is_rec, rec_mask, loc_mask),
        )

    def _foot_points(self, data: mjx.Data) -> jax.Array:
        # (2, 2): midway between heel and toe sites in world (x, z)
        s = data.site_xpos[self.sole_sites]  # (2, 2, 3)
        return s.mean(axis=1)[:, jnp.array([0, 2])]

    def _contacts(self, data: mjx.Data, is_recovering: jax.Array) -> tuple[jax.Array, jax.Array]:
        dist = data._impl.contact.dist
        active = dist < 0.0
        foot_l = jnp.any(active & self.contact_is_foot_l).astype(jnp.float32)
        foot_r = jnp.any(active & self.contact_is_foot_r).astype(jnp.float32)
        feet = jnp.stack([foot_l, foot_r])
        bad_never = jnp.any(active & self.contact_is_never)
        bad_disallowed = jnp.any(active & self.contact_is_disallowed)
        bad = bad_never | ((~is_recovering) & bad_disallowed)
        return feet, bad

    def _local_effectors(self, data: mjx.Data) -> jax.Array:
        frame = yaw_matrix(yaw_of(data.qpos[3:7]))
        rel = data.xpos[self.effector_ids] - data.qpos[:3]
        return (rel @ frame.T).ravel()

    def _features(
        self,
        move_id: jax.Array,
        side_id: jax.Array,
        kick_t: jax.Array,
        recover_id: jax.Array,
        recover_side: jax.Array,
        recover_t: jax.Array,
    ) -> jax.Array:
        safe_move = jnp.maximum(move_id, 0)
        dur = self.tables.move_durations[safe_move]
        b = smoothstep(kick_t / BLEND_IN) * smoothstep((dur - kick_t) / BLEND_OUT)
        kick_feat = jnp.where(
            move_id >= 0,
            jnp.array(
                [
                    (move_id == 0).astype(jnp.float32),
                    (move_id == 1).astype(jnp.float32),
                    jnp.where(side_id == 1, 1.0, -1.0),
                    kick_t / dur,
                    b,
                ],
                jnp.float32,
            ),
            jnp.zeros(5, jnp.float32),
        )
        safe_rec = jnp.maximum(recover_id, 0)
        total = FALL_TIME + self.tables.floor_durations[safe_rec, recover_side]
        rec_feat = jnp.where(
            recover_id >= 0,
            jnp.array(
                [
                    1.0,
                    (recover_id == 0).astype(jnp.float32),
                    jnp.where(recover_side == 1, 1.0, -1.0),
                    recover_t / total,
                    (recover_id == 2).astype(jnp.float32),
                ],
                jnp.float32,
            ),
            jnp.zeros(5, jnp.float32),
        )
        esquiva_feat = jnp.array([((recover_id < 0) & (move_id == self.esquiva_move_id)).astype(jnp.float32)], jnp.float32)
        extra_feat = ((recover_id < 0) & (move_id == self.extra_move_ids)).astype(jnp.float32)
        return jnp.concatenate(
            [jnp.where(recover_id >= 0, jnp.zeros(5, jnp.float32), kick_feat), rec_feat, esquiva_feat, extra_feat]
        )

    def _observe(
        self,
        data: mjx.Data,
        phase: jax.Array,
        bpm: jax.Array,
        frame_place: jax.Array,
        move_id: jax.Array,
        side_id: jax.Array,
        kick_t: jax.Array,
        level: jax.Array,
        anchor: jax.Array,
        recover_id: jax.Array,
        recover_side: jax.Array,
        recover_t: jax.Array,
        recover_onset: jax.Array,
        recover_place: jax.Array,
        recover_sink: jax.Array,
    ) -> jax.Array:
        q, v = data.qpos, data.qvel
        frame = yaw_matrix(yaw_of(q[3:7]))
        orientation = (frame @ quat_matrix(q[3:7]))[:, :2].ravel()
        ang = 2.0 * jnp.pi * phase
        ref_next, _, feet_next, _ = self._target(
            phase,
            bpm,
            frame_place,
            move_id,
            side_id,
            kick_t,
            level,
            anchor,
            recover_id,
            recover_side,
            recover_t,
            recover_onset,
            recover_place,
            recover_sink,
            self.control_dt,
        )
        feet_contact, _ = self._contacts(data, recover_id >= 0)
        offset = feet_next - self._foot_points(data)  # (2, 2)
        o0 = (frame @ jnp.array([offset[0, 0], 0.0, offset[0, 1]]))[jnp.array([0, 2])]
        o1 = (frame @ jnp.array([offset[1, 0], 0.0, offset[1, 1]]))[jnp.array([0, 2])]
        feet_offset = jnp.concatenate([o0, o1])
        tempo_in = (bpm - 95.0) / 35.0
        feat = self._features(move_id, side_id, kick_t, recover_id, recover_side, recover_t)
        return jnp.concatenate(
            [
                jnp.array([jnp.sin(ang), jnp.cos(ang), q[1]], jnp.float32),
                orientation,
                q[self.joint_qpos],
                0.1 * v[self.joint_qvel],
                frame @ v[:3],
                v[3:6] * 0.2,
                self._local_effectors(data),
                feet_contact,
                ref_next[self.joint_qpos] - q[self.joint_qpos],
                frame @ (ref_next[:3] - q[:3]),
                feet_offset,
                jnp.array([tempo_in], jnp.float32),
                feat,
            ]
        )

    # ---- Reset & Step -------------------------------------------------------

    def reset(self, rng: jax.Array, curriculum: jax.Array, fall_level: jax.Array) -> EnvState:
        (
            rng,
            k_phase,
            k_bpm,
            k_lv1,
            k_lv2,
            k_mv1,
            k_mv2,
            k_side,
            k_gf,
            k_entry,
            k_half,
            k_kt,
            k_anc,
            k_nq,
            k_nv,
            k_rec,
            k_dr,
            k_rv,
        ) = jax.random.split(rng, 18)
        phase = jax.random.uniform(k_phase, (), minval=0.0, maxval=1.0)
        bpm = MIN_BPM + (MAX_BPM - MIN_BPM) * (jax.random.uniform(k_bpm, ()) ** TEMPO_SKEW)
        frame = jnp.zeros(3, jnp.float32)
        act_scale = ACTION_SCALE * jax.random.uniform(k_dr, (), minval=0.97, maxval=1.03)

        # Level selection for kicks / esquiva
        lv_draw = jax.random.uniform(k_lv1, ())
        rand_lv = jax.random.randint(k_lv2, (), 0, curriculum + 1)
        level = jnp.where(lv_draw < 0.7, curriculum, rand_lv)
        if self.skill not in ("kicks", "esquiva"):
            level = jnp.array(3, jnp.int32)

        move_id = jnp.array(-1, jnp.int32)
        side_id = jax.random.randint(k_side, (), 0, 2)
        kick_t = jnp.array(0.0, jnp.float32)
        anchor = jnp.zeros(2, jnp.float32)
        pending_move = jnp.array(-1, jnp.int32)
        pending_side = side_id
        entry_centre = jnp.where(side_id == 1, 0.0, 0.5)
        entry = (entry_centre + jax.random.uniform(k_entry, (), minval=-0.1, maxval=0.1)) % 1.0
        pending_entry = entry
        from_ginga = jnp.array(False)
        noisy = jnp.array(False)

        if self.skill in ("kicks", "esquiva", "all"):
            chosen_move = self.skill_moves[jax.random.randint(k_mv1, (), 0, len(self.skill_moves))]
            do_move = True if self.skill in ("kicks", "esquiva") else (jax.random.uniform(k_mv2, ()) < 0.3)
            wait_ginga = (self.skill in ("kicks", "esquiva")) & (jax.random.uniform(k_gf, ()) < GINGA_FIRST)
            pending_move = jnp.where(do_move & wait_ginga, chosen_move, -1)
            move_id = jnp.where(do_move & (~wait_ginga), chosen_move, -1)
            from_ginga = do_move & wait_ginga

            start_zero = (self.skill in ("kicks", "esquiva")) & (jax.random.uniform(k_half, ()) < 0.5)
            dur = self.tables.move_durations[jnp.maximum(move_id, 0)]
            mid_t = jax.random.uniform(k_kt, (), minval=0.0, maxval=dur * 0.8)
            kick_t = jnp.where(move_id >= 0, jnp.where(start_zero, 0.0, mid_t), 0.0)
            noisy = (move_id >= 0) & start_zero
            from_ginga = from_ginga | noisy
            rand_anc = jax.random.uniform(k_anc, (2,), minval=-ANCHOR_LIMIT, maxval=ANCHOR_LIMIT) * 0.5
            anchor = jnp.where(noisy, rand_anc, jnp.zeros(2, jnp.float32))
            phase = jnp.where(move_id >= 0, entry, phase)

        # Recovery reset (skill == "recover")
        recover_id = jnp.array(-1, jnp.int32)
        recover_side = side_id
        recover_t = jnp.array(0.0, jnp.float32)
        recover_onset = jnp.zeros(self.m.nq, jnp.float32)
        recover_place = jnp.zeros(3, jnp.float32)
        recover_sink = jnp.array(0.0, jnp.float32)
        from_fall = jnp.array(False)

        if self.skill == "recover":
            k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8, k_r9, k_r10 = jax.random.split(k_rec, 10)
            draw = jax.random.uniform(k_r1, ())
            use_pool = draw < 0.5
            pool_idx = jax.random.randint(k_r2, (), 0, len(self.tables.falls_qpos))
            speeds = jnp.array(FALL_SPEEDS, jnp.float32)
            flv = jnp.where(
                jax.random.uniform(k_r3, ()) < 0.7,
                fall_level,
                jax.random.randint(k_r4, (), 0, fall_level + 1),
            )
            pq = self.tables.falls_qpos[pool_idx]
            pv = self.tables.falls_qvel[pool_idx] * speeds[flv]
            p_phase = self.tables.falls_phase[pool_idx]
            p_bpm = self.tables.falls_bpm[pool_idx]
            p_move = self.tables.falls_move[pool_idx, flv]
            p_side = self.tables.falls_side[pool_idx, flv]
            p_place = self.tables.falls_place[pool_idx, flv]

            # Non-pool: on floor (20%) or partway up (30%)
            f_move = jax.random.randint(k_r5, (), 0, 3)
            f_side = jax.random.randint(k_r6, (), 0, 2)
            f_dur = self.tables.floor_durations[f_move, f_side]
            f_t = FALL_TIME + jnp.where(draw < 0.7, 0.0, jax.random.uniform(k_r7, (), minval=0.0, maxval=0.85 * f_dur))
            f_place = jnp.array(
                [
                    jax.random.uniform(k_r8, (), minval=-1.0, maxval=1.0),
                    jax.random.uniform(k_r9, (), minval=-1.0, maxval=1.0),
                    jax.random.uniform(k_r10, (), minval=-jnp.pi, maxval=jnp.pi),
                ],
                jnp.float32,
            )
            f_onset, _, _ = place_pose(
                f_place, self.tables.floor_qpos[f_move, f_side, 0], jnp.zeros(self.m.nv), jnp.zeros((2, 2))
            )

            recover_id = jnp.where(use_pool, p_move, f_move)
            recover_side = jnp.where(use_pool, p_side, f_side)
            recover_t = jnp.where(use_pool, 0.0, f_t)
            recover_onset = jnp.where(use_pool, pq, f_onset)
            recover_place = jnp.where(use_pool, p_place, f_place)
            recover_sink = jnp.where(use_pool, pv[1], 0.0)
            phase = jnp.where(use_pool, p_phase, phase)
            bpm = jnp.where(use_pool, p_bpm, bpm)
            from_fall = use_pool

        q, v, _, _ = self._target(
            phase,
            bpm,
            frame,
            move_id,
            side_id,
            kick_t,
            level,
            anchor,
            recover_id,
            recover_side,
            recover_t,
            recover_onset,
            recover_place,
            recover_sink,
            0.0,
        )
        if self.skill == "recover":
            q = jnp.where(from_fall, pq, q)
            v = jnp.where(from_fall, pv, v)

        q_noise = jax.random.normal(k_nq, (len(self.joint_qpos),)) * 0.03
        v_noise = jax.random.normal(k_nv, (len(self.joint_qvel),)) * 0.2
        rv_noise = jax.random.normal(k_rv, (3,)) * 0.03
        q = q.at[self.joint_qpos].add(jnp.where(noisy, q_noise, 0.0))
        v = v.at[self.joint_qvel].add(jnp.where(noisy, v_noise, 0.0))
        v = v.at[:3].add(jnp.where(noisy, rv_noise, 0.0))

        dx = self._dx_template.replace(qpos=q, qvel=v, ctrl=jnp.zeros(self.act_size, jnp.float32))
        dx = mjx.forward(self.mx, dx)
        # Lift out of floor so rolled foot corners don't launch the body (except on pool falls, matching GingaEnv)
        floor_dists = jnp.where(self.contact_is_floor, dx._impl.contact.dist, 0.0)
        deepest = jnp.min(floor_dists)
        lift = jnp.maximum(-deepest, 0.0) + 0.002
        q = q.at[1].add(jnp.where(from_fall, 0.0, lift))
        dx = dx.replace(qpos=q)
        dx = mjx.forward(self.mx, dx)

        obs = self._observe(
            dx,
            phase,
            bpm,
            frame,
            move_id,
            side_id,
            kick_t,
            level,
            anchor,
            recover_id,
            recover_side,
            recover_t,
            recover_onset,
            recover_place,
            recover_sink,
        )
        return EnvState(
            data=dx,
            obs=obs,
            rng=rng,
            phase=phase,
            bpm=bpm,
            frame=frame,
            move_id=move_id,
            side_id=side_id,
            kick_t=kick_t,
            level=level,
            anchor=anchor,
            pending_move=pending_move,
            pending_side=pending_side,
            pending_entry=pending_entry,
            recover_id=recover_id,
            recover_side=recover_side,
            recover_t=recover_t,
            recover_onset=recover_onset,
            recover_place=recover_place,
            recover_sink=recover_sink,
            steps=jnp.array(0, jnp.int32),
            kicked=move_id >= 0,
            from_ginga=from_ginga,
            after_kick=jnp.array(-1.0, jnp.float32),
            after_recover=jnp.array(-1.0, jnp.float32),
            stuck_t=jnp.array(0.0, jnp.float32),
            from_fall=from_fall,
            landing_seen=jnp.where(recover_id >= 0, recover_id + 1, 0),
            act_scale=act_scale,
        )

    def step(
        self, state: EnvState, action: jax.Array, curriculum: jax.Array, fall_level: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        # 1. PD target one control step ahead + policy offset
        target_next, _, _, _ = self._target(
            state.phase,
            state.bpm,
            state.frame,
            state.move_id,
            state.side_id,
            state.kick_t,
            state.level,
            state.anchor,
            state.recover_id,
            state.recover_side,
            state.recover_t,
            state.recover_onset,
            state.recover_place,
            state.recover_sink,
            self.control_dt,
        )
        ctrl = target_next[self.joint_qpos] + state.act_scale * jnp.clip(action, -1.0, 1.0)
        ctrl = jnp.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
        dx = state.data.replace(ctrl=ctrl)

        # 2. Step physics FRAME_SKIP times
        dx = jax.lax.fori_loop(0, FRAME_SKIP, lambda _, d: mjx.step(self.mx, d), dx)

        # 3. Advance choreography
        was_recovering = state.recover_id >= 0
        was_falling = was_recovering & (state.recover_t < FALL_TIME)
        q_ref_before, _, _, _ = self._target(
            state.phase,
            state.bpm,
            state.frame,
            state.move_id,
            state.side_id,
            state.kick_t,
            state.level,
            state.anchor,
            state.recover_id,
            state.recover_side,
            state.recover_t,
            state.recover_onset,
            state.recover_place,
            state.recover_sink,
            0.0,
        )
        lag = jnp.linalg.norm(dx.qpos[:3] - q_ref_before[:3])
        pace = jnp.where(
            was_recovering & (state.recover_t >= FALL_TIME),
            jnp.clip(1.0 - (lag - WAIT_FROM) / WAIT_SPAN, MIN_PACE, 1.0),
            1.0,
        )
        dt = self.control_dt * pace

        # Advance recovery if active
        safe_rec = jnp.maximum(state.recover_id, 0)
        rec_total = FALL_TIME + self.tables.floor_durations[safe_rec, state.recover_side]
        new_rec_t = state.recover_t + dt
        rec_done = was_recovering & (new_rec_t >= rec_total)
        just_landed = was_falling & (new_rec_t >= FALL_TIME)
        land0 = self.tables.floor_qpos[safe_rec, state.recover_side, 0]
        landed_xy = jnp.stack([dx.qpos[0], dx.qpos[2]]) - place_xz(
            jnp.array([0.0, 0.0, state.recover_place[2]]), jnp.stack([land0[0], land0[2]])
        )
        recover_place = jnp.where(just_landed, state.recover_place.at[:2].set(landed_xy), state.recover_place)
        recover_id = jnp.where(rec_done, -1, state.recover_id)

        # Advance kick / esquiva if active
        was_kicking = ( ~was_recovering ) & (state.move_id >= 0)
        safe_move = jnp.maximum(state.move_id, 0)
        move_dur = self.tables.move_durations[safe_move]
        new_kick_t = state.kick_t + dt
        kick_done = was_kicking & (new_kick_t >= move_dur)

        # Reanchor triangle when kick/esquiva or floor recovery finishes
        _, _, g_feet_local = sample_ginga(self.tables, state.phase, state.bpm)
        local_placed = place_xz(state.frame, g_feet_local)
        feet_now = self._foot_points(dx)
        reanchored_frame = state.frame.at[:2].add(feet_now.mean(axis=0) - local_placed.mean(axis=0))
        rec_phase = self.tables.floor_phases[safe_rec, state.recover_side]
        _, _, r_feet_local = sample_ginga(self.tables, rec_phase, state.bpm)
        rec_placed = place_xz(recover_place, r_feet_local)
        rec_reanchored = recover_place.at[:2].add(feet_now.mean(axis=0) - rec_placed.mean(axis=0))
        frame = jnp.where(
            rec_done,
            rec_reanchored,
            jnp.where(kick_done, reanchored_frame, state.frame),
        )
        move_id = jnp.where(kick_done, -1, state.move_id)
        kick_t = jnp.where(was_kicking & (~kick_done), new_kick_t, state.kick_t)

        # Advance plain ginga & check pending move entry
        in_ginga = (~was_recovering) & (~was_kicking)
        phase_step = dt / (240.0 / state.bpm)
        has_pending = in_ginga & (state.pending_move >= 0)
        enter_now = has_pending & (((state.pending_entry - state.phase) % 1.0) <= phase_step)

        # Compute anchor when entering scheduled move
        pend_side = state.pending_side
        support_idx = jnp.where(pend_side == 1, 0, 1)
        _, _, planned_feet, _ = sample_move(
            self.tables, jnp.maximum(state.pending_move, 0), pend_side, state.level, jnp.array(0.0, jnp.float32)
        )
        raw_anc = unplace_xz(frame, feet_now[support_idx]) - planned_feet[support_idx]
        anc_norm = jnp.linalg.norm(raw_anc)
        entered_anc = jnp.where(anc_norm > 0.0, raw_anc * jnp.minimum(1.0, ANCHOR_LIMIT / (anc_norm + 1e-8)), raw_anc)

        phase = jnp.where(
            rec_done,
            self.tables.floor_phases[safe_rec, state.recover_side],
            jnp.where(
                enter_now,
                state.pending_entry,
                jnp.where(in_ginga, (state.phase + phase_step) % 1.0, state.phase),
            ),
        )
        move_id = jnp.where(enter_now, state.pending_move, move_id)
        side_id = jnp.where(enter_now, state.pending_side, state.side_id)
        kick_t = jnp.where(enter_now, 0.0, kick_t)
        anchor = jnp.where(enter_now, entered_anc, state.anchor)
        pending_move = jnp.where(enter_now, -1, state.pending_move)

        # Random tempo & random move scheduling during episode
        rng, k_t1, k_t2, k_m1, k_m2, k_m3, k_m4, k_reset = jax.random.split(state.rng, 8)
        change_tempo = jax.random.uniform(k_t1, ()) < (self.control_dt / TEMPO_CHANGE_EVERY)
        new_bpm = MIN_BPM + (MAX_BPM - MIN_BPM) * (jax.random.uniform(k_t2, ()) ** TEMPO_SKEW)
        bpm = jnp.where(change_tempo, new_bpm, state.bpm)

        busy = (move_id >= 0) | (pending_move >= 0) | (recover_id >= 0)
        if self.skill == "all":
            ask_move = (~busy) & (jax.random.uniform(k_m1, ()) < (self.control_dt / KICK_EVERY))
            s_new = jax.random.randint(k_m2, (), 0, 2)
            e_centre = jnp.where(s_new == 1, 0.0, 0.5)
            e_new = (e_centre + jax.random.uniform(k_m3, (), minval=-0.1, maxval=0.1)) % 1.0
            m_new = self.skill_moves[jax.random.randint(k_m4, (), 0, len(self.skill_moves))]
            pending_move = jnp.where(ask_move, m_new, pending_move)
            pending_side = jnp.where(ask_move, s_new, state.pending_side)
            pending_entry = jnp.where(ask_move, e_new, state.pending_entry)
        else:
            pending_side = state.pending_side
            pending_entry = state.pending_entry

        # 4. Reward & kinematics of current reference pose
        q_ref, v_ref, feet_ref, feet_mask = self._target(
            phase,
            bpm,
            frame,
            move_id,
            side_id,
            kick_t,
            state.level,
            anchor,
            recover_id,
             state.recover_side,
            new_rec_t,
            state.recover_onset,
            recover_place,
            state.recover_sink,
            0.0,
        )
        ref_dx = self._dx_template.replace(qpos=q_ref)
        ref_dx = mjx.kinematics(self.mx, ref_dx)

        q, v = dx.qpos, dx.qvel
        joint_err = jnp.sum((q[self.joint_qpos] - q_ref[self.joint_qpos]) ** 2)
        root_rot_err = quat_angle(q[3:7], q_ref[3:7]) ** 2
        vel_err = jnp.sum((v[self.joint_qvel] - v_ref[self.joint_qvel]) ** 2)
        ee_err = jnp.sum((self._local_effectors(dx) - self._local_effectors(ref_dx)) ** 2)
        is_falling_now = (recover_id >= 0) & (new_rec_t < FALL_TIME)
        root_err = jnp.where(is_falling_now, (q[1] - q_ref[1]) ** 2, jnp.sum((q[:3] - q_ref[:3]) ** 2))
        feet_err = jnp.sum(feet_mask[:, None] * (feet_now - feet_ref) ** 2)

        r_pose = jnp.exp(-2.0 * (joint_err + root_rot_err))
        r_vel = jnp.exp(-0.02 * vel_err)
        r_ee = jnp.exp(-40.0 * ee_err / len(EFFECTORS))
        r_root = jnp.exp(-10.0 * root_err)
        r_feet = jnp.exp(-20.0 * feet_err)
        base_reward = 0.4 * r_pose + 0.1 * r_vel + 0.15 * r_ee + 0.15 * r_root + 0.2 * r_feet

        # Strike & pivot terms when playing a kick (move_id in 0..num_kicks-1)
        is_kick_move = (move_id >= 0) & (move_id < self.num_kicks)
        kick_foot_body = jnp.where(side_id == 1, self.effector_ids[1], self.effector_ids[0])
        reach_err = jnp.sum(((dx.xpos[kick_foot_body] - q[:3]) - (ref_dx.xpos[kick_foot_body] - q_ref[:3])) ** 2)
        r_strike = jnp.where(is_kick_move, jnp.exp(-STRIKE_SCALE * reach_err), 0.0)

        support_side = jnp.where(side_id == 1, 0, 1)
        heel_id, toe_id = self.sole_sites[support_side, 0], self.sole_sites[support_side, 1]
        d_vec = dx.site_xpos[toe_id] - dx.site_xpos[heel_id]
        r_vec = ref_dx.site_xpos[toe_id] - ref_dx.site_xpos[heel_id]
        turn_err = (jnp.arctan2(d_vec[0], d_vec[2]) - jnp.arctan2(r_vec[0], r_vec[2]) + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        r_pivot = jnp.where(is_kick_move, jnp.exp(-PIVOT_SCALE * (turn_err**2)), 0.0)

        reward = jnp.where(
            is_kick_move,
            (1.0 - STRIKE_WEIGHT - PIVOT_WEIGHT) * base_reward + STRIKE_WEIGHT * r_strike + PIVOT_WEIGHT * r_pivot,
            base_reward,
        )

        # Recovery soft-landing & head-above-hips reward
        in_soft_window = (recover_id >= 0) & (new_rec_t < FALL_TIME + 0.3)
        head_y = dx.xpos[self.head_body_id, 1]
        hips_y = dx.xpos[self.hips_body_id, 1]
        r_head = jnp.exp(-10.0 * (jnp.maximum(hips_y + HEAD_ABOVE - head_y, 0.0) ** 2))
        soft_ids = jnp.array([self.hips_body_id, self.chest_body_id, self.head_body_id], jnp.int32)
        r_off = dx.xipos[soft_ids] - dx.subtree_com[1]
        vy_soft = dx.cvel[soft_ids, 4] + dx.cvel[soft_ids, 2] * r_off[:, 0] - dx.cvel[soft_ids, 0] * r_off[:, 2]
        ang_drop = 0.05 * jnp.sum(v[3:6] ** 2)
        r_soft = jnp.exp(-0.5 * (jnp.sum(jnp.minimum(vy_soft, 0.0) ** 2) + ang_drop))
        reward = jnp.where(in_soft_window, 0.75 * reward + 0.1 * r_soft + 0.15 * r_head, reward)

        # 5. Termination & truncation
        is_rec_now = recover_id >= 0
        _, bad_contact = self._contacts(dx, is_rec_now)
        drift = jnp.linalg.norm(jnp.stack([q[0] - q_ref[0], q[2] - q_ref[2]]))

        ref_head_y = ref_dx.xpos[self.head_body_id, 1]
        low_normal = (q[1] < jnp.minimum(LOW_HIPS, q_ref[1] - LOW_BELOW)) | (
            head_y < jnp.minimum(LOW_HEAD, ref_head_y - LOW_BELOW)
        )
        low = (~is_rec_now) & low_normal

        landed = is_rec_now & (new_rec_t >= FALL_TIME)
        drifted = jnp.where(is_rec_now, landed & (drift > LOST), drift > 0.5)
        new_stuck_t = jnp.where(landed & (pace <= MIN_PACE + 1e-5), state.stuck_t + self.control_dt, 0.0)
        stuck = is_rec_now & (new_stuck_t >= STUCK)
        dive = is_rec_now & (new_rec_t < FALL_TIME + DIVE_UNTIL) & (head_y < ref_head_y - DIVE)

        terminated = bad_contact | low | drifted | stuck | dive
        reward = jnp.where(terminated, 0.0, reward)

        steps = state.steps + 1
        kicked = state.kicked | (move_id >= 0)
        busy_now = (move_id >= 0) | (pending_move >= 0) | is_rec_now

        after_kick = jnp.where(
            (self.skill in ("kicks", "esquiva")) & (~busy_now),
            jnp.maximum(state.after_kick, 0.0) + self.control_dt,
            state.after_kick,
        )
        after_recover = jnp.where(
            rec_done,
            0.0,
            jnp.where(state.after_recover >= 0.0, state.after_recover + self.control_dt, -1.0),
        )

        truncated = steps >= self.max_steps
        if self.skill in ("kicks", "esquiva"):
            truncated = truncated | (after_kick >= MOVE_EPISODE_TAIL)
        if self.skill == "recover":
            truncated = truncated | (after_recover >= RECOVER_TAIL)

        done = terminated | truncated
        obs_next = self._observe(
            dx,
            phase,
            bpm,
            frame,
            move_id,
            side_id,
            kick_t,
            state.level,
            anchor,
            recover_id,
            state.recover_side,
            new_rec_t,
            state.recover_onset,
            recover_place,
            state.recover_sink,
        )

        next_state = EnvState(
            data=dx,
            obs=obs_next,
            rng=rng,
            phase=phase,
            bpm=bpm,
            frame=frame,
            move_id=move_id,
            side_id=side_id,
            kick_t=kick_t,
            level=state.level,
            anchor=anchor,
            pending_move=pending_move,
            pending_side=pending_side,
            pending_entry=pending_entry,
            recover_id=recover_id,
            recover_side=state.recover_side,
            recover_t=new_rec_t,
            recover_onset=state.recover_onset,
            recover_place=recover_place,
            recover_sink=state.recover_sink,
            steps=steps,
            kicked=kicked,
            from_ginga=state.from_ginga,
            after_kick=after_kick,
            after_recover=after_recover,
            stuck_t=new_stuck_t,
            from_fall=state.from_fall,
            landing_seen=jnp.where(is_rec_now, recover_id + 1, state.landing_seen),
            act_scale=state.act_scale,
        )

        # Auto-reset on done (same-step autoreset, keeping final_obs for bootstrap)
        reset_state = self.reset(k_reset, curriculum, fall_level)
        final_state = jax.tree.map(lambda r, n: jnp.where(done, r, n), reset_state, next_state)

        end_code = jnp.where(
            bad_contact,
            1,
            jnp.where(low, 2, jnp.where(drifted, 3, jnp.where(stuck, 4, jnp.where(dive, 5, 0)))),
        )
        # Count kick curriculum attempt when the kick itself finishes (before the 3s ginga tail), matching GingaEnv
        before_tail = state.after_kick < 0.0
        kick_finished = before_tail & (~busy_now) & kicked
        kick_failed_early = before_tail & terminated & (busy_now | kicked)
        elig = state.from_ginga & (state.level == curriculum)
        kick_attempt = elig & (kick_finished | kick_failed_early)
        kick_ok = elig & kick_finished & (~terminated)

        info = {
            "final_obs": obs_next,
            "episode_steps": steps,
            "pose": r_pose,
            "vel": r_vel,
            "ee": r_ee,
            "root": r_root,
            "feet": r_feet,
            "kicking": (move_id >= 0).astype(jnp.float32),
            "strike": r_strike,
            "pivot": r_pivot,
            "down": is_rec_now.astype(jnp.float32),
            "up": (after_recover >= 0.0).astype(jnp.float32),
            "end": end_code,
            "landing": state.landing_seen,
            "got_up": (after_recover >= 0.0).astype(jnp.float32),
            "from_fall": state.from_fall,
            "kick_attempt": kick_attempt,
            "kick_ok": kick_ok,
        }
        return final_state, reward, terminated, truncated, info
