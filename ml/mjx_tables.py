"""The reference clips as device tables, and the pure JAX functions that sample them."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from env_clips import Floor, Kicks, Reference
from env_consts import BENT, FALL_SPEEDS, LANDINGS, MOVES
from mjx_math import nlerp


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
