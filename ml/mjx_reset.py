"""MjxGingaEnv.reset: a ginga, a kick or esquiva (or its entry), a recorded fall or a way up, drawn from the rng."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx

from env_consts import ACTION_SCALE, ANCHOR_LIMIT, FALL_SPEEDS, FALL_TIME, GINGA_FIRST, MAX_BPM, MIN_BPM, TEMPO_SKEW
from mjx_math import place_pose
from mjx_state import EnvState


class MjxReset:
    """Episode start of MjxGingaEnv."""

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
