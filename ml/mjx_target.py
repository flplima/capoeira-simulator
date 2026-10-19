"""MjxGingaEnv's reference target and observation: what the body should be doing, and what the policy is told."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx

from env_consts import BLEND_IN, BLEND_OUT, FALL_TIME, GRAVITY
from mjx_math import nlerp, place_pose, quat_matrix, smoothstep, yaw_matrix, yaw_of
from mjx_tables import sample_floor, sample_ginga, sample_move


class MjxTarget:
    """Reference target, contacts and observation of MjxGingaEnv."""

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
