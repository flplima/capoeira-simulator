"""MjxGingaEnv.step: physics, choreography, reward, termination and the same-step auto-reset."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx

from env_consts import (
    ANCHOR_LIMIT,
    DIVE,
    DIVE_UNTIL,
    EFFECTORS,
    FALL_TIME,
    FRAME_SKIP,
    HEAD_ABOVE,
    KICK_EVERY,
    KICK_SETTLE,
    LOST,
    LOW_BELOW,
    LOW_HEAD,
    LOW_HIPS,
    MAX_BPM,
    MIN_BPM,
    MIN_PACE,
    MOVE_EPISODE_TAIL,
    PIVOT_WEIGHT,
    RECOVER_TAIL,
    STRIKE_WEIGHT,
    STUCK,
    TEMPO_CHANGE_EVERY,
    TEMPO_SKEW,
    WAIT_FROM,
    WAIT_SPAN,
)
from mjx_math import place_xz, quat_angle, unplace_xz
from mjx_state import EnvState
from mjx_tables import sample_ginga, sample_move


class MjxStep:
    """Control step of MjxGingaEnv."""

    def step(
        self, state: EnvState, action: jax.Array, curriculum: jax.Array, fall_level: jax.Array, sigma: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        """`sigma`: the tracking tolerances (env_consts.TERMS), one row for the ginga and one for the moves."""
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
        in_move = move_id >= 0
        is_kick_move = in_move & (move_id < self.num_kicks)
        joint_err = jnp.sum((q[self.joint_qpos] - q_ref[self.joint_qpos]) ** 2) + quat_angle(q[3:7], q_ref[3:7]) ** 2
        vel_err = jnp.sum((v[self.joint_qvel] - v_ref[self.joint_qvel]) ** 2)
        ee_err = jnp.sum((self._local_effectors(dx) - self._local_effectors(ref_dx)) ** 2) / len(EFFECTORS)
        is_falling_now = (recover_id >= 0) & (new_rec_t < FALL_TIME)
        root_err = jnp.where(is_falling_now, (q[1] - q_ref[1]) ** 2, jnp.sum((q[:3] - q_ref[:3]) ** 2))
        feet_err = jnp.sum(feet_mask[:, None] * (feet_now - feet_ref) ** 2)

        # A kick's own terms (move_id in 0..num_kicks-1): the kicking foot from the hips as the room sees it
        # (strike), and the support foot's heading, heel height and ball position (pivot)
        kick_foot_body = jnp.where(side_id == 1, self.effector_ids[1], self.effector_ids[0])
        reach_err = jnp.sum(((dx.xpos[kick_foot_body] - q[:3]) - (ref_dx.xpos[kick_foot_body] - q_ref[:3])) ** 2)
        support_side = jnp.where(side_id == 1, 0, 1)
        heel_id, toe_id = self.sole_sites[support_side, 0], self.sole_sites[support_side, 1]
        d_vec = dx.site_xpos[toe_id] - dx.site_xpos[heel_id]
        r_vec = ref_dx.site_xpos[toe_id] - ref_dx.site_xpos[heel_id]
        turn_err = ((jnp.arctan2(d_vec[0], d_vec[2]) - jnp.arctan2(r_vec[0], r_vec[2]) + jnp.pi) % (2.0 * jnp.pi) - jnp.pi) ** 2
        heel_err = (dx.site_xpos[heel_id, 1] - ref_dx.site_xpos[heel_id, 1]) ** 2
        ball_err = jnp.sum((dx.site_xpos[toe_id] - ref_dx.site_xpos[toe_id])[jnp.array([0, 2])] ** 2)

        # Each term's squared error in TERMS order, scored against the tolerances of the ginga or of this move on this leg
        err = jnp.stack([joint_err, vel_err, ee_err, root_err, feet_err, reach_err, turn_err, heel_err, ball_err])
        row = jnp.where(in_move, 1 + 2 * move_id + side_id, 0)
        scores = jnp.exp(-err / sigma[row])
        r_pose, r_vel, r_ee, r_root, r_feet = scores[:5]
        base_reward = 0.4 * r_pose + 0.1 * r_vel + 0.15 * r_ee + 0.15 * r_root + 0.2 * r_feet
        r_strike = jnp.where(is_kick_move, scores[5], 0.0)
        r_pivot = jnp.where(is_kick_move, jnp.prod(scores[6:]), 0.0)

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
        # A kick counts once the body has been back in the ginga KICK_SETTLE s, or as failed if it fell before,
        # matching GingaEnv
        unsettled = state.after_kick < KICK_SETTLE
        kick_finished = unsettled & (after_kick >= KICK_SETTLE) & kicked
        kick_failed_early = unsettled & terminated & (busy_now | kicked)
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
            "kicking": in_move.astype(jnp.float32),
            "kick": is_kick_move.astype(jnp.float32),
            "row": row,
            "err": err,
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
