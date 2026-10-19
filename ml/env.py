"""Gymnasium environment: a physically simulated capoeirista imitating the ginga.

DeepMimic-style motion imitation (Peng et al., 2018). The policy runs at about
33 Hz and outputs offsets to the reference joint angles; MuJoCo's PD position
actuators turn those targets into joint torques, which are bounded, so the
body has to balance with real forces, contacts and gravity. The reward says
how closely it tracks the reference ginga — joint angles, joint velocities,
hand and foot positions, the root, and where on the floor triangle each foot
is — and an episode ends when it falls or wanders off the triangle. The
triangle is fixed in the world, so a policy that scores well stays in place
and can ginga for as long as it runs.
"""

import gymnasium as gym
import mujoco
import numpy as np

from env_choreo import Choreography
from env_clips import Floor, Kicks, Reference
from env_consts import (
    ACTION_SCALE,
    ALLOWED_CONTACT,
    DIVE,
    DIVE_UNTIL,
    DODGES,
    EFFECTORS,
    ENDS,
    ENTRY_CENTRE,
    ENTRY_WINDOW,
    FALL_SPEEDS,
    FALL_TIME,
    FRAME_SKIP,
    GROUND_SITES,
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
    NEVER_CONTACT,
    RECOVER_TAIL,
    SKILL_MOVES,
    STRONG_ARMS,
    STUCK,
    TEMPO_CHANGE_EVERY,
    TEMPO_SKEW,
    TROUBLE_DROP,
    TROUBLE_TILT,
    WAIT_FROM,
    WAIT_SPAN,
    tempo_input,
)
from env_math import quat_matrix, yaw_matrix, yaw_of
from env_reset import EpisodeReset
from env_reward import RewardTerms


class GingaEnv(EpisodeReset, RewardTerms, gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        xml: str = "humanoid.xml",
        reference: str = "data/ginga_ref.npz",
        kicks: str = "data/kicks.npz",
        esquiva: str = "data/esquiva.npz",
        floor: str = "data/floor.npz",
        falls: str = "data/falls.npz",
        max_seconds: float = 20.0,
        seed: int | None = None,
        skill: str = "all",
    ):
        """skill: what episodes practise — "ginga" (no kicks), "kicks" (one kick per
        episode, from the ginga and back to it), "esquiva" (the same with an esquiva lateral), "recover" (falling from a lost balance
        — see falls.py — or starting partway, getting up and gingando on), or "all"
        (kicks asked for at random)."""
        self.skill = skill
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.ref = Reference(reference, self.m)
        self.choreo = Choreography(self.ref, Kicks([kicks, esquiva], self.m), Floor(floor, self.m))
        self.choreo.feet_now = self._foot_points
        self.ground_data = mujoco.MjData(self.m)
        self.ground_sites = [self.m.site(n).id for n in GROUND_SITES]
        self.ground_clear = np.array([KNEE_CLEAR if n.startswith("knee") else 0.0 for n in GROUND_SITES])
        self.choreo.sinking = self._sinking
        self.choreo.arms = [
            int(self.m.jnt_qposadr[j]) for j in range(self.m.njnt) if self.m.joint(j).name.startswith(("shoulder", "elbow"))
        ]
        # Kick-size curriculum (skill "kicks"): each environment starts with the
        # smallest kick and moves up a size once it finishes most of its kicks.
        self.curriculum = 0
        self.success = 0.0
        self.attempts = 0
        self.vary_moves = True
        self.ref_data = mujoco.MjData(self.m)
        self.control_dt = self.m.opt.timestep * FRAME_SKIP
        self.max_steps = int(max_seconds / self.control_dt)
        self.joint_qpos = np.arange(7, self.m.nq)
        self.joint_qvel = np.arange(6, self.m.nv)
        self.ctrl_lo = self.m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.m.actuator_ctrlrange[:, 1].copy()
        self.effector_ids = [self.m.body(n).id for n in EFFECTORS]
        self.allowed_geoms = {self.m.geom(n).id for n in ALLOWED_CONTACT}
        self.never_geoms = {self.m.geom(n).id for n in NEVER_CONTACT}
        self.soft_bodies = [self.m.body(n).id for n in ("hips", "chest", "head")]
        # Per arm motor: id, relaxed (kp, kv, force), braced (kp, kv, force) — see STRONG_ARMS.
        self.arms = []
        for i in range(self.m.nu):
            name = self.m.actuator(i).name
            joint = "shoulder" if name.startswith("shoulder") else "elbow" if name.startswith("elbow") else None
            if joint:
                kp, kv, force = self.m.actuator_gainprm[i, 0], self.m.actuator_biasprm[i, 2], self.m.actuator_forcerange[i, 1]
                strong_kp, strong_force = STRONG_ARMS[joint]
                self.arms.append((i, (kp, kv, force), (strong_kp, kv * np.sqrt(strong_kp / kp), strong_force)))
        self.braced = False
        self.fall_level, self.fall_success, self.fall_attempts = 0, 0.0, 0
        self.pool_level: int | None = None  # this episode's fall speed level, when it started from a recorded fall
        self.falls = None
        if skill == "recover":
            self.falls = dict(np.load(falls))
        self.floor = self.m.geom("floor").id
        self.foot_geoms = [self.m.geom("foot_l").id, self.m.geom("foot_r").id]
        self.sole_sites = [[self.m.site(f"heel_{s}").id, self.m.site(f"toe_{s}").id] for s in "lr"]
        self.t = 0.0
        self.vary_tempo = True
        self.after_recover = None
        self.landing: str | None = None
        self.push_force, self.push_left = np.zeros(3), 0
        nu = self.m.nu
        self.action_space = gym.spaces.Box(-1.0, 1.0, (nu,), np.float32)
        obs_size = len(self._observe_template())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    # ---- helpers -------------------------------------------------------
    def _observe_template(self) -> np.ndarray:
        mujoco.mj_forward(self.m, self.d)
        return self._observe()

    @property
    def phase(self) -> float:
        return self.choreo.phase

    @phase.setter
    def phase(self, value: float) -> None:
        self.choreo.phase = value

    @property
    def bpm(self) -> float:
        return self.choreo.bpm

    @bpm.setter
    def bpm(self, value: float) -> None:
        self.choreo.bpm = value

    def _local_effectors(self, data: mujoco.MjData) -> np.ndarray:
        frame = yaw_matrix(yaw_of(data.qpos[3:7]))
        return ((data.xpos[self.effector_ids] - data.qpos[:3]) @ frame.T).ravel()

    def _foot_points(self) -> np.ndarray:
        """Where each foot is on the floor: (x, z) midway between heel and toe."""
        return np.array([self.d.site_xpos[ids].mean(axis=0)[[0, 2]] for ids in self.sole_sites])

    def _contacts(self) -> tuple[np.ndarray, bool]:
        feet = np.zeros(2)
        bad = False
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            other = c.geom2 if c.geom1 == self.floor else c.geom1 if c.geom2 == self.floor else None
            if other is None:
                continue
            if other in self.foot_geoms:
                feet[self.foot_geoms.index(other)] = 1.0
            elif other in self.never_geoms or (self.choreo.recover is None and other not in self.allowed_geoms):
                bad = True
        return feet, bad

    def _observe(self) -> np.ndarray:
        q, v = self.d.qpos, self.d.qvel
        frame = yaw_matrix(yaw_of(q[3:7]))
        orientation = (frame @ quat_matrix(q[3:7]))[:, :2].ravel()
        phase = 2 * np.pi * self.phase
        ref_next, _, feet_next, _ = self.choreo.target(self.control_dt)
        feet, _ = self._contacts()
        return np.concatenate([
            [np.sin(phase), np.cos(phase), q[1]],
            orientation,
            q[self.joint_qpos],
            0.1 * v[self.joint_qvel],
            frame @ v[:3], v[3:6] * 0.2,
            self._local_effectors(self.d),
            feet,
            ref_next[self.joint_qpos] - q[self.joint_qpos],
            frame @ (ref_next[:3] - q[:3]),
            self._feet_offset(frame, feet_next),
            [tempo_input(self.bpm)],
            self.choreo.features(),
        ]).astype(np.float32)

    def _feet_offset(self, frame: np.ndarray, target: np.ndarray) -> np.ndarray:
        """How far each foot is from its spot on the triangle, in the heading frame."""
        offset = target - self._foot_points()
        return np.array([(frame @ np.array([o[0], 0.0, o[1]]))[[0, 2]] for o in offset]).ravel()

    def lean(self) -> np.ndarray:
        """Which way the body is going over, on the floor: where its centre of mass is past the middle of
        the feet, plus the hips' drift. (Not how the hips tip: the ginga itself leans them well forward.)"""
        return self.d.subtree_com[1][[0, 2]] - self._foot_points().mean(axis=0) + 0.3 * self.d.qvel[[0, 2]]

    def _sinking(self, q: np.ndarray) -> float:
        self.ground_data.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.ground_data)
        return float(max(-(self.ground_data.site_xpos[self.ground_sites, 1] - self.ground_clear).min(), 0.0))

    def bent(self) -> float:
        """How far the trunk leans forward (rad): the hips-to-head line from the vertical, towards where the hips face."""
        yaw = yaw_of(self.d.qpos[3:7])
        up = self.d.xpos[self.m.body("head").id] - self.d.qpos[:3]
        return float(np.arctan2(np.sin(yaw) * up[0] + np.cos(yaw) * up[2], up[1]))

    def low(self, ref: mujoco.MjData) -> bool:
        """Hips or head too low: under LOW_HIPS / LOW_HEAD, or — where the reference itself goes low, as in
        an esquiva — LOW_BELOW further down than the reference's (ref: the reference pose's kinematics)."""
        head = self.m.body("head").id
        head_below = LOW_BELOW + 0.08 if self.choreo.move in DODGES else LOW_BELOW
        return bool(
            self.d.qpos[1] < min(LOW_HIPS, ref.qpos[1] - LOW_BELOW)
            or self.d.xpos[head][1] < min(LOW_HEAD, ref.xpos[head][1] - head_below)
        )

    def trouble(self) -> bool:
        """Balance is lost: something besides hands and feet touches the floor, the head drops, or the
        hips are well off the reference — tipped, sunk or away. While a kick or dodge is active (or settling),
        tilt, hip-height lag and horizontal step are part of the move and only a real fall triggers recover."""
        q, q_ref = self.d.qpos, self.choreo.target(0.0)[0]
        _, bad = self._contacts()
        tilt = np.arccos(np.clip(quat_matrix(q[3:7])[:, 1] @ quat_matrix(q_ref[3:7])[:, 1], -1.0, 1.0))
        self.ref_data.qpos[:] = q_ref
        mujoco.mj_kinematics(self.m, self.ref_data)
        in_move = self.choreo.move is not None or self.choreo.since_move < 0.5
        return bool(
            bad
            or self.low(self.ref_data)
            or (
                not in_move
                and (
                    tilt > TROUBLE_TILT
                    or q[1] < q_ref[1] - TROUBLE_DROP
                    or np.linalg.norm(q[[0, 2]] - q_ref[[0, 2]]) > 0.5
                )
            )
        )

    def pace(self) -> float:
        """How fast the reference moves on: slower on the way up while the body lags behind it (WAIT_FROM)."""
        r = self.choreo.recover
        if r is None or r["t"] < FALL_TIME:
            return 1.0
        lag = float(np.linalg.norm(self.d.qpos[:3] - self.choreo.target(0.0)[0][:3]))
        return float(np.clip(1.0 - (lag - WAIT_FROM) / WAIT_SPAN, MIN_PACE, 1.0))

    def push(self, force: np.ndarray, seconds: float = 0.2) -> None:
        """Shove the chest with a horizontal force (N, world x/z) for a moment, like a partner would."""
        self.push_force = np.array([force[0], 0.0, force[1]])
        self.push_left = int(round(seconds / self.control_dt))

    # ---- gym API -------------------------------------------------------
    def step(self, action: np.ndarray):
        self.act(action)
        chest = self.m.body("chest").id
        self.d.xfrc_applied[chest, :3] = self.push_force if self.push_left > 0 else 0.0
        self.push_left = max(self.push_left - 1, 0)
        mujoco.mj_step(self.m, self.d, nstep=FRAME_SKIP)
        was_recovering = self.choreo.recover is not None
        self.advance()
        self.steps += 1

        reward, parts = self._reward()
        _, bad_contact = self._contacts()
        drift = np.linalg.norm(self.d.qpos[[0, 2]] - self.choreo.target(0.0)[0][[0, 2]])
        low = stuck = dive = False
        if self.choreo.recover is not None:
            # Down and up again: only the head may not touch the floor; wherever the body lands, the way up starts there,
            # but sliding away from it after landing is drifting (as in check.py), and a body lying there while the way up
            # waits for it is stuck.
            self.landing = self.choreo.recover["move"]
            landed = self.choreo.recover["t"] >= FALL_TIME
            drifted = landed and drift > LOST
            self.stuck = self.stuck + self.control_dt if landed and self.pace() <= MIN_PACE else 0.0
            stuck = self.stuck >= STUCK
            # (_reward left the reference pose's kinematics in ref_data)
            head = self.m.body("head").id
            dive = self.choreo.recover["t"] < FALL_TIME + DIVE_UNTIL and self.d.xpos[head][1] < self.ref_data.xpos[head][1] - DIVE
        else:
            low = self.low(self.ref_data)
            drifted = drift > 0.5
        terminated = bool(bad_contact or low or drifted or stuck or dive)
        end = "contact" if bad_contact else "low" if low else "drift" if drifted else "stuck" if stuck else "dive" if dive else "time"
        if was_recovering and self.choreo.recover is None:
            self.after_recover = 0.0
        if self.pool_level == self.fall_level and (terminated or self.after_recover is not None):
            self._fell(self.after_recover is not None)
            self.pool_level = None
        if self.after_recover is not None:
            self.after_recover += self.control_dt
        # Only whole kicks count (a start in the middle of a kick is easier).
        if self.skill in ("kicks", "esquiva") and self.from_ginga and self.choreo.kicks.level == self.curriculum:
            finished = self.after_kick is None and not self.choreo.busy and self.kicked
            if not self.counted and (terminated and (self.choreo.busy or self.kicked) or finished):
                self.counted = True
                self._attempt(not terminated)
        truncated = self.steps >= self.max_steps
        if self.choreo.move is not None:
            self.kicked = True
        if self.skill in ("kicks", "esquiva") and not self.choreo.busy:
            # A kick episode ends a while after the body is back in the ginga.
            self.after_kick = (self.after_kick or 0.0) + self.control_dt
            truncated = truncated or self.after_kick >= MOVE_EPISODE_TAIL
        if self.skill == "recover" and self.after_recover is not None:
            truncated = truncated or self.after_recover >= RECOVER_TAIL
        # Logged only, read at the end of an episode: why it ended, which landing it fell into, whether it got up.
        parts["end"] = ENDS.index(end)
        parts["landing"] = LANDINGS.index(self.landing) + 1 if self.landing else 0
        parts["got_up"] = float(self.after_recover is not None)
        parts["from_fall"] = float(self.from_fall)
        return self._observe(), (0.0 if terminated else reward), terminated, truncated, parts

    def _random_tempo(self) -> float:
        """Slow tempos are drawn more often: the long, low Angola ginga is the hardest to keep on the triangle."""
        return float(MIN_BPM + (MAX_BPM - MIN_BPM) * self.rng.random() ** TEMPO_SKEW)

    def _attempt(self, ok: bool) -> None:
        """Count a kick at the current curriculum size; move up once 75 % of recent ones end standing."""
        self.attempts += 1
        self.success += 0.1 * (float(ok) - self.success)
        if self.attempts >= 20 and self.success > 0.75 and self.curriculum < len(self.choreo.kicks.amplitudes) - 1:
            self.curriculum += 1
            self.attempts, self.success = 0, 0.0

    def _fell(self, up: bool) -> None:
        """Count a recorded fall at the current speed; move up once half of recent ones end standing."""
        self.fall_attempts += 1
        self.fall_success += 0.1 * (float(up) - self.fall_success)
        if self.fall_attempts >= 20 and self.fall_success > 0.5 and self.fall_level < len(FALL_SPEEDS) - 1:
            self.fall_level += 1
            self.fall_attempts, self.fall_success = 0, 0.0

    def brace(self, on: bool) -> None:
        """Arms braced (STRONG_ARMS) or relaxed."""
        if on == self.braced:
            return
        self.braced = on
        for i, relaxed, braced in self.arms:
            kp, kv, force = braced if on else relaxed
            self.m.actuator_gainprm[i, 0] = kp
            self.m.actuator_biasprm[i, 1:3] = -kp, kv
            self.m.actuator_forcerange[i] = -force, force

    def act(self, action: np.ndarray) -> None:
        """PD targets: the reference one control step ahead plus the policy's correction; the arms brace
        while falling and getting up."""
        self.brace(self.choreo.recover is not None)
        target = self.choreo.target(self.control_dt)[0]
        ctrl = target[self.joint_qpos] + ACTION_SCALE * np.clip(action, -1, 1)
        self.d.ctrl[:] = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)

    def advance(self) -> None:
        """Move on by one control step; while training, the toque sometimes changes and kicks are asked for."""
        r = self.choreo.recover
        falling = r is not None and r["t"] < FALL_TIME
        self.choreo.advance(self.control_dt * self.pace())
        if falling and r["t"] >= FALL_TIME:
            self.choreo.land(self.d.qpos)
        self.t += self.control_dt
        if self.vary_tempo and self.rng.random() < self.control_dt / TEMPO_CHANGE_EVERY:
            self.bpm = self._random_tempo()
        if self.vary_moves and not self.choreo.busy and self.rng.random() < self.control_dt / KICK_EVERY:
            side = "lr"[self.rng.integers(2)]
            entry = ENTRY_CENTRE[side] + self.rng.uniform(-ENTRY_WINDOW, ENTRY_WINDOW)
            practice = SKILL_MOVES.get(self.skill, KICK_MOVES)
            self.choreo.schedule(practice[self.rng.integers(len(practice))], side, entry)
