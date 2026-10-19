"""Starting an episode of GingaEnv: a ginga, a kick or esquiva (or its entry), a recorded fall, a landing pose or a way up."""

import mujoco
import numpy as np

from env_consts import (
    ANCHOR_LIMIT,
    ENTRY_CENTRE,
    ENTRY_WINDOW,
    FALL_SPEEDS,
    FALL_TIME,
    GINGA_FIRST,
    KICK_MOVES,
    LANDINGS,
    SKILL_MOVES,
)
from env_math import place_pose


def lift_out_of_floor(model: mujoco.MjModel, data: mujoco.MjData, floor: int) -> float:
    """How far to raise a pose so nothing sinks into the floor (the reference grounds the soles' centre
    lines, but a rolled foot's box corner can sit a centimetre deep, and a deep start launches the body)."""
    mujoco.mj_forward(model, data)
    deepest = min((c.dist for c in data.contact[: data.ncon] if floor in (c.geom1, c.geom2)), default=0.0)
    return max(-deepest, 0.0) + 0.002


class EpisodeReset:
    """GingaEnv.reset and the recovery episodes it can start."""

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        self.push_left = 0
        options = options or {}
        start = options.get("phase")
        self.phase = float(self.rng.uniform(0, 1) if start is None else start)
        bpm = options.get("bpm")
        self.bpm = float(self._random_tempo() if bpm is None else bpm)
        self.vary_tempo = bpm is None
        self.vary_moves = options.get("moves", self.skill == "all")
        self.t = 0.0
        self.after_kick = None
        self.kicked = False
        self.counted = False
        self.from_ginga = False
        self.after_recover = None
        self.landing = None
        self.stuck = 0.0
        self.pool_level = None
        self.from_fall = False  # started from a recorded fall (not on the floor or partway up)
        self.choreo.move, self.choreo.pending, self.choreo.recover = None, None, None
        self.choreo.since_move = np.inf
        self.choreo.frame = np.asarray(options.get("frame", (0.0, 0.0, 0.0)), float)
        if self.skill == "recover" or "recover" in options:
            if self._reset_recover(options):
                return self._observe(), {}
        if self.skill in ("kicks", "esquiva"):
            top = self.curriculum
            self.choreo.kicks.level = top if self.rng.random() < 0.7 else int(self.rng.integers(top + 1))
        if "level" in options:
            self.choreo.kicks.level = options["level"]
        move = options.get("move")
        rsi = 0.3 if self.vary_moves else 0.0
        practice = SKILL_MOVES.get(self.skill, KICK_MOVES)
        if move is None and "moves" not in options and (self.skill in ("kicks", "esquiva") or self.rng.random() < rsi):
            move = practice[self.rng.integers(len(practice))]
        anchor = options.get("anchor")
        noisy = False
        if move is not None and self.skill in ("kicks", "esquiva") and "kick_t" not in options and self.rng.random() < GINGA_FIRST:
            # Practice from the plain ginga: it waits for the entry, as in the game, so the network keeps
            # practising the ginga it took over from the ginga network (and forgets it otherwise: alone, the
            # esquiva and kick networks fell 4–12 and 5–18 times in 30 s at each toque; the ginga network, never).
            side = options.get("side") or "lr"[self.rng.integers(2)]
            self.choreo.schedule(move, side, ENTRY_CENTRE[side] + self.rng.uniform(-ENTRY_WINDOW, ENTRY_WINDOW))
            self.from_ginga, move = True, None
        if move is not None:
            # Start inside a kick (reference state initialisation covers every part of it).
            side = options.get("side") or "lr"[self.rng.integers(2)]
            self.phase = (ENTRY_CENTRE[side] + self.rng.uniform(-ENTRY_WINDOW, ENTRY_WINDOW)) % 1.0
            duration = self.choreo.kicks.duration(move)
            kick_t = options.get("kick_t")
            if kick_t is None and self.skill in ("kicks", "esquiva") and self.rng.random() < 0.5:
                # The whole kick, from its first frame: the state the ginga network hands over —
                # never exactly the reference, and never exactly on the corner (see Choreography).
                kick_t, noisy, self.from_ginga = 0.0, True, True
                anchor = self.rng.uniform(-ANCHOR_LIMIT, ANCHOR_LIMIT, 2) * 0.5 if anchor is None else anchor
            elif kick_t is None:
                kick_t = self.rng.uniform(0, duration * 0.8)
            self.choreo.start(move, side, float(kick_t), np.asarray((0.0, 0.0) if anchor is None else anchor, float))
        q, v, _, _ = self.choreo.target(0.0)
        self.d.qpos[:] = q
        self.d.qvel[:] = v
        if noisy:
            self.d.qpos[self.joint_qpos] += self.rng.normal(0, 0.03, len(self.joint_qpos))
            self.d.qvel[self.joint_qvel] += self.rng.normal(0, 0.2, len(self.joint_qvel))
        self.d.qpos[1] += lift_out_of_floor(self.m, self.d, self.floor)
        mujoco.mj_forward(self.m, self.d)
        self.steps = 0
        return self._observe(), {}

    def _reset_recover(self, options: dict) -> bool:
        """Start a recovery-practice episode; True when the state is already set (a fall from the pool).
        A fall from the pool half the time (going down is where the body fails), on the floor in a landing pose
        20 %, partway up 30 %."""
        given = options.get("recover")
        draw = self.rng.random()
        if given is None and self.falls is not None and draw < 0.5:
            # A real loss of balance, recorded from the trained ginga and kicks (falls.py).
            # Gentler first (FALL_SPEEDS): mostly the current speed, sometimes a slower one.
            i = int(self.rng.integers(len(self.falls["qpos"])))
            top = self.fall_level
            self.pool_level = top if self.rng.random() < 0.7 else int(self.rng.integers(top + 1))
            self.from_fall = True
            self.d.qpos[:] = self.falls["qpos"][i]
            self.d.qvel[:] = self.falls["qvel"][i] * FALL_SPEEDS[self.pool_level]
            self.phase, self.bpm = float(self.falls["phase"][i]), float(self.falls["bpm"][i])
            mujoco.mj_forward(self.m, self.d)
            self.choreo.begin_recover(self.d.qpos.copy(), self.lean(), self.bent(), float(self.d.qvel[1]))
            self.steps = 0
            return True
        # Partway down or up (reference state initialisation), anywhere on the floor, facing anywhere.
        given = given or {}
        move = given.get("move") or LANDINGS[self.rng.integers(len(LANDINGS))]
        side = given.get("side") or "lr"[self.rng.integers(2)]
        duration = self.choreo.floor.duration(move, side)
        t = given.get("t", FALL_TIME + (0.0 if draw < 0.7 else self.rng.uniform(0, 0.85 * duration)))
        place = np.asarray(given.get("place", (self.rng.uniform(-1, 1), self.rng.uniform(-1, 1), self.rng.uniform(-np.pi, np.pi))), float)
        onset = place_pose(place, self.choreo.floor.sample(move, side, 0.0)[0], np.zeros(self.m.nv), np.zeros((2, 2)))[0]
        self.choreo.recover = {"move": move, "side": side, "t": float(t), "onset": onset, "place": place}
        return False
