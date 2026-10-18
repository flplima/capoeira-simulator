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

# Small quaternion helpers (w, x, y, z) — scipy's Rotation is too slow per step.


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) of the rotation between two unit quaternions."""
    return 2.0 * np.arccos(min(1.0, abs(float(np.dot(a, b)))))


def nlerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    if np.dot(a, b) < 0:
        b = -b
    q = a + (b - a) * t
    return q / np.linalg.norm(q)


def yaw_of(q: np.ndarray) -> float:
    """Heading about the vertical Y axis: the direction the body's +Z axis points, projected on the floor."""
    w, x, y, z = q
    fx = 2 * (x * z + w * y)
    fz = 1 - 2 * (x * x + y * y)
    return float(np.arctan2(fx, fz))


def yaw_matrix(yaw: float) -> np.ndarray:
    """Rotation taking world vectors into the heading frame (undoing the yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def quat_matrix(q: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


# A placement on the floor: (x offset, z offset, yaw). The references live in
# their own frame (the ginga on its triangle, a get-up clip where it was
# solved); a placement puts them somewhere else in the world, turned about
# the vertical. Yaw turns +Z (forward) towards +X.


def place_xz(frame: np.ndarray, xz: np.ndarray) -> np.ndarray:
    """Reference-frame floor points (..., 2) → world."""
    c, s = np.cos(frame[2]), np.sin(frame[2])
    x, z = xz[..., 0], xz[..., 1]
    return np.stack([c * x + s * z + frame[0], -s * x + c * z + frame[1]], axis=-1)


def unplace_xz(frame: np.ndarray, xz: np.ndarray) -> np.ndarray:
    """World floor points (..., 2) → the reference frame."""
    c, s = np.cos(frame[2]), np.sin(frame[2])
    x, z = xz[..., 0] - frame[0], xz[..., 1] - frame[1]
    return np.stack([c * x - s * z, s * x + c * z], axis=-1)


def place_pose(frame: np.ndarray, q: np.ndarray, v: np.ndarray, feet: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A reference state (qpos, qvel, feet floor points) → world. The free joint's angular velocity
    is in the body's own frame, so only the linear part turns."""
    q, v = q.copy(), v.copy()
    q[[0, 2]] = place_xz(frame, q[[0, 2]])
    q[3:7] = quat_mul(np.array([np.cos(frame[2] / 2), 0.0, np.sin(frame[2] / 2), 0.0]), q[3:7])
    c, s = np.cos(frame[2]), np.sin(frame[2])
    v[0], v[2] = c * v[0] + s * v[2], -s * v[0] + c * v[2]
    return q, v, place_xz(frame, feet)

FRAME_SKIP = 15  # 0.002 s × 15 = 0.03 s per action
ACTION_SCALE = 0.5  # rad
EFFECTORS = ["foot_l", "foot_r", "hand_l", "hand_r", "head"]
ALLOWED_CONTACT = {"foot_l", "foot_r", "hand_l", "hand_r"}
# While falling and getting up, anything but the head and neck may touch the floor.
NEVER_CONTACT = {"head", "neck"}


MIN_BPM, MAX_BPM = 60.0, 130.0
TEMPO_SKEW = 1.3  # >1 favours slow tempos when sampling (see GingaEnv._random_tempo): slow is the hardest to keep, fast still needs practice
TEMPO_CHANGE_EVERY = 8.0  # s, on average, while training: the toque can change mid-game


def tempo_input(bpm: float) -> float:
    """The tempo as the policy sees it, about −1 (slowest) to 1 (fastest)."""
    return (bpm - 95.0) / 35.0


class Reference:
    """The looping reference cycle at any tempo, sampled by phase (0..1 over one cycle).

    The reference file holds one cycle per tempo on a grid (see reference.py);
    tempos in between blend their two neighbours.
    """

    def __init__(self, path: str, model: mujoco.MjModel):
        data = np.load(path)
        self.tempos = data["tempos"]
        self.periods = data["periods"]
        self.qpos = data["qpos"]  # (tempos, samples, nq)
        self.feet = data["feet"]  # (tempos, samples, 2 feet, x/z)
        self.triangle = data["triangle"]
        self.n = self.qpos.shape[1]
        self.model = model
        self.qvel = np.zeros((len(self.tempos), self.n, model.nv))
        for i, period in enumerate(self.periods):
            dt = period / self.n
            for k in range(self.n):
                mujoco.mj_differentiatePos(model, self.qvel[i, k], 2 * dt, self.qpos[i, k - 1], self.qpos[i, (k + 1) % self.n])

    @staticmethod
    def period(bpm: float) -> float:
        return 4 * 60 / bpm

    def _neighbours(self, bpm: float) -> tuple[int, int, float]:
        bpm = min(max(bpm, self.tempos[0]), self.tempos[-1])
        j = int(min(np.searchsorted(self.tempos, bpm), len(self.tempos) - 1))
        i = max(j - 1, 0)
        w = 0.0 if i == j else (bpm - self.tempos[i]) / (self.tempos[j] - self.tempos[i])
        return i, j, float(w)

    def _at(self, table: np.ndarray, phase: float) -> tuple[np.ndarray, np.ndarray, float]:
        x = (phase % 1.0) * self.n
        k = int(x) % self.n
        return table[k], table[(k + 1) % self.n], x - int(x)

    def sample(self, phase: float, bpm: float) -> tuple[np.ndarray, np.ndarray]:
        i, j, w = self._neighbours(bpm)
        out_q, out_v = [], []
        for t in (i, j):
            q0, q1, a = self._at(self.qpos[t], phase)
            q = q0 + (q1 - q0) * a
            q[3:7] = nlerp(q0[3:7], q1[3:7], a)
            v0, v1, _ = self._at(self.qvel[t], phase)
            out_q.append(q)
            out_v.append(v0 + (v1 - v0) * a)
        q = out_q[0] + (out_q[1] - out_q[0]) * w
        q[3:7] = nlerp(out_q[0][3:7], out_q[1][3:7], w)
        return q, out_v[0] + (out_v[1] - out_v[0]) * w

    def feet_at(self, phase: float, bpm: float) -> np.ndarray:
        i, j, w = self._neighbours(bpm)
        f = []
        for t in (i, j):
            f0, f1, a = self._at(self.feet[t], phase)
            f.append(f0 + (f1 - f0) * a)
        return f[0] + (f[1] - f[0]) * w


KICK_MOVES = ["martelo", "bencao", "meia-lua-de-frente", "queixada", "ponteira", "joelhada", "meia-lua-de-compasso"]
# While a kick is under way, this share of the reward pays for the kicking foot being where the reference's is,
# relative to the hips but not to their heading (GingaEnv._reward "strike"): 0.3 m off scores exp(-4·0.09) ≈ 0.7.
STRIKE_WEIGHT, STRIKE_SCALE = 0.15, 4.0
# …and this share for the support foot pointing where the reference's does (its heading, "pivot"): 30° off
# scores exp(-2·0.27) ≈ 0.58.
PIVOT_WEIGHT, PIVOT_SCALE = 0.1, 2.0
DODGES = ["esquiva", "esquiva-baixa", "cocorinha"]  # see esquiva.py
MOVES = KICK_MOVES + DODGES
EXTRA_MOVES = ["meia-lua-de-frente", "queixada", "ponteira", "joelhada", "meia-lua-de-compasso", "esquiva-baixa", "cocorinha"]
# What each practice skill asks for (GingaEnv skill), and which network plays each move (brain.py).
SKILL_MOVES = {"kicks": KICK_MOVES, "esquiva": DODGES, "all": MOVES}
# A kick (or an esquiva) blends in from the ginga over BLEND_IN seconds and back out over BLEND_OUT.
BLEND_IN, BLEND_OUT = 0.35, 0.35
# Where in the ginga cycle each leg can start a kick: around the base with that leg back.
ENTRY_CENTRE = {"r": 0.0, "l": 0.5}
ENTRY_WINDOW = 0.1
KICK_EVERY = 4.0  # s, on average, while training
ANCHOR_LIMIT = 0.25  # m: furthest a kick follows its support foot away from the corner
# s of ginga after a kick or an esquiva in a practice episode. The move's network drives the ginga that long
# before handing back (brain.MOVE_HANDBACK), so it practises getting all the way back into a settled ginga —
# the ginga network fell from what it was handed after 1 s.
MOVE_EPISODE_TAIL = 3.0
GINGA_FIRST = 0.3  # share of kick and esquiva practice episodes that start in the plain ginga and wait for the entry


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Kicks:
    """Reference kicks (see kicks.py) and esquivas (esquiva.py), per move, leg and size, sampled by time.
    A move's side is the leg that is back in the base it starts from."""

    def __init__(self, paths: list[str], model: mujoco.MjModel):
        self.clips = {}
        for path in paths:
            data = np.load(path)
            fps = float(data["fps"])
            self.amplitudes = data["amplitudes"].tolist()
            for move in MOVES:
                if f"{move}_duration" not in data:
                    continue
                duration = float(data[f"{move}_duration"])
                for side in "lr":
                  for level in range(len(self.amplitudes)):
                    qpos = data[f"{move}_{side}_{level}_qpos"]
                    qvel = np.zeros((len(qpos), model.nv))
                    for k in range(len(qpos)):
                        a, b = qpos[max(k - 1, 0)], qpos[min(k + 1, len(qpos) - 1)]
                        span = (min(k + 1, len(qpos) - 1) - max(k - 1, 0)) / fps
                        mujoco.mj_differentiatePos(model, qvel[k], span, a, b)
                    self.clips[move, side, level] = {
                        "duration": duration,
                        "fps": fps,
                        "qpos": qpos,
                        "qvel": qvel,
                        "feet": data[f"{move}_{side}_{level}_feet"],
                        "mask": data[f"{move}_{side}_{level}_mask"],
                    }
        self.level = len(self.amplitudes) - 1  # size used when not training (the full move)

    def duration(self, move: str) -> float:
        return self.clips[move, "r", 0]["duration"]

    def sample(self, move: str, side: str, t: float, level: int | None = None):
        clip = self.clips[move, side, self.level if level is None else level]
        x = min(max(t, 0.0), clip["duration"]) * clip["fps"]
        k = min(int(x), len(clip["qpos"]) - 2)
        a = x - k
        q0, q1 = clip["qpos"][k], clip["qpos"][k + 1]
        q = q0 + (q1 - q0) * a
        q[3:7] = nlerp(q0[3:7], q1[3:7], a)
        v = clip["qvel"][k] + (clip["qvel"][k + 1] - clip["qvel"][k]) * a
        feet = clip["feet"][k] + (clip["feet"][k + 1] - clip["feet"][k]) * a
        mask = np.minimum(clip["mask"][k], clip["mask"][k + 1])
        return q, v, feet, mask


LANDINGS = ["quatro", "negativa", "joelhos"]  # see floor.py
# Why an episode ended (logged by train.py as shares of the episodes that ended).
ENDS = ["time", "contact", "low", "drift", "stuck", "dive"]
FALL_TIME = 0.5  # s from losing balance to the landing pose (see _recover_target)
TROUBLE_TILT = 0.65  # rad: the hips tipped this far from the reference's (heading aside: a martelo's pivot is no fall)…
TROUBLE_DROP = 0.25  # m: …or this far below its height, and the body is falling
# Outside a recovery the hips below LOW_HIPS or the head below LOW_HEAD (m) is a fall — or, where the reference
# goes lower than that (an esquiva), LOW_BELOW further down than the reference's.
LOW_HIPS, LOW_HEAD, LOW_BELOW = 0.4, 0.8, 0.3
RECOVER_TAIL = 2.0  # s of ginga after getting up in a recovery-practice episode
# While falling and getting up the arms brace — a person catching their weight on their hands tenses them:
# the arm motors' stiffness (N·m/rad) and torque limit (N·m) become these (the ginga's are ~half; damping
# stays critical). The ginga and the kicks keep the relaxed arms they were trained with.
STRONG_ARMS = {"shoulder": (300.0, 100.0), "elbow": (250.0, 80.0)}
# On the way up, the reference waits for a body that lags behind it: its clock slows from full speed with the
# hips WAIT_FROM (m) off the reference's to MIN_PACE of it at WAIT_FROM + WAIT_SPAN — getting up from the floor
# is done at the body's own pace, not the clip's.
WAIT_FROM, WAIT_SPAN, MIN_PACE = 0.15, 0.3, 0.2
LOST = 0.6  # m: on the floor, this far from the way up and the body is lost (only after landing)
STUCK = 1.0  # s: after landing, the way up waiting at its slowest this long and the body is stuck on the floor
# Fall curriculum (skill "recover"): a recorded fall starts with its velocity scaled to one of these; each
# environment starts with the gentlest and moves up once most of its falls at that speed end up standing.
FALL_SPEEDS = (0.0, 0.35, 0.7, 1.0)
BENT = 0.9  # rad: a trunk already bent forward this far (hips to head) lands on hands and knees whichever way it goes
GRAVITY = 9.81  # m/s²
# Going down and just after landing, the head this far below the reference's ends the episode (skill recover): a
# body that keeps its hips up and pitches over them dives head first, and only hits the floor a moment later.
DIVE, DIVE_UNTIL = 0.3, 0.5  # m; s after landing
# Going down, no pose has these below the floor (a knee: its shin's surface, this far below the joint's site).
GROUND_SITES = ["heel_l", "toe_l", "heel_r", "toe_r", "knee_l", "knee_r", "fingers_l", "fingers_r"]
KNEE_CLEAR = 0.054
HEAD_ABOVE = 0.1  # m: going down, the head stays at least this far above the hips (every landing has it ~0.2 m up or more)


class Floor:
    """Falling and getting up (see floor.py): per landing and side, the way up from the landing pose
    to the ginga's base, with the feet's floor points, sampled by time."""

    def __init__(self, path: str, model: mujoco.MjModel):
        data = np.load(path)
        self.fps = float(data["fps"])
        sites = data["sites"].tolist()
        heel_toe = [[sites.index(f"heel_{s}"), sites.index(f"toe_{s}")] for s in "lr"]
        scratch = mujoco.MjData(model)
        sole = [[model.site(f"heel_{s}").id, model.site(f"toe_{s}").id] for s in "lr"]
        self.clips = {}
        for move in LANDINGS:
            for side in "lr":
                qpos = data[f"{move}_{side}_qpos"]
                qvel = np.zeros((len(qpos), model.nv))
                feet = np.zeros((len(qpos), 2, 2))
                for k in range(len(qpos)):
                    a, b = qpos[max(k - 1, 0)], qpos[min(k + 1, len(qpos) - 1)]
                    span = (min(k + 1, len(qpos) - 1) - max(k - 1, 0)) / self.fps
                    mujoco.mj_differentiatePos(model, qvel[k], span, a, b)
                    scratch.qpos[:] = qpos[k]
                    mujoco.mj_kinematics(model, scratch)
                    feet[k] = [scratch.site_xpos[ids].mean(axis=0)[[0, 2]] for ids in sole]
                mask = data[f"{move}_{side}_mask"][:, heel_toe].max(axis=2)
                self.clips[move, side] = {
                    "duration": float(data[f"{move}_{side}_duration"]),
                    "phase": float(data[f"{move}_{side}_phase"]),  # the ginga's phase at the base it ends in
                    "qpos": qpos,
                    "qvel": qvel,
                    "feet": feet,
                    "mask": mask,
                }

    def duration(self, move: str, side: str) -> float:
        return self.clips[move, side]["duration"]

    def sample(self, move: str, side: str, t: float):
        clip = self.clips[move, side]
        x = min(max(t, 0.0), clip["duration"]) * self.fps
        k = min(int(x), len(clip["qpos"]) - 2)
        a = x - k
        q0, q1 = clip["qpos"][k], clip["qpos"][k + 1]
        q = q0 + (q1 - q0) * a
        q[3:7] = nlerp(q0[3:7], q1[3:7], a)
        v = clip["qvel"][k] + (clip["qvel"][k + 1] - clip["qvel"][k]) * a
        feet = clip["feet"][k] + (clip["feet"][k + 1] - clip["feet"][k]) * a
        mask = np.minimum(clip["mask"][k], clip["mask"][k + 1])
        return q, v, feet, mask


class Choreography:
    """What the body should be doing: the ginga at the berimbau's tempo, or a kick.

    The ginga runs on its phase; a kick is scheduled for an entry phase (the
    base with the kicking leg back, give or take ENTRY_WINDOW), freezes the
    ginga there while it plays, blending in from and back out to the ginga
    pose, and then the ginga carries on from the same phase.

    The ginga is anchored to the triangle; a kick to the foot the body actually
    stands on. A real ginga never lands exactly on its corner, and a kick
    whose reference balances over the corner while the foot is 10 cm away
    asks the body to put its weight beside its foot. So when a kick starts,
    the whole kick reference shifts by the support foot's distance from its
    corner (at most ANCHOR_LIMIT), fading in and out with the blend.
    """

    def __init__(self, ref: Reference, kicks: Kicks, floor: Floor):
        self.ref, self.kicks, self.floor = ref, kicks, floor
        self.frame = np.zeros(3)  # where the ginga's triangle is (see place_xz); it moves when the body gets up elsewhere
        self.recover: dict | None = None  # falling and getting up: landing, side, time, onset pose, placement
        self.phase, self.bpm = 0.0, 66.0
        self.move: str | None = None  # kick being played
        self.side = "r"
        self.kick_t = 0.0
        self.since_move = np.inf
        self.pending: tuple[str, str, float] | None = None  # (move, side, entry phase)
        self.anchor = np.zeros(2)  # xz shift of the kick reference (see above)
        self.feet_now = None  # callable → the feet's floor points (x, z), set by the environment
        self.arms: list[int] = []  # qpos addresses of the shoulders and elbows, set by the environment
        self.sinking = None  # callable: how far (m) a pose's feet, knees or hands are below the floor; set by the environment

    @property
    def busy(self) -> bool:
        return self.move is not None or self.pending is not None or self.recover is not None

    def begin_recover(self, onset: np.ndarray, lean: np.ndarray, bent: float = 0.0, sink: float = 0.0) -> None:
        """Losing balance: pick the landing from where the body is going, and place it there.

        Falling forwards ends on hands and knees, backwards in the queda de quatro, sideways in
        the negativa — forwards, backwards and sideways as the body faces (the hips' heading), which is
        also the way the landing faces, a little way along the fall. The negativa folds the leg that
        carries the weight (the foot nearest where the body is going over) and stretches the free one:
        folding the free leg straightens the one the body stands on, and it can only pivot over it.
        A trunk already bent far forward (bent, rad: GingaEnv.bent) goes down onto its hands and knees:
        its head is over its hands, and sitting into a negativa from there dives head first.
        sink: the hips' vertical speed (m/s, up positive) — going down, they can't drop faster than falling."""
        yaw = yaw_of(onset[3:7])
        c, s = np.cos(yaw), np.sin(yaw)
        side_way, forward = c * lean[0] - s * lean[1], s * lean[0] + c * lean[1]  # in the body's frame: +x left, +z forward
        if bent > BENT:
            move, side = "joelhos", "r" if self.phase < 0.25 or self.phase >= 0.75 else "l"
        elif abs(forward) >= abs(side_way):
            move, side = "joelhos" if forward > 0 else "quatro", "r" if self.phase < 0.25 or self.phase >= 0.75 else "l"
        else:
            # negativa_r folds the left leg, negativa_l the right one
            if self.feet_now is not None:
                feet = self.feet_now()
                stance = int(np.argmin(np.linalg.norm(feet - (feet.mean(axis=0) + lean), axis=1)))  # 0: left
            else:
                stance = 1 if side_way < 0 else 0
            move, side = "negativa", "r" if stance == 0 else "l"
        landing = self.floor.sample(move, side, 0.0)[0]
        norm = np.linalg.norm(lean)
        target = onset[[0, 2]] + (0.25 * lean / norm if norm > 1e-6 else 0.0)
        place = np.array([0.0, 0.0, yaw])
        place[:2] = target - place_xz(place, landing[[0, 2]])
        self.move, self.pending = None, None
        self.recover = {"move": move, "side": side, "t": 0.0, "onset": onset.copy(), "place": place, "sink": sink}

    def land(self, root: np.ndarray) -> None:
        """The body has landed (the fall's time is up): the way up starts where it actually is."""
        r = self.recover
        landing = self.floor.sample(r["move"], r["side"], 0.0)[0]
        r["place"][:2] = root[[0, 2]] - place_xz(np.r_[0.0, 0.0, r["place"][2]], landing[[0, 2]])

    def schedule(self, move: str, side: str, entry: float) -> None:
        self.pending = (move, side, entry % 1.0)

    def start(self, move: str, side: str, at: float = 0.0, anchor: np.ndarray | None = None) -> None:
        self.move, self.side, self.kick_t, self.pending = move, side, at, None
        self.since_move = 0.0
        if anchor is None:
            support = 0 if side == "r" else 1
            planned = self.kicks.sample(move, side, 0.0)[2][support]
            anchor = unplace_xz(self.frame, self.feet_now()[support]) - planned if self.feet_now is not None else np.zeros(2)
            norm = np.linalg.norm(anchor)
            anchor = anchor * min(1.0, ANCHOR_LIMIT / norm) if norm > 0 else anchor
        self.anchor = np.asarray(anchor, float)

    def reanchor(self) -> np.ndarray:
        """Where the ginga's triangle goes when a kick or an esquiva ends: under the feet as they actually stand.
        A kick follows its support foot up to ANCHOR_LIMIT off its corner, the esquiva's wide step never lands
        back exactly on its corner, and a ginga that keeps stepping toward a
        triangle 0.1–0.4 m away walks off and falls; a capoeirista picks the ginga up from where they are.
        The triangle moves, without turning, so its feet targets at this phase sit where the feet are on
        average. (Turning it to the line between the feet swung the ginga 8–20° after nearly every esquiva:
        the back foot never lands exactly home, and the body drifted off following the turn.)"""
        local = place_xz(self.frame, self.ref.feet_at(self.phase, self.bpm))
        frame = self.frame.copy()
        frame[:2] += self.feet_now().mean(axis=0) - local.mean(axis=0)
        return frame

    def blend(self, t: float) -> float:
        duration = self.kicks.duration(self.move)
        return smoothstep(t / BLEND_IN) * smoothstep((duration - t) / BLEND_OUT)

    def target(self, ahead: float):
        """Reference (qpos, qvel, feet floor points, which feet count) `ahead` seconds from now, in the world."""
        if self.recover is not None:
            return self._recover_target(ahead)
        q, v, feet, mask = self._target(ahead)
        return (*place_pose(self.frame, q, v, feet), mask)

    def _recover_target(self, ahead: float):
        r = self.recover
        t = r["t"] + ahead
        if t >= FALL_TIME:
            q, v, feet, mask = self.floor.sample(r["move"], r["side"], t - FALL_TIME)
            return (*place_pose(r["place"], q, v, feet), mask)
        # Going down: from the pose balance was lost in to the landing, as a person reaches for the floor —
        # fast at first, settling into the landing, but never ahead of gravity: the hips drop no further than
        # they would falling freely from their speed at the start. The arms reach for the floor at once
        # (falling backwards, the hands are behind the hips and near the floor in 0.3 s).
        landing, _, feet, _ = self.floor.sample(r["move"], r["side"], 0.0)
        landing, _, feet = place_pose(r["place"], landing, np.zeros(len(r["onset"]) - 1), feet)
        onset, x = r["onset"], min(t / FALL_TIME, 1.0)
        ease = 1.0 - (1.0 - x) ** 2
        drop = onset[1] - landing[1]
        fallen = max(-r.get("sink", 0.0) * t + 0.5 * GRAVITY * t * t, 0.0)
        b_root = min(ease, fallen / drop) if drop > 0.01 else ease
        b = np.full(len(onset), b_root)
        b[self.arms] = ease
        q = onset + (landing - onset) * b
        q[3:7] = nlerp(onset[3:7], landing[3:7], b_root)
        # Blended joint by joint, the legs can swing through the floor on the way: the pose rises out of it.
        if self.sinking is not None:
            q[1] += self.sinking(q)
        return q, np.zeros(len(q) - 1), feet, np.zeros(2)

    def _target(self, ahead: float):
        """The ginga or kick in the triangle's own frame."""
        if self.move is None:
            phase = self.phase + ahead / self.ref.period(self.bpm)
            q, v = self.ref.sample(phase, self.bpm)
            return q, v, self.ref.feet_at(phase, self.bpm), np.ones(2)
        t = self.kick_t + ahead
        g, gv = self.ref.sample(self.phase, self.bpm)
        gf = self.ref.feet_at(self.phase, self.bpm)
        k, kv, kf, mask = self.kicks.sample(self.move, self.side, t)
        k[[0, 2]] += self.anchor
        kf = kf + self.anchor
        b = self.blend(t)
        q = g + (k - g) * b
        q[3:7] = nlerp(g[3:7], k[3:7], b)
        # While blending, the ginga's feet still count.
        return q, gv + (kv - gv) * b, gf + (kf - gf) * b, np.where(b < 1, 1.0, mask)

    def advance(self, dt: float) -> None:
        if self.recover is not None:
            self.since_move += dt
            r = self.recover
            r["t"] += dt
            if r["t"] >= FALL_TIME + self.floor.duration(r["move"], r["side"]):
                # Up in the base: the ginga carries on from here, its triangle where the body stands.
                self.frame = r["place"].copy()
                self.phase = self.floor.clips[r["move"], r["side"]]["phase"]
                self.recover = None
            return
        if self.move is not None:
            self.since_move = 0.0
            self.kick_t += dt
            if self.kick_t >= self.kicks.duration(self.move):
                if self.feet_now is not None:
                    self.frame = self.reanchor()
                self.move = None
            return
        self.since_move += dt
        step = dt / self.ref.period(self.bpm)
        if self.pending is not None:
            move, side, entry = self.pending
            if (entry - self.phase) % 1.0 <= step:
                self.phase = entry
                self.start(move, side)
                return
        self.phase = (self.phase + step) % 1.0

    def features(self) -> list[float]:
        """What the policy is told about the move: which kick, which leg, how far in, how blended;
        then whether it is falling or getting up, into which landing, on which side, how far along;
        then whether the move is an esquiva or one of the newer moves (appended later: older networks read the inputs before it)."""
        if self.recover is not None:
            r = self.recover
            total = FALL_TIME + self.floor.duration(r["move"], r["side"])
            return [0.0] * 5 + [1.0, float(r["move"] == "quatro"), 1.0 if r["side"] == "r" else -1.0, r["t"] / total, float(r["move"] == "joelhos")] + [0.0] + [0.0] * len(EXTRA_MOVES)
        return self._kick_features() + [0.0] * 5 + [float(self.move == "esquiva")] + [float(self.move == m) for m in EXTRA_MOVES]

    def _kick_features(self) -> list[float]:
        if self.move is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        duration = self.kicks.duration(self.move)
        return [
            float(self.move == "martelo"),
            float(self.move == "bencao"),
            1.0 if self.side == "r" else -1.0,
            self.kick_t / duration,
            self.blend(self.kick_t),
        ]


def lift_out_of_floor(model: mujoco.MjModel, data: mujoco.MjData, floor: int) -> float:
    """How far to raise a pose so nothing sinks into the floor (the reference grounds the soles' centre
    lines, but a rolled foot's box corner can sit a centimetre deep, and a deep start launches the body)."""
    mujoco.mj_forward(model, data)
    deepest = min((c.dist for c in data.contact[: data.ncon] if floor in (c.geom1, c.geom2)), default=0.0)
    return max(-deepest, 0.0) + 0.002


class GingaEnv(gym.Env):
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

    # ---- gym API -------------------------------------------------------
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

    def _reward(self) -> tuple[float, dict]:
        q_ref, v_ref, feet_ref, feet_mask = self.choreo.target(0.0)
        q, v = self.d.qpos, self.d.qvel
        joint_err = np.sum((q[self.joint_qpos] - q_ref[self.joint_qpos]) ** 2)
        root_rot_err = quat_angle(q[3:7], q_ref[3:7]) ** 2
        vel_err = np.sum((v[self.joint_qvel] - v_ref[self.joint_qvel]) ** 2)
        self.ref_data.qpos[:] = q_ref
        mujoco.mj_kinematics(self.m, self.ref_data)
        ee_err = np.sum((self._local_effectors(self.d) - self._local_effectors(self.ref_data)) ** 2)
        r = self.choreo.recover
        root_err = (q[1] - q_ref[1]) ** 2 if (r is not None and r["t"] < FALL_TIME) else np.sum((q[:3] - q_ref[:3]) ** 2)
        feet_err = np.sum(feet_mask[:, None] * (self._foot_points() - feet_ref) ** 2)
        parts = {
            "pose": np.exp(-2.0 * (joint_err + root_rot_err)),
            "vel": np.exp(-0.02 * vel_err),
            "ee": np.exp(-40.0 * ee_err / len(EFFECTORS)),
            "root": np.exp(-10.0 * root_err),
            "feet": np.exp(-20.0 * feet_err),
        }
        parts["kicking"] = float(self.choreo.move is not None)  # logged only: share of time spent kicking
        parts["down"] = float(self.choreo.recover is not None)  # logged only: share of time falling or getting up
        parts["up"] = float(self.after_recover is not None)  # logged only: share of time back in the ginga after getting up
        parts["size"] = float(self.fall_level if self.skill == "recover" else self.curriculum)  # logged only: curriculum level
        reward = 0.4 * parts["pose"] + 0.1 * parts["vel"] + 0.15 * parts["ee"] + 0.15 * parts["root"] + 0.2 * parts["feet"]
        parts["strike"] = parts["pivot"] = 0.0  # logged as their means while kicking
        if self.choreo.move in KICK_MOVES:
            # Where the kick goes: the kicking foot from the hips as the room sees it. ee turns everything into
            # the body's own heading, so a martelo thrown out to the side without turning the hips over (and
            # pivoting on the support foot) still scored there; this only pays for the foot where the reference's is.
            foot = self.effector_ids[EFFECTORS.index(f"foot_{self.choreo.side}")]
            reach_err = np.sum(((self.d.xpos[foot] - q[:3]) - (self.ref_data.xpos[foot] - q_ref[:3])) ** 2)
            parts["strike"] = np.exp(-STRIKE_SCALE * reach_err)
            # The support foot turning with the hips (the martelo's pivot on the ball of the foot), as the
            # reference's heel-to-toe line: the feet term only places the foot's middle, and the body pivoted
            # the other way.
            support = "l" if self.choreo.side == "r" else "r"
            heel, toe = self.sole_sites["lr".index(support)]
            heading = lambda data: np.arctan2(*(data.site_xpos[toe] - data.site_xpos[heel])[[0, 2]])
            turn_err = (heading(self.d) - heading(self.ref_data) + np.pi) % (2 * np.pi) - np.pi
            parts["pivot"] = np.exp(-PIVOT_SCALE * turn_err**2)
            reward = (1 - STRIKE_WEIGHT - PIVOT_WEIGHT) * reward + STRIKE_WEIGHT * parts["strike"] + PIVOT_WEIGHT * parts["pivot"]
        if r is not None and r["t"] < FALL_TIME + 0.3:
            # Going down softly: hips, chest and head lowered, not dropped, and angular velocity damped.
            speed = np.zeros(6)
            drop = 0.05 * float(np.sum(v[3:6] ** 2))
            for body in self.soft_bodies:
                mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, body, speed, 0)
                drop += min(speed[4], 0.0) ** 2
            parts["soft"] = np.exp(-0.5 * drop)
            # Hips first, head last: diving head first over the hips is how the head hits the floor.
            head, hips = self.d.xpos[self.soft_bodies[2]][1], self.d.xpos[self.soft_bodies[0]][1]
            parts["head"] = np.exp(-10.0 * max(hips + HEAD_ABOVE - head, 0.0) ** 2)
            reward = 0.75 * reward + 0.1 * parts["soft"] + 0.15 * parts["head"]
        return float(reward), parts
