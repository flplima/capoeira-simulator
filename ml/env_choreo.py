"""Choreography: what the body should be doing now (ginga, kick, esquiva, fall, way up) and the reference for it."""

import numpy as np

from env_clips import Floor, Kicks, Reference
from env_consts import ANCHOR_LIMIT, BENT, BLEND_IN, BLEND_OUT, EXTRA_MOVES, FALL_TIME, GRAVITY
from env_math import nlerp, place_pose, place_xz, smoothstep, unplace_xz, yaw_of


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
