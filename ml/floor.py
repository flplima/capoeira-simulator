"""Reference motions for falling and getting back up: queda de quatro, negativa and hands and knees.

    npx --yes tsx@4.19.2 scripts/sample-clips.ts queda-de-quatro negativa > ml/data/floor_clips.json   # from the web project
    uv run python floor.py                                                                               # → data/floor.npz

A capoeirista who loses balance doesn't crash: falling backwards they sit
into the queda de quatro (hands and feet on the floor, belly up), falling
sideways they drop into the negativa (one leg folded under, the other
stretched out, a hand on the floor), and falling forwards they catch
themselves on their hands and knees. The landings are the poses of the
game's keyframes in public/data/movements (hands and knees, not a move of
the game, is keyed here). The ways up are keyed here too (UP), so that the
body never has to throw itself up from the floor: every hand or foot lets go
only once the weight is over what stays down. From the queda de quatro the
hands push the hips forward over the feet into a low squat (the cocorinha's)
and the body rises from there; from the negativa the stretched leg drags its
heel in and plants the foot in front, the hand pushes the weight onto it,
and the body rises from a low lunge with the folded leg back; from hands and
knees one foot steps forward beside its hand, the hips move over it, the
hands push off into a low lunge, and the body rises with the other foot
back. Each ends in the ginga's base, and the ginga carries on from there.

So each floor move gives two pieces here: the landing pose (the keyframe
held on the floor), which the body aims for while it falls, and the way
up, from that pose to the ginga's base. Inverse kinematics puts them on
the floor: the hands and feet that the keyframes rest on touch it and
don't slide while they bear weight (but a dragged heel), the centre of
mass sits over the points that bear it, nothing sinks into the floor, no
thigh passes through the trunk, and the last frames melt into the ginga's
own base pose (the reference at a middle tempo), so the ginga can take
over where the body stands up. The clip is solved from its end backwards:
the end is the base on the triangle, and every earlier frame keeps the
floor points of the frame after it — except a point that lifts before the
end (a hand, a knee), which stays where it touched down in the landing
(see plant). A point takes and gives up its share of the weight gradually,
and a foot stepping to its next spot goes through the air.

A keyframe clip need not be followable by a torque-limited body, so the
solved clip is then paced (stretched wherever the body would rise, travel
or swing a limb too fast), free arms are slowed, and a few frames where the
solve jumps to another solution are blended over.
"""

import argparse
import json

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from env import nlerp
from kicks import GINGA_TEMPO, to_model
from reference import arm_clearance, d, fix_arms, m

# The ways up, as keyframes (time in keyframe-seconds; pose; channel values in degrees over it), for the
# right-side version (the left one mirrors it). A pose is "landing" (the source clip's pose held on the
# floor), "parallel" or "base_r"/"base_l" (from the queda de quatro's keyframes), "cocorinha" (the deep
# squat of public/data/movements/cocorinha.yaml, on both flat feet), or "hold" (the keyframe before).
# Every lift of a hand or a foot happens with the weight already over what stays down: nothing has to be
# thrown up from the floor, which a torque-limited body can't do.
COCORINHA = {"root.pitch": 5, "torso.flex": 45, "torso.side": 0, "torso.twist": 0, "head.flex": -35,
             "hip_l.flex": 123, "hip_l.abd": 20, "hip_l.rot": 0, "knee_l.flex": 150, "ankle_l.flex": -32,
             "hip_r.flex": 123, "hip_r.abd": 20, "hip_r.rot": 0, "knee_r.flex": 150, "ankle_r.flex": -32}
# Arms low and open in the squat, the hands outside the knees (hanging between them they pass through the legs).
LOW_ARMS = {"shoulder_l.flex": 40, "shoulder_l.abd": 50, "shoulder_l.rot": 0, "elbow_l.flex": 15,
            "shoulder_r.flex": 40, "shoulder_r.abd": 50, "shoulder_r.rot": 0, "elbow_r.flex": 15}
UP = {
    # Queda de quatro: the hands push the hips forward over the feet, into a low squat (the weight on the
    # feet before the hands leave the floor), and the body rises into the base from there.
    "quatro": [
        (0.0, "landing", {}),
        (0.15, "hold", {}),
        (0.6, "landing", {"root.pitch": -25, "torso.flex": 25, "head.flex": 10,
                          "hip_l.flex": 95, "knee_l.flex": 130, "ankle_l.flex": -15,
                          "hip_r.flex": 95, "knee_r.flex": 130, "ankle_r.flex": -15,
                          "shoulder_l.flex": -55, "elbow_l.flex": 15, "shoulder_r.flex": -55, "elbow_r.flex": 15}),
        (1.0, "cocorinha", LOW_ARMS),
        (1.3, "hold", {}),
        (1.75, "parallel", {}),
        (2.15, "base_r", {}),
    ],
    # Negativa: the stretched leg comes in and its foot plants in front, the hand still on the floor; the
    # weight moves onto that foot as the hand pushes off, into a low lunge over the folded leg's ball of the
    # foot, and the body rises into the base with that leg back (the other base from the queda de quatro's).
    "negativa": [
        (0.0, "landing", {}),
        (0.15, "hold", {}),
        (0.6, "landing", {"root.roll": -8, "torso.flex": 15, "torso.side": -8, "torso.twist": -10,
                          "hip_r.flex": 105, "hip_r.abd": 18, "knee_r.flex": 120, "ankle_r.flex": -25}),
        (1.0, "parallel", {"root.pitch": 25, "torso.flex": 15, "head.flex": -35,
                           "hip_r.flex": 100, "hip_r.abd": 15, "hip_r.rot": 0, "knee_r.flex": 115, "ankle_r.flex": -30,
                           "hip_l.flex": 55, "hip_l.abd": 12, "hip_l.rot": 5, "knee_l.flex": 125, "ankle_l.flex": 20}),
        (1.45, "base_l", {}),
    ],
    # Hands and knees: on all fours, one foot steps forward beside its hand, the hips move over it, the
    # hands push off into a low lunge, and the body rises into the base with the other foot back.
    "joelhos": [
        (0.0, "parallel", {"root.pitch": 62, "torso.flex": 12, "head.flex": -45,
                           "shoulder_l.flex": 75, "shoulder_l.abd": 12, "shoulder_l.rot": 0, "elbow_l.flex": 5,
                           "shoulder_r.flex": 75, "shoulder_r.abd": 12, "shoulder_r.rot": 0, "elbow_r.flex": 5,
                           "hip_l.flex": 70, "hip_l.abd": 10, "hip_l.rot": 0, "knee_l.flex": 100, "ankle_l.flex": 40,
                           "hip_r.flex": 70, "hip_r.abd": 10, "hip_r.rot": 0, "knee_r.flex": 100, "ankle_r.flex": 40}),
        (0.15, "hold", {}),
        # the left foot comes forward through the air, on its own side, and plants beside the left hand
        (0.5, "parallel", {"root.pitch": 55, "torso.flex": 15, "head.flex": -40,
                           "shoulder_l.flex": 75, "shoulder_l.abd": 12, "shoulder_l.rot": 0, "elbow_l.flex": 15,
                           "shoulder_r.flex": 75, "shoulder_r.abd": 12, "shoulder_r.rot": 0, "elbow_r.flex": 10,
                           "hip_l.flex": 120, "hip_l.abd": 18, "hip_l.rot": 0, "knee_l.flex": 120, "ankle_l.flex": -15,
                           "hip_r.flex": 65, "hip_r.abd": 10, "hip_r.rot": 0, "knee_r.flex": 100, "ankle_r.flex": 40}),
        # hands still down, the hips move forward over the front foot
        (0.75, "parallel", {"root.pitch": 45, "torso.flex": 18, "head.flex": -35,
                            "shoulder_l.flex": 70, "shoulder_l.abd": 12, "shoulder_l.rot": 0, "elbow_l.flex": 20,
                            "shoulder_r.flex": 70, "shoulder_r.abd": 12, "shoulder_r.rot": 0, "elbow_r.flex": 20,
                            "hip_l.flex": 110, "hip_l.abd": 15, "hip_l.rot": 0, "knee_l.flex": 105, "ankle_l.flex": -30,
                            "hip_r.flex": 35, "hip_r.abd": 10, "hip_r.rot": 5, "knee_r.flex": 100, "ankle_r.flex": 35}),
        # the hands push off, the back knee leaves the floor: a low lunge
        (1.05, "parallel", {"root.pitch": 25, "torso.flex": 22, "head.flex": -30,
                            "hip_l.flex": 95, "hip_l.abd": 15, "hip_l.rot": 5, "knee_l.flex": 105, "ankle_l.flex": -25,
                            "hip_r.flex": 10, "hip_r.abd": 12, "hip_r.rot": 10, "knee_r.flex": 85, "ankle_r.flex": -20}),
        (1.45, "base_r", {}),
    ],
}
# Where each landing comes from: the source clip and the keyframe time its pose is held from (see UP).
SOURCE = {"quatro": ("queda-de-quatro", 1.25), "negativa": ("negativa", 1.2), "joelhos": None}
MOVES = list(UP)
# The way up plays this much slower than the keyframes: standing up from the floor
# is the heaviest thing the body does, and its joints are torque-limited.
SLOW = 2.0
# The last keyframe-seconds melt into the ginga's base.
TO_GINGA = 0.35
# A point that starts or stops bearing weight takes or gives its share of it over this many keyframe-seconds
# (the centre of mass moves over the new support gradually, instead of jumping when a hand lets go).
HANDOVER = 0.3
# No faster than this (m/s, real time): the centre of mass rising, and moving along the floor. Where the
# keyframes go faster, the clip is stretched (a torque-limited body can't stand up from the floor in a jump).
RISE_SPEED, SHIFT_SPEED = 0.8, 1.0
LIMB_SPEED = 2.5  # m/s: nor any hand, knee or foot faster than this
# Which points rest on the floor, as keyframe-time windows, for the right-side
# version (the left one swaps l and r). Sites: heel and toe at each foot, the fingers at each hand.
HAND = {"l": ["fingers_l"], "r": ["fingers_r"]}  # the hand has no wrist joint: it rests on the fingers
FOOT = {"l": ["heel_l", "toe_l"], "r": ["heel_r", "toe_r"]}
DOWN = {
    "quatro": [
        (HAND["l"] + HAND["r"], 0.0, 1.2),  # the hands push until the hips are over the feet, in the squat
        (FOOT["l"] + ["toe_r"], 0.0, 9.0),
        (["heel_r"], 0.0, 1.85),  # the back foot's heel lifts as the body rises into the base
    ],
    "negativa": [
        (["toe_l"], 0.0, 9.0),  # the folded leg, on the ball of the foot
        (["heel_r"], 0.0, 0.55),  # the stretched leg, on its heel, dragged in (SLIDE)…
        (FOOT["r"], 0.55, 9.0),  # …and planted flat in front
        (HAND["r"], 0.0, 1.0),  # the hand pushes until the weight is over the feet
    ],
    "joelhos": [
        (HAND["l"] + HAND["r"], 0.0, 0.9),  # the hands push until the hips are over the front foot
        (["knee_r", "toe_r"], 0.0, 0.95),  # the back leg kneels on the ball of its foot…
        (["toe_r"], 0.95, 9.0),  # …then rises onto it
        (["knee_l", "toe_l"], 0.0, 0.2),  # the front leg steps forward…
        (["heel_l", "toe_l"], 0.5, 9.0),  # …and plants beside the hand
    ],
}
# Points that stay on the floor but slide along it, as keyframe-time windows (right-side version): the
# negativa's stretched leg drags its heel in rather than lifting it (lifting it leaves the body on a hand
# and a ball of the foot). A sliding point bears only SLIDE_SHARE of a planted point's weight.
SLIDE = {"negativa": [("heel_r", 0.0, 0.55)]}
SLIDE_SHARE = 0.3
# A kneeling knee rests on the shin's capsule, this far below the knee joint's site.
ABOVE_FLOOR = {"knee_l": 0.054, "knee_r": 0.054}
# The ginga's base keeps the back foot's heel up: at the very end only these bear weight.
BASE_DOWN = {"r": ["heel_l", "toe_l", "toe_r"], "l": ["heel_r", "toe_r", "toe_l"]}


def swap(name: str) -> str:
    return name[:-1] + {"l": "r", "r": "l"}[name[-1]]


def ends_in(move: str, side: str) -> str:
    """Which base ("r": right foot back, "l": left) the way up of this landing and side ends in."""
    base = UP[move][-1][1][-1]
    return base if side == "r" else swap(base)


def down_at(move: str, side: str, t: float, end: float) -> dict[str, tuple[float, float]]:
    """The points on the floor at keyframe time t, each with how much of its share of the weight it has
    taken (fading in over HANDOVER after it touches down) and how much it still holds (fading out over
    HANDOVER before it lifts)."""
    if t >= end - 1e-6:
        return {n: (1.0, 1.0) for n in BASE_DOWN[ends_in(move, side)]}
    out: dict[str, tuple[float, float]] = {}
    windows = [(n, a, b) for sites, a, b in DOWN[move] for n in sites]
    for sites, a, b in DOWN[move]:
        if a <= t < b:
            for n in sites:
                # (a point that stays down from one window into the next doesn't touch down or lift in between)
                taken = min(1.0, (t - a) / HANDOVER) if a > 0 and not any(a2 < a <= b2 for n2, a2, b2 in windows if n2 == n) else 1.0
                held = min(1.0, (b - t) / HANDOVER) if b < end and not any(a2 <= b < b2 for n2, a2, b2 in windows if n2 == n) else 1.0
                out[n if side == "r" else swap(n)] = (max(taken, 0.0), max(held, 0.0))
    return out


def plant(
    qpos: np.ndarray, times: np.ndarray, move: str, side: str, end: float, pins: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the clip onto the floor from its last frame backwards; returns qpos and, per frame, which
    sites bear weight (as a mask over SITES).

    The points still down at the end (the feet in the base) are pinned where the base puts them. A point
    that bears weight for a while and then lifts (a hand) stays where it first touched — the landing
    decides that, not the way up — so a first solve leaves those points free along the floor, and the
    second pins each where the first one put it when it touched down. The first solve is smoothed and
    becomes the pose the second one stays close to (smoothing would slide the points)."""
    first = pins is None
    joints = np.arange(7, m.nq)
    lo = np.r_[-np.inf, 0.05, -np.inf, [-np.pi] * 3, [m.jnt_range[m.jnt_qposadr.tolist().index(a)][0] for a in joints]]
    hi = np.r_[np.inf, 1.3, np.inf, [np.pi] * 3, [m.jnt_range[m.jnt_qposadr.tolist().index(a)][1] for a in joints]]
    site_ids = {m.site(i).name: i for i in range(m.nsite)}
    out = qpos.copy()
    mask = np.zeros((len(qpos), len(SITES)))
    floor_xz: dict[str, np.ndarray] = {}
    downs = [down_at(move, side, t, end) for t in times]
    # Per frame and point: the frame its stretch on the floor begins at, or None if that stretch lasts to the end;
    # and between two stretches, the step through the air: (point, frame it lifts after, frame it lands on).
    touched = [{} for _ in times]
    swings = [{} for _ in times]
    for n in SITES:
        stretch = []
        k = 0
        while k < len(times):
            if n not in downs[k]:
                k += 1
                continue
            e = k
            while e + 1 < len(times) and n in downs[e + 1]:
                e += 1
            for j in range(k, e + 1):
                touched[j][n] = None if e == len(times) - 1 else k
            stretch.append((k, e))
            k = e + 1
        for (s0, e0), (s1, _) in zip(stretch, stretch[1:]):
            for j in range(e0 + 1, s1):
                swings[j][n] = (s0, e0, s1)
    landed: dict[str, np.ndarray] = {}  # where each point's next stretch on the floor begins (solving backwards)
    for k in range(len(qpos) - 1, -1, -1):
        q = qpos[k]
        share = downs[k]
        names = list(share)
        mask[k] = [n in names and min(share[n]) >= 0.25 for n in SITES]
        if k == len(qpos) - 1:
            # The ginga's base itself, already on the triangle.
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            floor_xz = {n: d.site_xpos[site_ids[n]][[0, 2]].copy() for n in names}
            continue
        sliding = {n if side == "r" else swap(n) for n, a, b in SLIDE.get(move, []) if a <= times[k] < b}
        floor_xz = {n: p for n, p in floor_xz.items() if n in names and n not in sliding}
        if not first:
            floor_xz.update({n: pins[n, touched[k][n]] for n in names if touched[k][n] is not None and n not in sliding})
        # A point stepping from one place on the floor to the next goes through the air, clear of the floor.
        steps = {}
        if not first:
            for n, (s0, e0, s1) in swings[k].items():
                a = smoothstep((k - e0) / (s1 - e0))
                steps[n] = (pins[n, s0] + (landed[n] - pins[n, s0]) * a, ABOVE_FLOOR.get(n, 0.0) + STEP_HEIGHT * np.sin(np.pi * (k - e0) / (s1 - e0)))

        def pose(v: np.ndarray) -> None:
            d.qpos[:] = q
            d.qpos[:3] = v[:3]
            turn = np.zeros(4)
            angle = np.linalg.norm(v[3:6])
            mujoco.mju_axisAngle2Quat(turn, v[3:6] / (angle + 1e-12), angle)
            mujoco.mju_mulQuat(d.qpos[3:7], turn, q[3:7])
            d.qpos[joints] = v[6:]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)

        def residuals(v: np.ndarray) -> np.ndarray:
            pose(v)
            r = []
            for n in names:
                p = d.site_xpos[site_ids[n]]
                # A point on the floor doesn't slide; one handing its weight over is free to lift off.
                held = share[n][1]
                r.append([10.0 * held * (p[1] - ABOVE_FLOOR.get(n, 0.0))])
                if n in floor_xz:
                    r.append(100.0 * held * (p[[0, 2]] - floor_xz[n]))
            for n, (xz, y) in steps.items():
                p = d.site_xpos[site_ids[n]]
                r.append(20.0 * np.r_[p[[0, 2]] - xz, p[1] - y])
            # Nothing below the floor: the sites, the soles' corners and the shins.
            r.append(10.0 * np.minimum(d.site_xpos[:, 1] - CLEAR, 0.0))
            r.append(10.0 * np.minimum(below_floor(), 0.0))
            # No thigh through the trunk (folding a leg in front of a bent trunk).
            r.append(20.0 * np.minimum([mujoco.mj_geomDistance(m, d, a, b, 0.1, FROMTO) for a, b in TRUNK_THIGH], 0.0))
            # Never lying down, nor sitting: the head, chest and hips stay well up.
            r.append(np.array([10.0, 10.0, 40.0]) * np.minimum(d.xpos[[HEAD, CHEST, HIPS], 1] - np.array([0.45, 0.3, 0.25]), 0.0))
            # The weight over what carries it.
            weights = [(SLIDE_SHARE if n in sliding else 1.0) * min(share[n]) + 1e-3 for n in names]
            com = d.subtree_com[1][[0, 2]]
            r.append(15.0 * (com - np.average([d.site_xpos[site_ids[n]][[0, 2]] for n in names], axis=0, weights=weights)))
            # Before a point lets go, the weight is already over what stays down: it is never thrown up from the floor.
            staying = [n for n in names if min(share[n]) >= 1.0 and n not in sliding]
            leaving = min([share[n][1] for n in names], default=1.0)
            if staying and leaving < 0.5:
                r.append(40.0 * (1.0 - 2.0 * leaving) * outside(com, np.array([d.site_xpos[site_ids[n]][[0, 2]] for n in staying])))
            return np.concatenate([np.ravel(a) for a in r] + [0.5 * v[3:6], REGULAR * (v[6:] - q[joints])])

        # Start from the keyframe pose itself, grounded, and shifted onto the floor points it keeps.
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        start = np.r_[q[:3], 0, 0, 0, q[joints]]
        start[1] -= (d.site_xpos[:, 1] - CLEAR).min()
        kept = [n for n in names if n in floor_xz]
        if kept:
            start[[0, 2]] += np.mean([floor_xz[n] - d.site_xpos[site_ids[n]][[0, 2]] for n in kept], axis=0)
        x = least_squares(residuals, np.clip(start, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=1000).x
        pose(x)
        out[k] = d.qpos
        for n in names:
            if not (first and touched[k][n] is not None) and n not in sliding:
                floor_xz.setdefault(n, d.site_xpos[site_ids[n]][[0, 2]].copy())
            landed[n] = floor_xz.get(n, d.site_xpos[site_ids[n]][[0, 2]].copy())
    if first:
        # Where each lifting point touched down, then light smoothing, then the second solve (see above).
        found = {}
        for k in range(len(out)):
            d.qpos[:] = out[k]
            mujoco.mj_kinematics(m, d)
            for n, start in touched[k].items():
                if start == k:
                    found[n, k] = d.site_xpos[site_ids[n]][[0, 2]].copy()
        keep = out[-1].copy()
        out[:, :3] = gaussian_filter1d(out[:, :3], 1.0, axis=0, mode="nearest")
        out[:, 7:] = gaussian_filter1d(out[:, 7:], 1.0, axis=0, mode="nearest")
        out[-1] = keep
        return plant(out, times, move, side, end, found)
    # A frame the solver put on another solution than both its neighbours (a one-frame flip) takes their average.
    for k in range(1, len(out) - 1):
        gap = np.abs(out[k - 1, 7:] - out[k + 1, 7:]).max()
        if np.abs(out[k, 7:] - (out[k - 1, 7:] + out[k + 1, 7:]) / 2).max() > max(0.3, 2 * gap):
            quat = nlerp(out[k - 1, 3:7], out[k + 1, 3:7], 0.5)
            out[k] = (out[k - 1] + out[k + 1]) / 2
            out[k, 3:7] = quat
    # Nothing below the floor.
    for q in out[:-1]:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        q[1] -= min((d.site_xpos[:, 1] - CLEAR).min(), 0.0)
    return out, mask


SITES = [m.site(i).name for i in range(m.nsite)]
STEP_HEIGHT = 0.06  # m: how high a foot or knee clears the floor stepping to its next spot
SOLES = [m.geom(f"foot_{s}").id for s in "lr"]
SHINS = [m.geom(f"shin_{s}").id for s in "lr"]
HEAD, CHEST, HIPS = m.body("head").id, m.body("chest").id, m.body("hips").id
TRUNK_THIGH = [(m.geom(f"thigh_{s}").id, m.geom(t).id) for s in "lr" for t in ("spine", "chest")]
FROMTO = np.zeros(6)
# How firmly each joint keeps to the keyframes: the trunk and neck more (bending them is how a solve
# cheats its way to the balance it's asked for, folding the body in two), the limbs less.
REGULAR = np.array([0.6 if m.joint(i).name.startswith(("spine", "chest", "neck")) else 0.3 for i in range(1, m.njnt)])


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def outside(p: np.ndarray, points: np.ndarray) -> float:
    """How far the floor point p is outside the convex hull of points (both x, z; 0 inside)."""
    def to_segment(a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        t = np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
        return float(np.linalg.norm(p - a - t * ab))

    if len(points) < 3:
        return to_segment(points[0], points[-1])
    middle = points.mean(axis=0)
    ring = points[np.argsort(np.arctan2(*(points - middle).T[::-1]))]
    # (with a point inside the others' hull the ring still bounds the hull: such a point only adds an inward notch,
    # which can only overstate how far p is out — fine for a penalty that is 0 well inside)
    edges = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    inside = all((b - a)[0] * (p - a)[1] - (b - a)[1] * (p - a)[0] >= 0 for a, b in edges)
    return 0.0 if inside else min(to_segment(a, b) for a, b in edges)


def below_floor() -> np.ndarray:
    """Heights above the floor of each foot box's corners and of each shin capsule's surface at its ends
    (after mj_kinematics on d)."""
    out = []
    for g in SOLES:
        half = m.geom_size[g]
        corners = np.array([[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])])
        out.append((d.geom_xpos[g] + corners @ d.geom_xmat[g].reshape(3, 3).T)[:, 1])
    for g in SHINS:
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][1]
        out.append(np.array([(d.geom_xpos[g] + axis)[1], (d.geom_xpos[g] - axis)[1]]) - m.geom_size[g][0])
    return np.concatenate(out)


def pace(qpos: np.ndarray, times: np.ndarray, dt: float) -> np.ndarray:
    """Keyframe times for frames dt apart in real time, stretched wherever the planted clip (qpos, at
    `times`, dt apart) would raise or move its centre of mass faster than RISE_SPEED / SHIFT_SPEED, or a
    hand, knee or foot faster than LIMB_SPEED."""
    com, limbs = [], []
    ends = [SITES.index(n) for n in ("fingers_l", "fingers_r", "knee_l", "knee_r", "heel_l", "heel_r", "toe_l", "toe_r")]
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        com.append(d.subtree_com[1].copy())
        limbs.append(d.site_xpos[ends].copy())
    step = np.diff(np.array(com), axis=0)
    limb = np.linalg.norm(np.diff(np.array(limbs), axis=0), axis=2).max(axis=1)
    spans = np.maximum.reduce([
        np.full(len(step), dt),
        np.maximum(step[:, 1], 0) / RISE_SPEED,
        np.linalg.norm(step[:, [0, 2]], axis=1) / SHIFT_SPEED,
        limb / LIMB_SPEED,
    ])
    real = np.r_[0.0, np.cumsum(spans)]
    grid = np.linspace(0.0, real[-1], int(np.ceil(real[-1] / dt - 1e-6)) + 1)
    return np.interp(grid, real, times)


ARM_SPEED = 5.0  # rad/s: a free arm swings no faster than this (a hand letting go of the floor doesn't snap to the guard)


def calm_arms(qpos: np.ndarray, mask: np.ndarray, dt: float) -> np.ndarray:
    """Limit how fast each arm's joints move while its hand is off the floor: forwards from where the hand
    let go, then backwards from the base at the end."""
    out = qpos.copy()
    step = ARM_SPEED * dt
    for s in "lr":
        joints = [m.joint(n).qposadr[0] for n in (f"shoulder_{s}_flex", f"shoulder_{s}_abd", f"shoulder_{s}_rot", f"elbow_{s}")]
        free = mask[:, SITES.index(f"fingers_{s}")] == 0
        for k in range(1, len(out)):
            if free[k]:
                out[k, joints] = out[k - 1, joints] + np.clip(out[k, joints] - out[k - 1, joints], -step, step)
        for k in range(len(out) - 2, -1, -1):
            if free[k]:
                out[k, joints] = out[k + 1, joints] + np.clip(out[k, joints] - out[k + 1, joints], -step, step)
    return out


def despike(qpos: np.ndarray, dt: float, limit: float = 3.0) -> np.ndarray:
    """Where a hand, knee, foot or the head would move faster than `limit` (m/s) from one frame to the next —
    the solve switching to another solution for a few frames — the frames around it become the blend
    of the two on either side (the shortest stretch that brings it under the limit)."""
    ends = [m.site(n).id for n in ("fingers_l", "fingers_r", "knee_l", "knee_r", "heel_l", "heel_r", "toe_l", "toe_r", "nose")]

    def points(q: np.ndarray) -> np.ndarray:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[ends].copy()

    def blend(out: np.ndarray, a: int, b: int) -> None:
        for k in range(a + 1, b):
            s = smoothstep((k - a) / (b - a))
            quat = nlerp(out[a, 3:7], out[b, 3:7], s)
            out[k] = out[a] + (out[b] - out[a]) * s
            out[k, 3:7] = quat

    out = qpos.copy()
    for _ in range(len(out)):
        pts = np.array([points(q) for q in out])
        speed = np.linalg.norm(np.diff(pts, axis=0), axis=2).max(axis=1) / dt
        k = int(np.argmax(speed))
        if speed[k] <= limit:
            break
        for w in range(1, 8):
            a, b = max(k - w, 0), min(k + 1 + w, len(out) - 1)
            trial = out.copy()
            blend(trial, a, b)
            seg = np.array([points(q) for q in trial[a : b + 1]])
            if np.linalg.norm(np.diff(seg, axis=0), axis=2).max() / dt <= limit or w == 7:
                out = trial
                break
    return out


def resample(qpos: np.ndarray, times: np.ndarray, at: np.ndarray) -> np.ndarray:
    out = []
    for t in at:
        k = min(int(np.searchsorted(times, t, side="right")) - 1, len(qpos) - 2)
        a = float(np.clip((t - times[k]) / (times[k + 1] - times[k]), 0.0, 1.0))
        q = qpos[k] + (qpos[k + 1] - qpos[k]) * a
        q[3:7] = nlerp(qpos[k, 3:7], qpos[k + 1, 3:7], a)
        out.append(q)
    return np.array(out)


CLEAR = np.array([ABOVE_FLOOR.get(n, 0.0) for n in SITES])
NEGATED = {"root.x", "root.yaw", "root.roll", "torso.side", "torso.twist", "head.side", "head.twist"}


def mirror_frame(channels: list[str], frame: list[float]) -> list[float]:
    """The left-side version of a right-side keyframe pose (as mirrorPose in src/data/channels.ts)."""
    value = dict(zip(channels, frame))
    out = []
    for key in channels:
        joint, field = key.split(".")
        other = swap(joint) if joint[-2:] in ("_l", "_r") else joint
        out.append(value[f"{other}.{field}"] * (-1 if key in NEGATED else 1))
    return out


def way_up(clips: dict, move: str) -> dict:
    """The way up (UP) as a sampled clip like the web ones, smoothstepped between keyframes; both sides."""
    channels, fps = clips["channels"], clips["fps"]
    quatro = clips["queda-de-quatro"]["frames"]["r"]
    poses = {"base_r": np.array(quatro[0]), "parallel": np.array(quatro[int(round(0.4 * fps))])}
    poses["base_l"] = np.array(mirror_frame(channels, quatro[0]))
    poses["cocorinha"] = poses["parallel"].copy()
    for k, v in COCORINHA.items():
        poses["cocorinha"][channels.index(k)] = v
    if SOURCE[move] is not None:
        source, hold = SOURCE[move]
        poses["landing"] = np.array(clips[source]["frames"]["r"][int(round(hold * fps))])
    keys = []
    for t, base, values in UP[move]:
        pose = (keys[-1][1] if base == "hold" else poses[base]).copy()
        for k, v in values.items():
            pose[channels.index(k)] = v
        keys.append((t, pose))
    duration = keys[-1][0]
    right = []
    for k in range(int(round(duration * fps)) + 1):
        t = k / fps
        i = max(j for j in range(len(keys)) if keys[j][0] <= t + 1e-9)
        if i == len(keys) - 1:
            right.append(keys[i][1].tolist())
            continue
        (t0, a), (t1, b) = keys[i], keys[i + 1]
        s = (t - t0) / (t1 - t0)
        right.append((a + (b - a) * s * s * (3 - 2 * s)).tolist())
    return {"duration": duration, "frames": {"r": right, "l": [mirror_frame(channels, f) for f in right]}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/floor.npz")
    args = parser.parse_args()
    clips = json.load(open("data/floor_clips.json"))
    channels = clips["channels"]
    fps = clips["fps"]
    ginga = np.load("data/ginga_ref.npz")["qpos"][GINGA_TEMPO]
    out = {}
    for move in MOVES:
        clip = way_up(clips, move)
        for side in "lr":
            end = ends_in(move, side)
            base = ginga[0 if end == "r" else len(ginga) // 2]
            frames = clip["frames"][side]
            times = np.arange(len(frames)) / fps
            qpos = np.array([to_model(dict(zip(channels, f))) for f in frames])
            # Melt into the ginga's base over the last TO_GINGA seconds.
            for k, t in enumerate(times):
                s = np.clip(1 - (clip["duration"] - t) / TO_GINGA, 0.0, 1.0)
                s = s * s * (3 - 2 * s)
                quat = nlerp(qpos[k, 3:7], base[3:7], s)
                qpos[k] = qpos[k] + (base - qpos[k]) * s
                qpos[k, 3:7] = quat
            qpos[-1] = base
            # Plant once to see how fast the keyframes go, then again at the pace a body can follow (twice:
            # the pace changes the solve a little).
            at = times
            for _ in range(2):
                at = pace(plant(resample(qpos, times, at), at, move, side, clip["duration"])[0], at, SLOW / fps)
            qpos, mask = plant(resample(qpos, times, at), at, move, side, clip["duration"])
            # The guard arm clears the head; arms holding the body up stay where they are (easing between the two).
            hands = mask[:, [SITES.index(n) for n in HAND["l"] + HAND["r"]]].any(axis=1)
            keep = np.clip(gaussian_filter1d(hands.astype(float), 3.0, mode="nearest") * 2, 0.0, 1.0)[:, None]
            qpos = keep * qpos + (1 - keep) * fix_arms(qpos.copy(), periodic=False)
            qpos = despike(calm_arms(qpos, mask, SLOW / fps), SLOW / fps)
            qpos[-1] = base
            d.qpos[:] = qpos[0]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            duration = (len(qpos) - 1) / fps * SLOW
            out[f"{move}_{side}_qpos"] = qpos
            out[f"{move}_{side}_mask"] = mask
            out[f"{move}_{side}_duration"] = np.array(duration)
            out[f"{move}_{side}_phase"] = np.array(0.0 if end == "r" else 0.5)
            print(
                f"{move} ({side}): way up {duration:.2f} s; landing hips {qpos[0, 1]:.2f} m, head {d.xpos[m.body('head').id][1]:.2f} m, "
                f"centre of mass {d.subtree_com[1][1]:.2f} m up; arm–head clearance {arm_clearance(qpos) * 100:.1f} cm"
            )
    np.savez(args.out, fps=fps / SLOW, moves=np.array(MOVES), sites=np.array(SITES), **out)


if __name__ == "__main__":
    main()
