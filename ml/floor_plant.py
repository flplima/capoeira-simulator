"""Putting a way up on the floor: inverse kinematics from the clip's last frame backwards."""

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from env_math import nlerp
from floor_keys import ABOVE_FLOOR, SLIDE, SLIDE_SHARE, down_at, swap
from reference import d, m

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
CLEAR = np.array([ABOVE_FLOOR.get(n, 0.0) for n in SITES])


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

