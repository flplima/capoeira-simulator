"""Planting an esquiva on the triangle: inverse kinematics on the keyframed pose."""

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from esquiva_path import ARC, CLEAR, DODGE_SPECS, LIFT_UP, NECK_UP, RELEASE, com_path, drop, smoother
from reference import TRIANGLE, d, m


def plant(
    qpos: np.ndarray,
    side: str,
    amplitude: float,
    fps: float,
    base: np.ndarray,
    spec: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place the esquiva of the back leg `side` on the triangle; returns qpos, the feet's floor targets and which apply."""
    spec = spec or DODGE_SPECS["esquiva"]
    t_lift, t_wide, t_settle, t_rise, t_home, t_back = spec["times"]
    hips_low, head_low, share = spec["hips_low"], spec["head_low"], spec["share"]
    flat_wide = spec.get("flat_wide", 1.0)
    bent = "l" if side == "r" else "r"
    names = [n for s in "lr" for n in (f"hip_{s}_flex", f"hip_{s}_abd", f"hip_{s}_rot", f"knee_{s}", f"ankle_{s}", f"ankle_{s}_inv")]
    names += ["spine_flex", "spine_side", "chest_flex", "chest_side"]
    legs = np.array([m.joint(n).qposadr[0] for n in names])
    tilt = 0.6  # rad: how far the solver may turn the whole body from the keyframe
    lo = np.r_[-np.inf, 0.3, -np.inf, [-tilt] * 3, [m.jnt_range[m.joint(n).id][0] for n in names]]
    hi = np.r_[np.inf, 1.3, np.inf, [tilt] * 3, [m.jnt_range[m.joint(n).id][1] for n in names]]
    sole = {s: [m.site(f"heel_{s}").id, m.site(f"toe_{s}").id] for s in "lr"}
    foot_body = {s: m.body(f"foot_{s}").id for s in "lr"}
    foot_geoms = [m.geom(f"foot_{s}").id for s in "lr"]
    floor = m.geom("floor").id
    off_floor = [m.geom(f"{part}_{s}").id for s in "lr" for part in ("thigh", "shin", "forearm", "hand")]
    front = {"l": TRIANGLE[0], "r": TRIANGLE[1]}
    back = TRIANGLE[2]
    outward = np.sign(front[side][0])
    wide_target = spec["wide"](front, back, side, outward)
    wide = back + amplitude * (wide_target - back)
    n = len(qpos)
    times = np.arange(n) / fps
    end_t = times[-1]
    head = m.body("head").id

    d.qpos[:] = base
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    start, head_start, height = d.subtree_com[1][[0, 2]].copy(), d.xpos[head][1], d.subtree_com[1][1]
    foot = front[bent]
    bottom = foot + amplitude * share * (wide - foot)
    # Down at once, held, and back up: how far into the esquiva the body is (0 in the base, 1 at the bottom).
    depth = amplitude * (base[1] - hips_low)
    low = drop(times, depth) * (1 - drop(times - t_rise, depth, LIFT_UP))
    path = com_path(times, start, start, foot, bottom, 0.9 * height, spec["times"])
    neck = m.joint("neck_flex").qposadr[0]
    # At the ends the pose is the ginga's own, whose back foot's box may sit a few mm in the floor.
    ends = np.array([max(0.0, 1 - min(t, end_t - t) / 0.05) for t in times])
    hips = base[1] + amplitude * low * (hips_low - base[1])
    heads = head_start + amplitude * low * (head_low - head_start)
    head_x = spec["head_x"](foot, outward)
    fromto = np.zeros(6)

    lifts = np.zeros(n)
    out = qpos.copy()
    feet = np.zeros((n, 2, 2))
    mask = np.zeros((n, 2))
    x = None
    for k, (q, t) in enumerate(zip(qpos, times)):
        q = q.copy()
        q[neck] += amplitude * low[k] * NECK_UP
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        posed = {s: d.site_xpos[sole[s]].copy() for s in "lr"}

        def on(spot: np.ndarray, sites: np.ndarray, flat: bool, skim: float = 0.0) -> np.ndarray:
            """The foot's own shape (as posed) with its middle on the spot and its lowest point `skim` off the floor (or flat on it)."""
            middle = sites.mean(axis=0)
            moved = sites + np.array([spot[0] - middle[0], 0, spot[1] - middle[2]])
            if flat:
                moved[:, 1] = 0.0
            else:
                moved[:, 1] -= moved[:, 1].min() - skim
            return moved

        # The back foot: on its corner (on its ball, as in the base), a low step out landing flat on the wide spot,
        # a low step home, on its corner — paths with no jump in acceleration where the foot stops.
        if t < t_wide:
            u = (t - t_lift) / (t_wide - t_lift)
            spot, flat = back + smoother(u) * (wide - back), flat_wide * smoother((u - 0.5) / 0.5)
        else:
            u = (t - t_home) / (t_back - t_home)
            spot, flat = wide + smoother(u) * (back - wide), flat_wide * (1 - smoother(u / 0.5))
        lift = lifts[k] = ARC * np.sin(np.pi * min(max(u, 0.0), 1.0))
        resting = t < t_lift or t_wide + RELEASE <= t < t_home or t >= t_back + RELEASE
        moving = on(spot, posed[side], False, lift)
        moving[:, 1] = (1 - flat) * moving[:, 1] + flat * lift
        targets = {bent: on(front[bent], posed[bent], True), side: moving}
        feet[k, "lr".index(bent)] = front[bent]
        feet[k, "lr".index(side)] = spot
        mask[k, "lr".index(bent)] = 1.0
        mask[k, "lr".index(side)] = 1.0 if resting else 0.0
        flats = {bent: 1.0, side: flat}

        def pose(v: np.ndarray) -> None:
            d.qpos[:] = q
            d.qpos[:3] = v[:3]
            turn = np.zeros(4)
            mujoco.mju_axisAngle2Quat(turn, v[3:6] / (np.linalg.norm(v[3:6]) + 1e-12), np.linalg.norm(v[3:6]))
            mujoco.mju_mulQuat(d.qpos[3:7], turn, q[3:7])
            d.qpos[legs] = v[6:]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)

        def residuals(v: np.ndarray) -> np.ndarray:
            pose(v)
            r = [10.0 * (d.site_xpos[sole[s]] - targets[s]).ravel() for s in "lr"]
            # Flat soles: the foot's own up axis vertical, so the box doesn't sit on its edge.
            r += [5.0 * w * (d.xmat[foot_body[s]].reshape(3, 3)[:, 1] - [0.0, 1.0, 0.0]) for s, w in flats.items()]
            r.append([10.0 * min(mujoco.mj_geomDistance(m, d, g, floor, 0.2, fromto) - CLEAR, 0.0) for g in off_floor])
            # The feet on the floor, not in it (a rolled foot's box corner would sink).
            r.append([40.0 * min(mujoco.mj_geomDistance(m, d, g, floor, 0.2, fromto) + 0.008 * ends[k], 0.0) for g in foot_geoms])
            r.append([40.0 * min(mujoco.mj_geomDistance(m, d, foot_geoms["lr".index(side)], floor, 0.2, fromto) - 0.8 * lift, 0.0)])
            r.append(15.0 * (d.subtree_com[1][[0, 2]] - path[k]))
            r.append([10.0 * (v[1] - hips[k]), 6.0 * (d.xpos[head][1] - heads[k]), 8.0 * low[k] * amplitude * (d.xpos[head][0] - head_x)])
            # At the very ends the legs are the ginga's own (the hand-over has no step).
            return np.concatenate(r + [2.0 * v[3:6], (0.3 + 5.0 * ends[k]) * (v[6:] - q[legs])])

        guess = np.r_[q[:3], 0, 0, 0, q[legs]] if x is None else np.r_[x[:6], q[legs]]
        guess[1] = np.clip(guess[1], 0.31, 1.29)
        x = least_squares(residuals, np.clip(guess, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=400).x
        pose(x)
        out[k] = d.qpos
    columns = np.r_[0, 1, 2, 3, 4, 5, 6, legs]
    out[:, columns] = gaussian_filter1d(out[:, columns], 2.0, axis=0, mode="nearest")
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    for i, q in enumerate(out):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        q[1] -= min(d.site_xpos[j][1] for j in sole[bent])
        # Smoothing can press the sliding foot into the floor: lift it out with that leg alone.
        mujoco.mj_forward(m, d)
        # Smoothing can also press a stepping foot back onto the floor: it keeps at least 0.6 of its step height.
        clear = 0.6 * lifts[i] - 0.002 - 0.008 * ends[i]
        if mujoco.mj_geomDistance(m, d, foot_geoms["lr".index(side)], floor, 0.2, fromto) < clear:
            leg = np.array([m.joint(f"{j}_{side}{a}").qposadr[0] for j, a in (("hip", "_flex"), ("hip", "_abd"), ("hip", "_rot"), ("knee", ""), ("ankle", ""), ("ankle", "_inv"))])
            start = q[leg].copy()

            def lifted(v: np.ndarray) -> np.ndarray:
                d.qpos[:] = q
                d.qpos[leg] = v
                mujoco.mj_kinematics(m, d)
                gap = mujoco.mj_geomDistance(m, d, foot_geoms["lr".index(side)], floor, 0.2, fromto)
                return np.r_[40.0 * min(gap - clear, 0.0), v - start]

            bounds = (m.jnt_range[[m.jnt_qposadr.tolist().index(a) for a in leg], 0], m.jnt_range[[m.jnt_qposadr.tolist().index(a) for a in leg], 1])
            q[leg] = least_squares(lifted, np.clip(start, bounds[0] + 1e-6, bounds[1] - 1e-6), bounds=bounds).x
    return out, feet, mask


def limb_gaps(qpos: np.ndarray) -> tuple[float, float]:
    """Closest the arms come to the legs, and anything but the soles to the floor (m, negative: through)."""
    fromto = np.zeros(6)
    arms = [m.geom(f"{p}_{s}").id for s in "lr" for p in ("upperarm", "forearm", "hand")]
    legs = [m.geom(f"{p}_{s}").id for s in "lr" for p in ("thigh", "shin", "foot")]
    floor = m.geom("floor").id
    low = [m.geom(f"{p}_{s}").id for s in "lr" for p in ("thigh", "shin", "forearm", "hand")]
    arm_leg = floor_gap = np.inf
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        arm_leg = min(arm_leg, min(mujoco.mj_geomDistance(m, d, a, b, 0.5, fromto) for a in arms for b in legs))
        floor_gap = min(floor_gap, min(mujoco.mj_geomDistance(m, d, g, floor, 0.5, fromto) for g in low))
    return arm_leg, floor_gap
