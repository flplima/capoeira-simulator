"""Reference motion for the esquiva lateral the live ginga can dodge with.

    npx --yes tsx@4.19.2 scripts/sample-clips.ts esquiva-lateral > ml/data/esquiva_clip.json   # from the web project
    uv run python esquiva.py                                                                    # → data/esquiva.npz

The esquiva starts from the app's keyframes (public/data/movements/esquiva-lateral.yaml),
sampled at 30 fps for both sides. It leaves from the ginga's base and comes back to it, like
a kick: from the base with the right leg back, the body drops at once onto the bent front (left)
leg while the back leg opens straight out to the side in a low step, the trunk lies over the bent
knee with the head out to the left and the eyes still on the partner — going down as fast as
gravity and the legs allow (drop), the head below 1 m in about 0.33 s and the bottom at about 0.5 s —
holds, comes back up on the front leg more slowly (LIFT_UP: rising off one deeply bent leg is the
hard part) while the back foot steps home, and is in the base again. The feet end up about a metre apart: on the triangle's front
corners (0.68 m apart) a straight far leg would hold the hips up at 0.8 m. The keyframes grow out
of the ginga's own base (the reference at a middle tempo) and melt back into it, as the kicks do
(kicks.py); the head keeps the ginga's angle, lifting only as far as the trunk leans further.

Here the poses become model poses, and inverse kinematics plants them: the front foot stays
flat on its corner; the back foot sits on its corner (on its ball, as in the base), steps low to
the wide spot, lands flat and steps home, on paths with no jump in acceleration where it stops.
The centre of mass follows com_path: a push from the back foot, a pendulum roll onto the front
foot while the back one is in the air (2–2.5 cm off the floor at mid-step), a little toward the wide foot at the bottom, and the
same in reverse. Hip and head height, the trunk's lean (the head out over the bent leg), flat
soles and legs and hands clear of the floor are targets of the same solve, because a keyframe
pose need not reach them once its feet are planted and its weight balanced.
"""

import json

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, least_squares

from env import nlerp
from kicks import AMPLITUDES, GINGA_TEMPO, to_model
from reference import TRIANGLE, arm_clearance, d, fix_arms, m

MOVE = "esquiva-lateral"
# Keyframe times (s, public/data/movements/esquiva-lateral.yaml): the back foot leaves its corner, is
# down on the wide spot, the weight has settled at the bottom, the body starts back up, the foot leaves
# the wide spot, is back on its corner. (The hips reach the bottom when DROP lets them: see drop.)
LIFT, WIDE, SETTLE, RISE, HOME, BACK = 0.18, 0.45, 0.62, 0.85, 1.3, 1.7
# Going down the hips fall no faster than DROP[0] (m/s², under gravity's 9.81: the legs give way, they
# can't pull the body down) and are caught at no more than DROP[1] (on one deeply bent knee).
DROP = (7.0, 9.0)
# Coming back up the hips speed up at no more than LIFT_UP[0] and slow down at no more than LIFT_UP[1] (m/s²):
# rising off one deeply bent leg is the hard part, and a body that can't follow the rise sits down.
LIFT_UP = (4.0, 6.0)
# The wide spot: this far (m) outside the back foot's own front corner, and this far behind it.
WIDEN, BEHIND = 0.36, 0.1
# At the bottom the centre of mass is this share of the way from the front foot to the wide one.
SHARE = 0.25
# At the bottom the hips are this high (m), the bent thigh about level with the floor; the head this
# high, and this far outside the front foot, over the bent knee, out of the kick's way.
HIPS_LOW, HEAD_LOW, HEAD_OUT = 0.48, 0.75, 0.08
# Legs and hands stay this far (m) off the floor (only the soles touch it).
CLEAR = 0.02
# The back foot's step is this high (m) at its middle: low, almost a slide, but off the floor.
ARC = 0.045
# At the bottom the neck lifts this far (rad) from the ginga's, so the eyes stay on the partner as the
# trunk lies over the knee.
NECK_UP = -0.6
# The keyframes blend in from the ginga's base over this much keyframe time (s), and back out.
FROM_GINGA = 0.15
# How long a foot's floor target blends in or out (s) as it arrives or leaves.
RELEASE = 0.05
# The simulated body plays the esquiva this much slower than the keyframes.
SLOW = 1.0


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def smoother(x: float) -> float:
    """0 to 1 with no jump in speed or acceleration at either end."""
    x = min(max(x, 0.0), 1.0)
    return x**3 * (x * (6 * x - 15) + 10)


def hermite(p0: np.ndarray, v0: np.ndarray, p1: np.ndarray, v1: np.ndarray, span: float, u: float) -> np.ndarray:
    """Cubic from (p0, v0) to (p1, v1) over `span` seconds, at fraction u."""
    u = min(max(u, 0.0), 1.0)
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    return h00 * p0 + h10 * span * v0 + h01 * p1 + h11 * span * v1


def drop(times: np.ndarray, depth: float, limits: tuple[float, float] = DROP) -> np.ndarray:
    """How far along (0–1 of `depth` m) the hips are, going down as fast as DROP allows (or, with LIFT_UP,
    coming up): a smooth push of acceleration, then a smooth stop, with the speed zero at both ends."""
    fall, catch = limits
    # Half-sine accelerations: down for t1, up for t2; equal impulses (fall·t1 = catch·t2); the depth fixes t1.
    def depth_for(t1: float) -> float:
        t2 = fall * t1 / catch
        v = 2 * fall * t1 / np.pi
        return fall * t1**2 / np.pi + v * t2 - catch * t2**2 / np.pi
    t1 = brentq(lambda t: depth_for(t) - depth, 1e-4, 2.0) if depth > 1e-4 else 1e-4
    t2 = fall * t1 / catch
    dt = 1e-3
    grid = np.arange(0.0, t1 + t2 + dt, dt)
    acc = np.where(grid < t1, fall * np.sin(np.pi * grid / t1), -catch * np.sin(np.pi * np.clip(grid - t1, 0, t2) / t2))
    speed = np.cumsum(acc) * dt
    down = np.cumsum(speed) * dt
    return np.clip(np.interp(times, grid, down / max(down[-1], 1e-9)), 0.0, 1.0)


def com_path(
    times: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    foot: np.ndarray,
    bottom: np.ndarray,
    height: float,
    stage_times: tuple[float, float, float, float, float, float] = (LIFT, WIDE, SETTLE, RISE, HOME, BACK),
) -> np.ndarray:
    """The centre of mass (xz per sample). In the base it sits between the feet, well off the front foot. A
    short push from the back foot sets it moving toward the front one; with the back foot in the air it rolls
    on toward the front foot like a pendulum that just reaches it (the zero-moment point stays on that foot);
    with both feet down it goes a little toward the wide foot, holds, comes back, and leaves the same way in
    reverse, coming to rest where the base has it."""
    t_lift, t_wide, t_settle, t_rise, t_home, t_back = stage_times
    w = np.sqrt(9.81 / height)
    first = start + 0.15 * (foot - start)  # where the back foot leaves, moving toward the front foot at…
    first_v = -w * (first - foot)  # …just the speed to arrive over it
    last = end + 0.15 * (foot - end)
    last_v = w * (last - foot)
    landed = foot + (first - foot) * np.exp(-w * (t_wide - t_lift))
    leaving = foot + (last - foot) * np.exp(-w * (t_back - t_home))
    zero = np.zeros(2)
    path = np.zeros((len(times), 2))
    for k, t in enumerate(times):
        if t < t_lift:
            path[k] = hermite(start, zero, first, first_v, t_lift, t / t_lift)
        elif t < t_wide:
            path[k] = foot + (first - foot) * np.exp(-w * (t - t_lift))
        elif t < t_rise:
            path[k] = hermite(landed, -w * (landed - foot), bottom, zero, t_settle - t_wide, (t - t_wide) / (t_settle - t_wide))
        elif t < t_home:
            path[k] = hermite(bottom, zero, leaving, w * (leaving - foot), t_home - t_rise, (t - t_rise) / (t_home - t_rise))
        elif t < t_back:
            path[k] = foot + (last - foot) * np.exp(-w * (t_back - t))
        else:
            path[k] = hermite(last, last_v, end, zero, times[-1] - t_back, (t - t_back) / (times[-1] - t_back))
    return path


DODGE_SPECS = {
    "esquiva": {
        "clip": "esquiva-lateral",
        "times": (LIFT, WIDE, SETTLE, RISE, HOME, BACK),
        "wide": lambda front, back, side, out: front[side] + np.array([out * WIDEN, -BEHIND]),
        "flat_wide": 1.0,
        "share": SHARE,
        "hips_low": HIPS_LOW,
        "head_low": HEAD_LOW,
        "head_x": lambda foot, out: foot[0] - out * HEAD_OUT,
    },
    "esquiva-baixa": {
        "clip": "esquiva-baixa",
        "times": (0.18, 0.45, 0.62, 0.85, 1.30, 1.65),
        "wide": lambda front, back, side, out: back + np.array([0.0, -0.10]),
        "flat_wide": 0.0,
        "share": 0.20,
        "hips_low": 0.55,
        "head_low": 0.78,
        "head_x": lambda foot, out: foot[0],
    },
    "cocorinha": {
        "clip": "cocorinha",
        "times": (LIFT, WIDE, SETTLE, RISE, HOME, BACK),
        "wide": lambda front, back, side, out: front[side],
        "flat_wide": 1.0,
        "share": 0.45,
        "hips_low": 0.52,
        "head_low": 0.80,
        "head_x": lambda foot, out: 0.0,
    },
}


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


def main() -> None:
    clips = json.load(open("data/esquiva_clip.json"))
    channels, fps = clips["channels"], clips["fps"]
    ginga = np.load("data/ginga_ref.npz")["qpos"][GINGA_TEMPO]
    out = {}
    for move_name, spec in DODGE_SPECS.items():
        clip = clips[spec["clip"]]
        for side in "lr":
            # Start and end in the ginga's base with this leg back: phase 0 (right) or 0.5 (left).
            base = ginga[0 if side == "r" else len(ginga) // 2]
            keyframed = np.array([to_model(dict(zip(channels, f))) for f in clip["frames"][side]])
            neck = [m.joint(f"neck_{f}").qposadr[0] for f in ("flex", "side", "twist")]
            keyframed[:, neck] = base[neck]  # the head keeps the ginga's angle (the keyframes' is the app's own)
            duration = (len(keyframed) - 1) / fps
            for level, amplitude in enumerate(AMPLITUDES):
                qpos = keyframed.copy()
                for k, q in enumerate(qpos):
                    t = k / fps
                    s = smooth(min(t, duration - t) / FROM_GINGA) * amplitude
                    quat = nlerp(base[3:7], q[3:7], s)
                    q[:] = base + (q - base) * s
                    q[3:7] = quat
                qpos, feet, mask = plant(qpos, side, amplitude, fps, base, spec)
                qpos = fix_arms(qpos, periodic=False)
                out[f"{move_name}_{side}_{level}_qpos"] = qpos
                out[f"{move_name}_{side}_{level}_feet"] = feet
                out[f"{move_name}_{side}_{level}_mask"] = mask
                head, coms, slide = [], [], []
                for q in qpos:
                    d.qpos[:] = q
                    mujoco.mj_kinematics(m, d)
                    mujoco.mj_comPos(m, d)
                    head.append(d.xpos[m.body("head").id][1])
                    coms.append(d.subtree_com[1][[0, 2]].copy())
                    slide.append(d.site_xpos[[m.site(f"heel_{side}").id, m.site(f"toe_{side}").id]].copy())
                coms, slide = np.array(coms), np.array(slide)
                front = TRIANGLE[0 if side == "r" else 1]
                times = np.arange(len(qpos)) / fps
                sink = -np.diff(qpos[:, 1]) * fps / SLOW
                arm_leg, floor_gap = limb_gaps(qpos)
                print(
                    f"{move_name} ({side}, size {amplitude}): {duration * SLOW:.2f} s; hips {qpos[:, 1].min():.2f}–{qpos[:, 1].max():.2f} m "
                    f"(lowest at {times[np.argmin(qpos[:, 1])]:.2f} s), head down to {min(head):.2f} m (centre), "
                    f"feet up to {np.abs(feet[:, 0, 0] - feet[:, 1, 0]).max():.2f} m apart, "
                    f"centre of mass up to {np.linalg.norm(coms - front, axis=1).max() * 100:.0f} cm from the front foot, "
                    f"hips dropping up to {sink.max():.2f} m/s, rising up to {-sink.min():.2f} m/s, "
                    f"sliding sole {slide[:, :, 1].min() * 100:+.1f}…{slide[:, :, 1].max() * 100:+.1f} cm off the floor, "
                    f"arms to legs {arm_leg * 100:.1f} cm, legs and hands to the floor {floor_gap * 100:.1f} cm, "
                    f"arm–head {arm_clearance(qpos) * 100:.1f} cm"
                )
            out[f"{move_name}_duration"] = np.array(duration * SLOW)
    np.savez("data/esquiva.npz", fps=fps / SLOW, moves=np.array(list(DODGE_SPECS.keys())), amplitudes=np.array(AMPLITUDES), **out)


if __name__ == "__main__":
    main()
