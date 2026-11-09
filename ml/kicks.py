"""Reference motions for the kicks the live ginga can throw: martelo and bênção.

    npx --yes tsx@4.19.2 scripts/sample-clips.ts martelo bencao > ml/data/clips.json   # from the web project
    uv run python kicks.py                                                            # → data/kicks.npz

The kicks start from the keyframes in public/data/movements (tuned against
both camera views of the reference video, and checked against technique
notes: see the README), sampled at 30 fps for both legs — or, for the
martelo, from a real capoeirista's kick in a video (GVHMR, then
plant_clip.py; see VIDEO_KICKS). Here they become model poses, and inverse
kinematics plants them on the ginga triangle: the
support foot stays on its front corner the whole time (a foot that turns,
as in the martelo's pivot, turns on its ball: heel up, toe in place), and
the kicking foot leaves from the back corner
and comes back to it, so a kick starts and ends in the ginga's base. While
the kicking foot is off the floor, the same solve also keeps the centre of
mass over the support foot — leaning the body and trunk against the leg the
way a person does — because a keyframe pose need not balance, and a pose
whose weight hangs outside the foot can't be held by any controller.
"""

import json

import mujoco
import numpy as np
from scipy.optimize import least_squares

from env_math import nlerp, quat_mul, smoothstep, yaw_of
from plant_ik import IK_ADDR, LOWER, SOLE, UPPER, initial, legs_apart, on_floor, set_pose, smooth_and_ground
from reference import NAMES, TRIANGLE, arm_clearance, d, fix_arms, m, root_quat

KICKS = ["martelo", "bencao", "meia-lua-de-frente", "queixada", "ponteira", "joelhada", "meia-lua-de-compasso"]
# The simulated body plays the kicks this much slower than the keyframes: at
# full speed the hips rise 15 cm in a quarter second onto one leg, faster than
# its torque-limited joints can follow, and the body hops off the floor.
SLOW = 1.4
# How long the kicking foot stays pinned to the back corner at each end (s), and the blend in/out of that pin.
PINNED, RELEASE = 0.05, 0.35
# A support foot that turns under the weight (the martelo's pivot) does it on its ball: the body can't turn a
# flat foot, it drags all four corners. Its heel starts to rise once the foot has turned PIVOT_TURN[0] and is
# PIVOT_HEEL off the floor from PIVOT_TURN[1] on, and the weight rests BALL of the way from the foot's middle to
# its tip. A foot that only adjusts its angle a little (the bênção's) stays flat.
PIVOT_HEEL, PIVOT_TURN, BALL = 0.07, np.radians([15, 35]), 0.85
SWING_CLEARANCE = 0.08  # m: how high the kicking foot's pin arcs between its corner and the kick
# The kick holds the base this long (s) at each end, so the weight can travel onto the support foot
# while both feet are still down — and back after the kicking foot lands.
SHIFT = 0.3
# The kick grows out of the ginga's own base (the reference at a middle tempo) over this share of the
# keyframes' start, and melts back into it at the end, so the kick leaves from and lands in the pose
# the ginga network is actually in.
FROM_GINGA = 0.3  # s of keyframe time
GINGA_TEMPO = 2  # index into the reference tempos (96 bpm)
# Kick sizes for the training curriculum: the keyframe kick scaled toward the
# base pose (0.4 is a knee lift with a low kick, 1.0 the full kick). Each size
# gets its own balance solve.
AMPLITUDES = [0.4, 0.6, 0.8, 1.0]
# Kicks taken from a video instead of the keyframes: per leg, the clip plant_clip.py wrote, cut from just before
# the chamber through the landing to the frame the body stands settled again (clip_score.py finds those frames; a
# slow-motion replay goes through clip_speed.py first). The clip that stopped at the re-chamber, knee folded
# highest, and melted into the base from there left the body falling 0.3–1.2 s after the kick. Both legs are
# resampled to one length, because a move has one duration. The first and last pose are held FROM_GINGA longer at
# each end: the blend from the base, and the kicking foot leaving its corner, happen there.
VIDEO_KICKS = {"martelo": {"r": "data/martelo_r_ref_real.npz", "l": "data/martelo_l_ref_mirror.npz"}}


def to_model(frame: dict[str, float]) -> np.ndarray:
    """Web pose channels (degrees, anatomical) → a model qpos, before placement."""
    q = np.zeros(m.nq)
    q[3:7] = root_quat(*np.radians([frame["root.yaw"], frame["root.pitch"], frame["root.roll"]]))
    q[1] = 1.0
    angles = {}
    for part in ("spine", "chest"):
        for field, src in (("flex", "flex"), ("side", "side"), ("twist", "twist")):
            angles[f"{part}_{field}"] = frame[f"torso.{src}"] / 2
    for field in ("flex", "side", "twist"):
        angles[f"neck_{field}"] = frame[f"head.{field}"]
    for s in "lr":
        for field in ("flex", "abd", "rot"):
            angles[f"shoulder_{s}_{field}"] = frame[f"shoulder_{s}.{field}"]
            angles[f"hip_{s}_{field}"] = frame[f"hip_{s}.{field}"]
        angles[f"elbow_{s}"] = frame[f"elbow_{s}.flex"]
        angles[f"knee_{s}"] = frame[f"knee_{s}.flex"]
        angles[f"ankle_{s}"] = frame[f"ankle_{s}.flex"]
        angles[f"ankle_{s}_inv"] = 0.0
    for name in NAMES:
        j = m.joint(name)
        lo, hi = m.jnt_range[j.id]
        q[j.qposadr[0]] = np.clip(np.radians(angles[name]), lo, hi)
    return q


def from_video(path: str, frames: int, base: np.ndarray) -> np.ndarray:
    """A planted video clip's poses resampled to `frames`, turned and moved so they begin where base is, with the
    first and last pose held FROM_GINGA longer."""
    clip = np.load(path)
    at = np.linspace(0, len(clip["qpos"]) - 1, frames)
    qpos = np.stack([np.interp(at, np.arange(len(clip["qpos"])), channel) for channel in clip["qpos"].T], axis=1)
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    pad = int(round(FROM_GINGA * float(clip["fps"])))
    qpos = qpos[[0] * pad + list(range(frames)) + [frames - 1] * pad]
    turn = yaw_of(base[3:7]) - yaw_of(qpos[0, 3:7])
    c, s = np.cos(turn), np.sin(turn)
    x, z = qpos[:, 0] - qpos[0, 0], qpos[:, 2] - qpos[0, 2]
    qpos[:, 0], qpos[:, 2] = base[0] + c * x + s * z, base[2] - s * x + c * z
    spin = np.array([np.cos(turn / 2), 0.0, np.sin(turn / 2), 0.0])
    qpos[:, 3:7] = [quat_mul(spin, q) for q in qpos[:, 3:7]]
    return qpos


def pin_weight(t: float, duration: float) -> float:
    """1 while the kicking foot should sit on the back corner (start and end), 0 in the air."""
    edge = min(t, duration - t)
    return float(np.clip(1 - (edge - SHIFT - PINNED) / RELEASE, 0, 1))


def heading(sites: np.ndarray) -> float:
    """Which way a foot points on the floor, from its heel and toe."""
    return float(np.arctan2(*(sites[1] - sites[0])[[0, 2]]))


def on_ball(sites: np.ndarray, lift: float) -> np.ndarray:
    """A foot's heel and toe with the heel raised to at least lift, turning about the toe."""
    heel, toe = sites
    if heel[1] >= lift:
        return sites
    back = heel - toe
    scale = np.sqrt(back @ back - lift**2) / np.hypot(back[0], back[2])
    return np.array([[toe[0] + back[0] * scale, toe[1] + lift, toe[2] + back[2] * scale], toe])


def balance_path(start: np.ndarray, end: np.ndarray, lifted: np.ndarray, support: np.ndarray, height: float, dt: float) -> np.ndarray:
    """A centre-of-mass path (xz per sample) from start to end whose zero-moment point c - (h/g)·c''
    is on the support foot (a point per sample) whenever the kicking foot is lifted: the weight moves early,
    while both feet are down, and arrives over the foot as the other one leaves the floor."""
    n = len(lifted)
    zmp = np.where(lifted[:, None], support, np.where(np.arange(n)[:, None] < n // 2, start, end))
    a = height / 9.81 / dt**2
    system = np.zeros((n, n))
    rhs = zmp.copy()
    system[0, 0] = system[-1, -1] = 1.0
    rhs[0], rhs[-1] = start, end
    for k in range(1, n - 1):
        system[k, k - 1 : k + 2] = [-a, 1 + 2 * a, -a]
    return np.linalg.solve(system, rhs)


def plant(qpos: np.ndarray, kick: str, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place the kick on the triangle; returns qpos, the feet's floor targets and which targets apply."""
    support = "l" if kick == "r" else "r"
    corners = {"l": TRIANGLE[0], "r": TRIANGLE[1]}
    fps = (len(qpos) - 1) / duration
    out = qpos.copy()
    feet = np.zeros((len(qpos), 2, 2))
    mask = np.zeros((len(qpos), 2))
    x = None
    d.qpos[:] = qpos[0]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    start, height = d.subtree_com[1][[0, 2]].copy(), d.subtree_com[1][1]
    d.qpos[:] = qpos[-1]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    end = d.subtree_com[1][[0, 2]].copy()
    weights_in_time = [pin_weight(k / fps, duration) for k in range(len(qpos))]
    lifted = np.array(weights_in_time) < 0.99
    posed, planted = [], []
    for q, w in zip(qpos, weights_in_time):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        posed.append({s: d.site_xpos[SOLE[s]].copy() for s in "lr"})
        turned = abs((heading(posed[-1][support]) - heading(posed[0][support]) + np.pi) % (2 * np.pi) - np.pi)
        lift = PIVOT_HEEL * smoothstep((turned - PIVOT_TURN[0]) / (PIVOT_TURN[1] - PIVOT_TURN[0])) * (1 - w)
        foot = on_ball(on_floor(corners[support], posed[-1][support]), lift)
        # Turning on the ball, the toe stays where it started and the heel swings around it.
        planted.append(foot + ((planted[0][1] if planted else foot[1]) - foot[1]) * [1, 0, 1])
    middle = np.array([p.mean(axis=0)[[0, 2]] for p in planted])
    raised = np.array([min(p[0][1] / PIVOT_HEEL, 1.0) for p in planted])[:, None]
    weight_on = middle + (np.array([p[1][[0, 2]] for p in planted]) - middle) * BALL * raised
    com_path = balance_path(start, end, lifted, weight_on, height, 1 / fps)
    for k, q in enumerate(qpos):
        pose_sites, w = posed[k], weights_in_time[k]
        targets = {support: planted[k]}
        # The kicking foot's pin slides between the corner and where the pose carries the foot (kept beside the
        # support foot as posed), rather than pulling a foot still high in the air straight down to the corner;
        # on the way it arcs up to SWING_CLEARANCE off the floor and travels late, so it never drags along it.
        carried = pose_sites[kick] + (targets[support] - pose_sites[support]).mean(axis=0) * [1, 0, 1]
        travel = 1 - (1 - smoothstep(w)) ** 2
        targets[kick] = carried + (on_floor(TRIANGLE[2], pose_sites[kick]) - carried) * [travel, smoothstep(w), travel]
        targets[kick][:, 1] += SWING_CLEARANCE * 4 * w * (1 - w)
        weights = {support: 10.0, kick: 10.0 * min(4 * w, 1.0)}
        feet[k, "lr".index(support)] = middle[k]
        feet[k, "lr".index(kick)] = TRIANGLE[2]
        mask[k, "lr".index(support)] = 1.0
        mask[k, "lr".index(kick)] = 1.0 if w > 0.99 else 0.0

        def residuals(v: np.ndarray) -> np.ndarray:
            set_pose(q, v)
            r = [weights[s] * (d.site_xpos[SOLE[s]] - targets[s]).ravel() for s in "lr"]
            # The kicking foot never goes through the floor, nor through the standing leg or spine.
            r.append(10.0 * np.minimum(d.site_xpos[SOLE[kick]][:, 1], 0))
            r.append(legs_apart())
            # The weight rides a path the feet can hold: over the support foot while on one leg.
            r.append(15.0 * (d.subtree_com[1][[0, 2]] - com_path[k]))
            return np.concatenate(r + [0.5 * v[3:6], 0.3 * (v[6:] - q[IK_ADDR])])

        x = least_squares(residuals, initial(q, x), bounds=(LOWER, UPPER), max_nfev=400).x
        set_pose(q, x)
        out[k] = d.qpos
    return smooth_and_ground(out), feet, mask


def main() -> None:
    clips = json.load(open("data/clips.json"))
    channels = clips["channels"]
    ginga = np.load("data/ginga_ref.npz")["qpos"]
    out = {}
    for name in KICKS:
        clip = clips[name]
        if name in VIDEO_KICKS:
            # A video clip is already at the speed of life: thinned by SLOW, the slowed clock plays it at that speed.
            frames = int(round(np.mean([len(np.load(path)["qpos"]) for path in VIDEO_KICKS[name].values()]) / SLOW))
        for side in "lr":
            # Start and end in the ginga's base with this leg back: phase 0 (right) or 0.5 (left).
            base = ginga[GINGA_TEMPO][0 if side == "r" else len(ginga[GINGA_TEMPO]) // 2]
            if name in VIDEO_KICKS:
                keyframed, full_turn = from_video(VIDEO_KICKS[name][side], frames, base), False
                length = (len(keyframed) - 1) / clips["fps"]
            else:
                raw_frames = [dict(zip(channels, f)) for f in clip["frames"][side]]
                keyframed = np.array([to_model(f) for f in raw_frames])
                full_turn = any(abs(f["root.yaw"]) > 180.0 for f in raw_frames)
                length = clip["duration"]
            for level, amplitude in enumerate(AMPLITUDES):
                eff_amp = min(amplitude, 0.65) if name == "meia-lua-de-compasso" else amplitude
                qpos = keyframed.copy()
                for k, q in enumerate(qpos):
                    t = k / clips["fps"]
                    edge = smoothstep(min(t, length - t) / FROM_GINGA)
                    s = edge * eff_amp
                    if full_turn:
                        # A 360° turn (meia-lua-de-compasso) must complete its circle at every curriculum size
                        # instead of flipping across ±180° when nlerp scales toward 0°.
                        f = raw_frames[k]
                        q_turn = root_quat(*np.radians([f["root.yaw"], f["root.pitch"] * eff_amp, f["root.roll"] * eff_amp]))
                        quat = nlerp(base[3:7], q_turn, edge)
                    else:
                        quat = nlerp(base[3:7], q[3:7], s)
                    q[:] = base + (q - base) * s
                    q[3:7] = quat
                for k in range(1, len(qpos)):
                    if np.dot(qpos[k - 1, 3:7], qpos[k, 3:7]) < 0:
                        qpos[k, 3:7] *= -1
                hold = np.repeat(qpos[:1], int(round(SHIFT * clips["fps"] / SLOW)), axis=0)
                end_hold = np.repeat(qpos[-1:], len(hold), axis=0)
                qpos = np.concatenate([hold, qpos, end_hold])
                duration = (len(qpos) - 1) * SLOW / clips["fps"]
                qpos, feet, mask = plant(qpos, side, duration)
                qpos = fix_arms(qpos, periodic=False)
                out[f"{name}_{side}_{level}_qpos"] = qpos
                out[f"{name}_{side}_{level}_feet"] = feet
                out[f"{name}_{side}_{level}_mask"] = mask
                kick_toe = []
                for q in qpos:
                    d.qpos[:] = q
                    mujoco.mj_kinematics(m, d)
                    kick_toe.append(d.site_xpos[m.site(f"toe_{side}").id][1])
                print(
                    f"{name} ({side}, size {amplitude}): {duration:.2f} s; hips {qpos[:, 1].min():.2f}–{qpos[:, 1].max():.2f} m, "
                    f"kicking toe up to {max(kick_toe):.2f} m, arm–head clearance {arm_clearance(qpos) * 100:.1f} cm"
                )
        out[f"{name}_duration"] = np.array(duration)
    np.savez("data/kicks.npz", fps=clips["fps"] / SLOW, kicks=np.array(KICKS), amplitudes=np.array(AMPLITUDES), **out)


if __name__ == "__main__":
    main()
