"""Reference motions for the kicks the live ginga can throw: martelo and bênção.

    npx --yes tsx@4.19.2 scripts/sample-clips.ts martelo bencao > ml/data/clips.json   # from the web project
    uv run python kicks.py                                                            # → data/kicks.npz

The kicks start from the keyframes in public/data/movements (tuned against
both camera views of the reference video, and checked against technique
notes: see the README), sampled at 30 fps for both legs. Here they become
model poses, and inverse kinematics plants them on the ginga triangle: the
support foot stays on its front corner the whole time (turning on the spot
for the martelo's pivot), and the kicking foot leaves from the back corner
and comes back to it, so a kick starts and ends in the ginga's base. While
the kicking foot is off the floor, the same solve also keeps the centre of
mass over the support foot — leaning the body and trunk against the leg the
way a person does — because a keyframe pose need not balance, and a pose
whose weight hangs outside the foot can't be held by any controller.
"""

import json

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from env import nlerp
from reference import NAMES, TRIANGLE, arm_clearance, d, fix_arms, m, root_quat

KICKS = ["martelo", "bencao", "meia-lua-de-frente", "queixada", "ponteira", "joelhada", "meia-lua-de-compasso"]
# The simulated body plays the kicks this much slower than the keyframes: at
# full speed the hips rise 15 cm in a quarter second onto one leg, faster than
# its torque-limited joints can follow, and the body hops off the floor.
SLOW = 1.4
# How long the kicking foot stays pinned to the back corner at each end (s), and the blend in/out of that pin.
PINNED, RELEASE = 0.05, 0.35
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


def pin_weight(t: float, duration: float) -> float:
    """1 while the kicking foot should sit on the back corner (start and end), 0 in the air."""
    edge = min(t, duration - t)
    return float(np.clip(1 - (edge - SHIFT - PINNED) / RELEASE, 0, 1))


def balance_path(start: np.ndarray, end: np.ndarray, lifted: np.ndarray, support: np.ndarray, height: float, dt: float) -> np.ndarray:
    """A centre-of-mass path (xz per sample) from start to end whose zero-moment point c - (h/g)·c''
    is the support foot whenever the kicking foot is lifted: the weight moves early, while both feet
    are down, and arrives over the foot as the other one leaves the floor."""
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
    names = [n for s in "lr" for n in (f"hip_{s}_flex", f"hip_{s}_abd", f"hip_{s}_rot", f"knee_{s}", f"ankle_{s}", f"ankle_{s}_inv")]
    names += ["spine_flex", "spine_side", "chest_flex", "chest_side"]
    legs = np.array([m.joint(n).qposadr[0] for n in names])
    tilt = 0.6  # rad: how far the solver may turn the whole body from the keyframe
    lo = np.r_[-np.inf, 0.3, -np.inf, [-tilt] * 3, [m.jnt_range[m.joint(n).id][0] for n in names]]
    hi = np.r_[np.inf, 1.3, -np.inf if False else np.inf, [tilt] * 3, [m.jnt_range[m.joint(n).id][1] for n in names]]
    sole = {s: [m.site(f"heel_{s}").id, m.site(f"toe_{s}").id] for s in "lr"}
    leg_l = [m.geom(f"{p}_l").id for p in ("thigh", "shin", "foot")]
    leg_r = [m.geom(f"{p}_r").id for p in ("thigh", "shin", "foot")]
    spine_geom = m.geom("spine").id
    fromto = np.zeros(6)
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
    lifted = np.array([pin_weight(k / fps, duration) < 0.99 for k in range(len(qpos))])
    com_path = balance_path(start, end, lifted, corners[support], height, 1 / fps)
    for k, q in enumerate(qpos):
        t = k / fps
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        pose_sites = {s: d.site_xpos[sole[s]].copy() for s in "lr"}

        def on(corner: np.ndarray, sites: np.ndarray) -> np.ndarray:
            """The foot's own shape (as posed) moved so its middle is on the corner and its lowest point on the floor."""
            middle = sites.mean(axis=0)
            moved = sites + np.array([corner[0] - middle[0], 0, corner[1] - middle[2]])
            moved[:, 1] -= moved[:, 1].min()
            return moved

        w = pin_weight(t, duration)
        targets = {support: on(corners[support], pose_sites[support]), kick: on(TRIANGLE[2], pose_sites[kick])}
        weights = {support: 10.0, kick: 10.0 * w}
        feet[k, "lr".index(support)] = corners[support]
        feet[k, "lr".index(kick)] = TRIANGLE[2]
        mask[k, "lr".index(support)] = 1.0
        mask[k, "lr".index(kick)] = 1.0 if w > 0.99 else 0.0


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
            r = [weights[s] * (d.site_xpos[sole[s]] - targets[s]).ravel() for s in "lr"]
            # The kicking foot never goes through the floor, nor through the standing leg or spine.
            r.append(10.0 * np.minimum(d.site_xpos[sole[kick]][:, 1], 0))
            r.append([20.0 * min(mujoco.mj_geomDistance(m, d, gl, gr, 0.1, fromto) - 0.02, 0.0) for gl in leg_l for gr in leg_r])
            r.append([20.0 * min(mujoco.mj_geomDistance(m, d, g, spine_geom, 0.1, fromto) - 0.01, 0.0) for g in leg_l + leg_r])
            # The weight rides a path the feet can hold: over the support foot while on one leg.
            r.append(15.0 * (d.subtree_com[1][[0, 2]] - com_path[k]))
            return np.concatenate(r + [0.5 * v[3:6], 0.3 * (v[6:] - q[legs])])

        start = np.r_[q[:3], 0, 0, 0, q[legs]] if x is None else np.r_[x[:6], q[legs]]
        start[1] = np.clip(start[1], 0.31, 1.29)
        x = least_squares(residuals, np.clip(start, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=400).x
        pose(x)
        out[k] = d.qpos
    # Ensure quaternion sign continuity before Gaussian filtering (for 360° turns like meia-lua-de-compasso).
    for i in range(1, len(out)):
        if np.dot(out[i - 1, 3:7], out[i, 3:7]) < 0:
            out[i, 3:7] *= -1
    # Light smoothing of what the solver chose, then the support foot back on the floor.
    columns = np.r_[0, 1, 2, 3, 4, 5, 6, legs]
    out[:, columns] = gaussian_filter1d(out[:, columns], 1.0, axis=0, mode="nearest")
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    for q in out:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        q[1] -= min(d.site_xpos[i][1] for s in "lr" for i in sole[s])
    return out, feet, mask


def main() -> None:
    clips = json.load(open("data/clips.json"))
    channels = clips["channels"]
    ginga = np.load("data/ginga_ref.npz")["qpos"]
    out = {}
    for name in KICKS:
        clip = clips[name]
        for side in "lr":
            # Start and end in the ginga's base with this leg back: phase 0 (right) or 0.5 (left).
            base = ginga[GINGA_TEMPO][0 if side == "r" else len(ginga[GINGA_TEMPO]) // 2]
            raw_frames = [dict(zip(channels, f)) for f in clip["frames"][side]]
            keyframed = np.array([to_model(f) for f in raw_frames])
            full_turn = any(abs(f["root.yaw"]) > 180.0 for f in raw_frames)
            for level, amplitude in enumerate(AMPLITUDES):
                eff_amp = min(amplitude, 0.65) if name == "meia-lua-de-compasso" else amplitude
                qpos = keyframed.copy()
                for k, q in enumerate(qpos):
                    t = k / clips["fps"]
                    edge = min(max(min(t, clip["duration"] - t) / FROM_GINGA, 0.0), 1.0)
                    edge = edge * edge * (3 - 2 * edge)
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
