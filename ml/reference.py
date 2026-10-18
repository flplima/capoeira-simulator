"""Fuse the two camera views into one clean, symmetric, looping ginga cycle.

    uv run python reference.py   # → data/ginga_ref.npz

Monocular pose estimation is good across the image and poor in depth, so the
front camera is trusted for side-to-side motion (abduction, lateral bends,
roll) and the side camera for forward-and-back motion (flexion, pitch). Each
shot is time-warped onto a common phase — 0 when the right foot is furthest
back, 0.5 when the left one is — then the two halves of the cycle are
averaged with each other's mirror image (the ginga is the same movement on
both sides), and a Fourier low-pass makes it loop smoothly. Finally the root
is placed on the floor and carried by the stance foot, so the reference
travels the way the feet say it does.

Last, the legs are refitted with inverse kinematics so the feet step on the
ginga triangle from the movement file: each foot rests on its front corner and
steps back to the shared back corner, landing on the ball of the foot. The
upper body and the timing stay as in the video; the root follows the feet, so
the reference no longer wanders and a policy that tracks it stays in place.
"""

import mujoco
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

FPS = 30
# Phase events read off the fitted hip flexion (see README): the right foot is
# furthest back at phase 0, the left at phase 0.5.
FRONT = {"file": "data/ginga_fit.npz", "events": [(6.27, 0.0), (7.97, 0.5), (9.88, 1.0)]}
SIDE = {"file": "data/ginga_fit_side.npz", "events": [(13.01, 0.5), (16.6, 1.5)]}
PERIOD = np.mean([9.88 - 6.27, 16.6 - 13.01])
SAMPLES = int(round(PERIOD * FPS))
HARMONICS = 7
ROOT_HARMONICS = 3

m = mujoco.MjModel.from_xml_path("humanoid.xml")
d = mujoco.MjData(m)
NAMES = [m.joint(i).name for i in range(1, m.njnt)]


def view_weight(name: str) -> float:
    """How much to trust the front view (vs the side view) for a joint."""
    if name.endswith(("_abd", "_side", "_inv")):
        return 0.8
    if name.endswith(("_flex",)) or name.startswith(("knee", "elbow")) or name in ("ankle_l", "ankle_r"):
        return 0.25
    return 0.5


def root_angles(quat: np.ndarray) -> np.ndarray:
    """Root orientation as the web's yaw, pitch, roll (YXZ intrinsic; roll toward own left positive)."""
    yaw, pitch, roll = Rotation.from_quat(quat, scalar_first=True).as_euler("YXZ")
    return np.array([yaw, pitch, -roll])


def root_quat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    return Rotation.from_euler("YXZ", [yaw, pitch, -roll]).as_quat(scalar_first=True)


def on_phase(shot: dict) -> np.ndarray:
    """Resample a shot onto one cycle of the common phase: rows are [yaw, pitch, roll, joints...]."""
    data = np.load(shot["file"])
    times, qpos = data["t"], data["qpos"]
    angles = np.array([root_angles(q[3:7]) for q in qpos])
    angles[:, 0] = np.unwrap(angles[:, 0])
    rows = np.c_[angles, qpos[:, 7:]]
    event_t, event_phase = zip(*shot["events"])
    phase_of_time = interp1d(event_t, event_phase, fill_value="extrapolate")
    keep = (times >= event_t[0]) & (times <= event_t[-1])
    phase = phase_of_time(times[keep]) % 1.0
    order = np.argsort(phase)
    phase, rows = phase[order], rows[keep][order]
    grid = np.arange(SAMPLES) / SAMPLES
    # Periodic linear interpolation, then remove the heading the camera saw.
    phase = np.r_[phase - 1, phase, phase + 1]
    rows = np.r_[rows, rows, rows]
    out = interp1d(phase, rows, axis=0)(grid)
    out[:, 0] -= np.mean(out[:, 0])
    return out


def swap_side(name: str) -> str:
    parts = name.split("_")
    return "_".join({"l": "r", "r": "l"}.get(p, p) for p in parts)


MIRROR_SWAP = {n: swap_side(n) for n in NAMES}
NEGATE = {n for n in NAMES if n.startswith(("spine_", "chest_", "neck_")) and n.endswith(("_side", "_twist"))}


def mirror(row: np.ndarray) -> np.ndarray:
    out = row.copy()
    out[0], out[2] = -row[0], -row[2]  # yaw and roll change side
    for i, n in enumerate(NAMES):
        source = row[3 + NAMES.index(MIRROR_SWAP[n])]
        out[3 + i] = -source if n in NEGATE else source
    return out


def low_pass(rows: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(rows, axis=0)
    spectrum[HARMONICS + 1 :] = 0
    return np.fft.irfft(spectrum, n=len(rows), axis=0)


def place_root(rows: np.ndarray) -> np.ndarray:
    """Full qpos per sample: lowest sole on the floor, the stance foot carrying the body across it."""
    qpos = np.zeros((len(rows), m.nq))
    soles = ["heel_l", "toe_l", "heel_r", "toe_r"]
    previous_stance = None
    offset = np.zeros(2)
    for k, row in enumerate(rows):
        q = qpos[k]
        q[3:7] = root_quat(*row[:3])
        q[7:] = row[3:]
        q[1] = 1.0
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        heights = {s: d.site_xpos[m.site(s).id][1] for s in soles}
        q[1] -= min(heights.values())
        # Stance foot: the lower one. Keep it fixed horizontally while it stays the stance foot.
        foot = "l" if min(heights["heel_l"], heights["toe_l"]) < min(heights["heel_r"], heights["toe_r"]) else "r"
        ankle = d.site_xpos[m.site(f"ankle_{foot}").id][[0, 2]] + offset
        if previous_stance is not None and previous_stance[0] == foot:
            offset += previous_stance[1] - ankle
            ankle = previous_stance[1]
        previous_stance = (foot, ankle)
        q[0], q[2] = offset
    # The cycle must end where it started: spread any drift over the cycle.
    drift = 2 * qpos[-1, [0, 2]] - qpos[-2, [0, 2]] - qpos[0, [0, 2]]
    qpos[:, [0, 2]] -= np.outer(np.arange(len(rows)) / len(rows), drift)
    qpos[:, [0, 2]] -= qpos[:, [0, 2]].mean(axis=0)
    # Stance switches make the carried root jerky; a person's centre glides.
    # The height stays exact (lowest sole on the floor every sample).
    spectrum = np.fft.rfft(qpos[:, [0, 2]], axis=0)
    spectrum[ROOT_HARMONICS + 1 :] = 0
    qpos[:, [0, 2]] = np.fft.irfft(spectrum, n=len(qpos), axis=0)
    return qpos


TRIANGLE = np.array(yaml.safe_load(open("../public/data/movements/ginga.yaml"))["triangle"], float)  # front left, front right, back
LEG_JOINTS = [n for s in "lr" for n in (f"hip_{s}_flex", f"hip_{s}_abd", f"hip_{s}_rot", f"knee_{s}", f"ankle_{s}", f"ankle_{s}_inv")]
# Phase windows of the right foot's trip (the left one is half a cycle later):
# to the back corner, resting there on the ball of the foot, back to the front.
GO = (0.78, 0.97)
BACK = (0.97, 1.05)
RETURN = (1.05, 1.22)


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def step_plan(phase: float) -> tuple[float, float]:
    """(how far toward the back corner 0..1, how much the foot is in the air 0..1) for the right foot."""
    p = phase % 1.0
    if p < GO[0] - 0.5:  # wrap the return window into [0, 0.28)
        p += 1.0
    if GO[0] <= p < GO[1]:
        u = (p - GO[0]) / (GO[1] - GO[0])
        return smoothstep(u), np.sin(np.pi * u)
    if BACK[0] <= p < BACK[1]:
        return 1.0, 0.0
    if RETURN[0] <= p < RETURN[1]:
        u = (p - RETURN[0]) / (RETURN[1] - RETURN[0])
        return 1 - smoothstep(u), np.sin(np.pi * u)
    return 0.0, 0.0


def foot_targets(phase: float, side: str, heel: np.ndarray, toe: np.ndarray, lift: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heel and toe targets for one foot, and its floor point (midway heel–toe), from the reference's own foot shape."""
    back, air = step_plan(phase if side == "r" else phase + 0.5)
    front = TRIANGLE[0 if side == "l" else 1]
    point = front + (TRIANGLE[2] - front) * back
    half = (toe - heel)[[0, 2]] / 2
    # Planted: flat in front, heel raised at the back; in the air: the video's lift.
    raise_heel = back * max(heel[1] - toe[1], 0.0)
    rest = np.array([raise_heel, 0.0])
    lifted = lift * np.array([max(heel[1], 0.0), max(toe[1], 0.0)])
    heights = (1 - air) * rest + air * np.maximum(lifted, rest)
    h = np.array([point[0] - half[0], heights[0], point[1] - half[1]])
    t = np.array([point[0] + half[0], heights[1], point[1] + half[1]])
    return h, t, point


def balance_path(points: np.ndarray, height: float, period: float) -> np.ndarray:
    """A centre-of-mass path (xz per sample) that a body can actually ride on its feet.

    The point the floor pushes from (the zero-moment point) must stay under a
    foot: on the standing foot while the other one is in the air, between the
    two otherwise. For a body of centre height h that point is c - (h/g)·c'',
    so over a looping cycle each Fourier component of the path is the support
    point's divided by 1 + (h/g)·ω² — the faster the tempo, the less the body
    needs to sway to stay balanced.
    """
    n = len(points)
    air = np.array([[step_plan(k / n + (0.5 if side == "l" else 0.0))[1] > 0 for side in "lr"] for k in range(n)], float)
    share = 1 - air  # weight on each foot: none on a foot in the air
    share = gaussian_filter1d(share, 1.0, axis=0, mode="wrap")
    share[air > 0] = 0.0
    share /= share.sum(axis=1, keepdims=True)
    support = (share[:, :, None] * points).sum(axis=1)
    omega = 2 * np.pi * np.arange(n // 2 + 1) / period
    spectrum = np.fft.rfft(support, axis=0) / (1 + height / 9.81 * omega**2)[:, None]
    return np.fft.irfft(spectrum, n=n, axis=0)


def fit_triangle(qpos: np.ndarray, drop: float = 0.0, lift: float = 1.0, period: float = PERIOD) -> tuple[np.ndarray, np.ndarray]:
    """Refit root position and legs so the feet follow the triangle; returns qpos and the feet's floor points.

    drop lowers (negative) or raises the hips from the video's height, bending the legs; lift scales how high
    the stepping foot rises; period is the cycle's length in seconds, which sets how much the body sways.
    """
    legs = np.array([m.joint(n).qposadr[0] for n in LEG_JOINTS])
    lo = np.r_[-np.inf, 0.3, -np.inf, [m.jnt_range[m.joint(n).id][0] for n in LEG_JOINTS]]
    hi = np.r_[np.inf, 1.2, np.inf, [m.jnt_range[m.joint(n).id][1] for n in LEG_JOINTS]]
    sites = [m.site(n).id for n in ("heel_l", "toe_l", "heel_r", "toe_r")]
    out, points = qpos.copy(), np.zeros((len(qpos), 2, 2))
    root_path = com_path = None

    def solve_pass() -> None:
        x = None
        for k, q in enumerate(qpos):
            phase = k / len(qpos)
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            ref_sites = d.site_xpos[sites].copy()
            targets = []
            for i, side in enumerate("lr"):
                h, t, points[k, i] = foot_targets(phase, side, ref_sites[2 * i], ref_sites[2 * i + 1], lift)
                targets += [h, t]
            targets = np.array(targets)
            # The hips hang over the middle of the feet; the legs bend to put them there.
            centre = points[k].mean(axis=0)
            guess = np.r_[q[:3], q[legs]]

            def residuals(v: np.ndarray) -> np.ndarray:
                d.qpos[:] = q
                d.qpos[:3] = v[:3]
                d.qpos[legs] = v[3:]
                mujoco.mj_kinematics(m, d)
                if com_path is not None:
                    # Second pass: the centre of mass rides the balanced path, the hips' height the smoothed one.
                    mujoco.mj_comPos(m, d)
                    com = d.subtree_com[1][[0, 2]]
                    return np.r_[10 * (d.site_xpos[sites] - targets).ravel(), 0.3 * (v[3:] - q[legs]), 20.0 * (com - com_path[k]), 20.0 * (v[1] - root_path[k, 1])]
                return np.r_[10 * (d.site_xpos[sites] - targets).ravel(), 0.3 * (v[3:] - q[legs]), 8.0 * (v[1] - q[1] - drop), 2.0 * (v[[0, 2]] - centre)]

            start = guess if x is None else np.r_[x[:3], guess[3:]]
            x = least_squares(residuals, np.clip(start, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=300).x
            out[k, :3] = x[:3]
            out[k, legs] = x[3:]

    # First pass: the hips over the middle of the feet, which jumps whenever a foot moves.
    solve_pass()
    # The hips' height glides: keep only the slow part of it.
    spectrum = np.fft.rfft(out[:, :3], axis=0)
    spectrum[ROOT_HARMONICS + 2 :] = 0
    root_path = np.fft.irfft(spectrum, n=len(out), axis=0)
    # The centre of mass shifts over the standing foot before the other one lifts, as fast as balance needs.
    heights = []
    for q in out:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        heights.append(d.subtree_com[1][1])
    com_path = balance_path(points.copy(), float(np.mean(heights)), period)
    solve_pass()
    # Smooth the refitted channels like the rest, then re-ground every sample.
    columns = np.r_[0, 1, 2, legs]
    spectrum = np.fft.rfft(out[:, columns], axis=0)
    spectrum[2 * HARMONICS + 1 :] = 0
    out[:, columns] = np.fft.irfft(spectrum, n=len(out), axis=0)
    for q in out:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        q[1] -= min(d.site_xpos[s][1] for s in sites)
    return out, points


ARMS = {s: [f"shoulder_{s}_flex", f"shoulder_{s}_abd", f"shoulder_{s}_rot", f"elbow_{s}"] for s in "lr"}
HEAD = ["head", "neck"]
CLEARANCE = 0.02  # m between an arm and the head


def fix_arms(qpos: np.ndarray, periodic: bool = True) -> np.ndarray:
    """Move the guard arm out of the head: the video's forearm crosses the face, a solid body can't."""
    out = qpos.copy()
    fromto = np.zeros(6)
    for s, names in ARMS.items():
        adr = np.array([m.joint(n).qposadr[0] for n in names])
        lo = np.array([m.jnt_range[m.joint(n).id][0] for n in names])
        hi = np.array([m.jnt_range[m.joint(n).id][1] for n in names])
        arm = [m.geom(f"{part}_{s}").id for part in ("upperarm", "forearm", "hand")]
        head = [m.geom(n).id for n in HEAD]
        for k, q in enumerate(out):
            def gaps(v: np.ndarray) -> np.ndarray:
                d.qpos[:] = q
                d.qpos[adr] = v
                mujoco.mj_kinematics(m, d)
                return np.array([mujoco.mj_geomDistance(m, d, a, h, 1.0, fromto) for a in arm for h in head])

            if gaps(q[adr]).min() >= CLEARANCE:
                continue
            residuals = lambda v: np.r_[20 * np.maximum(CLEARANCE + 0.005 - gaps(v), 0), 0.5 * (v - q[adr])]
            q[adr] = least_squares(residuals, np.clip(q[adr], lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=200).x
        # Smooth the corrections into the motion.
        if periodic:
            spectrum = np.fft.rfft(out[:, adr], axis=0)
            spectrum[2 * HARMONICS + 1 :] = 0
            out[:, adr] = np.fft.irfft(spectrum, n=len(out), axis=0)
        else:
            out[:, adr] = gaussian_filter1d(out[:, adr], 1.0, axis=0, mode="nearest")
    return out


def arm_clearance(qpos: np.ndarray) -> float:
    fromto = np.zeros(6)
    pairs = [(m.geom(f"{p}_{s}").id, m.geom(h).id) for s in "lr" for p in ("upperarm", "forearm", "hand") for h in HEAD]
    worst = np.inf
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        worst = min(worst, min(mujoco.mj_geomDistance(m, d, a, b, 1.0, fromto) for a, b in pairs))
    return worst


# How the ginga changes with the berimbau (see README): Angola is slow, low and
# wide, with a fluid, leaning body and long steps; São Bento Grande is fast,
# upright and compact, with short, quick steps. Tempos in bpm, one cycle per
# bar of four beats; the triangle stays the same.
TEMPOS = [60, 78, 96, 114, 130]


def style(bpm: float) -> dict:
    t = (bpm - TEMPOS[0]) / (TEMPOS[-1] - TEMPOS[0])
    return {
        "period": 4 * 60 / bpm,
        "drop": -0.09 + 0.12 * t,  # hips 9 cm lower than the video at Angola, 3 cm higher at the fastest
        "lean": 1.15 - 0.3 * t,  # trunk flexion scale
        "lift": 1.0 - 0.45 * t,  # stepping-foot height scale
    }


def main() -> None:
    front, side = on_phase(FRONT), on_phase(SIDE)
    weights = np.r_[0.5, 0.2, 0.8, [view_weight(n) for n in NAMES]]  # yaw, pitch (side), roll (front)
    fused = weights * front + (1 - weights) * side
    half = SAMPLES // 2
    sym = fused.copy()
    for k in range(SAMPLES):
        other = mirror(fused[(k + half) % SAMPLES])
        sym[k] = (fused[k] + other) / 2
    rows = low_pass(sym)
    lo = np.array([m.jnt_range[m.joint(n).id][0] for n in NAMES])
    hi = np.array([m.jnt_range[m.joint(n).id][1] for n in NAMES])
    rows[:, 3:] = np.clip(rows[:, 3:], lo, hi)
    lean = [3 + NAMES.index(n) for n in ("spine_flex", "chest_flex")]
    out = {"tempos": np.array(TEMPOS, float), "joints": np.array(NAMES), "triangle": TRIANGLE}
    qpos_all, feet_all = [], []
    for bpm in TEMPOS:
        st = style(bpm)
        styled = rows.copy()
        styled[:, lean] = np.clip(styled[:, lean] * st["lean"], lo[np.array(lean) - 3], hi[np.array(lean) - 3])
        qpos, feet = fit_triangle(place_root(styled), st["drop"], st["lift"], st["period"])
        qpos = fix_arms(qpos)
        qpos_all.append(qpos)
        feet_all.append(feet)
        print(
            f"{bpm:3d} bpm: cycle {st['period']:.2f} s; height {qpos[:, 1].min():.2f}–{qpos[:, 1].max():.2f} m, "
            f"root travel x {np.ptp(qpos[:, 0]):.2f} z {np.ptp(qpos[:, 2]):.2f} m, arm–head clearance {arm_clearance(qpos) * 100:.1f} cm"
        )
    np.savez("data/ginga_ref.npz", qpos=np.array(qpos_all), feet=np.array(feet_all), periods=np.array([style(b)["period"] for b in TEMPOS]), **out)


if __name__ == "__main__":
    main()
