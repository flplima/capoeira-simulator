"""A reference the physics body can hold, from one GVHMR clip of the video.

    uv run python plant_clip.py data/martelo_hmr.npz data/martelo_fit_hmr.npz 7.45 9.3 data/martelo_r_ref_hmr.npz

Cut one clip per move, through its landing: where the video steps
fast between moves GVHMR rarely shows both feet down, and the weight would
have to jump from foot to foot.

The first file is hmr_modal.py's (every joint, with the body's path through
the room), the second retarget.py's fit of it (the joint angles, with the
hips held at one height). Here the body travels the path GVHMR recovered,
scaled to the model's legs; a foot the video shows standing (slow and near
the floor) stays planted where it stood, on one leg turning on its ball with
the heel up; and the centre of mass follows a path whose zero-moment point
is over the standing forefoot while the other foot is in the air, between
the feet when both are down. The last pose is held a moment, both feet down,
for the weight to come to rest in the middle of the base. The joint angles
stay the video's wherever that allows.
"""

import sys

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from scipy.optimize import least_squares

from plant_ik import FLOOR_GEOM, FOOT_GEOMS, FROMTO, IK_ADDR, LOWER, SOLE, UPPER, initial, feet_above_floor, legs_apart, set_pose, smooth_and_ground
from reference import d, fix_arms, m

FOOT = {s: [f"{side}_heel", f"{side}_big_toe", f"{side}_small_toe"] for s, side in (("l", "left"), ("r", "right"))}
STILL = 0.5  # m/s: a foot moving slower than this over the floor may be standing
LOW = 0.3  # m: ... and so may one this close to the floor under it (GVHMR floats a landed foot up to ~25 cm)
CARRIES = 0.25  # m: while one foot is this high, the other carries the body however much the video slides it
SHORTEST = 5  # frames: a shorter stand, or a shorter lift between two stands, is the video's noise
SHIFT_FRAMES = 2.0  # σ of the weight's move from one foot to the other, in frames
RAMP = 6  # frames over which a foot's pin fades in as it comes down to land
RELEASE = 2  # frames over which it fades out as the foot lifts, so the pin never holds back a chamber
STANCE = 0.3  # s: a clip ends with both feet down and the weight settled between them, to blend back into the ginga
HOLD = 0.4  # s the last pose of the video is held for that
HEEL_UP = 0.04  # m: on one foot the body turns on the ball of that foot (its forefoot), heel off the floor
FOREFOOT = 0.25  # share of the way from toe to heel where a foot on its ball carries the weight
HIPS_FRAMES = 3.0  # σ, in frames, of the smoothing of the hips' height between the two solves
ROOT_ABOVE_HIPS = 0.06  # m: the model's root sits this far above its hip joints, where SMPL-X puts the pelvis


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Each stretch of True as (first, past-the-last)."""
    edges = np.flatnonzero(np.diff(np.r_[0, flags.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def clean(flags: np.ndarray) -> np.ndarray:
    """Stands too short to be real are dropped, then lifts too short to be real are filled."""
    out = flags.copy()
    for a, b in runs(out):
        if b - a < SHORTEST:
            out[a:b] = False
    for a, b in runs(~out):
        if b - a < SHORTEST and a > 0 and b < len(out):
            out[a:b] = True
    return out


def contacts(joints: np.ndarray, names: list[str], fps: float) -> tuple[dict, dict, np.ndarray]:
    """Per foot: whether it stands each frame, the middle and the forefoot (xz) over time; plus the floor.
    GVHMR's floor drifts by several centimetres over a clip, so height counts only against the lowest foot nearby,
    and a standing foot that pivots under a kick can drift with it, so the foot under a raised leg always stands."""
    index = {n: i for i, n in enumerate(names)}
    points = {s: joints[:, [index[n] for n in FOOT[s]]] for s in "lr"}
    lowest = {s: points[s][:, :, 1].min(axis=1) for s in "lr"}
    floor = gaussian_filter1d(minimum_filter1d(np.minimum(lowest["l"], lowest["r"]), int(fps)), fps / 4)
    stands, middle, forefoot = {}, {}, {}
    for s in "lr":
        middle[s] = points[s][:, :2, :].mean(axis=1)[:, [0, 2]]
        toes = points[s][:, 1:, :].mean(axis=1)[:, [0, 2]]
        forefoot[s] = toes + FOREFOOT * (points[s][:, 0, [0, 2]] - toes)
        speed = gaussian_filter1d(np.linalg.norm(np.gradient(middle[s], axis=0), axis=1) * fps, 1.0)
        other = lowest["r" if s == "l" else "l"]
        stands[s] = clean(((speed < STILL) & (lowest[s] - floor < LOW)) | ((other - floor > CARRIES) & (lowest[s] < other)))
    # The body never jumps here: where the video shows no foot standing, the lower one does.
    neither = ~stands["l"] & ~stands["r"]
    for s, other in (("l", "r"), ("r", "l")):
        stands[s] = clean(stands[s] | (neither & (lowest[s] <= lowest[other])))
    for s in "lr":
        stands[s][-int(round(STANCE * fps)):] = True
    return stands, middle, forefoot, floor


def pins(stands: np.ndarray, spot_of: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, how hard a foot is held to the floor (1 standing, fading to 0 RAMP frames away) and where."""
    n = len(stands)
    weight, spot = np.zeros(n), np.zeros((n, 2))
    stretches = runs(stands)
    if not stretches:
        return weight, spot
    places = [np.median(spot_of[a:b], axis=0) for a, b in stretches]
    for k in range(n):
        gaps = [(a - k) / RAMP if k < a else max(k - (b - 1), 0) / RELEASE for a, b in stretches]
        nearest = int(np.argmin(gaps))
        u = max(0.0, 1 - gaps[nearest])
        weight[k], spot[k] = u * u * (3 - 2 * u), places[nearest]
    return weight, spot


def zmp_path(target: np.ndarray, weight: np.ndarray, prior: np.ndarray, height: float, dt: float) -> np.ndarray:
    """A centre-of-mass path (xz) whose zero-moment point c - (h/g)·c'' is on target where weight is high,
    staying near prior (the video's own) elsewhere."""
    n = len(target)
    # The weight can't jump from one foot to the other: the target moves across over a few frames.
    spread = gaussian_filter1d(weight, SHIFT_FRAMES)
    target = gaussian_filter1d(weight[:, None] * target, SHIFT_FRAMES, axis=0) / np.maximum(spread, 1e-9)[:, None]
    weight = spread
    a = height / 9.81 / dt**2
    zmp = np.eye(n)
    for k in range(1, n - 1):
        zmp[k, k - 1 : k + 2] = [-a, 1 + 2 * a, -a]
    settle = np.zeros((2, n))  # the body comes to rest at the end: no speed over the last two frames
    settle[0, -3:-1], settle[1, -2:] = [100.0, -100.0], [100.0, -100.0]
    rows = np.r_[np.sqrt(weight)[:, None] * zmp, 0.5 * np.eye(n), settle]
    rhs = np.r_[np.sqrt(weight)[:, None] * target, 0.5 * prior, np.zeros((2, 2))]
    return np.linalg.lstsq(rows, rhs, rcond=None)[0]


def com_of(q: np.ndarray) -> np.ndarray:
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    return d.subtree_com[1].copy()


def plant(fit: np.ndarray, joints: np.ndarray, names: list[str], fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """qpos that stands where the video stood; returns it with the feet's floor targets (xz) and which apply."""
    index = {n: i for i, n in enumerate(names)}
    thigh = np.linalg.norm(joints[:, index["left_hip"]] - joints[:, index["left_knee"]], axis=1).mean()
    shin = np.linalg.norm(joints[:, index["left_knee"]] - joints[:, index["left_ankle"]], axis=1).mean()
    hip, ankle = m.site("hip_joint_l").id, m.site("ankle_l").id
    d.qpos[:] = 0
    d.qpos[3] = 1
    mujoco.mj_kinematics(m, d)
    scale = (d.site_xpos[hip][1] - d.site_xpos[ankle][1]) / (thigh + shin)
    stands, middle, forefoot, floor = contacts(joints, names, fps)
    pelvis = joints[:, index["pelvis"]]
    guide = fit.copy()
    guide[:, 0], guide[:, 2] = scale * pelvis[:, 0], scale * pelvis[:, 2]
    guide[:, 1] = scale * (pelvis[:, 1] - floor) + ROOT_ABOVE_HIPS
    pin = {s: pins(stands[s], scale * forefoot[s]) for s in "lr"}
    carry = {s: pin[s][1] for s in "lr"}
    edge = int(round(STANCE * fps))

    video_com = np.array([com_of(q) for q in guide])
    both = stands["l"] & stands["r"]
    target, weight = np.zeros((len(fit), 2)), np.zeros(len(fit))
    for k in range(len(fit)):
        if both[k]:
            a, b = carry["l"][k], carry["r"][k]
            u = np.clip(np.dot(video_com[k, [0, 2]] - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
            if k >= len(fit) - edge:
                u = 0.5
            target[k], weight[k] = a + u * (b - a), 1.0
        elif stands["l"][k] or stands["r"][k]:
            target[k], weight[k] = carry["l" if stands["l"][k] else "r"][k], 10.0
    com_path = zmp_path(target, weight, video_com[:, [0, 2]], float(np.mean(video_com[:, 1])), 1 / fps)

    # A standing foot keeps its ball where it stood and turns on it: heel down on two feet, up on one,
    # changing over a few frames as the other foot lifts or lands.
    alone = {s: gaussian_filter1d(1.0 - stands["r" if s == "l" else "l"], SHIFT_FRAMES) for s in "lr"}

    def solve(guide: np.ndarray, height: float) -> np.ndarray:
        """Each frame in turn, starting from the last frame's root and turn; height is how closely the hips keep the guide's."""
        out, x = guide.copy(), None
        for k, q in enumerate(guide):
            heel = {s: (HEEL_UP * alone[s][k], 10.0 - 7.0 * alone[s][k]) for s in "lr"}

            def residuals(v: np.ndarray) -> np.ndarray:
                set_pose(q, v)
                r = []
                for s in "lr":
                    heel_at, toe_at = d.site_xpos[SOLE[s]]
                    w = pin[s][0][k]
                    ball_at = toe_at + FOREFOOT * (heel_at - toe_at)
                    r.append(30.0 * w * (ball_at[[0, 2]] - pin[s][1][k]))
                    r.append([30.0 * w * toe_at[1]])
                    r.append([heel[s][1] * w * (heel_at[1] - heel[s][0])])
                r += [10.0 * np.minimum(d.site_xpos[SOLE[s]][:, 1], 0) for s in "lr"]
                r.append(legs_apart())
                r.append(feet_above_floor())
                r.append(15.0 * (d.subtree_com[1][[0, 2]] - com_path[k]))
                r.append([height * (v[1] - q[1])])
                return np.concatenate(r + [0.5 * v[3:6], 0.3 * (v[6:] - q[IK_ADDR])])

            x = least_squares(residuals, initial(q, x), bounds=(LOWER, UPPER), max_nfev=400).x
            set_pose(q, x)
            out[k] = d.qpos
        return out

    # Smoothing root and joints apart lets a planted foot drift, so the smoothed motion is solved once more,
    # keeping the hips on a smoother height: the first solve can drop them all at once as a foot reaches to land.
    smoothed = smooth_and_ground(solve(guide, 1.0))
    smoothed[:, 1] = gaussian_filter1d(smoothed[:, 1], HIPS_FRAMES, mode="nearest")
    out = fix_arms(solve(smoothed, 10.0), periodic=False)
    feet = np.zeros((len(out), 2, 2))
    for k, q in enumerate(out):  # the middle of each foot, as the environment measures it, pivot included
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        feet[k] = [d.site_xpos[SOLE[s]].mean(axis=0)[[0, 2]] for s in "lr"]
    mask = np.stack([pin["l"][0] > 0.99, pin["r"][0] > 0.99], axis=1).astype(float)
    return out, feet, mask


def report(t: np.ndarray, qpos: np.ndarray, feet: np.ndarray, mask: np.ndarray, fps: float) -> None:
    """Where each foot stands, how far a standing foot slides or floats, and how far the weight is from the feet."""
    soles = np.zeros((len(qpos), 2, 2, 3))
    com = np.zeros((len(qpos), 3))
    for k, q in enumerate(qpos):
        com[k] = com_of(q)
        soles[k] = [d.site_xpos[SOLE[s]] for s in "lr"]
    for i, s in enumerate("lr"):
        for a, b in runs(mask[:, i] > 0):
            ball = (soles[a:b, i, 1] + FOREFOOT * (soles[a:b, i, 0] - soles[a:b, i, 1]))[:, [0, 2]]
            slide = np.linalg.norm(ball - np.median(ball, axis=0), axis=1).max()
            heel = soles[a:b, i, 0, 1]
            print(f"  {s} foot stands {t[a]:.2f}–{t[b - 1]:.2f} s: ball slides ≤ {slide * 100:.1f} cm, "
                  f"heel {heel.min() * 100:.1f}–{heel.max() * 100:.1f} cm up, toe ≤ {soles[a:b, i, 1, 1].max() * 100:.1f} cm up")
    sunk = 0.0
    for q in qpos:
        com_of(q)
        sunk = min([sunk] + [mujoco.mj_geomDistance(m, d, g, FLOOR_GEOM, 0.2, FROMTO) for g in FOOT_GEOMS])
    print(f"  feet into the floor ≤ {max(-sunk, 0) * 100:.1f} cm, fastest hip rise or drop {np.abs(np.diff(qpos[:, 1])).max() * fps:.1f} m/s, "
          f"body moving {np.linalg.norm(com[-1] - com[-2]) * fps:.2f} m/s at the end")
    accel = np.gradient(np.gradient(com[:, [0, 2]], axis=0), axis=0) * fps**2
    zmp = com[:, [0, 2]] - com[:, 1:2] / 9.81 * accel
    single = np.flatnonzero(mask.sum(axis=1) == 1)
    toe, heel = soles[:, :, 1, [0, 2]], soles[:, :, 0, [0, 2]]
    carry = toe + FOREFOOT * (heel - toe)
    gaps = np.array([np.linalg.norm(zmp[k] - carry[k, int(np.argmax(mask[k]))]) for k in single])
    if len(gaps):
        worst = single[np.argsort(gaps)[-3:]]
        print(f"  on one foot: zero-moment point {np.median(gaps) * 100:.0f} cm from its forefoot (median), "
              f"{gaps.max() * 100:.0f} cm at worst (at {', '.join(f'{t[k]:.2f}' for k in sorted(worst))} s)")
    print(f"  hips {qpos[:, 1].min():.2f}–{qpos[:, 1].max():.2f} m, travel x {np.ptp(qpos[:, 0]):.2f} m, z {np.ptp(qpos[:, 2]):.2f} m")


def main(capture: str, fitted: str, start: float, end: float, out: str) -> None:
    video, fit = np.load(capture), np.load(fitted)
    keep = (fit["t"] >= start - 1e-6) & (fit["t"] <= end + 1e-6)
    t = fit["t"][keep]
    rows = [int(np.argmin(np.abs(video["t"] - x))) for x in t]
    fps = float(video["fps"])
    # The last pose is held a while, so the weight can travel to the middle of the base and stop there.
    hold = int(round(HOLD * fps))
    t = np.r_[t, t[-1] + np.arange(1, hold + 1) / fps]
    rows += [rows[-1]] * hold
    poses = np.r_[fit["qpos"][keep], np.repeat(fit["qpos"][keep][-1:], hold, axis=0)]
    qpos, feet, mask = plant(poses, video["joints"][rows], [str(n) for n in video["names"]], fps)
    np.savez(out, t=t, qpos=qpos, feet=feet, mask=mask, fps=fps, joints=fit["joints"])
    print(f"{len(t)} frames, {t[0]:.2f}–{t[-1]:.2f} s → {out}")
    report(t, qpos, feet, mask, fps)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
