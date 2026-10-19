"""Pacing a planted way up to what a torque-limited body can follow."""

import mujoco
import numpy as np

from env_math import nlerp
from floor_plant import SITES, smoothstep
from reference import d, m

# No faster than this (m/s, real time): the centre of mass rising, and moving along the floor. Where the
# keyframes go faster, the clip is stretched (a torque-limited body can't stand up from the floor in a jump).
RISE_SPEED, SHIFT_SPEED = 0.8, 1.0
LIMB_SPEED = 2.5  # m/s: nor any hand, knee or foot faster than this


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
