"""Quaternion, heading and floor-placement helpers shared by the environment and the reference tools."""

import mujoco
import numpy as np

from env_consts import SIGMA_EMA, SIGMA_FLOOR, SIGMA_INIT

# Small quaternion helpers (w, x, y, z) — scipy's Rotation is too slow per step.


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) of the rotation between two unit quaternions."""
    return 2.0 * np.arccos(min(1.0, abs(float(np.dot(a, b)))))


def tighten(sigma: np.ndarray, mean_err: np.ndarray, count: np.ndarray, ema: np.ndarray, ratchet: bool) -> tuple[np.ndarray, np.ndarray]:
    """The tolerances after a training iteration (env_consts.TERMS): each term's running mean error where the term was
    seen (count > 0) — and, once `ratchet` (past SIGMA_WARMUP iterations), only ever tightened. Until then, and for a
    term seen for the first time, the running mean is just this iteration's mean, so the ratchet starts from a
    level the first, unrepresentative iterations haven't pulled down."""
    seen = count > 0
    first = ema >= SIGMA_INIT
    ema = np.where(seen, np.where(first | (not ratchet), mean_err, ema + SIGMA_EMA * (mean_err - ema)), ema)
    level = np.maximum(ema, SIGMA_FLOOR)
    return np.where(seen, np.minimum(sigma, level) if ratchet else level, sigma), ema


def nlerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    if np.dot(a, b) < 0:
        b = -b
    q = a + (b - a) * t
    return q / np.linalg.norm(q)


def yaw_of(q: np.ndarray) -> float:
    """Heading about the vertical Y axis: the direction the body's +Z axis points, projected on the floor."""
    w, x, y, z = q
    fx = 2 * (x * z + w * y)
    fz = 1 - 2 * (x * x + y * y)
    return float(np.arctan2(fx, fz))


def yaw_matrix(yaw: float) -> np.ndarray:
    """Rotation taking world vectors into the heading frame (undoing the yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def quat_matrix(q: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


# A placement on the floor: (x offset, z offset, yaw). The references live in
# their own frame (the ginga on its triangle, a get-up clip where it was
# solved); a placement puts them somewhere else in the world, turned about
# the vertical. Yaw turns +Z (forward) towards +X.


def place_xz(frame: np.ndarray, xz: np.ndarray) -> np.ndarray:
    """Reference-frame floor points (..., 2) → world."""
    c, s = np.cos(frame[2]), np.sin(frame[2])
    x, z = xz[..., 0], xz[..., 1]
    return np.stack([c * x + s * z + frame[0], -s * x + c * z + frame[1]], axis=-1)


def unplace_xz(frame: np.ndarray, xz: np.ndarray) -> np.ndarray:
    """World floor points (..., 2) → the reference frame."""
    c, s = np.cos(frame[2]), np.sin(frame[2])
    x, z = xz[..., 0] - frame[0], xz[..., 1] - frame[1]
    return np.stack([c * x - s * z, s * x + c * z], axis=-1)


def place_pose(frame: np.ndarray, q: np.ndarray, v: np.ndarray, feet: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A reference state (qpos, qvel, feet floor points) → world. The free joint's angular velocity
    is in the body's own frame, so only the linear part turns."""
    q, v = q.copy(), v.copy()
    q[[0, 2]] = place_xz(frame, q[[0, 2]])
    q[3:7] = quat_mul(np.array([np.cos(frame[2] / 2), 0.0, np.sin(frame[2] / 2), 0.0]), q[3:7])
    c, s = np.cos(frame[2]), np.sin(frame[2])
    v[0], v[2] = c * v[0] + s * v[2], -s * v[0] + c * v[2]
    return q, v, place_xz(frame, feet)


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)
