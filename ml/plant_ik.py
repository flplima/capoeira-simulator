"""Inverse kinematics shared by the solvers that plant a pose on the floor: the legs and lower
trunk move, the whole body may shift and turn a little, the arms and head stay as posed."""

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

from reference import LEG_JOINTS, d, m

IK_JOINTS = LEG_JOINTS + ["spine_flex", "spine_side", "chest_flex", "chest_side"]
IK_ADDR = np.array([m.joint(n).qposadr[0] for n in IK_JOINTS])
TILT = 0.6  # rad: how far the solver may turn the whole body from the pose it was given
# The solved vector: root position (3), a turn of the whole body (3, axis × angle), then IK_JOINTS.
LOWER = np.r_[-np.inf, 0.3, -np.inf, [-TILT] * 3, [m.jnt_range[m.joint(n).id][0] for n in IK_JOINTS]]
UPPER = np.r_[np.inf, 1.3, np.inf, [TILT] * 3, [m.jnt_range[m.joint(n).id][1] for n in IK_JOINTS]]
SOLE = {s: [m.site(f"heel_{s}").id, m.site(f"toe_{s}").id] for s in "lr"}
LEG_GEOMS = {s: [m.geom(f"{p}_{s}").id for p in ("thigh", "shin", "foot")] for s in "lr"}
SPINE_GEOM = m.geom("spine").id
FOOT_GEOMS = [m.geom(f"foot_{s}").id for s in "lr"]
FLOOR_GEOM = m.geom("floor").id
FROMTO = np.zeros(6)


def initial(q: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    """Where a frame's solve starts: the pose itself, or the previous frame's root and turn with this pose's joints."""
    x = np.r_[q[:3], 0, 0, 0, q[IK_ADDR]] if previous is None else np.r_[previous[:6], q[IK_ADDR]]
    x[1] = np.clip(x[1], LOWER[1] + 0.01, UPPER[1] - 0.01)
    return np.clip(x, LOWER + 1e-6, UPPER - 1e-6)


def set_pose(q: np.ndarray, v: np.ndarray) -> None:
    """Load pose q with the solved root, turn and joints v into d, with positions and centre of mass."""
    d.qpos[:] = q
    d.qpos[:3] = v[:3]
    turn = np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, v[3:6] / (np.linalg.norm(v[3:6]) + 1e-12), np.linalg.norm(v[3:6]))
    mujoco.mju_mulQuat(d.qpos[3:7], turn, q[3:7])
    d.qpos[IK_ADDR] = v[6:]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)


def on_floor(spot: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """The foot's own shape (as posed) moved so its middle is on spot (xz) and its lowest point on the floor."""
    middle = sites.mean(axis=0)
    moved = sites + np.array([spot[0] - middle[0], 0, spot[1] - middle[2]])
    moved[:, 1] -= moved[:, 1].min()
    return moved


def legs_apart() -> list[float]:
    """Residuals that are zero unless the legs touch each other or the spine."""
    out = [20.0 * min(mujoco.mj_geomDistance(m, d, a, b, 0.1, FROMTO) - 0.02, 0.0) for a in LEG_GEOMS["l"] for b in LEG_GEOMS["r"]]
    out += [20.0 * min(mujoco.mj_geomDistance(m, d, g, SPINE_GEOM, 0.1, FROMTO) - 0.01, 0.0) for g in LEG_GEOMS["l"] + LEG_GEOMS["r"]]
    return out


def feet_above_floor() -> list[float]:
    """Residuals that are zero unless a foot, tipped on its edge, goes into the floor."""
    return [40.0 * min(mujoco.mj_geomDistance(m, d, g, FLOOR_GEOM, 0.2, FROMTO), 0.0) for g in FOOT_GEOMS]


def smooth_and_ground(out: np.ndarray) -> np.ndarray:
    """Light smoothing of what the solver chose, then the lowest sole back on the floor."""
    out = out.copy()
    for i in range(1, len(out)):  # one sign per quaternion, so a full turn doesn't flip when smoothed
        if np.dot(out[i - 1, 3:7], out[i, 3:7]) < 0:
            out[i, 3:7] *= -1
    columns = np.r_[0, 1, 2, 3, 4, 5, 6, IK_ADDR]
    out[:, columns] = gaussian_filter1d(out[:, columns], 1.0, axis=0, mode="nearest")
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    for q in out:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        q[1] -= min(d.site_xpos[i][1] for s in "lr" for i in SOLE[s])
    return out
