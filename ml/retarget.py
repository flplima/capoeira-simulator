"""Fit the MuJoCo skeleton to MediaPipe landmarks, frame by frame.

    uv run python retarget.py data/ginga.npz <start_s> <end_s> data/ginga_fit.npz

MediaPipe's proportions are not the mannequin's, so the fit matches segment
directions (thigh, shin, foot, upper arm, forearm, pelvis and shoulder lines,
trunk, head) rather than positions. Joints the video can't see well — trunk
twist split between spine and chest, ankle inversion — are regularised toward
each other and toward the previous frame.
"""

import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

MP = {
    "nose": 0, "ear_l": 7, "ear_r": 8, "shoulder_l": 11, "shoulder_r": 12, "elbow_l": 13, "elbow_r": 14,
    "wrist_l": 15, "wrist_r": 16, "hip_l": 23, "hip_r": 24, "knee_l": 25, "knee_r": 26,
    "ankle_l": 27, "ankle_r": 28, "heel_l": 29, "heel_r": 30, "toe_l": 31, "toe_r": 32,
}
# (from, to, weight): model sites on the left, MediaPipe points on the right share names.
PAIRS = [
    ("hip_r", "hip_l", 3.0), ("shoulder_r", "shoulder_l", 2.0), ("mid_hip", "mid_shoulder", 3.0),
    ("hip_l", "knee_l", 2.0), ("knee_l", "ankle_l", 2.0), ("heel_l", "toe_l", 1.0),
    ("hip_r", "knee_r", 2.0), ("knee_r", "ankle_r", 2.0), ("heel_r", "toe_r", 1.0),
    ("shoulder_l", "elbow_l", 1.0), ("elbow_l", "wrist_l", 1.0),
    ("shoulder_r", "elbow_r", 1.0), ("elbow_r", "wrist_r", 1.0),
    ("mid_shoulder", "nose", 0.7), ("ear_r", "ear_l", 0.5),
]
SITE_FOR = {"hip_l": "hip_joint_l", "hip_r": "hip_joint_r"}


def to_y_up(points: np.ndarray) -> np.ndarray:
    """MediaPipe world (x right, y down, z away from camera) → x left-of-subject, y up, z toward camera."""
    return points * np.array([1.0, -1.0, -1.0])


def landmark(world: np.ndarray, name: str) -> np.ndarray:
    if name == "mid_hip":
        return (world[MP["hip_l"]] + world[MP["hip_r"]]) / 2
    if name == "mid_shoulder":
        return (world[MP["shoulder_l"]] + world[MP["shoulder_r"]]) / 2
    return world[MP[name]]


class Fitter:
    def __init__(self, xml: str = "humanoid.xml"):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.joint_ids = [self.m.joint(i).id for i in range(1, self.m.njnt)]
        self.lo = np.array([self.m.jnt_range[j][0] for j in self.joint_ids])
        self.hi = np.array([self.m.jnt_range[j][1] for j in self.joint_ids])
        self.names = [self.m.joint(j).name for j in self.joint_ids]
        self.site = {n: self.m.site(n).id for n in [self.m.site(i).name for i in range(self.m.nsite)]}
        idx = {n: i for i, n in enumerate(self.names)}
        self.tied = [(idx[f"spine_{a}"], idx[f"chest_{a}"]) for a in ("twist", "flex", "side")]
        self.quiet = [idx["ankle_l_inv"], idx["ankle_r_inv"], idx["neck_twist"]]

    def point(self, name: str) -> np.ndarray:
        if name == "mid_hip":
            return (self.point("hip_l") + self.point("hip_r")) / 2
        if name == "mid_shoulder":
            return (self.point("shoulder_l") + self.point("shoulder_r")) / 2
        return self.d.site_xpos[self.site[SITE_FOR.get(name, name)]]

    def pose(self, x: np.ndarray) -> np.ndarray:
        qpos = np.zeros(self.m.nq)
        qpos[1] = 0.97
        qpos[3:7] = Rotation.from_rotvec(x[:3]).as_quat(scalar_first=True)
        qpos[7:] = x[3:]
        return qpos

    def residuals(self, x: np.ndarray, target: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        self.d.qpos[:] = self.pose(x)
        mujoco.mj_kinematics(self.m, self.d)
        out = []
        for a, b, w in PAIRS:
            model = self.point(b) - self.point(a)
            seen = landmark(target, b) - landmark(target, a)
            out.extend(np.sqrt(w) * (model / np.linalg.norm(model) - seen / max(np.linalg.norm(seen), 1e-6)))
        joints = x[3:]
        out.extend(0.6 * (joints[i] - joints[j]) for i, j in self.tied)
        out.extend(0.3 * joints[i] for i in self.quiet)
        out.extend(0.05 * joints)
        if previous is not None:
            out.extend(0.4 * (x - previous))
        return np.array(out)

    def fit(self, target: np.ndarray, guess: np.ndarray) -> np.ndarray:
        lo = np.r_[[-np.inf] * 3, self.lo]
        hi = np.r_[[np.inf] * 3, self.hi]
        return least_squares(self.residuals, np.clip(guess, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), args=(target, guess), max_nfev=200).x


def main(src: str, start: float, end: float, out: str) -> None:
    data = np.load(src)
    keep = (data["t"] >= start) & (data["t"] <= end)
    times, world = data["t"][keep], to_y_up(data["world"][keep])
    fitter = Fitter()
    x = np.zeros(3 + len(fitter.joint_ids))
    fits = []
    for i, frame in enumerate(world):
        x = fitter.fit(frame, x)
        fits.append(fitter.pose(x))
        if i % 30 == 0:
            print(f"{times[i]:.2f}s  cost {np.sum(fitter.residuals(x, frame, None)[: len(PAIRS) * 3] ** 2):.3f}")
    np.savez(out, t=times, qpos=np.array(fits), joints=np.array(fitter.names))
    print(f"{len(fits)} frames → {out}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
