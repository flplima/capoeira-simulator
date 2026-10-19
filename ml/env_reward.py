"""GingaEnv's reward: tracking the reference, plus the strike and pivot of a kick and the soft landing of a fall."""

import mujoco
import numpy as np

from env_consts import EFFECTORS, FALL_TIME, HEAD_ABOVE, KICK_MOVES, PIVOT_SCALE, PIVOT_WEIGHT, STRIKE_SCALE, STRIKE_WEIGHT
from env_math import quat_angle


class RewardTerms:
    """GingaEnv._reward."""

    def _reward(self) -> tuple[float, dict]:
        q_ref, v_ref, feet_ref, feet_mask = self.choreo.target(0.0)
        q, v = self.d.qpos, self.d.qvel
        joint_err = np.sum((q[self.joint_qpos] - q_ref[self.joint_qpos]) ** 2)
        root_rot_err = quat_angle(q[3:7], q_ref[3:7]) ** 2
        vel_err = np.sum((v[self.joint_qvel] - v_ref[self.joint_qvel]) ** 2)
        self.ref_data.qpos[:] = q_ref
        mujoco.mj_kinematics(self.m, self.ref_data)
        ee_err = np.sum((self._local_effectors(self.d) - self._local_effectors(self.ref_data)) ** 2)
        r = self.choreo.recover
        root_err = (q[1] - q_ref[1]) ** 2 if (r is not None and r["t"] < FALL_TIME) else np.sum((q[:3] - q_ref[:3]) ** 2)
        feet_err = np.sum(feet_mask[:, None] * (self._foot_points() - feet_ref) ** 2)
        parts = {
            "pose": np.exp(-2.0 * (joint_err + root_rot_err)),
            "vel": np.exp(-0.02 * vel_err),
            "ee": np.exp(-40.0 * ee_err / len(EFFECTORS)),
            "root": np.exp(-10.0 * root_err),
            "feet": np.exp(-20.0 * feet_err),
        }
        parts["kicking"] = float(self.choreo.move is not None)  # logged only: share of time spent kicking
        parts["down"] = float(self.choreo.recover is not None)  # logged only: share of time falling or getting up
        parts["up"] = float(self.after_recover is not None)  # logged only: share of time back in the ginga after getting up
        parts["size"] = float(self.fall_level if self.skill == "recover" else self.curriculum)  # logged only: curriculum level
        reward = 0.4 * parts["pose"] + 0.1 * parts["vel"] + 0.15 * parts["ee"] + 0.15 * parts["root"] + 0.2 * parts["feet"]
        parts["strike"] = parts["pivot"] = 0.0  # logged as their means while kicking
        if self.choreo.move in KICK_MOVES:
            # Where the kick goes: the kicking foot from the hips as the room sees it. ee turns everything into
            # the body's own heading, so a martelo thrown out to the side without turning the hips over (and
            # pivoting on the support foot) still scored there; this only pays for the foot where the reference's is.
            foot = self.effector_ids[EFFECTORS.index(f"foot_{self.choreo.side}")]
            reach_err = np.sum(((self.d.xpos[foot] - q[:3]) - (self.ref_data.xpos[foot] - q_ref[:3])) ** 2)
            parts["strike"] = np.exp(-STRIKE_SCALE * reach_err)
            # The support foot turning with the hips (the martelo's pivot on the ball of the foot), as the
            # reference's heel-to-toe line: the feet term only places the foot's middle, and the body pivoted
            # the other way.
            support = "l" if self.choreo.side == "r" else "r"
            heel, toe = self.sole_sites["lr".index(support)]
            heading = lambda data: np.arctan2(*(data.site_xpos[toe] - data.site_xpos[heel])[[0, 2]])
            turn_err = (heading(self.d) - heading(self.ref_data) + np.pi) % (2 * np.pi) - np.pi
            parts["pivot"] = np.exp(-PIVOT_SCALE * turn_err**2)
            reward = (1 - STRIKE_WEIGHT - PIVOT_WEIGHT) * reward + STRIKE_WEIGHT * parts["strike"] + PIVOT_WEIGHT * parts["pivot"]
        if r is not None and r["t"] < FALL_TIME + 0.3:
            # Going down softly: hips, chest and head lowered, not dropped, and angular velocity damped.
            speed = np.zeros(6)
            drop = 0.05 * float(np.sum(v[3:6] ** 2))
            for body in self.soft_bodies:
                mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, body, speed, 0)
                drop += min(speed[4], 0.0) ** 2
            parts["soft"] = np.exp(-0.5 * drop)
            # Hips first, head last: diving head first over the hips is how the head hits the floor.
            head, hips = self.d.xpos[self.soft_bodies[2]][1], self.d.xpos[self.soft_bodies[0]][1]
            parts["head"] = np.exp(-10.0 * max(hips + HEAD_ABOVE - head, 0.0) ** 2)
            reward = 0.75 * reward + 0.1 * parts["soft"] + 0.15 * parts["head"]
        return float(reward), parts
