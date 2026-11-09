"""GingaEnv's reward: tracking the reference within the tolerances the training sets (env_consts.TERMS), plus the
strike and pivot of a kick and the soft landing of a fall."""

import mujoco
import numpy as np

from env_consts import EFFECTORS, FALL_TIME, HEAD_ABOVE, KICK_MOVES, MOVES, PIVOT_WEIGHT, STRIKE_WEIGHT, TERMS
from env_math import quat_angle


class RewardTerms:
    """GingaEnv._reward."""

    def _reward(self) -> tuple[float, dict]:
        q_ref, v_ref, feet_ref, feet_mask = self.choreo.target(0.0)
        q, v = self.d.qpos, self.d.qvel
        self.ref_data.qpos[:] = q_ref
        mujoco.mj_kinematics(self.m, self.ref_data)
        r = self.choreo.recover
        kicking = self.choreo.move in KICK_MOVES
        # Each term's squared error, in TERMS order; a kick's own terms stay 0 outside a kick.
        err = np.zeros(len(TERMS))
        err[0] = np.sum((q[self.joint_qpos] - q_ref[self.joint_qpos]) ** 2) + quat_angle(q[3:7], q_ref[3:7]) ** 2
        err[1] = np.sum((v[self.joint_qvel] - v_ref[self.joint_qvel]) ** 2)
        err[2] = np.sum((self._local_effectors(self.d) - self._local_effectors(self.ref_data)) ** 2) / len(EFFECTORS)
        err[3] = (q[1] - q_ref[1]) ** 2 if (r is not None and r["t"] < FALL_TIME) else np.sum((q[:3] - q_ref[:3]) ** 2)
        err[4] = np.sum(feet_mask[:, None] * (self._foot_points() - feet_ref) ** 2)
        if kicking:
            # Where the kick goes: the kicking foot from the hips as the room sees it. ee turns everything into
            # the body's own heading, so a martelo thrown out to the side without turning the hips over (and
            # pivoting on the support foot) still scored there; this only pays for the foot where the reference's is.
            foot = self.effector_ids[EFFECTORS.index(f"foot_{self.choreo.side}")]
            err[5] = np.sum(((self.d.xpos[foot] - q[:3]) - (self.ref_data.xpos[foot] - q_ref[:3])) ** 2)
            # The support foot turning with the hips as the reference's does (the martelo's pivot on the ball of
            # the foot): its heel-to-toe line, the heel's height and the ball staying where it is. The feet term
            # only places the foot's middle, and the body pivoted the other way, or flat, sliding the foot.
            support = "l" if self.choreo.side == "r" else "r"
            heel, toe = self.sole_sites["lr".index(support)]
            heading = lambda data: np.arctan2(*(data.site_xpos[toe] - data.site_xpos[heel])[[0, 2]])
            err[6] = ((heading(self.d) - heading(self.ref_data) + np.pi) % (2 * np.pi) - np.pi) ** 2
            err[7] = (self.d.site_xpos[heel][1] - self.ref_data.site_xpos[heel][1]) ** 2
            err[8] = np.sum((self.d.site_xpos[toe] - self.ref_data.site_xpos[toe])[[0, 2]] ** 2)
        row = 1 + 2 * MOVES.index(self.choreo.move) + "lr".index(self.choreo.side) if self.choreo.move else 0
        scores = np.exp(-err / self.sigma[row])
        parts = dict(zip(TERMS[:6], scores[:6].tolist()))
        parts["pivot"] = float(np.prod(scores[6:]))
        parts["err"], parts["row"], parts["kick"] = err, row, float(kicking)  # for the trainer's tolerances (env_math.tighten)
        parts["kicking"] = float(self.choreo.move is not None)  # logged only: share of time in a move
        parts["down"] = float(self.choreo.recover is not None)  # logged only: share of time falling or getting up
        parts["up"] = float(self.after_recover is not None)  # logged only: share of time back in the ginga after getting up
        parts["size"] = float(self.fall_level if self.skill == "recover" else self.curriculum)  # logged only: curriculum level
        reward = 0.4 * parts["pose"] + 0.1 * parts["vel"] + 0.15 * parts["ee"] + 0.15 * parts["root"] + 0.2 * parts["feet"]
        if kicking:
            reward = (1 - STRIKE_WEIGHT - PIVOT_WEIGHT) * reward + STRIKE_WEIGHT * parts["strike"] + PIVOT_WEIGHT * parts["pivot"]
        else:
            parts["strike"] = parts["pivot"] = 0.0  # logged as their means while kicking
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
