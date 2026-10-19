"""Reference clips sampled by the environment: the looping ginga, the kicks and esquivas, the way up from a fall."""

import mujoco
import numpy as np

from env_consts import LANDINGS, MOVES
from env_math import nlerp


class Reference:
    """The looping reference cycle at any tempo, sampled by phase (0..1 over one cycle).

    The reference file holds one cycle per tempo on a grid (see reference.py);
    tempos in between blend their two neighbours.
    """

    def __init__(self, path: str, model: mujoco.MjModel):
        data = np.load(path)
        self.tempos = data["tempos"]
        self.periods = data["periods"]
        self.qpos = data["qpos"]  # (tempos, samples, nq)
        self.feet = data["feet"]  # (tempos, samples, 2 feet, x/z)
        self.triangle = data["triangle"]
        self.n = self.qpos.shape[1]
        self.model = model
        self.qvel = np.zeros((len(self.tempos), self.n, model.nv))
        for i, period in enumerate(self.periods):
            dt = period / self.n
            for k in range(self.n):
                mujoco.mj_differentiatePos(model, self.qvel[i, k], 2 * dt, self.qpos[i, k - 1], self.qpos[i, (k + 1) % self.n])

    @staticmethod
    def period(bpm: float) -> float:
        return 4 * 60 / bpm

    def _neighbours(self, bpm: float) -> tuple[int, int, float]:
        bpm = min(max(bpm, self.tempos[0]), self.tempos[-1])
        j = int(min(np.searchsorted(self.tempos, bpm), len(self.tempos) - 1))
        i = max(j - 1, 0)
        w = 0.0 if i == j else (bpm - self.tempos[i]) / (self.tempos[j] - self.tempos[i])
        return i, j, float(w)

    def _at(self, table: np.ndarray, phase: float) -> tuple[np.ndarray, np.ndarray, float]:
        x = (phase % 1.0) * self.n
        k = int(x) % self.n
        return table[k], table[(k + 1) % self.n], x - int(x)

    def sample(self, phase: float, bpm: float) -> tuple[np.ndarray, np.ndarray]:
        i, j, w = self._neighbours(bpm)
        out_q, out_v = [], []
        for t in (i, j):
            q0, q1, a = self._at(self.qpos[t], phase)
            q = q0 + (q1 - q0) * a
            q[3:7] = nlerp(q0[3:7], q1[3:7], a)
            v0, v1, _ = self._at(self.qvel[t], phase)
            out_q.append(q)
            out_v.append(v0 + (v1 - v0) * a)
        q = out_q[0] + (out_q[1] - out_q[0]) * w
        q[3:7] = nlerp(out_q[0][3:7], out_q[1][3:7], w)
        return q, out_v[0] + (out_v[1] - out_v[0]) * w

    def feet_at(self, phase: float, bpm: float) -> np.ndarray:
        i, j, w = self._neighbours(bpm)
        f = []
        for t in (i, j):
            f0, f1, a = self._at(self.feet[t], phase)
            f.append(f0 + (f1 - f0) * a)
        return f[0] + (f[1] - f[0]) * w


class Kicks:
    """Reference kicks (see kicks.py) and esquivas (esquiva.py), per move, leg and size, sampled by time.
    A move's side is the leg that is back in the base it starts from."""

    def __init__(self, paths: list[str], model: mujoco.MjModel):
        self.clips = {}
        for path in paths:
            data = np.load(path)
            fps = float(data["fps"])
            self.amplitudes = data["amplitudes"].tolist()
            for move in MOVES:
                if f"{move}_duration" not in data:
                    continue
                duration = float(data[f"{move}_duration"])
                for side in "lr":
                  for level in range(len(self.amplitudes)):
                    qpos = data[f"{move}_{side}_{level}_qpos"]
                    qvel = np.zeros((len(qpos), model.nv))
                    for k in range(len(qpos)):
                        a, b = qpos[max(k - 1, 0)], qpos[min(k + 1, len(qpos) - 1)]
                        span = (min(k + 1, len(qpos) - 1) - max(k - 1, 0)) / fps
                        mujoco.mj_differentiatePos(model, qvel[k], span, a, b)
                    self.clips[move, side, level] = {
                        "duration": duration,
                        "fps": fps,
                        "qpos": qpos,
                        "qvel": qvel,
                        "feet": data[f"{move}_{side}_{level}_feet"],
                        "mask": data[f"{move}_{side}_{level}_mask"],
                    }
        self.level = len(self.amplitudes) - 1  # size used when not training (the full move)

    def duration(self, move: str) -> float:
        return self.clips[move, "r", 0]["duration"]

    def sample(self, move: str, side: str, t: float, level: int | None = None):
        clip = self.clips[move, side, self.level if level is None else level]
        x = min(max(t, 0.0), clip["duration"]) * clip["fps"]
        k = min(int(x), len(clip["qpos"]) - 2)
        a = x - k
        q0, q1 = clip["qpos"][k], clip["qpos"][k + 1]
        q = q0 + (q1 - q0) * a
        q[3:7] = nlerp(q0[3:7], q1[3:7], a)
        v = clip["qvel"][k] + (clip["qvel"][k + 1] - clip["qvel"][k]) * a
        feet = clip["feet"][k] + (clip["feet"][k + 1] - clip["feet"][k]) * a
        mask = np.minimum(clip["mask"][k], clip["mask"][k + 1])
        return q, v, feet, mask


class Floor:
    """Falling and getting up (see floor.py): per landing and side, the way up from the landing pose
    to the ginga's base, with the feet's floor points, sampled by time."""

    def __init__(self, path: str, model: mujoco.MjModel):
        data = np.load(path)
        self.fps = float(data["fps"])
        sites = data["sites"].tolist()
        heel_toe = [[sites.index(f"heel_{s}"), sites.index(f"toe_{s}")] for s in "lr"]
        scratch = mujoco.MjData(model)
        sole = [[model.site(f"heel_{s}").id, model.site(f"toe_{s}").id] for s in "lr"]
        self.clips = {}
        for move in LANDINGS:
            for side in "lr":
                qpos = data[f"{move}_{side}_qpos"]
                qvel = np.zeros((len(qpos), model.nv))
                feet = np.zeros((len(qpos), 2, 2))
                for k in range(len(qpos)):
                    a, b = qpos[max(k - 1, 0)], qpos[min(k + 1, len(qpos) - 1)]
                    span = (min(k + 1, len(qpos) - 1) - max(k - 1, 0)) / self.fps
                    mujoco.mj_differentiatePos(model, qvel[k], span, a, b)
                    scratch.qpos[:] = qpos[k]
                    mujoco.mj_kinematics(model, scratch)
                    feet[k] = [scratch.site_xpos[ids].mean(axis=0)[[0, 2]] for ids in sole]
                mask = data[f"{move}_{side}_mask"][:, heel_toe].max(axis=2)
                self.clips[move, side] = {
                    "duration": float(data[f"{move}_{side}_duration"]),
                    "phase": float(data[f"{move}_{side}_phase"]),  # the ginga's phase at the base it ends in
                    "qpos": qpos,
                    "qvel": qvel,
                    "feet": feet,
                    "mask": mask,
                }

    def duration(self, move: str, side: str) -> float:
        return self.clips[move, side]["duration"]

    def sample(self, move: str, side: str, t: float):
        clip = self.clips[move, side]
        x = min(max(t, 0.0), clip["duration"]) * self.fps
        k = min(int(x), len(clip["qpos"]) - 2)
        a = x - k
        q0, q1 = clip["qpos"][k], clip["qpos"][k + 1]
        q = q0 + (q1 - q0) * a
        q[3:7] = nlerp(q0[3:7], q1[3:7], a)
        v = clip["qvel"][k] + (clip["qvel"][k + 1] - clip["qvel"][k]) * a
        feet = clip["feet"][k] + (clip["feet"][k + 1] - clip["feet"][k]) * a
        mask = np.minimum(clip["mask"][k], clip["mask"][k + 1])
        return q, v, feet, mask
