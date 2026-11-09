"""Play a slow-motion replay at the speed of life: resample a stretch of hmr_modal.py's .npz by a factor.

    uv run python clip_speed.py data/hmr/ZCLVQIfaPHE_6.8s.npz 4 13.5 33 data/martelo_real.npz

The frames between start and end, replayed `speed` times slower than life (the time a foot spends in the air in the
replay over the time in the full-speed take of the same move), are resampled to FPS at real time; the clip keeps
its start time, so the move's times in it shrink towards that. A replay has several times the frames of the take,
and its poses are the cleaner ones, so it makes the better reference once its clock is right.
"""

import sys

import numpy as np

from hmr_points import FPS


def main() -> None:
    src, speed, start, end, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    clip = np.load(src)
    t = clip["t"]
    real = start + np.arange(0.0, (end - start) / speed, 1 / FPS)
    at = start + (real - start) * speed

    def resample(a: np.ndarray) -> np.ndarray:
        flat = a.reshape(len(t), -1)
        return np.stack([np.interp(at, t, flat[:, k]) for k in range(flat.shape[1])], axis=1).reshape((len(real), *a.shape[1:]))

    data = {k: clip[k] for k in clip.files}
    data.update({k: resample(clip[k]) for k in ("world", "visibility", "root", "joints")}, t=real)
    np.savez(out, **data)
    print(f"{len(real)} frames, {start}–{real[-1]:.2f} s at the speed of life ← {start}–{end} s of {src}, {speed:g}× slower")


if __name__ == "__main__":
    main()
