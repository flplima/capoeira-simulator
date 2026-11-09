"""The other leg's version of a move, from one leg's clip: mirror hmr_modal.py's .npz left for right.

    uv run python clip_mirror.py data/martelo_real.npz data/martelo_real_mirror.npz

A video often shows a move well on one leg only (the other one chained into a combination, or mistracked), and a
mirror image of a kick is a kick with the other leg: every joint's x is negated and the left and right joints
swap names, then the clip is placed like any other (hmr_points.to_capture).
"""

import sys

import numpy as np

from hmr_points import to_capture


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    clip = np.load(src)
    names = [str(n) for n in clip["names"]]
    swapped = [n.replace("left_", "~").replace("right_", "left_").replace("~", "right_") for n in names]
    joints = clip["joints"][:, [names.index(n) for n in swapped]] * np.array([-1.0, 1.0, 1.0])
    np.savez(out, **to_capture(joints, names, float(clip["t"][0])))
    print(f"{len(joints)} frames mirrored → {out}")


if __name__ == "__main__":
    main()
