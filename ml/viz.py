"""Render qpos frames from an .npz (t, qpos) into a strip of images, front and side.

    uv run python viz.py data/ginga_fit.npz out.png [t1 t2 ...]
    uv run python viz.py data/floor.npz out.png negativa_r [t1 t2 ...]   # one clip of floor.npz (the way up, from the landing)
    uv run python viz.py data/falls.npz out.png 0 1 2 3                  # states of a pool, by index

When the file also holds the targets (check.py --record), two more rows show
the reference the body was following at the same moments, under the body.
"""

import sys

import mujoco
import numpy as np
from PIL import Image


def render(qposes, xml="humanoid.xml", size=(240, 320)):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=size[1], width=size[0])
    rows = []
    for camera in ("front", "side"):  # fixed Y-up cameras defined in the model
        tiles = []
        for q in qposes:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=camera)
            tiles.append(renderer.render())
        rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0)


if __name__ == "__main__":
    data = dict(np.load(sys.argv[1]))
    args = sys.argv[3:]
    if "t" not in data and "qpos" in data:
        data["t"] = np.arange(len(data["qpos"]), dtype=float)  # a pool of states (falls.npz): times are indices
    elif "t" not in data:
        clip = args.pop(0)
        data = {"qpos": data[f"{clip}_qpos"], "t": np.arange(len(data[f"{clip}_qpos"])) / float(data["fps"])}
    times = [float(x) for x in args] or list(np.linspace(data["t"][0], data["t"][-1], 8))
    picks = [int(np.argmin(np.abs(data["t"] - t))) for t in times]
    image = render([data["qpos"][i] for i in picks])
    if "target" in data:
        image = np.concatenate([image, render([data["target"][i] for i in picks])], axis=0)
    Image.fromarray(image).save(sys.argv[2])
    print("saved", sys.argv[2], [round(float(data["t"][i]), 2) for i in picks])
    if "label" in data:
        print("labels", [str(data["label"][i]) for i in picks])
