"""Reference motion for the esquiva lateral the live ginga can dodge with.

    npx --yes tsx@4.19.2 scripts/sample-clips.ts esquiva-lateral > ml/data/esquiva_clip.json   # from the web project
    uv run python esquiva.py                                                                    # → data/esquiva.npz

The esquiva starts from the app's keyframes (public/data/movements/esquiva-lateral.yaml),
sampled at 30 fps for both sides. It leaves from the ginga's base and comes back to it, like
a kick: from the base with the right leg back, the body drops at once onto the bent front (left)
leg while the back leg opens straight out to the side in a low step, the trunk lies over the bent
knee with the head out to the left and the eyes still on the partner — going down as fast as
gravity and the legs allow (drop), the head below 1 m in about 0.33 s and the bottom at about 0.5 s —
holds, comes back up on the front leg more slowly (LIFT_UP: rising off one deeply bent leg is the
hard part) while the back foot steps home, and is in the base again. The feet end up about a metre apart: on the triangle's front
corners (0.68 m apart) a straight far leg would hold the hips up at 0.8 m. The keyframes grow out
of the ginga's own base (the reference at a middle tempo) and melt back into it, as the kicks do
(kicks.py); the head keeps the ginga's angle, lifting only as far as the trunk leans further.

Here the poses become model poses, and inverse kinematics plants them: the front foot stays
flat on its corner; the back foot sits on its corner (on its ball, as in the base), steps low to
the wide spot, lands flat and steps home, on paths with no jump in acceleration where it stops.
The centre of mass follows com_path: a push from the back foot, a pendulum roll onto the front
foot while the back one is in the air (2–2.5 cm off the floor at mid-step), a little toward the wide foot at the bottom, and the
same in reverse. Hip and head height, the trunk's lean (the head out over the bent leg), flat
soles and legs and hands clear of the floor are targets of the same solve, because a keyframe
pose need not reach them once its feet are planted and its weight balanced.
"""

import json

import mujoco
import numpy as np

from env_math import nlerp
from esquiva_path import DODGE_SPECS, FROM_GINGA, SLOW, smooth
from esquiva_plant import limb_gaps, plant
from kicks import AMPLITUDES, GINGA_TEMPO, to_model
from reference import TRIANGLE, arm_clearance, d, fix_arms, m


def main() -> None:
    clips = json.load(open("data/esquiva_clip.json"))
    channels, fps = clips["channels"], clips["fps"]
    ginga = np.load("data/ginga_ref.npz")["qpos"][GINGA_TEMPO]
    out = {}
    for move_name, spec in DODGE_SPECS.items():
        clip = clips[spec["clip"]]
        for side in "lr":
            # Start and end in the ginga's base with this leg back: phase 0 (right) or 0.5 (left).
            base = ginga[0 if side == "r" else len(ginga) // 2]
            keyframed = np.array([to_model(dict(zip(channels, f))) for f in clip["frames"][side]])
            neck = [m.joint(f"neck_{f}").qposadr[0] for f in ("flex", "side", "twist")]
            keyframed[:, neck] = base[neck]  # the head keeps the ginga's angle (the keyframes' is the app's own)
            duration = (len(keyframed) - 1) / fps
            for level, amplitude in enumerate(AMPLITUDES):
                qpos = keyframed.copy()
                for k, q in enumerate(qpos):
                    t = k / fps
                    s = smooth(min(t, duration - t) / FROM_GINGA) * amplitude
                    quat = nlerp(base[3:7], q[3:7], s)
                    q[:] = base + (q - base) * s
                    q[3:7] = quat
                qpos, feet, mask = plant(qpos, side, amplitude, fps, base, spec)
                qpos = fix_arms(qpos, periodic=False)
                out[f"{move_name}_{side}_{level}_qpos"] = qpos
                out[f"{move_name}_{side}_{level}_feet"] = feet
                out[f"{move_name}_{side}_{level}_mask"] = mask
                head, coms, slide = [], [], []
                for q in qpos:
                    d.qpos[:] = q
                    mujoco.mj_kinematics(m, d)
                    mujoco.mj_comPos(m, d)
                    head.append(d.xpos[m.body("head").id][1])
                    coms.append(d.subtree_com[1][[0, 2]].copy())
                    slide.append(d.site_xpos[[m.site(f"heel_{side}").id, m.site(f"toe_{side}").id]].copy())
                coms, slide = np.array(coms), np.array(slide)
                front = TRIANGLE[0 if side == "r" else 1]
                times = np.arange(len(qpos)) / fps
                sink = -np.diff(qpos[:, 1]) * fps / SLOW
                arm_leg, floor_gap = limb_gaps(qpos)
                print(
                    f"{move_name} ({side}, size {amplitude}): {duration * SLOW:.2f} s; hips {qpos[:, 1].min():.2f}–{qpos[:, 1].max():.2f} m "
                    f"(lowest at {times[np.argmin(qpos[:, 1])]:.2f} s), head down to {min(head):.2f} m (centre), "
                    f"feet up to {np.abs(feet[:, 0, 0] - feet[:, 1, 0]).max():.2f} m apart, "
                    f"centre of mass up to {np.linalg.norm(coms - front, axis=1).max() * 100:.0f} cm from the front foot, "
                    f"hips dropping up to {sink.max():.2f} m/s, rising up to {-sink.min():.2f} m/s, "
                    f"sliding sole {slide[:, :, 1].min() * 100:+.1f}…{slide[:, :, 1].max() * 100:+.1f} cm off the floor, "
                    f"arms to legs {arm_leg * 100:.1f} cm, legs and hands to the floor {floor_gap * 100:.1f} cm, "
                    f"arm–head {arm_clearance(qpos) * 100:.1f} cm"
                )
            out[f"{move_name}_duration"] = np.array(duration * SLOW)
    np.savez("data/esquiva.npz", fps=fps / SLOW, moves=np.array(list(DODGE_SPECS.keys())), amplitudes=np.array(AMPLITUDES), **out)


if __name__ == "__main__":
    main()
