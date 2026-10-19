"""Reference motions for falling and getting back up: queda de quatro, negativa and hands and knees.

    npx --yes tsx@4.19.2 scripts/sample-clips.ts queda-de-quatro negativa > ml/data/floor_clips.json   # from the web project
    uv run python floor.py                                                                               # → data/floor.npz

A capoeirista who loses balance doesn't crash: falling backwards they sit
into the queda de quatro (hands and feet on the floor, belly up), falling
sideways they drop into the negativa (one leg folded under, the other
stretched out, a hand on the floor), and falling forwards they catch
themselves on their hands and knees. The landings are the poses of the
game's keyframes in public/data/movements (hands and knees, not a move of
the game, is keyed here). The ways up are keyed here too (UP), so that the
body never has to throw itself up from the floor: every hand or foot lets go
only once the weight is over what stays down. From the queda de quatro the
hands push the hips forward over the feet into a low squat (the cocorinha's)
and the body rises from there; from the negativa the stretched leg drags its
heel in and plants the foot in front, the hand pushes the weight onto it,
and the body rises from a low lunge with the folded leg back; from hands and
knees one foot steps forward beside its hand, the hips move over it, the
hands push off into a low lunge, and the body rises with the other foot
back. Each ends in the ginga's base, and the ginga carries on from there.

So each floor move gives two pieces here: the landing pose (the keyframe
held on the floor), which the body aims for while it falls, and the way
up, from that pose to the ginga's base. Inverse kinematics puts them on
the floor: the hands and feet that the keyframes rest on touch it and
don't slide while they bear weight (but a dragged heel), the centre of
mass sits over the points that bear it, nothing sinks into the floor, no
thigh passes through the trunk, and the last frames melt into the ginga's
own base pose (the reference at a middle tempo), so the ginga can take
over where the body stands up. The clip is solved from its end backwards:
the end is the base on the triangle, and every earlier frame keeps the
floor points of the frame after it — except a point that lifts before the
end (a hand, a knee), which stays where it touched down in the landing
(see plant). A point takes and gives up its share of the weight gradually,
and a foot stepping to its next spot goes through the air.

A keyframe clip need not be followable by a torque-limited body, so the
solved clip is then paced (stretched wherever the body would rise, travel
or swing a limb too fast), free arms are slowed, and a few frames where the
solve jumps to another solution are blended over.
"""

import argparse
import json

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

from env_math import nlerp
from floor_keys import HAND, MOVES, ends_in, way_up
from floor_pace import calm_arms, despike, pace, resample
from floor_plant import SITES, plant
from kicks import GINGA_TEMPO, to_model
from reference import arm_clearance, d, fix_arms, m

# The way up plays this much slower than the keyframes: standing up from the floor
# is the heaviest thing the body does, and its joints are torque-limited.
SLOW = 2.0
# The last keyframe-seconds melt into the ginga's base.
TO_GINGA = 0.35


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/floor.npz")
    args = parser.parse_args()
    clips = json.load(open("data/floor_clips.json"))
    channels = clips["channels"]
    fps = clips["fps"]
    ginga = np.load("data/ginga_ref.npz")["qpos"][GINGA_TEMPO]
    out = {}
    for move in MOVES:
        clip = way_up(clips, move)
        for side in "lr":
            end = ends_in(move, side)
            base = ginga[0 if end == "r" else len(ginga) // 2]
            frames = clip["frames"][side]
            times = np.arange(len(frames)) / fps
            qpos = np.array([to_model(dict(zip(channels, f))) for f in frames])
            # Melt into the ginga's base over the last TO_GINGA seconds.
            for k, t in enumerate(times):
                s = np.clip(1 - (clip["duration"] - t) / TO_GINGA, 0.0, 1.0)
                s = s * s * (3 - 2 * s)
                quat = nlerp(qpos[k, 3:7], base[3:7], s)
                qpos[k] = qpos[k] + (base - qpos[k]) * s
                qpos[k, 3:7] = quat
            qpos[-1] = base
            # Plant once to see how fast the keyframes go, then again at the pace a body can follow (twice:
            # the pace changes the solve a little).
            at = times
            for _ in range(2):
                at = pace(plant(resample(qpos, times, at), at, move, side, clip["duration"])[0], at, SLOW / fps)
            qpos, mask = plant(resample(qpos, times, at), at, move, side, clip["duration"])
            # The guard arm clears the head; arms holding the body up stay where they are (easing between the two).
            hands = mask[:, [SITES.index(n) for n in HAND["l"] + HAND["r"]]].any(axis=1)
            keep = np.clip(gaussian_filter1d(hands.astype(float), 3.0, mode="nearest") * 2, 0.0, 1.0)[:, None]
            qpos = keep * qpos + (1 - keep) * fix_arms(qpos.copy(), periodic=False)
            qpos = despike(calm_arms(qpos, mask, SLOW / fps), SLOW / fps)
            qpos[-1] = base
            d.qpos[:] = qpos[0]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            duration = (len(qpos) - 1) / fps * SLOW
            out[f"{move}_{side}_qpos"] = qpos
            out[f"{move}_{side}_mask"] = mask
            out[f"{move}_{side}_duration"] = np.array(duration)
            out[f"{move}_{side}_phase"] = np.array(0.0 if end == "r" else 0.5)
            print(
                f"{move} ({side}): way up {duration:.2f} s; landing hips {qpos[0, 1]:.2f} m, head {d.xpos[m.body('head').id][1]:.2f} m, "
                f"centre of mass {d.subtree_com[1][1]:.2f} m up; arm–head clearance {arm_clearance(qpos) * 100:.1f} cm"
            )
    np.savez(args.out, fps=fps / SLOW, moves=np.array(MOVES), sites=np.array(SITES), **out)


if __name__ == "__main__":
    main()
