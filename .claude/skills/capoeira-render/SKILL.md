---
name: capoeira-render
description: Render the capoeira physics body, its reference motions and recorded falls (front and side cameras) to judge whether a ginga, kick, fall or get-up looks like real capoeira and whether a reference is physically sensible. Use when reviewing a trained policy, a reference motion, or a plan for new references.
---

# Rendering the body to judge the movement

Run from `capoeira/ml`. `$SCRATCH` below is your session scratchpad directory (or
`mktemp -d`); recordings and images go there, never into the repo. Read each PNG
with the Read tool — look at every image you make.

## 1. Record a run of the trained body

```sh
uv run python check.py runs/expert_ginga6/last.pt --kicks <kicks.pt> [--recover <recover.pt> --push-every 4 --push 350] \
  --minutes 1 --kick-every 5 --size 3 --toques 90 --record $SCRATCH/run.npz
```

The file holds, per control step: `t`, `qpos` (the body), `target` (the reference
it was following at that moment) and `label` (`"<toque>: ginga"`, `"…: martelo"`,
`"…: recover negativa r"`…). Find the moments worth looking at from the labels:

```sh
uv run python -c "
import numpy as np; z = np.load('$SCRATCH/run.npz'); prev = None
for t, l in zip(z['t'], z['label']):
    if l != prev: print(round(float(t), 2), l); prev = l"
```

## 2. Render chosen moments

```sh
uv run python viz.py $SCRATCH/run.npz $SCRATCH/fall1.png 35.1 35.4 35.7 36.0 36.4 36.8 37.2
```

Rows, top to bottom: **body front, body side, reference front, reference side**;
one column per time. The cameras follow the centre of mass; the floor is grey.
The body faces +z: in the front view it faces you.

Good picks:
- a whole ginga cycle (one bar: 4 beats ≈ 4 × 60/bpm s) in 8 columns;
- a kick or an esquiva from 0.3 s before its label starts to 0.5 s after it ends;
- a fall: the moment the label turns `recover`, +0.25 s, +0.5 s (landing), then
  every 0.5 s of the way up, and 1 s after the label is back to `ginga`.
  Render at least three falls, from different directions.

## 3. References and other inputs

```sh
uv run python viz.py data/ginga_fit.npz $SCRATCH/master.png          # the master in the video (front camera)
uv run python viz.py data/floor.npz $SCRATCH/neg.png negativa_r      # a landing and its way up (clip <move>_<side>)
uv run python viz.py data/falls.npz $SCRATCH/pool.png 0 50 100 150   # recorded moments of losing balance, by index
```

| File | What it is |
|---|---|
| `data/ginga_fit.npz`, `data/ginga_fit_side.npz` | the master in the video, fitted to the body (front / side camera) — ground truth for the ginga's style |
| `data/ginga_ref.npz` | the ginga reference per tempo (`qpos[tempo, frame]`; slice one tempo into a `t`/`qpos` file in the scratchpad to render it) |
| `data/kicks.npz` | martelo and bênção per leg and size |
| `data/esquiva.npz` | the esquiva lateral per side (the leg back in the base it starts from) and size, `esquiva_<side>_<level>_qpos` |
| `data/floor.npz` | each landing pose and its way up, `<move>_<side>_qpos`, first frame = the landing |
| `data/falls.npz` | 3000 recorded states of the trained body at the moment it lost balance (the starts recovery practises from) |
| `../public/data/movements/*.yaml` | the hand-made keyframes the references start from (ginga triangle, kicks, esquiva, landings) |

## What to look for

- **Ginga:** knees bent and hips low (not standing tall), back foot stepping
  behind on the triangle's back corners, the guard arm in front of the face
  switching with the legs, chest leaning slightly forward, weight shifting
  side to side in the rhythm — compare to `ginga_fit`.
- **Martelo:** a round kick with the instep, knee chambered first, hip turned
  over, the support heel pivoting; the leg comes back to the ginga, not down in front.
- **Bênção:** a front push kick with the sole, knee lifted then pushed straight
  forward, torso leaning back for balance.
- **Falls:** the landing must suit the way the body is going over — a person
  catches a forward fall on their hands, sits back into a backward one, and
  drops to the side into a sideways one. Check this on the recorded falls
  (`falls.npz` and the runs) against the landing `check.py` chose. The body
  lowers itself — hands reach the floor before the hips, no head or neck
  contact, no sliding of hands or feet that bear weight.
- **Landing references** (`floor.npz`): reachable from a fall within 0.5 s, the
  weight over the hands and feet that carry it, nothing through the floor, and
  a way up a torque-limited body can do (pushing off hands, a foot under the
  hips before rising), ending in the ginga's base.
- **Getting up:** pushes off the hands, feet to parallel, rises into the base
  and the ginga resumes at that tempo without a stumble.
- **Everywhere:** no limb through another, no foot sinking into or floating over
  the floor while it bears weight, no jitter between frames.

Report per move: **pass / fail**, the image path and column times that show the
problem, and whether it is the **body** (doesn't follow a good reference → more
training) or the **reference** (the reference rows already look wrong, or the
reference doesn't suit the situation → the reference or the choice of landing
must change).
