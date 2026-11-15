# capoeira simulator

**English** · [Português](README.pt-BR.md)

Can a machine learn Capoeira?

> [!WARNING]
> _Early days, and entirely vibe-coded. The body still falls more than any capoeirista would. Expect breaking changes._

## what is it?

A body in your browser that learned capoeira by trial and error. It is a 70 kg
mannequin in [MuJoCo](https://mujoco.org), the physics engine, compiled to
WebAssembly. Its joints have torque limits, like muscles.

Neural networks drive it. They were trained with reinforcement learning (PPO,
in the DeepMimic style) to imitate a capoeirista on video. Nothing is
recorded: every 0.03 s a network looks at the body and picks targets for the
joints, and physics decides the rest. So the ginga never ends, never loops on
a seam, and follows the tempo of the berimbau.

Ask for a move at any time. The body picks the leg and the moment to leave
the ginga (both learned in training), does the move and goes back to the
ginga. A side panel explains each move: how to do it, tips and sources.

## what it can do today

The ginga holds at the three toques, with the tempo anywhere from 70 to
120 bpm. From the ginga the body does four moves and one turn, and it walks
around the roda:

| Key | Move | Ends standing |
| --- | --- | --- |
| <kbd>1</kbd> | Meia-lua de frente | 165 of 180 |
| <kbd>2</kbd> | Joelhada | 172 of 180 |
| <kbd>3</kbd> | Esquiva lateral | 176 of 180 |
| <kbd>4</kbd> | Cocorinha | 169 of 180 |
| <kbd>5</kbd> | Giro de base | 180 of 180 |

The counts are 60 tries at each toque. They come from
`scripts/measure-moves.ts`, which runs the same simulation as the browser
without drawing it. A move is in the app when it ends standing nine times in
ten and reads as the move.

Other keys: <kbd>space</kbd> play or pause, <kbd>.</kbd> one control step,
<kbd>-</kbd>/<kbd>+</kbd> speed, <kbd>WASD</kbd> or the arrows walk,
<kbd>Q</kbd>/<kbd>R</kbd> turn, <kbd>O</kbd> orbit the roda, <kbd>M</kbd>
berimbau. Drag to turn the camera. The page opens in English, and the flags
switch it to Portuguese.

Trained, but out of the app until they pass that bar:

- **Martelo** (the one learned from video): 134 of 180. It falls on the
  landing about one time in four.
- **Queixada** 147, **troca de base** 142 and **esquiva baixa** 141, of 180.
- **Ponteira** 67 and **bênção** 29, of 90.
- **Meia-lua de compasso**: it stands 176 of 180 times, but it is not a
  compasso yet. No hands on the floor, no spin.
- **Rolê** 21 and **passagem de negativa** 2, of 90.

## what it cannot do yet

- **Get up after a fall.** The recovery network got up 0 of 29 times in
  tests ([numbers, in Portuguese](ml/README-TRAINING.md#resultado-do-treino)).
  Today the app stands the body up again after 1.2 s.
- **Play with someone.** There is one body and no partner, so no jogo yet.
- **Change skills smoothly.** Ginga, kicks, esquivas and recovery are four
  separate networks. Every hand-off between them is where the body falls.

## running it

```bash
npm install
npm run dev        # → http://localhost:3895
npm test           # the simulation against Python, the movement files, interpolation, the rig
npm run build
```

An untracked `.envrc` sets the environment (see `.envrc.example`). Only `PORT`
is used, and `vite.config.ts` falls back to the same port without it. The
engine (about 10 MB) loads when the page opens.

## how it works

```
public/data/sim/ginga/          the model, the networks and the references (ml/export_web.py)
        │  fetch + MuJoCo WASM (src/sim/simulation.ts)
        ▼
LiveGinga                       every 0.03 s: observation → network (ginga, kicks,
                                esquiva or recovery) → joint targets → physics
        ▼
apply (simulation.ts)           MuJoCo angles → the rig's bone quaternions
        ▼
SkinnedMesh + FloorMarkers      17 bones, one mesh; plumb lines and circles on the floor, support polygon
```

- **State** (`src/machine.ts`): an XState machine holds what is discrete:
  playing or paused, speed, toque and tempo, the berimbau, the panel's tab.
  The simulation lives outside React (`src/playback.ts`), so nothing
  re-renders every frame.
- **Rig** (`src/rig/rig.ts`): a real bone hierarchy bound to one skinned mesh,
  with the same bones as the MuJoCo model. It looks like an artist's wooden
  mannequin in white abadá pants and a yellow corda.
- **Floor** (`src/physics/markers.ts`): each hand and foot has a plumb line to
  a circle, filled when it touches the floor. A diamond marks the centre of
  mass (segments from de Leva, 1996). The green line is the support polygon,
  orange when the centre of mass leaves it.
- **Triangle**: the `triangle` in `public/data/movements/ginga.yaml` gives the
  three corners the feet step on. Training uses them, and the app draws them
  where the simulation put them: under the feet after a move, and wherever
  the body was stood up after a fall.

### berimbau and tempo

The berimbau is synthesized in the browser with Tone.js (a steel string over a
gourd for the low *dong* and high *ding*, plus the caxixi), looping in the
chosen toque:

| Toque | Tempo | Beat (one 4-beat bar) | Ginga |
| --- | --- | --- | --- |
| Angola | 72 bpm | tch-tch DONG ding (rest) | low, wide, fluid, torso leaning more |
| São Bento Pequeno | 90 bpm | tch-tch DING dong (rest) | medium |
| São Bento Grande | 120 bpm | tch-tch DING dong dong | high, short and quick, upright |

The tempo goes from 70 to 120 bpm in steps of 2, and the app starts on São
Bento Pequeno. A full ginga (both sides)
lasts one bar, and with the music on its phase locks to the bar. Speed
(0.25× to 1.25×) slows music and body together. Sources on the toques:
[Capoeira toques](https://en.wikipedia.org/wiki/Capoeira_toques) and
[Capoeira Angola](https://en.wikipedia.org/wiki/Capoeira_Angola) (Wikipedia),
[Ginga (CapoeiraWiki)](https://capoeirawiki.org/wiki/Ginga),
[Angola vs Regional (Abadá Luxemburgo)](https://abadacapoeiraluxemburgo.com/capoeira-angola-capoeira-regional-diferencas/).

## training

The pipeline lives in [`ml/`](ml/README.md) (Python, uv), and its docs are in
Portuguese: video → 3D pose → the mannequin's joint angles → reference
motions → PPO → export to `public/data/sim/ginga/`. Training runs on
[Modal](https://modal.com)'s GPUs by default, and the same code runs locally.
The moves are in [ml/README-MOVES.md](ml/README-MOVES.md), the reward and the
results in [ml/README-TRAINING.md](ml/README-TRAINING.md).

## movement files

Each move starts as a YAML file in `public/data/movements/`: the text the
panel shows (summary, steps, tips, sources) and keyframes posed by hand.
`scripts/sample-clips.ts` samples them for training. Shared poses live in
`public/data/poses.yaml`. [RESEARCH.md](RESEARCH.md) (in Portuguese) has the
20 basic moves and their sources.

```yaml
id: bencao                    # = the file name
name: Bênção
category: ataque direto
side: right                   # the side the keyframes are written for
summary: …
steps: [ … ]
keyframes:
  - t: 0                      # seconds
    pose: base_r              # start from a named pose…
  - t: 0.35
    label: armando o joelho   # …or from the previous frame, overriding joints
    hip_r: { flex: 100, abd: 5 }
    knee_r: 105
```

Angles are degrees from rest: standing straight, arms hanging, facing +Z,
the body's left on +X. A missing field means "same as the starting pose".

| joint | fields | positive means |
|---|---|---|
| `root` | `x`, `z`, `lift` (m) | position on the floor; lift above the automatic support height |
| | `yaw`, `pitch`, `roll` | turn left · lean forward · lean to its own left (in that order) |
| `torso`, `head` | `flex`, `side`, `twist` | bend forward · bend to its own left · turn left |
| `shoulder_l/r`, `hip_l/r` | `flex`, `abd`, `rot` | limb forward/up · limb away from the midline · internal rotation (abduction first) |
| `elbow_l/r`, `knee_l/r` | number | bend |
| `ankle_l/r` | number | point the toes (negative: toes up) |

A foot is flat when `ankle = hip.flex − knee − root.pitch`. The ginga poses
tip the pelvis forward (`root.pitch` 10 and 15), so a keyframe that rewrites a
standing leg's `hip.flex` must add that pitch. A keyframe with `pose:` takes
the pose's shape but keeps `root.x`, `root.z` and `root.yaw` (a pose is a
shape, not a place).

The contact sheet shows every keyframe side by side, from several angles:
`/sheet.html?m=martelo` (add `&n=12` for evenly spaced samples, `&mirror=1`
for the left side, `&views=front,top`). A file that fails validation shows
its errors there and in `npm test`, which also checks joint limits and that a
move ends facing where it started without walking away.

## next steps

1. **Getting up after a fall.** The recovery network gets up 0 of 29 times,
   so today the app stands the body up again after 1.2 s.
2. **Bring back the moves taken out** because they fall too often or don't
   read as the move yet (listed above).
3. **More moves from video, on both legs**: queixada and armada, from the same
   slow-motion series as the martelo.
4. **One tracking policy for every clip** instead of one network per skill.
   The hand-offs between networks are where the body falls.
5. **A guide to adding a move from a video**, and videos from capoeiristas.
   Slow motion or 60 fps, the whole body in frame, one move from the ginga
   back to the ginga, from the front and from the side.

All of this runs on Modal's US$ 30 monthly credit. An H100 costs about
US$ 0.07 a minute, and a 40-minute run is about 80 M steps. Making it work on
that budget is part of the point.

## credits and licences

- **Martelo reference**: fitted to "Martelo -- Capoeira movimento + slow
  motion", by the [Capoeira Nantes](https://www.youtube.com/watch?v=ZCLVQIfaPHE)
  YouTube channel. No footage is in this repository, only the joint angles
  fitted to it (in `public/data/sim/ginga/reference.json`). The martelo
  is trained but out of the app for now.
- **Ginga reference**: fitted to a two-camera (front and side) video of a
  capoeirista with MediaPipe Pose (Apache-2.0).
- **[GVHMR](https://github.com/zju3dv/GVHMR)** (Shen et al., SIGGRAPH Asia
  2024) turns the martelo video into 3D pose. It is licensed for educational,
  research and non-profit use only; commercial use needs the authors'
  permission. It is not vendored here: `ml/hmr_modal.py` clones it when the
  Modal image builds.
- **[SMPL](https://smpl.is.tue.mpg.de) and [SMPL-X](https://smpl-x.is.tue.mpg.de)**:
  the body models GVHMR needs. They need a sign-up, have their own
  non-commercial research licences, and are not in this repository.
- **[MuJoCo](https://github.com/google-deepmind/mujoco)** (Apache-2.0, Google
  DeepMind), including its WebAssembly build `@mujoco/mujoco`.
- **Papers**: DeepMimic (Peng et al. 2018) for the reward and the imitation
  setup, KungfuBot (Xie et al. 2025) for adaptive tracking tolerances,
  Abdolhosseini et al. 2019 for the symmetry loss, de Leva 1996 for segment
  masses.
- **Keyframe moves** were posed by hand from the sources in
  [RESEARCH.md](RESEARCH.md).

So: the code is MIT. The motion references and the networks trained on them
come from that research-only chain and from third-party video, so this
project is non-commercial. Reusing those files commercially means clearing
GVHMR, SMPL-X and the video's owners yourself.

## licence

[MIT](LICENSE), for the code.
