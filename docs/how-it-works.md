# how it works

The details behind the [README](../README.md): the moves still out of the
app, what the body can't do yet, the app's parts, the berimbau, the training
and the movement files.

## moves still out of the app

A move gets into the app when it ends standing nine times in ten over the
three toques and looks like the move. `scripts/measure-moves.ts` counts the
first part: it runs the same simulation as the browser, without drawing it.

Trained, but out until they pass that bar:

- **Martelo** (the one learned from video): 134 of 180. It falls on the
  landing about one time in four.
- **Queixada** 147, **troca de base** 142 and **esquiva baixa** 141, of 180.
- **Ponteira** 67 and **bênção** 29, of 90.
- **Meia-lua de compasso**: it stands 176 of 180 times, but it is not a
  compasso yet. No hands on the floor, no spin.
- **Rolê** 21 and **passagem de negativa** 2, of 90.

Bringing one back is one line in `src/offered.ts`, the list every button and
key comes from.

## what it cannot do yet

- **Get up after a fall.** The recovery network got up 0 of 29 times in
  tests ([numbers, in Portuguese](../ml/README-TRAINING.md#resultado-do-treino)).
  Today the app stands the body up again after 1.2 s.
- **Play with someone.** There is one body and no partner, so no jogo yet.
- **Change skills smoothly.** Ginga, kicks, esquivas and recovery are four
  separate networks. Every hand-off between them is where the body falls.

## keys

| Key | Does |
| --- | --- |
| <kbd>1</kbd> to <kbd>5</kbd> | the moves, in the order of the buttons |
| <kbd>space</kbd> | play or pause |
| <kbd>.</kbd> | one control step (0.03 s) |
| <kbd>-</kbd> / <kbd>+</kbd> | slower or faster, body and music together |
| <kbd>WASD</kbd> or the arrows | walk |
| <kbd>Q</kbd> / <kbd>R</kbd> | turn |
| <kbd>O</kbd> | circle the roda, facing the centre |
| <kbd>M</kbd> | berimbau on or off |

Drag to turn the camera. `?lang=pt` opens the page in Portuguese.

## the app

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
  where the simulation put them.
- **Parity**: `src/sim/` re-implements `ml/env*.py`. `test/live.test.ts`
  replays a fixture that `ml/export_web.py` writes, so the browser and the
  trainer can't drift apart.

## berimbau and tempo

The berimbau is synthesized in the browser, from no recordings. When the music
starts, the app renders a few takes of each sound and plays one at random on
every stroke, so no two bars sound quite the same. The steel wire is struck,
not plucked: a hard, bright hit, its overtones a little sharp because steel is
stiff. The gourd shapes it. Pressed to the belly, it makes the third overtone
ring out, nasal. Pulled away after the strike, it opens into a broader,
brighter sound: the "uá". The open note, the *dong*, sits at 104 Hz and rings
for over a second. The coin pressed on the wire gives the *ding*, 1.4
semitones higher. Resting lightly on the wire, it gives the *chiado*, a short,
dead buzz. The caxixi is in the stick hand, so its seeds rattle with every
stroke, on their own on a beat without one, and softly on the back-swing in
between. Tone.js keeps the beat and loops the chosen toque:

| Toque | Tempo | Beat (one 4-beat bar) | Ginga |
| --- | --- | --- | --- |
| Angola | 72 bpm | tch-tch DONG ding (rest) | low, wide, fluid, torso leaning more |
| São Bento Pequeno | 90 bpm | tch-tch DING dong (rest) | medium |
| São Bento Grande | 120 bpm | tch-tch DING dong dong | high, short and quick, upright |

The tempo goes from 70 to 120 bpm in steps of 2, and the app starts on São
Bento Pequeno. A full ginga (both sides) lasts one bar, and with the music on
its phase locks to the bar. A new toque glides the tempo at 6 bpm a second,
because a jump made the body fall. Speed (0.25× to 1.25×) slows music and
body together.

Sources on the toques:
[Capoeira toques](https://en.wikipedia.org/wiki/Capoeira_toques) and
[Capoeira Angola](https://en.wikipedia.org/wiki/Capoeira_Angola) (Wikipedia),
[Ginga (CapoeiraWiki)](https://capoeirawiki.org/wiki/Ginga),
[Angola vs Regional (Abadá Luxemburgo)](https://abadacapoeiraluxemburgo.com/capoeira-angola-capoeira-regional-diferencas/).

## training

The pipeline lives in [`ml/`](../ml/README.md) (Python, uv), and its docs are
in Portuguese: video → 3D pose → the mannequin's joint angles → reference
motions → PPO → export to `public/data/sim/ginga/`. Training runs on
[Modal](https://modal.com)'s GPUs by default, and the same code runs locally.
The moves are in [ml/README-MOVES.md](../ml/README-MOVES.md), the reward and
the results in [ml/README-TRAINING.md](../ml/README-TRAINING.md).

An H100 costs about US$ 0.07 a minute, and a 40-minute run is about 80 M
steps. Modal's US$ 30 monthly credit pays for all of it.

## movement files

Each move starts as a YAML file in `public/data/movements/`: the text the
panel shows (summary, steps, tips, sources, and an `en:` block for English)
and keyframes posed by hand. `scripts/sample-clips.ts` samples them for
training. Shared poses live in `public/data/poses.yaml`, and
[RESEARCH.md](../RESEARCH.md) (in Portuguese) has the 20 basic moves and their
sources. `public/data/movements/bencao.yaml` is a short one to start from.

A keyframe has a time in seconds and either a named `pose` to start from or
the previous frame, then overrides joints. Angles are degrees from rest:
standing straight, arms hanging, facing +Z, the body's left on +X. A missing
field means "same as the starting pose".

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
