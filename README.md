# capoeira simulator

**English** · [Português](README.pt-BR.md)

Can a machine learn capoeira?

https://github.com/user-attachments/assets/dfddc6f9-33ba-4223-b176-dea17ac0d75b

Sound on! The berimbau is synthesized in your browser too.

Try it in your browser: **[capoeira.felipelima.xyz](https://capoeira.felipelima.xyz)**

> [!WARNING]
> _Early days, and entirely vibe-coded. The body still falls more than any capoeirista would._

## what is it?

A 70 kg mannequin in [MuJoCo](https://mujoco.org), the physics engine,
running in your browser. Its joints have torque limits, like muscles. Nothing
in the video is an animation.

Neural networks move it. They learned by trial and error to imitate a
capoeirista on video (reinforcement learning, in the
[DeepMimic](https://github.com/xbpeng/DeepMimic) style). Every 0.03 s a
network looks at the body and picks targets for the joints, and physics does
the rest. So the ginga follows the berimbau and never loops.

Ask for a move at any time. The body picks the leg and the moment to leave
the ginga, does the move and goes back to the ginga.

## what it can do today

| Key | Move | Ends standing |
| --- | --- | --- |
| <kbd>1</kbd> | Meia-lua de frente | 165 of 180 |
| <kbd>2</kbd> | Joelhada | 172 of 180 |
| <kbd>3</kbd> | Esquiva lateral | 176 of 180 |
| <kbd>4</kbd> | Cocorinha | 169 of 180 |
| <kbd>5</kbd> | Giro de base | 180 of 180 |

That's 60 tries at each of the three toques, from 72 to 120 bpm. A move gets
into the app when it ends standing nine times in ten and looks like the move.
Nine more are trained and wait outside until they pass that bar: the list
and their numbers are in [docs/how-it-works.md](docs/how-it-works.md#moves-still-out-of-the-app).

## how it compares

| | keyframe animation | motion capture | capoeira simulator |
|---|---|---|---|
| where the motion comes from | an animator poses it | a recording of a real person | a network that learned from a video |
| weight and balance | drawn | recorded | simulated: 70 kg, torque-limited joints |
| follows the berimbau's tempo | if someone re-times it | if someone re-times it | ✓, from 70 to 120 bpm |
| loops | ✓ | ✓ | ✗ |
| can fall | ✗ | ✗ | ✓ (more than it should) |

## running it

```bash
npm install
npm run dev        # → http://localhost:3895
npm test           # the simulation against Python, the movement files, the rig
```

The page opens in English, and the flags switch it to Portuguese. The keys
are under the buttons: <kbd>1</kbd> to <kbd>5</kbd> for the moves,
<kbd>WASD</kbd> to walk, <kbd>O</kbd> to circle the roda.

## how it works

```
video ──► 3D pose ──► the mannequin's joint angles ──► reference motions
                                                              │
                                    PPO on Modal's GPUs ◄─────┘
                                              │
public/data/sim/ginga/ ◄──── export ──────────┘
        │
        ▼
the browser: every 0.03 s, body state ──► network ──► joint targets ──► MuJoCo
```

The training pipeline lives in [`ml/`](ml/README.md) (Python, docs in
Portuguese). The app is Vite, React and Three.js, and `src/sim/` re-runs the
same environment as `ml/env.py`, checked against it in `npm test`. The rig,
the berimbau, the movement files and the training are in
[docs/how-it-works.md](docs/how-it-works.md).

## next steps

1. **Getting up after a fall.** The recovery network gets up 0 of 29 times,
   so today the app stands the body up again after 1.2 s.
2. **Bring back the moves left out**, starting with the martelo, the one
   learned from video. It falls on the landing about one time in four.
3. **More moves from video, on both legs**: queixada and armada, from the
   same slow-motion series as the martelo.
4. **One network for every clip** instead of one per skill. The hand-offs
   between networks are where the body falls.
5. **A guide to adding a move from your own video.** Slow motion or 60 fps,
   the whole body in frame, from the ginga back to the ginga.

All of this runs on Modal's US$ 30 monthly credit. Making it work on that
budget is part of the fun.

## credits and licences

- **Martelo reference**: fitted to "Martelo -- Capoeira movimento + slow
  motion", by the [Capoeira Nantes](https://www.youtube.com/watch?v=ZCLVQIfaPHE)
  YouTube channel. No footage is in this repository, only the joint angles
  fitted to it (in `public/data/sim/ginga/reference.json`).
- **Ginga reference**: fitted to a two-camera (front and side) video of a
  capoeirista with MediaPipe Pose (Apache-2.0).
- **[GVHMR](https://github.com/zju3dv/GVHMR)** (Shen et al., SIGGRAPH Asia
  2024) turns the martelo video into 3D pose. It is licensed for educational,
  research and non-profit use only, and commercial use needs the authors'
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

## contributing

Any contribution is welcome, and so are AI agents! [AGENTS.md](AGENTS.md) has
the rules, and [docs/CI-TRIAGE.md](docs/CI-TRIAGE.md) maps each CI job to the
command that runs it locally.

## licence

[MIT](LICENSE), for the code.
