# data

Everything in this folder, and the recording in
[`test/fixtures/`](../../test/fixtures), is under
[Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(CC BY-NC 4.0). The full text is in [LICENSE](LICENSE). The code is
[MIT](../../LICENSE).

You can copy, change and share these files for anything non-commercial.
Credit "capoeira simulator, by Felipe Lima" with a link to
https://github.com/flplima/capoeira-simulator.

## where each file comes from

| file | what it is | made from |
|---|---|---|
| `sim/ginga/reference.json` | the motions the networks learned to imitate | the ginga: a two-camera video of a capoeirista. The martelo: "Martelo -- Capoeira movimento + slow motion", by Capoeira Nantes. The other moves: the keyframes in `movements/` |
| `sim/ginga/policy.bin`, `sim/ginga/policy.json` | the trained networks | `reference.json` |
| `sim/ginga/humanoid.xml` | the mannequin, as a MuJoCo model | [`ml/model.py`](../../ml/model.py) |
| `movements/`, `poses.yaml`, `index.yaml` | the moves posed by hand, and the text the app shows about them | the sources in [RESEARCH.md](../../RESEARCH.md). Some keyframes were adjusted while watching the ginga video |
| `test/fixtures/ginga-live.json` | a run of the networks that the tests replay | the networks above |

## what this licence can't give you

It covers only my own rights. Two kinds of rights aren't mine to give:

- **The videos.** The martelo is fitted to Capoeira Nantes' video, and the
  ginga to a video of a capoeirista. The videos aren't mine, and neither is
  the movement of the people in them.
- **The tools.** The martelo went through
  [GVHMR](https://github.com/zju3dv/GVHMR), licensed for non-commercial
  research only, and the [SMPL-X](https://smpl-x.is.tue.mpg.de) body model,
  which has its own non-commercial licence.

So commercial use needs their permission too, not only mine.
