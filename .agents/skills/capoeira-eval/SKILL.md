---
name: capoeira-eval
description: Train, test and export the capoeira physics policies (ginga, kicks, fall-and-get-up) — the modal_train.py / train_mjx.py / train.py / check.py / choose.py / export_web.py commands, how to read their output, the trial-run gate, when to stop a run, and the pass bars. Use when starting, watching or judging a training run or shipping networks to the app.
---

# Training and testing the capoeira policies

Everything runs from `ml/` with `uv run python …`. `ml/README.md` explains
the pipeline; this file is the operating manual.

## The networks

| Skill | Run folder | What it drives |
|---|---|---|
| `ginga` | `runs/expert_ginga<N>` | the ginga at every tempo (60–130 bpm) |
| `kicks` | `runs/expert_kicks<N>` | martelo and bênção, from the entry until back in the ginga |
| `recover` | `runs/expert_recover<N>` | falling onto a landing pose and getting up into the ginga |
| `esquiva` | `runs/expert_esquiva<N>` | the esquiva lateral, from the base until back in the ginga |

Current best: ginga `runs/expert_ginga6/last.pt`, kicks `runs/expert_kicks9/best.pt`,
recover — none yet (`runs/expert_recover` got up 0 times; it is the baseline to beat), esquiva `runs/expert_esquiva6/last.pt`.
New runs get the next free number; never overwrite or delete a run folder.

## Where training runs

**Modal first.** `ml/modal_train.py` runs `train_mjx.py` on a Modal GPU (an L4 by
default). Checkpoints and `log.jsonl` land in the Modal volume `capoeira-mjx-runs`,
committed every 3 minutes, so a run that is cut off keeps what it had. Runs
are independent: two can train at once, but start the full run only after the
trial was judged.

**Local fallback.** The same code runs on this Mac when Modal is unavailable or for
a smoke test: `train_mjx.py` picks the device JAX finds (64 environments on a CPU,
2048 on a GPU), and the older CPU PPO `train.py` still works. The Mac runs
**one local training at a time** (it fills the CPU).

Both write the same `.pt` files, so `check.py`, `choose.py` and `export_web.py`
do not care which one trained a policy.

Is something already training?

```sh
uv run modal app list                       # Modal: running apps ("capoeira-mjx")
ps -Ao pid,etime,command | grep "[t]rain"   # this Mac: train.py / train_mjx.py
```

The `[t]` matters: a plain `pgrep -f "train"` inside a shell loop matches the
loop's own command line and waits forever (this cost 3½ hours once).

## Starting a run — trial first, always

Every new setup (changed reward, reference, environment or skill) gets a
**15-minute trial** before the 4-hour run. From `ml/`:

```sh
# Modal (default): --detach keeps the run going after this shell returns
uv run modal run --detach modal_train.py --no-wait --skill <ginga|kicks|esquiva|recover> \
  --out runs/<out> --init runs/<checkpoint>.pt --minutes 15 --steps 100e6 [--lr 5e-5] [--log-std -2.3] \
  [--gpu A100-40GB --envs 4096]   # another GPU than the L4, with the environments its memory takes
  [--mirror]                      # kicks and esquiva: also learn from the left-right mirror of every batch, both legs at once

# Local fallback (CPU/GPU on this Mac)
mkdir -p runs/<out>
nohup uv run python train_mjx.py --skill <skill> --out runs/<out> \
  --init runs/<checkpoint>.pt --minutes 15 --steps 100e6 [--lr 5e-5] [--log-std -2.3] \
  > runs/<out>/stdout.txt 2> runs/<out>/stderr.txt &
```

(`train.py` takes the same flags plus `--envs 10`, and is the choice when JAX is
not available.) The Modal run name is the last part of `--out`; use the next free
number and never reuse one that exists in the volume or in `runs/`.

When the trial ends: bring it home (below), read its log, run the matching
`check.py` for one minute, and record a short run for the reviewer (`--record`).
Only a trial whose numbers move the right way earns the full run: same command,
a new `--out`, `--init runs/<trial>/last.pt`, `--minutes 240`.

- Start from the best checkpoint of the same skill (or the ginga for a new skill, as the esquiva).
- **Polishing a policy that already works: `--lr 5e-5`.** The default shakes
  a trained policy and it gets worse before it gets better.
- Don't wait in a shell loop. Start the run, report where it runs (Modal app id
  or local PID), start time and expected end, and return; the coordinator
  schedules the check-ins.

## Watching a run

Modal: `uv run modal run modal_train.py::tail --run <out>` prints the last lines
of `log.jsonl` from the volume; `uv run modal app logs <app-id>` streams the
stdout. When it has ended, or to test a checkpoint mid-run,
`uv run modal run modal_train.py::fetch --run <out>` copies `best.pt`, `last.pt`
and `log.jsonl` into `runs/<out>/`. Local: read `runs/<out>/log.jsonl` directly.
`best.pt` is the iteration with the highest share of moves finished standing (kicks
and esquiva runs; the measure `eval` and `check.py` use) or the highest reward ×
episode length (other skills); it starts over at every curriculum size.

`runs/<out>/log.jsonl`, one line per iteration:

- `reward` 0–1, `episode_s` (mean episode length in seconds — longer is better),
  `std` (exploration noise).
- `tol`: the tracking tolerances per term, for the ginga and for each move on each
  leg (`martelo_l`, `martelo_r`, …), as RMS errors (rad for `pose` and `turn`, m for
  the rest). The trainer sets each to that term's running mean error and only ever
  tightens it, so **`tol` falling is progress**, and a leg or move whose row stays
  far above its twin's is the one not learning (compare `martelo_l` with
  `martelo_r`). `pose`/`vel`/`ee`/`root`/`feet`/`strike`/`pivot` are the scores
  against those tolerances and sit around 0.4–0.6 by construction; compare `tol`,
  not them, between runs.
- Shares of time, not scores: `kicking` (inside a kick), `down` (falling or
  getting up), `up` (back in the ginga after getting up). `size` = curriculum level: kick or esquiva size (max ≈ 3), or with `--skill recover` the
  fall speed recorded falls start with (0 = at rest, 1 = 35 %, 2 = 70 %, 3 = full).
- **Why episodes end**, over the last 200: `end_time` (ran out the clock — good),
  `end_contact` (a body part that may not touch the floor did; while recovering,
  only the head and neck), `end_low` (hips or head too low outside a recovery: below 0.4 / 0.8 m, or 0.3 m below the reference's where it goes lower, as in the esquiva),
  `end_drift` (too far from the reference; while recovering, only after landing: 0.6 m off the way up), `end_stuck`
  (recovering: after landing, lying there 1 s while the way up waits at its slowest), `end_dive`
  (recovering: going down or within 0.5 s of landing, the head 0.3 m below the reference's).
- **Recover only**, over the last 200 episodes that fell: `fell_<landing>` (share
  per landing pose) and `got_up` (share that reached the ginga again). Training
  also starts 50 % of episodes on the floor or partway up, so `got_up` in training reads higher than in `check.py`.

- **`eval`** (kicks and esquiva runs, the last line of the log): the trainer's own
  test at the end of the run, on `last.pt` and `best.pt` — every environment asks for one
  move from the ginga, played without exploration noise, and `{move: {leg: [finished
  standing, asked]}}` counts them the way `check.py` does. With 8192 environments
  that is ~580 kicks per move and leg, against `check.py`'s ~26: trust `eval` to rank
  checkpoints and runs; use `check.py` for the ginga hand-off and the eye.

Red flags: `episode_s` collapsing, `pose` < 0.1 for long, `std` growing, `nan`,
or one `end_*` reason dominating (> 0.6) — that names what to fix. `stderr.txt`
holds crashes; the torch "requires_grad … to a scalar" warning is harmless.

### Stopping a run early

Compare the mean of the last 30 minutes of `reward`, `episode_s` and (recover)
`got_up` with the 30 minutes before. If none improved for **90 minutes**, the run
has stalled: stop it (Modal: `uv run modal app stop <app-id>`; local: `kill <pid>` of the
training process, never by name pattern), keep its `last.pt` (`fetch` it first on
Modal), and report it as stalled with the `end_*`
shares. More time won't fix a stalled run; the setup needs a change.

## Testing

`check.py` plays the policy the way the browser does: long, switching tempo every 30 s.

```sh
# ginga alone, every tempo
uv run python check.py runs/expert_ginga6/last.pt --toques 60,66,90,120,130 --minutes 5 --kick-every 0
# kicks, one size level at a time (1 = 60 %, 2 = 80 %, 3 = full)
uv run python check.py runs/expert_ginga6/last.pt --kicks <kicks.pt> --toques 66,90,120 --minutes 3 --kick-every 5 --size 3
# the esquiva alone (--moves picks what is asked for; default: every move with a network)
uv run python check.py runs/expert_ginga6/last.pt --esquiva <esquiva.pt> --toques 66,90,120 --minutes 3 --kick-every 5 --size 3
# falls and getting up: shoves from random directions, plus kicks
uv run python check.py runs/expert_ginga6/last.pt --kicks <kicks.pt> --recover <recover.pt> --push-every 4 --push 350 --kick-every 5 --size 3 --minutes 3
```

With `--recover` it also prints every fall grouped **by landing** with how it
ended — `up` (stood up and gingou again), `head` (head or neck hit the floor),
`dive` (going down or within 0.5 s of landing, the head 0.3 m below the reference's, as in training), `drift` (after landing, 0.6 m off the way up — which starts where the body
landed), `flat` (hips on the floor), `stuck` (lying 1 s while the way up
waits) — and the median second it failed at (the
landing is reached at 0.5 s; earlier means it failed while going down). Then
**by which way the body went over** (forward, back, left, right, as the body
faced) and which landing that picked — forward → `joelhos` (hands and knees),
back → `quatro`, sideways → `negativa` (folding the leg that carries the weight), and a trunk bent over 0.9 rad forward → `joelhos` whichever way — and how many falls began during a kick. Report these tables as they are.

Add `--record <file>.npz` to keep the run for the reviewer (see `capoeira-render`).

### Pass bars

| What | Trial (15 min) worth a full run | Ship |
|---|---|---|
| ginga falls | — | 0 per tempo in 5 min |
| planted foot from its corner | — | median ≤ 5 cm |
| furthest from the triangle centre | — | ≤ 0.35 m |
| kicks finished standing (still up 1 s into the ginga after it) | — | ≥ 14/18 per kick |
| esquivas finished standing (`--esquiva`) | > 0 at size 0, `size` rising in the log | ≥ 17/18 at full size |
| recover: falls that end `up` (check.py) | > 0, and `got_up` rising in the log | ≥ 80 % |
| recover: `head` endings | falling vs. the previous run | ≤ 10 % of falls |
| recover: median failure time | later than the previous run | — |
| kicks with `--recover` on | not worse than without it | same bars as kicks |

Always report numbers per tempo, per kick and per landing, next to the previous best run.

## After training

```sh
uv run python choose.py runs/expert_ginga6/last.pt --kicks <kicks.pt> [--esquiva <esquiva.pt>] --size <s>   # leg and entry per tempo → data/choices.json (moves left out keep theirs)
uv run python export_web.py runs/expert_ginga6/last.pt ../public/data/sim/ginga \
  --kicks <kicks.pt> --recover <recover.pt> --esquiva <esquiva.pt> --size <s>          # networks + fixture for the app
cd .. && npx vitest run && npm run build                                               # the browser port must match Python
```

`export_web.py` rewrites `test/fixtures/ginga-live.json`; if `test/live.test.ts`
then fails, the TypeScript port (`src/sim/simulation.ts`) and `env.py` disagree —
report it, don't paper over it.

## Never

- Commit or push (the coordinator asks the user first).
- Copy anything from `ml/data`, `ml/models` or `ml/runs` outside the project or into
  git — the reference video, its frames and landmarks are private.
- Run two local trainings at once, or long checks while a local training runs (a 1–3 minute check is fine). Modal runs are exempt.
- Start a Modal run without saying so: it spends the credit.
