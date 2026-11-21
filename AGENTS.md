# Capoeira

A browser capoeira simulator: a MuJoCo (WebAssembly) body with 70 kg and torque-limited joints, driven by neural networks trained with reinforcement learning to imitate a reference video, over a berimbau-paced roda. `src/` is the app (Vite + React + Three.js + xstate); `ml/` is the training pipeline (Python, uv); `public/data/sim/ginga/` holds the networks the app loads.

**This project is under active development, not production.** Breaking changes are welcome. No backwards compatibility required — delete, rename, and restructure freely.

See [README.md](README.md) for the app and [ml/README.md](ml/README.md) for the pipeline (capture → retarget → reference → PPO → export). [RESEARCH.md](RESEARCH.md) holds the sources. [docs/CI-TRIAGE.md](docs/CI-TRIAGE.md) maps each failing CI job to its local command.

## Running it

| Want | Do | Notes |
| --- | --- | --- |
| The dev server | `npm run dev` | Port from `PORT` (3895); `.envrc` (untracked) provides the environment, see `.envrc.example` |
| Drive the app | `agent-browser --session <slug> open http://localhost:3895` | Launched muted so the berimbau isn't heard |
| Type-check, tests, build | `npx tsc -b`, `npx vitest run`, `npm run build` | |
| The pipeline | `cd ml && uv run python <script>.py` | Everything in `ml/` runs from that folder |

`ml/data`, `ml/models` and `ml/runs` are private (the reference video's frames and landmarks, the pose model, checkpoints). They are gitignored and never leave the project.

## Training: Modal first, local as the fallback

Training runs on Modal's GPU by default (`ml/modal_train.py` wraps `ml/train_mjx.py`); the same code runs locally for a smoke test or when offline. Every trainer writes the same `.pt` checkpoint format, which `check.py`, `choose.py` and `export_web.py` read. The commands and pass bars are in the `capoeira-eval` skill (`.agents/skills/capoeira-eval/`).

- Every new setup gets a **15-minute trial** before the 4-hour run.
- One local training at a time (it fills the CPU). Modal runs are independent, but start the full run only after the trial was judged.
- A run that stalls for 90 minutes is stopped.
- Only the trainer agent starts and stops runs, and only runs it was told about.

## Working with sub-agents

The main session coordinates and is the only one that edits code. Two sub-agents in `.claude/agents/` do the rest:

- **trainer** — starts, monitors and (when stalled) stops training runs, tests with `check.py`, answers data questions, exports to the app, runs vitest and the build. Reports numbers; never edits.
- **reviewer** — judges movement and references from renders (`check.py --record` + `viz.py`) and the app in the browser. Read-only, and briefed with checkpoints and files only — not the trainer's or the main session's conclusions — so its judgement stays independent.

A round:

1. Main session changes the environment, reward or references. In parallel the reviewer judges the references and recorded falls, and the trainer answers data questions.
2. Trainer runs a 15-minute trial, tests it and records a run; the reviewer looks at the recording. Main session decides: full run or change the setup.
3. Trainer runs the full 4 hours; main session checks in hourly (scheduled check-in or Monitor, never a shell loop).
4. Trainer tests, reviewer looks at the body and the app; main session decides: retrain, fix a reference, or export.

The manuals both agents follow are the `capoeira-eval` and `capoeira-render` skills.

## Coding Guidelines

1. **No legacy code** — remove dead code immediately. No commented-out code, no unused imports.
2. **No "not doing" comments** — comments explain what code does, not what it doesn't do.
3. **DRY** — extract repeated logic. If you write similar code twice, refactor it.
4. **Files stay under 5,000 tokens (20,000 bytes).** `npm run check:policy` fails the commit and CI otherwise. Split by responsibility and import from the concrete module — no facade files that re-export. Exempt: lockfiles, `public/data`, `test/fixtures`, binaries.
5. **The TypeScript port must match Python.** `src/sim/` re-implements `ml/env*.py`; `test/live.test.ts` replays a fixture that `export_web.py` writes. A change to one side regenerates the fixture and passes that test.
6. **Python modules the Modal image imports** are listed in `ml/modal_train.py`. Adding or splitting one means updating that list.

## Test Guidelines

- **Test what the user sees**, not what the state contains: a body that is up and gingando, not a flag.
- **Never commit skipped tests** (`it.skip`, `test.skip`, `describe.skip`, `xit`, `xtest`, `xdescribe`, `.todo`). `npm run check:policy` rejects them.
- **Fix any failure you meet**, even one that predates your change or is unrelated. A failing test is never someone else's problem.
- **A local pass is weak evidence for anything timing-sensitive.** When a test waits for state, treat "not ready yet" as keep waiting, never as done.

### Red-main policy

A broken `main` stops everyone. If CI turns red on `main`, land a fix, revert the offending commit, or quarantine the failing test within 24 hours. Never build features on a red `main`.

### Before pushing

`npm run check:fast` is the floor (policy, `tsc -b`, vitest, Python compile). Then run the local equivalent of every CI job your diff touches — [docs/CI-TRIAGE.md](docs/CI-TRIAGE.md) has the table. The pre-push hook runs `tsc -b` and vitest; `CAPOEIRA_SKIP_PREPUSH=1 git push` skips it for a genuine emergency.

## Documentation

- **No project-specific code in docs**: describe architecture and conventions in prose and tables; reference file paths instead of pasting code.
- **Diagrams are Mermaid (vertical, `flowchart TD`) or ASCII.** GitHub draws Mermaid in markdown; the READMEs use it.
- **One instruction file**: this `AGENTS.md` is the only one; `CLAUDE.md` just imports it.

## Git

Every change goes on its own branch, in its own worktree (`.claude/worktrees/<slug>`), and opens a pull request stacked on top of the open ones with `gh stack` (related or not), so the owner reviews and merges the whole stack at once. In a stack the base is the PR below: keep it current with `gh stack sync`, not `git merge main`. A PR stays a draft until CI is green. Never commit or push to `main`, and never merge.

**Stage files explicitly — never `git add -A`.** `ml/runs`, `ml/data` and `ml/models` sit next to tracked code; a sweeping add is how private data would get committed.

**Commit and push only your own branch; the PR is how the user reviews it.** Ask before anything else that leaves the machine: a training run on Modal, a deploy.

Use [gitmoji](https://gitmoji.dev/) for commit messages:

| Emoji | Description |
| --- | --- |
| ✨ | New feature |
| 🐛 | Bug fix |
| ♻️ | Refactor |
| 🎨 | Improve structure/format |
| ⚡ | Performance |
| 🔥 | Remove code/files |
| ✅ | Tests |
| 📝 | Documentation |
| 🔧 | Configuration |
| 🧠 | Training pipeline or new networks |

### Bug-fix commit tagging: missed-by layer

Tag every bug-fix commit (`🐛`) with the layer that should have caught the bug:

- Format: `🐛 <description> [missed-by: <layer>]`
- Layers: `unit-tests` (vitest), `parity` (the Python/TypeScript fixture), `check` (`check.py` on a trained policy), `render` (the reviewer's renders), `browser` (the app in a real browser), or `none`.
- `[missed-by: none]` means a gap in the testing setup, and marks where a new test or check belongs.
