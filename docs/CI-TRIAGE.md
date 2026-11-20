# CI Triage Guide

Quick path for agents to diagnose a failure in the GitHub Actions run.

## First-pass workflow

1. Find the failing run in the Actions tab.
2. Open the failing job's log first (not the successful jobs).
3. Run the matching local command from the table below.
4. Re-run the smallest equivalent check until it passes, then push.

## Job-to-command map

| Workflow job | Local command | Primary failure surface |
| --- | --- | --- |
| `policy` | `npm run check:policy` | A file over the 5,000-token cap, a skipped test, a doc naming a missing npm script |
| `web` | `npx tsc -b`, then `npx vitest run`, then `npm run build` | Types, unit tests (including the Python/TypeScript parity fixture), the bundle |
| `ml-syntax` | `python3 -m compileall -q ml -x '\.venv'` | A Python syntax error in the training pipeline |
| `deploy demo` (`build`) | `npm run build -- --base=./`, then `test -s dist/index.html` | The relative-base build that goes to GitHub Pages; skipped while the repo is private |

`npm run check:fast` runs the policy, the type-check, the unit tests and the
Python compile in one go; it is the floor before a push.

## Before pushing

Run the local equivalent of every job your diff can touch:

| Changed | Also run |
| --- | --- |
| `src/**` | `npx vitest run` and `npm run build` |
| `ml/env*.py`, `ml/export_web.py` | `cd ml && uv run python export_web.py …` to regenerate `test/fixtures/ginga-live.json`, then `npx vitest run` (the TypeScript port must match Python) |
| `ml/train*.py`, `ml/modal_train.py` | the 15-minute trial from `capoeira-eval` |
| `.github/workflows/**` | nothing local; read the run |
| `vite.config.ts`, `index.html`, `sheet.html`, `public/**` | `npm run build -- --base=./`, serve `dist/` under a subpath and open it |

## Not in CI

Training, `check.py` and the render reviews need the reference data, the models
and a GPU or hours of CPU, none of which a runner has. They are judged by the
trainer and reviewer agents (see `AGENTS.md`).
