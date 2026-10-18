# Capoeira — working with sub-agents

The main session coordinates and is the only one that edits code. Two
sub-agents in `.claude/agents/` do the rest:

- **trainer** — starts, monitors and (when stalled) stops `ml/train.py` runs,
  tests with `check.py`, answers data questions, exports to the app, runs vitest
  and the build. Reports numbers; never edits.
- **reviewer** — judges movement and references from renders (`check.py
  --record` + `viz.py`) and the app in the browser. Read-only, and briefed with
  checkpoints and files only — not the trainer's or the main session's
  conclusions — so its judgement stays independent.

A round:

1. Main session changes the environment, reward or references. In parallel the
   reviewer judges the references and recorded falls, and the trainer answers
   data questions.
2. Trainer runs a **15-minute trial**, tests it and records a run; the reviewer
   looks at the recording. Main session decides: full run or change the setup.
3. Trainer runs the full 4 hours; main session checks in hourly (scheduled
   check-in or Monitor, never a shell loop); a run that stalls for 90 minutes is stopped.
4. Trainer tests, reviewer looks at the body and the app; main session decides:
   retrain, fix a reference, or export.

One training at a time on this Mac. The manuals both agents follow are
`.claude/skills/capoeira-eval` and `.claude/skills/capoeira-render`. Nothing is
committed or pushed without asking the user.
