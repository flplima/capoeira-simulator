"""Record how the trained body actually loses its balance, for the recovery network to practise on.

    uv run python falls.py runs/expert_ginga/last.pt --kicks runs/expert_kicks/last.pt   # → data/falls.npz

The ginga and kick networks play at random tempos, with kicks asked for now
and then (at a size they don't always manage) and a shove from a partner every
few seconds. The moment the body is falling (GingaEnv.trouble) its whole state
is saved and a new game starts. Recovery episodes start from these states, so
the network learns to fall from the falls that really happen.
"""

import argparse

import numpy as np

from brain import Brain
from env import GingaEnv

PUSH_EVERY = 2.5  # s, on average
PUSH_FORCE = (120.0, 450.0)  # N
PUSH_TIME = (0.1, 0.3)  # s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--kicks")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--size", type=int, default=2, help="kick size level asked for")
    parser.add_argument("--out", default="data/falls.npz")
    args = parser.parse_args()
    brain = Brain(args.checkpoint, args.kicks)
    env = GingaEnv(max_seconds=1e9, seed=0, skill="all" if args.kicks else "ginga")
    env.choreo.kicks.level = args.size
    rng = np.random.default_rng(0)
    saved = {"qpos": [], "qvel": [], "phase": [], "bpm": []}
    obs, _ = env.reset()
    brain.reset()
    playing = 0.0
    while len(saved["qpos"]) < args.count:
        if rng.random() < env.control_dt / PUSH_EVERY:
            angle = rng.uniform(-np.pi, np.pi)
            env.push(rng.uniform(*PUSH_FORCE) * np.array([np.cos(angle), np.sin(angle)]), rng.uniform(*PUSH_TIME))
        obs, _, _, _, _ = env.step(brain(env, obs))
        playing += env.control_dt
        if playing > 0.5 and env.trouble():
            for key, value in (("qpos", env.d.qpos), ("qvel", env.d.qvel), ("phase", env.phase), ("bpm", env.bpm)):
                saved[key].append(np.copy(value))
            if len(saved["qpos"]) % 250 == 0:
                print(f"{len(saved['qpos'])} falls")
            obs, _ = env.reset()
            brain.reset()
            playing = 0.0
    np.savez(args.out, **{k: np.array(v) for k, v in saved.items()})
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
