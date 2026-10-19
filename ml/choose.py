"""Learn when and with which leg to answer a kick request, from simulated trials.

    uv run python choose.py runs/expert_ginga/last.pt --kicks runs/expert_kicks/last.pt   # → data/choices.json

A kick can start from the base with either leg back, a little before or
after the exact base (ENTRY_WINDOW). Which start is best depends on where in
the ginga the request arrives and on the tempo: waiting costs time, an early
or late entry can cost balance. So for every move, tempo and moment of the
request, the trained policy tries every leg and entry in simulation, from a
settled ginga, and the start that scores best — stays up, tracks the kick
well, answers soon — is kept. The browser looks the answer up.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from brain import Brain
from env import GingaEnv
from env_consts import DODGES, ENTRY_CENTRE, ENTRY_WINDOW, FRAME_SKIP, KICK_MOVES

TEMPOS = [60.0, 78.0, 96.0, 114.0, 130.0]
BINS = 16  # request moments per ginga cycle
OFFSETS = np.linspace(-ENTRY_WINDOW, ENTRY_WINDOW, 5)
WAIT_COST = 0.15  # score lost per second of waiting
AFTER = 1.0  # s of ginga after the kick that must stay upright too


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--kicks", help="kick network")
    parser.add_argument("--esquiva", help="esquiva network")
    parser.add_argument("--out", default="data/choices.json")
    parser.add_argument("--size", type=int, help="kick size level (kicks.py AMPLITUDES index; default the full kick)")
    args = parser.parse_args()
    brain = Brain(args.checkpoint, args.kicks, esquiva=args.esquiva)
    moves = (KICK_MOVES if args.kicks else []) + (DODGES if args.esquiva else [])
    env = GingaEnv(max_seconds=1e9, seed=0)
    m, d = env.m, env.d
    if args.size is not None:
        env.choreo.kicks.level = args.size

    def policy_step() -> tuple[float, bool]:
        action = brain(env, env._observe())
        env.act(action)
        mujoco.mj_step(m, d, nstep=FRAME_SKIP)
        env.choreo.advance(env.control_dt)
        reward, _ = env._reward()
        fallen = env.low(env.ref_data) or env._contacts()[1]  # (_reward left the reference's kinematics in ref_data)
        return reward, fallen

    # Moves without a network this time keep the choices they had.
    old = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    table: dict = {"tempos": TEMPOS, "bins": BINS, "moves": old.get("moves", {}) if old.get("tempos") == TEMPOS and old.get("bins") == BINS else {}}
    for move in moves:
        rows = []
        for bpm in TEMPOS:
            row = []
            for b in range(BINS):
                request = b / BINS
                # A settled ginga arriving at the request moment: start a cycle earlier.
                env.reset(options={"phase": request, "bpm": bpm, "moves": False})
                brain.reset()
                for _ in range(int(round(env.ref.period(bpm) / env.control_dt))):
                    policy_step()
                env.phase = request
                saved = (d.qpos.copy(), d.qvel.copy(), d.act.copy(), d.time)
                since = brain.since_kick
                best = None
                for side in "lr":
                    for offset in OFFSETS:
                        d.qpos[:], d.qvel[:], d.act[:] = saved[0], saved[1], saved[2]
                        d.time = saved[3]
                        mujoco.mj_forward(m, d)
                        env.choreo.move, env.choreo.phase = None, request
                        env.choreo.frame[:] = 0.0
                        env.choreo.anchor[:] = 0.0
                        brain.since_kick = since
                        brain.mover = None
                        entry = (ENTRY_CENTRE[side] + offset) % 1.0
                        env.choreo.schedule(move, side, entry)
                        wait = ((entry - request) % 1.0) * env.ref.period(bpm)
                        rewards, fell, steps = [], False, 0
                        limit = int((wait + env.choreo.kicks.duration(move) + 0.5 + AFTER) / env.control_dt)
                        after = None
                        while steps < limit:
                            reward, fallen = policy_step()
                            steps += 1
                            if env.choreo.move is not None:
                                rewards.append(reward)
                            elif rewards and after is None:
                                after = steps
                            if fallen:
                                fell = True
                                break
                            if after is not None and steps - after > AFTER / env.control_dt:
                                break
                        quality = float(np.mean(rewards)) if rewards and not fell else 0.0
                        score = quality - WAIT_COST * wait
                        if best is None or score > best["score"]:
                            best = {"side": side, "entry": round(float(entry), 4), "score": round(score, 4), "quality": round(quality, 4), "wait": round(wait, 3)}
                row.append(best)
            rows.append(row)
            sides = "".join(c["side"] for c in row)
            print(f"{move:8s} {bpm:5.0f} bpm  legs by request moment: {sides}  mean quality {np.mean([c['quality'] for c in row]):.2f}")
        table["moves"][move] = rows
    json.dump(table, open(args.out, "w"), indent=1)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
