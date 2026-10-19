"""Check a trained policy the way the browser runs it: long, at every toque.

    uv run python check.py runs/expert_ginga/last.pt [--kicks runs/expert_kicks/last.pt] [--minutes 3]

Runs the policy from the reference pose, changing toque every 30 s (Angola,
São Bento Pequeno, São Bento Grande, and back) and asking for a kick every
5 s (martelo, bênção and the esquiva lateral in turn — those with a network, or
--moves — leg and entry from data/choices.json when it exists, else the next base), and reports falls, how far the body strays from
the triangle, how close planted feet stay to the corners, how high the hips
ride at each tempo, and how many kicks were finished standing. With
--recover, losing balance hands over to the recovery network, which should
land, get up and ginga on (the triangle moves to where the body stands up);
--push-every shoves the body now and then, the way the app's button does.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from brain import Brain
from env import GingaEnv
from env_consts import DIVE, DIVE_UNTIL, DODGES, ENTRY_CENTRE, FALL_TIME, FRAME_SKIP, KICK_MOVES, LOST, MIN_PACE, STUCK
from env_math import place_xz

TOQUES = [("Angola", 66.0), ("São Bento Pequeno", 90.0), ("São Bento Grande", 120.0)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--kicks", help="kick network")
    parser.add_argument("--esquiva", help="esquiva network")
    parser.add_argument("--moves", help="moves to ask for, in turn (default: those with a network)")
    parser.add_argument("--minutes", type=float, default=3.0)
    parser.add_argument("--toques", default="66,90,120", help="tempos to visit, 30 s each")
    parser.add_argument("--kick-every", type=float, default=5.0, help="s between kick requests (0: none)")
    parser.add_argument("--size", type=int, help="kick size level (kicks.py AMPLITUDES index; default the full kick)")
    parser.add_argument("--recover", help="recovery network: fall, get up and carry on instead of starting over")
    parser.add_argument("--push-every", type=float, default=0.0, help="s between shoves (0: none)")
    parser.add_argument("--push", type=float, default=300.0, help="shove force (N), for 0.2 s")
    parser.add_argument("--record", help="save the run to this .npz (t, qpos, target, label) for viz.py")
    args = parser.parse_args()
    brain = Brain(args.checkpoint, args.kicks, args.recover, args.esquiva)
    asking = args.moves.split(",") if args.moves else (KICK_MOVES if args.kicks else []) + (DODGES if args.esquiva else [])
    rng = np.random.default_rng(1)
    global TOQUES
    names = {66.0: "Angola", 90.0: "São Bento Pequeno", 120.0: "São Bento Grande"}
    TOQUES = [(names.get(float(b), f"{b} bpm"), float(b)) for b in args.toques.split(",")]

    env = GingaEnv(max_seconds=1e9, seed=0)
    obs, _ = env.reset(options={"phase": 0.0, "bpm": TOQUES[0][1], "moves": False})
    m, d = env.m, env.d
    centre = env.ref.triangle.mean(axis=0)
    steps = int(args.minutes * 60 / env.control_dt)
    per_toque = int(30 / env.control_dt)
    stats = {name: {"falls": 0, "up": 0, "failed": 0, "wander": 0.0, "corner": [], "height": []} for name, _ in TOQUES}
    kicks = {move: {"asked": 0, "done": 0, "fell": 0} for move in asking}
    choices = json.loads(Path("data/choices.json").read_text()) if Path("data/choices.json").exists() else None
    per_kick = int(args.kick_every / env.control_dt) if args.kick_every > 0 and asking else 0
    if args.size is not None:
        env.choreo.kicks.level = args.size
    asked = 0
    down = False
    per_push = int(args.push_every / env.control_dt) if args.push_every > 0 else 0
    record = {"t": [], "qpos": [], "target": [], "label": []}
    falls = []  # per fall: landing, which way the body was going over (as it faced), during a kick?, how it ended
    for step in range(steps):
        name, bpm = TOQUES[(step // per_toque) % len(TOQUES)]
        env.bpm = bpm
        c = env.choreo
        was_kicking = c.move
        if per_kick and step % per_kick == per_kick - 1 and not c.busy:
            move = asking[asked % len(asking)]
            asked += 1
            if choices and move in choices["moves"]:
                t = int(np.argmin([abs(x - bpm) for x in choices["tempos"]]))
                choice = choices["moves"][move][t][int(env.phase * choices["bins"]) % choices["bins"]]
                side, entry = choice["side"], choice["entry"]
            else:
                side = min("rl", key=lambda s: (ENTRY_CENTRE[s] - env.phase) % 1.0)
                entry = ENTRY_CENTRE[side]
            c.schedule(move, side, entry)
            kicks[move]["asked"] += 1
        if per_push and step % per_push == per_push - 1 and not c.busy:
            angle = rng.uniform(-np.pi, np.pi)
            env.push(args.push * np.array([np.cos(angle), np.sin(angle)]))
        was_recovering = c.recover is not None
        action = brain(env, obs)
        env.act(action)
        chest = m.body("chest").id
        d.xfrc_applied[chest, :3] = env.push_force if env.push_left > 0 else 0.0
        env.push_left = max(env.push_left - 1, 0)
        mujoco.mj_step(m, d, nstep=FRAME_SKIP)
        env.advance()
        obs = env._observe()
        if args.record:
            record["t"].append(step * env.control_dt)
            record["qpos"].append(d.qpos.copy())
            record["target"].append(c.target(0.0)[0].copy())
            record["label"].append(f"{name}: " + (f"recover {c.recover['move']} {c.recover['side']}" if c.recover is not None else c.move or "ginga"))
        s = stats[name]
        if brain.recover:
            if c.recover is None and not was_recovering and env.trouble():
                s["falls"] += 1
                if was_kicking:
                    kicks[was_kicking]["fell"] += 1
                lean = env.lean()
                env.choreo.begin_recover(d.qpos.copy(), lean, env.bent(), float(d.qvel[1]))
                yaw = c.recover["place"][2]
                side_way = np.cos(yaw) * lean[0] - np.sin(yaw) * lean[1]
                forward = np.sin(yaw) * lean[0] + np.cos(yaw) * lean[1]
                way = ("forward" if forward > 0 else "back") if abs(forward) >= abs(side_way) else ("left" if side_way > 0 else "right")
                falls.append({"landing": f"{c.recover['move']} {c.recover['side']}", "way": way, "kick": was_kicking, "end": None})
                stuck = 0.0
                obs = env._observe()
                continue
            if c.recover is not None:
                # While going down the head touching or diving (env.DIVE); once landed (the way up starts where the body
                # is), also drifting off it, or lying there while the way up waits at its slowest (env.STUCK).
                landed = c.recover["t"] >= FALL_TIME
                env.ref_data.qpos[:] = c.target(0.0)[0]
                mujoco.mj_kinematics(m, env.ref_data)
                head = m.body("head").id
                dive = c.recover["t"] < FALL_TIME + DIVE_UNTIL and d.xpos[head][1] < env.ref_data.xpos[head][1] - DIVE
                drift = np.linalg.norm(d.qpos[[0, 2]] - c.target(0.0)[0][[0, 2]]) if landed else 0.0
                stuck = stuck + env.control_dt if landed and env.pace() <= MIN_PACE else 0.0
                bad = env._contacts()[1]
                if bad or dive or drift > LOST or d.qpos[1] < 0.05 or stuck >= STUCK:
                    falls[-1]["end"] = "head" if bad else "dive" if dive else "drift" if drift > LOST else "flat" if d.qpos[1] < 0.05 else "stuck"
                    falls[-1]["at"] = round(c.recover["t"], 2)
                    s["failed"] += 1
                    env.reset(options={"phase": env.phase, "bpm": bpm, "moves": False})
                    brain.reset()
                    obs = env._observe()
                continue
            if was_recovering:
                s["up"] += 1
                falls[-1]["end"] = "up"
            fallen = False
        else:
            env.ref_data.qpos[:] = c.target(0.0)[0]
            mujoco.mj_kinematics(m, env.ref_data)
            fallen = env.low(env.ref_data) or env._contacts()[1]
        if not brain.recover and fallen and not down:
            s["falls"] += 1
            if was_kicking:
                kicks[was_kicking]["fell"] += 1
        elif was_kicking and c.move is None:
            kicks[was_kicking]["done"] += 1
        down = fallen
        if fallen:
            env.reset(options={"phase": env.phase, "bpm": bpm, "moves": False})
            brain.reset()
            obs = env._observe()
            continue
        triangle = place_xz(c.frame, env.ref.triangle)
        s["wander"] = max(s["wander"], float(np.linalg.norm(d.qpos[[0, 2]] - place_xz(c.frame, centre))))
        s["height"].append(d.qpos[1])
        for f, ids in enumerate(env.sole_sites):
            if min(d.site_xpos[i][1] for i in ids) < 0.01:
                s["corner"].append(np.min(np.linalg.norm(triangle - env._foot_points()[f], axis=1)))
    print(f"{args.minutes:.1f} min, toque changes every 30 s")
    for name, s in stats.items():
        print(
            f"  {name:18s} falls {s['falls']}" + (f" (got up {s['up']}, hit the head or lost it {s['failed']})" if brain.recover else "") + f", at most {s['wander']:.2f} m from the triangle centre, "
            f"planted foot {np.median(s['corner']) * 100:.1f} cm from its corner (median), hips {np.mean(s['height']):.2f} m"
        )
    for move, k in kicks.items():
        print(f"  {move:18s} asked {k['asked']}, finished standing {k['done']}, fell during it {k['fell']}")
    if falls:
        print("  falls, by landing (how they ended: up = stood and gingou again; head = head or neck hit the floor;")
        print("  dive = going down or just landed, the head 0.3 m below the reference's;")
        print("  drift = 0.6 m off the way up after landing; flat = hips on the floor; stuck = lying while the way up waits;")
        print("  at = median seconds after the fall began)")
        for landing in sorted({f["landing"] for f in falls}):
            group = [f for f in falls if f["landing"] == landing]
            ends = {e: sum(f["end"] == e for f in group) for e in ("up", "head", "dive", "drift", "flat", "stuck")}
            ats = [f["at"] for f in group if "at" in f]
            print(f"    {landing:12s} {len(group):3d}: " + ", ".join(f"{e} {n}" for e, n in ends.items()) + (f"; at {np.median(ats):.2f} s" if ats else ""))
        print("  falls, by which way the body went over → landing chosen")
        for way in ("forward", "back", "left", "right"):
            group = [f for f in falls if f["way"] == way]
            if group:
                chosen = {l: sum(f["landing"] == l for f in group) for l in sorted({f["landing"] for f in group})}
                print(f"    {way:8s} {len(group):3d}: " + ", ".join(f"{l} {n}" for l, n in chosen.items()))
        during = [f for f in falls if f["kick"]]
        print(f"  {len(during)} of {len(falls)} falls began during a kick")
    if args.record:
        np.savez_compressed(args.record, **{k: np.array(v) for k, v in record.items()})
        print("recorded", args.record)


if __name__ == "__main__":
    main()
