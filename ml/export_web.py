"""Export what the browser needs to run the ginga live: model, policy, reference.

    uv run python export_web.py runs/expert_ginga/last.pt ../public/data/sim/ginga --kicks runs/expert_kicks/last.pt \
        --recover runs/expert_recover/last.pt --esquiva runs/expert_esquiva/last.pt

The browser runs the same MuJoCo model (same engine version, compiled to
WebAssembly) and evaluates the policy network itself every control step, so
the body decides its next move in real time instead of replaying a recording.
Written files:

- humanoid.xml     the MuJoCo model
- policy.json      the networks (ginga, kicks, recover, esquiva; see brain.py): sizes and
                   observation normalisation; control constants
- policy.bin       actor weights (float32: per network, each layer's weight,
                   row-major, then bias)
- reference.json   the reference cycle per tempo (qpos and foot targets), the
                   kicks and the esquiva, and the ways up from each landing (floor.py)

Also writes test/fixtures/ginga-live.json: a few simulated states with the
observation and action the Python side computes for them, so a web test can
check its port of the observation and of the network match exactly.
"""

import argparse
import copy
import json
import shutil

import mujoco
from pathlib import Path

import numpy as np
import torch

from brain import HANDBACK, MOVE_HANDBACK, Brain
from env import GingaEnv
from env_consts import (
    ACTION_SCALE,
    ALLOWED_CONTACT,
    BENT,
    BLEND_IN,
    BLEND_OUT,
    EFFECTORS,
    FALL_TIME,
    FRAME_SKIP,
    GROUND_SITES,
    LOST,
    LOW_BELOW,
    LOW_HEAD,
    LOW_HIPS,
    MAX_BPM,
    MIN_BPM,
    MIN_PACE,
    MOVES,
    NEVER_CONTACT,
    STRONG_ARMS,
    STUCK,
    TROUBLE_DROP,
    TROUBLE_TILT,
    WAIT_FROM,
    WAIT_SPAN,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    parser.add_argument("--fixture", default="../test/fixtures/ginga-live.json")
    parser.add_argument("--choices", default="data/choices.json", help="from choose.py")
    parser.add_argument("--kicks", help="kick network checkpoint, then move=checkpoint for a kick with its own (brain.kick_networks)")
    parser.add_argument("--recover", help="recovery network checkpoint")
    parser.add_argument("--esquiva", help="esquiva network checkpoint")
    parser.add_argument("--size", type=int, help="kick size level to ship (kicks.py AMPLITUDES index; default the full kick)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    brain = Brain(args.checkpoint, args.kicks, args.recover, args.esquiva)
    blobs, networks, offset = [], {}, 0
    for name, net in brain.networks.items():
        layers = []
        for layer in (l for l in net.agent.actor if isinstance(l, torch.nn.Linear)):
            w = layer.weight.detach().numpy().astype(np.float32)
            b = layer.bias.detach().numpy().astype(np.float32)
            layers.append({"in": w.shape[1], "out": w.shape[0], "offset": offset})
            blobs += [w.ravel(), b]
            offset += w.size + b.size
        networks[name] = {
            "layers": layers,
            "mean": [float(x) for x in net.norm.mean],
            "std": [float(x) for x in np.sqrt(net.norm.var + 1e-8)],
            "training": net.stats,
        }
    (out / "policy.bin").write_bytes(np.concatenate(blobs).tobytes())

    env = GingaEnv(max_seconds=1e9, seed=0)
    m = env.m
    if args.size is not None:
        env.choreo.kicks.level = args.size
    policy = {
        "obs": int(env.observation_space.shape[0]),
        "act": brain.ginga.act,
        "networks": networks,
        "moveNetworks": brain.move_network,
        "handback": HANDBACK,
        "handbackKicks": MOVE_HANDBACK,
        "handbackEsquiva": MOVE_HANDBACK,
        "activation": "elu",
        "clip": 10,
        "actionScale": ACTION_SCALE,
        "frameSkip": FRAME_SKIP,
        "timestep": float(m.opt.timestep),
        "effectors": EFFECTORS,
        "allowedContact": sorted(ALLOWED_CONTACT),
        "neverContact": sorted(NEVER_CONTACT),
        "low": {"hips": LOW_HIPS, "head": LOW_HEAD, "below": LOW_BELOW},
        "minBpm": MIN_BPM,
        "maxBpm": MAX_BPM,
    }
    (out / "policy.json").write_text(json.dumps(policy, separators=(",", ":")))
    shutil.copy("humanoid.xml", out / "humanoid.xml")

    ref = env.ref
    reference = {
        "tempos": ref.tempos.tolist(),
        "samples": ref.n,
        "nq": int(m.nq),
        "qpos": [np.round(q, 5).ravel().tolist() for q in ref.qpos],
        "feet": [np.round(f, 5).ravel().tolist() for f in ref.feet],
        "triangle": ref.triangle.tolist(),
        "kicks": {
            "blendIn": BLEND_IN,
            "blendOut": BLEND_OUT,
            "clips": {
                f"{move}_{side}": {
                    "duration": clip["duration"],
                    "fps": clip["fps"],
                    "qpos": np.round(clip["qpos"], 5).ravel().tolist(),
                    "feet": np.round(clip["feet"], 5).ravel().tolist(),
                    "mask": clip["mask"].ravel().tolist(),
                }
                for (move, side, level), clip in env.choreo.kicks.clips.items()
                if level == env.choreo.kicks.level
            },
        },
        "floor": {
            "fps": env.choreo.floor.fps,
            "fallTime": FALL_TIME,
            "troubleTilt": TROUBLE_TILT,
            "lost": LOST,
            "stuck": STUCK,
            "bent": BENT,
            "ground": {"sites": GROUND_SITES, "clear": env.ground_clear.tolist()},
            "waitFrom": WAIT_FROM,
            "waitSpan": WAIT_SPAN,
            "minPace": MIN_PACE,
            "strongArms": STRONG_ARMS,
            "troubleDrop": TROUBLE_DROP,
            "clips": {
                f"{move}_{side}": {
                    "duration": clip["duration"],
                    "qpos": np.round(clip["qpos"], 5).ravel().tolist(),
                    "feet": np.round(clip["feet"], 5).ravel().tolist(),
                    "phase": clip["phase"],
                }
                for (move, side), clip in env.choreo.floor.clips.items()
            },
        },
        "choices": json.loads(Path(args.choices).read_text()) if Path(args.choices).exists() else None,
    }
    (out / "reference.json").write_text(json.dumps(reference, separators=(",", ":")))

    # Parity fixture: from a few states mid-ginga, the observation, action and
    # resulting state over three control steps. Both sides start with
    # mj_forward, then run exactly what the live loop runs.
    # Also: the ginga somewhere else on the floor, a fall just begun, one about to land (the way up moves to
    # where the body is), and partway up from a negativa and from hands and knees.
    samples = []
    cases = [(66.0, None, None, {}), (120.0, None, None, {}), (90.0, "martelo", "r", {}), (78.0, "bencao", "l", {})]
    cases += [(84.0, "esquiva", "r", {}), (104.0, "esquiva", "l", {})]
    # A kick and an esquiva ending within the three compared steps: the triangle moves to the feet
    # (Choreography.reanchor).
    cases += [(96.0, "esquiva", "l", {"kick_t": env.choreo.kicks.duration("esquiva") - 12 * env.control_dt - 0.05, "frame": (0.2, 0.1, 0.4)})]
    cases += [(88.0, "martelo", "l", {"kick_t": env.choreo.kicks.duration("martelo") - 12 * env.control_dt - 0.05, "frame": (-0.3, 0.2, 1.1)})]
    cases += [(96.0, None, None, {"frame": (0.5, -0.4, 0.9)}), (96.0, "fall", None, {"frame": (0.5, -0.4, 0.9)})]
    cases += [(96.0, "landing", None, {"frame": (0.5, -0.4, 0.9)})]
    cases += [(110.0, "up", None, {"recover": {"move": "negativa", "side": "l", "t": FALL_TIME + 0.6, "place": (-0.3, 0.7, -1.2)}})]
    cases += [(70.0, "up", None, {"recover": {"move": "joelhos", "side": "r", "t": FALL_TIME + 1.1, "place": (0.2, 0.1, 2.0)}})]
    for bpm, move, side, extra in cases:
        options = {"phase": 0.3, "bpm": bpm, "moves": False, **extra}
        if move in MOVES:
            options = {"move": move, "side": side, "kick_t": 0.1, "anchor": (0.04, -0.06), **options}
        obs, _ = env.reset(options=options)
        brain.reset()
        for _ in range(12 if move else 40):
            obs, *_ = env.step(brain(env, obs))
        if move in ("fall", "landing"):
            env.choreo.begin_recover(env.d.qpos.copy(), env.lean() + np.array([0.3, 0.1]), env.bent(), float(env.d.qvel[1]))
            for _ in range(3 if move == "fall" else int(FALL_TIME / env.control_dt) - 1):
                obs, *_ = env.step(brain(env, env._observe()))
        for _ in range(5):
            if steady(env, brain):
                break
            obs, *_ = env.step(brain(env, env._observe()))
        c = env.choreo
        r = c.recover
        start = {
            "qpos": env.d.qpos.tolist(),
            "qvel": env.d.qvel.tolist(),
            "phase": env.phase,
            "bpm": bpm,
            "move": c.move,
            "side": c.side,
            "kickT": c.kick_t,
            "sinceKick": min(brain.since_kick, 1e9),
            "anchor": c.anchor.tolist(),
            "placement": c.frame.tolist(),
            "recover": r and {"move": r["move"], "side": r["side"], "t": r["t"], "onset": r["onset"].tolist(), "place": r["place"].tolist(), "sink": r.get("sink", 0.0)},
            "sinceRecover": min(brain.since_recover, 1e9),
        }
        mujoco.mj_forward(m, env.d)
        steps = []
        for _ in range(3):
            obs = env._observe()
            action = brain(env, obs)
            env.act(action)
            mujoco.mj_step(m, env.d, nstep=FRAME_SKIP)
            env.advance()
            steps.append({"obs": obs.tolist(), "action": action.tolist(), "qpos": env.d.qpos.tolist()})
        samples.append({"start": start, "steps": steps})
    Path(args.fixture).parent.mkdir(parents=True, exist_ok=True)
    Path(args.fixture).write_text(json.dumps(samples))
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"{', '.join(networks)} networks, {offset} weights; {size:.1f} MB in {out}; fixture {args.fixture}")


def steady(env: GingaEnv, brain: Brain) -> bool:
    """Whether three control steps from here barely move when the pose moves by 5 µm, as float32 rounding and
    two MuJoCo builds move it. A contact right at its edge is chaotic: from a foot 10 µm into the floor, 5 µm
    in the pose moved the kicking leg 1.5 mm three steps later in Python itself, and the browser can't match
    that to 0.1 mm — so the fixture moves on a step from such a state. Whether a nudge finds the edge depends on
    its direction: a knee landing on the floor passed three of them and moved 4.9 mm on the fourth."""
    rng = np.random.default_rng(0)
    ends = []
    for noise in (0.0,) + (5e-6,) * 12:
        e, b = copy.deepcopy((env, brain))
        e.d.qpos[:] += rng.uniform(-noise, noise, e.d.qpos.shape)
        mujoco.mj_forward(e.m, e.d)
        for _ in range(3):
            e.act(b(e, e._observe()))
            mujoco.mj_step(e.m, e.d, nstep=FRAME_SKIP)
            e.advance()
        ends.append(e.d.qpos.copy())
    return max(float(np.abs(q - ends[0]).max()) for q in ends[1:]) < 5e-5


if __name__ == "__main__":
    main()
