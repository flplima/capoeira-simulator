"""Run MJX Capoeira training on Modal's serverless GPU tier ($30/mo free credit).

This is the default way to train; `train_mjx.py` runs the same loop on this
machine (CPU or whatever device JAX finds) and writes the same checkpoints.

First-time setup:
    uv run modal setup

Start a run on an NVIDIA L4 GPU. `--detach` keeps it going after this shell
returns, and `--no-wait` makes the entrypoint start it and exit; the run keeps
committing `best.pt`, `last.pt` and `log.jsonl` to Modal Volume
`capoeira-mjx-runs` every 3 minutes:
    uv run modal run --detach modal_train.py --no-wait --skill kicks --minutes 30 --out runs/mjx_kicks --init runs/expert_kicks11/last.pt

Without `--no-wait` the entrypoint blocks and saves the checkpoints to `runs/<out>/` itself.
`--gpu A100-40GB` (or A10G, H100…) picks another GPU than the L4, with `--envs` to match its memory.

Watch a run and bring its checkpoints home:
    uv run modal run modal_train.py::tail --run mjx_kicks
    uv run modal run modal_train.py::fetch --run mjx_kicks

Stop one: `uv run modal app list`, then `uv run modal app stop <app-id>`.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import modal

HERE = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]>=0.4.35",
        "mujoco==3.15.0",
        "mujoco-mjx==3.15.0",
        "optax>=0.2.4",
        "numpy>=1.26,<2.3",
        "scipy>=1.13",
        "gymnasium>=1.0",
        "torch>=2.4",
    )
    .add_local_file(HERE / "humanoid.xml", "/workspace/humanoid.xml")
    .add_local_file(HERE / "env_consts.py", "/workspace/env_consts.py")
    .add_local_file(HERE / "env_math.py", "/workspace/env_math.py")
    .add_local_file(HERE / "env_clips.py", "/workspace/env_clips.py")
    .add_local_file(HERE / "mjx_math.py", "/workspace/mjx_math.py")
    .add_local_file(HERE / "mjx_model.py", "/workspace/mjx_model.py")
    .add_local_file(HERE / "mjx_tables.py", "/workspace/mjx_tables.py")
    .add_local_file(HERE / "mjx_state.py", "/workspace/mjx_state.py")
    .add_local_file(HERE / "mjx_target.py", "/workspace/mjx_target.py")
    .add_local_file(HERE / "mjx_reset.py", "/workspace/mjx_reset.py")
    .add_local_file(HERE / "mjx_step.py", "/workspace/mjx_step.py")
    .add_local_file(HERE / "env_mjx.py", "/workspace/env_mjx.py")
    .add_local_file(HERE / "mjx_eval.py", "/workspace/mjx_eval.py")
    .add_local_file(HERE / "train_nets.py", "/workspace/train_nets.py")
    .add_local_file(HERE / "train_log.py", "/workspace/train_log.py")
    .add_local_file(HERE / "train_mjx.py", "/workspace/train_mjx.py")
    .add_local_dir(HERE / "data", "/workspace/data")
)

vol = modal.Volume.from_name("capoeira-mjx-runs", create_if_missing=True)
app = modal.App("capoeira-mjx", image=image)


# Room for the 4-hour full run, plus compiling and the last checkpoint.
@app.function(gpu="L4", timeout=60 * 60 * 5, volumes={"/runs": vol})
def remote_train(
    run_name: str,
    steps: float,
    envs: int,
    horizon: int,
    minutes: float,
    skill: str,
    lr: float,
    log_std: float,
    epochs: int,
    minibatches: int,
    start_size: int,
    patience: int,
    focus: str,
    only: str,
    gated: bool,
    init_bytes: bytes | None,
) -> dict[str, bytes]:
    import os
    import time

    os.chdir("/workspace")
    cache_dir = Path("/runs/.jax_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)

    out_dir = Path("/runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "train_mjx.py",
        "--steps",
        str(steps),
        "--envs",
        str(envs),
        "--horizon",
        str(horizon),
        "--minutes",
        str(minutes),
        "--skill",
        skill,
        "--lr",
        str(lr),
        "--log-std",
        str(log_std),
        "--epochs",
        str(epochs),
        "--minibatches",
        str(minibatches),
        "--start-size",
        str(start_size),
        "--patience",
        str(patience),
        "--focus",
        focus,
        "--only",
        only,
        "--cache-dir",
        str(cache_dir),
        "--out",
        str(out_dir),
    ]
    if gated:
        cmd.append("--gated")
    if init_bytes is not None:
        init_path = Path("/tmp/init.pt")
        init_path.write_bytes(init_bytes)
        cmd += ["--init", str(init_path)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    last_commit = time.time()
    for line in proc.stdout:
        print(line, end="", flush=True)
        if time.time() - last_commit >= 180.0:
            vol.commit()
            last_commit = time.time()
    rc = proc.wait()
    vol.commit()
    if rc != 0:
        raise RuntimeError(f"train_mjx.py exited with status {rc}")

    artifacts: dict[str, bytes] = {}
    for name in ("best.pt", "last.pt", "log.jsonl"):
        p = out_dir / name
        if p.exists():
            artifacts[name] = p.read_bytes()
    return artifacts


@app.local_entrypoint()
def main(
    steps: float = 40e6,
    envs: int = 2048,
    horizon: int = 64,
    minutes: float = 20.0,
    out: str = "runs/mjx_ginga",
    init: str = "",
    skill: str = "all",
    lr: float = 1e-4,
    log_std: float = -2.3,
    epochs: int = 4,
    minibatches: int = 4,
    start_size: int = -1,
    patience: int = 100,
    focus: str = "",
    only: str = "",
    gated: bool = False,
    gpu: str = "L4",
    wait: bool = True,
) -> None:
    init_bytes = None
    if init:
        init_path = Path(init)
        if not init_path.is_absolute():
            init_path = HERE / init_path
        init_bytes = init_path.read_bytes()

    local_out = Path(out)
    run_name = local_out.name

    train_args = dict(
        run_name=run_name,
        steps=steps,
        envs=envs,
        horizon=horizon,
        minutes=minutes,
        skill=skill,
        lr=lr,
        log_std=log_std,
        epochs=epochs,
        minibatches=minibatches,
        start_size=start_size,
        patience=patience,
        focus=focus,
        only=only,
        gated=gated,
        init_bytes=init_bytes,
    )
    train = remote_train.with_options(gpu=gpu)
    if not wait:
        call = train.spawn(**train_args)
        print(f"Started {run_name} on a Modal {gpu} (call {call.object_id}); `modal_train.py::fetch --run {run_name}` brings it home.")
        return
    artifacts = train.remote(**train_args)

    if not local_out.is_absolute():
        local_out = HERE / local_out
    local_out.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        (local_out / name).write_bytes(data)
    print(f"Saved {', '.join(artifacts.keys())} to {local_out}")


RUN_FILES = ("best.pt", "last.pt", "log.jsonl")


def _read_volume_file(path: str) -> bytes:
    return b"".join(vol.read_file(path))


@app.local_entrypoint()
def fetch(run: str, out: str = "") -> None:
    """Copy a run's checkpoints and log from the volume into runs/<run>/."""
    local_out = Path(out) if out else HERE / "runs" / run
    local_out.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in RUN_FILES:
        try:
            (local_out / name).write_bytes(_read_volume_file(f"{run}/{name}"))
        except FileNotFoundError:
            continue
        saved.append(name)
    if not saved:
        raise SystemExit(f"Nothing for run {run!r} in volume capoeira-mjx-runs")
    print(f"Saved {', '.join(saved)} to {local_out}")


@app.local_entrypoint()
def tail(run: str, lines: int = 5) -> None:
    """Print the last lines of a run's log.jsonl straight from the volume."""
    try:
        log = _read_volume_file(f"{run}/log.jsonl").decode()
    except FileNotFoundError:
        raise SystemExit(f"No log.jsonl for run {run!r} in volume capoeira-mjx-runs")
    print("\n".join(log.strip().splitlines()[-lines:]))
