"""Run MJX Capoeira training on Modal's serverless GPU tier ($30/mo free credit).

First-time setup:
    uv run modal setup

Launch a training run on an NVIDIA L4 GPU (default) and save checkpoints to both
Modal Volume `capoeira-mjx-runs` and local `runs/<out>/`:
    uv run modal run modal_train.py --skill kicks --steps 40000000 --minutes 30 --out runs/mjx_kicks --init runs/expert_kicks11/last.pt
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
    .add_local_file(HERE / "env.py", "/workspace/env.py")
    .add_local_file(HERE / "env_mjx.py", "/workspace/env_mjx.py")
    .add_local_file(HERE / "train_mjx.py", "/workspace/train_mjx.py")
    .add_local_dir(HERE / "data", "/workspace/data")
)

vol = modal.Volume.from_name("capoeira-mjx-runs", create_if_missing=True)
app = modal.App("capoeira-mjx", image=image)


@app.function(gpu="L4", timeout=60 * 60 * 2, volumes={"/runs": vol})
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
        "--cache-dir",
        str(cache_dir),
        "--out",
        str(out_dir),
    ]
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
) -> None:
    init_bytes = None
    if init:
        init_path = Path(init)
        if not init_path.is_absolute():
            init_path = HERE / init_path
        init_bytes = init_path.read_bytes()

    local_out = Path(out)
    run_name = local_out.name

    artifacts = remote_train.remote(
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
        init_bytes=init_bytes,
    )

    if not local_out.is_absolute():
        local_out = HERE / local_out
    local_out.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        (local_out / name).write_bytes(data)
    print(f"Saved {', '.join(artifacts.keys())} to {local_out}")
