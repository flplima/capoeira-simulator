"""Recover a body's 3D motion, path through the room included, from one shot of a video with GVHMR on Modal's GPU.

GVHMR (Shen et al., SIGGRAPH Asia 2024) fits an SMPL-X body to every frame in
a gravity-aligned world, from one camera that may move. The output has the
same arrays capture.py writes, so retarget.py fits it unchanged:

    uv run modal run hmr_modal.py --video data/youtube/ZCLVQIfaPHE.mp4 --start 7.3 --end 12.2 --out data/martelo_hmr.npz --static
    uv run python retarget.py data/martelo_hmr.npz 7.3 12.2 data/martelo_fit_hmr.npz

Run one shot at a time (no camera cut inside start..end); `--static` skips
the camera-motion estimate when the camera doesn't move. Next to the .npz it
writes an .mp4 with the fitted body over the video and in the world. Many
shots at once, in parallel containers, from the JSON `video_shots.py` writes:

    uv run modal run hmr_modal.py::batch --jobs data/youtube/segments.json --out data/hmr

The SMPL and SMPL-X body models need a free sign-up (smpl.is.tue.mpg.de and
smpl-x.is.tue.mpg.de) and go in models/body_models/smpl/SMPL_NEUTRAL.pkl and
models/body_models/smplx/SMPLX_NEUTRAL.npz; the image build copies them in
and downloads GVHMR's other weights.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import modal
import numpy as np

from hmr_points import to_capture

HERE = Path(__file__).resolve().parent
GVHMR = "/gvhmr"
COMMIT = "ee960bb6e2ea2d381aa97f08e9b71ef320b624b1"
# GVHMR's own weights, mirrored on Hugging Face (Google Drive refuses the big ones once their download quota runs out).
WEIGHTS = "camenduru/GVHMR"
WEIGHTS_REVISION = "21b32d5389e2e59c0737d4c4095bbc0b8c23f66b"
WEIGHT_FILES = [
    "gvhmr/gvhmr_siga24_release.ckpt",
    "hmr2/epoch=10-step=25000.ckpt",
    "vitpose/vitpose-h-multi-coco.pth",
    "yolo/yolov8x.pt",
]

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "build-essential", "libgl1", "libglib2.0-0")
    .run_commands(
        f"git clone https://github.com/zju3dv/GVHMR {GVHMR} && git -C {GVHMR} checkout {COMMIT}",
        "pip install numpy==1.23.5 cython",
        "pip install --no-build-isolation chumpy cython_bbox",
        f"pip install -r {GVHMR}/requirements.txt && pip install -e {GVHMR}",
    )
    .run_commands(
        "python -c \"import sys; from huggingface_hub import hf_hub_download; "
        "[hf_hub_download(sys.argv[1], f, revision=sys.argv[2], local_dir=sys.argv[3]) for f in sys.argv[4:]]\" "
        f"{WEIGHTS} {WEIGHTS_REVISION} {GVHMR}/inputs/checkpoints " + " ".join(f"'{f}'" for f in WEIGHT_FILES)
    )
    .apt_install("libtk8.6")  # GVHMR's body_model.py imports `turtle`, which loads Tk
    .add_local_dir(HERE / "models/body_models", f"{GVHMR}/inputs/checkpoints/body_models")
    .add_local_python_source("hmr_points")
)
app = modal.App("capoeira-hmr", image=image)


@app.function(gpu="L4", timeout=30 * 60)
def recover(video: bytes, start: float, end: float, static: bool) -> dict:
    """Run GVHMR on start..end of the video; return every SMPL-X joint per frame and the overlay video."""
    import os
    import shutil
    import tempfile

    import torch
    from smplx.joint_names import JOINT_NAMES

    # A folder of its own per call: GVHMR's demo reuses whatever results it finds in its output folder, and one
    # container serves several calls in a row.
    work = Path(tempfile.mkdtemp(prefix="hmr"))
    source, clip = work / "source", work / "clip.mp4"
    source.write_bytes(video)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-i", str(source), "-t", str(end - start),
         "-r", "30", "-an", "-c:v", "libx264", "-crf", "17", str(clip)],
        check=True,
    )
    flags = ["-s"] if static else []
    subprocess.run(["python", "tools/demo/demo.py", f"--video={clip}", f"--output_root={work}", *flags], cwd=GVHMR, check=True)

    os.chdir(GVHMR)
    from hmr4d.utils.smplx_utils import make_smplx

    out = work / clip.stem
    params = torch.load(out / "hmr4d_results.pt")["smpl_params_global"]
    with torch.no_grad():
        joints = make_smplx("supermotion")(**params).joints.numpy()
    result = {
        "joints": joints,
        "names": JOINT_NAMES[: joints.shape[1]],
        "video": (out / f"{clip.stem}_3_incam_global_horiz.mp4").read_bytes(),
    }
    shutil.rmtree(work)
    return result


def _save(result: dict, start: float, out: Path) -> None:
    capture = to_capture(np.asarray(result["joints"]), list(result["names"]), start)
    np.savez(out, **capture)
    out.with_suffix(".mp4").write_bytes(result["video"])
    print(f"{len(capture['t'])} frames from {start}s → {out} (+ {out.with_suffix('.mp4').name})")


@app.local_entrypoint()
def main(video: str, start: float, end: float, out: str, static: bool = False) -> None:
    _save(recover.remote(Path(video).read_bytes(), start, end, static), start, Path(out))


def _target(out_dir: Path, job: dict) -> Path:
    return out_dir / f"{Path(job['video']).stem}_{job['start']:g}s.npz"


@app.local_entrypoint()
def batch(jobs: str, out: str = "data/hmr", static: bool = True) -> None:
    """Every shot in the JSON at once (those without an .npz in `out` yet); one <video>_<start>s.npz (+ .mp4) per
    shot. A shot GVHMR fails on (nobody in the frame…) is reported and skipped."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [job for job in json.loads(Path(jobs).read_text()) if not _target(out_dir, job).exists()]
    inputs = [(Path(job["video"]).read_bytes(), job["start"], job["end"], static) for job in todo]
    for job, result in zip(todo, recover.starmap(inputs, return_exceptions=True)):
        if isinstance(result, Exception):
            print(f"{job['video']} {job['start']}–{job['end']} s failed: {str(result).strip().splitlines()[-1][:200]}")
        else:
            _save(result, job["start"], _target(out_dir, job))
