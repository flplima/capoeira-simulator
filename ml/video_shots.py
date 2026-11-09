"""Cut videos into the shots GVHMR runs on, one job per shot, past any title card.

    uv run python video_shots.py --intro 6.8 data/youtube/*.mp4 > data/youtube/segments.json
    uv run modal run hmr_modal.py::batch --jobs data/youtube/segments.json --out data/hmr

ffmpeg's scene detector finds the cuts. Everything before `--intro` seconds is
dropped (a channel's title card; when it fades instead of cutting, the first
shot starts there), and shots shorter than SHORTEST are skipped. The JSON
lists one job (video, start, end) per shot for `hmr_modal.py::batch`.
"""

import argparse
import json
import subprocess

SHORTEST = 2.0  # s: a shorter shot holds no whole move
SCENE = 0.3  # ffmpeg's scene-change score above which a frame starts a new shot


def probe(*args: str) -> str:
    return subprocess.check_output(["ffprobe", "-v", "error", *args], text=True)


def duration(video: str) -> float:
    return float(probe("-show_entries", "format=duration", "-of", "csv=p=0", video))


def cuts(video: str) -> list[float]:
    out = probe("-f", "lavfi", "-i", f"movie={video},select=gt(scene\\,{SCENE})", "-show_entries", "frame=pts_time", "-of", "csv=p=0")
    return [float(line.rstrip(",")) for line in out.split() if line.strip()]


def shots(video: str, intro: float) -> list[dict]:
    edges = [0.0, *cuts(video), duration(video)]
    jobs = []
    for a, b in zip(edges, edges[1:]):
        a = max(a, intro)
        if b - a >= SHORTEST:
            jobs.append({"video": video, "start": round(a, 2), "end": round(b, 2)})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--intro", type=float, default=0.0, help="s of title card to skip at the start of every video")
    args = parser.parse_args()
    print(json.dumps([job for video in args.videos for job in shots(video, args.intro)], indent=1))


if __name__ == "__main__":
    main()
