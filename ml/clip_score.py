"""Score GVHMR clips as training references: the moves they hold, where each starts and ends standing, how clean the
feet are — the physics filter before plant_clip.py (after KungfuBot, Xie et al. 2025).

    uv run python clip_score.py data/hmr/*.npz [--json data/hmr/moves.json]

For every clip (hmr_modal.py's .npz: every SMPL-X joint per frame, Y up, floor at 0) it finds the moves — a foot
rising above KICK_HEIGHT (a kick) or the hips above the head (an aú, a macaco…) — and for each the window from the
last frame standing still before it to the first after it: both feet down and the centre of mass within STABLE of
the line between them, for at least SETTLED frames. Per move it prints the window and its length, the peak foot or
hip height, the moving foot's peak speed (a replay in slow motion kicks at a fraction of a kick's speed), the worst
stability margin at the window's ends and how far a standing foot slid. A window that never settles at one end is
open there; plant_clip.py wants a closed one, cut through the landing.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

KICK_HEIGHT = 0.6  # m: an ankle this high is a kick (the hips stand at ~0.9)
STABLE = 0.25  # m: the centre of mass this close to the support is standing still (KungfuBot allows 0.1 on corrected feet; GVHMR's stand 10–20 cm off)
SETTLED = 5  # frames both feet stand stably before a move starts or after it ends
STILL = 0.5  # m/s: a foot slower than this, and…
LOW = 0.3  # m: …this close to the floor, stands (GVHMR floats a landed foot up to ~25 cm)…
FLOOR_WINDOW = 2.0  # s: …where the floor is the lowest foot this near in time (GVHMR's floor drifts along a clip)
PEAK_GAP = 0.5  # s: two peaks closer than this are one move
INVERTED = 0.1  # m: the hips this far above the head, with a hand on the floor (HANDS_DOWN), is an aú, a macaco…
HANDS_DOWN = 0.3  # m
REPLAY = 0.7  # a move whose foot peaks below this share of the fastest in the same video is a slow-motion replay
# Share of body mass per segment (de Leva 1996, rounded), by the SMPL-X joints that bound it.
MASS = {
    ("head",): 0.07,
    ("pelvis", "neck"): 0.43,
    ("left_shoulder", "left_elbow"): 0.027,
    ("right_shoulder", "right_elbow"): 0.027,
    ("left_elbow", "left_wrist"): 0.022,
    ("right_elbow", "right_wrist"): 0.022,
    ("left_hip", "left_knee"): 0.14,
    ("right_hip", "right_knee"): 0.14,
    ("left_knee", "left_ankle"): 0.06,
    ("right_knee", "right_ankle"): 0.06,
}
FOOT = {side: [f"{side}_heel", f"{side}_big_toe", f"{side}_small_toe"] for side in ("left", "right")}


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Each stretch of True as (first, past-the-last)."""
    edges = np.flatnonzero(np.diff(np.r_[0, flags.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


class Clip:
    def __init__(self, path: str):
        data = np.load(path)
        self.name, self.t, self.fps = path, data["t"], float(data["fps"])
        joints, index = data["joints"], {n: i for i, n in enumerate(data["names"])}
        self.at = lambda name: joints[:, index[name]]
        total = sum(MASS.values())
        self.com = sum(share * joints[:, [index[b] for b in bones]].mean(axis=1) for bones, share in MASS.items()) / total
        self.feet = {side: joints[:, [index[n] for n in names]].mean(axis=1) for side, names in FOOT.items()}
        floor = minimum_filter1d(np.minimum(self.feet["left"][:, 1], self.feet["right"][:, 1]), int(FLOOR_WINDOW * self.fps))
        self.speed = {side: np.linalg.norm(np.gradient(f, 1 / self.fps, axis=0), axis=1) for side, f in self.feet.items()}
        self.standing = {side: (self.speed[side] < STILL) & (f[:, 1] < floor + LOW) for side, f in self.feet.items()}
        self.margin = np.array([self._margin(i) for i in range(len(self.t))])
        both = self.standing["left"] & self.standing["right"] & (self.margin < STABLE)
        self.settled = np.zeros(len(self.t), bool)
        for a, b in runs(both):
            if b - a >= SETTLED:
                self.settled[a:b] = True

    def _margin(self, i: int) -> float:
        """Horizontal distance from the centre of mass to the support: the standing foot, or the line between both."""
        points = [self.feet[side][i, [0, 2]] for side in FOOT if self.standing[side][i]]
        p = self.com[i, [0, 2]]
        if not points:
            return np.inf
        if len(points) == 1:
            return float(np.linalg.norm(p - points[0]))
        a, b = points
        u = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0.0, 1.0)
        return float(np.linalg.norm(p - (a + u * (b - a))))

    def peaks(self) -> list[tuple[str, int]]:
        """(kind, frame) per move: the highest frame of each stretch where a foot is kicking or the body is inverted."""
        ankle = np.maximum(self.at("left_ankle")[:, 1], self.at("right_ankle")[:, 1])
        height = gaussian_filter1d(ankle, 1.0)
        hips, head = self.at("pelvis")[:, 1], self.at("head")[:, 1]
        hands = np.minimum(self.at("left_wrist")[:, 1], self.at("right_wrist")[:, 1])
        found = []
        for kind, flags, value in (("kick", height > KICK_HEIGHT, height), ("inverted", (hips > head + INVERTED) & (hands < HANDS_DOWN), hips)):
            for a, b in runs(flags):
                peak = a + int(np.argmax(value[a:b]))
                if not found or all(abs(peak - f) > PEAK_GAP * self.fps or k != kind for k, f in found):
                    found.append((kind, peak))
        return sorted(found, key=lambda kf: kf[1])

    def window(self, peak: int) -> tuple[int | None, int | None]:
        """The last settled frame before the peak and the first after it (None: never settles on that side)."""
        before = np.flatnonzero(self.settled[:peak])
        after = np.flatnonzero(self.settled[peak:])
        return (int(before[-1]) if len(before) else None), (int(peak + after[0]) if len(after) else None)

    def slide(self, a: int, b: int) -> float:
        """How far a standing foot moved over the floor inside the window, at worst (m)."""
        worst = 0.0
        for side in FOOT:
            for s, e in runs(self.standing[side][a:b]):
                path = self.feet[side][a + s : a + e, [0, 2]]
                worst = max(worst, float(np.linalg.norm(path - path[0], axis=1).max()))
        return worst

    def moves(self) -> list[dict]:
        out = []
        for kind, peak in self.peaks():
            a, b = self.window(peak)
            lo, hi = a if a is not None else 0, b if b is not None else len(self.t) - 1
            height = max(self.at("left_ankle")[peak, 1], self.at("right_ankle")[peak, 1]) if kind == "kick" else self.at("pelvis")[peak, 1]
            fast = max(self.speed[side][lo : hi + 1].max() for side in FOOT)
            out.append({
                "clip": self.name,
                "kind": kind,
                "peak": round(float(self.t[peak]), 2),
                "start": round(float(self.t[a]), 2) if a is not None else None,
                "end": round(float(self.t[b]), 2) if b is not None else None,
                "seconds": round((hi - lo) / self.fps, 2),
                "height": round(float(height), 2),
                "foot_speed": round(float(fast), 1),
                "margin": round(float(max(self.margin[lo], self.margin[hi])), 3),
                "slide": round(self.slide(lo, hi + 1), 3),
            })
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--json", help="also write every move here")
    args = parser.parse_args()
    found = [move for path in args.clips for move in Clip(path).moves()]
    # A video that shows a move at full speed and then slowed down: the replay's foot is the slow one.
    fastest: dict[str, float] = {}
    for m in found:
        m["video"] = Path(m["clip"]).stem.rsplit("_", 1)[0]
        fastest[m["video"]] = max(fastest.get(m["video"], 0.0), m["foot_speed"])
    for m in found:
        m["slow_motion"] = m["foot_speed"] < REPLAY * fastest[m["video"]]
    print(f"{'clip':34s} {'kind':8s} {'peak':>6s} {'start':>6s} {'end':>6s} {'s':>5s} {'height':>6s} {'m/s':>5s} slow {'margin':>6s} {'slide':>5s}")
    for m in found:
        start, end = (f"{m[k]:6.2f}" if m[k] is not None else "  open" for k in ("start", "end"))
        print(f"{m['clip'][-34:]:34s} {m['kind']:8s} {m['peak']:6.2f} {start} {end} {m['seconds']:5.2f} {m['height']:6.2f} {m['foot_speed']:5.1f} {'yes ' if m['slow_motion'] else 'no  '} {m['margin']:6.3f} {m['slide']:5.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(found, f, indent=1)
        print(f"{len(found)} moves → {args.json}")


if __name__ == "__main__":
    main()
