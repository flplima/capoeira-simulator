"""The keyframes of the ways up from each landing, and which points bear weight when."""

import numpy as np

# The ways up, as keyframes (time in keyframe-seconds; pose; channel values in degrees over it), for the
# right-side version (the left one mirrors it). A pose is "landing" (the source clip's pose held on the
# floor), "parallel" or "base_r"/"base_l" (from the queda de quatro's keyframes), "cocorinha" (the deep
# squat of public/data/movements/cocorinha.yaml, on both flat feet), or "hold" (the keyframe before).
# Every lift of a hand or a foot happens with the weight already over what stays down: nothing has to be
# thrown up from the floor, which a torque-limited body can't do.
COCORINHA = {"root.pitch": 5, "torso.flex": 45, "torso.side": 0, "torso.twist": 0, "head.flex": -35,
             "hip_l.flex": 123, "hip_l.abd": 20, "hip_l.rot": 0, "knee_l.flex": 150, "ankle_l.flex": -32,
             "hip_r.flex": 123, "hip_r.abd": 20, "hip_r.rot": 0, "knee_r.flex": 150, "ankle_r.flex": -32}
# Arms low and open in the squat, the hands outside the knees (hanging between them they pass through the legs).
LOW_ARMS = {"shoulder_l.flex": 40, "shoulder_l.abd": 50, "shoulder_l.rot": 0, "elbow_l.flex": 15,
            "shoulder_r.flex": 40, "shoulder_r.abd": 50, "shoulder_r.rot": 0, "elbow_r.flex": 15}
UP = {
    # Queda de quatro: the hands push the hips forward over the feet, into a low squat (the weight on the
    # feet before the hands leave the floor), and the body rises into the base from there.
    "quatro": [
        (0.0, "landing", {}),
        (0.15, "hold", {}),
        (0.6, "landing", {"root.pitch": -25, "torso.flex": 25, "head.flex": 10,
                          "hip_l.flex": 95, "knee_l.flex": 130, "ankle_l.flex": -15,
                          "hip_r.flex": 95, "knee_r.flex": 130, "ankle_r.flex": -15,
                          "shoulder_l.flex": -55, "elbow_l.flex": 15, "shoulder_r.flex": -55, "elbow_r.flex": 15}),
        (1.0, "cocorinha", LOW_ARMS),
        (1.3, "hold", {}),
        (1.75, "parallel", {}),
        (2.15, "base_r", {}),
    ],
    # Negativa: the stretched leg comes in and its foot plants in front, the hand still on the floor; the
    # weight moves onto that foot as the hand pushes off, into a low lunge over the folded leg's ball of the
    # foot, and the body rises into the base with that leg back (the other base from the queda de quatro's).
    "negativa": [
        (0.0, "landing", {}),
        (0.15, "hold", {}),
        (0.6, "landing", {"root.roll": -8, "torso.flex": 15, "torso.side": -8, "torso.twist": -10,
                          "hip_r.flex": 105, "hip_r.abd": 18, "knee_r.flex": 120, "ankle_r.flex": -25}),
        (1.0, "parallel", {"root.pitch": 25, "torso.flex": 15, "head.flex": -35,
                           "hip_r.flex": 100, "hip_r.abd": 15, "hip_r.rot": 0, "knee_r.flex": 115, "ankle_r.flex": -30,
                           "hip_l.flex": 55, "hip_l.abd": 12, "hip_l.rot": 5, "knee_l.flex": 125, "ankle_l.flex": 20}),
        (1.45, "base_l", {}),
    ],
    # Hands and knees: on all fours, one foot steps forward beside its hand, the hips move over it, the
    # hands push off into a low lunge, and the body rises into the base with the other foot back.
    "joelhos": [
        (0.0, "parallel", {"root.pitch": 62, "torso.flex": 12, "head.flex": -45,
                           "shoulder_l.flex": 75, "shoulder_l.abd": 12, "shoulder_l.rot": 0, "elbow_l.flex": 5,
                           "shoulder_r.flex": 75, "shoulder_r.abd": 12, "shoulder_r.rot": 0, "elbow_r.flex": 5,
                           "hip_l.flex": 70, "hip_l.abd": 10, "hip_l.rot": 0, "knee_l.flex": 100, "ankle_l.flex": 40,
                           "hip_r.flex": 70, "hip_r.abd": 10, "hip_r.rot": 0, "knee_r.flex": 100, "ankle_r.flex": 40}),
        (0.15, "hold", {}),
        # the left foot comes forward through the air, on its own side, and plants beside the left hand
        (0.5, "parallel", {"root.pitch": 55, "torso.flex": 15, "head.flex": -40,
                           "shoulder_l.flex": 75, "shoulder_l.abd": 12, "shoulder_l.rot": 0, "elbow_l.flex": 15,
                           "shoulder_r.flex": 75, "shoulder_r.abd": 12, "shoulder_r.rot": 0, "elbow_r.flex": 10,
                           "hip_l.flex": 120, "hip_l.abd": 18, "hip_l.rot": 0, "knee_l.flex": 120, "ankle_l.flex": -15,
                           "hip_r.flex": 65, "hip_r.abd": 10, "hip_r.rot": 0, "knee_r.flex": 100, "ankle_r.flex": 40}),
        # hands still down, the hips move forward over the front foot
        (0.75, "parallel", {"root.pitch": 45, "torso.flex": 18, "head.flex": -35,
                            "shoulder_l.flex": 70, "shoulder_l.abd": 12, "shoulder_l.rot": 0, "elbow_l.flex": 20,
                            "shoulder_r.flex": 70, "shoulder_r.abd": 12, "shoulder_r.rot": 0, "elbow_r.flex": 20,
                            "hip_l.flex": 110, "hip_l.abd": 15, "hip_l.rot": 0, "knee_l.flex": 105, "ankle_l.flex": -30,
                            "hip_r.flex": 35, "hip_r.abd": 10, "hip_r.rot": 5, "knee_r.flex": 100, "ankle_r.flex": 35}),
        # the hands push off, the back knee leaves the floor: a low lunge
        (1.05, "parallel", {"root.pitch": 25, "torso.flex": 22, "head.flex": -30,
                            "hip_l.flex": 95, "hip_l.abd": 15, "hip_l.rot": 5, "knee_l.flex": 105, "ankle_l.flex": -25,
                            "hip_r.flex": 10, "hip_r.abd": 12, "hip_r.rot": 10, "knee_r.flex": 85, "ankle_r.flex": -20}),
        (1.45, "base_r", {}),
    ],
}
# Where each landing comes from: the source clip and the keyframe time its pose is held from (see UP).
SOURCE = {"quatro": ("queda-de-quatro", 1.25), "negativa": ("negativa", 1.2), "joelhos": None}
MOVES = list(UP)
# A point that starts or stops bearing weight takes or gives its share of it over this many keyframe-seconds
# (the centre of mass moves over the new support gradually, instead of jumping when a hand lets go).
HANDOVER = 0.3
# Which points rest on the floor, as keyframe-time windows, for the right-side
# version (the left one swaps l and r). Sites: heel and toe at each foot, the fingers at each hand.
HAND = {"l": ["fingers_l"], "r": ["fingers_r"]}  # the hand has no wrist joint: it rests on the fingers
FOOT = {"l": ["heel_l", "toe_l"], "r": ["heel_r", "toe_r"]}
DOWN = {
    "quatro": [
        (HAND["l"] + HAND["r"], 0.0, 1.2),  # the hands push until the hips are over the feet, in the squat
        (FOOT["l"] + ["toe_r"], 0.0, 9.0),
        (["heel_r"], 0.0, 1.85),  # the back foot's heel lifts as the body rises into the base
    ],
    "negativa": [
        (["toe_l"], 0.0, 9.0),  # the folded leg, on the ball of the foot
        (["heel_r"], 0.0, 0.55),  # the stretched leg, on its heel, dragged in (SLIDE)…
        (FOOT["r"], 0.55, 9.0),  # …and planted flat in front
        (HAND["r"], 0.0, 1.0),  # the hand pushes until the weight is over the feet
    ],
    "joelhos": [
        (HAND["l"] + HAND["r"], 0.0, 0.9),  # the hands push until the hips are over the front foot
        (["knee_r", "toe_r"], 0.0, 0.95),  # the back leg kneels on the ball of its foot…
        (["toe_r"], 0.95, 9.0),  # …then rises onto it
        (["knee_l", "toe_l"], 0.0, 0.2),  # the front leg steps forward…
        (["heel_l", "toe_l"], 0.5, 9.0),  # …and plants beside the hand
    ],
}
# Points that stay on the floor but slide along it, as keyframe-time windows (right-side version): the
# negativa's stretched leg drags its heel in rather than lifting it (lifting it leaves the body on a hand
# and a ball of the foot). A sliding point bears only SLIDE_SHARE of a planted point's weight.
SLIDE = {"negativa": [("heel_r", 0.0, 0.55)]}
SLIDE_SHARE = 0.3
# A kneeling knee rests on the shin's capsule, this far below the knee joint's site.
ABOVE_FLOOR = {"knee_l": 0.054, "knee_r": 0.054}
# The ginga's base keeps the back foot's heel up: at the very end only these bear weight.
BASE_DOWN = {"r": ["heel_l", "toe_l", "toe_r"], "l": ["heel_r", "toe_r", "toe_l"]}



def swap(name: str) -> str:
    return name[:-1] + {"l": "r", "r": "l"}[name[-1]]


def ends_in(move: str, side: str) -> str:
    """Which base ("r": right foot back, "l": left) the way up of this landing and side ends in."""
    base = UP[move][-1][1][-1]
    return base if side == "r" else swap(base)


def down_at(move: str, side: str, t: float, end: float) -> dict[str, tuple[float, float]]:
    """The points on the floor at keyframe time t, each with how much of its share of the weight it has
    taken (fading in over HANDOVER after it touches down) and how much it still holds (fading out over
    HANDOVER before it lifts)."""
    if t >= end - 1e-6:
        return {n: (1.0, 1.0) for n in BASE_DOWN[ends_in(move, side)]}
    out: dict[str, tuple[float, float]] = {}
    windows = [(n, a, b) for sites, a, b in DOWN[move] for n in sites]
    for sites, a, b in DOWN[move]:
        if a <= t < b:
            for n in sites:
                # (a point that stays down from one window into the next doesn't touch down or lift in between)
                taken = min(1.0, (t - a) / HANDOVER) if a > 0 and not any(a2 < a <= b2 for n2, a2, b2 in windows if n2 == n) else 1.0
                held = min(1.0, (b - t) / HANDOVER) if b < end and not any(a2 <= b < b2 for n2, a2, b2 in windows if n2 == n) else 1.0
                out[n if side == "r" else swap(n)] = (max(taken, 0.0), max(held, 0.0))
    return out



NEGATED = {"root.x", "root.yaw", "root.roll", "torso.side", "torso.twist", "head.side", "head.twist"}


def mirror_frame(channels: list[str], frame: list[float]) -> list[float]:
    """The left-side version of a right-side keyframe pose (as mirrorPose in src/data/channels.ts)."""
    value = dict(zip(channels, frame))
    out = []
    for key in channels:
        joint, field = key.split(".")
        other = swap(joint) if joint[-2:] in ("_l", "_r") else joint
        out.append(value[f"{other}.{field}"] * (-1 if key in NEGATED else 1))
    return out


def way_up(clips: dict, move: str) -> dict:
    """The way up (UP) as a sampled clip like the web ones, smoothstepped between keyframes; both sides."""
    channels, fps = clips["channels"], clips["fps"]
    quatro = clips["queda-de-quatro"]["frames"]["r"]
    poses = {"base_r": np.array(quatro[0]), "parallel": np.array(quatro[int(round(0.4 * fps))])}
    poses["base_l"] = np.array(mirror_frame(channels, quatro[0]))
    poses["cocorinha"] = poses["parallel"].copy()
    for k, v in COCORINHA.items():
        poses["cocorinha"][channels.index(k)] = v
    if SOURCE[move] is not None:
        source, hold = SOURCE[move]
        poses["landing"] = np.array(clips[source]["frames"]["r"][int(round(hold * fps))])
    keys = []
    for t, base, values in UP[move]:
        pose = (keys[-1][1] if base == "hold" else poses[base]).copy()
        for k, v in values.items():
            pose[channels.index(k)] = v
        keys.append((t, pose))
    duration = keys[-1][0]
    right = []
    for k in range(int(round(duration * fps)) + 1):
        t = k / fps
        i = max(j for j in range(len(keys)) if keys[j][0] <= t + 1e-9)
        if i == len(keys) - 1:
            right.append(keys[i][1].tolist())
            continue
        (t0, a), (t1, b) = keys[i], keys[i + 1]
        s = (t - t0) / (t1 - t0)
        right.append((a + (b - a) * s * s * (3 - 2 * s)).tolist())
    return {"duration": duration, "frames": {"r": right, "l": [mirror_frame(channels, f) for f in right]}}

