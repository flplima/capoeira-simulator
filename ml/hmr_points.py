"""Turn GVHMR's SMPL-X joints into the 33 MediaPipe points capture.py writes, so retarget.py fits either.

GVHMR gives every joint in a gravity-aligned world (Y up, metres) with the
body's path through the room. The body is turned and shifted so it starts at
the origin facing +z with the floor at y = 0; `world` keeps MediaPipe's
hip-centred camera axes (x right, y down, z away), and `root` keeps the path.
"""

import numpy as np

FPS = 30.0  # GVHMR re-times every clip to 30 frames per second
# MediaPipe's 33 landmarks in order, each taken from the SMPL-X joint of the same place.
MEDIAPIPE = [
    "nose", "left_eye", "left_eye", "left_eye", "right_eye", "right_eye", "right_eye", "left_ear", "right_ear",
    "nose", "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_big_toe", "right_big_toe",
]
FEET = ["left_heel", "right_heel", "left_big_toe", "right_big_toe"]


def stand_at_origin(joints: np.ndarray, names: list[str]) -> np.ndarray:
    """Turn about the vertical and shift so the body starts at the origin facing +z, with the floor at y = 0."""
    index = {name: i for i, name in enumerate(names)}
    left = joints[0, index["left_hip"]] - joints[0, index["right_hip"]]
    angle = np.arctan2(left[2], left[0])  # turning by this puts the body's left on +x, so it faces +z
    c, s = np.cos(angle), np.sin(angle)
    turn = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    turned = joints @ turn.T
    feet = turned[:, [index[name] for name in FEET], 1].min(axis=1)
    start = turned[0, index["pelvis"]]
    return turned - np.array([start[0], np.percentile(feet, 2), start[2]])


def to_capture(joints: np.ndarray, names: list[str], start: float) -> dict[str, np.ndarray]:
    """The arrays capture.py writes (t, world, visibility, fps), plus the body's path and every joint."""
    index = {name: i for i, name in enumerate(names)}
    placed = stand_at_origin(joints, names)
    root = placed[:, index["pelvis"]]
    points = placed[:, [index[name] for name in MEDIAPIPE]]
    world = (points - (points[:, [23]] + points[:, [24]]) / 2) * np.array([1.0, -1.0, -1.0])
    frames = len(joints)
    return {
        "t": start + np.arange(frames) / FPS,
        "world": world,
        "visibility": np.ones((frames, len(MEDIAPIPE))),
        "fps": FPS,
        "root": root,
        "joints": placed,
        "names": np.array(names),
    }
