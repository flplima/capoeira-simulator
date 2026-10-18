"""Estimate 3D body landmarks from a reference video with MediaPipe Pose.

    uv run python capture.py <video> <start_s> <end_s> <out.npz>

Writes per-frame world landmarks (metres, hip-centred, camera-oriented),
normalised image landmarks and visibility for the 33 MediaPipe points.
"""

import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode


def main(video: str, start: float, end: float, out: str) -> None:
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/pose_landmarker_heavy.task"),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    times, world, image, visibility = [], [], [], []
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            ok, frame = capture.read()
            if not ok or t > end:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
            if not result.pose_world_landmarks:
                continue
            times.append(t)
            world.append([[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]])
            image.append([[p.x, p.y, p.z] for p in result.pose_landmarks[0]])
            visibility.append([p.visibility for p in result.pose_landmarks[0]])
    np.savez(out, t=np.array(times), world=np.array(world), image=np.array(image), visibility=np.array(visibility), fps=fps)
    print(f"{len(times)} frames with a pose between {start}s and {end}s → {out}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
