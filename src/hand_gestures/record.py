"""Record real dynamic gestures from a webcam as a labelled test set.

    hg-record --out data/recordings.npz --landmarker models/hand_landmarker.task

Keys: 0-9 / a choose the gesture class (shown on screen), SPACE records one window (~1 s),
q quits. Every window is appended to the .npz, so sessions can be continued later. Needs OpenCV
with GUI support.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hand_gestures.landmarks import to_isotropic
from hand_gestures.synth import DYNAMIC_CLASSES

KEYS = "0123456789a"


def append(path: Path, points: np.ndarray, valid: np.ndarray, label: int) -> int:
    if path.exists():
        old = np.load(path)
        data = {
            "points": np.concatenate([old["points"], points[None]]),
            "valid": np.concatenate([old["valid"], valid[None]]),
            "labels": np.concatenate([old["labels"], [label]]),
        }
    else:
        data = {"points": points[None], "valid": valid[None], "labels": np.array([label])}
    np.savez(path, classes=np.array(DYNAMIC_CLASSES), **data)
    return len(data["labels"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/recordings.npz"))
    parser.add_argument("--landmarker", default="models/hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--length", type=int, default=32)
    args = parser.parse_args(argv)

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.landmarker), running_mode=vision.RunningMode.VIDEO
        )
    )
    capture = cv2.VideoCapture(args.camera)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    label, frame_index, saved = 0, 0, 0
    recording: list | None = None  # landmarks of the window being recorded
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            frame_index * 33,
        )
        frame_index += 1
        if recording is not None:
            recording.append(result.hand_landmarks[0] if result.hand_landmarks else None)
            if len(recording) == args.length:
                valid = np.array([p is not None for p in recording])
                points = np.stack(
                    [
                        to_isotropic(np.array([[q.x, q.y] for q in p], np.float32), w, h)
                        if p is not None
                        else np.zeros((21, 2), np.float32)
                        for p in recording
                    ]
                )
                saved = append(args.out, points, valid, label)
                recording = None
        status = f"class [{KEYS[label]}] {DYNAMIC_CLASSES[label]} | saved {saved}"
        status += " | RECORDING" if recording is not None else " | SPACE to record"
        color = (0, 0, 255) if recording is not None else (0, 200, 0)
        cv2.putText(bgr, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("record", bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and recording is None:
            recording = []
        elif chr(key) in KEYS:
            label = KEYS.index(chr(key))
    capture.release()
    landmarker.close()
    print(f"{saved} windows in {args.out}")


if __name__ == "__main__":
    main()
