"""Live demo on a webcam or video file: hand skeletons, static gestures and dynamic gesture events.

    hg-demo --static runs/static_mlp/best.pt --dynamic runs/dynamic_tcn/best.pt --source 0 --show
    hg-demo --static runs/static_mlp/best.pt --source clip.mp4 --out annotated.mp4

``--show`` needs OpenCV with GUI support (``pip install opencv-python``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from hand_gestures.engine import DEFAULT_LANDMARKER, FrameResult, GestureEngine
from hand_gestures.landmarks import EDGES


def draw(
    bgr: np.ndarray, result: FrameResult, fps: float, last_event: tuple[str, float] | None
) -> np.ndarray:
    h, w = bgr.shape[:2]
    for hand in result.hands:
        px = np.stack([hand.points[:, 0] * h, hand.points[:, 1] * h], axis=1).astype(
            int
        )  # isotropic -> pixels
        for a, b in EDGES:
            cv2.line(bgr, tuple(px[a]), tuple(px[b]), (255, 255, 255), 2, cv2.LINE_AA)
        for p in px:
            cv2.circle(bgr, tuple(p), 3, (0, 140, 255), -1, cv2.LINE_AA)
        x0, y0 = px.min(0)
        cv2.putText(
            bgr,
            f"{hand.gesture} {hand.confidence:.2f} ({hand.handedness})",
            (x0, max(20, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
        )
    if last_event and time.time() - last_event[1] < 1.5:
        cv2.putText(
            bgr,
            last_event[0].replace("_", " ").upper(),
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (60, 220, 60),
            3,
        )
    cv2.putText(bgr, f"{fps:.0f} FPS", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return bgr


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path)
    parser.add_argument("--source", default="0", help="webcam index or video file")
    parser.add_argument("--landmarker", default=DEFAULT_LANDMARKER)
    parser.add_argument("--out", type=Path, help="write the annotated video here")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    engine = GestureEngine(args.static, args.dynamic, args.landmarker)
    capture = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
    writer, frame_index, fps, last_event = None, 0, 0.0, None
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        started = time.perf_counter()
        result = engine.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(frame_index * 1000 / source_fps))
        fps = 0.9 * fps + 0.1 / max(time.perf_counter() - started, 1e-6) if frame_index else 0.0
        if result.event:
            last_event = (result.event, time.time())
            print(f"frame {frame_index}: {result.event}")
        frame = draw(bgr, result, fps, last_event)
        if args.out:
            if writer is None:
                writer = cv2.VideoWriter(
                    str(args.out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (frame.shape[1], frame.shape[0]),
                )
            writer.write(frame)
        if args.show:
            cv2.imshow("hand gestures", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_index += 1
    capture.release()
    if writer:
        writer.release()
    engine.close()


if __name__ == "__main__":
    main()
