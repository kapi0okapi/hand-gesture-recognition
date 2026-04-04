"""Two paradigms on the same held-out photos (people never seen in training):

* keypoints: MediaPipe hand landmarks -> landmark classifier (this repo)
* detector:  YOLO trained end-to-end on HaGRID photos, box + gesture in one pass

Each ground-truth hand counts as correct when a prediction overlaps it (IoU >= 0.3 for the
landmark box, 0.5 for YOLO) with the right gesture. Missed hands count as wrong.

    hg-compare --data data/hagrid_yolo --static runs/static_mlp/best.pt \\
        --yolo runs/yolo_hagrid/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from hand_gestures.engine import StaticClassifier
from hand_gestures.landmarks import to_isotropic


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ground_truth(label_file: Path, classes: list[str]) -> list[tuple[np.ndarray, str]]:
    hands = []
    for line in label_file.read_text().split("\n"):
        if line.strip():
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = map(float, (cx, cy, w, h))
            hands.append((np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]), classes[int(c)]))
    return hands


def score(
    gt: list[tuple[np.ndarray, str]], preds: list[tuple[np.ndarray, str]], min_iou: float
) -> tuple[int, int]:
    """(hands found, hands found with the right gesture), greedy one-to-one matching."""
    used, found, correct = set(), 0, 0
    for box, label in gt:
        candidates = [(iou(box, p), i) for i, (p, _) in enumerate(preds) if i not in used]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= min_iou:
            used.add(best[1])
            found += 1
            correct += preds[best[1]][1] == label
    return found, correct


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="folder written by hagrid_images")
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--yolo", type=Path)
    parser.add_argument("--landmarker", default="models/hand_landmarker.task")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=Path("comparison.json"))
    parser.add_argument("--device", default="cpu", help="for the detector; MediaPipe always runs on the CPU")
    args = parser.parse_args(argv)

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    classifier = StaticClassifier(args.static)
    classes = classifier.classes
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.landmarker),
            num_hands=2,
            min_hand_detection_confidence=0.3,
        )
    )
    yolo = None
    if args.yolo:
        from ultralytics import YOLO

        yolo = YOLO(str(args.yolo))

    totals = {"hands": 0, "keypoints": [0, 0, 0.0], "detector": [0, 0, 0.0]}  # found, correct, seconds
    images = sorted((args.data / "images" / args.split).glob("*.jpg"))
    for path in images:
        gt = ground_truth(args.data / "labels" / args.split / f"{path.stem}.txt", classes)
        if not gt:
            continue
        bgr = cv2.imread(str(path))
        h, w = bgr.shape[:2]
        totals["hands"] += len(gt)

        started = time.perf_counter()
        result = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        )
        preds = []
        if result.hand_landmarks:
            raw = np.array([[[p.x, p.y] for p in hand] for hand in result.hand_landmarks], np.float32)
            labels = [classes[i] for i in classifier(to_isotropic(raw, w, h)).argmax(1)]
            for hand, label in zip(raw, labels, strict=True):
                margin = 0.1 * (hand.max(0) - hand.min(0))
                preds.append((np.r_[hand.min(0) - margin, hand.max(0) + margin], label))
        totals["keypoints"][2] += time.perf_counter() - started
        found, correct = score(gt, preds, 0.3)
        totals["keypoints"][0] += found
        totals["keypoints"][1] += correct

        if yolo is not None:
            started = time.perf_counter()
            boxes = yolo.predict(bgr, imgsz=640, conf=0.25, nms=False, device=args.device, verbose=False)[
                0
            ].boxes
            totals["detector"][2] += time.perf_counter() - started
            preds = [
                (b / [w, h, w, h], yolo.names[int(c)])
                for b, c in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), strict=True)
            ]
            found, correct = score(gt, preds, 0.5)
            totals["detector"][0] += found
            totals["detector"][1] += correct

    n_images, n_hands = len(images), max(totals["hands"], 1)
    report = {"images": n_images, "hands": totals["hands"]}
    for method in ("keypoints", "detector") if yolo is not None else ("keypoints",):
        found, correct, seconds = totals[method]
        report[method] = {
            "hand_recall": found / n_hands,
            "gesture_accuracy": correct / n_hands,
            "ms_per_image": 1000 * seconds / max(n_images, 1),
        }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
