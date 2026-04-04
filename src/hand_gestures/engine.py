"""Real-time gesture engine: MediaPipe hand landmarks -> static gesture per hand (smoothed) and
dynamic gesture events from the main hand's last second of motion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from hand_gestures.landmarks import to_isotropic
from hand_gestures.models import build_static, build_temporal
from hand_gestures.synth import DYNAMIC_CLASSES, FRAME_DIM, frame_features
from hand_gestures.train_static import model_input
from hand_gestures.utils.runtime import load_checkpoint

DEFAULT_LANDMARKER = "models/hand_landmarker.task"


@dataclass
class Hand:
    points: np.ndarray  # (21, 2) isotropic image units (x * width / height, y)
    handedness: str
    gesture: str = ""
    confidence: float = 0.0


@dataclass
class FrameResult:
    hands: list[Hand] = field(default_factory=list)
    event: str | None = None  # dynamic gesture recognised in this frame


class StaticClassifier:
    def __init__(self, checkpoint: str | Path):
        state = load_checkpoint(Path(checkpoint))
        self.model_name = state["config"]["model"]["name"]
        self.classes = list(state["classes"])
        self.model = build_static(self.model_name, len(self.classes))
        self.model.load_state_dict(state["model"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, points: np.ndarray) -> np.ndarray:
        """(N, 21, 2) -> class probabilities (N, C)."""
        x = torch.from_numpy(model_input(points, self.model_name))
        return self.model(x).softmax(-1).numpy()


class DynamicClassifier:
    def __init__(self, checkpoint: str | Path):
        state = load_checkpoint(Path(checkpoint))
        cfg = state["config"]
        self.length = cfg["synth"]["length"]
        self.model = build_temporal(cfg["model"], len(DYNAMIC_CLASSES), FRAME_DIM)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, points: np.ndarray, valid: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(frame_features(points, valid))[None]
        return self.model(x).softmax(-1)[0].numpy()


class GestureEngine:
    def __init__(
        self,
        static_checkpoint: str | Path,
        dynamic_checkpoint: str | Path | None = None,
        landmarker: str | Path = DEFAULT_LANDMARKER,
        max_hands: int = 2,
        smoothing: float = 0.6,
        event_threshold: float = 0.85,
        cooldown_frames: int = 15,
    ):
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(landmarker)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.static = StaticClassifier(static_checkpoint)
        self.dynamic = DynamicClassifier(dynamic_checkpoint) if dynamic_checkpoint else None
        self.smoothing, self.event_threshold, self.cooldown_frames = (
            smoothing,
            event_threshold,
            cooldown_frames,
        )
        self._ema: dict[str, np.ndarray] = {}
        length = self.dynamic.length if self.dynamic else 32
        self._track: deque = deque(maxlen=length)
        self._cooldown = 0

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> FrameResult:
        import mediapipe as mp

        h, w = rgb.shape[:2]
        detection = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), timestamp_ms
        )
        hands = [
            Hand(to_isotropic(np.array([[p.x, p.y] for p in lms], np.float32), w, h), handed[0].category_name)
            for lms, handed in zip(detection.hand_landmarks, detection.handedness, strict=True)
        ]
        result = FrameResult(hands)
        if hands:
            probs = self.static(np.stack([hand.points for hand in hands]))
            for hand, p in zip(hands, probs, strict=True):
                smoothed = self.smoothing * self._ema.get(hand.handedness, p) + (1 - self.smoothing) * p
                self._ema[hand.handedness] = smoothed
                hand.gesture = self.static.classes[int(smoothed.argmax())]
                hand.confidence = float(smoothed.max())
        result.event = self._dynamic_step(hands)
        return result

    def _dynamic_step(self, hands: list[Hand]) -> str | None:
        if self.dynamic is None:
            return None
        main = max(hands, key=lambda hand: np.ptp(hand.points[:, 1]), default=None)  # the largest hand
        self._track.append(main.points if main is not None else None)
        self._cooldown = max(0, self._cooldown - 1)
        if len(self._track) < self._track.maxlen or self._cooldown:
            return None
        valid = np.array([p is not None for p in self._track])
        if valid.mean() < 0.6:
            return None
        points = np.stack([p if p is not None else np.zeros((21, 2), np.float32) for p in self._track])
        probs = self.dynamic(points, valid)
        label = int(probs.argmax())
        if label != 0 and probs[label] >= self.event_threshold:
            self._cooldown = self.cooldown_frames
            self._track.clear()
            return DYNAMIC_CLASSES[label]
        return None

    def close(self) -> None:
        self.landmarker.close()
