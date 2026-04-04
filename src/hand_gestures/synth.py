"""Synthetic dynamic-gesture sequences built from real HaGRID hand shapes.

Public skeleton datasets for dynamic hand gestures are small or hard to obtain, but a dynamic
gesture is mostly a known hand shape moved along a known path. Real hand shapes from HaGRID are
therefore animated with minimum-jerk motion profiles (the smooth bell-shaped velocity of human
reaching movements), then corrupted like a live landmark detector would: jitter, dropped frames,
random position, size and handedness. Real recordings (``hg-record``) serve as the test set.

Every frame becomes the same feature vector at training and at run time (``frame_features``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from hand_gestures.landmarks import WRIST, normalize, palm_size

DYNAMIC_CLASSES = (
    "none",
    "swipe_left",
    "swipe_right",
    "swipe_up",
    "swipe_down",
    "push",
    "pull",
    "rotate_cw",
    "rotate_ccw",
    "grab",
    "release",
)
FRAME_DIM = 46  # 42 shape + 2 relative position + 1 log-scale + 1 valid flag
OPEN_HANDS = ("palm", "stop", "four")
FISTS = ("fist",)


def frame_features(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(T, 21, 2) points in isotropic image units + (T,) validity -> (T, 46) features.
    Position and size are measured relative to the first valid frame, in palm lengths."""
    out = np.zeros((len(points), FRAME_DIM), np.float32)
    if not valid.any():
        return out
    k0 = int(np.argmax(valid))
    ref_pos, ref_size = points[k0, WRIST], palm_size(points[k0])
    sizes = palm_size(points)
    out[:, :42] = normalize(points).reshape(len(points), 42)
    out[:, 42:44] = (points[:, WRIST] - ref_pos) / ref_size
    out[:, 44] = np.log(np.maximum(sizes, 1e-6) / ref_size)
    out[:, 45] = 1.0
    out[~valid] = 0.0
    return out


def min_jerk(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def rotate(shape: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate (T, 21, 2) shapes about the wrist; positive angles are clockwise on screen (y down)."""
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    x, y = shape[..., 0], shape[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


@dataclass
class SynthConfig:
    length: int = 32  # frames per window (~1 s at 30 fps)
    min_duration: float = 0.35  # seconds
    max_duration: float = 0.8
    fps: float = 30.0
    jitter: float = 0.04  # landmark noise in palm lengths
    drop_prob: float = 0.05  # frames without a detected hand


class ShapeBank:
    """Normalised real hand shapes grouped by HaGRID class."""

    def __init__(self, points: np.ndarray, labels: np.ndarray, classes: list[str]):
        self.shapes = normalize(points).astype(np.float32)
        self.by_class = {c: np.flatnonzero(labels == i) for i, c in enumerate(classes)}
        self.all = np.arange(len(labels))

    def sample(self, rng: np.random.Generator, names: tuple[str, ...] = ()) -> np.ndarray:
        pool = np.concatenate([self.by_class[n] for n in names if n in self.by_class] or [self.all])
        shape = self.shapes[rng.choice(pool)].copy()
        if rng.random() < 0.5:  # left or right hand
            shape[:, 0] *= -1
        angle = np.deg2rad(rng.uniform(-20, 20))
        return rotate(shape[None], np.array([angle]))[0]


def synth_sequence(
    bank: ShapeBank, cls: str, rng: np.random.Generator, cfg: SynthConfig
) -> tuple[np.ndarray, np.ndarray]:
    """One window of a dynamic gesture: raw points (T, 21, 2) in image units and validity (T,)."""
    t_len = cfg.length
    duration = int(np.clip(rng.uniform(cfg.min_duration, cfg.max_duration) * cfg.fps, 6, t_len - 2))
    start = int(rng.integers(0, t_len - duration + 1))
    progress = min_jerk((np.arange(t_len) - start) / duration)
    size = rng.uniform(0.06, 0.2)  # palm size in image units
    center = np.tile(rng.uniform(0.3, 0.7, 2), (t_len, 1))
    scale = np.full(t_len, size)
    angle = np.zeros(t_len)

    if cls in ("grab", "release"):
        a, b = bank.sample(rng, OPEN_HANDS), bank.sample(rng, FISTS)
        if cls == "release":
            a, b = b, a
        shapes = (1 - progress)[:, None, None] * a + progress[:, None, None] * b
    else:
        shapes = np.repeat(bank.sample(rng, OPEN_HANDS if cls != "none" else ())[None], t_len, axis=0)

    if cls.startswith("swipe"):
        direction = {"swipe_left": 180.0, "swipe_right": 0.0, "swipe_up": -90.0, "swipe_down": 90.0}[cls]
        theta = np.deg2rad(direction + rng.uniform(-15, 15))
        amplitude = rng.uniform(2.5, 6.0) * size
        center = center + progress[:, None] * amplitude * np.array([np.cos(theta), np.sin(theta)])
    elif cls in ("push", "pull"):
        factor = rng.uniform(0.4, 1.0) if cls == "push" else -rng.uniform(0.3, 0.5)
        scale = size * (1 + factor * progress)
    elif cls.startswith("rotate"):
        sign = 1.0 if cls == "rotate_cw" else -1.0
        angle = sign * np.deg2rad(rng.uniform(45, 120)) * progress
    elif cls == "none":
        mode = rng.integers(4)
        if mode == 1:  # small random wobble
            center = center + np.cumsum(rng.normal(0, 0.05 * size, (t_len, 2)), axis=0)
        elif mode == 2:  # slow drift, well below swipe speed
            drift = rng.normal(0, 1, 2)
            center = (
                center
                + np.linspace(0, 1, t_len)[:, None]
                * drift
                / np.linalg.norm(drift)
                * rng.uniform(0.3, 1.2)
                * size
            )
        elif mode == 3:  # changing between two unrelated static gestures
            other = bank.sample(rng)
            shapes = (1 - progress)[:, None, None] * shapes + progress[:, None, None] * other

    shapes = rotate(shapes, angle)
    points = shapes * scale[:, None, None] + center[:, None, :]
    points = points + rng.normal(0, cfg.jitter, points.shape) * scale[:, None, None]
    valid = rng.random(t_len) >= cfg.drop_prob
    valid[rng.integers(t_len)] = True
    return points.astype(np.float32), valid


class SyntheticGestures(Dataset):
    """Class-balanced synthetic windows, regenerated each epoch from (seed, epoch, index)."""

    def __init__(self, bank: ShapeBank, length: int, seed: int, cfg: SynthConfig | None = None):
        self.bank, self.length, self.seed = bank, length, seed
        self.cfg = cfg or SynthConfig()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        rng = np.random.default_rng((self.seed, self.epoch, i))
        label = i % len(DYNAMIC_CLASSES)
        points, valid = synth_sequence(self.bank, DYNAMIC_CLASSES[label], rng, self.cfg)
        return torch.from_numpy(frame_features(points, valid)), label
