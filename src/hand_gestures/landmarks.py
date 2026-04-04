"""21-point hand skeleton (MediaPipe / HaGRID layout): normalisation, features and augmentation.

Points are (x, y) with y pointing down. Normalisation puts the wrist at the origin and divides by
the size of the palm (a nearly rigid quadrilateral), so features do not depend on where the hand is
or how far it is from the camera. In-plane orientation is kept on purpose: HaGRID has classes
such as ``stop`` vs. ``stop_inverted`` that differ only by rotation.
"""

from __future__ import annotations

import numpy as np

WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = (THUMB, INDEX, MIDDLE, RING, PINKY)
TIPS = (4, 8, 12, 16, 20)
EDGES = (
    [(WRIST, f[0]) for f in FINGERS]
    + [(f[i], f[i + 1]) for f in FINGERS for i in range(3)]
    + [(5, 9), (9, 13), (13, 17)]
)

# HaGRID stores landmarks normalised separately by image width and height; most photos are
# portrait 4:3 (683 x 512), so x is multiplied by width/height to make the axes comparable.
HAGRID_ASPECT = 0.75


def palm_size(points: np.ndarray) -> np.ndarray:
    p = points
    sides = [
        np.linalg.norm(p[..., a, :] - p[..., b, :], axis=-1) for a, b in ((0, 5), (0, 9), (0, 17), (5, 17))
    ]
    return np.maximum(np.mean(sides, axis=0), 1e-6)


def normalize(points: np.ndarray) -> np.ndarray:
    """(..., 21, 2) -> wrist at origin, palm size 1."""
    centered = points - points[..., WRIST : WRIST + 1, :]
    return centered / palm_size(points)[..., None, None]


def joint_angles(points: np.ndarray) -> np.ndarray:
    """Bending angle at each of the 15 finger joints (radians, 0 = straight)."""
    angles = []
    for finger in FINGERS:
        chain = (WRIST, *finger)
        for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:], strict=True):
            u = points[..., b, :] - points[..., a, :]
            v = points[..., c, :] - points[..., b, :]
            norms = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
            cos = (u * v).sum(-1) / np.maximum(norms, 1e-12)
            angles.append(np.arccos(np.clip(cos, -1, 1)))
    return np.stack(angles, axis=-1)


def tip_distances(points: np.ndarray) -> np.ndarray:
    """The 10 pairwise distances between fingertips (on normalised points)."""
    tips = points[..., TIPS, :]
    i, j = np.triu_indices(5, k=1)
    return np.linalg.norm(tips[..., i, :] - tips[..., j, :], axis=-1)


def features(points: np.ndarray) -> np.ndarray:
    """(..., 21, 2) raw points -> (..., 67): normalised coordinates, fingertip distances, joint angles."""
    norm = normalize(points)
    flat = norm.reshape(*norm.shape[:-2], 42)
    return np.concatenate([flat, tip_distances(norm), joint_angles(norm)], axis=-1).astype(np.float32)


def augment(points: np.ndarray, rng: np.random.Generator, max_rotation_deg: float = 20.0) -> np.ndarray:
    """Random in-plane rotation, aspect change, mirroring (left vs. right hand) and jitter."""
    p = normalize(points)
    angle = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    c, s = np.cos(angle), np.sin(angle)
    p = p @ np.array([[c, s], [-s, c]])
    p[..., 0] *= rng.uniform(0.75, 1.33)
    if rng.random() < 0.5:
        p[..., 0] *= -1
    return p + rng.normal(0, 0.03, p.shape)


def to_isotropic(points: np.ndarray, width: float, height: float) -> np.ndarray:
    """Image-normalised landmarks (x / width, y / height) -> comparable units on both axes."""
    out = np.array(points, dtype=np.float32, copy=True)
    out[..., 0] *= width / height
    return out
