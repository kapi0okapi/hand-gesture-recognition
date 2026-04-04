import numpy as np

from hand_gestures.synth import (
    DYNAMIC_CLASSES,
    FRAME_DIM,
    ShapeBank,
    SynthConfig,
    SyntheticGestures,
    frame_features,
    synth_sequence,
)

FIXTURE = np.load("tests/fixtures/hagrid_small.npz")
BANK = ShapeBank(FIXTURE["points"], FIXTURE["labels"], [str(c) for c in FIXTURE["classes"]])
CLEAN = SynthConfig(jitter=0.0, drop_prob=0.0)


def end_state(cls: str, seed: int = 0) -> np.ndarray:
    points, valid = synth_sequence(BANK, cls, np.random.default_rng(seed), CLEAN)
    return frame_features(points, valid)[-1]


def test_motions_have_the_expected_direction():
    assert end_state("swipe_right")[42] > 1.5 and end_state("swipe_left")[42] < -1.5
    assert end_state("swipe_up")[43] < -1.5 and end_state("swipe_down")[43] > 1.5  # image y points down
    assert end_state("push")[44] > 0.2 and end_state("pull")[44] < -0.2


def test_rotation_direction():
    def wrist_to_middle_angle(cls):
        points, valid = synth_sequence(BANK, cls, np.random.default_rng(3), CLEAN)
        v0, v1 = points[0, 9] - points[0, 0], points[-1, 9] - points[-1, 0]
        return np.arctan2(v0[0] * v1[1] - v0[1] * v1[0], v0 @ v1)  # > 0 = clockwise on screen (y down)

    assert wrist_to_middle_angle("rotate_cw") > 0.5 and wrist_to_middle_angle("rotate_ccw") < -0.5


def test_dropped_frames_are_zeroed():
    points, valid = synth_sequence(BANK, "swipe_left", np.random.default_rng(1), SynthConfig(drop_prob=0.5))
    feats = frame_features(points, valid)
    assert feats.shape == (32, FRAME_DIM) and np.all(feats[~valid] == 0) and np.all(feats[valid, 45] == 1)


def test_dataset_is_balanced_and_deterministic():
    ds = SyntheticGestures(BANK, length=22, seed=0)
    labels = [ds[i][1] for i in range(22)]
    assert sorted(set(labels)) == list(range(len(DYNAMIC_CLASSES))) and labels.count(0) == 2
    assert np.array_equal(ds[5][0].numpy(), ds[5][0].numpy())
