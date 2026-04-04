import numpy as np
import pytest
import torch

from hand_gestures.landmarks import augment, features, joint_angles, normalize, palm_size, to_isotropic
from hand_gestures.models import GestureGRU, GestureTCN, HandGCN, MLPClassifier

FIXTURE = np.load("tests/fixtures/hagrid_small.npz")
HAND = FIXTURE["points"][0]


def test_normalize_is_translation_and_scale_invariant():
    moved = HAND * 3.0 + np.array([0.4, -0.2])
    np.testing.assert_allclose(normalize(moved), normalize(HAND), atol=1e-5)
    assert np.allclose(normalize(HAND)[0], 0) and np.isclose(palm_size(normalize(HAND)), 1.0, atol=1e-5)


def test_straight_finger_has_zero_angles():
    hand = np.zeros((21, 2), np.float32)
    for k, j in enumerate(range(1, 21)):
        hand[j] = [(j - 1) // 4 * 0.2 - 0.4, -(k % 4 + 1) * 0.1]  # five straight vertical fingers
    np.testing.assert_allclose(joint_angles(hand)[1:3], 0.0, atol=1e-3)  # thumb joints 2 and 3


def test_feature_dimensions_and_augmentation():
    assert features(HAND).shape == (67,) and features(FIXTURE["points"][:5]).shape == (5, 67)
    aug = augment(HAND, np.random.default_rng(0))
    assert aug.shape == (21, 2) and not np.allclose(aug, normalize(HAND))


def test_to_isotropic():
    p = to_isotropic(np.array([[0.5, 0.5]]), width=640, height=480)
    np.testing.assert_allclose(p, [[0.5 * 640 / 480, 0.5]])


@pytest.mark.parametrize("model,shape", [(MLPClassifier(34), (4, 67)), (HandGCN(34), (4, 21, 2))])
def test_static_models(model, shape):
    assert model(torch.randn(*shape)).shape == (4, 34)


def test_tcn_is_causal():
    model = GestureTCN(11, in_dim=46).eval()
    x = torch.randn(1, 32, 46)
    y = x.clone()
    y[:, -1] += 5.0  # change only the last frame
    with torch.no_grad():
        hx = model.blocks(model.input(x.transpose(1, 2)))
        hy = model.blocks(model.input(y.transpose(1, 2)))
    torch.testing.assert_close(hx[..., :-1], hy[..., :-1])  # earlier outputs never see the future
    assert GestureGRU(11, in_dim=46)(x).shape == (1, 11)
