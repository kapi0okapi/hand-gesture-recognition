"""Training smoke tests on a small sample of real HaGRID landmarks, export, and dataset tools."""

import functools
import http.server
import io
import json
import threading
import zipfile

import numpy as np
import pytest

from hand_gestures.compare import iou, score
from hand_gestures.data.hagrid import convert
from hand_gestures.data.hagrid_images import fetch_member
from hand_gestures.train_dynamic import TrainConfig as DynamicConfig
from hand_gestures.train_dynamic import train as train_dynamic
from hand_gestures.train_static import TrainConfig, train
from hand_gestures.utils.config import load_config

FIXTURE = "tests/fixtures/hagrid_small.npz"


@pytest.mark.parametrize("model", ["mlp", "gcn"])
def test_static_training(tmp_path, model):
    cfg = load_config(
        TrainConfig,
        "configs/smoke_static.yaml",
        [f"data.path={FIXTURE}", f"model.name={model}", "optim.epochs=3"],
    )
    result = train(cfg, tmp_path / model)
    assert result["test_accuracy"] > 0.5  # 4 classes, 240 training hands, people unseen in training
    report = json.loads((tmp_path / model / "test_report.json").read_text())
    assert len(report["confusion"]) == 4


def test_dynamic_training_and_export(tmp_path):
    cfg = load_config(
        DynamicConfig,
        None,
        [
            f"data.landmarks={FIXTURE}",
            "data.train_size=256",
            "data.val_size=64",
            "data.batch_size=64",
            "data.num_workers=0",
            "epochs=1",
            "device=cpu",
        ],
    )
    train_dynamic(cfg, tmp_path / "dyn")
    static_cfg = load_config(TrainConfig, "configs/smoke_static.yaml", [f"data.path={FIXTURE}"])
    train(static_cfg, tmp_path / "static")
    from hand_gestures import export

    export.main(
        [
            "--static",
            str(tmp_path / "static" / "best.pt"),
            "--dynamic",
            str(tmp_path / "dyn" / "best.pt"),
            "--out",
            str(tmp_path / "onnx"),
        ]
    )
    assert (tmp_path / "onnx" / "static_gesture.onnx").exists() and (
        tmp_path / "onnx" / "dynamic_gesture.onnx"
    ).exists()


def test_convert_annotations(tmp_path):
    record = {
        "bboxes": [[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
        "labels": ["like", "no_gesture"],
        "user_id": "u1",
        "hand_landmarks": [[[0.2, 0.2]] * 21, []],
    }
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as z:
        for split in ("train", "test"):
            z.writestr(f"annotations/{split}/like.json", json.dumps({f"{split}_img": record}))
            z.writestr(f"annotations/{split}/no_gesture.json", json.dumps({}))
    stats = convert(tmp_path / "a.zip", tmp_path / "out.npz")
    data = np.load(tmp_path / "out.npz")
    assert stats["hands"] == 2  # the hand without landmarks is skipped
    assert list(data["classes"]) == ["like", "no_gesture"] and data["points"].shape == (2, 21, 2)


def test_fetch_member_from_remote_zip(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("a/first.jpg", b"x" * 5000)
        z.writestr("a/second.jpg", b"payload-" * 100)
    (tmp_path / "images.zip").write_bytes(buffer.getvalue())
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        info = zipfile.ZipFile(tmp_path / "images.zip").getinfo("a/second.jpg")
        import requests

        data = fetch_member(requests.Session(), f"http://127.0.0.1:{server.server_port}/images.zip", info)
    finally:
        server.shutdown()
    assert data == b"payload-" * 100


def test_matching():
    gt = [(np.array([0.1, 0.1, 0.3, 0.3]), "like"), (np.array([0.6, 0.6, 0.8, 0.8]), "palm")]
    preds = [(np.array([0.12, 0.1, 0.3, 0.32]), "like"), (np.array([0.6, 0.6, 0.8, 0.8]), "fist")]
    assert score(gt, preds, 0.5) == (2, 1) and iou(gt[0][0], gt[0][0]) == 1.0
