"""Train a static gesture classifier on HaGRID hand landmarks.

    hg-train-static --config configs/static_mlp.yaml
    hg-train-static --config configs/static_mlp.yaml --set model.name=gcn --set name=static_gcn

Writes metrics.csv, best.pt (by validation macro-F1), and for the best epoch a confusion matrix
and per-class report on the test subjects.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hand_gestures.landmarks import augment, features, normalize
from hand_gestures.models import build_static
from hand_gestures.utils.config import load_config, save_config
from hand_gestures.utils.runtime import (
    MetricLogger,
    Precision,
    StopSignal,
    cosine_with_warmup,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    select_device,
    setup_logging,
)

log = logging.getLogger("hand_gestures.train_static")
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


@dataclass
class DataConfig:
    path: str = "${DATA}/hagrid/landmarks.npz"
    batch_size: int = 1024
    num_workers: int = 8
    max_train: int | None = None  # subsample for quick runs


@dataclass
class ModelConfig:
    name: str = "mlp"  # mlp | gcn


@dataclass
class OptimConfig:
    epochs: int = 30
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.1


@dataclass
class TrainConfig:
    name: str = "static_mlp"
    seed: int = 0
    device: str = "auto"
    amp: bool = False  # tiny models: full precision is fast enough
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


def model_input(points: np.ndarray, model_name: str) -> np.ndarray:
    """Raw (..., 21, 2) points -> what the chosen model consumes."""
    return features(points) if model_name == "mlp" else normalize(points).astype(np.float32)


class LandmarkDataset(Dataset):
    def __init__(self, points: np.ndarray, labels: np.ndarray, model_name: str, train: bool):
        self.points, self.labels = points, labels.astype(np.int64)
        self.model_name, self.train = model_name, train
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        p = self.points[i]
        if self.train:
            if self._rng is None:  # one generator per DataLoader worker
                self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            p = augment(p, self._rng)
        return torch.from_numpy(model_input(p, self.model_name)), self.labels[i]


def load_split(path: Path, split: str, max_items: int | None = None, seed: int = 0):
    data = np.load(path)
    sel = np.flatnonzero(data["split"] == SPLIT_IDS[split])
    if max_items and len(sel) > max_items:
        sel = np.sort(np.random.default_rng(seed).choice(sel, max_items, replace=False))
    return data["points"][sel], data["labels"][sel], [str(c) for c in data["classes"]]


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        probs.append(model(x.to(device)).float().softmax(-1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def classification_report(probs: np.ndarray, targets: np.ndarray, classes: list[str]) -> dict:
    pred = probs.argmax(1)
    n = len(classes)
    cm = np.bincount(targets * n + pred, minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            c: {"precision": round(float(p), 4), "recall": round(float(r), 4)}
            for c, p, r in zip(classes, precision, recall, strict=True)
        },
        "confusion": cm.tolist(),
    }


def train(cfg: TrainConfig, run_dir: Path, resume: bool = True) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    path = Path(os.path.expandvars(cfg.data.path))
    train_pts, train_y, classes = load_split(path, "train", cfg.data.max_train, cfg.seed)
    val_pts, val_y, _ = load_split(path, "val")
    log.info("train %d / val %d hands, %d classes", len(train_y), len(val_y), len(classes))

    kw = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "persistent_workers": cfg.data.num_workers > 0,
    }
    train_loader = DataLoader(
        LandmarkDataset(train_pts, train_y, cfg.model.name, True), shuffle=True, drop_last=True, **kw
    )
    val_loader = DataLoader(LandmarkDataset(val_pts, val_y, cfg.model.name, False), **kw)

    model = build_static(cfg.model.name, len(classes)).to(device)
    o = cfg.optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=o.weight_decay)
    steps = len(train_loader)
    scheduler = cosine_with_warmup(optimizer, int(o.warmup_epochs * steps), o.epochs * steps)
    precision, logger, stop = Precision(device, cfg.amp), MetricLogger(run_dir), StopSignal()
    start_epoch, best = 0, {"macro_f1": -1.0}
    last_path = run_dir / "last.pt"
    if resume and last_path.exists():
        state = load_checkpoint(last_path, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best = state["epoch"], state["best"]

    def save_last(next_epoch: int) -> None:
        save_checkpoint(
            last_path,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            epoch=next_epoch,
            best=best,
            config=dataclasses.asdict(cfg),
            classes=classes,
        )

    for epoch in range(start_epoch, o.epochs):
        model.train()
        started, total = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with precision.autocast():
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=o.label_smoothing)
            precision.step(loss, optimizer, model)
            scheduler.step()
            total += loss.item()
            if stop:
                save_last(epoch)
                return best
        report = classification_report(*predict(model, val_loader, device), classes)
        logger.log(
            epoch,
            {
                "train/loss": total / steps,
                "val/accuracy": report["accuracy"],
                "val/macro_f1": report["macro_f1"],
            },
        )
        log.info(
            "epoch %d/%d loss=%.4f val acc=%.4f macro-F1=%.4f (%.0fs)",
            epoch + 1,
            o.epochs,
            total / steps,
            report["accuracy"],
            report["macro_f1"],
            time.time() - started,
        )
        if report["macro_f1"] > best["macro_f1"]:
            best = {k: report[k] for k in ("accuracy", "macro_f1")}
            save_checkpoint(
                run_dir / "best.pt",
                model=model.state_dict(),
                config=dataclasses.asdict(cfg),
                classes=classes,
                epoch=epoch,
                val=best,
            )
        save_last(epoch + 1)

    state = load_checkpoint(run_dir / "best.pt", device)
    model.load_state_dict(state["model"])
    test_pts, test_y, _ = load_split(path, "test")
    test = classification_report(
        *predict(model, DataLoader(LandmarkDataset(test_pts, test_y, cfg.model.name, False), **kw), device),
        classes,
    )
    (run_dir / "test_report.json").write_text(json.dumps({"classes": classes, **test}, indent=1))
    log.info("test (unseen people): accuracy %.4f, macro-F1 %.4f", test["accuracy"], test["macro_f1"])
    logger.close()
    return {**best, "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-dir", type=str)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    cfg = load_config(TrainConfig, args.config, args.set)
    run_dir = Path(os.path.expandvars(args.run_dir or os.path.join(os.environ.get("RUNS", "runs"), cfg.name)))
    print(json.dumps(train(cfg, run_dir, resume=not args.no_resume), indent=2))


if __name__ == "__main__":
    main()
