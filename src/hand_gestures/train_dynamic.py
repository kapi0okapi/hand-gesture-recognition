"""Train the dynamic gesture network on synthetic sequences; optionally evaluate on real recordings.

hg-train-dynamic --config configs/dynamic_tcn.yaml
hg-train-dynamic --config configs/dynamic_tcn.yaml --set model.name=gru --set name=dynamic_gru
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
from torch.utils.data import DataLoader, TensorDataset

from hand_gestures.models import build_temporal
from hand_gestures.synth import (
    DYNAMIC_CLASSES,
    FRAME_DIM,
    ShapeBank,
    SynthConfig,
    SyntheticGestures,
    frame_features,
)
from hand_gestures.train_static import classification_report, load_split
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

log = logging.getLogger("hand_gestures.train_dynamic")


@dataclass
class DataConfig:
    landmarks: str = "${DATA}/hagrid/landmarks.npz"  # source of real hand shapes
    max_shapes: int = 200_000
    train_size: int = 50_000  # fresh synthetic windows per epoch
    val_size: int = 5_000
    batch_size: int = 256
    num_workers: int = 8
    recordings: str = ""  # optional .npz from hg-record, evaluated after training


@dataclass
class TrainConfig:
    name: str = "dynamic_tcn"
    seed: int = 0
    device: str = "auto"
    model: str = "tcn"  # tcn | gru
    epochs: int = 30
    lr: float = 2e-3
    weight_decay: float = 1e-4
    synth: SynthConfig = field(default_factory=SynthConfig)
    data: DataConfig = field(default_factory=DataConfig)


def recordings_dataset(path: Path) -> TensorDataset:
    data = np.load(path)
    feats = np.stack([frame_features(p, v) for p, v in zip(data["points"], data["valid"], strict=True)])
    return TensorDataset(torch.from_numpy(feats), torch.from_numpy(data["labels"].astype(np.int64)))


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        probs.append(model(x.to(device)).float().softmax(-1).cpu().numpy())
        targets.append(y.numpy())
    return classification_report(np.concatenate(probs), np.concatenate(targets), list(DYNAMIC_CLASSES))


def train(cfg: TrainConfig, run_dir: Path, resume: bool = True) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    points, labels, classes = load_split(
        Path(os.path.expandvars(cfg.data.landmarks)), "train", cfg.data.max_shapes
    )
    bank = ShapeBank(points, labels, classes)
    train_set = SyntheticGestures(bank, cfg.data.train_size, cfg.seed, cfg.synth)
    val_set = SyntheticGestures(bank, cfg.data.val_size, cfg.seed + 10_000, cfg.synth)
    kw = {"batch_size": cfg.data.batch_size, "num_workers": cfg.data.num_workers}
    val_loader = DataLoader(val_set, persistent_workers=cfg.data.num_workers > 0, **kw)

    model = build_temporal(cfg.model, len(DYNAMIC_CLASSES), FRAME_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.data.train_size // cfg.data.batch_size
    scheduler = cosine_with_warmup(optimizer, steps, cfg.epochs * steps)
    precision, logger, stop = Precision(device, enabled=False), MetricLogger(run_dir), StopSignal()
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
        )

    for epoch in range(start_epoch, cfg.epochs):
        train_set.set_epoch(epoch)
        model.train()
        started, total = time.time(), 0.0
        for x, y in DataLoader(train_set, shuffle=True, drop_last=True, **kw):
            loss = F.cross_entropy(model(x.to(device)), y.to(device), label_smoothing=0.05)
            precision.step(loss, optimizer, model, max_grad_norm=1.0)
            scheduler.step()
            total += loss.item()
            if stop:
                save_last(epoch)
                return best
        report = evaluate(model, val_loader, device)
        logger.log(
            epoch,
            {
                "train/loss": total / steps,
                "val/accuracy": report["accuracy"],
                "val/macro_f1": report["macro_f1"],
            },
        )
        log.info(
            "epoch %d/%d loss=%.4f synthetic val acc=%.4f (%.0fs)",
            epoch + 1,
            cfg.epochs,
            total / steps,
            report["accuracy"],
            time.time() - started,
        )
        if report["macro_f1"] > best["macro_f1"]:
            best = {"accuracy": report["accuracy"], "macro_f1": report["macro_f1"]}
            save_checkpoint(
                run_dir / "best.pt",
                model=model.state_dict(),
                config=dataclasses.asdict(cfg),
                classes=list(DYNAMIC_CLASSES),
                epoch=epoch,
                val=best,
            )
        save_last(epoch + 1)

    if cfg.data.recordings:
        model.load_state_dict(load_checkpoint(run_dir / "best.pt", device)["model"])
        real = evaluate(
            model,
            DataLoader(recordings_dataset(Path(os.path.expandvars(cfg.data.recordings))), batch_size=64),
            device,
        )
        (run_dir / "real_report.json").write_text(json.dumps(real, indent=1))
        log.info("real recordings: accuracy %.4f macro-F1 %.4f", real["accuracy"], real["macro_f1"])
        best = {**best, "real_accuracy": real["accuracy"]}
    logger.close()
    return best


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
