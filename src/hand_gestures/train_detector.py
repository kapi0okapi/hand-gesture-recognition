"""Train the end-to-end YOLO gesture detector on the HaGRID image subset (resumable).

hg-train-detector --data data/hagrid_yolo/data.yaml --model yolo26s.pt --name yolo_hagrid
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--name", default="yolo_hagrid")
    parser.add_argument("--project", default=os.environ.get("RUNS", "runs"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    project = Path(os.path.expandvars(args.project)).resolve()  # relative paths would land under runs/
    last = project / args.name / "weights" / "last.pt"
    if last.exists():
        YOLO(str(last)).train(resume=True)
        return
    YOLO(args.model).train(
        data=os.path.expandvars(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project),
        name=args.name,
        exist_ok=True,
        cos_lr=True,
        patience=20,
        fliplr=0.5,
        plots=True,
        nms=False,  # validate (and pick best.pt) on YOLO26's NMS-free head, as used by hg-compare
    )


if __name__ == "__main__":
    main()
