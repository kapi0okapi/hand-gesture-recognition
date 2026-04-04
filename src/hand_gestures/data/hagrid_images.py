"""A balanced HaGRID image subset in YOLO format, fetched from the 128 GB archive with HTTP
range requests (only the selected photos are downloaded).

    python -m hand_gestures.data.hagrid_images --annotations data/hagrid/annotations.zip \\
        --out data/hagrid_yolo --train-per-class 600 --val-per-class 60 --test-per-class 120

The split follows HaGRID's subject split, so the detector is tested on people it never saw.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from hand_gestures.data.hagrid import IMAGES_512_URL, iter_annotation_files
from hand_gestures.data.remote_zip import HttpRangeFile

log = logging.getLogger(__name__)


def select(annotations: Path, per_split: dict[str, int], seed: int = 0) -> tuple[dict, list[str]]:
    """{split: {image_id: (class_folder, boxes, labels)}} with a fixed number of photos per class."""
    rng = np.random.default_rng(seed)
    chosen: dict[str, dict] = {s: {} for s in per_split}
    with zipfile.ZipFile(annotations) as archive:
        records = list(iter_annotation_files(archive))
        classes = sorted({name for _, name, _ in records})
        for split, name, info in records:
            if split not in per_split:
                continue
            data = json.loads(archive.read(info))
            ids = sorted(data)
            for image_id in rng.permutation(ids)[: per_split[split]]:
                item = data[image_id]
                chosen[split][image_id] = (name, item["bboxes"], item["labels"])
    return chosen, classes


def fetch_member(session: requests.Session, url: str, info: zipfile.ZipInfo) -> bytes:
    """Bytes of one archive member: read the local header to find the data, then inflate it."""
    end = info.header_offset + 30 + len(info.filename.encode()) + 512 + info.compress_size
    response = session.get(url, headers={"Range": f"bytes={info.header_offset}-{end}"}, timeout=60)
    response.raise_for_status()
    raw = response.content
    if response.status_code == 200:  # server ignored the range and sent the whole file
        raw = raw[info.header_offset : end + 1]
    name_len, extra_len = int.from_bytes(raw[26:28], "little"), int.from_bytes(raw[28:30], "little")
    start = 30 + name_len + extra_len
    data = raw[start : start + info.compress_size]
    return zlib.decompress(data, -15) if info.compress_type == zipfile.ZIP_DEFLATED else data


def build(annotations: Path, out: Path, per_split: dict[str, int], workers: int = 16) -> dict:
    chosen, classes = select(annotations, per_split)
    log.info("reading the archive index (~1M entries, ~40 s)")
    index = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(IMAGES_512_URL), buffer_size=1 << 22))
    members = {Path(i.filename).stem: i for i in index.infolist() if i.filename.endswith(".jpg")}
    session = requests.Session()
    counts = {}
    for split, items in chosen.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        def save(item: tuple[str, tuple], split: str = split) -> bool:
            image_id, (folder, boxes, labels) = item
            target = out / "images" / split / f"{folder}_{image_id}.jpg"
            if image_id not in members:
                return False
            if not target.exists():
                target.write_bytes(fetch_member(session, IMAGES_512_URL, members[image_id]))
            lines = [
                f"{classes.index(lab)} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}"
                for (x, y, w, h), lab in zip(boxes, labels, strict=True)
                if lab in classes
            ]
            (out / "labels" / split / f"{folder}_{image_id}.txt").write_text("\n".join(lines) + "\n")
            return True

        with ThreadPoolExecutor(workers) as pool:
            counts[split] = sum(pool.map(save, items.items()))
        log.info("%s: %d images", split, counts[split])
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"
    )
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=600)
    parser.add_argument("--val-per-class", type=int, default=60)
    parser.add_argument("--test-per-class", type=int, default=120)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    per_split = {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class}
    print(build(args.annotations, args.out, per_split, args.workers))


if __name__ == "__main__":
    main()
