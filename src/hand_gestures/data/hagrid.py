"""HaGRID v2 hand landmarks -> one compact .npz for training.

HaGRID (Kapitanov et al., WACV 2024; v2 2024) has ~1M photos of 33 gestures + "no_gesture" from
~65k people, with 21 hand landmarks per hand. Only the annotations are needed for the landmark
classifier (719 MB instead of 1.5 TB of images). The official split is by person, so test
subjects never appear in training.

    python -m hand_gestures.data.hagrid --out data/hagrid             # download + convert
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from hand_gestures.landmarks import HAGRID_ASPECT

log = logging.getLogger(__name__)

ANNOTATIONS_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/"
    "annotations.zip"
)
IMAGES_512_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagridv2_512.zip"
SPLITS = ("train", "val", "test")


def iter_annotation_files(archive: zipfile.ZipFile):
    for info in archive.infolist():
        parts = info.filename.split("/")
        if len(parts) == 3 and parts[1] in SPLITS and parts[2].endswith(".json") and not info.is_dir():
            yield parts[1], parts[2][:-5], info


def convert(zip_path: Path, out_file: Path) -> dict:
    points, labels, splits, users, images = [], [], [], [], []
    with zipfile.ZipFile(zip_path) as archive:
        records = list(iter_annotation_files(archive))
        classes = sorted({name for _, name, _ in records})
        user_ids: dict[str, int] = {}
        for split, name, info in records:
            data = json.loads(archive.read(info))
            for image_id, item in data.items():
                for label, hand in zip(item["labels"], item.get("hand_landmarks", []), strict=False):
                    if len(hand) != 21 or label not in classes:
                        continue  # some hands have no landmarks (automatic annotation)
                    points.append(hand)
                    labels.append(classes.index(label))
                    splits.append(SPLITS.index(split))
                    users.append(user_ids.setdefault(item["user_id"], len(user_ids)))
                    images.append(f"{name}/{image_id}")
            log.info("%s/%s: %d hands so far", split, name, len(labels))
    xy = np.asarray(points, dtype=np.float32)
    xy[..., 0] *= HAGRID_ASPECT
    np.savez_compressed(
        out_file,
        points=xy,
        labels=np.asarray(labels, np.int16),
        split=np.asarray(splits, np.int8),
        user=np.asarray(users, np.int32),
        image=np.asarray(images),
        classes=np.asarray(classes),
    )
    counts = np.bincount(labels, minlength=len(classes))
    return {
        "hands": len(labels),
        "classes": len(classes),
        "per_split": np.bincount(splits, minlength=3).tolist(),
        "smallest_class": classes[int(counts.argmin())],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/hagrid"))
    parser.add_argument("--zip", type=Path, help="already downloaded annotations.zip")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    zip_path = args.zip or args.out / "annotations.zip"
    if not zip_path.exists():
        log.info("downloading %s (719 MB)", ANNOTATIONS_URL)
        urllib.request.urlretrieve(ANNOTATIONS_URL, zip_path)
    print(json.dumps(convert(zip_path, args.out / "landmarks.npz"), indent=2))


if __name__ == "__main__":
    main()
