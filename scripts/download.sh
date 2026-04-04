#!/bin/bash
# Run on a login node (needs internet): MediaPipe hand model, HaGRID annotations with landmarks
# (719 MB), pretrained YOLO weights and fonts, and the image subset for the detector comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${DATA:-data}"
mkdir -p models "$DATA/hagrid"
URL=https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
[[ -f models/hand_landmarker.task ]] || curl -fL -o models/hand_landmarker.task "$URL"
python -m hand_gestures.data.hagrid --out "$DATA/hagrid"                     # download + landmarks.npz
python - <<'PY'
from ultralytics import YOLO
from ultralytics.utils.checks import check_font
YOLO("yolo26s.pt"); check_font("Arial.ttf")
PY
# ~25k photos (~3 GB) fetched from the 128 GB archive with range requests
python -m hand_gestures.data.hagrid_images --annotations "$DATA/hagrid/annotations.zip" --out "$DATA/hagrid_yolo" \
    --train-per-class 600 --val-per-class 60 --test-per-class 120
