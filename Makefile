.PHONY: install dev lint format test jobs demo

install:
	pip install -e ".[train]"

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -q

demo:     ## webcam demo (needs GUI OpenCV: pip install opencv-python)
	hg-demo --static runs/static_mlp/best.pt --dynamic runs/dynamic_tcn/best.pt --source 0 --show

jobs:
	python3 slurm/generate.py --all
