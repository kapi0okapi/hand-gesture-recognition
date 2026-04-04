"""Export the static and dynamic gesture models to ONNX and check them against PyTorch.

hg-export --static runs/static_mlp/best.pt --dynamic runs/dynamic_tcn/best.pt --out models/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hand_gestures.engine import DynamicClassifier, StaticClassifier
from hand_gestures.synth import FRAME_DIM


def export(model: torch.nn.Module, example: torch.Tensor, path: Path) -> float:
    model.eval()
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        opset_version=18,
    )
    import onnxruntime as ort

    x = np.random.default_rng(0).standard_normal((3, *example.shape[1:])).astype(np.float32)
    onnx_out = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()
    return float(np.abs(onnx_out - torch_out).max())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    static = StaticClassifier(args.static)
    shape = (2, 67) if static.model_name == "mlp" else (2, 21, 2)
    diff = export(static.model, torch.randn(*shape), args.out / "static_gesture.onnx")
    (args.out / "static_gesture.json").write_text(
        json.dumps({"classes": static.classes, "input": static.model_name}, indent=1)
    )
    print(f"static_gesture.onnx: max |onnx - torch| = {diff:.2e}")
    if args.dynamic:
        dynamic = DynamicClassifier(args.dynamic)
        diff = export(
            dynamic.model, torch.randn(2, dynamic.length, FRAME_DIM), args.out / "dynamic_gesture.onnx"
        )
        print(f"dynamic_gesture.onnx: max |onnx - torch| = {diff:.2e}")


if __name__ == "__main__":
    main()
