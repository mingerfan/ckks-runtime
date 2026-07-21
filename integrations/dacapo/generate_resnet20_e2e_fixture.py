#!/usr/bin/env python3

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    ROOT / "third_party" / "dacapo" / "examples" / "data" /
    "resnet20.silu.model"
)


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path: Path) -> torch.nn.Module:
    from poly.models.ResNet import resnet20

    model = torch.nn.DataParallel(resnet20())
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().double()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible ResNet-20 input and PyTorch result"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"ResNet-20 model not found: {model_path}")

    generator = random.Random(args.seed)
    values = [generator.random() for _ in range(3 * 32 * 32)]
    input_tensor = torch.tensor(values, dtype=torch.double).reshape(1, 3, 32, 32)
    model = load_model(model_path)
    with torch.no_grad():
        # The Dacapo packing keeps every activation at 1/32 of the clear value.
        output = (model(input_tensor).reshape(-1) / 32.0).tolist()

    fixture = {
        "format_version": 1,
        "model": "resnet20",
        "seed": args.seed,
        "distribution": "uniform_[0,1)",
        "model_sha256": sha256(model_path),
        "input_shape": [1, 3, 32, 32],
        "input": values,
        "python_output_scale": 32.0,
        "python_output": output,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"fixture: {output_path}")


if __name__ == "__main__":
    main()
