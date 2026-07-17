#!/usr/bin/env python3

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "third_party" / "dacapo" / "examples" / "data" / "mlp.model"
DEFAULT_OPERATOR_SPEC = (
    ROOT / "docs" / "operator-spec" / "v1" / "profiles" /
    "poseidon-ckks-cpu.v1.json"
)


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_mlp(values: list[float], model: dict) -> list[float]:
    weight1 = model["linear1.weight"].detach().cpu().numpy()
    bias1 = model["linear1.bias"].detach().cpu().numpy()
    weight2 = model["linear2.weight"].detach().cpu().numpy()
    bias2 = model["linear2.bias"].detach().cpu().numpy()

    hidden = []
    for row in range(100):
        total = 0.0
        for column in range(784):
            total += values[column] * float(weight1[row][column])
        total += float(bias1[row])
        hidden.append(total * total)

    output = []
    for row in range(10):
        total = 0.0
        for column in range(100):
            total += hidden[column] * float(weight2[row][column])
        output.append(total + float(bias2[row]))
    return output


def make_e2e_operator_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["spec_id"] = "poseidon-ckks-cpu-seal-e2e-v1"
    spec["context"]["context_id"] = "poseidon-ckks-cpu-seal-e2e"
    spec["levels"]["lower_bound"] = 1
    profiles = [
        profile for profile in spec["boot_profiles"]
        if profile["profile_id"] == "poseidon-cpu-boot-emulation-v1"
    ]
    if len(profiles) != 1:
        raise ValueError("base OperatorSpec is missing the CPU Boot profile")
    profiles[0]["output_level"] = 13
    profiles[0]["output_scale_log2"] = 60
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one reproducible random MLP input and Python result"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--operator-spec-output", type=Path, required=True,
        help="Write the test-only Poseidon CPU OperatorSpec here",
    )
    parser.add_argument(
        "--base-operator-spec", type=Path, default=DEFAULT_OPERATOR_SPEC,
    )
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MLP model not found: {model_path}")
    generator = random.Random(args.seed)
    values = [generator.random() for _ in range(784)]
    model = torch.load(model_path, map_location=torch.device("cpu"))
    fixture = {
        "format_version": 1,
        "seed": args.seed,
        "distribution": "uniform_[0,1)",
        "model_sha256": sha256(model_path),
        "input": values,
        "python_output": evaluate_mlp(values, model),
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    operator_spec_path = args.operator_spec_output.resolve()
    operator_spec_path.parent.mkdir(parents=True, exist_ok=True)
    operator_spec_path.write_text(
        json.dumps(
            make_e2e_operator_spec(args.base_operator_spec.resolve()),
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"fixture: {output_path}")
    print(f"OperatorSpec: {operator_spec_path}")


if __name__ == "__main__":
    main()
