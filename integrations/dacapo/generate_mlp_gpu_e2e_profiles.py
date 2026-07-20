#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPERATOR_SPEC = (
    ROOT / "docs" / "operator-spec" / "v2" / "profiles" /
    "dacapo-heaan-gpu.v1.json"
)
DEFAULT_COMPILER_PROFILE = ROOT / "third_party" / "dacapo" / "profiled_HEAAN_GPU.json"
LOGICAL_LEVEL_UPPER = 4
PHYSICAL_LEVEL_LOWER = 4
LEVEL_COUNT = 17
BOOT_LATENCY_US = 10_000
BOOT_PROFILE_ID = "poseidon-gpu-host-boot-v1"


def make_operator_spec(base_path: Path) -> dict:
    spec = json.loads(base_path.read_text(encoding="utf-8"))
    if spec.get("spec_format_version") != 2:
        raise ValueError("base OperatorSpec must use format V2")
    expected_operators = {
        "add_cc", "add_cp", "sub_cc", "sub_cp", "mul_cc", "mul_cp",
        "negate", "rotate", "rescale", "mod_switch", "relinearize", "boot",
    }
    if set(spec.get("operators", {})) != expected_operators:
        raise ValueError("base OperatorSpec has an unexpected operator set")

    spec["spec_id"] = "poseidon-ckks-gpu-mlp-e2e-v1"
    spec["status"] = "validated"
    spec["target_id"] = "poseidon-ckks-gpu"
    spec["rescale_mode"] = "lazy"
    spec["noise_unit"] = None
    spec["provenance"] = {
        "kind": "test-fixture",
        "repository": "ckks-runtime",
        "revision": "local",
        "path": "integrations/dacapo/generate_mlp_gpu_e2e_profiles.py",
        "source_sha256": "sha256:" + "0" * 64,
    }
    spec["context"] = {
        "context_id": "poseidon-ckks-gpu-mlp-e2e",
        "poly_degree": 4096,
        "rns_moduli_log2": [30] * LEVEL_COUNT,
        "max_modulus_log2": 30,
        "default_scale_log2": 40,
    }
    spec["levels"] = {
        "lower_bound": PHYSICAL_LEVEL_LOWER,
        "upper_bound": LEVEL_COUNT - 1,
    }

    latency = list(range(1, LEVEL_COUNT + 1))
    for name in expected_operators - {"boot"}:
        entry = {
            "supported": True,
            "latency_us_by_level": latency,
            "noise_by_level": None,
        }
        if name == "rescale":
            entry["max_levels_per_op"] = 4
        spec["operators"][name] = entry
    spec["operators"]["boot"] = {"supported": True}
    spec["boot_profiles"] = [{
        "profile_id": BOOT_PROFILE_ID,
        "implementation": "decrypt_reencrypt",
        "input_level_min": PHYSICAL_LEVEL_LOWER,
        "input_level_max": LEVEL_COUNT - 1,
        "input_components": 2,
        "output_level": LEVEL_COUNT - 1,
        "output_scale_log2": 120,
        "output_components": 2,
        "latency_us_by_input_level": [BOOT_LATENCY_US] * LEVEL_COUNT,
        "noise_by_input_level": None,
        "host_requirements": {
            "needs_secret_key": True,
            "needs_host_compute": True,
        },
    }]
    return spec


def make_compiler_profile(base_path: Path) -> dict:
    profile = json.loads(base_path.read_text(encoding="utf-8"))
    for key in (
        "polynomialDegree", "rescalingFactor", "levelUpperBound",
        "levelLowerBound", "bootstrapLevelUpperBound",
        "bootstrapLevelLowerBound", "latencyTable",
    ):
        if key not in profile:
            raise ValueError(f"base compiler profile is missing {key}")
    profile["polynomialDegree"] = 4096
    profile["rescalingFactor"] = 120
    profile["levelUpperBound"] = LOGICAL_LEVEL_UPPER
    profile["levelLowerBound"] = 1
    profile["bootstrapLevelUpperBound"] = LOGICAL_LEVEL_UPPER
    profile["bootstrapLevelLowerBound"] = 1
    boot_latency = profile["latencyTable"].get("earth.bootstrap_single")
    if not isinstance(boot_latency, list) or not boot_latency:
        raise ValueError("base compiler profile has no Boot latency table")
    profile["latencyTable"]["earth.bootstrap_single"] = [BOOT_LATENCY_US] * len(
        boot_latency
    )
    return profile


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DaCapo/Poseidon GPU profiles for the MLP E2E test"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-operator-spec", type=Path,
                        default=DEFAULT_OPERATOR_SPEC)
    parser.add_argument("--base-compiler-profile", type=Path,
                        default=DEFAULT_COMPILER_PROFILE)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    operator_spec = output_dir / "operator-spec.json"
    compiler_profile = output_dir / "compiler-profile.json"
    write_json(operator_spec, make_operator_spec(args.base_operator_spec.resolve()))
    write_json(compiler_profile,
               make_compiler_profile(args.base_compiler_profile.resolve()))
    print(f"OperatorSpec: {operator_spec}")
    print(f"compiler profile: {compiler_profile}")


if __name__ == "__main__":
    main()
