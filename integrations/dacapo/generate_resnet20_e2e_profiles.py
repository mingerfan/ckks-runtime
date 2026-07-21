#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import generate_mlp_gpu_e2e_profiles as gpu_profiles


ROOT = Path(__file__).resolve().parents[2]
DACAPO = ROOT / "third_party" / "dacapo"
CPU_BASE_SPEC = (
    ROOT / "docs" / "operator-spec" / "v2" / "profiles" /
    "dacapo-heaan-cpu.v1.json"
)
CPU_BASE_COMPILER_PROFILE = DACAPO / "profiled_HEAAN_CPU.json"
GPU_BASE_SPEC = (
    ROOT / "docs" / "operator-spec" / "v2" / "profiles" /
    "dacapo-heaan-gpu.v1.json"
)
GPU_BASE_COMPILER_PROFILE = DACAPO / "profiled_HEAAN_GPU.json"

POLY_DEGREE = 32768
LEVEL_COUNT = 17
LEVEL_UPPER = LEVEL_COUNT - 1
BOOT_LATENCY_US = 30_000


def truncate_latency(value: object, count: int) -> list[int]:
    if not isinstance(value, list) or len(value) < count:
        raise ValueError("latency table is shorter than the requested level count")
    return [int(entry) for entry in value[:count]]


def make_cpu_operator_spec(base_path: Path) -> dict:
    spec = json.loads(base_path.read_text(encoding="utf-8"))
    spec["spec_id"] = "poseidon-ckks-cpu-resnet20-32768-v1"
    spec["status"] = "validated"
    spec["target_id"] = "poseidon-ckks-cpu"
    spec["rescale_mode"] = "eager"
    spec["noise_unit"] = None
    spec["provenance"] = {
        "kind": "test-fixture",
        "repository": "ckks-runtime",
        "revision": "local",
        "path": "integrations/dacapo/generate_resnet20_e2e_profiles.py",
        "source_sha256": "sha256:" + "0" * 64,
    }
    spec["context"] = {
        "context_id": "poseidon-ckks-cpu-resnet20-32768",
        "poly_degree": POLY_DEGREE,
        "rns_moduli_log2": [51] * LEVEL_COUNT,
        "max_modulus_log2": 51,
        "default_scale_log2": 40,
    }
    spec["levels"] = {"lower_bound": 1, "upper_bound": LEVEL_UPPER}
    for name, operator in spec["operators"].items():
        if name == "boot":
            spec["operators"][name] = {"supported": True}
            continue
        if operator["supported"] is not True:
            continue
        operator["latency_us_by_level"] = truncate_latency(
            operator["latency_us_by_level"], LEVEL_COUNT
        )
        operator["noise_by_level"] = None
        if name == "rescale":
            operator["max_levels_per_op"] = 1
    spec["boot_profiles"] = [{
        "profile_id": "poseidon-cpu-host-boot-resnet20-v1",
        "implementation": "decrypt_reencrypt",
        "input_level_min": 1,
        "input_level_max": LEVEL_UPPER,
        "input_components": 2,
        "output_level": LEVEL_UPPER,
        "output_scale_log2": 51,
        "output_components": 2,
        "latency_us_by_input_level": [BOOT_LATENCY_US] * LEVEL_COUNT,
        "noise_by_input_level": None,
        "host_requirements": {
            "needs_secret_key": True,
            "needs_host_compute": True,
        },
    }]
    return spec


def make_cpu_compiler_profile(base_path: Path) -> dict:
    profile = json.loads(base_path.read_text(encoding="utf-8"))
    profile["polynomialDegree"] = POLY_DEGREE
    profile["levelLowerBound"] = 1
    profile["levelUpperBound"] = LEVEL_UPPER
    profile["bootstrapLevelLowerBound"] = 3
    profile["bootstrapLevelUpperBound"] = LEVEL_UPPER
    for name, latency in profile["latencyTable"].items():
        profile["latencyTable"][name] = truncate_latency(latency, LEVEL_COUNT)
    return profile


def make_gpu_operator_spec(base_path: Path) -> dict:
    spec = gpu_profiles.make_operator_spec(
        base_path, POLY_DEGREE, "synthetic"
    )
    spec["spec_id"] = "poseidon-ckks-gpu-resnet20-32768-v1"
    spec["context"] = {
        "context_id": "poseidon-ckks-gpu-resnet20-32768",
        "poly_degree": POLY_DEGREE,
        "rns_moduli_log2": [30] * LEVEL_COUNT,
        "max_modulus_log2": 30,
        "default_scale_log2": 20,
    }
    spec["levels"] = {"lower_bound": 1, "upper_bound": LEVEL_UPPER}
    latencies = gpu_profiles.operator_latencies(
        "poseidon-65536", LEVEL_COUNT
    )
    for name, latency in latencies.items():
        spec["operators"][name]["latency_us_by_level"] = latency
    spec["boot_profiles"][0]["input_level_min"] = 1
    spec["boot_profiles"][0]["output_scale_log2"] = 30
    spec["provenance"]["path"] = (
        "integrations/dacapo/generate_resnet20_e2e_profiles.py"
    )
    return spec


def make_gpu_compiler_profile(base_path: Path) -> dict:
    profile = json.loads(base_path.read_text(encoding="utf-8"))
    profile["polynomialDegree"] = POLY_DEGREE
    profile["rescalingFactor"] = 30
    profile["levelLowerBound"] = 1
    profile["levelUpperBound"] = LEVEL_UPPER
    profile["bootstrapLevelLowerBound"] = 3
    profile["bootstrapLevelUpperBound"] = LEVEL_UPPER
    physical = gpu_profiles.operator_latencies(
        "poseidon-65536", LEVEL_COUNT
    )
    profile["latencyTable"].update({
        "earth.add_double": physical["add_cc"],
        "earth.add_single": physical["add_cp"],
        "earth.bootstrap_single": [
            gpu_profiles.BOOT_LATENCY_US
        ] * LEVEL_COUNT,
        "earth.modswitch_single": physical["mod_switch"],
        "earth.mul_double": gpu_profiles.mildly_scaled_latency(
            gpu_profiles.POSEIDON_65536_COMBINED_MUL_RELIN_RESCALE_US_AT_L40,
            LEVEL_COUNT,
        ),
        "earth.mul_single": physical["mul_cp"],
        "earth.negate_single": physical["negate"],
        "earth.rescale_single": physical["rescale"],
        "earth.rotate_single": physical["rotate"],
    })
    for name, latency in profile["latencyTable"].items():
        profile["latencyTable"][name] = truncate_latency(latency, LEVEL_COUNT)
    return profile


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate degree-32768 Poseidon profiles for ResNet-20"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    write_json(
        output_dir / "cpu" / "operator-spec.json",
        make_cpu_operator_spec(CPU_BASE_SPEC),
    )
    write_json(
        output_dir / "cpu" / "compiler-profile.json",
        make_cpu_compiler_profile(CPU_BASE_COMPILER_PROFILE),
    )
    write_json(
        output_dir / "gpu" / "operator-spec.json",
        make_gpu_operator_spec(GPU_BASE_SPEC),
    )
    write_json(
        output_dir / "gpu" / "compiler-profile.json",
        make_gpu_compiler_profile(GPU_BASE_COMPILER_PROFILE),
    )
    write_json(
        output_dir / "gpu" / "communication-profile.json",
        gpu_profiles.make_communication_profile("synthetic"),
    )
    print(f"ResNet-20 profiles: {output_dir}")


if __name__ == "__main__":
    main()
