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
DEFAULT_LEVEL_COUNT = 17
BOOT_LATENCY_US = 10_000
BOOT_PROFILE_ID = "poseidon-gpu-host-boot-v1"
PROFILE_PRESETS = ("synthetic", "poseidon-8192", "poseidon-65536")

POSEIDON_8192_LEVEL_COUNT = 17
POSEIDON_8192_POLY_DEGREE = 8192
POSEIDON_8192_GPU_WALL_US_AT_L17 = {
    # RTX 4060 Laptop GPU; median of three Release runs with 20 warmups and
    # 100 timed iterations at degree=8192, Q=17, P=2. Values are rounded to
    # the integer-microsecond resolution required by OperatorSpec V2.
    "add_cc": 7,
    "add_cp": 7,
    "sub_cc": 8,
    "sub_cp": 7,
    # Derived from 675.505 us combined - 514.775 us Relin - 142.568 us Rescale.
    "mul_cc": 18,
    "mul_cp": 9,
    "negate": 8,
    "rotate": 517,
    "rescale": 143,
    # No standalone ModSwitch benchmark is available; use measured HYBRID
    # BConv/ModUp (87.895 us) as a conservative proxy.
    "mod_switch": 88,
    "relinearize": 515,
}
POSEIDON_8192_COMBINED_MUL_RELIN_RESCALE_US_AT_L17 = 676

POSEIDON_65536_LEVEL_COUNT = 40
POSEIDON_65536_POLY_DEGREE = 65536
POSEIDON_GPU_LOW_LEVEL_FACTOR = 0.80
POSEIDON_65536_GPU_WALL_US_AT_L40 = {
    "add_cc": 6,
    "add_cp": 7,
    "sub_cc": 8,
    "sub_cp": 7,
    # Raw MulCC is derived from 500 us combined - 370 us Relin - 110 us Rescale.
    "mul_cc": 20,
    "mul_cp": 8,
    "negate": 8,
    "rotate": 380,
    "rescale": 110,
    # No ModSwitch result was supplied; use keyswitch_bconv_modup as a proxy.
    "mod_switch": 40,
    "relinearize": 370,
}
POSEIDON_65536_COMBINED_MUL_RELIN_RESCALE_US_AT_L40 = 500

# count=32, degree=65536, components=2. The operational point uses the
# end-to-end object loop rather than the contiguous-buffer microbenchmark.
POSEIDON_65536_INTRA_RANK_RATE_POINTS = (
    (16, 46.8),
    (80, 153.6),
    (160, 214.0),
    (240, 252.2),
    (320, 276.3),
    (400, 293.8),
    (480, 305.4),
    (560, 315.8),
    (640, 323.6),
)
POSEIDON_65536_CONTIGUOUS_RATE_LIMIT_GBPS = 395.7


def level_count(profile_preset: str) -> int:
    if profile_preset == "poseidon-8192":
        return POSEIDON_8192_LEVEL_COUNT
    if profile_preset == "poseidon-65536":
        return POSEIDON_65536_LEVEL_COUNT
    return DEFAULT_LEVEL_COUNT


def mildly_scaled_latency(highest_level_us: int, count: int) -> list[int]:
    if count < 2:
        return [highest_level_us]
    result = []
    for index in range(count):
        position = index / (count - 1)
        factor = POSEIDON_GPU_LOW_LEVEL_FACTOR + (
            1.0 - POSEIDON_GPU_LOW_LEVEL_FACTOR
        ) * position ** 0.7
        result.append(max(1, round(highest_level_us * factor)))
    result[-1] = highest_level_us
    return result


def operator_latencies(profile_preset: str, count: int) -> dict[str, list[int]]:
    if profile_preset == "poseidon-8192":
        return {
            name: mildly_scaled_latency(latency, count)
            for name, latency in POSEIDON_8192_GPU_WALL_US_AT_L17.items()
        }
    if profile_preset == "poseidon-65536":
        return {
            name: mildly_scaled_latency(latency, count)
            for name, latency in POSEIDON_65536_GPU_WALL_US_AT_L40.items()
        }
    latency = list(range(1, count + 1))
    return {
        name: latency
        for name in (
            "add_cc", "add_cp", "sub_cc", "sub_cp", "mul_cc", "mul_cp",
            "negate", "rotate", "rescale", "mod_switch", "relinearize",
        )
    }


def compiler_latencies(profile_preset: str,
                       count: int) -> dict[str, list[int]]:
    if profile_preset == "poseidon-8192":
        combined_latency = (
            POSEIDON_8192_COMBINED_MUL_RELIN_RESCALE_US_AT_L17
        )
    elif profile_preset == "poseidon-65536":
        combined_latency = (
            POSEIDON_65536_COMBINED_MUL_RELIN_RESCALE_US_AT_L40
        )
    else:
        return {}
    physical = operator_latencies(profile_preset, count)
    combined_mul = mildly_scaled_latency(
        combined_latency,
        count,
    )
    physical_levels = [
        count - 1 - (LOGICAL_LEVEL_UPPER - logical_level) * 4
        for logical_level in range(LOGICAL_LEVEL_UPPER + 1)
    ]
    if min(physical_levels) < 0:
        raise ValueError("physical modulus chain is too short for compiler levels")

    def select(table: list[int]) -> list[int]:
        return [table[index] for index in physical_levels]

    return {
        "earth.add_double": select(physical["add_cc"]),
        "earth.add_single": select(physical["add_cp"]),
        "earth.bootstrap_single": [BOOT_LATENCY_US] * len(physical_levels),
        "earth.modswitch_single": select(physical["mod_switch"]),
        "earth.mul_double": select(combined_mul),
        "earth.mul_single": select(physical["mul_cp"]),
        "earth.negate_single": select(physical["negate"]),
        "earth.rescale_single": select(physical["rescale"]),
        "earth.rotate_single": select(physical["rotate"]),
    }


def make_operator_spec(base_path: Path, poly_degree: int,
                       profile_preset: str) -> dict:
    spec = json.loads(base_path.read_text(encoding="utf-8"))
    if spec.get("spec_format_version") != 2:
        raise ValueError("base OperatorSpec must use format V2")
    expected_operators = {
        "add_cc", "add_cp", "sub_cc", "sub_cp", "mul_cc", "mul_cp",
        "negate", "rotate", "rescale", "mod_switch", "relinearize", "boot",
    }
    if set(spec.get("operators", {})) != expected_operators:
        raise ValueError("base OperatorSpec has an unexpected operator set")

    count = level_count(profile_preset)
    profile_ids = {
        "poseidon-8192": "poseidon-ckks-gpu-8192-calibrated-v1",
        "poseidon-65536": "poseidon-ckks-gpu-65536-estimated-v1",
    }
    context_ids = {
        "poseidon-8192": "poseidon-ckks-gpu-8192-calibrated",
        "poseidon-65536": "poseidon-ckks-gpu-65536-estimated",
    }
    spec["spec_id"] = profile_ids.get(
        profile_preset, "poseidon-ckks-gpu-mlp-e2e-v1"
    )
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
        "context_id": context_ids.get(
            profile_preset, "poseidon-ckks-gpu-mlp-e2e"
        ),
        "poly_degree": poly_degree,
        "rns_moduli_log2": [30] * count,
        "max_modulus_log2": 30,
        "default_scale_log2": 40,
    }
    spec["levels"] = {
        "lower_bound": PHYSICAL_LEVEL_LOWER,
        "upper_bound": count - 1,
    }

    latencies = operator_latencies(profile_preset, count)
    for name in expected_operators - {"boot"}:
        entry = {
            "supported": True,
            "latency_us_by_level": latencies[name],
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
        "input_level_max": count - 1,
        "input_components": 2,
        "output_level": count - 1,
        "output_scale_log2": 120,
        "output_components": 2,
        "latency_us_by_input_level": [BOOT_LATENCY_US] * count,
        "noise_by_input_level": None,
        "host_requirements": {
            "needs_secret_key": True,
            "needs_host_compute": True,
        },
    }]
    return spec


def make_compiler_profile(base_path: Path, poly_degree: int,
                          profile_preset: str) -> dict:
    profile = json.loads(base_path.read_text(encoding="utf-8"))
    for key in (
        "polynomialDegree", "rescalingFactor", "levelUpperBound",
        "levelLowerBound", "bootstrapLevelUpperBound",
        "bootstrapLevelLowerBound", "latencyTable",
    ):
        if key not in profile:
            raise ValueError(f"base compiler profile is missing {key}")
    profile["polynomialDegree"] = poly_degree
    profile["rescalingFactor"] = 120
    profile["levelUpperBound"] = LOGICAL_LEVEL_UPPER
    profile["levelLowerBound"] = 1
    profile["bootstrapLevelUpperBound"] = LOGICAL_LEVEL_UPPER
    profile["bootstrapLevelLowerBound"] = 1
    boot_latency = profile["latencyTable"].get("earth.bootstrap_single")
    if not isinstance(boot_latency, list) or not boot_latency:
        raise ValueError("base compiler profile has no Boot latency table")
    measured_latencies = compiler_latencies(
        profile_preset, level_count(profile_preset)
    )
    if measured_latencies:
        profile["latencyTable"].update(measured_latencies)
    else:
        profile["latencyTable"]["earth.bootstrap_single"] = [
            BOOT_LATENCY_US
        ] * len(boot_latency)
    return profile


def make_communication_profile(profile_preset: str) -> dict:
    profile = {
        "format_version": 1,
        "coefficient_bytes": 4,
        "links": {
            "host_device": {
                "startup_latency_us": 8,
                "max_rate_bytes_per_us": 12000,
                "saturation_bytes": 1048576,
            },
            "intra_rank": {
                "startup_latency_us": 5,
                "max_rate_bytes_per_us": 16000,
                "saturation_bytes": 524288,
            },
            "inter_rank": {
                "startup_latency_us": 20,
                "max_rate_bytes_per_us": 3000,
                "saturation_bytes": 4194304,
            },
        },
    }
    if profile_preset == "poseidon-65536":
        intra_rank = profile["links"]["intra_rank"]
        intra_rank["max_rate_bytes_per_us"] = round(
            POSEIDON_65536_CONTIGUOUS_RATE_LIMIT_GBPS * 1000
        )
        intra_rank["saturation_bytes"] = 16 * 1024 * 1024
        intra_rank["rate_points"] = [
            {
                "payload_bytes": payload_mib * 1024 * 1024,
                "rate_bytes_per_us": round(rate_gbps * 1000),
            }
            for payload_mib, rate_gbps in POSEIDON_65536_INTRA_RANK_RATE_POINTS
        ]
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
    parser.add_argument("--poly-degree", type=int, default=4096)
    parser.add_argument(
        "--profile-preset", choices=PROFILE_PRESETS, default="synthetic",
        help="Use measured/estimated Poseidon data instead of synthetic fixture costs",
    )
    args = parser.parse_args()

    if args.poly_degree < 2 or args.poly_degree & (args.poly_degree - 1):
        raise ValueError("--poly-degree must be a power of two")
    if (args.profile_preset == "poseidon-65536" and
            args.poly_degree != POSEIDON_65536_POLY_DEGREE):
        raise ValueError("poseidon-65536 requires --poly-degree 65536")
    if (args.profile_preset == "poseidon-8192" and
            args.poly_degree != POSEIDON_8192_POLY_DEGREE):
        raise ValueError("poseidon-8192 requires --poly-degree 8192")

    output_dir = args.output_dir.resolve()
    operator_spec = output_dir / "operator-spec.json"
    compiler_profile = output_dir / "compiler-profile.json"
    communication_profile = output_dir / "communication-profile.json"
    write_json(
        operator_spec,
        make_operator_spec(
            args.base_operator_spec.resolve(), args.poly_degree,
            args.profile_preset,
        ),
    )
    write_json(
        compiler_profile,
        make_compiler_profile(
            args.base_compiler_profile.resolve(), args.poly_degree,
            args.profile_preset,
        ),
    )
    write_json(
        communication_profile,
        make_communication_profile(args.profile_preset),
    )
    print(f"OperatorSpec: {operator_spec}")
    print(f"compiler profile: {compiler_profile}")
    print(f"communication profile: {communication_profile}")


if __name__ == "__main__":
    main()
