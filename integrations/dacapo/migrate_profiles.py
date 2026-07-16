#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DACAPO = ROOT / "third_party" / "dacapo"
OUTPUT = ROOT / "docs" / "operator-spec" / "v2" / "profiles"
DACAPO_REVISION = "a2c9ce41a57062cdf77ea4bac5b02be747109448"
DACAPO_REPOSITORY = "git@github.com:mingerfan/dacapo-modified.git"
RELINEARIZE_PLACEHOLDER_LATENCY_US = 1


@dataclass(frozen=True)
class ProfileConfig:
    source_name: str
    output_name: str
    spec_id: str
    target_id: str
    context_id: str
    runtime: str
    boot_implementation: str


PROFILES = (
    ProfileConfig(
        "profiled_HEAAN_CPU.json",
        "dacapo-heaan-cpu.v1.json",
        "dacapo-heaan-cpu-v1",
        "dacapo-heaan-cpu",
        "dacapo-heaan-fva",
        "HEAAN-HEVM",
        "native",
    ),
    ProfileConfig(
        "profiled_HEAAN_GPU.json",
        "dacapo-heaan-gpu.v1.json",
        "dacapo-heaan-gpu-v1",
        "dacapo-heaan-gpu",
        "dacapo-heaan-fva",
        "HEAAN-HEVM",
        "native",
    ),
    ProfileConfig(
        "profiled_SEAL_CPU.json",
        "dacapo-seal-cpu.v1.json",
        "dacapo-seal-cpu-v1",
        "dacapo-seal-cpu",
        "dacapo-seal-n15-l14",
        "SEAL-HEVM",
        "decrypt_reencrypt",
    ),
)


OPERATOR_KEYS = {
    "add_cc": "earth.add_double",
    "add_cp": "earth.add_single",
    "sub_cc": None,
    "sub_cp": None,
    "mul_cc": "earth.mul_double",
    "mul_cp": "earth.mul_single",
    "negate": "earth.negate_single",
    "rotate": "earth.rotate_single",
    "rescale": "earth.rescale_single",
    "mod_switch": "earth.modswitch_single",
    "relinearize": None,
}


def read_source_bytes(path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(DACAPO), "show", f"{DACAPO_REVISION}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def source_sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def require_integer(value, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a nonnegative integer")
    return value


def require_number(value, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{path} must be a nonnegative number")
    return value


def normalize_table(profile: dict, table_name: str, key: str | None, length: int):
    if key is None:
        return None
    table = profile.get(table_name, {})
    if key not in table:
        return None
    raw = table[key]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{table_name}.{key} must be a non-empty array")
    reader = require_integer if table_name == "latencyTable" else require_number
    values = [0]
    values.extend(reader(value, f"{table_name}.{key}[{index}]") for index, value in enumerate(raw))
    if len(values) < length:
        values.extend([values[-1]] * (length - len(values)))
    return values[:length]


def convert_boot_levels(profile: dict) -> tuple[int, int, int]:
    lower = require_integer(profile["bootstrapLevelLowerBound"], "bootstrapLevelLowerBound")
    upper = require_integer(profile["bootstrapLevelUpperBound"], "bootstrapLevelUpperBound")
    if lower > upper:
        raise ValueError("bootstrap level range is empty")

    init_level = upper
    earth_input_min = 0
    earth_input_max = upper - lower
    runtime_inputs = sorted((init_level - earth_input_min, init_level - earth_input_max))
    runtime_output = init_level
    return runtime_inputs[0], runtime_inputs[1], runtime_output


def operator_entry(profile: dict, key: str | None, level_count: int, rescale: bool = False) -> dict:
    latency = normalize_table(profile, "latencyTable", key, level_count)
    noise = normalize_table(profile, "noiseTable", key, level_count)
    result = {
        "supported": latency is not None or noise is not None,
        "latency_us_by_level": latency,
        "noise_by_level": noise,
    }
    if rescale:
        result["max_levels_per_op"] = 1
    return result


def migrate(config: ProfileConfig) -> dict:
    source_content = read_source_bytes(config.source_name)
    profile = json.loads(source_content.decode("utf-8"))
    required = {
        "runtime",
        "rescalingFactor",
        "polynomialDegree",
        "levelLowerBound",
        "levelUpperBound",
        "bootstrapLevelLowerBound",
        "bootstrapLevelUpperBound",
        "latencyTable",
    }
    missing = sorted(required - profile.keys())
    if missing:
        raise ValueError(f"{config.source_name} is missing fields: {', '.join(missing)}")
    if profile["runtime"] != config.runtime:
        raise ValueError(f"{config.source_name} has unexpected runtime {profile['runtime']!r}")

    level_lower = require_integer(profile["levelLowerBound"], "levelLowerBound")
    level_upper = require_integer(profile["levelUpperBound"], "levelUpperBound")
    if level_lower > level_upper:
        raise ValueError("level range is empty")
    level_count = level_upper + 1
    modulus_bits = require_integer(profile["rescalingFactor"], "rescalingFactor")
    poly_degree = require_integer(profile["polynomialDegree"], "polynomialDegree")
    if poly_degree < 2 or poly_degree & (poly_degree - 1):
        raise ValueError("polynomialDegree must be a power of two")

    operators = {}
    for name, key in OPERATOR_KEYS.items():
        operators[name] = operator_entry(profile, key, level_count, name == "rescale")
    operators["relinearize"] = {
        "supported": True,
        "latency_us_by_level": [
            0 if level < level_lower else RELINEARIZE_PLACEHOLDER_LATENCY_US
            for level in range(level_count)
        ],
        "noise_by_level": None,
    }
    operators["boot"] = {"supported": True}

    boot_input_min, boot_input_max, boot_output = convert_boot_levels(profile)
    if boot_input_min < level_lower or boot_input_max > level_upper or boot_output > level_upper:
        raise ValueError("converted bootstrap range is outside the CKKS level range")

    boot_key = "earth.bootstrap_single"
    boot_latency = normalize_table(profile, "latencyTable", boot_key, level_count)
    boot_noise = normalize_table(profile, "noiseTable", boot_key, level_count)
    has_noise = any(entry.get("noise_by_level") is not None for entry in operators.values())
    has_noise = has_noise or boot_noise is not None

    needs_host = config.boot_implementation == "decrypt_reencrypt"
    return {
        "spec_format_version": 2,
        "spec_id": config.spec_id,
        "version": 1,
        "status": "imported",
        "target_id": config.target_id,
        "rescale_mode": "eager",
        "noise_unit": "dacapo-legacy-estimator" if has_noise else None,
        "provenance": {
            "kind": "dacapo-profile-json",
            "repository": DACAPO_REPOSITORY,
            "revision": DACAPO_REVISION,
            "path": config.source_name,
            "source_sha256": source_sha256(source_content),
        },
        "context": {
            "context_id": config.context_id,
            "poly_degree": poly_degree,
            "rns_moduli_log2": [modulus_bits] * level_count,
            "max_modulus_log2": modulus_bits,
            "default_scale_log2": 40,
        },
        "levels": {"lower_bound": level_lower, "upper_bound": level_upper},
        "operators": operators,
        "boot_profiles": [
            {
                "profile_id": config.spec_id.removesuffix("-v1") + "-boot-v1",
                "implementation": config.boot_implementation,
                "input_level_min": boot_input_min,
                "input_level_max": boot_input_max,
                "input_components": 2,
                "output_level": boot_output,
                "output_scale_log2": 40,
                "output_components": 2,
                "latency_us_by_input_level": boot_latency,
                "noise_by_input_level": boot_noise,
                "host_requirements": {
                    "needs_secret_key": needs_host,
                    "needs_host_compute": needs_host,
                },
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    subprocess.run(
        ["git", "-C", str(DACAPO), "cat-file", "-e", f"{DACAPO_REVISION}^{{commit}}"],
        check=True,
    )

    for config in PROFILES:
        content = json.dumps(migrate(config), indent=2, ensure_ascii=False) + "\n"
        path = OUTPUT / config.output_name
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise RuntimeError(f"generated profile is stale: {path}")
        else:
            path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
