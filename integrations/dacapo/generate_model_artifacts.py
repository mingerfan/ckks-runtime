#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[2]
DACAPO = ROOT / "third_party" / "dacapo"
MODELS = {
    "mlp": "MLP",
    "resnet20": "ResNet",
}


def sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def require_string(value, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def require_positive_integer(value, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def parse_device_counts(value: Optional[str]) -> list[int]:
    if value is None:
        return []
    parts = value.split("x")
    if not parts or any(not part.isdecimal() for part in parts):
        raise ValueError(
            "--device-counts must contain positive integers separated by x"
        )
    counts = [int(part) for part in parts]
    if any(count < 1 or count > (1 << 31) - 1 for count in counts):
        raise ValueError("--device-counts entries must be positive int32 values")
    return counts


def read_operator_spec(path: Path) -> tuple[dict, str]:
    content = path.read_bytes()
    spec = json.loads(content)
    if not isinstance(spec, dict):
        raise ValueError("OperatorSpec root must be an object")
    if spec.get("spec_format_version") not in {1, 2}:
        raise ValueError("Dacapo artifact generation requires OperatorSpec V1 or V2")
    require_string(spec.get("spec_id"), "$.spec_id")
    require_positive_integer(spec.get("version"), "$.version")
    require_string(spec.get("target_id"), "$.target_id")

    context = spec.get("context")
    if not isinstance(context, dict):
        raise ValueError("$.context must be an object")
    require_string(context.get("context_id"), "$.context.context_id")

    operators = spec.get("operators")
    if not isinstance(operators, dict):
        raise ValueError("$.operators must be an object")
    relinearize = operators.get("relinearize")
    if not isinstance(relinearize, dict) or relinearize.get("supported") is not True:
        raise ValueError("OperatorSpec must support relinearize")

    return spec, sha256(content)


def compiler_profile_from_provenance(spec: dict) -> Path:
    provenance = spec.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("kind") != "dacapo-profile-json":
        raise ValueError(
            "--compiler-profile is required when OperatorSpec provenance "
            "does not reference a Dacapo profile JSON"
        )
    source_path = require_string(provenance.get("path"), "$.provenance.path")
    source_digest = require_string(
        provenance.get("source_sha256"), "$.provenance.source_sha256"
    )
    compiler_profile = (DACAPO / source_path).resolve()
    if compiler_profile.parent != DACAPO.resolve():
        raise ValueError("$.provenance.path must name a file in the Dacapo root")
    if sha256(compiler_profile.read_bytes()) != source_digest:
        raise ValueError("Dacapo compiler profile does not match OperatorSpec provenance")

    return compiler_profile


def select_boot_profile(spec: dict, profile_id: Optional[str]) -> dict:
    boot_profiles = spec.get("boot_profiles")
    if not isinstance(boot_profiles, list) or not boot_profiles:
        raise ValueError("OperatorSpec must contain a boot profile")
    if profile_id is None:
        if len(boot_profiles) != 1:
            raise ValueError(
                "--boot-profile is required when OperatorSpec has multiple boot profiles"
            )
        boot = boot_profiles[0]
    else:
        matches = [
            profile for profile in boot_profiles
            if isinstance(profile, dict) and profile.get("profile_id") == profile_id
        ]
        if len(matches) != 1:
            raise ValueError("--boot-profile does not name exactly one profile")
        boot = matches[0]
    if not isinstance(boot, dict):
        raise ValueError("boot profile must be an object")
    require_string(boot.get("profile_id"), "boot profile id")
    implementation = require_string(
        boot.get("implementation"), "boot profile implementation"
    )
    if implementation not in {"native", "decrypt_reencrypt"}:
        raise ValueError("unsupported boot implementation")
    return boot


def trace_model(model: str, output_dir: Path, python: Path,
                hecate_build: Path) -> Path:
    source_name = MODELS[model]
    benchmark = DACAPO / "examples" / "benchmarks" / f"{source_name}.py"
    frontend = hecate_build / "lib" / "libHecateFrontend.so"
    if not frontend.is_file():
        raise FileNotFoundError(f"Hecate frontend library not found: {frontend}")

    trace_dir = output_dir / "traced"
    trace_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HECATE"] = str(DACAPO)
    env["HECATE_BUILD"] = str(hecate_build)
    python_paths = [
        str(DACAPO / "python" / "hecate"),
        str(DACAPO / "python" / "poly"),
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    subprocess.run([str(python), str(benchmark)], cwd=output_dir,
                   env=env, check=True)
    generated = trace_dir / f"{source_name}.mlir"
    if not generated.is_file():
        raise FileNotFoundError(f"Python frontend did not generate {generated}")
    traced = output_dir / f"{model}.traced.earth.mlir"
    generated.replace(traced)
    trace_dir.rmdir()
    return traced


def build_optimizer_command(args, traced: Path, spec: dict,
                            spec_digest: str, compiler_profile: Path,
                            boot: dict,
                            output_dir: Path) -> list[str]:
    optimized = output_dir / f"{args.model}.optimized.mlir"
    command = [
        str(args.hecate_opt),
        "--dacapo",
        f"--ckks-config={compiler_profile}",
        f"--waterline={args.waterline}",
        "--enable-debug-printer",
        "--mlir-print-debuginfo",
        "--mlir-pretty-debuginfo",
        "--mlir-print-local-scope",
        "--mlir-disable-threading",
        f"--runtime-plan-id={args.plan_id}",
        f"--runtime-plan-target-id={spec['target_id']}",
        "--runtime-plan-capability-version=1",
        f"--runtime-plan-operator-spec-id={spec['spec_id']}",
        f"--runtime-plan-operator-spec-version={spec['version']}",
        f"--runtime-plan-operator-spec-sha256={spec_digest}",
        f"--runtime-plan-context-id={spec['context']['context_id']}",
        "--runtime-plan-ntt=true",
        f"--runtime-plan-boot-profile={boot['profile_id']}",
        f"--runtime-plan-boot-implementation={boot['implementation']}",
        f"--runtime-plan-inline-payload-max-bytes={args.inline_payload_max_bytes}",
    ]
    if args.device_counts:
        command.extend([
            f"--runtime-plan-device-counts={args.device_counts}",
            "--runtime-plan-operator-spec-path="
            f"{args.placement_operator_spec}",
            "--runtime-plan-intra-rank-communication-cost="
            f"{args.intra_rank_communication_cost}",
            "--runtime-plan-inter-rank-communication-cost="
            f"{args.inter_rank_communication_cost}",
        ])
    else:
        command.append("--runtime-plan-device-count=0")
    command.extend([str(traced), "-o", str(optimized)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Dacapo MLIR and RuntimePlan review artifacts"
    )
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--operator-spec", type=Path, required=True)
    parser.add_argument(
        "--compiler-profile", type=Path,
        help="Dacapo profile used for level and scale analysis; defaults to OperatorSpec provenance",
    )
    parser.add_argument(
        "--boot-profile",
        help="Boot profile id; required when OperatorSpec contains more than one profile",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--traced-mlir", type=Path,
                        help="Use an existing Earth MLIR file instead of running Python tracing")
    parser.add_argument("--hecate-build", type=Path,
                        default=DACAPO / "build" / "nix")
    parser.add_argument("--hecate-opt", type=Path,
                        default=DACAPO / "build" / "nix" / "bin" / "hecate-opt")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--waterline", type=int, default=40)
    parser.add_argument("--plan-id", type=int, default=1)
    parser.add_argument("--inline-payload-max-bytes", type=int, default=4096)
    parser.add_argument(
        "--device-counts",
        help="Enable placement with x-separated per-rank counts, e.g. 8 or 8x8",
    )
    parser.add_argument(
        "--placement-operator-spec", type=Path,
        help="OperatorSpec V2 used for placement latency; defaults to --operator-spec",
    )
    parser.add_argument(
        "--intra-rank-communication-cost", type=int, default=1000,
        help="Fixed point-to-point cost between devices in one rank",
    )
    parser.add_argument(
        "--inter-rank-communication-cost", type=int, default=10000,
        help="Fixed point-to-point cost between ranks",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.waterline < 0:
        raise ValueError("--waterline must be nonnegative")
    if args.plan_id < 0:
        raise ValueError("--plan-id must be nonnegative")
    if args.inline_payload_max_bytes < 0:
        raise ValueError("--inline-payload-max-bytes must be nonnegative")
    if args.inline_payload_max_bytes > (1 << 53) - 1:
        raise ValueError("--inline-payload-max-bytes exceeds the safe JSON integer range")
    parse_device_counts(args.device_counts)
    if args.intra_rank_communication_cost < 1:
        raise ValueError("--intra-rank-communication-cost must be positive")
    if args.inter_rank_communication_cost < 1:
        raise ValueError("--inter-rank-communication-cost must be positive")

    args.operator_spec = args.operator_spec.resolve()
    if args.compiler_profile:
        args.compiler_profile = args.compiler_profile.resolve()
    args.placement_operator_spec = (
        args.placement_operator_spec or args.operator_spec
    ).resolve()
    args.hecate_build = args.hecate_build.resolve()
    args.hecate_opt = args.hecate_opt.resolve()
    args.python = Path(os.path.abspath(args.python))
    output_dir = (args.output_dir or
                  (DACAPO / "review_artifacts" / args.model)).resolve()
    spec, spec_digest = read_operator_spec(args.operator_spec)
    compiler_profile = (
        args.compiler_profile or compiler_profile_from_provenance(spec)
    )
    if not compiler_profile.is_file():
        raise FileNotFoundError(f"Dacapo compiler profile not found: {compiler_profile}")
    boot = select_boot_profile(spec, args.boot_profile)
    if args.device_counts and spec["spec_format_version"] != 2:
        raise ValueError("placement requires OperatorSpec V2")

    if args.traced_mlir:
        traced = args.traced_mlir.resolve()
        if not args.dry_run and not traced.is_file():
            raise FileNotFoundError(f"traced MLIR not found: {traced}")
    elif args.dry_run:
        traced = output_dir / f"{args.model}.traced.earth.mlir"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        traced = trace_model(args.model, output_dir, args.python,
                             args.hecate_build)

    command = build_optimizer_command(
        args, traced, spec, spec_digest, compiler_profile, boot, output_dir
    )
    if args.dry_run:
        print(shlex.join(command))
        return

    if not args.hecate_opt.is_file():
        raise FileNotFoundError(f"hecate-opt not found: {args.hecate_opt}")
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=DACAPO, check=True)

    runtime_plans = sorted(output_dir.glob(
        f"{args.model}.optimized.*.runtime-plan.json"
    ))
    if not runtime_plans:
        raise FileNotFoundError("hecate-opt did not generate a RuntimePlan JSON file")
    print(f"traced MLIR: {traced}")
    print(f"optimized MLIR: {output_dir / f'{args.model}.optimized.mlir'}")
    for path in runtime_plans:
        print(f"RuntimePlan: {path}")
        plan = json.loads(path.read_text(encoding="utf-8"))
        if "plaintext_bundle" in plan:
            bundle = Path(str(path).removesuffix(".runtime-plan.json") + ".bundle")
            if not bundle.is_dir():
                raise FileNotFoundError(f"RuntimePlan bundle directory not found: {bundle}")
            print(f"plaintext bundle: {bundle}")


if __name__ == "__main__":
    main()
