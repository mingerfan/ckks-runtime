#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


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


def read_operator_spec(path: Path) -> tuple[dict, str, Path]:
    content = path.read_bytes()
    spec = json.loads(content)
    if not isinstance(spec, dict):
        raise ValueError("OperatorSpec root must be an object")
    if spec.get("spec_format_version") != 2:
        raise ValueError("Dacapo artifact generation requires OperatorSpec V2")
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

    provenance = spec.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("kind") != "dacapo-profile-json":
        raise ValueError("OperatorSpec provenance must reference a Dacapo profile JSON")
    source_path = require_string(provenance.get("path"), "$.provenance.path")
    source_digest = require_string(
        provenance.get("source_sha256"), "$.provenance.source_sha256"
    )
    compiler_profile = (DACAPO / source_path).resolve()
    if compiler_profile.parent != DACAPO.resolve():
        raise ValueError("$.provenance.path must name a file in the Dacapo root")
    if sha256(compiler_profile.read_bytes()) != source_digest:
        raise ValueError("Dacapo compiler profile does not match OperatorSpec provenance")

    boot_profiles = spec.get("boot_profiles")
    if not isinstance(boot_profiles, list) or len(boot_profiles) != 1:
        raise ValueError("Dacapo artifact generation requires exactly one boot profile")
    boot = boot_profiles[0]
    if not isinstance(boot, dict):
        raise ValueError("$.boot_profiles[0] must be an object")
    require_string(boot.get("profile_id"), "$.boot_profiles[0].profile_id")
    implementation = require_string(
        boot.get("implementation"), "$.boot_profiles[0].implementation"
    )
    if implementation not in {"native", "decrypt_reencrypt"}:
        raise ValueError("unsupported boot implementation")

    return spec, sha256(content), compiler_profile


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
                            output_dir: Path) -> list[str]:
    boot = spec["boot_profiles"][0]
    optimized = output_dir / f"{args.model}.optimized.mlir"
    return [
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
        "--runtime-plan-device-count=0",
        "--runtime-plan-ntt=true",
        f"--runtime-plan-boot-profile={boot['profile_id']}",
        f"--runtime-plan-boot-implementation={boot['implementation']}",
        f"--runtime-plan-inline-payload-max-bytes={args.inline_payload_max_bytes}",
        str(traced),
        "-o",
        str(optimized),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Dacapo MLIR and RuntimePlan review artifacts"
    )
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--operator-spec", type=Path, required=True)
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

    args.operator_spec = args.operator_spec.resolve()
    args.hecate_build = args.hecate_build.resolve()
    args.hecate_opt = args.hecate_opt.resolve()
    args.python = args.python.resolve()
    output_dir = (args.output_dir or
                  (DACAPO / "review_artifacts" / args.model)).resolve()
    spec, spec_digest, compiler_profile = read_operator_spec(args.operator_spec)

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
        args, traced, spec, spec_digest, compiler_profile, output_dir
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
