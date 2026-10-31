#!/usr/bin/env python3
"""Check native 25Q alignment and distinct initial/Boot output prefixes."""

import argparse
import json
from pathlib import Path
import runpy
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runpy.run_path(
        str(root / "integrations/dacapo/generate_resnet20_manual_profiles.py")
    )["generate"](out)
    compiler = str(args.hecate_opt.resolve())
    profile = "poseidon-resnet20-manual-q50-scale40"
    common = [
        compiler,
        f"--ckks-config={out}/compiler.json",
        f"--runtime-plan-physical-level-operator-spec-path={out}/operator-spec.json",
        f"--runtime-plan-boot-profile={profile}",
    ]
    raw = "tensor<1x!earth.ci<0 * 0>>"
    checks = 0
    for name, level, scale, drops, anchor, expected_boots in [
        ("all-25-q", 25, 40, 0, 30, 1),
        ("one-q-alignment", 24, 41, 1, 30, 1),
        ("two-q-alignment", 23, 71, 2, 30, 1),
        ("no-q-scale41", 25, 41, None, 30, 0),
        ("one-q-scale71", 24, 71, None, 30, 0),
        ("two-q-scale179", 23, 179, None, 30, 0),
        ("initial-q34", 0, 40, 0, 33, 0),
        ("shifted-anchor-25-q", 28, 40, 0, 33, 1),
    ]:
        source = out / (name + ".mlir")
        source.write_text(
            f"module {{ func.func @budget(%x: {raw}) -> {raw} attributes {{"
            f"segment_inputType = [tensor<1x!earth.ci<{scale} * {level}>>], "
            f"segment_input_scales = array<f64: {float(scale)}>}} {{"
            f'%y = "earth.mul"(%x, %x) : ({raw}, {raw}) -> {raw}\n'
            f"return %y : {raw}\n}}}}\n"
        )
        destination = out / name
        command = [
            "python3",
            str(root / "integrations/dacapo/generate_model_artifacts.py"),
            "mlp",
            "--operator-spec",
            str(out / "operator-spec.json"),
            "--compiler-profile",
            str(out / "compiler.json"),
            "--hecate-opt",
            compiler,
            "--traced-mlir",
            str(source),
            "--device-counts",
            "1",
            "--plan-format",
            "json",
            "--input-level",
            str(anchor),
            "--waterline",
            "40",
            "--boot-placement",
            "greedy",
            "--output-dir",
            str(destination),
        ]
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (out / (name + ".log")).write_text(run.stdout + run.stderr)
        if drops is None:
            assert (
                run.returncode and "insufficient Q/encoding precision" in run.stderr
            ), run.stderr
        else:
            assert run.returncode == 0, run.stderr
            plan = json.loads(next(destination.glob("*.runtime-plan.json")).read_text())
            desc = {v["id"]: v for v in plan["values"]}
            boots = [op for op in plan["execution"] if op.get("op") == "boot"]
            assert len(boots) == expected_boots
            assert (
                sum(op.get("op") == "rescale" for op in plan["execution"]) == drops + 1
            )
            for boot in boots:
                before, after = desc[boot["inputs"][0]], desc[boot["output"]]
                assert before["level"] == 5 and abs(before["scale_log2"] - 40) < 1e-8
                assert after["level"] == 30 and after["scale_log2"] == 40
            if name == "initial-q34":
                assert desc[plan["external_inputs"][0]]["level"] == 33
        checks += 1
        print(name, "passed", flush=True)

    trace = out / "deep.mlir"
    trace.write_text(
        runpy.run_path(
            str(root / "third_party/dacapo/test/runtime-plan/check-native-planning.py")
        )["source"](18, residual=True)
    )
    command = [
        "python3",
        str(root / "integrations/dacapo/generate_model_artifacts.py"),
        "mlp",
        "--operator-spec",
        str(out / "operator-spec.json"),
        "--compiler-profile",
        str(out / "compiler.json"),
        "--hecate-opt",
        compiler,
        "--traced-mlir",
        str(trace),
        "--device-counts",
        "1",
        "--plan-format",
        "json",
        "--input-level",
        "33",
        "--waterline",
        "40",
    ]
    for strategy in ("greedy", "depth-dp", "optimized"):
        destination = out / ("shifted-" + strategy)
        run = subprocess.run(
            command + ["--boot-placement", strategy, "--output-dir", str(destination)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        (out / ("shifted-" + strategy + ".log")).write_text(run.stdout + run.stderr)
        assert run.returncode == 0, run.stderr
        plan = json.loads(next(destination.glob("*.runtime-plan.json")).read_text())
        desc = {v["id"]: v for v in plan["values"]}
        assert desc[plan["external_inputs"][0]]["level"] == 33
        boots = [op for op in plan["execution"] if op.get("op") == "boot"]
        assert boots
        for boot in boots:
            before, after = desc[boot["inputs"][0]], desc[boot["output"]]
            assert before["level"] == 5 and abs(before["scale_log2"] - 40) < 1e-8
            assert after["level"] == 30 and after["scale_log2"] == 40
        checks += 1
        print("shifted", strategy, "passed", flush=True)

    run = subprocess.run(
        command + ["--scale-strategy", "target", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert (
        run.returncode
        and "unrecognized arguments: --scale-strategy target" in run.stderr
    ), run.stderr
    checks += 1
    for flag in (
        "--native-target-scale-planning",
        "--native-target-scale-goals=unused.json",
        "--native-target-bootstrap-schedule=unused.json",
    ):
        run = subprocess.run(
            common + [flag, str(trace), "-o", str(out / "removed.mlir")],
            capture_output=True,
            text=True,
        )
        assert (
            run.returncode
            and "Unknown command line argument" in run.stderr
            and flag.split("=")[0] in run.stderr
        ), run.stderr
        checks += 1
    print(checks, "native budget and removed-target option checks passed", flush=True)


if __name__ == "__main__":
    main()
