#!/usr/bin/env python3
"""Check backwards scale planning, precise encodings and explicit rejection."""

import argparse
import itertools
import json
import math
from pathlib import Path
import runpy
import shlex
import subprocess


CT = "tensor<1x!earth.ci<0 * 0>>"
PT = "tensor<1x!earth.pl<0 * 0>>"


def source(case):
    lines = [f"module {{ func.func @target(%x: {CT}) -> {CT} {{"]

    def constant(name, value):
        lines.append(
            f'%{name} = "earth.constant"() {{value = dense<{value}> : tensor<32768xf64>, rms_var = {abs(value)} : f64}} : () -> {PT}'
        )

    def op(name, kind, *args):
        operands = ", ".join("%" + v for v in args)
        types = ", ".join(PT if v.startswith("c") else CT for v in args)
        lines.append(f'%{name} = "earth.{kind}"({operands}) : ({types}) -> {CT}')

    if case == "linear":
        constant("c0", 1.01)
        constant("c1", -0.125)
        op("a", "mul", "x", "c0")
        op("b", "mul", "x", "c1")
        op("y", "add", "a", "b")

        def expected(x):
            return 0.885 * x
    elif case == "nested-plain":
        constant("c0", 0.75)
        constant("c1", 0.125)
        op("a", "mul", "x", "c0")
        op("y", "mul", "a", "c1")

        def expected(x):
            return 0.09375 * x
    elif case == "unit-coefficients":
        constant("c0", 1.0)
        constant("c1", -1.0)
        op("a", "mul", "x", "c0")
        op("b", "mul", "x", "c1")
        op("square", "mul", "x", "x")
        op("first", "add", "square", "a")
        op("y", "add", "first", "b")

        def expected(x):
            return x * x
    elif case == "shared-basis":
        constant("c0", -1.0)
        constant("c1", 0.75)
        constant("c2", 0.125)
        op("twice", "add", "x", "x")
        op("square", "mul", "twice", "x")
        op("t2", "add", "square", "c0")
        op("product", "mul", "twice", "t2")
        op("negative", "negate", "x")
        op("t3", "add", "product", "negative")
        op("a", "mul", "x", "c1")
        op("b", "mul", "t3", "c2")
        op("y", "add", "a", "b")

        def expected(x):
            return 0.75 * x + 0.125 * (4 * x * x * x - 3 * x)
    elif case in ("explicit-boot", "boot-scale-rejection"):
        op("square", "mul", "x", "x")
        attrs = " {planned_scale = 45 : i64}" if case == "boot-scale-rejection" else ""
        lines.append(f'%refresh = "earth.bootstrap"(%square){attrs} : ({CT}) -> {CT}')
        op("y", "mul", "refresh", "refresh")

        def expected(x):
            return x**4
    else:
        previous = "x"
        for i in range(24):
            name = "square" + str(i)
            op(name, "mul", previous, previous)
            previous = name
        lines.append(f"return %{previous} : {CT}")
        lines.append("}}")

        def expected(x):
            for _ in range(24):
                x *= x
            return x

        return "\n".join(lines) + "\n", expected
    lines += [f"return %y : {CT}", "}}"]
    return "\n".join(lines) + "\n", expected


def check(plan, spec, expected):
    desc = {v["id"]: v for v in plan["values"]}
    operations = list(
        itertools.chain(plan["initialization"], plan["execution"], plan["finalization"])
    )
    producers = {op["output"]: op for op in operations if "output" in op}
    boot_alignment = set()
    for op in operations:
        if op.get("op") != "boot":
            continue
        value = op["inputs"][0]
        while value in producers:
            source = producers[value]
            if source.get("op") == "mul_cp":
                boot_alignment.add(value)
                break
            if source.get("op") not in ("rescale", "mod_switch"):
                break
            value = source["inputs"][0]
    constants = {}
    for x in (-0.75, -0.25, 0.0, 0.25, 0.75):
        values = {plan["external_inputs"][0]: x}
        for op in itertools.chain(
            plan["initialization"], plan["execution"], plan["finalization"]
        ):
            kind = op.get("op", op["kind"])
            outputs = op.get("outputs", [op["output"]] if "output" in op else [])
            if not outputs:
                continue
            out = outputs[0]
            after = desc[out]
            if kind == "encode":
                data = op["payload"]["values"]
                assert all(v == data[0] for v in data)
                values[out] = constants[out] = data[0]
                continue
            inputs = op["inputs"]
            before = [desc[v] for v in inputs]
            args = [values[v] for v in inputs]
            if kind == "transfer":
                values[out] = args[0]
                if inputs[0] in constants:
                    constants[out] = constants[inputs[0]]
                continue
            if kind == "boot":
                # Clear evaluation checks structure; encrypted Boot is checked
                # separately against the manual profile's sine-map reference.
                assert before[0]["level"] == 5
                assert abs(before[0]["scale_log2"] - 40) < 1e-8
                assert after["level"] == 30 and after["scale_log2"] == 40
                values[out] = args[0]
            elif kind in ("mul_cc", "mul_cp"):
                assert (
                    abs(sum(v["scale_log2"] for v in before) - after["scale_log2"])
                    < 1e-8
                )
                if kind == "mul_cp" and constants[inputs[1]] == 1:
                    s = before[1]["scale_log2"]
                    precision = 30 if out in boot_alignment else 40
                    assert abs(s) < 1e-8 or s >= precision - 1e-8
                values[out] = args[0] * args[1]
            elif kind in ("add_cc", "add_cp"):
                assert abs(before[0]["scale_log2"] - before[1]["scale_log2"]) < 1e-8
                values[out] = args[0] + args[1]
            else:
                values[out] = -args[0] if kind == "negate" else args[0]
                assert kind in ("negate", "relinearize", "mod_switch", "rescale")
                if kind == "rescale":
                    scale = before[0]["scale_log2"] - sum(
                        math.log2(spec["context"]["rns_moduli"][i])
                        for i in range(before[0]["level"], after["level"], -1)
                    )
                    assert (
                        abs(scale - after["scale_log2"]) < 1e-8 and scale >= 40 - 1e-8
                    )
        assert abs(values[plan["final_outputs"][0]] - expected(x)) < 1e-12


def boot_count(plan):
    return sum(
        op.get("op", op["kind"]) == "boot"
        for phase in ("initialization", "execution", "finalization")
        for op in plan[phase]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _, spec = runpy.run_path(
        str(root / "integrations/dacapo/generate_resnet20_manual_profiles.py")
    )["generate"](out)
    for case in (
        "linear",
        "nested-plain",
        "unit-coefficients",
        "shared-basis",
        "explicit-boot",
        "boot-scale-rejection",
        "automatic-boot",
    ):
        text, expected = source(case)
        trace = out / (case + ".mlir")
        trace.write_text(text)
        cmd = [
            "python3",
            str(root / "integrations/dacapo/generate_model_artifacts.py"),
            "mlp",
            "--operator-spec",
            str(out / "operator-spec.json"),
            "--compiler-profile",
            str(out / "compiler.json"),
            "--hecate-opt",
            str(args.hecate_opt.resolve()),
            "--traced-mlir",
            str(trace),
            "--output-dir",
            str(out / case),
            "--device-counts",
            "1",
            "--plan-format",
            "json",
            "--boot-placement",
            "greedy",
            "--waterline",
            "40",
            "--scale-strategy",
            "target",
            "--inline-payload-max-bytes",
            "1048576",
        ]
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        (out / (case + ".log")).write_text(run.stdout + run.stderr)
        if case == "boot-scale-rejection":
            assert run.returncode and "Boot output scale disagrees" in run.stderr, (
                run.stderr
            )
        else:
            assert run.returncode == 0, run.stderr
            plan = json.loads(
                next((out / case).glob("*.runtime-plan.json")).read_text()
            )
            assert boot_count(plan) == (
                1 if case in ("explicit-boot", "automatic-boot") else 0
            )
            check(plan, spec, expected)
        print(case, "passed", flush=True)
    external = out / "external-plain.mlir"
    external.write_text(
        f'module {{ func.func @external(%x: {CT}, %p: {PT}) -> {CT} {{ %y = "earth.mul"(%x, %p) : ({CT}, {PT}) -> {CT}\n return %y : {CT} }} }}'
    )
    cmd[cmd.index("--traced-mlir") + 1] = str(external)
    cmd[cmd.index("--output-dir") + 1] = str(out / "external-plain")
    run = subprocess.run(cmd, capture_output=True, text=True)
    assert run.returncode and "requires locally encoded plaintexts" in run.stderr, (
        run.stderr
    )
    print("external plaintext rejection passed", flush=True)
    fixed = out / "fixed-schedule.json"
    wrapper = out / "hecate-fixed-schedule"
    wrapper.write_text(
        "#!/bin/sh\nexec "
        + shlex.quote(str(args.hecate_opt.resolve()))
        + " --native-target-bootstrap-schedule="
        + shlex.quote(str(fixed))
        + ' "$@"\n'
    )
    wrapper.chmod(0o755)
    scheduled = cmd[:]
    scheduled[scheduled.index("--hecate-opt") + 1] = str(wrapper)
    scheduled[scheduled.index("--traced-mlir") + 1] = str(out / "automatic-boot.mlir")
    scheduled[scheduled.index("--output-dir") + 1] = str(out / "fixed-schedule")
    fixed.write_text("[[14, 14]]\n")
    run = subprocess.run(scheduled, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    plan = json.loads(
        next((out / "fixed-schedule").glob("*.runtime-plan.json")).read_text()
    )
    assert boot_count(plan) == 1
    check(plan, spec, source("automatic-boot")[1])
    print("fixed schedule passed", flush=True)
    fixed.write_text("[]\n")
    run = subprocess.run(scheduled, capture_output=True, text=True, timeout=120)
    assert run.returncode and "no precision-safe target scale plan" in run.stderr
    print("fixed schedule does not insert extra Boots passed", flush=True)
    cmd[cmd.index("greedy")] = "optimized"
    run = subprocess.run(cmd, capture_output=True, text=True)
    assert (
        run.returncode
        and "target scale strategy requires native moduli and greedy" in run.stderr
    )
    print("11 target scale checks passed", flush=True)


if __name__ == "__main__":
    main()
