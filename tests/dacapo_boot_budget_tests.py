#!/usr/bin/env python3
"""Check the full native Q budget and value-dependent Boot alignment."""

import argparse
import json
from pathlib import Path
import runpy
import subprocess


def trace(states, kind):
    types = [f"tensor<1x!earth.ci<{scale} * {level}>>" for level, scale in states]
    raw = "tensor<1x!earth.ci<0 * 0>>"
    output = raw if kind == "tail" else "tensor<1x!earth.ci<40 * 0>>"
    args = ", ".join(f"%x{i}: {raw}" for i in range(len(types)))
    scales = ", ".join(str(float(scale)) for _, scale in states)
    text = f"module {{ func.func @budget({args}) -> {output} attributes {{segment_inputType = [{', '.join(types)}], segment_input_scales = array<f64: {scales}>}} {{\n"
    source, source_type = "%x0", raw
    if kind == "add":
        source_type = raw
        text += f'%sum = "earth.add"(%x0, %x1) : ({raw}, {raw}) -> {source_type}\n'
        source = "%sum"
    elif kind == "double":
        text += f'%d = "earth.add"(%x0, %x0) : ({raw}, {raw}) -> {raw}\n'
        text += f'%n = "earth.negate"(%d) : ({raw}) -> {raw}\n'
        text += f'%p = "earth.mul"(%d, %x0) : ({raw}, {raw}) -> {raw}\n'
        text += f'%sum = "earth.add"(%p, %n) : ({raw}, {raw}) -> {raw}\n'
        source = "%sum"
    elif kind == "rotate":
        text += (
            f'%r = "earth.rotate"(%x0) {{offset = array<i64: 1>}} : ({raw}) -> {raw}\n'
        )
        text += f'%p = "earth.mul"(%r, %x0) : ({raw}, {raw}) -> {raw}\n'
        source = "%p"
    elif kind == "tail":
        plain = "tensor<1x!earth.pl<0 * 0>>"
        text += f'%c = "earth.constant"() {{value = dense<0.75> : tensor<32768xf64>, rms_var = 0.75 : f64}} : () -> {plain}\n'
        text += f'%p = "earth.mul"(%x0, %c) : ({raw}, {plain}) -> {raw}\n'
        return text + f"return %p : {raw}\n}}}}\n"
    text += f'%y = "earth.bootstrap"({source}) {{targetScale = 40 : i64}} : ({source_type}) -> {output}\nreturn %y : {output}\n}}}}\n'
    return text


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
    cases = [
        ("all-25-q", [(25, 40)], "", 0, 1, None),
        ("one-q-alignment", [(24, 41)], "", 1, 1, None),
        ("two-q-alignment", [(23, 71)], "", 2, 1, None),
        ("no-q-scale41", [(25, 41)], "", None, 0, None),
        ("one-q-scale71", [(24, 71)], "", None, 0, None),
        ("two-q-scale179", [(23, 179)], "", None, 0, None),
        ("shared-branch-rescale", [(23, 48), (25, 40)], "add", 2, 1, None),
        ("double-frontier", [(25, 40)], "double", 3, 2, None),
        ("rotate-frontier", [(25, 40)], "rotate", 3, 2, None),
        ("terminal-below-boot-level", [(24, 40)], "tail", 2, 0, None),
        ("initial-q34", [(0, 40)], "", 0, 1, 33),
        ("shifted-anchor-25-q", [(28, 40)], "", 0, 1, 33),
    ]
    for name, states, kind, drops, boot_count, input_level in cases:
        source = out / (name + ".mlir")
        source.write_text(trace(states, kind))
        command = [
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
            str(source),
            "--output-dir",
            str(out / name),
            "--device-counts",
            "1",
            "--plan-format",
            "json",
            "--boot-placement",
            "greedy",
            "--scale-strategy",
            "target",
            "--waterline",
            "40",
        ]
        if input_level is not None:
            command += ["--input-level", str(input_level)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (out / (name + ".log")).write_text(result.stdout + result.stderr)
        if drops is None:
            assert (
                result.returncode
                and "insufficient Q/encoding precision" in result.stderr
            ), result.stderr
        else:
            assert result.returncode == 0, result.stderr
            plan = json.loads(
                next((out / name).glob("*.runtime-plan.json")).read_text()
            )
            desc = {v["id"]: v for v in plan["values"]}
            boots = [op for op in plan["execution"] if op.get("op") == "boot"]
            assert len(boots) == boot_count
            for boot in boots:
                before = desc[boot["inputs"][0]]
                assert before["level"] == 5 and abs(before["scale_log2"] - 40) < 1e-8
                assert desc[boot["output"]]["level"] == 30
            if name == "initial-q34":
                assert desc[plan["external_inputs"][0]]["level"] == 33
            if kind == "tail":
                assert desc[plan["final_outputs"][0]]["level"] == 4
            assert sum(op.get("op") == "rescale" for op in plan["execution"]) == drops
        print(name, "passed", flush=True)
    deep = runpy.run_path(
        str(root / "third_party/dacapo/test/runtime-plan/check-native-planning.py")
    )["source"](18, residual=True)
    source = out / "shifted-threshold.mlir"
    source.write_text(deep)
    for strategy in ("greedy", "depth-dp", "optimized"):
        command[command.index("--traced-mlir") + 1] = str(source)
        command[command.index("--output-dir") + 1] = str(out / ("shifted-" + strategy))
        command[command.index("--scale-strategy") + 1] = "threshold"
        command[command.index("--boot-placement") + 1] = strategy
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (out / ("shifted-" + strategy + ".log")).write_text(
            result.stdout + result.stderr
        )
        assert result.returncode == 0, result.stderr
        plan = json.loads(
            next(
                (out / ("shifted-" + strategy)).glob("*.runtime-plan.json")
            ).read_text()
        )
        desc = {v["id"]: v for v in plan["values"]}
        assert desc[plan["external_inputs"][0]]["level"] == 33
        boots = [op for op in plan["execution"] if op.get("op") == "boot"]
        assert boots
        for boot in boots:
            before = desc[boot["inputs"][0]]
            assert before["level"] == 5 and abs(before["scale_log2"] - 40) < 1e-8
            assert desc[boot["output"]]["level"] == 30
        print("shifted threshold", strategy, "passed", flush=True)
    print("15 full-Q budget checks passed", flush=True)


if __name__ == "__main__":
    main()
