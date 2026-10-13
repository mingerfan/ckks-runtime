#!/usr/bin/env python3
"""Compare direct DaCapo V4 binary export with JSON through the production reader."""
import argparse
from pathlib import Path
import subprocess
import tempfile


SOURCE = r'''module {
func.func @waterline(%arg0: tensor<1x!ckks.poly<2 * 40 * 1>> {dist.rank = 0 : i64, dist.device = 0 : i64}) -> tensor<1x!ckks.poly<2 * 40 * 1>> attributes {dist.device_counts = array<i64: 1>} {
  %a = "ckks.encode"() <{payload = dense<2.0> : tensor<1xf64>}> {dist.logical_id = 1 : i64, dist.rank = 0 : i64, dist.device = -1 : i64} : () -> tensor<1x!ckks.poly<1 * 40 * 1>>
  %b = "ckks.encode"() <{payload = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<8xf64>}> {dist.logical_id = 2 : i64, dist.rank = 0 : i64, dist.device = -1 : i64} : () -> tensor<1x!ckks.poly<1 * 40 * 1>>
  %p = "dist.transfer"(%a) <{transfer_id = 0 : i64, source_rank = 0 : i64, source_device = -1 : i64, destination_rank = 0 : i64, destination_device = 0 : i64, initialization = true}> {dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<1 * 40 * 1>>) -> tensor<1x!ckks.poly<1 * 40 * 1>>
  %q = "dist.transfer"(%b) <{transfer_id = 1 : i64, source_rank = 0 : i64, source_device = -1 : i64, destination_rank = 0 : i64, destination_device = 0 : i64, initialization = true}> {dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<1 * 40 * 1>>) -> tensor<1x!ckks.poly<1 * 40 * 1>>
  %v = "ckks.addcp"(%arg0, %p) {dist.logical_id = 3 : i64, dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 1>>, tensor<1x!ckks.poly<1 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
  %y = "ckks.addcp"(%v, %q) {dist.logical_id = 4 : i64, dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 1>>, tensor<1x!ckks.poly<1 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
  return %y : tensor<1x!ckks.poly<2 * 40 * 1>>
}
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--reader-test", type=Path, required=True)
    parser.add_argument("--dacapo-source", type=Path, required=True)
    args = parser.parse_args()
    compiler = args.hecate_opt.resolve()
    reader = args.reader_test.resolve()
    dacapo_source = args.dacapo_source.resolve()
    spec = dacapo_source / "test/runtime-plan/placement-operator-spec.json"
    with tempfile.TemporaryDirectory(prefix="ckks-compiler-v4-binary-") as directory:
        temporary = Path(directory)
        source = temporary / "input.mlir"
        source.write_text(SOURCE)
        paths = {}
        for plan_format in ("json", "binary"):
            prefix = temporary / plan_format
            passes = [
                f"plan-plaintext-waterline{{operator-spec={spec} gpu-budget-bytes=8192 fraction=0.5 "
                "host-budget-bytes=16384 pinned-budget-bytes=8192 raw-budget-bytes=128}",
                "plan-runtime-memory{enable-reuse=false}",
                f"emit-runtime-plan{{plan-format={plan_format} bundle-format=pack inline-payload-max-bytes=32 "
                f"prefix={prefix} target-id=test operator-spec-id=test "
                f"operator-spec-sha256=sha256:{'0' * 64} context-id=test-context}}",
            ]
            subprocess.run([str(compiler), str(source), f"-p=builtin.module(func.func({','.join(passes)}))",
                            "-o", str(temporary / f"{plan_format}.mlir")], check=True, cwd=dacapo_source)
            paths[plan_format] = Path(f"{prefix}.waterline.runtime-plan.{'json' if plan_format == 'json' else 'bin'}")
        subprocess.run([str(reader), str(paths["json"]), str(paths["binary"])], check=True)


if __name__ == "__main__":
    main()
