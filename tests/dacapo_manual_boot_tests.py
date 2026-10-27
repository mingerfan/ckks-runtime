#!/usr/bin/env python3
"""Check manual Boot input alignment, Q reservation and actual-prime export."""
import argparse
import json
import math
import re
from pathlib import Path
import runpy
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    compiler = root/'third_party/dacapo'
    out = args.output_dir.resolve()
    generate = runpy.run_path(str(root/'integrations/dacapo/generate_resnet20_manual_profiles.py'))['generate']
    _, spec = generate(out)
    source = runpy.run_path(str(compiler/'test/runtime-plan/check-native-planning.py'))['source']
    (out/'squares.mlir').write_text(source(18, residual=True))
    lines = source(18, residual=True).splitlines()
    ct, pt = 'tensor<1x!earth.ci<0 * 0>>', 'tensor<1x!earth.pl<0 * 0>>'
    seed = [f'%half = "earth.constant"() {{value = dense<0.5> : tensor<32768xf64>, '
            f'rms_var = 0.5 : f64}} : () -> {pt}',
            f'%seed = "earth.mul"(%x, %half) : ({ct}, {pt}) -> {ct}']
    (out/'scaled.mlir').write_text('\n'.join(lines[:1]+seed+
        [line.replace('%x', '%seed') for line in lines[1:]]))
    checks = 0
    for strategy, scaled in [('greedy', False), ('depth-dp', False), ('optimized', False), ('depth-dp', True)]:
        case = 'scaled-depth-dp' if scaled else strategy
        target = out/case
        result = subprocess.run([
            'python3', str(root/'integrations/dacapo/generate_model_artifacts.py'), 'mlp',
            '--operator-spec', str(out/'operator-spec.json'), '--compiler-profile', str(out/'compiler.json'),
            '--hecate-opt', str(args.hecate_opt.resolve()), '--traced-mlir', str(out/('scaled.mlir' if scaled else 'squares.mlir')),
            '--output-dir', str(target), '--device-counts', '1', '--plan-format', 'json',
            '--boot-placement', strategy, '--waterline', '40',
            '--inline-payload-max-bytes', '1048576'],
            capture_output=True, text=True, timeout=180)
        (out/(case+'.log')).write_text(result.stdout+result.stderr)
        assert result.returncode == 0, (strategy, result.stderr[-6000:])
        plan = json.loads(next(target.glob('*.runtime-plan.json')).read_text())
        values = {v['id']: v for v in plan['values']}
        boots = [op for op in plan['execution'] if op.get('op') == 'boot']
        assert boots
        if scaled:
            # The live 0.5*x branch has a non-40 actual scale, so this checks
            # alignment of a nonzero signal as well as the deep square branch.
            assert len(boots) == 2
        for boot in boots:
            before, after = values[boot['inputs'][0]], values[boot['output']]
            assert before['level'] == 5 and abs(before['scale_log2']-40) < 1e-8, before
            assert after['level'] == 30 and after['scale_log2'] == 40, after
        for op in plan['execution']:
            if op.get('op') != 'rescale': continue
            before, after = values[op['inputs'][0]], values[op['output']]
            expected = before['scale_log2'] - sum(math.log2(q) for q in
                spec['context']['rns_moduli'][after['level']+1:before['level']+1])
            assert abs(after['scale_log2']-expected) < 1e-8
        print(case, len(boots), 'manual Boot input/output contracts passed', flush=True)
        checks += 1

    # Isolate alignment: exact scale needs no Rescale, scale41 needs one,
    # scale71 needs two to retain >=30 bits of unit-plaintext precision.
    for scale, level, drops in [(40, 5, 0), (40, 7, 0), (41, 7, 1), (71, 7, 2), (71, 6, None)]:
        source_type = f'tensor<1x!ckks.poly<2 * {scale} * {level}>>'
        result_type = 'tensor<1x!ckks.poly<2 * 40 * 30>>'
        text = (f'module {{ func.func @align(%x: {source_type}) -> {result_type} '
            f'attributes {{init_level = 30 : i64}} {{ %y = "ckks.bootstrapc"(%x) : '
            f'({source_type}) -> {result_type}\n return %y : {result_type} }} }}')
        (out/'align.mlir').write_text(text)
        pipeline = ('builtin.module(func.func(materialize-ckks-physical-levels{'
            f'operator-spec={out}/operator-spec.json '
            'boot-profile=poseidon-resnet20-manual-q50-scale40 levels-per-logical-level=1}))')
        result = subprocess.run([str(args.hecate_opt.resolve()),
            f'--ckks-config={out}/compiler.json', f'--pass-pipeline={pipeline}',
            str(out/'align.mlir'), '-o', str(out/'aligned.mlir')], text=True, capture_output=True)
        if drops is None:
            assert result.returncode != 0 and 'insufficient Q/encoding precision' in result.stderr, result.stderr
        else:
            assert result.returncode == 0, result.stderr
            text = (out/'aligned.mlir').read_text()
            assert text.count('"ckks.rescalec"') == drops, text
            assert re.search(r'!ckks.poly<2 \* (40|41) \* 5>', text), text
        checks += 1
    print(checks, 'manual Boot alignment checks passed')


if __name__ == '__main__':
    main()
