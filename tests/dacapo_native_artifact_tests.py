#!/usr/bin/env python3
"""Exercise native artifact generation with uniform and mixed actual Q chains.

The manual Q50 fixture checks ordinary arithmetic at full Q. Its Boot profile
only anchors the compiler input; that case must contain no Boot operations.
These checks do not execute encrypted inference or validate native Boot accuracy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hecate-opt', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    compiler = root / 'third_party/dacapo'
    source = runpy.run_path(str(compiler / 'test/runtime-plan/check-greedy.py'))['source']
    fixture = json.loads((root/'tests/fixtures/resnet20-manual-q50.json').read_text())
    cases = [
        ('uniform30', 16, [(1 << 30)-(i+1)*32768+1 for i in range(50)], 8, 12),
        ('mixed30-31-32', 16,
         [(1 << (30+i%3))-(i+1)*32768+1 for i in range(50)], 8, 12),
        ('manual-q50', fixture['poly_degree'], fixture['rns_moduli'], 49, 5),
    ]
    checks = 0
    with tempfile.TemporaryDirectory(prefix='native-artifacts-') as temporary:
        out = args.output_dir.resolve() if args.output_dir else Path(temporary)
        out.mkdir(parents=True, exist_ok=True)
        for name, degree, moduli, anchor, depth in cases:
            work = out/name; work.mkdir()
            config = json.loads((compiler/'config.json').read_text())
            config.update(polynomialDegree=degree, rescalingFactor=30, bootstrapLevelLowerBound=1,
                          bootstrapLevelUpperBound=anchor-3, levelLowerBound=1, levelUpperBound=anchor-3)
            spec = json.loads((compiler/'test/runtime-plan/lazy-physical-level-operator-spec.json').read_text())
            for operator_name, operator in spec['operators'].items():
                operator['supported'] = True
                if operator_name != 'boot':
                    operator.update(latency_us_by_level=[10]*len(moduli), noise_by_level=None)
            spec['context'].update(poly_degree=degree, default_scale_log2=40, rns_moduli=moduli,
                                   rns_moduli_log2=[q.bit_length() for q in moduli],
                                   max_modulus_log2=max(q.bit_length() for q in moduli))
            spec['rescale_mode'] = 'native'
            spec['boot_profiles'] = [dict(profile_id='native45', implementation='native',
                input_level_min=4, input_level_max=49, input_components=2,
                output_level=anchor, output_scale_log2=45, output_components=2,
                latency_us_by_input_level=[85000]*len(moduli), noise_by_input_level=None,
                host_requirements=dict(needs_secret_key=False, needs_host_compute=False))]
            profile = work/'config.json'; profile.write_text(json.dumps(config))
            operator_spec = work/'spec.json'; operator_spec.write_text(json.dumps(spec))
            traced = work/'input.mlir'; traced.write_text(source(depth))

            def generate(strategy, destination, dry_run=False):
                command = [sys.executable, str(root/'integrations/dacapo/generate_model_artifacts.py'),
                    'mlp', '--operator-spec', str(operator_spec), '--compiler-profile', str(profile),
                    '--hecate-opt', str(args.hecate_opt.resolve()), '--traced-mlir', str(traced),
                    '--output-dir', str(destination), '--device-counts', '1', '--plan-format', 'json',
                    '--boot-placement', strategy, '--waterline', '40']
                if dry_run: command.append('--dry-run')
                run = subprocess.run(command, text=True, capture_output=True, timeout=90)
                (work/(destination.name+'.log')).write_text(run.stdout+run.stderr)
                if dry_run: return run
                assert run.returncode == 0, (name, strategy, run.stderr)
                plans = list(destination.glob('*.runtime-plan.json')); assert len(plans) == 1
                return json.loads(plans[0].read_text())

            baseline = None
            for strategy in ['optimized', 'greedy', 'depth-dp']:
                plan = generate(strategy, work/strategy)
                if strategy == 'greedy': baseline = plan
                values = {v['id']: v for v in plan['values']}
                boots = [s for s in plan['execution'] if s.get('op') == 'boot']
                assert plan['format_version'] == 5
                assert bool(boots) == (name != 'manual-q50'), (name, boots)
                assert all(values[s['output']]['level'] == anchor and
                           values[s['output']]['scale_log2'] == 45 for s in boots)
                assert all(4 <= v['level'] <= anchor for v in values.values())
                dropped_widths = set()
                for step in plan['execution']:
                    if step.get('op') != 'rescale': continue
                    before, after = values[step['inputs'][0]], values[step['output']]
                    removed = moduli[after['level']+1:before['level']+1]
                    assert removed
                    expected = before['scale_log2']-sum(math.log2(q) for q in removed)
                    assert math.isclose(after['scale_log2'], expected, abs_tol=1e-8)
                    assert math.isclose(step['attrs']['target_scale_log2'], expected, abs_tol=1e-8)
                    assert after['scale_log2'] >= 40
                    dropped_widths.update(q.bit_length() for q in removed)
                assert dropped_widths == ({30} if name == 'uniform30' else {30,31,32}), dropped_widths
                assert plan['target']['operator_spec']['source_sha256'] == 'sha256:'+hashlib.sha256(operator_spec.read_bytes()).hexdigest()
                checks += 1
                print(name, strategy, 'actual-prime export passed', flush=True)

            # A legacy scalar factor must not affect the actual-prime schedule.
            config['rescalingFactor'] = 120; profile.write_text(json.dumps(config))
            changed = generate('greedy', work/'factor120')
            assert baseline['values'] == changed['values']
            assert baseline['execution'] == changed['execution']
            checks += 1
            valid = operator_spec.read_text()
            bad = json.loads(valid); bad['context']['rns_moduli'].pop()
            operator_spec.write_text(json.dumps(bad))
            run = generate('greedy', work/'invalid-length', dry_run=True)
            assert run.returncode != 0 and 'actual Q array' in run.stderr, run.stderr
            checks += 1
            bad = json.loads(valid); bad['context']['rns_moduli_log2'][0] += 1
            operator_spec.write_text(json.dumps(bad))
            run = generate('greedy', work/'invalid-width', dry_run=True)
            assert run.returncode != 0 and 'declared bit width' in run.stderr, run.stderr
            checks += 1
            operator_spec.write_text(valid)
        print(f'{checks} native artifact checks passed', flush=True)


if __name__ == '__main__':
    main()
