#!/usr/bin/env python3
"""Run the artifact generator through all planners with native scale45 Boot."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hecate-opt', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    compiler = root / 'third_party/dacapo'
    source = runpy.run_path(str(compiler / 'test/runtime-plan/check-greedy.py'))['source']
    with tempfile.TemporaryDirectory(prefix='native-artifacts-') as temporary:
        out = Path(temporary)
        config = json.loads((compiler / 'config.json').read_text())
        config.update(polynomialDegree=16, rescalingFactor=30, bootstrapLevelLowerBound=1,
                      bootstrapLevelUpperBound=5, levelLowerBound=1, levelUpperBound=5)
        spec = json.loads((compiler / 'test/runtime-plan/lazy-physical-level-operator-spec.json').read_text())
        for operator in spec['operators'].values():
            operator.update(supported=True, latency_us_by_level=[10] * 46)
        spec['context']['default_scale_log2'] = 40
        spec['context']['rns_moduli'] = [(1 << 30) - (i + 1) * 32768 + 1 for i in range(50)]
        spec['rescale_mode'] = 'native'
        spec['boot_profiles'] = [dict(profile_id='native45', implementation='native',
            input_level_min=4, input_level_max=49, input_components=2,
            output_level=8, output_scale_log2=45, output_components=2,
            latency_us_by_input_level=[85000]*46, noise_by_input_level=None,
            host_requirements=dict(needs_secret_key=False, needs_host_compute=False))]
        profile = out / 'config.json'; profile.write_text(json.dumps(config))
        operator_spec = out / 'spec.json'; operator_spec.write_text(json.dumps(spec))
        traced = out / 'input.mlir'; traced.write_text(source(12))
        for strategy in ['optimized', 'greedy', 'depth-dp']:
            destination = out / strategy
            command = [sys.executable, str(root / 'integrations/dacapo/generate_model_artifacts.py'),
                'mlp', '--operator-spec', str(operator_spec), '--compiler-profile', str(profile),
                '--hecate-opt', str(args.hecate_opt.resolve()), '--traced-mlir', str(traced),
                '--output-dir', str(destination), '--device-counts', '1', '--plan-format', 'json',
                '--boot-placement', strategy, '--waterline', '40']
            run = subprocess.run(command, text=True, capture_output=True, timeout=60)
            assert run.returncode == 0, (strategy, run.stderr)
            plans = list(destination.glob('*.runtime-plan.json'))
            assert len(plans) == 1
            plan = json.loads(plans[0].read_text()); values = {v['id']: v for v in plan['values']}
            boots = [s for s in plan['execution'] if s.get('op') == 'boot']
            assert plan['format_version'] == 5 and boots, plan
            assert all(values[s['output']]['level'] == 8 and
                       values[s['output']]['scale_log2'] == 45 for s in boots)
            assert all(4 <= v['level'] <= 8 for v in values.values())
            print(strategy, 'native scale45 artifact generation passed')
        config['rescalingFactor'] = 120; profile.write_text(json.dumps(config))
        run = subprocess.run(command + ['--dry-run'], text=True, capture_output=True)
        assert run.returncode != 0 and 'matching compiler rescalingFactor' in run.stderr, run.stderr


if __name__ == '__main__':
    main()
