#!/usr/bin/env python3
"""Generate the manual Q50 Boot contract for compilation/accuracy tests.

Latency tables are placeholders, not measurements. GPU setup must use the
matching public Q/P fixture and build_resnet20_manual().
"""
import argparse
import hashlib
import json
from pathlib import Path


def generate(destination: Path):
    root = Path(__file__).resolve().parents[2]
    compiler = root / 'third_party/dacapo'
    fixture = json.loads((root/'tests/fixtures/resnet20-manual-q50.json').read_text())
    q = fixture['rns_moduli']
    config = json.loads((compiler/'config.json').read_text())
    config.update(polynomialDegree=fixture['poly_degree'], rescalingFactor=30,
                  bootstrapLevelLowerBound=1, bootstrapLevelUpperBound=24,
                  levelLowerBound=1, levelUpperBound=24)
    for name, table in config['latencyTable'].items():
        config['latencyTable'][name] = [max(table)] * 51
    spec = json.loads((compiler/'test/runtime-plan/lazy-physical-level-operator-spec.json').read_text())
    spec.update(spec_id='poseidon-resnet20-manual-q50', target_id='poseidon-ckks-gpu',
                status='validated', rescale_mode='native')
    fixture_path = root/'tests/fixtures/resnet20-manual-q50.json'
    spec['provenance'] = dict(kind='test-fixture', repository='mingerfan/poseidon',
        revision='local', path='tests/fixtures/resnet20-manual-q50.json',
        source_sha256='sha256:'+hashlib.sha256(fixture_path.read_bytes()).hexdigest())
    spec['context'].update(context_id='poseidon-resnet20-manual-q50',
        poly_degree=fixture['poly_degree'], default_scale_log2=40,
        rns_moduli=q, rns_moduli_log2=[v.bit_length() for v in q],
        max_modulus_log2=max(v.bit_length() for v in q))
    spec['levels'] = dict(lower_bound=4, upper_bound=49)
    for name, operator in spec['operators'].items():
        operator['supported'] = True
        if name != 'boot':
            operator.update(latency_us_by_level=[10]*50, noise_by_level=None)
    spec['boot_profiles'] = [dict(
        profile_id='poseidon-resnet20-manual-q50-scale40', implementation='native',
        input_level_min=5, input_level_max=5, input_scale_log2=40, input_components=2,
        output_level=30, output_scale_log2=40, output_components=2,
        latency_us_by_input_level=[85000 if i == 5 else 0 for i in range(50)],
        noise_by_input_level=None,
        host_requirements=dict(needs_secret_key=False, needs_host_compute=False))]
    destination.mkdir(parents=True, exist_ok=True)
    for name, document in [('compiler.json', config), ('operator-spec.json', spec)]:
        (destination/name).write_text(json.dumps(document, indent=2)+'\n')
    return config, spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    generate(parser.parse_args().output_dir)
