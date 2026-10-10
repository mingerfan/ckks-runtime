#!/usr/bin/env python3
"""Exercise conversion and failed publication with tiny local artifacts."""
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / 'tools/pack_plaintext_bundle.py'


class PackBundleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'old.bundle'
        (self.source / 'data').mkdir(parents=True)
        self.payloads = [struct.pack('<2d', 1.5, -2.0), struct.pack('<d', 3.0)]
        entries = []
        for payload in self.payloads:
            digest = hashlib.sha256(payload).hexdigest()
            (self.source / 'data' / (digest + '.bin')).write_bytes(payload)
            entries.append({'content': 'sha256:' + digest, 'byte_length': len(payload)})
        raw = json.dumps({'bundle_format_version': 1, 'bundle_id': 'test', 'version': 1,
                          'blobs': entries}).encode()
        (self.source / 'manifest.json').write_bytes(raw)
        self.old_digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
        self.plan = self.root / 'old.plan.json'
        self.plan.write_text(json.dumps({'execution': [], 'plaintext_bundle': {
            'id': 'test', 'version': 1, 'manifest_sha256': self.old_digest}}) + '\n')
        self.output = self.root / 'new.bundle'
        self.output_plan = self.root / 'new.plan.json'

    def convert(self, success):
        result = subprocess.run([sys.executable, str(TOOL), str(self.source), str(self.output),
                                 '--plan', str(self.plan), '--output-plan', str(self.output_plan)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return result

    def assert_clean_failure(self):
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output_plan.exists())
        self.assertFalse(self.output.with_name('new.bundle.tmp').exists())
        self.assertFalse(self.output_plan.with_name('new.plan.json.tmp').exists())

    def test_conversion_and_existing_output(self):
        result = self.convert(True)
        report = json.loads(result.stdout)
        self.assertEqual(report['blob_hashes_recomputed'], 0)
        self.assertEqual(report['pack_bytes'], 24)
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), ['data.bin', 'manifest.json'])
        self.assertEqual((self.output / 'data.bin').read_bytes(), b''.join(self.payloads))
        raw = (self.output / 'manifest.json').read_bytes()
        manifest = json.loads(raw)
        self.assertEqual([e['offset'] for e in manifest['blobs']], [0, 16])
        new_digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
        expected = self.plan.read_bytes().replace(self.old_digest.encode(), new_digest.encode())
        self.assertEqual(self.output_plan.read_bytes(), expected)
        self.convert(False)
        self.assertEqual(self.output_plan.read_bytes(), expected)

    def test_truncated_blob(self):
        next((self.source / 'data').iterdir()).write_bytes(b'')
        self.convert(False)
        self.assert_clean_failure()

    def test_ambiguous_plan(self):
        self.plan.write_text(json.dumps({'manifest_sha256': self.old_digest, 'other': self.old_digest}))
        self.convert(False)
        self.assert_clean_failure()

    def test_unowned_plan_staging_preserved(self):
        staging = self.output_plan.with_name('new.plan.json.tmp')
        staging.write_bytes(b'belongs to another writer')
        self.convert(False)
        self.assertFalse(self.output.exists())
        self.assertEqual(staging.read_bytes(), b'belongs to another writer')
        self.assertFalse(self.output.with_name('new.bundle.tmp').exists())


if __name__ == '__main__':
    unittest.main()
