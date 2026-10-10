#!/usr/bin/env python3
"""Merge a compiler-generated V1 bundle without recomputing blob hashes."""
import argparse
import hashlib
import json
import mmap
import os
import re
import shutil
import sys
import time
from pathlib import Path


def unique_members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate manifest key: ' + key)
        result[key] = value
    return result


def patch_plan(source, destination, old_digest, new_digest):
    # Compiler plans contain a unique, unescaped manifest_sha256 field. Copy
    # their bytes unchanged, then update just this same-length reference.
    with source.open('rb') as src, destination.open('wb') as dst:
        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
    with destination.open('r+b') as output, mmap.mmap(output.fileno(), 0) as data:
        needle = old_digest.encode('ascii')
        position = data.find(needle)
        if position < 0 or data.find(needle, position + len(needle)) >= 0:
            raise ValueError('plan must contain exactly one original manifest digest')
        prefix = data[max(0, position - 256):position]
        if not re.search(rb'"manifest_sha256"\s*:\s*"$', prefix):
            raise ValueError('unsupported compiler plan manifest reference')
        data[position:position + len(needle)] = new_digest.encode('ascii')
        data.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output-plan', type=Path)
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    if bool(args.plan) != bool(args.output_plan):
        parser.error('--plan and --output-plan must be specified together')
    if args.output.exists() or (args.output_plan and args.output_plan.exists()):
        raise FileExistsError('output already exists; use a new path')
    start = time.monotonic()
    raw = (args.bundle / 'manifest.json').read_bytes()
    old_digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw, object_pairs_hook=unique_members)
    del raw
    if set(manifest) != {'bundle_format_version', 'bundle_id', 'version', 'blobs'} or type(manifest['bundle_format_version']) is not int or manifest['bundle_format_version'] != 1:
        raise ValueError('expected a V1 loose-file bundle manifest')
    if not isinstance(manifest['bundle_id'], str) or not manifest['bundle_id'] or type(manifest['version']) is not int or not 0 < manifest['version'] <= (1 << 31) - 1:
        raise ValueError('invalid bundle identity')
    if not isinstance(manifest['blobs'], list):
        raise ValueError('blobs must be an array')
    staging = args.output.with_name(args.output.name + '.tmp')
    plan_temp = args.output_plan.with_name(args.output_plan.name + '.tmp') if args.output_plan else None
    staging.mkdir()
    published = False
    plan_owned = False
    try:
        offset = 0
        seen = set()
        with (staging / 'data.bin').open('xb', buffering=8 * 1024 * 1024) as output:
            for index, entry in enumerate(manifest['blobs']):
                if set(entry) != {'content', 'byte_length'}:
                    raise ValueError('invalid blob fields')
                content, length = entry['content'], entry['byte_length']
                if not isinstance(content, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', content) or content in seen:
                    raise ValueError('invalid or duplicate content ID')
                if type(length) is not int or length < 8 or length % 8 or offset + length > (1 << 53) - 1:
                    raise ValueError('invalid blob length')
                seen.add(content)
                path = args.bundle / 'data' / (content[7:] + '.bin')
                if path.stat().st_size != length:
                    raise ValueError('blob byte length mismatch: ' + str(path))
                with path.open('rb') as source:
                    remaining = length
                    while remaining:
                        block = source.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError('truncated blob: ' + str(path))
                        output.write(block)
                        remaining -= len(block)
                    if source.read(1):
                        raise ValueError('blob length changed: ' + str(path))
                entry['offset'] = offset
                offset += length
                if args.progress and (index + 1) % 100000 == 0:
                    print(f'packed {index + 1}/{len(manifest["blobs"])} blobs, {offset} bytes', file=sys.stderr, flush=True)
        manifest['bundle_format_version'] = 2
        manifest['pack_byte_length'] = offset
        digest = hashlib.sha256()
        with (staging / 'manifest.json').open('xb', buffering=1024 * 1024) as output:
            def write(data):
                output.write(data)
                digest.update(data)
            metadata = {key: value for key, value in manifest.items() if key != 'blobs'}
            write(json.dumps(metadata, separators=(',', ':'), allow_nan=False).encode()[:-1] + b',"blobs":[')
            for index, entry in enumerate(manifest['blobs']):
                if index:
                    write(b',')
                write(json.dumps(entry, separators=(',', ':'), allow_nan=False).encode())
            write(b']}\n')
        new_digest = 'sha256:' + digest.hexdigest()
        merge_seconds = time.monotonic() - start
        patch_start = time.monotonic()
        if args.plan:
            with plan_temp.open('xb'):
                pass
            plan_owned = True
            patch_plan(args.plan, plan_temp, old_digest, new_digest)
        patch_seconds = time.monotonic() - patch_start
        os.rename(staging, args.output)
        published = True
        if args.plan:
            os.replace(plan_temp, args.output_plan)
            plan_owned = False
        print(json.dumps({'bundle': str(args.output), 'plan': str(args.output_plan) if args.plan else None,
                          'blob_count': len(manifest['blobs']), 'pack_bytes': offset,
                          'merge_seconds': merge_seconds, 'plan_patch_seconds': patch_seconds,
                          'blob_hashes_recomputed': 0,
                          'reference': {'id': manifest['bundle_id'], 'version': manifest['version'],
                                        'manifest_sha256': new_digest}}))
    except BaseException:
        if published:
            shutil.rmtree(args.output)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if plan_owned and plan_temp.exists():
            plan_temp.unlink()


if __name__ == '__main__':
    main()
