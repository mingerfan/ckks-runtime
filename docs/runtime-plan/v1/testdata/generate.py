#!/usr/bin/env python3

import copy
import hashlib
import json
import pathlib
import shutil
import struct

ROOT = pathlib.Path(__file__).resolve().parents[4]
TESTDATA = pathlib.Path(__file__).resolve().parent
PROFILES = ROOT / "docs" / "operator-spec" / "v1" / "profiles"


def encoded(document):
    return (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(document))


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def place_host(rank):
    return {"kind": "host", "rank": rank}


def place_device(rank, index=0):
    return {"kind": "device", "rank": rank, "index": index}


def value(value_id, kind, place, level, scale, components=None):
    return {
        "id": str(value_id),
        "kind": kind,
        "place": place,
        "context": "ctx-main",
        "level": level,
        "scale_log2": scale,
        "ntt": True,
        "components": components if components is not None else (1 if kind == "plaintext" else 2),
    }


def transfer(ordinal, transfer_id, input_id, output_id, source, destination, kind, hint="auto"):
    return {
        "ordinal": ordinal,
        "kind": "transfer",
        "transfer_id": str(transfer_id),
        "hint": hint,
        "inputs": [str(input_id)],
        "outputs": [str(output_id)],
        "sources": [source],
        "destinations": [destination],
        "output_kinds": [kind],
    }


def compute(ordinal, op, place, inputs, output, attrs=None):
    result = {
        "ordinal": ordinal,
        "kind": "compute",
        "op": op,
        "place": place,
        "inputs": [str(item) for item in inputs],
        "output": str(output),
    }
    if attrs is not None:
        result["attrs"] = attrs
    return result


def encode_inline(ordinal, output, values):
    return {"ordinal": ordinal, "kind": "encode",
            "payload": {"kind": "inline", "values": values}, "output": str(output)}


def encode_bundle(ordinal, output, content):
    return {"ordinal": ordinal, "kind": "encode",
            "payload": {"kind": "bundle", "content": content}, "output": str(output)}


def operator_entry(levels):
    return {"supported": True, "latency_us_by_level": list(range(levels)), "noise_by_level": None}


def make_profile(spec_id, target_id, rescale_mode, modulus_bits, max_rescale):
    level_count = 14
    operators = {name: operator_entry(level_count) for name in (
        "add_cc", "add_cp", "sub_cc", "sub_cp", "mul_cc", "mul_cp",
        "negate", "rotate", "mod_switch", "relinearize")}
    operators["rescale"] = operator_entry(level_count)
    operators["rescale"]["max_levels_per_op"] = max_rescale
    operators["boot"] = {"supported": True}
    return {
        "spec_format_version": 1,
        "spec_id": spec_id,
        "version": 1,
        "status": "placeholder",
        "target_id": target_id,
        "rescale_mode": rescale_mode,
        "context": {
            "context_id": "ctx-main",
            "poly_degree": 32768,
            "rns_moduli_log2": [modulus_bits] * level_count,
            "max_modulus_log2": modulus_bits,
            "default_scale_log2": 40,
        },
        "levels": {"lower_bound": 2, "upper_bound": 13},
        "operators": operators,
        "boot_profiles": [
            {
                "profile_id": "poseidon-native-boot-v1",
                "implementation": "native",
                "input_level_min": 2,
                "input_level_max": 13,
                "input_components": 2,
                "output_level": 12,
                "output_scale_log2": 40,
                "output_components": 2,
                "latency_us": 1000000,
                "host_requirements": {"needs_secret_key": False, "needs_host_compute": False},
            },
            {
                "profile_id": "poseidon-cpu-boot-emulation-v1",
                "implementation": "decrypt_reencrypt",
                "input_level_min": 2,
                "input_level_max": 13,
                "input_components": 2,
                "output_level": 12,
                "output_scale_log2": 40,
                "output_components": 2,
                "latency_us": 2500000,
                "host_requirements": {"needs_secret_key": True, "needs_host_compute": True},
            },
        ],
    }


def target(name, spec_digests, world_size=1, device_counts=None):
    if device_counts is None:
        device_counts = [1] * world_size
    spec_id = f"poseidon-ckks-{name}-v1"
    return {
        "target_id": f"poseidon-ckks-{name}",
        "capability_version": 1,
        "operator_spec": {"id": spec_id, "version": 1, "source_sha256": spec_digests[spec_id]},
        "world_size": world_size,
        "device_counts": device_counts,
    }


def plan(plan_id, target_value, values, external_inputs, initialization, execution, finalization, final_outputs, bundle=None):
    result = {
        "format_version": 1,
        "plan_id": str(plan_id),
        "target": target_value,
        "values": values,
        "external_inputs": [str(item) for item in external_inputs],
        "initialization": initialization,
        "execution": execution,
        "finalization": finalization,
        "final_outputs": [str(item) for item in final_outputs],
    }
    if bundle is not None:
        result["plaintext_bundle"] = bundle
    return result


def make_artifacts():
    cpu = make_profile("poseidon-ckks-cpu-v1", "poseidon-ckks-cpu", "eager", 60, 1)
    gpu = make_profile("poseidon-ckks-gpu-v1", "poseidon-ckks-gpu", "lazy", 28, 2)
    write_json(PROFILES / "poseidon-ckks-cpu.v1.json", cpu)
    write_json(PROFILES / "poseidon-ckks-gpu.v1.json", gpu)
    spec_digests = {
        cpu["spec_id"]: digest((PROFILES / "poseidon-ckks-cpu.v1.json").read_bytes()),
        gpu["spec_id"]: digest((PROFILES / "poseidon-ckks-gpu.v1.json").read_bytes()),
    }

    bundle_dir = TESTDATA / "bundles" / "v005-demo"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    (bundle_dir / "data").mkdir(parents=True)
    raw_a = b"".join(struct.pack("<d", 0.0 if item == 0.0 else item) for item in [1.0, -0.0, 3.5, -2.0])
    raw_b = b"".join(struct.pack("<d", item) for item in [7.0, 8.0])
    content_a, content_b = digest(raw_a), digest(raw_b)
    (bundle_dir / "data" / (content_a[7:] + ".bin")).write_bytes(raw_a)
    (bundle_dir / "data" / (content_b[7:] + ".bin")).write_bytes(raw_b)
    manifest = {
        "bundle_format_version": 1,
        "bundle_id": "v005-demo-weights",
        "version": 1,
        "blobs": [
            {"content": content_a, "byte_length": len(raw_a)},
            {"content": content_b, "byte_length": len(raw_b)},
        ],
    }
    write_json(bundle_dir / "manifest.json", manifest)
    bundle_ref = {"id": manifest["bundle_id"], "version": manifest["version"],
                  "manifest_sha256": digest((bundle_dir / "manifest.json").read_bytes())}

    host0, host1 = place_host(0), place_host(1)
    dev0, dev1 = place_device(0), place_device(1)
    valid = {}
    valid["v001_inline_encode_host_compute.json"] = plan(
        1, target("cpu", spec_digests),
        [value(1, "ciphertext", host0, 5, 40), value(2, "plaintext", host0, 5, 40),
         value(3, "ciphertext", host0, 5, 40)],
        [1], [encode_inline(0, 2, [1, -0.0, 3.5, -2])],
        [compute(1, "add_cp", host0, [1, 2], 3)], [], [3])
    valid["v002_device_mul_relin_rescale_rotate.json"] = plan(
        2, target("gpu", spec_digests),
        [value(1, "ciphertext", host0, 5, 40), value(2, "ciphertext", dev0, 5, 40),
         value(3, "ciphertext", dev0, 5, 80, 3), value(4, "ciphertext", dev0, 5, 80),
         value(5, "ciphertext", dev0, 4, 40), value(6, "ciphertext", dev0, 4, 40),
         value(7, "ciphertext", host0, 4, 40)],
        [1], [transfer(0, 100, 1, 2, host0, dev0, "ciphertext")],
        [compute(1, "mul_cc", dev0, [2, 2], 3), compute(2, "relinearize", dev0, [3], 4),
         compute(3, "rescale", dev0, [4], 5, {"target_level": 4, "target_scale_log2": 40}),
         compute(4, "rotate", dev0, [5], 6, {"steps": -1})],
        [transfer(5, 101, 6, 7, dev0, host0, "ciphertext")], [7])
    valid["v003_bundle_multi_rank.json"] = plan(
        3, target("gpu", spec_digests, 2, [1, 1]),
        [value(1, "ciphertext", host0, 5, 40), value(2, "plaintext", host1, 5, 40),
         value(3, "ciphertext", dev0, 5, 40), value(4, "plaintext", dev1, 5, 40),
         value(5, "ciphertext", dev1, 5, 40), value(6, "ciphertext", host1, 5, 40),
         value(7, "ciphertext", dev1, 5, 80)],
        [1], [encode_bundle(0, 2, content_a),
              transfer(1, 200, 1, 3, host0, dev0, "ciphertext"),
              transfer(2, 201, 2, 4, host1, dev1, "plaintext")],
        [{"ordinal": 3, "kind": "replicate", "transfer_id": "202", "hint": "tree",
          "inputs": ["3"], "outputs": ["5", "6"], "sources": [dev0],
          "destinations": [dev1, host1], "output_kinds": ["ciphertext", "ciphertext"]},
         compute(4, "mul_cp", dev1, [5, 4], 7)], [], [6, 7], bundle_ref)
    valid["v004_host_boot.json"] = plan(
        4, target("gpu", spec_digests),
        [value(1, "ciphertext", host0, 2, 40), value(2, "ciphertext", host0, 12, 40)],
        [1], [], [compute(0, "boot", host0, [1], 2,
            {"target_level": 12, "target_scale_log2": 40, "target_components": 2,
             "operator_profile": "poseidon-cpu-boot-emulation-v1", "implementation": "decrypt_reencrypt"})], [], [2])
    valid["v005_bundle_reuse.json"] = plan(
        5, target("cpu", spec_digests),
        [value(1, "plaintext", host0, 5, 40), value(2, "plaintext", host0, 4, 20)],
        [], [encode_bundle(0, 1, content_a), encode_bundle(1, 2, content_a)], [], [], [1, 2], bundle_ref)
    valid["v006_native_boot_device.json"] = plan(
        6, target("gpu", spec_digests),
        [value(1, "ciphertext", host0, 2, 40), value(2, "ciphertext", dev0, 2, 40),
         value(3, "ciphertext", dev0, 12, 40)],
        [1], [transfer(0, 300, 1, 2, host0, dev0, "ciphertext")],
        [compute(1, "boot", dev0, [2], 3,
            {"target_level": 12, "target_scale_log2": 40, "target_components": 2,
             "operator_profile": "poseidon-native-boot-v1", "implementation": "native"})], [], [3])

    valid_dir = TESTDATA / "valid"
    invalid_dir = TESTDATA / "invalid"
    if valid_dir.exists(): shutil.rmtree(valid_dir)
    if invalid_dir.exists(): shutil.rmtree(invalid_dir)
    for name, document in valid.items(): write_json(valid_dir / name, document)

    invalid = {}
    invalid["i001_unknown_format_version.json"] = copy.deepcopy(valid["v001_inline_encode_host_compute.json"])
    invalid["i001_unknown_format_version.json"]["format_version"] = 2
    invalid["i002_float_scale_log2.json"] = copy.deepcopy(valid["v001_inline_encode_host_compute.json"])
    invalid["i002_float_scale_log2.json"]["values"][0]["scale_log2"] = 40.0
    invalid["i003_duplicate_value_id.json"] = copy.deepcopy(valid["v001_inline_encode_host_compute.json"])
    invalid["i003_duplicate_value_id.json"]["values"].append(copy.deepcopy(invalid["i003_duplicate_value_id.json"]["values"][0]))
    invalid["i004_comm_list_length_mismatch.json"] = copy.deepcopy(valid["v002_device_mul_relin_rescale_rotate.json"])
    invalid["i004_comm_list_length_mismatch.json"]["initialization"][0]["outputs"].append("7")
    invalid["i005_unknown_boot_profile.json"] = copy.deepcopy(valid["v004_host_boot.json"])
    invalid["i005_unknown_boot_profile.json"]["execution"][0]["attrs"]["operator_profile"] = "missing"
    invalid["i006_boot_target_desc_mismatch.json"] = copy.deepcopy(valid["v004_host_boot.json"])
    invalid["i006_boot_target_desc_mismatch.json"]["execution"][0]["attrs"]["target_level"] = 11
    invalid["i007_use_before_define.json"] = copy.deepcopy(valid["v002_device_mul_relin_rescale_rotate.json"])
    invalid["i007_use_before_define.json"]["execution"][0]["inputs"][0] = "7"
    invalid["i008_unknown_field.json"] = copy.deepcopy(valid["v001_inline_encode_host_compute.json"])
    invalid["i008_unknown_field.json"]["comment"] = "not allowed"
    invalid["i009_place_mismatch.json"] = copy.deepcopy(valid["v002_device_mul_relin_rescale_rotate.json"])
    invalid["i009_place_mismatch.json"]["execution"][0]["place"] = host0
    invalid["i010_device_external_input.json"] = copy.deepcopy(valid["v001_inline_encode_host_compute.json"])
    invalid["i010_device_external_input.json"]["values"][0]["place"] = dev0
    invalid["i011_operator_spec_digest_mismatch.json"] = copy.deepcopy(valid["v001_inline_encode_host_compute.json"])
    invalid["i011_operator_spec_digest_mismatch.json"]["target"]["operator_spec"]["source_sha256"] = "sha256:" + "0" * 64
    invalid["i012_manifest_digest_mismatch.json"] = copy.deepcopy(valid["v005_bundle_reuse.json"])
    invalid["i012_manifest_digest_mismatch.json"]["plaintext_bundle"]["manifest_sha256"] = "sha256:" + "0" * 64
    invalid["i013_encode_outside_initialization.json"] = copy.deepcopy(valid["v001_inline_encode_host_compute.json"])
    moved = invalid["i013_encode_outside_initialization.json"]["initialization"].pop()
    invalid["i013_encode_outside_initialization.json"]["execution"].insert(0, moved)
    invalid["i014_rotate_normalizes_to_zero.json"] = copy.deepcopy(valid["v002_device_mul_relin_rescale_rotate.json"])
    invalid["i014_rotate_normalizes_to_zero.json"]["execution"][3]["attrs"]["steps"] = 16384
    invalid["i015_unused_value_desc.json"] = copy.deepcopy(valid["v001_inline_encode_host_compute.json"])
    invalid["i015_unused_value_desc.json"]["values"].append(value(99, "plaintext", host0, 5, 40))
    for name, document in invalid.items(): write_json(invalid_dir / name, document)


if __name__ == "__main__":
    make_artifacts()
