#!/usr/bin/env python3
"""重新生成协议测试样例集(profiles + testdata)。

用法: python3 generate.py  (在本目录下运行)

指纹按 RuntimePlan V1 规范第 8 节计算:删除顶层 fingerprint 字段后,
按 JCS 等价的 canonical form(sort_keys、无空白、ensure_ascii=False)
序列化为 UTF-8,取 SHA-256。协议中所有数字都是整数,无需浮点规范化。

注意:invalid/ 中的个别文件在生成后做了刻意破坏(见 mutate 部分),
其中 i008 的指纹故意不符,其余文件的指纹都是对其(错误)内容的正确指纹,
保证每份文件恰好只有一个错误。
"""

import copy
import hashlib
import json
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES = HERE.parent.parent.parent / "operator-spec" / "v1" / "profiles"


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(obj) -> str:
    stripped = {k: v for k, v in obj.items() if k != "fingerprint"}
    digest = hashlib.sha256(canonical(stripped).encode("utf-8")).hexdigest()
    return "sha256:" + digest


def with_fingerprint(obj):
    obj = copy.deepcopy(obj)
    obj["fingerprint"] = fingerprint(obj)
    return obj


def write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


# ---------------------------------------------------------------- profiles

LEVEL_UPPER = 13
TABLE_LEN = LEVEL_UPPER + 1


def latency_table(base: int, step: int):
    return [base + step * i for i in range(TABLE_LEN)]


def op_entry(base: int, step: int):
    return {
        "supported": True,
        "latency_us_by_level": latency_table(base, step),
        "noise_by_level": None,
    }


def make_operators(scale: int, rescale_max_levels: int):
    return {
        "add_cc": op_entry(50 * scale, 60 * scale),
        "add_cp": op_entry(40 * scale, 50 * scale),
        "sub_cc": op_entry(50 * scale, 60 * scale),
        "sub_cp": op_entry(40 * scale, 50 * scale),
        "mul_cc": op_entry(4000 * scale, 6000 * scale),
        "mul_cp": op_entry(200 * scale, 180 * scale),
        "negate": op_entry(30 * scale, 40 * scale),
        "rotate": op_entry(3800 * scale, 5500 * scale),
        "rescale": {
            "supported": True,
            "max_levels_per_op": rescale_max_levels,
            "latency_us_by_level": latency_table(1900 * scale, 900 * scale),
            "noise_by_level": None,
        },
        "mod_switch": op_entry(48 * scale, 60 * scale),
        "relinearize": op_entry(3500 * scale, 5000 * scale),
        "boot": {"supported": True},
    }


BOOT_EMULATION_PROFILE = {
    "profile_id": "poseidon-cpu-boot-emulation-v1",
    "implementation": "decrypt_reencrypt",
    "input_level_min": 2,
    "input_level_max": LEVEL_UPPER,
    "input_components": 2,
    "output_level": 12,
    "output_scale_log2": 40,
    "output_components": 2,
    "latency_us": 2500000,
    "host_requirements": {"needs_secret_key": True, "needs_host_compute": True},
}

cpu_spec = with_fingerprint({
    "spec_format_version": 1,
    "spec_id": "poseidon-ckks-cpu-v1",
    "version": 1,
    "status": "placeholder",
    "target_id": "poseidon-ckks-cpu",
    "rescale_mode": "eager",
    "context": {
        "context_id": "ctx-main",
        "poly_degree": 32768,
        "rns_moduli_log2": [60] + [40] * 12 + [60],
        "max_modulus_log2": 60,
        "default_scale_log2": 40,
    },
    "levels": {"lower_bound": 2, "upper_bound": LEVEL_UPPER},
    "operators": make_operators(scale=1, rescale_max_levels=1),
    "boot_profiles": [BOOT_EMULATION_PROFILE],
})

gpu_operators = make_operators(scale=1, rescale_max_levels=2)
for entry in gpu_operators.values():
    table = entry.get("latency_us_by_level")
    if table is not None:
        entry["latency_us_by_level"] = [max(1, v // 10) for v in table]

gpu_spec = with_fingerprint({
    "spec_format_version": 1,
    "spec_id": "poseidon-ckks-gpu-v1",
    "version": 1,
    "status": "placeholder",
    "target_id": "poseidon-ckks-gpu",
    "rescale_mode": "lazy",
    "context": {
        "context_id": "ctx-main",
        "poly_degree": 32768,
        "rns_moduli_log2": [28] * TABLE_LEN,
        "max_modulus_log2": 28,
        "default_scale_log2": 40,
    },
    "levels": {"lower_bound": 2, "upper_bound": LEVEL_UPPER},
    "operators": gpu_operators,
    "boot_profiles": [BOOT_EMULATION_PROFILE],
})

write(PROFILES / "poseidon-ckks-cpu.v1.json", cpu_spec)
write(PROFILES / "poseidon-ckks-gpu.v1.json", gpu_spec)

CPU_REF = {"id": cpu_spec["spec_id"], "version": 1, "fingerprint": cpu_spec["fingerprint"]}
GPU_REF = {"id": gpu_spec["spec_id"], "version": 1, "fingerprint": gpu_spec["fingerprint"]}


# ---------------------------------------------------------------- helpers

def host(rank):
    return {"kind": "host", "rank": rank}


def dev(rank, index):
    return {"kind": "device", "rank": rank, "index": index}


def value(vid, place, kind="ciphertext", level=5, scale_log2=40, ntt=True, components=2):
    return {
        "id": vid, "kind": kind, "place": place, "context": "ctx-main",
        "level": level, "scale_log2": scale_log2, "ntt": ntt, "components": components,
    }


def compute(ordinal, op, place, inputs, output, attrs=None):
    ins = {"ordinal": ordinal, "kind": "compute", "op": op, "place": place,
           "inputs": inputs, "output": output}
    if attrs is not None:
        ins["attrs"] = attrs
    return ins


def comm(ordinal, kind, transfer_id, inputs, outputs, sources, destinations,
         output_kinds, hint="auto"):
    return {"ordinal": ordinal, "kind": kind, "transfer_id": transfer_id, "hint": hint,
            "inputs": inputs, "outputs": outputs, "sources": sources,
            "destinations": destinations, "output_kinds": output_kinds}


def plan(plan_id, target, values, external_inputs, required_keys,
         execution, final_outputs, initialization=None, finalization=None,
         plaintext_bundle=None):
    document = {
        "format_version": 1,
        "plan_id": plan_id,
        "target": target,
        "values": values,
        "external_inputs": external_inputs,
        "required_keys": required_keys,
        "initialization": initialization or [],
        "execution": execution,
        "finalization": finalization or [],
        "final_outputs": final_outputs,
    }
    if plaintext_bundle is not None:
        document["plaintext_bundle"] = plaintext_bundle
    return with_fingerprint(document)


def cpu_target(**kw):
    base = {"target_id": "poseidon-ckks-cpu", "capability_version": 1,
            "operator_spec": CPU_REF, "rescale_mode": "eager", "boot_mode": "native",
            "world_size": 1, "device_counts": [1], "required_capabilities": []}
    return base | kw


def gpu_target(**kw):
    base = {"target_id": "poseidon-ckks-gpu", "capability_version": 1,
            "operator_spec": GPU_REF, "rescale_mode": "lazy", "boot_mode": "native",
            "world_size": 1, "device_counts": [1], "required_capabilities": []}
    return base | kw


# ---------------------------------------------------------------- valid plans
# IO-2:external_inputs 只允许 Host,进设备一律走显式 transfer(通常放 initialization)。

d00 = dev(0, 0)
h0 = host(0)

v001 = plan(
    "1", cpu_target(required_capabilities=["transfer"]),
    values=[value("1", h0), value("2", d00), value("3", d00)],
    external_inputs=["1"], required_keys=[],
    initialization=[comm(0, "transfer", "1", ["1"], ["2"], [h0], [d00], ["ciphertext"])],
    execution=[compute(1, "add_cc", d00, ["2", "2"], "3")],
    final_outputs=["3"],
)

d01 = dev(0, 1)
v002 = plan(
    "2", gpu_target(device_counts=[2], required_capabilities=["transfer"]),
    values=[
        value("1", h0),
        value("2", d00),
        value("3", d00, scale_log2=80, components=3),
        value("4", d00, scale_log2=80),
        value("5", d00, level=4),
        value("6", d01, level=4),
    ],
    external_inputs=["1"],
    required_keys=[{"kind": "relin", "place": d00}],
    initialization=[comm(0, "transfer", "100", ["1"], ["2"], [h0], [d00], ["ciphertext"])],
    execution=[
        compute(1, "mul_cc", d00, ["2", "2"], "3"),
        compute(2, "relinearize", d00, ["3"], "4"),
        compute(3, "rescale", d00, ["4"], "5",
                {"target_level": 4, "target_scale_log2": 40}),
        comm(4, "transfer", "101", ["5"], ["6"], [d00], [d01], ["ciphertext"]),
    ],
    final_outputs=["6"],
)

d10 = dev(1, 0)
v003 = plan(
    "3", gpu_target(world_size=2, device_counts=[1, 1],
                    required_capabilities=["transfer", "replicate"]),
    values=[value("1", h0), value("2", d00), value("3", d10), value("4", host(1))],
    external_inputs=["1"], required_keys=[],
    initialization=[comm(0, "transfer", "1", ["1"], ["2"], [h0], [d00], ["ciphertext"])],
    execution=[
        comm(1, "replicate", "7", ["2"], ["3", "4"], [d00], [d10, host(1)],
             ["ciphertext", "ciphertext"], hint="broadcast"),
    ],
    final_outputs=["3", "4"],
)

BOOT_ATTRS = {"target_level": 12, "target_scale_log2": 40, "target_components": 2,
              "operator_profile": "poseidon-cpu-boot-emulation-v1",
              "implementation": "decrypt_reencrypt"}
v004 = plan(
    "4", gpu_target(boot_mode="decrypt_reencrypt",
                    required_capabilities=["transfer", "host_compute",
                                           "boot_decrypt_reencrypt"]),
    values=[
        value("1", h0, level=2),
        value("2", d00, level=2),
        value("3", h0, level=2),
        value("4", h0, level=12),
        value("5", d00, level=12),
    ],
    external_inputs=["1"],
    required_keys=[{"kind": "secret", "place": h0}],
    initialization=[comm(0, "transfer", "1", ["1"], ["2"], [h0], [d00], ["ciphertext"])],
    execution=[
        comm(1, "transfer", "2", ["2"], ["3"], [d00], [h0], ["ciphertext"]),
        compute(2, "boot", h0, ["3"], "4", BOOT_ATTRS),
        comm(3, "transfer", "3", ["4"], ["5"], [h0], [d00], ["ciphertext"]),
    ],
    final_outputs=["5"],
)

# -------- v005:引用明文数据包(权重包)的推理片段 --------

BUNDLE_DIR = HERE / "bundles" / "v005-demo"
# 数据文件是编码前的浮点 slot 向量:小端 float64 数组(这里是 8 个 slot 的 0/1 mask)。
BUNDLE_DATA = struct.pack("<8d", 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
BUNDLE_CONTENT = hashlib.sha256(BUNDLE_DATA).hexdigest()

bundle_manifest = with_fingerprint({
    "bundle_format_version": 1,
    "bundle_id": "v005-demo-weights",
    "version": 1,
    "context": "ctx-main",
    "entries": [
        {"value_id": "1", "name": "demo.mask",
         "content": "sha256:" + BUNDLE_CONTENT, "byte_length": len(BUNDLE_DATA)},
    ],
})
data_path = BUNDLE_DIR / "data" / (BUNDLE_CONTENT + ".bin")
data_path.parent.mkdir(parents=True, exist_ok=True)
data_path.write_bytes(BUNDLE_DATA)
print(f"wrote {data_path}")
write(BUNDLE_DIR / "manifest.json", bundle_manifest)

BUNDLE_REF = {"id": bundle_manifest["bundle_id"], "version": 1,
              "fingerprint": bundle_manifest["fingerprint"]}

v005 = plan(
    "5", cpu_target(required_capabilities=["transfer"]),
    values=[
        value("1", h0, kind="plaintext", components=1),
        value("2", h0),
        value("3", d00, kind="plaintext", components=1),
        value("4", d00),
        value("5", d00, scale_log2=80),
    ],
    external_inputs=["1", "2"], required_keys=[],
    initialization=[
        comm(0, "transfer", "1", ["1"], ["3"], [h0], [d00], ["plaintext"]),
        comm(1, "transfer", "2", ["2"], ["4"], [h0], [d00], ["ciphertext"]),
    ],
    execution=[compute(2, "mul_cp", d00, ["4", "3"], "5")],
    final_outputs=["5"],
    plaintext_bundle=BUNDLE_REF,
)

VALID = {
    "v001_minimal_single_device.json": v001,
    "v002_mul_rescale_transfer.json": v002,
    "v003_replicate_multi_rank.json": v003,
    "v004_host_boot_emulation.json": v004,
    "v005_plaintext_bundle.json": v005,
}
for name, obj in VALID.items():
    write(HERE / "valid" / name, obj)


# ---------------------------------------------------------------- invalid plans

def refingerprint(obj):
    obj = {k: v for k, v in obj.items() if k != "fingerprint"}
    return with_fingerprint(obj)


invalid = {}

p = copy.deepcopy(v001)
p["format_version"] = 2
invalid["i001_unknown_format_version.json"] = refingerprint(p)

p = copy.deepcopy(v001)
p["values"][0]["scale_log2"] = 40.5
invalid["i002_float_scale_log2.json"] = refingerprint(p)

p = copy.deepcopy(v001)
p["values"].insert(1, copy.deepcopy(p["values"][0]))
invalid["i003_duplicate_value_id.json"] = refingerprint(p)

p = copy.deepcopy(v003)
p["execution"][0]["output_kinds"] = ["ciphertext"]  # outputs/destinations 仍是 2 项
invalid["i004_comm_list_length_mismatch.json"] = refingerprint(p)

p = copy.deepcopy(v004)
p["target"]["boot_mode"] = "native"  # 指令仍是 decrypt_reencrypt
invalid["i005_boot_mode_mismatch.json"] = refingerprint(p)

p = copy.deepcopy(v004)
p["values"][3]["level"] = 11  # boot attrs 仍声明 target_level 12
invalid["i006_boot_target_desc_mismatch.json"] = refingerprint(p)

p = copy.deepcopy(v001)
p["values"].append(value("4", d00))
p["execution"] = [
    compute(1, "add_cc", d00, ["4", "4"], "3"),  # 使用尚未定义的 4
    compute(2, "negate", d00, ["2"], "4"),
]
invalid["i007_use_before_define.json"] = refingerprint(p)

p = copy.deepcopy(v001)  # 指纹与内容不符(翻转末位十六进制)
fp = p["fingerprint"]
p["fingerprint"] = fp[:-1] + ("0" if fp[-1] != "0" else "1")
invalid["i008_bad_fingerprint.json"] = p

p = copy.deepcopy(v001)
p["comment"] = "unknown field must be rejected"
invalid["i009_unknown_field.json"] = refingerprint(p)

p = copy.deepcopy(v001)
p["values"][1]["place"] = h0  # 使用它的指令 place 仍是 device(0,0)
invalid["i010_place_mismatch.json"] = refingerprint(p)

p = copy.deepcopy(v004)
p["required_keys"] = []  # decrypt_reencrypt boot 需要 host secret key
invalid["i011_missing_required_key.json"] = refingerprint(p)

p = copy.deepcopy(v001)
fp = p["target"]["operator_spec"]["fingerprint"]
p["target"]["operator_spec"]["fingerprint"] = fp[:-1] + ("0" if fp[-1] != "0" else "1")
invalid["i012_operator_spec_fingerprint_mismatch.json"] = refingerprint(p)

# 外部输入直接声明在 Device 上,没有任何上传指令(违反 IO-2)
invalid["i013_device_external_input.json"] = plan(
    "13", cpu_target(),
    values=[value("1", d00), value("2", d00)],
    external_inputs=["1"], required_keys=[],
    execution=[compute(0, "add_cc", d00, ["1", "1"], "2")],
    final_outputs=["2"],
)

p = copy.deepcopy(v005)
fp = p["plaintext_bundle"]["fingerprint"]
p["plaintext_bundle"]["fingerprint"] = fp[:-1] + ("0" if fp[-1] != "0" else "1")
invalid["i014_plaintext_bundle_fingerprint_mismatch.json"] = refingerprint(p)

for name, obj in invalid.items():
    write(HERE / "invalid" / name, obj)

print("done")
