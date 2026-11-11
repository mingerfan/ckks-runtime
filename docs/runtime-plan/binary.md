# RuntimePlan binary format

This is the current binary file contract. Container V1 is independent of semantic RuntimePlan V1–V5, V7 and V8; V6 remains reserved and is rejected. See the [RuntimePlan contract](README.md), [V7 ApiCall](v7.md) and [V8 Buffer](v8.md). The compiler generates binary plans by default. Select JSON when inspecting instructions:

```sh
hecate-opt ... --runtime-plan-format=binary
hecate-opt ... --runtime-plan-format=json --runtime-plan-pretty=true
```

The corresponding `emit-runtime-plan` pass option is `plan-format=binary|json` (default binary). Binary export requires `bundle-format=pack` and `pretty=false`. Output files are `<prefix>.<function>.runtime-plan.bin`, `<prefix>.<function>.bundle/manifest.bin`, and `data.bin`. JSON mode uses `.runtime-plan.json` and `manifest.json`. Use a separate prefix for each representation: existing bundles are only reused when the requested manifest and payloads match. The model artifact generator accepts `--plan-format=binary|json`, default binary, and reads only the small binary metadata header when reporting artifacts.

`RuntimePlanReader::read_file` accepts `.bin` or `.json`. The binary reader reads the entire file, scans record boundaries and allocates arrays, then decodes independent blocks into those arrays with four threads by default. Set `PlanReadOptions.threads`, or `CKKS_RUNTIME_PLAN_READ_THREADS=1..64`. It releases the temporary input buffer after construction. `PlanVerifier` then applies the same ordered lifecycle and semantic checks as JSON. This does not parallelize execution or make the plan partially resident. CPU, GPU, and MPI Poseidon execution tools use this reader.

## Header and metadata

Both files start with the following fields. All integers, signed fields, and IEEE float64 bits are explicitly little-endian; no C++ memory layouts are serialized.

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | 8 bytes | `CKKSPL01` for a plan, `CKKSMF01` for a manifest |
| 8 | u32 | Container version, currently `1` |
| 12 | u32 | Metadata byte length |
| 16 | UTF-8 JSON bytes | Small metadata object, at most 1 MiB |

There is no literal `METADATA` marker. The header gives its position and length. Unknown magic, version, tags, enums, malformed metadata, invalid counts, truncated records, and trailing bytes are errors. The binary container version is independent of RuntimePlan semantic `format_version` (V1–V5, V7 or V8) and bundle storage version (V2 pack). A binary file does not become JSON when decoding fails.

Plan metadata uses the existing JSON root schema with empty arrays for the six record sections. This metadata-only skeleton is parsed separately, not as a complete plan: the V8 feature requirement is checked on the full plan after its records are decoded. Metadata carries `format_version`, decimal `plan_id`, `target` (including OperatorSpec reference), and optional bundle reference. A binary bundle reference is:

```json
{"id":"model-plaintexts","version":1,"manifest_format":"binary"}
```

It selects exactly `manifest.bin`; there is no manifest source digest. Legacy JSON bundle references retain `manifest_sha256` and select `manifest.json`. Bundle ID and version must match the manifest in both modes.

Binary loading does not compute a whole-file SHA-256 or rehash blob payloads. It checks metadata, record fields, ranges, exact pack file size, slot capacity and finite float64 values. OperatorSpec still uses its existing small-file digest. For the existing API's MPI preflight, the reader supplies a digest of the small canonical metadata only, separately from `LoadedRuntimePlan.source_sha256` (empty for binary). That preflight compares metadata across ranks; it does not certify equality of every instruction or payload byte. JSON keeps its existing source digest behavior.

## Plan records

After the header: `u64 string_count`, then strings encoded as `u32 byte_length` plus UTF-8 bytes. Context names, bundle content IDs, Boot profiles, ApiCall callees and V8 buffer layouts refer to this table by u32 index. Intern buffer layout strings, including the empty string used by V8 CT/PT descriptors.

Six arrays follow in order: values, external inputs, initialization, execution, finalization, final outputs. Each starts with u64 record count. Input/output arrays contain u64 ValueIds. A V1–V4 descriptor is exactly 35 bytes: u64 ID, u8 kind, place (u8 kind, i32 rank, i32 index), u32 context index, i32 level, i32 scale_log2, u8 NTT boolean, i32 components.

V5 and V7 descriptors are exactly 39 bytes: `scale_log2` is IEEE float64 instead of i32. V7 keeps V5's scale representation unchanged. These older record widths do not change.

### V8 ValueDesc record

Every V8 ValueDesc is exactly **51 bytes**, regardless of value kind. Its first 39 bytes are the unchanged V5/V7 prefix; the final 12 bytes are `u64 buffer_bytes` and `u32 buffer_layout_string_index`, both little-endian.

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u64 | ValueId |
| 8 | u8 | Value kind |
| 9 | 9 bytes | Place: u8 kind, i32 rank, i32 index |
| 18 | u32 | Context string-table index |
| 22 | i32 | level |
| 26 | float64 | scale_log2 |
| 34 | u8 | NTT boolean |
| 35 | i32 | components |
| 39 | u64 | buffer_bytes |
| 47 | u32 | buffer_layout string-table index |

Buffer uses value-kind wire code **2**. Its context must exist in the same OperatorSpec, its layout string must be non-empty, and its prefix must have `level=0`, `scale_log2=0`, `ntt=false`, `components=0`. A byte count of zero is valid.

V8 Ciphertext and Plaintext records also carry the suffix: `buffer_bytes=0` and a reference to the interned **empty string**. The layout field is a normal string-table reference, not an absent-field sentinel. JSON is intentionally different: CT/PT retain their existing shape and reject both buffer fields. A JSON Buffer requires `buffer_bytes` as a canonical uint64 decimal **string** and a non-empty `buffer_layout`; see [V8](v8.md).

A complete JSON or binary plan uses semantic V8 if and only if it contains a Buffer or `external_inputs` references a Device Ciphertext. Readers, writers and the verifier reject V8 when neither condition holds, and reject either feature in pre-V8 plans. The compiler selects V8 for either feature and retains the existing version-selection rules otherwise. A bufferless Device-external plan still uses 51-byte V8 value records, each with the zero/empty CT/PT suffix. Internal Device values alone do not select V8.

V8 external inputs may name ciphertexts already resident on their declared Device; binding validates the supplied object and does not insert a transfer. This adds no binary fields: external input arrays remain u64 ValueIds. V1–V5/V7 external inputs and all plaintext external inputs remain Host-only; Buffer external inputs remain forbidden. See [V8](v8.md).

### Instruction records

Instructions start with u64 ordinal and u8 tag:

| Tag | Body fields, in order |
| --- | --- |
| 0 Encode | u64 output; u8 payload tag; inline: u64 count + float64 values, or bundle: u32 content index |
| 1 Compute | u8 operator; u64 input count + u64 IDs; u64 output; place; u8 attribute tag + attributes; u8 reuse-present + optional u64 input index |
| 2 Communication | u64 transfer ID; u8 kind; u8 hint; inputs, outputs, sources, destinations, output kinds, each with u64 count |
| 3 Release | u64 ValueId |
| 4 Fence | no additional fields |
| 5 PreparePlaintext (V4+) | Same output and payload fields as Encode |

Compute operator codes 0–11: AddCC, AddCP, SubCC, SubCP, MulCC, MulCP, Negate, Rotate, Rescale, ModSwitch, Relinearize, Boot. Attribute tags 0–4: none, Rotate (i32 steps), Rescale (two i32 targets), ModSwitch (i32 level), Boot (three i32 targets, u32 profile index, u8 implementation). For V5, V7 and V8, Rescale attributes are i32 target level followed by float64 target scale; Boot attributes keep their integer targets. Communication kind codes: Transfer 0, Replicate 1. Hint codes: Auto, PointToPoint, Broadcast, Tree, Ring, HostStaged. Value kinds: Plaintext 0, Ciphertext 1, Buffer 2 (V8 only); places: Host 0, Device 1; Boot implementation: Native 0, DecryptReencrypt 1. Boolean values must be 0 or 1. Strings are bounded to 1 MiB and individual typed arrays and plan files to 8 GiB.

### V7/V8 ApiCall record

ApiCall uses the existing Compute instruction tag **1**, operator code **13**, and attribute tag **6**. These codes and the attribute payload are unchanged in V8. Operator code **12** and attribute tag **5** are reserved for the separate V6 BootStep protocol; this reader rejects them, including in V7 and V8 plans. Instruction tag 5 above is unrelated and remains valid for PreparePlaintext.

The attribute payload is exactly **12 bytes**, excluding its u8 tag: `u32 callee_string_index`, then `u64 call_id`. The callee must be non-empty. JSON represents `call_id` as a canonical unsigned decimal **string**; binary stores its full uint64 value directly.

The input count may be zero. Output and place use the normal Compute fields. V7 requires ciphertext inputs and output; V8 permits Ciphertext or Buffer for each input and the single output, but never Plaintext. All inputs and the output retain the same Place and context. ApiCall cannot reuse an input: the common `reuse-present` byte must be 0, with no reuse index. Release records and ordered lifetime checks are unchanged; a shared Buffer can feed two suffix calls and be released after its last use.

Calls require semantic V7 or V8; they are not accepted in V1–V5. Buffer is produced and consumed only by ApiCall, never by Encode/Prepare, ordinary arithmetic, Transfer or Replicate, and cannot be an external input or final output. Numerical transformations and backend startup/value validation follow [V7](v7.md) and [V8](v8.md), not the ordinary arithmetic/Boot rules. A complete native Boot remains an opaque ApiCall, not BootStep or a decrypt/re-encrypt substitute.

## Manifest records

Metadata contains `bundle_format_version:2`, `bundle_id`, `version`, `pack_byte_length`, and an empty `blobs` array. Then u64 entry count and fixed 48-byte records: 32 raw content-ID digest bytes, u64 offset, u64 byte length. The digest bytes are identifiers generated during constant deduplication, not a request to rehash payloads when loading. Entries must be unique, aligned to eight bytes, have nonzero aligned lengths, remain within the pack, and collectively cover it with no overlaps or gaps.

Compiler output uses staging files and directories. A temporary record file lets the compiler place metadata and the string table first without retaining the entire plan in RAM. It reuses the JSON builder's IDs and records; it does not convert a previously generated giant JSON file. Temporary data is removed on failure and publication errors preserve an existing entry point.

The old `PLNEXP01`/`MNFEXP01` files remain experiment-only. `runtime_binary_io_experiment promote-plan` and `promote-manifest` migrate existing experimental data without whole-file hashing; those tools are for controlled benchmark preparation, not the compiler's production publication path.
