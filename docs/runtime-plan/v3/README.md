# RuntimePlan V3: local plaintext batches

V3 inherits V2 Release and reuse. It allows Host Encode in execution and adds
`{"ordinal": 123, "kind": "fence"}` with no inputs or outputs. Fence is valid
only in execution. V1/V2 keep their phase restrictions. V3 requires world_size=1;
it is not an MPI/NCCL barrier protocol.

Fence completes current communication groups, publishes pending results, and
calls Api::drain before the next batch. Completed groups are discarded only
after Pending references have been replaced. Cross-batch values, use counts and
final outputs remain intact. Release still means last logical use, not device
completion. AllValuesAfterRun remains incompatible with Release/reuse.

Sequential execution and per-device workers support V3. Worker execution uses
one serial CPU queue for Encode and Host compute, including decrypt/re-encrypt
Boot; device queues handle local transfers and computation. A Fence joins the
current queues, delivers communication results and drains the API. Worker failure
wakes dependency waiters. Task and communication handles are dropped per batch.

V3 bundle loading reads the manifest and checks local Encode references in both
initialization and execution. Blobs are read and verified when Encode runs, with
no raw-data cache. File length is checked before allocation; content digest,
finite values, capacity and negative-zero normalization remain mandatory.
V1/V2 keep eager loading. Runtime ValueDesc lookup uses an index.

DaCapo exports V3 with `--runtime-plan-plaintext-schedule=stream` and explicit
positive `--runtime-plan-{gpu,host,pinned,raw}-budget-bytes`. GPU budget is per
card, after the caller reserves space for keys, pools and other unmodeled memory.
Host budget includes encoded plaintexts and Host ciphertexts. Pinned budget
includes Host/Device transfer staging; raw budget covers one file buffer plus
its decoded double vector. `--runtime-plan-workspace-bytes-per-op` adds a
cumulative reservation to each GPU compute in the batch. Zero means unmeasured,
not zero actual workspace. `--runtime-plan-prefetch-bytes=0` prepares each weight
at its consumer; a positive bound moves a prefix of batch preparations ahead of
independent computation. It is a byte lookahead, not a measured latency model.

The common export pipeline runs PlanPlaintextStreaming after placement and
communication, then reruns Release/reuse planning. Encode CSE must not run after
streaming. Shared weights get distinct ValueIds/TransferIds when re-encoded in
later batches. Ciphertext computation and communication ordering is preserved.
Only direct Host Encode uploads are supported; unsupported plaintext final
outputs, multirank plans and indivisible over-budget operations fail explicitly.

`*.streaming.json` reports the conservative batch bound: retained objects at the
batch start plus every new allocation until Fence. It does not subtract Release
before Fence, and conservatively charges allocations later removed by reuse.
`*.memory.json` remains the older sequential-completion RNS estimate. Neither is
a total device memory guarantee: key/parameter storage, unknown temporary
workspaces, encoder workspace, allocator pools and fragmentation are excluded.
The runtime consumes the compiler schedule; it does not impose a dynamic quota
on arbitrary handwritten V3 plans. Actual upload overlap needs CUDA trace
measurement; existing allocation readiness waits are preserved.

RuntimeTiming exposes Encode/read counts, bytes and wall time, the no-cache raw
read upper bound, and Fence count/wall time. In worker mode Fence time includes
executing and joining the batch queues; in sequential mode it measures the final
communication wait and API drain. These timings must not be summed as disjoint
GPU execution intervals. POSEIDON_RUNTIME_TRACE emits per-value Encode/read and
Fence records; POSEIDON_THREAD_TRACE additionally records upload packing spans.
