#!/usr/bin/env python3
"""Generate long, deliberately parallel RuntimePlan V1 GPU probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def device(index: int) -> dict[str, int | str]:
    return {"kind": "device", "rank": 0, "index": index}


HOST = {"kind": "host", "rank": 0}


class PlanBuilder:
    def __init__(self, spec_path: Path, device_count: int, plan_id: int) -> None:
        spec_bytes = spec_path.read_bytes()
        spec = json.loads(spec_bytes)
        self.context = spec["context"]["context_id"]
        self.values: list[dict] = []
        self.initialization: list[dict] = []
        self.execution: list[dict] = []
        self.next_value_id = 0
        self.next_ordinal = 0
        self.next_transfer_id = 0
        self.plan = {
            "format_version": 1,
            "plan_id": str(plan_id),
            "target": {
                "target_id": spec["target_id"],
                "capability_version": 1,
                "operator_spec": {
                    "id": spec["spec_id"],
                    "version": spec["version"],
                    "source_sha256": "sha256:" + hashlib.sha256(spec_bytes).hexdigest(),
                },
                "world_size": 1,
                "device_counts": [device_count],
            },
            "values": self.values,
            "external_inputs": [],
            "initialization": self.initialization,
            "execution": self.execution,
            "finalization": [],
            "final_outputs": [],
        }

    def value(self, place: dict, level: int, scale_log2: int = 40) -> str:
        value_id = str(self.next_value_id)
        self.next_value_id += 1
        self.values.append(
            {
                "id": value_id,
                "kind": "ciphertext",
                "place": place,
                "context": self.context,
                "level": level,
                "scale_log2": scale_log2,
                "ntt": True,
                "components": 2,
            }
        )
        return value_id

    def instruction(self, phase: list[dict], body: dict) -> None:
        phase.append({"ordinal": self.next_ordinal, **body})
        self.next_ordinal += 1

    def transfer(self, phase: list[dict], source: str, source_place: dict,
                 destination_place: dict, level: int) -> str:
        output = self.value(destination_place, level)
        self.instruction(
            phase,
            {
                "kind": "transfer",
                "transfer_id": str(self.next_transfer_id),
                "hint": "point_to_point",
                "inputs": [source],
                "outputs": [output],
                "sources": [source_place],
                "destinations": [destination_place],
                "output_kinds": ["ciphertext"],
            },
        )
        self.next_transfer_id += 1
        return output

    def compute(self, op: str, inputs: list[str], place: dict, level: int,
                attrs: dict | None = None) -> str:
        output = self.value(place, level)
        body = {
            "kind": "compute",
            "op": op,
            "place": place,
            "inputs": inputs,
            "output": output,
        }
        if attrs is not None:
            body["attrs"] = attrs
        self.instruction(self.execution, body)
        return output

    def finish(self, output: str) -> dict:
        self.plan["final_outputs"] = [output]
        return self.plan


def rotate_chain(builder: PlanBuilder, value: str, gpu: int, level: int,
                 count: int) -> str:
    for _ in range(count):
        value = builder.compute(
            "rotate", [value], device(gpu), level, {"steps": 1}
        )
    return value


def build_single_gpu(spec_path: Path, segment_length: int,
                     communication_rounds: int, input_level: int,
                     compute_level: int) -> dict:
    builder = PlanBuilder(spec_path, device_count=1, plan_id=655360001)
    external = builder.value(HOST, input_level)
    builder.plan["external_inputs"] = [external]
    on_device = builder.transfer(
        builder.initialization, external, HOST, device(0), input_level
    )

    chains = []
    for _ in range(4):
        value = builder.compute(
            "mod_switch", [on_device], device(0), compute_level,
            {"target_level": compute_level},
        )
        chains.append(rotate_chain(builder, value, 0, compute_level, segment_length))

    for chain_index, value in enumerate(chains):
        for _ in range(communication_rounds):
            value = builder.compute(
                "add_cc", [value, value], device(0), compute_level
            )
        chains[chain_index] = rotate_chain(
            builder, value, 0, compute_level, segment_length
        )

    output = chains[0]
    for value in chains[1:]:
        output = builder.compute(
            "add_cc", [output, value], device(0), compute_level
        )
    return builder.finish(output)


def distributed_counts(total: int, buckets: int) -> list[int]:
    quotient, remainder = divmod(total, buckets)
    return [quotient + (index < remainder) for index in range(buckets)]


def build_four_gpu_burst(spec_path: Path, segment_length: int,
                         communication_rounds: int, input_level: int,
                         compute_level: int) -> dict:
    builder = PlanBuilder(spec_path, device_count=4, plan_id=655360004)
    external = builder.value(HOST, input_level)
    builder.plan["external_inputs"] = [external]
    device_inputs = [
        builder.transfer(
            builder.initialization, external, HOST, device(gpu), input_level
        )
        for gpu in range(4)
    ]

    chains = []
    for gpu in range(4):
        value = builder.compute(
            "mod_switch", [device_inputs[gpu]], device(gpu), compute_level,
            {"target_level": compute_level},
        )
        chains.append(
            rotate_chain(builder, value, gpu, compute_level, segment_length)
        )

    incoming: list[list[str]] = [[] for _ in range(4)]
    for round_index in range(communication_rounds):
        distance = round_index % 3 + 1
        for destination in range(4):
            source = (destination + distance) % 4
            incoming[destination].append(
                builder.transfer(
                    builder.execution,
                    chains[source],
                    device(source),
                    device(destination),
                    compute_level,
                )
            )

    for gpu in range(4):
        value = chains[gpu]
        for received in incoming[gpu]:
            value = builder.compute(
                "add_cc", [value, received], device(gpu), compute_level
            )
        chains[gpu] = rotate_chain(
            builder, value, gpu, compute_level, segment_length
        )

    gathered = [chains[0]]
    for gpu in range(1, 4):
        gathered.append(
            builder.transfer(
                builder.execution,
                chains[gpu],
                device(gpu),
                device(0),
                compute_level,
            )
        )
    output = gathered[0]
    for value in gathered[1:]:
        output = builder.compute(
            "add_cc", [output, value], device(0), compute_level
        )
    return builder.finish(output)


def build_four_gpu_interleaved(
    spec_path: Path, segment_length: int, communication_rounds: int,
    input_level: int, compute_level: int, online_transfers: bool = True,
) -> dict:
    builder = PlanBuilder(spec_path, device_count=4, plan_id=655360005)
    external = builder.value(HOST, input_level)
    builder.plan["external_inputs"] = [external]
    device_inputs = [
        builder.transfer(
            builder.initialization, external, HOST, device(gpu), input_level
        )
        for gpu in range(4)
    ]
    chains = [
        builder.compute(
            "mod_switch", [device_inputs[gpu]], device(gpu), compute_level,
            {"target_level": compute_level},
        )
        for gpu in range(4)
    ]

    # Post each transfer before independent local work. The receive is consumed
    # only after that work, giving the P2P stream a useful overlap window.
    rotations_per_round = distributed_counts(
        2 * segment_length, communication_rounds
    )
    for round_index, rotation_count in enumerate(rotations_per_round):
        distance = round_index % 3 + 1
        incoming: list[str] = []
        for destination in range(4):
            source = (destination + distance) % 4
            incoming.append(
                builder.transfer(
                    builder.execution,
                    chains[source],
                    device(source),
                    device(destination),
                    compute_level,
                )
                if online_transfers
                else chains[destination]
            )

        for gpu in range(4):
            value = rotate_chain(
                builder, chains[gpu], gpu, compute_level, rotation_count
            )
            chains[gpu] = builder.compute(
                "add_cc", [value, incoming[gpu]], device(gpu), compute_level
            )

    gathered = [chains[0]]
    for gpu in range(1, 4):
        gathered.append(
            builder.transfer(
                builder.execution,
                chains[gpu],
                device(gpu),
                device(0),
                compute_level,
            )
        )
    output = gathered[0]
    for value in gathered[1:]:
        output = builder.compute(
            "add_cc", [output, value], device(0), compute_level
        )
    return builder.finish(output)


def summarize(plan: dict) -> dict:
    computes = [item for item in plan["execution"] if item["kind"] == "compute"]
    return {
        "devices": plan["target"]["device_counts"][0],
        "values": len(plan["values"]),
        "initialization_instructions": len(plan["initialization"]),
        "execution_instructions": len(plan["execution"]),
        "execution_transfers": sum(
            item["kind"] == "transfer" for item in plan["execution"]
        ),
        "compute_ops": {
            op: sum(item["op"] == op for item in computes)
            for op in ("mod_switch", "rotate", "add_cc")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--segment-length", type=int, default=100)
    parser.add_argument("--communication-rounds", type=int, default=48)
    parser.add_argument("--input-level", type=int, default=39)
    parser.add_argument("--compute-level", type=int, default=31)
    args = parser.parse_args()
    if args.segment_length <= 0 or args.communication_rounds <= 0:
        parser.error("segment length and communication rounds must be positive")
    if args.compute_level >= args.input_level:
        parser.error("compute level must be lower than input level")

    plans = {
        "1gpu.runtime-plan.json": build_single_gpu(
            args.operator_spec, args.segment_length, args.communication_rounds,
            args.input_level, args.compute_level
        ),
        "4gpu.runtime-plan.json": build_four_gpu_interleaved(
            args.operator_spec, args.segment_length, args.communication_rounds,
            args.input_level, args.compute_level
        ),
        "4gpu-burst-transfers.runtime-plan.json": build_four_gpu_burst(
            args.operator_spec, args.segment_length, args.communication_rounds,
            args.input_level, args.compute_level
        ),
        "4gpu-no-online-transfers.runtime-plan.json": build_four_gpu_interleaved(
            args.operator_spec, args.segment_length, args.communication_rounds,
            args.input_level, args.compute_level, online_transfers=False
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, plan in plans.items():
        path = args.output_dir / filename
        path.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"{path}: {json.dumps(summarize(plan), sort_keys=True)}")


if __name__ == "__main__":
    main()
