#!/usr/bin/env python3
"""Convert a one-GPU RuntimePlan into an equivalent local CPU plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def host_place() -> dict[str, int | str]:
    return {"kind": "host", "rank": 0}


def map_place(place: dict) -> dict[str, int | str]:
    if place.get("rank") != 0:
        raise ValueError("only rank 0 places are supported")
    kind = place.get("kind")
    if kind == "host":
        return host_place()
    if kind == "device" and place.get("index") == 0:
        return host_place()
    raise ValueError("the plan contains a place other than Host(0) or Device(0)")


def encode_json(value: dict) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def resolve(value_id: str, aliases: dict[str, str]) -> str:
    path: list[str] = []
    while value_id in aliases:
        path.append(value_id)
        value_id = aliases[value_id]
    for item in path:
        aliases[item] = value_id
    return value_id


def comparable_desc(value: dict) -> dict:
    return {
        key: item for key, item in value.items()
        if key not in {"id", "place"}
    }


def convert_plan(plan: dict, spec_id: str, spec_digest: str) -> tuple[dict, int]:
    target = plan["target"]
    if (target.get("world_size") != 1 or
            target.get("device_counts") != [1] or
            target.get("target_id") != "poseidon-ckks-gpu"):
        raise ValueError("input must be a one-rank, one-GPU Poseidon plan")

    values = {value["id"]: value for value in plan["values"]}
    for value in values.values():
        value["place"] = map_place(value["place"])

    aliases: dict[str, str] = {}
    removed_outputs: set[str] = set()
    removed_transfers = 0
    next_ordinal = 0

    for phase_name in ("initialization", "execution", "finalization"):
        converted: list[dict] = []
        for instruction in plan[phase_name]:
            instruction = dict(instruction)
            kind = instruction["kind"]
            if "inputs" in instruction:
                instruction["inputs"] = [
                    resolve(value_id, aliases)
                    for value_id in instruction["inputs"]
                ]
            if kind == "compute":
                instruction["place"] = map_place(instruction["place"])
            elif kind in {"transfer", "replicate"}:
                instruction["sources"] = [
                    map_place(place) for place in instruction["sources"]
                ]
                instruction["destinations"] = [
                    map_place(place) for place in instruction["destinations"]
                ]
                if len(instruction["inputs"]) != 1:
                    raise ValueError("local communication must have exactly one input")
                source_id = instruction["inputs"][0]
                source_desc = values[source_id]
                for output_id in instruction["outputs"]:
                    if comparable_desc(values[output_id]) != comparable_desc(source_desc):
                        raise ValueError(
                            f"local transfer changes metadata for ValueId {output_id}"
                        )
                    aliases[output_id] = source_id
                    removed_outputs.add(output_id)
                removed_transfers += 1
                continue
            instruction["ordinal"] = next_ordinal
            next_ordinal += 1
            converted.append(instruction)
        plan[phase_name] = converted

    plan["values"] = [
        value for value_id, value in values.items()
        if value_id not in removed_outputs
    ]
    plan["external_inputs"] = [
        resolve(value_id, aliases) for value_id in plan["external_inputs"]
    ]
    plan["final_outputs"] = [
        resolve(value_id, aliases) for value_id in plan["final_outputs"]
    ]
    target["target_id"] = "poseidon-ckks-cpu"
    target["device_counts"] = [0]
    target["operator_spec"] = {
        "id": spec_id,
        "version": plan["target"]["operator_spec"]["version"],
        "source_sha256": spec_digest,
    }
    return plan, removed_transfers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-plan", required=True, type=Path)
    parser.add_argument("--gpu-operator-spec", required=True, type=Path)
    parser.add_argument("--cpu-plan", required=True, type=Path)
    parser.add_argument("--cpu-operator-spec", required=True, type=Path)
    args = parser.parse_args()

    plan = json.loads(args.gpu_plan.read_text(encoding="utf-8"))
    spec = json.loads(args.gpu_operator_spec.read_text(encoding="utf-8"))
    if spec.get("target_id") != "poseidon-ckks-gpu":
        raise ValueError("input OperatorSpec is not a Poseidon GPU spec")
    spec["target_id"] = "poseidon-ckks-cpu"
    spec["spec_id"] = spec["spec_id"] + "-cpu-equivalent"
    spec_bytes = encode_json(spec)
    spec_digest = "sha256:" + hashlib.sha256(spec_bytes).hexdigest()
    converted, removed_transfers = convert_plan(
        plan, spec["spec_id"], spec_digest
    )

    args.cpu_operator_spec.parent.mkdir(parents=True, exist_ok=True)
    args.cpu_plan.parent.mkdir(parents=True, exist_ok=True)
    args.cpu_operator_spec.write_bytes(spec_bytes)
    args.cpu_plan.write_bytes(encode_json(converted))
    compute_count = sum(
        instruction["kind"] == "compute"
        for phase in ("initialization", "execution", "finalization")
        for instruction in converted[phase]
    )
    print(
        f"cpu_plan={args.cpu_plan} compute={compute_count} "
        f"removed_local_transfers={removed_transfers}"
    )
    print(f"cpu_operator_spec={args.cpu_operator_spec} {spec_digest}")


if __name__ == "__main__":
    main()
