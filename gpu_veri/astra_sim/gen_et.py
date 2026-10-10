#!/usr/bin/env python3
"""Generate Chakra ET traces for the 2x(2x2) Mesh-Rail experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

NPUS = 8
PROBE_BYTES = 1 << 20
STAGES = {
    "s1": {"collective": "ALL_GATHER", "dim": "Y", "dims": [False, True, False]},
    "s2": {"collective": "ALL_TO_ALL", "dim": "W", "dims": [False, False, True]},
    "s3": {"collective": "REDUCE_SCATTER", "dim": "X", "dims": [True, False, False]},
}


def load_chakra(astra_sim_dir: Path) -> dict[str, Any]:
    """Load Chakra from either pip or ASTRA-SIM's pinned nested submodule."""
    try:
        from chakra.schema.protobuf import et_def_pb2 as schema
        from chakra.src.third_party.utils.protolib import encodeMessage
    except ImportError:
        sys.path.insert(0, str(astra_sim_dir))
        try:
            from extern.graph_frontend.chakra.schema.protobuf import et_def_pb2 as schema
            from extern.graph_frontend.chakra.src.third_party.utils.protolib import (
                encodeMessage,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import Chakra. Initialize nested submodules and build ASTRA-SIM "
                "first so et_def_pb2.py is generated."
            ) from exc
    return {"schema": schema, "encode": encodeMessage}


def _attribute(schema: Any, name: str, *, integer: int | None = None,
               boolean: bool | None = None,
               bools: list[bool] | None = None) -> Any:
    attr = schema.AttributeProto(name=name)
    if integer is not None:
        attr.int64_val = integer
    elif boolean is not None:
        attr.bool_val = boolean
    elif bools is not None:
        attr.bool_list.values.extend(bools)
    else:
        raise ValueError("attribute needs an integer or bool list")
    return attr


def collective_node(schema: Any, node_id: int, name: str, collective: str,
                    comm_size: int, dims: list[bool], deps: list[int]) -> Any:
    node = schema.Node(id=node_id, name=name, type=schema.COMM_COLL_NODE)
    node.attr.extend([
        _attribute(schema, "is_cpu_op", boolean=False),
        _attribute(schema, "comm_type", integer=getattr(schema, collective)),
        _attribute(schema, "comm_size", integer=comm_size),
        _attribute(schema, "involved_dim", bools=dims),
    ])
    node.ctrl_deps.extend(deps)
    return node


def write_trace(prefix: Path, nodes: list[Callable[[Any], Any]], chakra: dict[str, Any]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    schema, encode = chakra["schema"], chakra["encode"]
    for npu in range(NPUS):
        with Path(f"{prefix}.{npu}.et").open("wb") as handle:
            encode(handle, schema.GlobalMetadata(version="0.0.4"))
            for make_node in nodes:
                encode(handle, make_node(schema))


def make_stage_factory(stage: str, comm_size: int, node_id: int = 1,
                       deps: list[int] | None = None) -> Callable[[Any], Any]:
    spec = STAGES[stage]
    return lambda schema: collective_node(
        schema, node_id, stage.upper(), spec["collective"], comm_size,
        spec["dims"], deps or [],
    )


def generate_probes(output_dir: Path, astra_sim_dir: Path) -> None:
    chakra = load_chakra(astra_sim_dir)
    for stage in STAGES:
        write_trace(
            output_dir / f"probe_{stage}",
            [make_stage_factory(stage, PROBE_BYTES)],
            chakra,
        )


def comm_size_for(per_link_bytes: int, factor: float) -> int:
    if factor <= 0:
        raise ValueError("comm_size factor must be positive")
    value = round(per_link_bytes / factor)
    if value <= 0 or abs(value * factor - per_link_bytes) > 1.0:
        raise ValueError(
            f"cannot represent {per_link_bytes} per-link bytes with factor {factor}"
        )
    return value


def generate_workloads(profiles_path: Path, semantics_path: Path, output_dir: Path,
                       astra_sim_dir: Path) -> dict[str, Any]:
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))["profiles"]
    semantics = json.loads(semantics_path.read_text(encoding="utf-8"))["stages"]
    chakra = load_chakra(astra_sim_dir)
    manifest: dict[str, Any] = {"npus": NPUS, "workloads": {}}

    for size_text, profile in sorted(profiles.items(), key=lambda item: int(item[0])):
        size = int(size_text)
        mib = int(profile["data_size_mib"])
        entries: dict[str, Any] = {}
        stage_sizes: dict[str, int] = {}
        for stage in STAGES:
            per_link = int(profile["stages"][stage]["transfer_bytes"])
            comm_size = comm_size_for(per_link, float(semantics[stage]["factor"]))
            stage_sizes[stage] = comm_size
            prefix = output_dir / f"D_{mib}MiB_{stage}"
            write_trace(prefix, [make_stage_factory(stage, comm_size)], chakra)
            entries[stage] = {
                "prefix": str(prefix),
                "comm_size": comm_size,
                "per_link_bytes": per_link,
                "dim": STAGES[stage]["dim"],
            }

        combined = output_dir / f"D_{mib}MiB_all"
        write_trace(combined, [
            make_stage_factory("s1", stage_sizes["s1"], 1, []),
            make_stage_factory("s2", stage_sizes["s2"], 2, [1]),
            make_stage_factory("s3", stage_sizes["s3"], 3, [2]),
        ], chakra)
        entries["total"] = {"prefix": str(combined)}

        # L2 uses the same A2A primitive and byte count on fast-Y and slow-W.
        l2_comm_size = comm_size_for(size // 2, float(semantics["s2"]["factor"]))
        for label, dims, dim in (
            ("l2_fast_y", [False, True, False], "Y"),
            ("l2_slow_w", [False, False, True], "W"),
        ):
            prefix = output_dir / f"D_{mib}MiB_{label}"
            factory = lambda schema, d=dims: collective_node(
                schema, 1, label, "ALL_TO_ALL", l2_comm_size, d, []
            )
            write_trace(prefix, [factory], chakra)
            entries[label] = {
                "prefix": str(prefix), "comm_size": l2_comm_size,
                "per_link_bytes": size // 2, "dim": dim,
            }

        manifest["workloads"][size_text] = entries

    manifest_path = output_dir / "workloads.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--astra-sim-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--output-dir", type=Path, required=True)
    workloads = subparsers.add_parser("workloads")
    workloads.add_argument("--profiles", type=Path, required=True)
    workloads.add_argument("--semantics", type=Path, required=True)
    workloads.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "probe":
        generate_probes(args.output_dir, args.astra_sim_dir)
    else:
        generate_workloads(
            args.profiles, args.semantics, args.output_dir, args.astra_sim_dir
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
