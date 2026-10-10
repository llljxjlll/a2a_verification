#!/usr/bin/env python3
"""Run ASTRA-SIM validation levels L0-L4 and compare against A100 data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import gen_et

# The pinned congestion-aware executable rejects every topology with more than
# one dimension. The 2x2x2 experiment therefore runs on the multidimensional
# congestion-unaware backend; see FINDINGS_sim.md for the explicit limitation.
BACKENDS = ("unaware",)
STAGES = ("s1", "s2", "s3")
FINISH_RE = re.compile(r"sys\[(\d+)\] finished,\s*(\d+) cycles")
FACTOR_CANDIDATES = (0.25, 0.5, 1.0, 2.0, 4.0)
# The pinned analytical backend calls its input GB/s but implements 1 GB=2^30 B.
# GPU CSVs use decimal GB/s, so network-file values must be scaled down.
ASTRA_GBPS_TO_BYTES_PER_NS = (1 << 30) / 1_000_000_000


def executable(astra_sim_dir: Path, backend: str) -> Path:
    suffix = "Unaware" if backend == "unaware" else "Aware"
    return (
        astra_sim_dir / "build/astra_analytical/build/bin"
        / f"AstraSim_Analytical_Congestion_{suffix}"
    )


def parse_completion_cycles(output: str) -> int:
    matches = [(int(rank), int(cycles)) for rank, cycles in FINISH_RE.findall(output)]
    ranks = {rank for rank, _ in matches}
    if ranks != set(range(gen_et.NPUS)):
        raise ValueError(
            f"expected completion lines for ranks 0..{gen_et.NPUS - 1}, "
            f"found {sorted(ranks)}"
        )
    return max(cycles for _, cycles in matches)


def run_one(binary: Path, prefix: Path, system: Path, network: Path,
            memory: Path, log_path: Path) -> int:
    internal_log_dir = log_path.parent / "internal"
    internal_log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        f"--workload-configuration={prefix}",
        f"--system-configuration={system}",
        f"--network-configuration={network}",
        f"--remote-memory-configuration={memory}",
        "--comm-group-configuration=empty",
        f"--logging-folder={internal_log_dir}",
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n" + completed.stdout,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ASTRA-SIM exited with {completed.returncode}; see {log_path}"
        )
    try:
        return parse_completion_cycles(completed.stdout)
    except ValueError as exc:
        raise RuntimeError(f"cannot parse ASTRA-SIM output; see {log_path}: {exc}") from exc


def infer_factor(cycles: int, probe_bytes: int = gen_et.PROBE_BYTES,
                 bandwidth_bytes_ns: float = 100.0 * ASTRA_GBPS_TO_BYTES_PER_NS,
                 ) -> tuple[float, float, float]:
    raw = cycles * bandwidth_bytes_ns / probe_bytes
    selected = min(FACTOR_CANDIDATES, key=lambda value: abs(value - raw))
    error_pct = abs(raw / selected - 1.0) * 100.0
    if error_pct > 1.0:
        raise ValueError(
            f"raw comm_size factor {raw:.6f} is not within 1% of a supported "
            f"factor {selected:g}"
        )
    return selected, raw, error_pct


def load_gpu_rows(path: Path) -> dict[tuple[int, str], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[tuple[int, str], float] = {}
    for row in rows:
        if row.get("mode") == "sweep" and row.get("stage") in STAGES:
            indexed[(int(row["data_size_bytes"]), row["stage"])] = float(
                row["t_median_us"]
            )
    return indexed


def pct_error(actual: float, expected: float) -> float:
    return abs(actual / expected - 1.0) * 100.0 if expected else math.inf


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_network_config(profile: dict[str, Any], path: Path) -> None:
    """Materialize a plan-compliant network instead of trusting stale YAML."""
    if profile.get("dimension_order") != ["X", "Y", "W"]:
        raise ValueError("profile dimension order must be [X, Y, W]")
    decimal_bandwidth = [float(value) for value in profile["bandwidth_gbps"]]
    latency = [float(value) for value in profile["latency_ns"]]
    if len(decimal_bandwidth) != 3 or len(latency) != 3:
        raise ValueError("profile must contain exactly three dimensions")
    astra_bandwidth = [
        value / ASTRA_GBPS_TO_BYTES_PER_NS for value in decimal_bandwidth
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Generated from the A100 empirical profile by run_astra_sim.py.\n"
        "topology: [ FullyConnected, FullyConnected, FullyConnected ]\n"
        "npus_count: [ 2, 2, 2 ]\n"
        "# ASTRA uses 1 GB=2^30 B; values below encode decimal GPU GB/s.\n"
        "bandwidth: [ " + ", ".join(f"{value:.9f}" for value in astra_bandwidth)
        + " ]  # ASTRA GB/s (GiB/s convention)\n"
        "latency: [ " + ", ".join(f"{value:.3f}" for value in latency)
        + " ]  # ns\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--astra-sim-dir", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system", type=Path, required=True)
    parser.add_argument("--probe-network", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    traces = args.output_dir / "workloads"
    networks = args.output_dir / "networks"
    profiles_doc = json.loads(args.profiles.read_text(encoding="utf-8"))
    profiles = profiles_doc["profiles"]
    gpu_rows = load_gpu_rows(args.gpu_csv)
    failures: list[str] = []

    binaries = {backend: executable(args.astra_sim_dir, backend) for backend in BACKENDS}
    missing = [str(path) for path in binaries.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing ASTRA-SIM binaries: " + ", ".join(missing))

    # L0: independently measure each collective's comm_size -> per-link ratio.
    gen_et.generate_probes(traces, args.astra_sim_dir)
    probe_results: dict[str, dict[str, dict[str, float]]] = {}
    for backend in BACKENDS:
        probe_results[backend] = {}
        for stage in STAGES:
            cycles = run_one(
                binaries[backend], traces / f"probe_{stage}", args.system,
                args.probe_network, args.memory,
                logs / f"L0_{backend}_{stage}.log",
            )
            factor, raw, error = infer_factor(cycles)
            probe_results[backend][stage] = {
                "cycles": cycles, "factor": factor, "raw_factor": raw,
                "snap_error_pct": error,
            }

    semantics: dict[str, Any] = {"probe_bytes": gen_et.PROBE_BYTES, "stages": {}}
    for stage in STAGES:
        semantics["stages"][stage] = {
            "collective": gen_et.STAGES[stage]["collective"],
            "factor": probe_results["unaware"][stage]["factor"],
            "per_link_bytes_per_comm_size": probe_results["unaware"][stage]["factor"],
            "backends": {
                backend: probe_results[backend][stage] for backend in BACKENDS
            },
        }
    semantics_path = args.output_dir / "comm_size_semantics.json"
    semantics_path.write_text(json.dumps(semantics, indent=2) + "\n", encoding="utf-8")

    workload_manifest = gen_et.generate_workloads(
        args.profiles, semantics_path, traces, args.astra_sim_dir
    )["workloads"]

    # L1/L3/L4: isolated stages and the strictly serialized three-stage trace.
    result_rows: list[dict[str, Any]] = []
    run_times: dict[tuple[int, str, str], float] = {}
    l2_rows: list[dict[str, Any]] = []
    for size_text, profile in sorted(profiles.items(), key=lambda item: int(item[0])):
        size = int(size_text)
        mib = int(profile["data_size_mib"])
        network = networks / f"network_D_{mib}MiB.yml"
        write_network_config(profile, network)
        entries = workload_manifest[size_text]

        for backend in BACKENDS:
            for stage in (*STAGES, "total"):
                cycles = run_one(
                    binaries[backend], Path(entries[stage]["prefix"]), args.system,
                    network, args.memory,
                    logs / f"D_{mib}MiB_{backend}_{stage}.log",
                )
                t_us = cycles / 1000.0
                run_times[(size, backend, stage)] = t_us
                if stage == "total":
                    gpu_us = sum(gpu_rows[(size, item)] for item in STAGES)
                    formula_us = sum(
                        float(profile["stages"][item]["astra_sim_latency_ns"]) / 1000.0
                        + int(profile["stages"][item]["transfer_bytes"])
                        / (float(profile["stages"][item]["astra_sim_bandwidth_gbps"])
                           * 1000.0)
                        for item in STAGES
                    )
                    row = {
                        "D": size, "stage": "total", "dim": "Y-W-X",
                        "comm_size": "", "per_link_bytes": "", "backend": backend,
                        "t_us": t_us, "gpu_t_us": gpu_us,
                        "error_pct": pct_error(t_us, gpu_us),
                        "model_formula_us": formula_us,
                        "formula_error_pct": pct_error(t_us, formula_us),
                        "run_kind": "L3",
                    }
                else:
                    entry = entries[stage]
                    stage_profile = profile["stages"][stage]
                    gpu_us = gpu_rows.get(
                        (size, stage),
                        float(stage_profile["measured_time_us"]),
                    )
                    formula_us = (
                        float(stage_profile["astra_sim_latency_ns"]) / 1000.0
                        + int(entry["per_link_bytes"])
                        / (float(stage_profile["astra_sim_bandwidth_gbps"]) * 1000.0)
                    )
                    row = {
                        "D": size, "stage": stage, "dim": entry["dim"],
                        "comm_size": entry["comm_size"],
                        "per_link_bytes": entry["per_link_bytes"],
                        "backend": backend, "t_us": t_us, "gpu_t_us": gpu_us,
                        "error_pct": pct_error(t_us, gpu_us),
                        "model_formula_us": formula_us,
                        "formula_error_pct": pct_error(t_us, formula_us),
                        "run_kind": "L1/L4",
                    }
                result_rows.append(row)

            # L2: identical AllToAll and bytes, placed once on Y and once on W.
            l2_times: dict[str, float] = {}
            for label in ("l2_fast_y", "l2_slow_w"):
                cycles = run_one(
                    binaries[backend], Path(entries[label]["prefix"]), args.system,
                    network, args.memory,
                    logs / f"D_{mib}MiB_{backend}_{label}.log",
                )
                l2_times[label] = cycles / 1000.0
            bytes_count = int(entries["l2_fast_y"]["per_link_bytes"])
            fast_expected = (
                float(profile["latency_ns"][1]) / 1000.0
                + bytes_count / (float(profile["bandwidth_gbps"][1]) * 1000.0)
            )
            slow_expected = (
                float(profile["latency_ns"][2]) / 1000.0
                + bytes_count / (float(profile["bandwidth_gbps"][2]) * 1000.0)
            )
            l2_rows.append({
                "D": size, "backend": backend,
                "fast_y_us": l2_times["l2_fast_y"],
                "slow_w_us": l2_times["l2_slow_w"],
                "observed_ratio": l2_times["l2_slow_w"] / l2_times["l2_fast_y"],
                "expected_ratio": slow_expected / fast_expected,
            })

    fields = [
        "D", "stage", "dim", "comm_size", "per_link_bytes", "backend",
        "t_us", "gpu_t_us", "error_pct", "model_formula_us",
        "formula_error_pct", "run_kind",
    ]
    with (args.output_dir / "astra_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)
    l2_fields = [
        "D", "backend", "fast_y_us", "slow_w_us", "observed_ratio",
        "expected_ratio",
    ]
    with (args.output_dir / "l2_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=l2_fields)
        writer.writeheader()
        writer.writerows(l2_rows)

    # Acceptance checks and report.
    for row in result_rows:
        if float(row["formula_error_pct"]) > 1.0:
            failures.append(
                f"L1/L3: D={row['D']} {row['stage']} {row['backend']} 与解析公式 "
                f"误差 {float(row['formula_error_pct']):.2f}% > 1%"
            )
        if float(row["error_pct"]) > 3.0:
            failures.append(
                f"L4: D={row['D']} {row['stage']} {row['backend']} 与 GPU "
                f"误差 {float(row['error_pct']):.2f}% > 3%"
            )
    for size_text in profiles:
        size = int(size_text)
        for backend in BACKENDS:
            isolated_sum = sum(run_times[(size, backend, stage)] for stage in STAGES)
            combined = run_times[(size, backend, "total")]
            if pct_error(combined, isolated_sum) > 1.0:
                failures.append(
                    f"L3: D={size} {backend} 串联时间与三阶段之和不一致"
                )
    for row in l2_rows:
        if pct_error(float(row["observed_ratio"]), float(row["expected_ratio"])) > 1.0:
            failures.append(
                f"L2: D={row['D']} {row['backend']} 慢/快维度比值与配置不符"
            )

    lines = [
        "# 2×(2×2) Mesh-Rail ASTRA-SIM 对照结果", "",
        "> 本报告由 `astra_sim/run_astra_sim.py` 生成。时间单位为 µs；"
        "analytical backend 的 cycle 按 1 ns 换算。网络带宽已校正 ASTRA-SIM "
        "使用 2^30 B/GB、GPU CSV 使用 10^9 B/GB 的单位差异。", "",
        "## L0：`comm_size` 语义探针", "",
        "| 阶段 | collective | 后端 | cycles | 原始每链路倍率 | 采用倍率 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for stage in STAGES:
        for backend in BACKENDS:
            item = probe_results[backend][stage]
            lines.append(
                f"| {stage} | {gen_et.STAGES[stage]['collective']} | {backend} | "
                f"{int(item['cycles'])} | {fmt(item['raw_factor'], 6)} | "
                f"{item['factor']:g} |"
            )

    lines += [
        "", "## L1 / L4：单维度公式与 GPU 逐点对照", "",
        "| D (MiB) | 阶段 | 维度 | 后端 | ASTRA-SIM | 解析公式 | 公式误差 | GPU median | GPU 误差 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        if row["stage"] == "total":
            continue
        lines.append(
            f"| {int(row['D']) / (1 << 20):g} | {row['stage']} | {row['dim']} | "
            f"{row['backend']} | {fmt(float(row['t_us']))} | "
            f"{fmt(float(row['model_formula_us']))} | "
            f"{fmt(float(row['formula_error_pct']), 2)}% | "
            f"{fmt(float(row['gpu_t_us']))} | {fmt(float(row['error_pct']), 2)}% |"
        )

    lines += [
        "", "## L2：慢层维度映射", "",
        "同一 `ALL_TO_ALL`、同一每链路字节数分别锁到 Y 和 W；期望比值由"
        "该消息点的实测网络配置计算。", "",
        "| D (MiB) | 后端 | Y-fast (µs) | W-slow (µs) | 实测比值 | 配置期望 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in l2_rows:
        lines.append(
            f"| {int(row['D']) / (1 << 20):g} | {row['backend']} | "
            f"{fmt(row['fast_y_us'])} | {fmt(row['slow_w_us'])} | "
            f"{fmt(row['observed_ratio'])} | {fmt(row['expected_ratio'])} |"
        )

    lines += [
        "", "## L3：三阶段严格串联", "",
        "| D (MiB) | 后端 | 串联仿真 | 单阶段仿真之和 | GPU 三阶段之和 | 仿真/GPU 误差 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for size_text, profile in sorted(profiles.items(), key=lambda item: int(item[0])):
        size = int(size_text)
        for backend in BACKENDS:
            total = run_times[(size, backend, "total")]
            isolated = sum(run_times[(size, backend, stage)] for stage in STAGES)
            gpu_total = sum(gpu_rows[(size, stage)] for stage in STAGES)
            lines.append(
                f"| {int(profile['data_size_mib'])} | {backend} | {fmt(total)} | "
                f"{fmt(isolated)} | {fmt(gpu_total)} | "
                f"{fmt(pct_error(total, gpu_total), 2)}% |"
            )

    lines += [
        "", "### 三阶段耗时占比", "",
        "| D (MiB) | 后端 | S1 | S2 | S3 | GPU S1/S2/S3 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for size_text, profile in sorted(profiles.items(), key=lambda item: int(item[0])):
        size = int(size_text)
        gpu_stage_times = [gpu_rows[(size, stage)] for stage in STAGES]
        gpu_sum = sum(gpu_stage_times)
        gpu_shares = "/".join(f"{value / gpu_sum * 100:.1f}%" for value in gpu_stage_times)
        for backend in BACKENDS:
            sim_stage_times = [run_times[(size, backend, stage)] for stage in STAGES]
            sim_sum = sum(sim_stage_times)
            shares = [value / sim_sum * 100 for value in sim_stage_times]
            lines.append(
                f"| {int(profile['data_size_mib'])} | {backend} | "
                f"{shares[0]:.1f}% | {shares[1]:.1f}% | {shares[2]:.1f}% | "
                f"{gpu_shares} |"
            )

    lines += ["", "## 验收结论", ""]
    if failures:
        lines.append("**未通过。**")
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("**Congestion_Unaware 后端的 L0–L4 全部通过。**")
    lines += [
        "", "## 后端适用性", "",
        "锁定版本的 `AstraSim_Analytical_Congestion_Aware` 在读取三维网络时会"
        "直接退出并报告 `only support 1-dim topology`，因此不能用于本次 "
        "2×2×2 Mesh-Rail。这里没有将该后端包装成虚假的三维结果；完整对照使用"
        "支持多维拓扑的 `Congestion_Unaware`。",
        "", "## 结论边界", "",
        "每个消息点使用由同一 GPU 实测点反推的带宽/延迟参数，因此这里验证的是"
        "通信量映射、维度映射和 ASTRA-SIM 逐点复现，不是跨消息大小预测。",
        "2×(2×2) 的每个维度只有两个节点，可用 `FullyConnected` 表示单链路；"
        "论文中的 6×6 开边界 mesh 不能由 analytical 后端精确表达。该实验也不验证"
        "6×6 中心热点、多跳路由或 16-wafer torus 拥塞。", "",
    ]
    (args.output_dir / "FINDINGS_sim.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    if failures:
        print(f"ASTRA-SIM validation produced {len(failures)} failure(s).")
        print(f"See {args.output_dir / 'FINDINGS_sim.md'}")
        return 1 if args.strict else 0
    print(f"ASTRA-SIM validation passed. Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
