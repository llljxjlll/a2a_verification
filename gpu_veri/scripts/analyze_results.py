#!/usr/bin/env python3
"""Analyze Mesh-Rail GPU measurements and render FINDINGS.md."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

STAGES = ("s1", "s2", "s3")
DIMENSION_STAGE = (("X", "s3"), ("Y", "s1"), ("W", "s2"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "mode", "stage", "pair", "data_size_bytes", "per_link_bytes", "tier",
        "target_bw_gbps", "achieved_bw_gbps", "ratio_R", "chunk_bytes",
        "pace_clamped", "sm_clock_mhz", "iters", "t_mean_us", "t_median_us",
        "t_p99_us", "t_min_us", "t_max_us",
    }
    if not rows:
        raise ValueError(f"{path} contains no rows")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path} misses columns: {sorted(missing)}")
    return rows


def through_origin_fit(points: Iterable[tuple[float, float]]) -> tuple[float, float]:
    points = list(points)
    denom = sum(x * x for x, _ in points)
    slope = sum(x * y for x, y in points) / denom
    residual = sum((y - slope * x) ** 2 for x, y in points)
    baseline = sum(y * y for _, y in points)
    return slope, 1.0 - residual / baseline if baseline else 1.0


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def sample_clock_report(path: Path) -> tuple[str, int]:
    if not path.exists():
        return "逐样本频率文件不存在。", 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    clocks: list[int] = []
    for row in rows:
        for key in ("sm_clock_gpu0_mhz", "sm_clock_gpu1_mhz"):
            value = int(row[key])
            if value:
                clocks.append(value)
    if not clocks:
        return "`nvidia-smi` 未返回逐样本 SM 频率。", 0
    counts = Counter(clocks)
    mode, mode_count = counts.most_common(1)[0]
    rejected = len(clocks) - mode_count
    distribution = ", ".join(f"{clock} MHz: {count}" for clock, count in sorted(counts.items()))
    return f"频率分布：{distribution}；按众数 {mode} MHz 筛选会剔除 {rejected}/{len(clocks)} 个端点样本。", rejected


def globally_filter_samples(
    rows: list[dict[str, str]], sample_path: Path
) -> tuple[list[dict[str, str]], str]:
    """Recompute sweep statistics using each physical GPU's global modal clock."""
    if not sample_path.exists():
        return rows, "未找到逐样本文件，未执行全局频率众数筛选。"
    with sample_path.open(newline="", encoding="utf-8") as handle:
        samples = [row for row in csv.DictReader(handle) if row["mode"] == "sweep"]
    if not samples:
        return rows, "逐样本文件中没有 sweep 样本，未执行全局频率众数筛选。"

    clock_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for sample in samples:
        for suffix in ("0", "1"):
            gpu = int(sample[f"physical_gpu{suffix}"])
            clock = int(sample[f"sm_clock_gpu{suffix}_mhz"])
            if clock > 0:
                clock_counts[gpu][clock] += 1
    modes = {
        gpu: counts.most_common(1)[0][0]
        for gpu, counts in clock_counts.items()
        if counts
    }
    if not modes:
        return rows, "`nvidia-smi` 未返回有效逐样本频率，未执行频率筛选。"

    kept: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    total: Counter[tuple[str, str, str, int]] = Counter()
    pair_modes: dict[tuple[str, str, str, int], tuple[int, int]] = {}
    for sample in samples:
        key = (
            sample["mode"], sample["stage"], sample["pair"],
            int(sample["data_size_bytes"]),
        )
        total[key] += 1
        gpu0, gpu1 = int(sample["physical_gpu0"]), int(sample["physical_gpu1"])
        clock0 = int(sample["sm_clock_gpu0_mhz"])
        clock1 = int(sample["sm_clock_gpu1_mhz"])
        mode0, mode1 = modes.get(gpu0, 0), modes.get(gpu1, 0)
        pair_modes[key] = (mode0, mode1)
        if (mode0 == 0 or clock0 == mode0) and (mode1 == 0 or clock1 == mode1):
            kept[key].append(float(sample["pair_max_us"]))

    filtered: list[dict[str, str]] = []
    rejected = 0
    for original in rows:
        row = dict(original)
        key = (row["mode"], row["stage"], row["pair"], int(row["data_size_bytes"]))
        values = kept.get(key, [])
        if key in total:
            rejected += total[key] - len(values)
            row["iters"] = str(len(values))
            mode0, mode1 = pair_modes[key]
            row["sm_clock_mhz"] = str(min(value for value in (mode0, mode1) if value > 0)) if mode0 or mode1 else "0"
            if values:
                ordered = sorted(values)
                median = statistics.median(ordered)
                p99 = ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)]
                row["t_mean_us"] = str(statistics.mean(ordered))
                row["t_median_us"] = str(median)
                row["t_p99_us"] = str(p99)
                row["t_min_us"] = str(ordered[0])
                row["t_max_us"] = str(ordered[-1])
                row["achieved_bw_gbps"] = str(
                    int(row["per_link_bytes"]) / (median * 1_000.0)
                )
        filtered.append(row)

    distribution = ", ".join(
        f"GPU{gpu}={clock} MHz" for gpu, clock in sorted(modes.items())
    )
    return filtered, f"全局逐卡众数：{distribution}；共剔除 {rejected}/{len(samples)} 个 sweep 样本。"


def concurrency_report(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return ["未提供并发度扫描结果。"]
    rows = read_rows(path)
    groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if not row["mode"].startswith("concurrency-"):
            continue
        groups[int(row["mode"].split("-")[1])].append(float(row["t_median_us"]))
    if not groups:
        return ["并发 CSV 中没有 concurrency-* 行。"]
    baseline = statistics.median(groups[min(groups)])
    result = ["| 并发 pair 数 | pair 数量 | median 的中位数 (µs) | 相对 1-pair 差异 |", "|---:|---:|---:|---:|"]
    for count in sorted(groups):
        value = statistics.median(groups[count])
        result.append(f"| {count} | {len(groups[count])} | {fmt(value)} | {fmt((value / baseline - 1) * 100)}% |")
    return result


def write_astra_profiles(
    rows: list[dict[str, str]], calibration: dict, out_dir: Path
) -> dict[int, dict]:
    """Write one empirical ASTRA-SIM network configuration per message size.

    ASTRA-SIM models time as a latency term plus serialization time.  The CSV
    bandwidth includes both, so feeding it to ASTRA-SIM together with latency
    would double count startup cost.  For each measured point we therefore use
    the 1-byte operation latency from calibration and derive the serialization
    bandwidth that reconstructs the measured median:

        B_astra(D) = bytes(D) / (T_median(D) - L_operation)

    S1 uses the measured AllGather latency.  S2 and S3 both use the measured
    point-to-point send/recv latency; pacing changes bandwidth, not the NCCL
    primitive's startup latency.
    """
    indexed = {(int(row["data_size_bytes"]), row["stage"]): row for row in rows}
    sizes = sorted({size for size, _ in indexed})
    latency_by_stage = {
        "s1": float(calibration["fast_tier"]["latency_us"]),
        "s2": float(calibration["slow_tier"]["latency_us"]),
        "s3": float(calibration["slow_tier"]["latency_us"]),
    }
    profiles: dict[int, dict] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        missing = [stage for stage in STAGES if (size, stage) not in indexed]
        if missing:
            raise ValueError(f"cannot export D={size}: missing stages {missing}")
        stages: dict[str, dict] = {}
        for stage in STAGES:
            row = indexed[(size, stage)]
            measured_us = float(row["t_median_us"])
            latency_us = latency_by_stage[stage]
            serialization_us = measured_us - latency_us
            if serialization_us <= 0:
                raise ValueError(
                    f"cannot decompose D={size} {stage}: measured {measured_us} us "
                    f"is not greater than latency {latency_us} us"
                )
            transfer_bytes = int(row["per_link_bytes"])
            stages[stage] = {
                "transfer_bytes": transfer_bytes,
                "measured_time_us": measured_us,
                "measured_message_latency_us": measured_us,
                "measured_effective_bandwidth_gbps": float(row["achieved_bw_gbps"]),
                "target_bandwidth_gbps": float(row["target_bw_gbps"]),
                "astra_sim_bandwidth_gbps": transfer_bytes / (serialization_us * 1_000.0),
                "astra_sim_latency_ns": latency_us * 1_000.0,
                "latency_source": "allgather_1byte" if stage == "s1" else "sendrecv_1byte",
            }

        bandwidth = [stages[stage]["astra_sim_bandwidth_gbps"] for _, stage in DIMENSION_STAGE]
        latency = [stages[stage]["astra_sim_latency_ns"] for _, stage in DIMENSION_STAGE]
        mib = size // (1 << 20)
        filename = f"network_D_{mib}MiB.yml"
        yaml = [
            "# Empirical per-message ASTRA-SIM profile generated from A100 measurements.",
            "# Dimension order is [X, Y, W] = [S3 fast, S1 fast, S2 slow].",
            f"# Original per-rank A2A buffer D = {mib} MiB.",
            "topology: [ Ring, Ring, Ring ]",
            "npus_count: [ 2, 2, 2 ]",
            "bandwidth: [ " + ", ".join(f"{value:.9f}" for value in bandwidth) + " ]  # GB/s",
            "latency: [ " + ", ".join(f"{value:.3f}" for value in latency) + " ]  # ns",
            "",
        ]
        (out_dir / filename).write_text("\n".join(yaml), encoding="utf-8")
        profiles[size] = {
            "data_size_bytes": size,
            "data_size_mib": mib,
            "network_config": filename,
            "dimension_order": [dimension for dimension, _ in DIMENSION_STAGE],
            "stage_order": [stage for _, stage in DIMENSION_STAGE],
            "bandwidth_gbps": bandwidth,
            "latency_ns": latency,
            "stages": stages,
        }

    manifest = {
        "schema_version": 1,
        "profile_type": "empirical_per_message",
        "warning": (
            "Each profile is derived from the same message-size measurement it reproduces; "
            "use it for pointwise ASTRA-SIM comparison, not as evidence of cross-size prediction."
        ),
        "dimension_order": [dimension for dimension, _ in DIMENSION_STAGE],
        "profiles": {str(size): profile for size, profile in profiles.items()},
    }
    (out_dir / "profiles.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--concurrency", type=Path)
    parser.add_argument("--out", type=Path, default=Path("FINDINGS.md"))
    parser.add_argument(
        "--astra-dir", type=Path,
        help="output directory for five empirical per-message ASTRA-SIM profiles",
    )
    parser.add_argument("--strict", action="store_true", help="return nonzero when an acceptance check fails")
    args = parser.parse_args()

    rows = [row for row in read_rows(args.csv) if row["mode"] == "sweep"]
    if not rows:
        raise ValueError("no sweep rows found")
    with args.calibration.open(encoding="utf-8") as handle:
        calibration = json.load(handle)
    if calibration.get("pace_timer") != "globaltimer":
        raise ValueError("calibration did not use globaltimer")

    sample_path = args.samples or args.csv.with_suffix(".samples.csv")
    rows, global_clock_text = globally_filter_samples(rows, sample_path)
    if sample_path.exists():
        filtered_path = args.csv.with_name(args.csv.stem + ".filtered.csv")
        with filtered_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    valid_rows = [row for row in rows if int(row["pace_clamped"]) == 0]
    failures: list[str] = []
    warnings: list[str] = []
    if len(valid_rows) != len(rows):
        failures.append(f"{len(rows) - len(valid_rows)} 个 sweep 行发生 pacing 钳位")
    for row in valid_rows:
        if int(row["iters"]) < 100:
            failures.append(
                f"D={row['data_size_bytes']} {row['stage']} 频率筛选后仅剩 "
                f"{row['iters']} 个样本（要求至少 100）"
            )

    indexed = {(int(row["data_size_bytes"]), row["stage"]): row for row in valid_rows}
    sizes = sorted({size for size, _ in indexed})
    expected_sizes = [value * (1 << 20) for value in (2, 8, 32, 64, 256)]
    if sizes != expected_sizes:
        failures.append("数据量扫描点不完整")

    astra_dir = args.astra_dir or args.out.parent / "astra_sim_profiles"
    astra_profiles: dict[int, dict] = {}
    if sizes == expected_sizes and all((size, stage) in indexed for size in sizes for stage in STAGES):
        astra_profiles = write_astra_profiles(valid_rows, calibration, astra_dir)

    fast = float(calibration["fast_tier"]["B_achieved_gbps"])
    slow = float(calibration["slow_tier"]["B_achieved_gbps"])
    lines = [
        "# Mesh-Rail A2A 真机验证结果",
        "",
        "> 本文件由 `scripts/analyze_results.py` 生成。所有时间使用两端 CUDA event 的较大值；带宽采用十进制 GB/s。",
        "",
        "## 标定摘要",
        "",
        "| 层级 | B_raw (GB/s) | B_target (GB/s) | B_achieved (GB/s) | L_eff (µs) | O (ns/chunk) | 轮数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("fast_tier", "fast"), ("slow_tier", "slow")):
        tier = calibration[name]
        lines.append(
            f"| {label} | {fmt(tier['B_raw_gbps'])} | {fmt(tier['B_target_gbps'])} | "
            f"{fmt(tier['B_achieved_gbps'])} | {fmt(tier['latency_us'], 3)} | "
            f"{fmt(tier['O_ns'])} | {tier['calib_iters']} |"
        )
        if tier["calib_iters"] > 3:
            warnings.append(f"{label} 层闭环标定用了 {tier['calib_iters']} 轮，存在非线性")
        for endpoint, achieved in tier.get("cross_validation", {}).items():
            if abs(float(achieved) - float(tier["B_target_gbps"])) / float(tier["B_target_gbps"]) > 0.03:
                warnings.append(
                    f"{label} 层 {endpoint} 未通过单一带宽模型的 3% 交叉验证；"
                    "该点改用实测 ASTRA-SIM profile"
                )

    lines += [
        "",
        "## 分阶段逐点对照",
        "",
        "| D (MiB) | S1 实测/预测 (µs) | S2 实测/预测 (µs) | S3 实测/预测 (µs) | 总实测/预测 (µs) | S1/S2/S3 占比 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for size in sizes:
        measured: dict[str, float] = {}
        predicted = {
            "s1": size / (fast * 1_000.0),
            "s2": (size / 2) / (slow * 1_000.0),
            "s3": (size / 2) / (fast * 1_000.0),
        }
        missing = [stage for stage in STAGES if (size, stage) not in indexed]
        if missing:
            failures.append(f"D={size} 缺少阶段 {missing}")
            continue
        for stage in STAGES:
            row = indexed[(size, stage)]
            measured[stage] = float(row["t_median_us"])
            target = float(row["target_bw_gbps"])
            achieved = float(row["achieved_bw_gbps"])
            if abs(achieved - target) / target > 0.03:
                warnings.append(
                    f"D={size} {stage} 的目标带宽误差 >=3%；ASTRA-SIM profile 使用实测值"
                )
        total_m = sum(measured.values())
        total_p = sum(predicted.values())
        shares = "/".join(fmt(100 * measured[stage] / total_m, 1) + "%" for stage in STAGES)
        cells = [f"{fmt(measured[s])}/{fmt(predicted[s])}" for s in STAGES]
        lines.append(
            f"| {size / (1 << 20):g} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{fmt(total_m)}/{fmt(total_p)} | {shares} |"
        )

    lines += ["", "## 线性度（强制过原点）", "", "| 阶段 | µs/byte | R²(origin) |", "|---|---:|---:|"]
    for stage in STAGES:
        points = [(float(size), float(indexed[(size, stage)]["t_median_us"])) for size in sizes if (size, stage) in indexed]
        slope, r2 = through_origin_fit(points)
        lines.append(f"| {stage.upper()} | {slope:.9g} | {r2:.6f} |")
        if r2 < 0.99:
            warnings.append(
                f"{stage} 过原点线性拟合 R²={r2:.4f} < 0.99；"
                "单一跨消息大小带宽模型不成立"
            )

    lines += [
        "",
        "## ASTRA-SIM 逐消息参数",
        "",
        "> 每个 D 使用该点实测 median 与 1-byte 操作延迟反推序列化带宽。"
        "这些参数用于逐点复现，不构成跨消息大小预测。",
        "",
        "| D (MiB) | X/S3: B (GB/s), L (ns) | Y/S1: B (GB/s), L (ns) | W/S2: B (GB/s), L (ns) |",
        "|---:|---:|---:|---:|",
    ]
    for size in sizes:
        if size not in astra_profiles:
            continue
        profile = astra_profiles[size]
        cells = [
            f"{profile['bandwidth_gbps'][i]:.3f}, {profile['latency_ns'][i]:.3f}"
            for i in range(3)
        ]
        lines.append(f"| {size / (1 << 20):g} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += ["", f"配置目录：`{astra_dir}`。"]

    clock_text, _ = sample_clock_report(sample_path)
    lines += [
        "",
        "## 频率与并发链路独立性",
        "",
        f"- 标定文件记录锁频：`{calibration.get('sm_clock_locked', False)}`，频率 `{calibration.get('sm_clock_locked_mhz', 0)} MHz`。",
        f"- {clock_text}",
        f"- {global_clock_text}",
        "",
        *concurrency_report(args.concurrency),
        "",
        "## 验证强度与已知限制",
        "",
        "当前模式为每个消息点分别从真机提取 `B(D)` 与 `L_eff`。因此 ASTRA-SIM 单点接近是参数回放而非跨消息大小预测；可验证的是相同参数口径下的阶段映射、通信量和模拟器执行路径。",
        "",
        "本实验仍有以下边界：2×2 mesh 不产生论文大规模拓扑中的局部热点；模拟链路共享 NVSwitch 而非物理隔离；三个阶段均退化为二成员单跳操作；2–256 MiB 只覆盖带宽主导区，未验证延迟项；单 pair 分阶段之和是算法总时间下界，不能给出四 pair 的 straggler 或跨阶段 overlap。与 ASTRA-SIM 对照时必须保持同一测量口径。",
        "",
        "## 验收结论",
        "",
    ]
    if failures:
        lines.append("**未通过。**")
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("**数据完整性通过。** 五个消息点均已生成独立 ASTRA-SIM 参数。")
    if warnings:
        lines += ["", "### 建模警告", ""]
        lines.extend(f"- {warning}" for warning in warnings)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.out} and {astra_dir}; "
        f"acceptance failures: {len(failures)}, modeling warnings: {len(warnings)}"
    )
    return 2 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
