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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--concurrency", type=Path)
    parser.add_argument("--out", type=Path, default=Path("FINDINGS.md"))
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
            failures.append(f"{label} 层闭环标定用了 {tier['calib_iters']} 轮，可能存在非线性")
        for endpoint, achieved in tier.get("cross_validation", {}).items():
            if abs(float(achieved) - float(tier["B_target_gbps"])) / float(tier["B_target_gbps"]) > 0.03:
                failures.append(f"{label} 层 {endpoint} 未通过 3% 交叉验证")

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
                failures.append(f"D={size} {stage} 的达成带宽误差 >=3%")
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
            failures.append(f"{stage} 过原点线性拟合 R²={r2:.4f} < 0.99")

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
        "标定已经从真机取走 `B` 与 `L_eff`，所以总时间单点接近并不是强证据。真正的对照量是三个阶段的独立耗时占比，以及耗时随 D 的过原点线性度。R=7.5 时 S3 只占约 10%，即使 S3 有 30% 建模误差，也只造成约 3% 总时间误差，因此 S3 的验证强度天然较弱。",
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
        lines.append("**通过。** 无 pacing 钳位；各点带宽在目标 3% 内；扫描完整且各阶段过原点线性度满足 R²≥0.99。")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}; acceptance failures: {len(failures)}")
    return 2 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
