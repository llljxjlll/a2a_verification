#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_results", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AnalysisTests(unittest.TestCase):
    def test_through_origin_fit(self) -> None:
        slope, r2 = MODULE.through_origin_fit([(1, 2), (2, 4), (4, 8)])
        self.assertTrue(math.isclose(slope, 2.0))
        self.assertTrue(math.isclose(r2, 1.0))

    def test_clock_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            path.write_text(
                "sm_clock_gpu0_mhz,sm_clock_gpu1_mhz\n"
                "1215,1215\n1215,1200\n",
                encoding="utf-8",
            )
            text, rejected = MODULE.sample_clock_report(path)
            self.assertIn("1215 MHz", text)
            self.assertEqual(rejected, 1)

    def test_complete_synthetic_report(self) -> None:
        fields = [
            "mode", "stage", "pair", "data_size_bytes", "per_link_bytes",
            "tier", "target_bw_gbps", "achieved_bw_gbps", "ratio_R",
            "chunk_bytes", "pace_clamped", "sm_clock_mhz", "iters",
            "t_mean_us", "t_median_us", "t_p99_us", "t_min_us", "t_max_us",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "results.csv"
            calibration = root / "calibration.json"
            report = root / "FINDINGS.md"
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for size_mib in (2, 8, 32, 64, 256):
                    size = size_mib * (1 << 20)
                    for stage, pair, factor, bandwidth in (
                        ("s1", "0-2", 1.0, 40.0),
                        ("s2", "0-4", 0.5, 40.0 / 7.5),
                        ("s3", "0-1", 0.5, 40.0),
                    ):
                        elapsed = size * factor / (bandwidth * 1_000.0)
                        writer.writerow({
                            "mode": "sweep", "stage": stage, "pair": pair,
                            "data_size_bytes": size,
                            "per_link_bytes": int(size * factor),
                            "tier": "slow" if stage == "s2" else "fast",
                            "target_bw_gbps": bandwidth,
                            "achieved_bw_gbps": bandwidth, "ratio_R": 7.5,
                            "chunk_bytes": 4 << 20, "pace_clamped": 0,
                            "sm_clock_mhz": 1215, "iters": 100,
                            "t_mean_us": elapsed, "t_median_us": elapsed,
                            "t_p99_us": elapsed, "t_min_us": elapsed,
                            "t_max_us": elapsed,
                        })
            calibration.write_text(json.dumps({
                "fast_tier": {
                    "B_raw_gbps": 200, "B_target_gbps": 40,
                    "B_achieved_gbps": 40, "latency_us": 5, "O_ns": 10,
                    "calib_iters": 2,
                    "cross_validation": {
                        "D_2MiB_achieved_gbps": 40,
                        "D_256MiB_achieved_gbps": 40,
                    },
                },
                "slow_tier": {
                    "B_raw_gbps": 200, "B_target_gbps": 40 / 7.5,
                    "B_achieved_gbps": 40 / 7.5, "latency_us": 5,
                    "O_ns": 10, "calib_iters": 2,
                    "cross_validation": {
                        "D_2MiB_achieved_gbps": 40 / 7.5,
                        "D_256MiB_achieved_gbps": 40 / 7.5,
                    },
                },
                "chunk_bytes": 4 << 20, "sm_clock_locked_mhz": 1215,
                "sm_clock_locked": True, "pace_timer": "globaltimer",
            }), encoding="utf-8")
            argv = ["analyze_results.py", "--csv", str(summary),
                    "--calibration", str(calibration), "--out", str(report),
                    "--strict"]
            with patch.object(sys, "argv", argv):
                self.assertEqual(MODULE.main(), 0)
            self.assertIn("**通过。**", report.read_text(encoding="utf-8"))

    def test_global_per_gpu_clock_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sample_path = Path(directory) / "results.samples.csv"
            sample_path.write_text(
                "mode,stage,pair,data_size_bytes,iteration,endpoint0_us,endpoint1_us,"
                "pair_max_us,physical_gpu0,physical_gpu1,sm_clock_gpu0_mhz,"
                "sm_clock_gpu1_mhz,pace_clamped\n"
                "sweep,s1,0-2,2097152,0,10,11,11,0,2,1215,1215,0\n"
                "sweep,s1,0-2,2097152,1,99,99,99,0,2,1200,1215,0\n"
                "sweep,s1,0-2,2097152,2,12,13,13,0,2,1215,1215,0\n",
                encoding="utf-8",
            )
            rows = [{
                "mode": "sweep", "stage": "s1", "pair": "0-2",
                "data_size_bytes": "2097152", "per_link_bytes": "2097152",
                "tier": "fast", "target_bw_gbps": "40",
                "achieved_bw_gbps": "1", "ratio_R": "7.5",
                "chunk_bytes": str(4 << 20), "pace_clamped": "0",
                "sm_clock_mhz": "0", "iters": "3", "t_mean_us": "0",
                "t_median_us": "0", "t_p99_us": "0", "t_min_us": "0",
                "t_max_us": "0",
            }]
            filtered, report = MODULE.globally_filter_samples(rows, sample_path)
            self.assertEqual(filtered[0]["iters"], "2")
            self.assertEqual(float(filtered[0]["t_median_us"]), 12.0)
            self.assertIn("GPU0=1215 MHz", report)


if __name__ == "__main__":
    unittest.main()
