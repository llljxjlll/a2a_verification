#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parents[1] / "astra_sim"
sys.path.insert(0, str(SCRIPT_DIR))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = load("mesh_rail_gen_et", "gen_et.py")
RUN = load("mesh_rail_run_astra", "run_astra_sim.py")


class AstraSimTests(unittest.TestCase):
    def test_completion_parser_uses_slowest_rank(self) -> None:
        output = "\n".join(
            f"[info] sys[{rank}] finished, {1000 + rank} cycles, exposed communication 1 cycles."
            for rank in range(8)
        )
        self.assertEqual(RUN.parse_completion_cycles(output), 1007)

    def test_completion_parser_rejects_missing_rank(self) -> None:
        output = "\n".join(
            f"sys[{rank}] finished, 1000 cycles, exposed communication 1 cycles."
            for rank in range(7)
        )
        with self.assertRaises(ValueError):
            RUN.parse_completion_cycles(output)

    def test_probe_factor_snapping(self) -> None:
        cycles = math.ceil((1 << 20) / 2 / 100.0)
        factor, raw, error = RUN.infer_factor(cycles)
        self.assertEqual(factor, 0.5)
        self.assertLess(error, 0.01)
        self.assertGreater(raw, 0.5)

    def test_comm_size_inverts_probe_factor(self) -> None:
        self.assertEqual(GEN.comm_size_for(2 << 20, 1.0), 2 << 20)
        self.assertEqual(GEN.comm_size_for(1 << 20, 0.5), 2 << 20)

    def test_network_is_always_fully_connected(self) -> None:
        profile = {
            "dimension_order": ["X", "Y", "W"],
            "bandwidth_gbps": [10, 20, 3],
            "latency_ns": [1000, 2000, 3000],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.yml"
            RUN.write_network_config(profile, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn(
            "topology: [ FullyConnected, FullyConnected, FullyConnected ]", text
        )
        self.assertIn("npus_count: [ 2, 2, 2 ]", text)


if __name__ == "__main__":
    unittest.main()
