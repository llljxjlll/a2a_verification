#!/usr/bin/env python3
"""CPU-only checks for the mapping, byte counts, and analytic predictions."""

from __future__ import annotations

import math

MIB = 1 << 20
FAST_GBPS = 40.0
RATIO = 7.5

COORDINATES = {
    rank: (rank % 2, (rank // 2) % 2, rank // 4) for rank in range(8)
}
PAIRS = {
    "s1": {(0, 2), (1, 3), (4, 6), (5, 7)},
    "s2": {(0, 4), (1, 5), (2, 6), (3, 7)},
    "s3": {(0, 1), (2, 3), (4, 5), (6, 7)},
}
FORBIDDEN = {(0, 3), (1, 2), (4, 7), (5, 6)}


def transfer_bytes(stage: str, data_size: int) -> int:
    return data_size if stage == "s1" else data_size // 2


def expected_us(stage: str, data_size: int) -> float:
    bandwidth = FAST_GBPS if stage != "s2" else FAST_GBPS / RATIO
    return transfer_bytes(stage, data_size) / (bandwidth * 1_000.0)


def main() -> None:
    for rank, (x, y, w) in COORDINATES.items():
        assert rank == x + 2 * y + 4 * w

    assert not FORBIDDEN.intersection(set.union(*PAIRS.values()))
    for a, b in PAIRS["s1"]:
        ax, ay, aw = COORDINATES[a]
        bx, by, bw = COORDINATES[b]
        assert ax == bx and ay != by and aw == bw
    for a, b in PAIRS["s2"]:
        ax, ay, aw = COORDINATES[a]
        bx, by, bw = COORDINATES[b]
        assert (ax, ay) == (bx, by) and aw != bw
    for a, b in PAIRS["s3"]:
        ax, ay, aw = COORDINATES[a]
        bx, by, bw = COORDINATES[b]
        assert ax != bx and ay == by and aw == bw

    for data_size in (2, 8, 32, 64, 256):
        data_size *= MIB
        times = {stage: expected_us(stage, data_size) for stage in PAIRS}
        total = sum(times.values())
        shares = {stage: value / total for stage, value in times.items()}
        assert math.isclose(total, 5.25 * data_size / (FAST_GBPS * 1_000.0))
        assert math.isclose(shares["s1"], 1 / 5.25)
        assert math.isclose(shares["s2"], 3.75 / 5.25)
        assert math.isclose(shares["s3"], 0.5 / 5.25)

    # The small S2/S3 point must use its actual 1 MiB chunk, never 4 MiB.
    assert transfer_bytes("s2", 2 * MIB) == 1 * MIB
    print("Static plan validation passed: mapping, legal links, byte counts, and predictions.")


if __name__ == "__main__":
    main()
