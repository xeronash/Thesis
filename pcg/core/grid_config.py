"""
grid_config.py

Shared output resolution for BOTH terrain algorithms.

Both diamond-square and Perlin render onto the same square grid so the
benchmarks compare the same terrain sizes.

    GRID_N    : grid power. n=8 gives a (2^8 + 1) = 257 grid on each side.
    GRID_SIZE : side length in cells, 2^n + 1.
    GRID_LAST : last valid index, 2^n. Used by diamond-square for the
                subdivision structure (Perlin only needs GRID_SIZE).
"""

GRID_N = 8
GRID_SIZE = (2 ** GRID_N) + 1
GRID_LAST = GRID_SIZE - 1
