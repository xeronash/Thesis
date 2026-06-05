"""
gen_full_hashed.py

Full-grid diamond-square with hash-keyed displacements
"""

import numpy as np

from pcg.core.grid_config import GRID_SIZE
from pcg.terrain.ds.helper import ROUGHNESS
from pcg.core.hashing import hash_cell, bits_to_signed
from pcg.core.seeds import validate_seed


def ds_gen_full_hashed(world_seed: int, *,
                       roughness: float = ROUGHNESS) -> np.ndarray:
    """Full (GRID_SIZE x GRID_SIZE) heightmap from hashed displacements."""

    validate_seed(world_seed)

    size = GRID_SIZE
    grid = np.zeros((size, size))

    # Four corners (level 0)
    grid[0, 0] = bits_to_signed(hash_cell(world_seed, 0, 0, 0))
    grid[0, -1] = bits_to_signed(hash_cell(world_seed, 0, 0, size - 1))
    grid[-1, 0] = bits_to_signed(hash_cell(world_seed, 0, size - 1, 0))
    grid[-1, -1] = bits_to_signed(hash_cell(world_seed, 0, size - 1, size - 1))

    step = size - 1
    amplitude = 1.0
    level = 1

    while step > 1:
        half = step // 2

        # Diamond step: square centers
        for i in range(0, size - 1, step):
            for j in range(0, size - 1, step):
                avg = (grid[i, j] + grid[i + step, j]
                       + grid[i, j + step] + grid[i + step, j + step]) / 4

                bits = hash_cell(world_seed, level, i + half, j + half)
                grid[i + half, j + half] = (
                    avg + bits_to_signed(bits) * amplitude
                    )

        # Square step: edge midpoints
        for i in range(0, size, half):
            start = half if (i // half) % 2 == 0 else 0
            for j in range(start, size, step):
                nb = []
                if i - half >= 0:
                    nb.append(grid[i - half, j])
                if i + half < size:
                    nb.append(grid[i + half, j])
                if j - half >= 0:
                    nb.append(grid[i, j - half])
                if j + half < size:
                    nb.append(grid[i, j + half])

                bits = hash_cell(world_seed, level, i, j)
                grid[i, j] = (
                    sum(nb) / len(nb) + bits_to_signed(bits) * amplitude
                    )

        step = half
        amplitude *= 2 ** (-roughness)
        level += 1

    return grid
