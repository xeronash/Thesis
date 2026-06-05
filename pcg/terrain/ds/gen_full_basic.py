"""
gen_full_basic.py

Classical diamond-square with a PRNG
"""

import random
import numpy as np

from pcg.core.grid_config import GRID_SIZE
from pcg.terrain.ds.helper import ROUGHNESS
from pcg.core.seeds import validate_seed


def ds_gen_full_basic(world_seed: int, *,
                      roughness: float = ROUGHNESS) -> np.ndarray:
    """Full (GRID_SIZE x GRID_SIZE) heightmap from a sequential PRNG"""

    validate_seed(world_seed)
    rng = random.Random(world_seed)

    size = GRID_SIZE
    grid = np.zeros((size, size))

    # Four corners (level 0)
    grid[0, 0] = rng.uniform(-1.0, 1.0)
    grid[0, -1] = rng.uniform(-1.0, 1.0)
    grid[-1, 0] = rng.uniform(-1.0, 1.0)
    grid[-1, -1] = rng.uniform(-1.0, 1.0)

    step = size - 1
    amplitude = 1.0

    while step > 1:
        half = step // 2

        # Diamond step: square centers
        for i in range(0, size - 1, step):
            for j in range(0, size - 1, step):
                avg = (grid[i, j] + grid[i + step, j]
                       + grid[i, j + step] + grid[i + step, j + step]) / 4

                grid[i + half, j + half] = (
                    avg + rng.uniform(-1.0, 1.0) * amplitude
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

                grid[i, j] = (
                    sum(nb) / len(nb) + rng.uniform(-1.0, 1.0) * amplitude
                    )

        step = half
        amplitude *= 2 ** (-roughness)

    return grid
