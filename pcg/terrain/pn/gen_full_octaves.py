"""
gen_full_octaves.py

Full-grid full octaves perlin terrain
"""

import numpy as np

from pcg.core.seeds import validate_seed
from pcg.core.grid_config import GRID_SIZE
from pcg.terrain.pn.helper import (
    pn_sample,
    DEFAULT_FREQUENCY,
    DEFAULT_OCTAVES,
    DEFAULT_PERSISTENCE,
    DEFAULT_LACUNARITY,
)


def pn_gen_full_octaves(
    world_seed,
    *,
    frequency=DEFAULT_FREQUENCY,
    octaves=DEFAULT_OCTAVES,
    persistence=DEFAULT_PERSISTENCE,
    lacunarity=DEFAULT_LACUNARITY,
) -> np.ndarray:
    """Full (GRID_SIZE x GRID_SIZE) multi-octave Perlin heightmap"""

    validate_seed(world_seed)

    out = np.zeros((GRID_SIZE, GRID_SIZE))

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):

            out[i, j] = pn_sample(
                (i * frequency, j * frequency), world_seed,
                octaves=octaves, persistence=persistence,
                lacunarity=lacunarity,
            )

    return out
