"""
gen_full_basic.py

Full-grid Perlin single octave terrain
"""

import numpy as np

from pcg.core.grid_config import GRID_SIZE
from pcg.terrain.pn.helper import pn_sample, DEFAULT_FREQUENCY
from pcg.core.seeds import validate_seed


def pn_gen_full_basic(world_seed, *,
                      frequency=DEFAULT_FREQUENCY) -> np.ndarray:
    """Full (GRID_SIZE x GRID_SIZE) single-octave Perlin heightmap"""

    validate_seed(world_seed)

    out = np.zeros((GRID_SIZE, GRID_SIZE))

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):

            out[i, j] = pn_sample((i * frequency, j * frequency), world_seed,
                                  octaves=1)

    return out
