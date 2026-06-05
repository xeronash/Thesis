"""
gen_rect.py

Rectangular region of Perlin noise
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


def pn_gen_rect(
    origin,
    size,
    world_seed,
    *,
    frequency=DEFAULT_FREQUENCY,
    octaves=DEFAULT_OCTAVES,
    persistence=DEFAULT_PERSISTENCE,
    lacunarity=DEFAULT_LACUNARITY,
) -> np.ndarray:
    """Return a (h x w) array for the rect at `origin = (i0, j0)`"""

    validate_seed(world_seed)

    i0, j0 = origin
    h, w = size

    if h <= 0 or w <= 0:
        raise ValueError(f"size must be positive; got {size}")
    if i0 < 0 or j0 < 0 or i0 + h > GRID_SIZE or j0 + w > GRID_SIZE:
        raise ValueError(
            f"rect origin={origin} size={size} out of bounds "
            f"for grid size {GRID_SIZE}"
        )

    out = np.zeros((h, w))

    for di in range(h):
        for dj in range(w):

            out[di, dj] = pn_sample(
                ((i0 + di) * frequency, (j0 + dj) * frequency), world_seed,
                octaves=octaves, persistence=persistence,
                lacunarity=lacunarity,
            )

    return out
