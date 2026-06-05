"""
gen_rect.py

Rectangular diamond-square region query

This is the single-cell walk-up engine seeded with every cell in the rect,
all sharing one memo
"""

import numpy as np

from pcg.core.grid_config import GRID_SIZE
from pcg.terrain.ds.helper import ROUGHNESS
from pcg.terrain.ds.gen_single import _resolve
from pcg.core.seeds import validate_seed


def ds_gen_rect(origin, size, world_seed: int, *,
                roughness: float = ROUGHNESS) -> np.ndarray:
    """
    Return a (h x w) array for the rect starting at 'origin = (i0, j0)'
    """

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

    memo = {}
    out = np.zeros((h, w))

    for di in range(h):
        for dj in range(w):

            out[di, dj] = _resolve((i0 + di, j0 + dj), world_seed, memo,
                                   roughness)

    return out
