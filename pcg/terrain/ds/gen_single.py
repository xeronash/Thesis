"""
gen_single.py

Single-cell point query recursive walk-up
"""

from pcg.core.grid_config import GRID_SIZE
from pcg.terrain.ds.helper import ROUGHNESS, level_of, parents
from pcg.core.hashing import hash_cell, bits_to_signed
from pcg.core.seeds import validate_seed


def _resolve(coord, world_seed, memo, roughness):
    """Recursive walk-up with memo"""
    if coord in memo:
        return memo[coord]

    i, j = coord
    L = level_of(i, j)
    if L == 0:
        v = bits_to_signed(hash_cell(world_seed, 0, i, j))
        memo[coord] = v
        return v

    amplitude = 2.0 ** (-roughness * (L - 1))

    ps = parents(i, j, L)
    parent_avg = (sum(_resolve(p, world_seed, memo, roughness) for p in ps) /
                  len(ps))

    disp = bits_to_signed(hash_cell(world_seed, L, i, j))

    v = parent_avg + disp * amplitude
    memo[coord] = v

    return v


def ds_gen_single(coord, world_seed, *, roughness: float = ROUGHNESS):
    """
    Height at a single grid cell 'coord = (i, j)'
    """
    validate_seed(world_seed)

    i, j = coord

    if not (0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE):
        raise ValueError(f"coord {coord} out of bounds")

    memo = {}

    return _resolve((int(i), int(j)), world_seed, memo, roughness)
