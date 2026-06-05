"""
helper.py

Helper functions for the different diamond_square algorithm versions

    - ROUGHNESS               : amplitude falloff per level
    - v2 / level_of           : which subdivision level a cell belongs to
    - is_center / parents     : which 4 cells a cell depends on
"""

from pcg.core.grid_config import GRID_N, GRID_SIZE, GRID_LAST


# Roughness controls how fast the displacement amplitude shrinks each level
ROUGHNESS = 1.0


def v2(x: int) -> int:
    """2-adic valuation, number of trailing zero bits in x
    Returns GRID_N for x == 0 so corner coordinates resolve to level 0"""
    if x == 0:
        return GRID_N
    count = 0
    while x & 1 == 0:
        x >>= 1
        count += 1
    return count


def level_of(i: int, j: int) -> int:
    """Return the level (0..GRID_N) at which cell (i, j) was placed"""
    return GRID_N - min(v2(i), v2(j))


def is_center(i: int, j: int, level: int) -> bool:
    """
    True if (i, j) is a center cell (placed by the diamond step), both
    coordinates are odd multiples of half_L. Such cells have diagonal
    (X-shape) parents. Otherwise the cell is an edge midpoint (square step)
    with cardinal (+-shape) parents
    """
    half_L = GRID_LAST >> level
    return ((i // half_L) % 2 == 1) and ((j // half_L) % 2 == 1)


def parents(i: int, j: int, level: int) -> list:
    """
    The cells whose values are needed to compute (i, j) at this level.

    - Center (square step) = 4 diagonal parents at distance half_L
    - Edge midpoint (diamond step) = 4 cardinal parents at distance half_L
      dropping any that fall outside the border of the grid (terrain)
    """
    half_L = GRID_LAST >> level
    if is_center(i, j, level):
        candidates = [
            (i - half_L, j - half_L), (i + half_L, j - half_L),
            (i - half_L, j + half_L), (i + half_L, j + half_L),
        ]
    else:
        candidates = [
            (i - half_L, j), (i + half_L, j),
            (i, j - half_L), (i, j + half_L),
        ]
    return [
        (pi, pj) for (pi, pj) in candidates
        if 0 <= pi < GRID_SIZE and 0 <= pj < GRID_SIZE
    ]
