"""
gen_single.py

Single-cell Perlin query (multi-octave by default)
"""

from pcg.core.seeds import validate_seed
from pcg.terrain.pn.helper import (
    pn_sample,
    DEFAULT_FREQUENCY,
    DEFAULT_OCTAVES,
    DEFAULT_PERSISTENCE,
    DEFAULT_LACUNARITY
)


def pn_gen_single(
    coord,
    world_seed,
    *,
    frequency=DEFAULT_FREQUENCY,
    octaves=DEFAULT_OCTAVES,
    persistence=DEFAULT_PERSISTENCE,
    lacunarity=DEFAULT_LACUNARITY,
) -> float:
    """Height at a single grid cell `coord = (i, j)`."""

    validate_seed(world_seed)
    i, j = coord

    return pn_sample(
        (i * frequency, j * frequency), world_seed,
        octaves=octaves, persistence=persistence, lacunarity=lacunarity,
    )
