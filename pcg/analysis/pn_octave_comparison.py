"""
pn_octave_comparison.py

Generates the octaves figure for the Perlin section: the same world rendered at
2, 4, and 6 octaves, side by side, sharing one colour scale.
"""

from pcg.terrain.pn.gen_full_octaves import pn_gen_full_octaves
from pcg.render.pn_oct_comp_plot import render_octave_comparison
from pcg.terrain.pn.helper import (DEFAULT_PERSISTENCE, DEFAULT_LACUNARITY,
                                   DEFAULT_FREQUENCY)


SEED = 0x28d25ddd626642c1
OCTAVE_LIST = [2, 4, 6]

grids = [
    pn_gen_full_octaves(SEED,
                        frequency=DEFAULT_FREQUENCY,
                        octaves=o,
                        persistence=DEFAULT_PERSISTENCE,
                        lacunarity=DEFAULT_LACUNARITY)
    for o in OCTAVE_LIST
]

render_octave_comparison(
    grids, OCTAVE_LIST, seed=SEED,
    shared_params={"persistence": DEFAULT_PERSISTENCE,
                   "lacunarity": DEFAULT_LACUNARITY,
                   "frequency": DEFAULT_FREQUENCY},
    mode="both",
)
