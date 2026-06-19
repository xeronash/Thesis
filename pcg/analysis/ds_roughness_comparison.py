"""
ds_roughness_comparison.py
"""

from pcg.core.seeds import TESTING_SEED, random_seed
from pcg.core.grid_config import GRID_N, GRID_SIZE
from pcg.terrain.ds.gen_full_basic import ds_gen_full_basic
from pcg.render.ds_roughness_comp_plot import render_roughness_comparison


# SEED = random_seed()
SEED = 0xcb3e37afe0b55803
ROUGHNESS_LIST = [0.5, 1.0, 4.0]

grids = [
    ds_gen_full_basic(SEED, roughness=r)
    for r in ROUGHNESS_LIST
]

render_roughness_comparison(
    grids,
    ROUGHNESS_LIST,
    seed=SEED,
    shared_params={"grid_n": GRID_N, "size": GRID_SIZE},
    mode="both",
)
