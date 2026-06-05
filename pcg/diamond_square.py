"""
diamond_square.py
"""

from pcg.core.seeds import TESTING_SEED, random_seed
from pcg.core.grid_config import GRID_N, GRID_SIZE
from pcg.terrain.ds.helper import ROUGHNESS
from pcg.terrain.ds.gen_full_basic import ds_gen_full_basic
from pcg.terrain.ds.gen_full_hashed import ds_gen_full_hashed
from pcg.terrain.ds.gen_single import ds_gen_single
from pcg.terrain.ds.gen_rect import ds_gen_rect
from pcg.render.heightmap import plot_2d_3d

ROUGHNESS_RUN = ROUGHNESS       # Global default = 1.0 (halfing each step)
# ROUGHNESS_RUN = 0.4

# --- Seed -------------------------------------------------------------------
SEED = TESTING_SEED
# SEED = random_seed()

# Shared params stamped into every DS plot title.
DS_PARAMS = {"grid_n": GRID_N, "size": GRID_SIZE, "roughness": ROUGHNESS_RUN}


# === 1) FULL GRID - basic PRNG =============================================
grid = ds_gen_full_basic(SEED, roughness=ROUGHNESS_RUN)
plot_2d_3d(grid, algo="ds", name="ds_gen_full_basic", seed=SEED, params=DS_PARAMS)


# === 2) FULL GRID - hashed displacements ===================================
# grid = ds_gen_full_hashed(SEED, roughness=ROUGHNESS_RUN)
# plot_2d_3d(grid, algo="ds", name="ds_gen_full_hashed", seed=SEED, params=DS_PARAMS)


# === 3) SINGLE CELL - point query ==========================================
# coord = (GRID_SIZE // 2, GRID_SIZE // 2)
# coord = (55, 214)
# height = ds_gen_single(coord, SEED, roughness=ROUGHNESS_RUN)
# print(f"ds_gen_single{coord} = {height}")

# assert grid[55, 214] == height


# === 4) RECT - shared-memo region query ====================================
# origin, size = (96, 96), (64, 64)
# rect = ds_gen_rect(origin, size, SEED, roughness=ROUGHNESS_RUN)
# plot_2d_3d(rect, algo="ds", name="ds_gen_rect", seed=SEED,
#        params={**DS_PARAMS, "origin": origin, "rect_size": size})
