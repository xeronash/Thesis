"""
perlin_noise.py
"""

from pcg.core.seeds import TESTING_SEED, random_seed
from pcg.core.grid_config import GRID_SIZE
from pcg.terrain.pn.gen_full_basic import pn_gen_full_basic
from pcg.terrain.pn.gen_full_octaves import pn_gen_full_octaves
from pcg.terrain.pn.gen_single import pn_gen_single
from pcg.terrain.pn.gen_rect import pn_gen_rect
from pcg.render.heightmap import plot_2d_3d


# --- Seed -------------------------------------------------------------------
SEED = TESTING_SEED
# SEED = random_seed()
# SEED = 0x28d25ddd626642c1

# --- Terrain parameters (tune these) ---------------------------------------
OCTAVES      = 6        # number of fBm layers for the multi-octave runs
PERSISTENCE  = 0.5      # amplitude multiplier per octave
LACUNARITY   = 2.0      # frequency multiplier per octave
FREQUENCY    = 1 / 128  # grid->noise scaling for the multi-octave runs
BASIC_FREQ   = 1 / 32   # higher freq for the 1-octave run so structure shows


# === 1) FULL GRID - basic, single octave ===================================
# grid = pn_gen_full_basic(SEED, frequency=BASIC_FREQ)
# plot_2d_3d(grid, algo="pn", name="pn_gen_full_basic", seed=SEED,
#        params={"octaves": 1, "frequency": BASIC_FREQ})


# === 2) FULL GRID - multi-octave fBm terrain ===============================
# grid = pn_gen_full_octaves(TESTING_SEED, frequency=FREQUENCY, octaves=OCTAVES,
#                            persistence=PERSISTENCE, lacunarity=LACUNARITY)
# plot_2d_3d(grid, algo="pn", name="pn_gen_full_octaves", seed=SEED,
#        params={"octaves": OCTAVES, "persistence": PERSISTENCE,
#                "lacunarity": LACUNARITY, "frequency": FREQUENCY})


# === 3) SINGLE CELL - point query ==========================================
# coord = (GRID_SIZE // 2, GRID_SIZE // 2)
# coord = (55, 127)
# height = pn_gen_single(coord, SEED, frequency=FREQUENCY, octaves=OCTAVES,
#                        persistence=PERSISTENCE, lacunarity=LACUNARITY)
# print(f"pn_gen_single{coord} = {height}")

# assert grid[55, 127] == height


# === 4) RECT - independent-sample loop =====================================
# origin, size = (96, 96), (64, 64)
# rect = pn_gen_rect(origin, size, SEED, frequency=FREQUENCY, octaves=OCTAVES,
#                    persistence=PERSISTENCE, lacunarity=LACUNARITY)
# plot_2d_3d(rect, algo="pn", name="pn_gen_rect", seed=SEED,
#        params={"octaves": OCTAVES, "persistence": PERSISTENCE,
#                "lacunarity": LACUNARITY, "frequency": FREQUENCY,
#                "origin": origin, "rect_size": size})
