"""
verify.py

Correctness + determinism suite. Run from pcg/:

    python verify.py

Checks (each prints PASS/FAIL; the script exits non-zero if anything fails):

  DS equivalence
    - ds_gen_full_hashed == ds_gen_rect over the whole grid   (all cells)
    - ds_gen_full_hashed == ds_gen_single                     (random sample)
    - ds_gen_rect == looping ds_gen_single over the same cells (shared vs
      independent memo give the SAME values)
  DS determinism / statelessness
    - calling each generator twice gives identical output
    - results are independent of call order (interleave calls)
  DS validation
    - bad seeds and out-of-bounds coords raise ValueError
  PN consistency
    - pn_gen_full_octaves == looping pn_gen_single            (random sample)
    - pn_gen_rect == slice of the full grid
  Cross-seed
    - DS equivalence holds for several different seeds

Note: ds_gen_full_basic (sequential PRNG) is deliberately NOT compared for
equality against the hashed variants - it uses a different random source, so a
different (but equally valid) terrain is expected. Only the hashed family must
agree.
"""

import sys
import numpy as np

from pcg.core.grid_config import GRID_SIZE
from pcg.core.seeds import TESTING_SEED, random_seed

from pcg.terrain.ds.gen_full_basic import ds_gen_full_basic
from pcg.terrain.ds.gen_full_hashed import ds_gen_full_hashed
from pcg.terrain.ds.gen_single import ds_gen_single
from pcg.terrain.ds.gen_rect import ds_gen_rect

from pcg.terrain.pn.gen_full_octaves import pn_gen_full_octaves
from pcg.terrain.pn.gen_single import pn_gen_single
from pcg.terrain.pn.gen_rect import pn_gen_rect


TOL = 1e-9
_results = []


def check(name, ok, detail=""):
    _results.append(ok)
    flag = "PASS" if ok else "FAIL"
    line = f"  [{flag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)


def sample_cells(n, seed=0):
    rng = np.random.default_rng(seed)
    return list(zip(rng.integers(0, GRID_SIZE, n), rng.integers(0, GRID_SIZE, n)))


def verify_ds(seed):
    print(f"DS, seed=0x{seed:x}")
    full = ds_gen_full_hashed(seed)

    # whole-grid rect uses the single-query engine with one shared memo
    whole = ds_gen_rect((0, 0), (GRID_SIZE, GRID_SIZE), seed)
    check("full_hashed == rect(whole grid)  [all cells]",
          np.allclose(full, whole, atol=TOL),
          f"max|diff|={np.max(np.abs(full-whole)):.2e}")

    # fresh-memo single query on a sample
    cells = sample_cells(400, seed=1)
    diffs = [abs(ds_gen_single((int(i), int(j)), seed) - full[i, j]) for i, j in cells]
    check("full_hashed == single  [400 random cells]",
          max(diffs) < TOL, f"max|diff|={max(diffs):.2e}")

    # shared-memo rect values == independent single queries
    rect = ds_gen_rect((90, 70), (40, 50), seed)
    indep = np.array([[ds_gen_single((90 + di, 70 + dj), seed) for dj in range(50)]
                      for di in range(40)])
    check("rect (shared memo) == independent single queries",
          np.allclose(rect, indep, atol=TOL),
          f"max|diff|={np.max(np.abs(rect-indep)):.2e}")


def verify_determinism(seed):
    print("DS determinism / statelessness")
    check("full_hashed twice identical", np.array_equal(ds_gen_full_hashed(seed),
                                                         ds_gen_full_hashed(seed)))
    check("full_basic twice identical", np.array_equal(ds_gen_full_basic(seed),
                                                        ds_gen_full_basic(seed)))
    # interleaving calls must not change results (no surviving state)
    a1 = ds_gen_single((101, 37), seed)
    _ = ds_gen_rect((0, 0), (8, 8), seed)
    _ = ds_gen_full_hashed(seed)
    a2 = ds_gen_single((101, 37), seed)
    check("single value stable across interleaved calls", abs(a1 - a2) < TOL)


def verify_validation():
    print("DS validation")
    def raises(fn):
        try:
            fn(); return False
        except ValueError:
            return True
        except Exception:
            return False
    check("bad seed (negative) raises", raises(lambda: ds_gen_single((0, 0), -1)))
    check("bad seed (too big) raises", raises(lambda: ds_gen_single((0, 0), 2**64)))
    check("bad seed (float) raises", raises(lambda: ds_gen_single((0, 0), 1.5)))
    check("OOB coord raises", raises(lambda: ds_gen_single((GRID_SIZE, 0), TESTING_SEED)))
    check("bad rect size raises", raises(lambda: ds_gen_rect((0, 0), (0, 5), TESTING_SEED)))


def verify_pn(seed):
    print(f"PN consistency, seed=0x{seed:x}")
    full = pn_gen_full_octaves(seed)
    cells = sample_cells(400, seed=2)
    diffs = [abs(pn_gen_single((int(i), int(j)), seed) - full[i, j]) for i, j in cells]
    check("full_octaves == single  [400 random cells]",
          max(diffs) < 1e-12, f"max|diff|={max(diffs):.2e}")
    rect = pn_gen_rect((90, 70), (40, 50), seed)
    check("rect == slice of full grid",
          np.allclose(rect, full[90:130, 70:120], atol=1e-12),
          f"max|diff|={np.max(np.abs(rect-full[90:130,70:120])):.2e}")


def main():
    seeds = [TESTING_SEED, random_seed(), random_seed()]
    for s in seeds:
        verify_ds(s)
        print()
    verify_determinism(TESTING_SEED); print()
    verify_validation(); print()
    verify_pn(TESTING_SEED); print()

    n_pass = sum(_results); n = len(_results)
    print("=" * 48)
    print(f"  {n_pass}/{n} checks passed")
    print("=" * 48)
    sys.exit(0 if n_pass == n else 1)


if __name__ == "__main__":
    main()
