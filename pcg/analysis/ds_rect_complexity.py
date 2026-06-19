"""
ds_rect_complexity.py

Figure (A) for the rectangular diamond-square query.
"""

import os
import random
import statistics

import matplotlib.pyplot as plt

OUT = os.path.join("outputs", "analysis")

AREA_N = 10
AREA_SIDES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1025]
PLACEMENTS = 3
CONE_SAMPLES = 5000
SEED = 1


def _v2(x, n):
    if x == 0:
        return n
    return (x & -x).bit_length() - 1


def level_of(i, j, n):
    return n - min(_v2(i, n), _v2(j, n))


def parents(i, j, level, n):
    grid_last = 1 << n
    half_l = grid_last >> level
    is_center = (i // half_l) % 2 == 1 and (j // half_l) % 2 == 1
    if is_center:
        cand = [(i - half_l, j - half_l), (i - half_l, j + half_l),
                (i + half_l, j - half_l), (i + half_l, j + half_l)]
    else:
        cand = [(i - half_l, j), (i + half_l, j),
                (i, j - half_l), (i, j + half_l)]
    return [(a, b) for (a, b) in cand
            if 0 <= a <= grid_last and 0 <= b <= grid_last]


def cone_size(coord, n):
    seen = set()
    stack = [coord]
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        L = level_of(c[0], c[1], n)
        if L > 0:
            stack.extend(parents(c[0], c[1], L, n))
    return len(seen)


def rect_union_size(i0, j0, h, w, n):
    seen = set()
    stack = [(i0 + di, j0 + dj) for di in range(h) for dj in range(w)]
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        L = level_of(c[0], c[1], n)
        if L > 0:
            stack.extend(parents(c[0], c[1], L, n))
    return len(seen)


def _origin(rng, n, h, w):
    last_i, last_j = (1 << n) + 1 - h, (1 << n) + 1 - w
    i0 = rng.randrange(last_i) if last_i > 0 else 0
    j0 = rng.randrange(last_j) if last_j > 0 else 0
    return i0, j0


def mean_cone(n, rng):
    N = (1 << n) + 1
    return statistics.mean(
        cone_size((rng.randrange(N), rng.randrange(N)), n)
        for _ in range(CONE_SAMPLES)
    )


def measure_area(n, sides, rng):
    """Panel (A): per-cell cost vs area """
    full = ((1 << n) + 1) ** 2
    rows = []
    for s in sides:
        if s > (1 << n) + 1:
            continue
        reps = PLACEMENTS if (s * s) < 0.25 * full else 1
        vals = [rect_union_size(*_origin(rng, n, s, s), s, s, n)
                for _ in range(reps)]
        shared = statistics.mean(vals)
        rows.append({"area": s * s, "shared": shared, "amort": shared / (s * s)})
    return rows


def figure_rect_complexity(area_rows, area_n, area_cone):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    # ---- (A) per-cell cost vs area, single cell -------------
    a = [r["area"] for r in area_rows]
    pc = [r["amort"] for r in area_rows]
    N = (1 << area_n) + 1
    full = N * N

    ax.plot(a, pc, "o-", color="#2166ac", ms=5, lw=1.4,
            label="ds_gen_rect (measured)")
    ax.axhline(area_cone, ls=":", color="dimgray", lw=1.3, 
               label=r"naive, no sharing: 1 cone / cell  ($\Theta(hw\log N)$)")
    ax.axhline(1.0, ls="--", color="black", lw=1.3, 
               label=r"optimal: 1 resolve / cell  ($\Theta(hw)$)")

    # rect as a fraction of the whole grid
    trans = ax.get_xaxis_transform()
    for frac, lab in [(0.05, "5%"), (0.1, "10%"), (0.5, "50%"), (1.0, "full grid")]:
        x = frac * full
        ax.axvline(x, ls="-", color="0.65", lw=0.9)
        ax.text(x * 0.86, 0.5, lab, transform=trans, rotation=90, 
                va="center", ha="center", fontsize=7.5, color="0.35")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"rect area  $h\,w$")
    ax.set_ylabel("resolves per output cell")
    ax.set_title("Rectangular query cost versus query area (grid n = 10)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "rect_complexity_minimal.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def main():
    rng = random.Random(SEED)

    area_rows = measure_area(AREA_N, AREA_SIDES, rng)
    area_cone = mean_cone(AREA_N, rng)
    N = (1 << AREA_N) + 1
    print(
        f"panel A (grid n={AREA_N}, N^2={N*N}): per cell "
        f"{area_rows[0]['amort']:.1f} (1 cell) -> {area_rows[-1]['amort']:.2f} "
        f"(full grid); naive ceiling = {area_cone:.0f}"
    )
    figure_rect_complexity(area_rows, AREA_N, area_cone)


if __name__ == "__main__":
    main()
