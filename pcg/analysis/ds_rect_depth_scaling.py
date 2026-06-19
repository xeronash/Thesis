"""
ds_rect_depth_scaling.py

Depth-scaling figure for the rectangular diamond-square query.
"""

import os
import random
import statistics

import matplotlib.pyplot as plt

OUT = os.path.join("outputs", "analysis")

DEPTH_SIDE = 64        # fixed rectangle side length
DEPTH_MIN = 8          # 64x64 is a sub-region for n >= 8
DEPTH_NS = range(DEPTH_MIN, 15)
PLACEMENTS = 3
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


def measure_depth(side, ns, rng):
    """Per-cell cost for a fixed rectangle as grid depth grows."""
    rows = []
    for n in ns:
        if side >= (1 << n) + 1:
            continue
        vals = [rect_union_size(*_origin(rng, n, side, side), side, side, n)
                for _ in range(PLACEMENTS)]
        shared = statistics.mean(vals)
        rows.append({"n": n, "rect_per_cell": shared / (side * side)})
    return rows


def figure_rect_depth_scaling(depth):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    dn = [r["n"] for r in depth]
    amort = [r["rect_per_cell"] for r in depth]

    ax.plot(dn, amort, "s-", color="#2166ac", ms=5, lw=1.4,
            label=rf"shared rectangular query ({DEPTH_SIDE}×{DEPTH_SIDE})")
    ax.axhline(1.0, ls="--", color="black", lw=1.3,
               label=r"one resolve per output cell")
    ax.set_ylim([0.95, 1.5])
    ax.set_xlabel(r"grid depth $n$  ($N = 2^n + 1$)")
    ax.set_ylabel("resolves per output cell")
    ax.set_title("Amortized rectangular query cost versus grid depth")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "rect_depth_scaling.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def main():
    rng = random.Random(SEED)
    depth = measure_depth(DEPTH_SIDE, DEPTH_NS, rng)

    print(
        f"depth scaling ({DEPTH_SIDE}x{DEPTH_SIDE}, "
        f"n={depth[0]['n']}..{depth[-1]['n']}): "
        f"amortized per cell {depth[0]['rect_per_cell']:.3f} -> "
        f"{depth[-1]['rect_per_cell']:.3f}"
    )

    figure_rect_depth_scaling(depth)


if __name__ == "__main__":
    main()
