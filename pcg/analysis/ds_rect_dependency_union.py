"""
rect_dependency_union.py

Visualize the dependency union and dependency reuse for one rectangular
diamond-square query.
"""

import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle

from pcg.core.grid_config import GRID_N, GRID_SIZE
from pcg.terrain.ds.helper import level_of, parents

OUT = os.path.join("outputs", "analysis")


def rect_cells(origin, size):
    i0, j0 = origin
    h, w = size
    return [(i0 + di, j0 + dj) for di in range(h) for dj in range(w)]


def cone_cells(cell):
    """
    Full dependency cone of one output cell.
    Returns the set of all cells needed to resolve that cell.
    """
    seen = set()
    stack = [cell]

    while stack:
        c = stack.pop()
        if c in seen:
            continue

        seen.add(c)
        L = level_of(*c)

        if L > 0:
            stack.extend(parents(c[0], c[1], L))

    return seen


def dependency_union(cells):
    """
    Union of dependency cones for all output cells in 'cells'.
    Returns {cell: level}.
    """
    union = {}
    stack = list(cells)

    while stack:
        c = stack.pop()

        if c in union:
            continue

        L = level_of(*c)
        union[c] = L

        if L > 0:
            stack.extend(parents(c[0], c[1], L))

    return union


def dependency_multiplicity(cells):
    """
    For each dependency cell c, count how many output cells depend on c.

    Returns:
        Counter mapping cell -> multiplicity
    """
    mult = Counter()
    for cell in cells:
        for dep in cone_cells(cell):
            mult[dep] += 1
    return mult


def figure_rect_union_and_reuse(origin, size, label):
    cells = rect_cells(origin, size)
    box = set(cells)
    union = dependency_union(cells)
    mult = dependency_multiplicity(cells)

    area = size[0] * size[1]
    distinct = len(union)
    exterior = distinct - area
    naive_total = sum(mult.values())
    saved = naive_total - distinct
    reused_cells = sum(1 for v in mult.values() if v > 1)
    hit_rate = saved / naive_total if naive_total > 0 else 0.0

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15.5, 4.8),
        constrained_layout=True,
    )

    # (A) Dependency union, colored by level
    ax = axes[0]

    cmap = plt.get_cmap("viridis", GRID_N + 1)
    norm = BoundaryNorm(
        boundaries=np.arange(-0.5, GRID_N + 1.5, 1),
        ncolors=GRID_N + 1,
    )

    ii = [i for (i, j) in union]
    jj = [j for (i, j) in union]
    cell_levels = [union[(i, j)] for (i, j) in zip(ii, jj)]

    sc = ax.scatter(
        jj,
        ii,
        c=cell_levels,
        cmap=cmap,
        norm=norm,
        s=10,
        edgecolors="black",
        linewidths=0.15,
    )

    i0, j0 = origin
    h, w = size
    ax.add_patch(
        Rectangle(
            (j0 - 0.5, i0 - 0.5),
            w,
            h,
            fill=False,
            edgecolor="red",
            linewidth=2.0,
        )
    )

    ax.set_xlim(-2, GRID_SIZE + 1)
    ax.set_ylim(-2, GRID_SIZE + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"(A) Dependency union coloured by level\n"
        f"{h}×{w} query: {area} inside + {exterior} outside = {distinct} distinct cells",
        fontsize=10,
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        fraction=0.046,
        pad=0.03,
        ticks=np.arange(0, GRID_N + 1),
    )
    cbar.set_label("cell level")

    # Shared data for panels (B) and (C)
    levels = list(range(GRID_N + 1))
    naive_accesses = []
    memo_resolves = []

    for L in levels:
        vals = [mult[c] for c in union if union[c] == L]

        naive_accesses.append(sum(vals))
        memo_resolves.append(len(vals))

    x = np.arange(len(levels))

    # (B) Naive dependency accesses by level
    ax = axes[1]

    ax.bar(
        x,
        naive_accesses,
        width=0.8,
        label="naive accesses",
    )

    max_naive = max(naive_accesses) if naive_accesses else 0
    y_offset = max_naive * 0.015 if max_naive > 0 else 1.0

    for xi, total in zip(x, naive_accesses):
        if total == 0:
            continue

        ax.text(
            xi,
            total + y_offset,
            str(int(total)),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90 if total > 999 else 0,
        )

    ax.set_ylim(0, max_naive * 1.18 if max_naive > 0 else 1)
    ax.set_xlabel("cell level")
    ax.set_ylabel("dependency accesses")
    ax.set_title(
        f"(B) Independent single-cell queries\n"
        f"{area} output cells: {naive_total} total dependency accesses",
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.grid(axis="y", alpha=0.3)

    # (C) Shared-memo resolves by level
    ax = axes[2]

    ax.bar(
        x,
        memo_resolves,
        width=0.8,
        label="shared-memo resolves",
    )

    max_memo = max(memo_resolves) if memo_resolves else 0
    y_offset = max_memo * 0.015 if max_memo > 0 else 1.0

    for xi, total in zip(x, memo_resolves):
        if total == 0:
            continue

        ax.text(
            xi,
            total + y_offset,
            str(int(total)),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, max_memo * 1.15 if max_memo > 0 else 1)
    ax.set_xlabel("cell level")
    ax.set_ylabel("distinct dependency cells resolved")
    ax.set_title(
        f"(C) Rectangular query with shared memoization\n"
        f"{distinct} distinct dependency cells resolved once",
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Dependency union and reuse for a rectangular query",
        fontsize=13,
    )

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"rect_dependency_union_n{GRID_N}.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def main():
    # Central square only
    s = max(8, GRID_SIZE // 10)
    mid = GRID_SIZE // 2
    origin = (mid - s // 2, mid - s // 2)
    size = (s, s)
    label = "central square"

    cells = rect_cells(origin, size)
    union = dependency_union(cells)
    mult = dependency_multiplicity(cells)

    area = size[0] * size[1]
    distinct = len(union)
    exterior = distinct - area
    naive_total = sum(mult.values())
    saved = naive_total - distinct
    hit_rate = saved / naive_total if naive_total > 0 else 0.0

    print(
        f"{label} {size} at {origin}: "
        f"{area} inside + {exterior} outside = {distinct} distinct cells"
    )
    print(
        f"naive total resolves = {naive_total}, "
        f"repeated resolves avoided by memoization = {saved}, "
        f"memo hit rate = {100 * hit_rate:.1f}%"
    )

    figure_rect_union_and_reuse(origin, size, label)


if __name__ == "__main__":
    main()
