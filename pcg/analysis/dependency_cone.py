import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from pcg.core.grid_config import GRID_N, GRID_SIZE
from pcg.terrain.ds.helper import level_of, parents

OUT = os.path.join("outputs", "analysis")


def dependency_cone(coord):
    """
    Return all cells touched by the recursive walk-up for one target coordinate
    """
    cone = {}
    stack = [coord]

    while stack:
        c = stack.pop()

        if c in cone:
            continue

        L = level_of(*c)
        cone[c] = L

        if L > 0:
            stack.extend(parents(c[0], c[1], L))

    return cone


def figure_query_cones(targets):
    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(4.8 * len(targets), 4.8),
        constrained_layout=True,
    )

    if len(targets) == 1:
        axes = [axes]

    level_colors = [
        "#440154",  # 0 dark purple
        "#482878",  # 1 purple
        "#3E4989",  # 2 blue-purple
        "#31688E",  # 3 blue
        "#26828E",  # 4 teal
        "#1F9E89",  # 5 green-teal
        "#35B779",  # 6 green
        "#6DCD59",  # 7 light green
        "#FDE725",  # 8 yellow
    ]

    cmap = ListedColormap(level_colors)
    norm = BoundaryNorm(
        boundaries=np.arange(-0.5, GRID_N + 1.5, 1),
        ncolors=len(level_colors)
    )

    sc = None

    for ax, target in zip(axes, targets):
        cone = dependency_cone(target)

        ii = [i for (i, j) in cone]
        jj = [j for (i, j) in cone]
        levels = [cone[(i, j)] for (i, j) in zip(ii, jj)]

        sc = ax.scatter(
            jj,
            ii,
            c=levels,
            cmap=cmap,
            norm=norm,
            s=24,
            edgecolors="black",
            linewidths=0.25,
        )

        ti, tj = target
        ax.scatter(
            [tj],
            [ti],
            s=140,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
        )

        ax.set_xlim(-2, GRID_SIZE + 1)
        ax.set_ylim(-2, GRID_SIZE + 1)
        ax.set_aspect("equal")

        ax.set_title(
            f"target {target}, level {level_of(*target)}\n"
            f"{len(cone)} cells touched",
            fontsize=10,
        )

        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(
        sc,
        ax=axes,
        fraction=0.025,
        pad=0.02,
        ticks=np.arange(0, GRID_N + 1)
    )
    cbar.set_label("cell level")

    fig.suptitle(
        "Example dependency cones for single-point diamond-square queries",
        fontsize=13,
    )

    os.makedirs(OUT, exist_ok=True)

    p = os.path.join(OUT, f"dependency_cones_n{GRID_N}.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("wrote", p)


def main():
    targets = [
        (64, 192),   # small cone
        (129, 129),  # central/deep cone
        (130, 7),    # larger cone near an edge
    ]

    for target in targets:
        cone = dependency_cone(target)
        print(
            f"target {target}: "
            f"level {level_of(*target)}, "
            f"cone size {len(cone)}"
        )

    figure_query_cones(targets)


if __name__ == "__main__":
    main()
