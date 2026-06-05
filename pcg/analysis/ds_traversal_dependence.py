"""
demo_order_dependence.py

Thesis figure: traversal-order dependence in diamond-square
"""

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pcg.core.grid_config import GRID_SIZE
from pcg.terrain.ds.helper import ROUGHNESS
from pcg.core.hashing import hash_cell, bits_to_signed
from pcg.core.seeds import TESTING_SEED

OUT = Path(__file__).resolve().parents[2] / "outputs" / "analysis"


def _reorder(cells, order):
    if order == "normal":
        return cells
    if order == "reverse":
        return cells[::-1]
    raise ValueError(order)


def _ds_fill(seed, source="hash", order="normal", swap_coords=False):
    """`source` selects sequential PRNG vs coordinate hash
       `order`  selects normal vs reversed per-level traversal"""
    size = GRID_SIZE
    g = np.zeros((size, size))
    rng = random.Random(seed)

    def disp(level, i, j):
        if source == "prng":
            return rng.uniform(-1.0, 1.0)
        a, b = (j, i) if swap_coords else (i, j)
        return bits_to_signed(hash_cell(seed, level, a, b))

    # Corners fixed; only the per-level diamond/square traversal order varies.
    for (i, j) in [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]:
        g[i, j] = disp(0, i, j)

    step = size - 1
    amp = 1.0
    level = 1
    while step > 1:
        half = step // 2

        centers = [(i + half, j + half)
                   for i in range(0, size - 1, step)
                   for j in range(0, size - 1, step)]

        for (ci, cj) in _reorder(centers, order):
            i, j = ci - half, cj - half
            avg = (g[i, j] + g[i + step, j]
                   + g[i, j + step] + g[i + step, j + step]) / 4
            g[ci, cj] = avg + disp(level, ci, cj) * amp

        mids = []
        for i in range(0, size, half):
            start = half if (i // half) % 2 == 0 else 0
            for j in range(start, size, step):
                mids.append((i, j))

        for (i, j) in _reorder(mids, order):
            nb = []
            if i - half >= 0:
                nb.append(g[i - half, j])
            if i + half < size:
                nb.append(g[i + half, j])
            if j - half >= 0:
                nb.append(g[i, j - half])
            if j + half < size:
                nb.append(g[i, j + half])
            g[i, j] = sum(nb) / len(nb) + disp(level, i, j) * amp

        step = half
        amp *= 2 ** (-ROUGHNESS)
        level += 1
    return g


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _terrain_panel(ax, grid, title, vmin, vmax):
    ax.imshow(grid, cmap="terrain", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])


def _diff_panel(ax, diff, title, vmax):
    im = ax.imshow(diff, cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def figure_order_difference(seed):
    """Symmetric single-row figure: two groups of three (normal | reverse | diff)."""
    basic_n = _ds_fill(seed, source="prng", order="normal")
    basic_r = _ds_fill(seed, source="prng", order="reverse")
    hash_n = _ds_fill(seed, source="hash", order="normal")
    hash_r = _ds_fill(seed, source="hash", order="reverse")

    basic_diff = np.abs(basic_n - basic_r)
    hash_diff = np.abs(hash_n - hash_r)
    cb = _corr(basic_n, basic_r)
    basic_max = float(basic_diff.max())

    print(
        f"classical corr={cb:+.3f} max|diff|={basic_max:.4g} ; "
        f"hashed identical={np.array_equal(hash_n, hash_r)} "
        f"max|diff|={hash_diff.max():.4g}"
    )

    terr = [basic_n, basic_r, hash_n, hash_r]
    vmin = min(float(g.min()) for g in terr)
    vmax = max(float(g.max()) for g in terr)

    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(9.6, 2.1))
    gs = fig.add_gridspec(
        2, 7,
        height_ratios=[1.0, 0.045],
        width_ratios=[1, 1, 1, 0.18, 1, 1, 1],
        wspace=0.09, hspace=0.10,
        left=0.015, right=0.985, top=0.80, bottom=0.06,
    )

    ax_cn = fig.add_subplot(gs[0, 0])
    ax_cr = fig.add_subplot(gs[0, 1])
    ax_cd = fig.add_subplot(gs[0, 2])
    ax_hn = fig.add_subplot(gs[0, 4])
    ax_hr = fig.add_subplot(gs[0, 5])
    ax_hd = fig.add_subplot(gs[0, 6])

    cax_c = fig.add_subplot(gs[1, 2])
    cax_h = fig.add_subplot(gs[1, 6])

    _terrain_panel(ax_cn, basic_n, "normal", vmin, vmax)
    _terrain_panel(ax_cr, basic_r, "reverse", vmin, vmax)
    im_c = _diff_panel(ax_cd, basic_diff, "$|\\Delta|$ Height", basic_max)

    _terrain_panel(ax_hn, hash_n, "normal", vmin, vmax)
    _terrain_panel(ax_hr, hash_r, "reverse", vmin, vmax)
    im_h = _diff_panel(ax_hd, np.zeros_like(hash_diff), r"$|\Delta|$ Height", basic_max)
    ax_hd.text(0.5, 0.5, r"$\max|\Delta| = 0$",
               ha="center", va="center", color="white", fontsize=8.5,
               transform=ax_hd.transAxes)

    cbar_c = fig.colorbar(im_c, cax=cax_c, orientation="horizontal")
    cbar_c.ax.tick_params(labelsize=6.5, pad=1)
    # cbar_c.set_label("Absolute height difference", fontsize=6.5, labelpad=1)

    cbar_h = fig.colorbar(im_h, cax=cax_h, orientation="horizontal")
    cbar_h.ax.tick_params(labelsize=6.5, pad=1)
    # cbar_h.set_label("Absolute height difference", fontsize=6.5, labelpad=1)

    def center_x(a, b):
        pa, pb = a.get_position(), b.get_position()
        return (pa.x0 + pb.x1) / 2

    fig.text(center_x(ax_cn, ax_cd), 0.910, "Classical  —  order-dependent",
             ha="center", va="center", fontsize=10, fontweight="bold")
    fig.text(center_x(ax_hn, ax_hd), 0.910, "Hashed  —  order-independent",
             ha="center", va="center", fontsize=10, fontweight="bold")

    xdiv = (ax_cd.get_position().x1 + ax_hn.get_position().x0) / 2
    fig.add_artist(plt.Line2D([xdiv, xdiv], [0.05, 0.86], color="0.8", lw=1.0))

    cy_c = cax_c.get_position().y0 + cax_c.get_position().height / 2
    cy_h = cax_h.get_position().y0 + cax_h.get_position().height / 2

    fig.text(center_x(ax_cn, ax_cd), cy_c, f"seed = 0x{seed:x}",
             ha="right", va="center", fontsize=8, style="italic", color="0.55")

    fig.text(center_x(ax_hn, ax_hd), cy_h, f"seed = 0x{seed:x}",
             ha="right", va="center", fontsize=8, style="italic", color="0.55")

    fig.suptitle("Traversal order dependence in diamond-square",
                 fontsize=13, y=1.06)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "ds_traversal_order_difference.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
    return p


def main():
    figure_order_difference(TESTING_SEED)


if __name__ == "__main__":
    main()
