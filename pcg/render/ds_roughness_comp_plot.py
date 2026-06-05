"""
ds_roughness_comp_plot.py

Multi-panel comparison plots for diamond-square terrain generated with the same
seed but different roughness values.

The panels share one colour range in 2D and one z-range in 3D, so the roughness
change is visually comparable across the row.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pcg.core.seeds import TESTING_SEED
from pcg.render.helper import seed_hex, params_str


def _roughness_token(r):
    """Make a roughness value safe for filenames, e.g. 0.3 -> 0p3."""
    return str(r).replace(".", "p").replace("-", "m")


def plot_panels_2d(grids, labels, path, suptitle=None, cmap="terrain"):
    """Row of 2D heightmaps sharing one colour scale and one colorbar."""
    vmin = min(float(g.min()) for g in grids)
    vmax = max(float(g.max()) for g in grids)

    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    im = None
    for ax, g, lab in zip(axes, grids, labels):
        im = ax.imshow(g, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(lab)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.8, label="height")

    if suptitle:
        fig.suptitle(suptitle)

    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_panels_3d(grids, labels, path, suptitle=None, cmap="terrain",
                   vertical_exaggeration=0.5, elevation=35, azimuth=-60,
                   stride=None):
    """Row of 3D surfaces sharing one z-range, one view, and one colorbar."""
    size = grids[0].shape[0]
    if stride is None:
        stride = max(1, size // 128)

    Zs = [g[::stride, ::stride] * vertical_exaggeration for g in grids]
    zmin = min(float(z.min()) for z in Zs)
    zmax = max(float(z.max()) for z in Zs)
    norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)

    coord = np.arange(0, size, stride)
    X, Y = np.meshgrid(coord, coord)

    n = len(grids)
    fig = plt.figure(figsize=(6 * n, 5))
    axes = []

    for k, (Z, lab) in enumerate(zip(Zs, labels), start=1):
        ax = fig.add_subplot(1, n, k, projection="3d")
        ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm, linewidth=0,
                        antialiased=True, rstride=1, cstride=1)
        ax.set_zlim(zmin, zmax)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_title(lab)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zticklabels([])
        axes.append(ax)

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=axes, shrink=0.6, label="height")

    if suptitle:
        fig.suptitle(suptitle)

    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_roughness_comparison(grids, roughness_list, seed, shared_params=None,
                                mode="2d", cmap="terrain",
                                base_dir=os.path.join("outputs", "analysis"),
                                family=None, tag=None):
    """
    Save one side-by-side figure of the same diamond-square world at several
    roughness values.

        grids          : list of heightmaps, one per roughness value.
        roughness_list : roughness values, lined up by index with `grids`.
        seed           : world seed, rendered as hex in the filename/title.
        shared_params  : parameters held constant across panels.
        mode           : "2d", "3d", or "both".
        base_dir       : output folder. Defaults to outputs/analysis.
        family         : optional subfolder inside base_dir.

    Returns the list of written paths.
    """
    if len(grids) != len(roughness_list):
        raise ValueError("grids and roughness_list must be the same length")

    folder = os.path.join(base_dir, family) if family else base_dir
    os.makedirs(folder, exist_ok=True)

    seed_s = seed_hex(seed)
    if tag is None and int(seed) == TESTING_SEED:
        tag = "TEST"
    prefix = f"{tag}_" if tag else ""

    labels = [f"roughness = {r}" for r in roughness_list]
    rough_s = "-".join(_roughness_token(r) for r in roughness_list)
    pstr = params_str(shared_params)
    # suptitle = (f"ds_gen_full_basic roughness comparison   seed=0x{seed_s}"
    #             + (f"   |   {pstr}" if pstr else ""))

    written = []

    if mode in ("2d", "both"):
        p = os.path.join(folder, "dn_roughness_comparison_2d.png")
        plot_panels_2d(grids, labels, p, cmap=cmap)
        written.append(p)

    if mode in ("3d", "both"):
        p = os.path.join(folder, "dn_roughness_comparison_3d.png")
        plot_panels_3d(grids, labels, p, cmap=cmap)
        written.append(p)

    print("[render] roughness comparison: wrote " + ", ".join(written))
    return written
