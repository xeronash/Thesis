"""
viz_compare.py

Multi-panel comparison plots: figures that place several heightmaps of the same
world side by side, for example one Perlin seed rendered at increasing octave
counts.

This is kept separate from visualization.py, which renders one heightmap per
image. The reason a separate routine is needed is the SHARED scale. All panels
use one common colour range (and one common z-range in 3D), so a colour means
the same height in every panel and only the octave count changes between them.
Rendering each panel with its own auto-scaled range would make a low-octave and
a high-octave terrain look equally tall, which hides the very thing the figure
is meant to show.

"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pcg.core.seeds import TESTING_SEED
from pcg.render.helper import seed_hex, params_str


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
    # if suptitle:
    #     fig.suptitle(suptitle)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_panels_3d(grids, labels, path, suptitle=None, cmap="terrain",
                   vertical_exaggeration=0.5, elevation=35, azimuth=-60,
                   stride=None):
    """Row of 3D surfaces sharing one z-range, one view, and one colorbar.

    Height is shown by the single colorbar on the right, so the per-panel
    z-axis ticks are removed to avoid repeating the same scale three times.
    """
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

    # if suptitle:
    #     fig.suptitle(suptitle)
    #     fig.suptitle("pn_gen_full_octaves")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_octave_comparison(grids, octaves_list, seed, shared_params=None,
                             mode="2d", cmap="terrain",
                             base_dir=os.path.join("outputs", "analysis"),
                             family=None, tag=None):
    """
    Save one side-by-side figure of the same world at several octave counts.

        grids         : list of heightmaps, one per octave count.
        octaves_list  : the octave counts, lined up by index with `grids`.
        seed          : world seed (rendered as hex in name + title).
        shared_params : parameters held constant across panels
                        (persistence, lacunarity, frequency), stamped in title.
        mode          : "2d", "3d", or "both".
        base_dir      : output folder. Defaults to outputs/analysis.
        family        : optional subfolder inside base_dir. Defaults to None,
                        so images are written directly into outputs/analysis.

    Returns the list of written paths.
    """
    if len(grids) != len(octaves_list):
        raise ValueError("grids and octaves_list must be the same length")

    folder = os.path.join(base_dir, family) if family else base_dir
    os.makedirs(folder, exist_ok=True)

    seed_s = seed_hex(seed)
    if tag is None and int(seed) == TESTING_SEED:
        tag = "TEST"
    prefix = f"{tag}_" if tag else ""

    labels = [f"{o} octaves" for o in octaves_list]
    oct_s = "-".join(str(o) for o in octaves_list)
    pstr = params_str(shared_params)
    suptitle = (f"pn_gen_full_octaves   seed=0x{seed_s}"
                + (f"   |   {pstr}" if pstr else ""))

    written = []
    if mode in ("2d", "both"):
        p = os.path.join(folder, f"{prefix}{seed_s}_octaves_{oct_s}_2d.png")
        plot_panels_2d(grids, labels, p, suptitle=suptitle, cmap=cmap)
        written.append(p)
    if mode in ("3d", "both"):
        p = os.path.join(folder, f"{prefix}{seed_s}_octaves_{oct_s}_3d.png")
        plot_panels_3d(grids, labels, p, suptitle=suptitle, cmap=cmap)
        written.append(p)

    print("[render] octave comparison: wrote " + ", ".join(written))
    return written
