"""
heightmap.py


"""

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from pcg.core.seeds import TESTING_SEED
from pcg.render.helper import seed_hex, params_str

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"


def plot_heightmap(grid, path="heightmap.png", title=None, cmap="terrain"):
    """Save a top-down 2D view of a heightmap."""
    plt.figure(figsize=(7, 6))
    plt.imshow(grid, cmap=cmap, origin="lower")
    plt.colorbar(label="height")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_surface(grid, path="heightmap_3d.png", title=None, cmap="terrain",
                 vertical_exaggeration=0.5, elevation=35, azimuth=-60,
                 stride=None):
    """Save a 3D surface plot of a heightmap."""
    size = grid.shape[0]
    if stride is None:
        stride = max(1, size // 128)

    x = np.arange(0, size, stride)
    y = np.arange(0, size, stride)
    X, Y = np.meshgrid(x, y)
    Z = grid[::stride, ::stride] * vertical_exaggeration

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True,
                           rstride=1, cstride=1)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("height")

    if title:
        ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.6, label="height")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_2d_3d(grid, algo, name, seed, params=None, tag=None,
               make_2d=True, make_3d=True, cmap="terrain",
               base_dir=OUTPUTS_DIR):
    """
    Save 2D and/or 3D figures into Plots/<algo>/ with self-documenting names
    and titles. Returns the list of written paths.

        algo     : "ds" or "pn" (the subfolder).
        name     : e.g. "ds_gen_full_hashed" (goes in filename + title).
        seed     : world seed (rendered as hex in name + title).
        params   : dict of the parameter values used, stamped into the title.
        tag      : optional filename marker, e.g. "TEST". If left None, the
                    file is auto-marked "TEST_" when seed == TESTING_SEED, so
                    test-seed figures are obvious at a glance in the folder.
                    The title always shows the real seed regardless.
    """
    folder = os.path.join(base_dir, algo)
    os.makedirs(folder, exist_ok=True)

    seed_s = seed_hex(seed)

    if tag is None and int(seed) == TESTING_SEED:
        tag = "TEST"
    prefix = f"{tag}_" if tag else ""

    pstr = params_str(params)
    subtitle = f"seed=0x{seed_s}" + (f"   |   {pstr}" if pstr else "")

    written = []
    if make_2d:
        p = os.path.join(folder, f"{prefix}{seed_s}_{name}_2d.png")
        plot_heightmap(grid, path=p, title=f"{name}\n{subtitle}", cmap=cmap)
        written.append(p)
    if make_3d:
        p = os.path.join(folder, f"{prefix}{seed_s}_{name}_3d.png")
        plot_surface(grid, path=p, title=f"{name}\n{subtitle}", cmap=cmap)
        written.append(p)

    print(f"[render] {name}: wrote " + ", ".join(written))
    return written
