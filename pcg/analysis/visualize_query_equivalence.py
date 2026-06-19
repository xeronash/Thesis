"""
visualize_query_equivalence.py

visualize that rectangle queries and independent single-cell queries reproduce
the same values as the corresponding full terrain functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.lines import Line2D

from pcg.core.grid_config import GRID_SIZE
from pcg.core.seeds import TESTING_SEED

from pcg.terrain.ds.gen_full_hashed import ds_gen_full_hashed
from pcg.terrain.ds.gen_single import ds_gen_single
from pcg.terrain.ds.gen_rect import ds_gen_rect

from pcg.terrain.pn.gen_full_octaves import pn_gen_full_octaves
from pcg.terrain.pn.gen_single import pn_gen_single
from pcg.terrain.pn.gen_rect import pn_gen_rect


DS_TOL = 1e-9
PN_TOL = 1e-12
DEFAULT_ORIGIN = (90, 70)
DEFAULT_SHAPE = (40, 50)
OUT = Path(__file__).resolve().parents[2] / "outputs" / "analysis"
DIFF_VMAX = 1e-15
DIFF_TICKS = np.linspace(0.0, DIFF_VMAX, 6)


def _rect_bounds(origin: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    i0, j0 = origin
    h, w = shape
    i1, j1 = i0 + h, j0 + w
    if i0 < 0 or j0 < 0 or h <= 0 or w <= 0 or i1 > GRID_SIZE or j1 > GRID_SIZE:
        raise ValueError(
            f"Invalid rectangle origin={origin}, shape={shape} for GRID_SIZE={GRID_SIZE}"
        )
    return i0, j0, i1, j1


def _slice(full: np.ndarray, origin: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    i0, j0, i1, j1 = _rect_bounds(origin, shape)
    return full[i0:i1, j0:j1]


def _loop_single(
    single_fn: Callable[[tuple[int, int], int], float],
    origin: tuple[int, int],
    shape: tuple[int, int],
    seed: int,
) -> np.ndarray:
    i0, j0, _, _ = _rect_bounds(origin, shape)
    h, w = shape
    return np.array(
        [[single_fn((i0 + di, j0 + dj), seed) for dj in range(w)] for di in range(h)],
        dtype=float,
    )


def _loop_single_full(
    single_fn: Callable[[tuple[int, int], int], float],
    seed: int,
) -> np.ndarray:
    return np.array(
        [[single_fn((i, j), seed) for j in range(GRID_SIZE)] for i in range(GRID_SIZE)],
        dtype=float,
    )


def _make_comparison(
    name: str,
    full_fn: Callable[[int], np.ndarray],
    rect_fn: Callable[[tuple[int, int], tuple[int, int], int], np.ndarray],
    single_fn: Callable[[tuple[int, int], int], float],
    seed: int,
    origin: tuple[int, int],
    shape: tuple[int, int],
    tol: float,
) -> dict:
    full = full_fn(seed)
    ref = _slice(full, origin, shape)
    rect = rect_fn(origin, shape, seed)
    indep = _loop_single(single_fn, origin, shape, seed)
    single_full = _loop_single_full(single_fn, seed)

    return {
        "name": name,
        "reference_full": full,
        "reference_slice": ref,
        "single_full": single_full,
        "rect": rect,
        "independent": indep,
        "diff_rect": np.abs(ref - rect),
        "diff_independent": np.abs(ref - indep),
        "tol": tol,
    }


def _terrain_limits(*arrays: np.ndarray) -> tuple[float, float]:
    return min(float(a.min()) for a in arrays), max(float(a.max()) for a in arrays)


def _terrain_panel(
    ax: plt.Axes,
    grid: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    rectangle: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> None:
    ax.imshow(grid, cmap="terrain", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=7.9, pad=2.0)
    ax.set_xticks([])
    ax.set_yticks([])

    if rectangle is not None:
        origin, shape = rectangle
        i0, j0, _, _ = _rect_bounds(origin, shape)
        h, w = shape
        ax.add_patch(
            Rectangle(
                (j0 - 0.5, i0 - 0.5),
                w,
                h,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
            )
        )


def _diff_panel(
    ax: plt.Axes,
    diff: np.ndarray,
    title: str,
    vmax: float,
) -> plt.AxesImage:
    im = ax.imshow(diff, cmap="magma", origin="lower", vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=7.7, pad=1.8)
    ax.set_xticks([])
    ax.set_yticks([])

    max_diff = float(np.max(diff))
    ax.text(
        0.5,
        0.5,
        f"Max |Δ height|\n= {max_diff:.2e}",
        ha="center",
        va="center",
        color="white",
        fontsize=7.0,
        linespacing=1.15,
        transform=ax.transAxes,
    )
    return im


def _zoom_lines(
    fig: plt.Figure,
    ax_src: plt.Axes,
    ax_dst: plt.Axes,
    origin: tuple[int, int],
    shape: tuple[int, int],
) -> None:
    """Draw guide lines from the rectangle in ax_src to the rect panel in ax_dst."""
    i0, j0, _, _ = _rect_bounds(origin, shape)
    h, w = shape

    src_corners = [
        (j0 + w - 0.5, i0 + h - 0.5),
        (j0 + w - 0.5, i0 - 0.5),
    ]

    dst_corners = [
        (-0.5, h - 0.5),
        (-0.5, -0.5),
    ]

    for (xa, ya), (xb, yb) in zip(src_corners, dst_corners):
        con = ConnectionPatch(
            xyA=(xa, ya),
            coordsA=ax_src.transData,
            xyB=(xb, yb),
            coordsB=ax_dst.transData,
            color="0.40",
            lw=0.8,
            linestyle=(0, (3, 2)),
            alpha=0.9,
            zorder=5,
        )
        con.set_clip_on(False)
        fig.add_artist(con)


def _side_label(ax: plt.Axes, text: str) -> None:
    ax.axis("off")
    ax.text(
        0.98,
        0.5,
        text,
        ha="center",
        va="center",
        rotation=90,
        fontsize=9.2,
        fontweight="bold",
        transform=ax.transAxes,
    )


def _style_colorbar(cbar) -> None:
    cbar.set_ticks(DIFF_TICKS)
    cbar.set_ticklabels([f"{tick / DIFF_VMAX:.1f}" for tick in DIFF_TICKS])
    cbar.ax.tick_params(labelsize=6.0, pad=1)
    cbar.set_label("absolute height difference", fontsize=6.8, labelpad=4)
    cbar.ax.set_title("1e−15", fontsize=5.8, pad=3)


def figure_query_equivalence(
    seed: int = TESTING_SEED,
    origin: tuple[int, int] = DEFAULT_ORIGIN,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    out_dir: Path = OUT,
) -> Path:
    ds = _make_comparison(
        name="Diamond-square",
        full_fn=ds_gen_full_hashed,
        rect_fn=ds_gen_rect,
        single_fn=ds_gen_single,
        seed=seed,
        origin=origin,
        shape=shape,
        tol=DS_TOL,
    )

    pn = _make_comparison(
        name="Perlin noise",
        full_fn=pn_gen_full_octaves,
        rect_fn=pn_gen_rect,
        single_fn=pn_gen_single,
        seed=seed,
        origin=origin,
        shape=shape,
        tol=PN_TOL,
    )

    plt.rcParams.update({"font.size": 9})

    fig = plt.figure(figsize=(9.2, 4.0))
    outer = fig.add_gridspec(
        2,
        4,
        width_ratios=[0.065, 2.05, 0.54, 0.045],
        wspace=0.015,
        hspace=0.28,
        left=0.03,
        right=0.985,
        top=0.84,
        bottom=0.08,
    )

    diff_vmax = DIFF_VMAX

    divider_left_axes = []
    divider_right_axes = []
    terr_div_left_axes = []
    terr_div_right_axes = []
    row_axes = []

    for row, comp in enumerate([ds, pn]):
        ax_label = fig.add_subplot(outer[row, 0])
        cax = fig.add_subplot(outer[row, 3])

        terr_gs = outer[row, 1].subgridspec(
            1, 4, width_ratios=[1.0, 0.16, 1.0, 1.0], wspace=0.006
        )
        ax_full = fig.add_subplot(terr_gs[0, 0])
        ax_slice = fig.add_subplot(terr_gs[0, 2])
        ax_single_full = fig.add_subplot(terr_gs[0, 3])

        diff_gs = outer[row, 2].subgridspec(2, 1, hspace=0.26)
        ax_rect = fig.add_subplot(diff_gs[0, 0])
        ax_ind = fig.add_subplot(diff_gs[1, 0])

        terr_vmin, terr_vmax = _terrain_limits(
            comp["reference_full"], comp["reference_slice"], comp["single_full"]
        )

        _side_label(ax_label, comp["name"])

        _terrain_panel(
            ax_full,
            comp["reference_full"],
            "Full terrain",
            terr_vmin,
            terr_vmax,
            rectangle=(origin, shape),
        )

        _terrain_panel(
            ax_slice,
            comp["reference_slice"],
            "Rect",
            terr_vmin,
            terr_vmax,
        )

        _terrain_panel(
            ax_single_full,
            comp["single_full"],
            "Single (full)",
            terr_vmin,
            terr_vmax,
        )

        _zoom_lines(fig, ax_full, ax_slice, origin, shape)

        im = _diff_panel(
            ax_rect,
            comp["diff_rect"],
            "Rect",
            diff_vmax,
        )

        _diff_panel(
            ax_ind,
            comp["diff_independent"],
            "Single",
            diff_vmax,
        )

        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        _style_colorbar(cbar)

        divider_left_axes.append(ax_single_full)
        divider_right_axes.append(ax_rect)
        terr_div_left_axes.append(ax_full)
        terr_div_right_axes.append(ax_slice)
        row_axes.extend([ax_full, ax_slice, ax_single_full, ax_rect, ax_ind])
    x_div = (
        divider_left_axes[0].get_position().x1
        + divider_right_axes[0].get_position().x0
    ) / 2.0
    x_div_terr = (
        terr_div_left_axes[0].get_position().x1
        + terr_div_right_axes[0].get_position().x0
    ) / 2.0

    y_bottom = min(ax.get_position().y0 for ax in row_axes)
    y_top = max(ax.get_position().y1 for ax in row_axes)

    for x in (x_div_terr, x_div):
        fig.add_artist(
            Line2D(
                [x, x],
                [y_bottom, y_top],
                transform=fig.transFigure,
                color="0.80",
                lw=0.9,
            )
        )

    fig.suptitle(
        "Query equivalence against full terrain generators",
        fontsize=12.2,
        y=0.965,
    )

    fig.text(
        0.5,
        0.89,
        f"seed = 0x{seed:x}    rectangle origin = {origin}, shape = {shape}",
        ha="center",
        va="center",
        fontsize=8.0,
        style="italic",
        color="0.45",
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "query_equivalence_difference.png"

    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    for comp in [ds, pn]:
        rect_max = float(comp["diff_rect"].max())
        ind_max = float(comp["diff_independent"].max())
        single_full_max = float(np.abs(comp["reference_full"] - comp["single_full"]).max())
        print(
            f"{comp['name']}: "
            f"rect max|diff|={rect_max:.3e}, "
            f"independent singles max|diff|={ind_max:.3e}, "
            f"single-full max|diff|={single_full_max:.3e}, "
            f"tol={comp['tol']:.1e}"
        )

    print("wrote", png_path)
    return png_path


def main() -> None:
    figure_query_equivalence()


if __name__ == "__main__":
    main()
