"""
analyze_cone_scaling.py

Measure how the dependency-cone size of ds_gen_single scales with grid depth.
"""

import csv
import os
import random
import statistics

import matplotlib.pyplot as plt

OUT = os.path.join("outputs", "analysis")

MAX_N = 14
EXHAUSTIVE_MAX = 8
SAMPLES = 100000
SEED = 1


def _v2(x, n):
    """2-adic valuation; v2(0) = n by convention."""
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
        cand = [
            (i - half_l, j - half_l),
            (i - half_l, j + half_l),
            (i + half_l, j - half_l),
            (i + half_l, j + half_l),
        ]
    else:
        cand = [
            (i - half_l, j),
            (i + half_l, j),
            (i, j - half_l),
            (i, j + half_l),
        ]

    return [
        (a, b)
        for (a, b) in cand
        if 0 <= a <= grid_last and 0 <= b <= grid_last
    ]


def cone_size(coord, n):
    """Number of distinct ancestors = number of cells _resolve touches."""
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


def measure(n, rng):
    """
    Return (mean_cone, max_cone, exhaustive).
    If exhaustive is False, max_cone is only the observed maximum among samples
    """
    N = (1 << n) + 1

    if n <= EXHAUSTIVE_MAX:
        sizes = [cone_size((i, j), n) for i in range(N) for j in range(N)]
        exhaustive = True
    else:
        sizes = [
            cone_size((rng.randrange(N), rng.randrange(N)), n)
            for _ in range(SAMPLES)
        ]
        exhaustive = False

    return statistics.mean(sizes), max(sizes), exhaustive


def figure_cone_scaling(max_n):
    rng = random.Random(SEED)

    ns = list(range(3, max_n + 1))
    means = []
    maxs = []
    rows = []

    for n in ns:
        mean_cone, max_cone, exhaustive = measure(n, rng)

        means.append(mean_cone)
        maxs.append(max_cone)

        N = (1 << n) + 1
        cells = N ** 2
        kind = "exhaustive" if exhaustive else f"sampled ({SAMPLES:,})"
        max_label = "max" if exhaustive else "observed max"

        rows.append(
            {
                "n": n,
                "N": N,
                "cells": cells,
                "max_label": max_label,
                "max_cone": max_cone,
                "mean_cone": mean_cone,
                "method": kind,
                "exhaustive": exhaustive,
                "samples": "" if exhaustive else SAMPLES,
            }
        )

        print(
            f"n={n:2d}  "
            f"N={N:6d}  "
            f"cells={cells:>12,d}  "
            f"{max_label}={max_cone:4d}  "
            f"mean={mean_cone:7.2f}  "
            f"({kind})"
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.plot(
        ns,
        maxs,
        "o-",
        color="#b2182b",
        label="observed max cone size",
    )

    ax.plot(
        ns,
        means,
        "s-",
        color="#2166ac",
        label="mean cone size",
    )

    ax.set_xlabel(r"grid depth $n$  (side length $N = 2^n + 1$)")
    ax.set_ylabel("dependency-cone size (cells touched)")
    ax.set_title("Dependency-cone size grows linearly with grid depth")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    ax.text(
        0.98,
        0.03,
        f"n≤{EXHAUSTIVE_MAX}: exhaustive; larger n: {SAMPLES:,} samples",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="dimgray",
        bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.8),
    )

    fig.tight_layout()

    os.makedirs(OUT, exist_ok=True)

    csv_path = os.path.join(OUT, f"dependency_cone_scaling_n{max_n}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "N",
                "cells",
                "max_label",
                "max_cone",
                "mean_cone",
                "method",
                "exhaustive",
                "samples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    p = os.path.join(OUT, f"dependency_cone_scaling_n{max_n}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("wrote", csv_path)
    print("wrote", p)


def main():
    figure_cone_scaling(max_n=MAX_N)


if __name__ == "__main__":
    main()
