"""
helper.py

Everything the Perlin algorithms share

    Defaults (terrain):
        DEFAULT_OCTAVES      = 6
        DEFAULT_PERSISTENCE  = 0.5      # amplitude multiplier per octave
        DEFAULT_LACUNARITY   = 2.0      # frequency multiplier per octave
        DEFAULT_FREQUENCY    = 1/128    # grid coord -> noise coord scaling

References:
    Perlin (1985) - original noise function.
    Perlin (2002) - "Improved Noise" with the smoother fade curve
"""

import math

from pcg.core.hashing import MASK64, hash_cell


# Terrain defaults
DEFAULT_OCTAVES = 6
DEFAULT_PERSISTENCE = 0.5
DEFAULT_LACUNARITY = 2.0
DEFAULT_FREQUENCY = 1 / 128


# Gradient table, 8 unit vectors evenly spaced around the circle
_DIAG = math.sqrt(0.5)
_GRADIENTS = [
    (1.0,  0.0),
    (-1.0,  0.0),
    (0.0,  1.0),
    (0.0, -1.0),
    (_DIAG,  _DIAG),
    (_DIAG, -_DIAG),
    (-_DIAG,  _DIAG),
    (-_DIAG, -_DIAG)
]


def _gradient_at(world_seed, octave, lx, ly):
    """Gradient vector at integer lattice point (lx, ly) for this octave
       Octave hashed for unique gradient at each octave depth
       lx/ly may be negative, so they are masked to 64-bit"""
    bits = hash_cell(world_seed, octave, lx & MASK64, ly & MASK64)
    return _GRADIENTS[bits & 7]


def _fade(t):
    """Perlins improved (2002) fade: 6t^5 - 15t^4 + 10t^3"""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a, b, t):
    return a + t * (b - a)


def _perlin_single(coord, world_seed, octave):
    """One raw Perlin octave at continuous coords (x, y)."""
    x, y = coord
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = x - x0
    dy = y - y0

    g00 = _gradient_at(world_seed, octave, x0, y0)
    g10 = _gradient_at(world_seed, octave, x1, y0)
    g01 = _gradient_at(world_seed, octave, x0, y1)
    g11 = _gradient_at(world_seed, octave, x1, y1)

    n00 = g00[0] * dx + g00[1] * dy
    n10 = g10[0] * (dx - 1) + g10[1] * dy
    n01 = g01[0] * dx + g01[1] * (dy - 1)
    n11 = g11[0] * (dx - 1) + g11[1] * (dy - 1)

    u = _fade(dx)
    v = _fade(dy)
    return _lerp(_lerp(n00, n10, u), _lerp(n01, n11, u), v)


def pn_sample(
    coord,
    world_seed,
    *,
    octaves=DEFAULT_OCTAVES,
    persistence=DEFAULT_PERSISTENCE,
    lacunarity=DEFAULT_LACUNARITY,
    normalize=True,
):
    """
    fBm Perlin value at continuous NOISE coordinates `coord = (x, y)`.

    Sums `octaves` layers; each octave doubles frequency (lacunarity) and
    scales amplitude by `persistence`. With normalize=True the result is
    divided by the total amplitude so it stays comparable across octave counts.
    """
    x, y = coord
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    amplitude_sum = 0.0

    for octave in range(octaves):

        total += _perlin_single((x * frequency, y * frequency), world_seed,
                                octave) * amplitude

        amplitude_sum += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / amplitude_sum if normalize else total
