"""
hashing.py

Contents:
    - MASK64                       : 64-bit unsigned wraparound mask.
    - _mix64(x)                    : splitmix64 finalizer
    - hash_cell(seed, level, i, j) : 64 well-mixed bits.
    - bits_to_signed(bits)         : 64 bits -> float in [-1, 1).
"""

# 64-bit unsigned wraparound mask
MASK64 = 0xFFFFFFFFFFFFFFFF

# splitmix64 finalizer constants
_SPLITMIX_C1 = 0xBF58476D1CE4E5B9
_SPLITMIX_C2 = 0x94D049BB133111EB


def _mix64(x: int) -> int:
    """splitmix64 finalizer. A one-shot mixer with strong avalanche"""
    x = (x ^ (x >> 30)) & MASK64
    x = (x * _SPLITMIX_C1) & MASK64
    x = (x ^ (x >> 27)) & MASK64
    x = (x * _SPLITMIX_C2) & MASK64
    x = (x ^ (x >> 31)) & MASK64
    return x


def hash_cell(seed: int, level: int, i: int, j: int) -> int:
    """ Hash (seed, level, i, j) into 64 well-mixed bits"""
    state = _mix64(seed ^ level)
    state = _mix64(state ^ i)
    state = _mix64(state ^ j)
    return state


def bits_to_signed(bits: int) -> float:
    """Convert 64 random bits to a uniform float in [-1, 1)"""
    top53 = (bits & MASK64) >> 11
    return top53 / float(1 << 53) * 2.0 - 1.0


def bits_to_unit(bits: int) -> float:
    """
    Convert 64 random bits to a uniform float in [0, 1).
    """
    top53 = (bits & MASK64) >> 11
    return top53 / float(1 << 53)
