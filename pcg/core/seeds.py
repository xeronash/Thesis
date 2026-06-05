"""
seeds.py

64-bit world seeds.

    - SEED_MAX        : the valid range upper bound (2^64 - 1)
    - random_seed()   : draw a fresh seed from the OS entropy pool
    - validate_seed() : check that a value is a valid 64-bit unsigned int
    - TESTING_SEED    : one fixed seed for reproducible debugging
"""

import os

# 64-bit unsigned int range
SEED_MAX = 2**64 - 1
TESTING_SEED = 0x8C1FA2A44789C64F


def random_seed() -> int:
    """Generate a fresh 64-bit unsigned integer"""
    return int.from_bytes(os.urandom(8), "big")


def validate_seed(seed) -> None:
    """check if seed is not a valid 64-bit unsigned integer"""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int; got {type(seed).__name__}")
    if not (0 <= seed <= SEED_MAX):
        raise ValueError(
            f"seed must be a 64-bit unsigned int (0..{SEED_MAX}); got {seed}"
        )
