"""
helper.py

shared helper functions between the rendering scripts
"""


def seed_hex(seed) -> str:
    """make filename compact"""
    return format(int(seed), "x")


def params_str(params) -> str:
    if not params:
        return ""
    return ", ".join(f"{k}={v}" for k, v in params.items())
