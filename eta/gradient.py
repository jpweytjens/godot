"""Gradient binning and gradient-aware route utilities."""

import math


def gradient_bin(gradient_frac: float) -> int:
    """Assign a gradient (fraction) to a 3%-wide bin, clamped to [-21, 21].

    Parameters
    ----------
    gradient_frac : float
        Gradient as a fraction (e.g. 0.05 for 5%).

    Returns
    -------
    int
        Bin center in percent (e.g. 3, 6, -9).
    """
    pct = gradient_frac * 100
    b = math.floor(pct / 3) * 3
    return max(-21, min(21, b))
