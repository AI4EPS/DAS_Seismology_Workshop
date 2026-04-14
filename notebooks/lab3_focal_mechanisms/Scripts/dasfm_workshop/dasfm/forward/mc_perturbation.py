"""Monte Carlo location perturbation for source parameter uncertainty.

The perturbation strategy (Gaussian noise in random horizontal direction +
independent vertical noise) and the union-across-trials approach for
combining acceptable solutions are adapted from SKHASH/HASH:
    Hardebeck, J.L., & Shearer, P.M. (2002). A new method for determining
    first-motion focal mechanisms. BSSA, 92(6), 2264-2276.
    Skoumal, R.J., Hardebeck, J.L., & Shearer, P.M. (2024). SKHASH: A Python
    package for computing earthquake focal mechanisms. SRL, 95(4), 2519-2526.
"""

from __future__ import annotations

import numpy as np


def perturb_sources(source_x, source_y, source_z, vert_unc, horz_unc, rng):
    """Return perturbed source coordinates (sx, sy, sz).

    Parameters
    ----------
    source_x, source_y, source_z : array (n_ev,)
        Unperturbed source Cartesian coordinates.
    vert_unc : float
        Vertical uncertainty (km), applied as Gaussian noise to z.
    horz_unc : float
        Horizontal uncertainty (km), applied as Gaussian noise in random direction.
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    sx, sy, sz : array (n_ev,) float64
        Perturbed source coordinates (sz clipped to >= 0).
    """
    n_ev = len(source_x)
    rand_angle = rng.uniform(0, 2 * np.pi, size=n_ev)
    rand_dist = rng.normal(size=n_ev) * horz_unc
    sx = source_x + rand_dist * np.cos(rand_angle)
    sy = source_y + rand_dist * np.sin(rand_angle)
    sz = source_z + rng.normal(size=n_ev) * vert_unc
    sz = np.clip(sz, 0, None)
    return sx, sy, sz
