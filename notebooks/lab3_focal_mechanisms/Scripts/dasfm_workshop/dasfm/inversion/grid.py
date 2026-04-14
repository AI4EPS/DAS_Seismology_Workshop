"""
SKHASH-style focal mechanism grid generation and SDR → moment tensor conversion.

The grid generation algorithm is adapted from SKHASH/HASH:
    Hardebeck, J.L., & Shearer, P.M. (2002). A new method for determining
    first-motion focal mechanisms. BSSA, 92(6), 2264-2276.
    Skoumal, R.J., Hardebeck, J.L., & Shearer, P.M. (2024). SKHASH: A Python
    package for computing earthquake focal mechanisms. SRL, 95(4), 2519-2526.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
#  Grid generation
# ---------------------------------------------------------------------------
def make_sdr_grid_skhash_style(dang: float = 5.0):
    """
    Generate uniformly-sampled focal mechanism grid (SKHASH style).

    Returns:
        strike_deg, dip_deg, rake_deg: 1-D arrays (degrees).
    """
    p_dict = {"dang": dang}
    dir_cos_dict = dir_cos_setup(p_dict)
    faultnorms = dir_cos_dict["b3"]  # (3, ncoor)
    slips = dir_cos_dict["b1"]       # (3, ncoor)
    return sdr_from_vector(faultnorms, slips)


def dir_cos_setup(p_dict: dict) -> dict:
    """Build direction-cosine arrays for all grid points."""
    start_angle = max(p_dict["dang"], 1e-3)
    the = np.arange(start_angle, 90.001, p_dict["dang"])
    num_izeta = int(np.floor(179.9 / p_dict["dang"]))

    rthe = np.deg2rad(the)
    costhe, sinthe = np.cos(rthe), np.sin(rthe)

    fnumang = 360.0 / p_dict["dang"]
    temp_n_phi = fnumang * sinthe
    temp_n_phi[temp_n_phi < 1] = 1
    dphi = 360.0 / np.round(temp_n_phi)
    num_iphi = np.round(360.0 / dphi).astype(int)

    total_points = np.sum(num_iphi) * (num_izeta + 1)
    dir_cos_dict = {
        "ncoor": total_points,
        "b1": np.zeros((3, total_points)),
        "b2": np.zeros((3, total_points)),
        "b3": np.zeros((3, total_points)),
    }

    irot = 0
    for ithe in range(len(the)):
        bb1 = np.zeros(3)
        bb3 = np.zeros(3)
        for iphi in range(num_iphi[ithe]):
            phi = iphi * dphi[ithe]
            rphi = np.deg2rad(phi)
            cosphi, sinphi = np.cos(rphi), np.sin(rphi)

            bb3[2] = costhe[ithe]
            bb3[0] = sinthe[ithe] * cosphi
            bb3[1] = sinthe[ithe] * sinphi

            bb1[2] = -sinthe[ithe]
            bb1[0] = costhe[ithe] * cosphi
            bb1[1] = costhe[ithe] * sinphi

            bb2 = np.cross(bb1, bb3) * -1

            for izeta in range(num_izeta + 1):
                if irot >= dir_cos_dict["ncoor"]:
                    break
                zeta = izeta * p_dict["dang"]
                rzeta = np.deg2rad(zeta)
                coszeta, sinzeta = np.cos(rzeta), np.sin(rzeta)

                dir_cos_dict["b3"][:, irot] = bb3
                dir_cos_dict["b1"][0, irot] = bb1[0] * coszeta + bb2[0] * sinzeta
                dir_cos_dict["b1"][1, irot] = bb1[1] * coszeta + bb2[1] * sinzeta
                dir_cos_dict["b1"][2, irot] = bb1[2] * coszeta + bb2[2] * sinzeta
                dir_cos_dict["b2"][0, irot] = bb2[0] * coszeta - bb1[0] * sinzeta
                dir_cos_dict["b2"][1, irot] = bb2[1] * coszeta - bb1[1] * sinzeta
                dir_cos_dict["b2"][2, irot] = bb2[2] * coszeta - bb1[2] * sinzeta
                irot += 1

    if irot < dir_cos_dict["ncoor"]:
        dir_cos_dict["b1"] = dir_cos_dict["b1"][:, :irot]
        dir_cos_dict["b2"] = dir_cos_dict["b2"][:, :irot]
        dir_cos_dict["b3"] = dir_cos_dict["b3"][:, :irot]
        dir_cos_dict["ncoor"] = irot

    return dir_cos_dict


def sdr_from_vector(faultnorms, slips):
    """
    Fault normal & slip vectors → strike/dip/rake (degrees).

    Reference: Aki & Richards p. 115; HASH (Hardebeck & Shearer 2002).
    """
    num_vect = faultnorms.shape[1]
    if faultnorms.shape != slips.shape or faultnorms.shape[0] != 3:
        raise ValueError("faultnorms and slips must be 3-by-n arrays of the same shape")

    phi = np.zeros(num_vect)
    delt = np.zeros(num_vect)
    lam = np.zeros(num_vect)

    undef_flag = (1 - np.abs(faultnorms[2, :])) <= 1e-7
    if np.any(undef_flag):
        ui = np.where(undef_flag)[0]
        phi[ui] = np.arctan2(-slips[0, ui], slips[1, ui])
        clam = np.cos(phi[ui]) * slips[0, ui] + np.sin(phi[ui]) * slips[1, ui]
        slam = np.sin(phi[ui]) * slips[0, ui] - np.cos(phi[ui]) * slips[1, ui]
        lam[ui] = np.arctan2(slam, clam)

    if np.any(~undef_flag):
        di = np.where(~undef_flag)[0]
        phi[di] = np.arctan2(-faultnorms[0, di], faultnorms[1, di])
        a = np.sqrt(faultnorms[0, di] ** 2 + faultnorms[1, di] ** 2)
        delt[di] = np.arctan2(a, -faultnorms[2, di])
        clam = np.cos(phi[di]) * slips[0, di] + np.sin(phi[di]) * slips[1, di]
        slam = -slips[2, di] / np.sin(delt[di])
        lam[di] = np.arctan2(slam, clam)

        tmp_ind = di[np.where(delt[di] > 0.5 * np.pi)[0]]
        if len(tmp_ind) > 0:
            delt[tmp_ind] = np.pi - delt[tmp_ind]
            phi[tmp_ind] += np.pi
            lam[tmp_ind] = -lam[tmp_ind]

    strike = np.mod(np.rad2deg(phi), 360)
    dip = np.rad2deg(delt)
    rake = np.mod(np.rad2deg(lam), 360)
    return strike, dip, rake


# ---------------------------------------------------------------------------
#  SDR → moment tensor
# ---------------------------------------------------------------------------
def sdr_to_mt(strike, dip, rake):
    """
    Strike/dip/rake (degrees) → full 3×3×N moment tensor (Up-South-East).

    Equivalent to MATLAB sdr2mt (Garrett Euler).
    """
    strike = np.asarray(strike, float).reshape(-1)
    dip = np.asarray(dip, float).reshape(-1)
    rake = np.asarray(rake, float).reshape(-1)
    N = strike.size

    sind = lambda x: np.sin(np.deg2rad(x))
    cosd = lambda x: np.cos(np.deg2rad(x))

    mt = np.zeros((N, 6))
    mt[:, 0] = sind(2 * dip) * sind(rake)
    mt[:, 1] = sind(dip) * cosd(rake) * sind(2 * strike) + sind(2 * dip) * sind(rake) * sind(strike) ** 2
    mt[:, 2] = sind(dip) * cosd(rake) * sind(2 * strike) - sind(2 * dip) * sind(rake) * cosd(strike) ** 2
    mt[:, 3] = cosd(dip) * cosd(rake) * cosd(strike) + cosd(2 * dip) * sind(rake) * sind(strike)
    mt[:, 4] = cosd(dip) * cosd(rake) * sind(strike) - cosd(2 * dip) * sind(rake) * cosd(strike)
    mt[:, 5] = sind(dip) * cosd(rake) * cosd(2 * strike) + 0.5 * sind(2 * dip) * sind(rake) * sind(2 * strike)

    mt[:, 1] *= -1
    mt[:, 3] *= -1
    mt[:, 5] *= -1

    M = np.zeros((3, 3, N))
    M[0, 0, :] = mt[:, 1]
    M[1, 1, :] = mt[:, 2]
    M[2, 2, :] = mt[:, 0]
    M[0, 1, :] = M[1, 0, :] = -mt[:, 5]
    M[0, 2, :] = M[2, 0, :] = mt[:, 3]
    M[1, 2, :] = M[2, 1, :] = -mt[:, 4]
    return M
