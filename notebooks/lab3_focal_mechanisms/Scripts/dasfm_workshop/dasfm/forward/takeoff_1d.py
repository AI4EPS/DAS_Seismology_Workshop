"""1-D velocity model takeoff angle & traveltime lookup table (SKHASH-style).

Computes takeoff angles and traveltimes via 1-D ray tracing through a layered
velocity model.  Based on HASH/SKHASH (Hardebeck & Shearer 2002; Skoumal et al.
2024).

Usage
-----
>>> deptab, delttab, takeoff_table, tt_table = create_lookup_table(vel_depth, vel_vp)
>>> takeoff_deg = lookup_table(takeoff_table, depth_km, dist_km, deptab, delttab)
>>> traveltime_s = lookup_table(tt_table, depth_km, dist_km, deptab, delttab)

Takeoff convention: 0° = straight down, 90° = horizontal, 180° = straight up.
"""
from __future__ import annotations

import numpy as np


def create_lookup_table(
    vel_depth: np.ndarray,
    vel_vp: np.ndarray,
    look_dep: tuple[float, float, float] = (0, 39, 3),
    look_del: tuple[float, float, float] = (0, 200, 2),
    nump: int = 9000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 2-D takeoff-angle and traveltime lookup tables from a 1-D velocity model.

    Parameters
    ----------
    vel_depth : 1-D array — depth nodes [km] (increasing).
    vel_vp    : 1-D array — Vp at each depth node [km/s].
    look_dep  : (min_depth, max_depth, step) for depth axis [km].
    look_del  : (min_dist, max_dist, step) for distance axis [km].
    nump      : number of ray parameters to trace.

    Returns
    -------
    deptab    : 1-D array of depth bins [km].
    delttab   : 1-D array of distance bins [km].
    takeoff_table : 2-D array (n_del, n_dep) of takeoff angles [deg].
    tt_table  : 2-D array (n_del, n_dep) of first-arrival traveltimes [s].
    """
    deptab = np.arange(look_dep[0], look_dep[1] + look_dep[2], look_dep[2])
    delttab = np.arange(look_del[0], look_del[1] + look_del[2], look_del[2])

    takeoff_table, tt_table = create_takeoff_table(
        np.column_stack([vel_depth, vel_vp]),
        deptab, delttab, nump,
        nx0=len(delttab), nd0=len(deptab),
    )
    return deptab, delttab, takeoff_table, tt_table


def lookup_table(
    table: np.ndarray,
    depth_km: np.ndarray,
    dist_km: np.ndarray,
    deptab: np.ndarray,
    delttab: np.ndarray,
) -> np.ndarray:
    """
    Query a lookup table (takeoff or traveltime) via bilinear interpolation.

    Parameters
    ----------
    table    : (n_del, n_dep) lookup table from create_lookup_table.
    depth_km : source depths [km], any shape.
    dist_km  : source-receiver distances [km], same shape as depth_km.
    deptab   : depth axis of the table.
    delttab  : distance axis of the table.

    Returns
    -------
    values : interpolated values, same shape as depth_km.
    """
    depth_km = np.asarray(depth_km, dtype=np.float64)
    dist_km = np.asarray(dist_km, dtype=np.float64)

    # Clamp to table range
    depth_km = np.clip(depth_km, deptab[0], deptab[-1])
    dist_km = np.clip(dist_km, delttab[0], delttab[-1])

    dep_step = deptab[1] - deptab[0] if len(deptab) > 1 else 1.0
    del_step = delttab[1] - delttab[0] if len(delttab) > 1 else 1.0

    # Depth indices
    id1 = np.clip(((depth_km - deptab[0]) / dep_step).astype(int), 0, len(deptab) - 2)
    id2 = id1 + 1

    # Distance indices
    ix1 = np.clip(((dist_km - delttab[0]) / del_step).astype(int), 0, len(delttab) - 2)
    ix2 = ix1 + 1

    # Bilinear interpolation
    xfrac = (dist_km - delttab[ix1]) / (delttab[ix2] - delttab[ix1])
    dfrac = (depth_km - deptab[id1]) / (deptab[id2] - deptab[id1])

    t1 = table[ix1, id1] + xfrac * (table[ix2, id1] - table[ix1, id1])
    t2 = table[ix1, id2] + xfrac * (table[ix2, id2] - table[ix1, id2])
    values = t1 + dfrac * (t2 - t1)

    return values


# Keep old name as alias for backward compatibility
lookup_takeoff = lookup_table


# ---------------------------------------------------------------------------
#  Internal: ray tracing (adapted from SKHASH/HASH)
# ---------------------------------------------------------------------------
def create_takeoff_table(vmodel_depthvp, deptab, delttab, nump, nx0, nd0):
    """
    1-D ray tracing to build takeoff angle and traveltime tables.

    Returns
    -------
    takeoff_table : (nx0, nd0) takeoff angles [deg].
    tt_table  : (nx0, nd0) first-arrival traveltimes [s].
    """
    # Extend model if needed
    if vmodel_depthvp[-1, 0] < deptab[-1]:
        vmodel_depthvp = np.vstack((vmodel_depthvp, vmodel_depthvp[-1, :]))
        vmodel_depthvp[-1, 0] = deptab[-1] + 1
        vmodel_depthvp[-1, 1] = vmodel_depthvp[-1, 1] + 0.001

    takeoff_table = np.zeros((nx0, len(deptab))) - 999
    tt_table = np.full((nx0, len(deptab)), np.nan)

    ndel = len(delttab)
    ndep = len(deptab)
    pmin = 0

    z = vmodel_depthvp[:, 0]
    alpha = vmodel_depthvp[:, 1]

    npts = len(z)
    z = np.hstack((z, z[npts - 1]))
    alpha = np.hstack((alpha, alpha[npts - 1]))

    # Insert depth table nodes into model
    for i in range(npts - 1, 0, -1):
        for idep in range(ndep - 1, -1, -1):
            if (z[i - 1] <= (deptab[idep] - 0.00001)) and (z[i] >= (deptab[idep] + 0.00001)):
                z = np.insert(z, i, z[i - 1])
                alpha = np.insert(alpha, i, alpha[i - 1])
                z[i] = deptab[idep]
                frac = (z[i] - z[i - 1]) / (z[i + 1] - z[i - 1])
                alpha[i] = alpha[i - 1] + frac * (alpha[i + 1] - alpha[i - 1])

    slow = 1 / alpha
    pmax = slow[0]
    pstep = (pmax - pmin) / nump

    npmax = int((pmax + pstep / 2 - pmin) / pstep) + 1

    depxcor = np.zeros((npmax, nd0))
    depucor = np.zeros((npmax, nd0))
    deptcor = np.zeros((npmax, nd0))

    depxcor[:, np.where(deptab != 0)[0]] = -999
    deptcor[:, np.where(deptab != 0)[0]] = -999

    ptab = np.linspace(pmin, pmin + pstep * (npmax - 1), num=npmax)

    h_array = z[1:] - z[:-1]
    utop = slow[:-1]
    ubot = slow[1:]

    # LAYERTRACE equivalent
    dx = np.zeros((npmax, len(utop)))
    dt = np.zeros((npmax, len(utop)))
    irtr = np.zeros((npmax, len(utop)), dtype=int)

    qs = np.full((npmax, len(utop)), np.nan)
    qr = np.full((npmax, len(utop)), np.nan)
    ytop = utop - ptab[:, np.newaxis]
    ytop_pos = ytop > 0
    qs[ytop_pos] = (ytop * (utop + ptab[:, np.newaxis]))[ytop_pos]
    qs[ytop_pos] = np.sqrt(qs[ytop_pos])

    qr = np.arctan2(qs, ptab[:, np.newaxis])

    b = np.ma.divide(-np.log(ubot / utop), h_array).filled(np.nan)

    etau = qs - qr * ptab[:, np.newaxis]
    ex = qr

    ybot = ubot - ptab[:, np.newaxis]
    y_sub = ybot <= 0
    y_pos = ybot > 0
    irtr[y_sub] = 2
    irtr[y_pos] = 1
    irtr[~ytop_pos] = 0

    dx[y_sub] = ex[y_sub]
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = dx / b
        dtau = etau / b
    dt[y_sub] = dtau[y_sub] + (dx * ptab[:, np.newaxis])[y_sub]

    q = np.full(ybot.shape, np.nan)
    q[y_pos] = (ybot * (ubot + ptab[:, np.newaxis]))[y_pos]
    qs2 = np.sqrt(q)

    qr2 = np.arctan2(qs2, ptab[:, np.newaxis])
    etau = etau - qs2 + ptab[:, np.newaxis] * qr2
    ex = ex - qr2

    with np.errstate(divide="ignore", invalid="ignore"):
        exb = ex / b
        dtau = etau / b
    dx[y_pos] = exb[y_pos]
    dt[y_pos] = dtau[y_pos] + (exb * ptab[:, np.newaxis])[y_pos]

    # NaN after ray turns
    x = (irtr == 0) | (irtr == 2)
    idx = np.arange(npmax), x.argmax(axis=1)
    tmp = x[idx] == True

    idx_1 = idx[0][tmp], idx[1][tmp] + 1
    tmp_1 = idx_1[1] < (len(utop) - 1)
    idx_1 = idx_1[0][tmp_1], idx_1[1][tmp_1]
    for xx in range(len(idx_1[0])):
        row, col = idx_1[0][xx], idx_1[1][xx]
        dx[row, col:] = np.nan
        dt[row, col:] = np.nan

    deltab = np.nansum(dx, axis=1) * 2
    tttab = np.nansum(dt, axis=1) * 2

    idx_2 = idx[0][tmp], idx[1][tmp]
    tmp_2 = idx_2[1] < (len(utop) - 1)
    idx_2 = idx_2[0][tmp_2], idx_2[1][tmp_2]
    for xx in range(len(idx_2[0])):
        row, col = idx_2[0][xx], idx_2[1][xx]
        dx[row, col:] = np.nan
        dt[row, col:] = np.nan

    depxcor = np.cumsum(dx, axis=1)
    deptcor = np.cumsum(dt, axis=1)
    output_col_ind = np.where(np.isin(z, deptab))[0] - 1
    depxcor = depxcor[:, output_col_ind]
    deptcor = deptcor[:, output_col_ind]
    depxcor[:, 0] = 0
    deptcor[:, 0] = 0
    depucor[:] = slow[output_col_ind + 1]
    depucor[np.isnan(depxcor)] = -999
    depxcor[np.isnan(depxcor)] = -999
    deptcor[np.isnan(deptcor)] = -999

    x = np.diff(depxcor, axis=0) <= 0
    idx = x.argmax(axis=0) + 1
    tmp = x[idx, np.arange(nd0)] == False
    idx[tmp] = npmax - 1

    for idep in range(ndep):
        # upgoing rays
        xsave_up = depxcor[:(idx[idep]), idep]
        tsave_up = deptcor[:(idx[idep]), idep]
        usave_up = depucor[:(idx[idep]), idep]
        psave_up = -1 * ptab[:(idx[idep])]

        # downgoing rays
        down_idx = np.where((depxcor[:, idep] != -999) & (deltab != -999))[0][::-1]
        xsave_down = deltab[down_idx] - depxcor[down_idx, idep]
        tsave_down = tttab[down_idx] - deptcor[down_idx, idep]
        usave_down = depucor[down_idx, idep]
        psave_down = ptab[down_idx]

        xsave = np.hstack([xsave_up, xsave_down])
        tsave = np.hstack([tsave_up, tsave_down])
        usave = np.hstack([usave_up, usave_down])
        psave = np.hstack([psave_up, psave_down])

        scr1 = np.zeros(ndel)
        tt_col = np.full(ndel, np.nan)
        for idel in range(1, ndel):
            del_x = delttab[idel]
            ind = np.where((xsave[:-1] <= del_x) & (xsave[1:] >= del_x))[0] + 1
            if len(ind) == 0:
                continue
            frac = (del_x - xsave[ind - 1]) / (xsave[ind] - xsave[ind - 1])
            t_interp = tsave[ind - 1] + frac * (tsave[ind] - tsave[ind - 1])
            min_idx = np.argmin(t_interp)
            min_ind = ind[min_idx]
            scr1[idel] = psave[min_ind] / usave[min_ind]
            tt_col[idel] = t_interp[min_idx]

        angle = np.rad2deg(np.arcsin(np.clip(scr1, -1, 1)))
        angle_flag = angle >= 0
        angle *= -1
        angle[angle_flag] += 180
        takeoff_table[:, idep] = angle
        tt_table[:, idep] = tt_col

    if delttab[0] == 0:
        takeoff_table[0, :] = 0.0
        tt_table[0, :] = 0.0
    takeoff_table = np.round(takeoff_table, 1)
    tt_table = np.round(tt_table, 6)
    return takeoff_table, tt_table


def compute_1d_ray_params(
    takeoff_table, tt_table, deptab, delttab,
    ev_depth_km, sr_dist_km,
    source_x, source_y, source_z,
    ev_lon, ev_lat, rec_lon, rec_lat, cos_lat_mid,
    n_ev, n_rec,
    perturb=False, nmc=30, vert_unc=1.0, horz_unc=1.0,
    lookup_tables=None,
    logger=None,
) -> list[dict]:
    """Compute takeoff / traveltime / raypath length from 1-D lookup tables.

    Returns a list of per-trial dicts (one element for nominal runs, ``nmc``
    elements for MC runs, ``n_vel_models`` elements for a multi-model non-MC
    sweep). Trial 0 is always the unperturbed / primary-model result.

    The caller is responsible for computing azimuth geometrically and
    wrapping each trial in a :class:`dasfm.io.data.RayParamTable`.

    Parameters
    ----------
    takeoff_table, tt_table : np.ndarray
        Nominal-model lookup tables from :func:`create_lookup_table`.
    deptab, delttab : np.ndarray
        Depth and distance axes of the lookup tables.
    ev_depth_km : np.ndarray, shape (n_ev,)
    sr_dist_km : np.ndarray, shape (n_ev, n_rec)
    source_x, source_y, source_z : np.ndarray, shape (n_ev,)
    ev_lon, ev_lat : np.ndarray, shape (n_ev,)
    rec_lon, rec_lat : np.ndarray, shape (n_rec,)
    cos_lat_mid : float
    n_ev, n_rec : int
    perturb : bool
        If ``True``, run Monte Carlo location perturbation (``nmc`` trials).
    nmc : int
        Number of MC trials (only used when ``perturb=True``).
    vert_unc, horz_unc : float
        Vertical and horizontal Gaussian perturbation sigmas (km).
    lookup_tables : list of ``(deptab, delttab, takeoff_table, tt_table)`` tuples, optional
        Multiple velocity model tables. With ``perturb=True`` each trial
        randomly selects a model (trial 0 uses model 0). With
        ``perturb=False`` each model becomes one trial (no location jitter).
    logger : Logger or None
        Optional logger for per-trial progress messages.

    Returns
    -------
    trials : list of dict
        Each element has keys ``traveltime``, ``takeoff``, ``raypath_length``
        (shape ``(n_ev, n_rec)``, float32) and ``source_x``, ``source_y``,
        ``source_z`` (shape ``(n_ev,)``, float64).
    """
    n_vel_models = len(lookup_tables) if lookup_tables else 1
    trials: list[dict] = []

    def _single_trial(dep_v, del_v, takeoff_v, tt_v, sx, sy, sz, dep_mc, dist_mc):
        """Run one lookup + assemble a trial dict."""
        depth_2d = np.broadcast_to(dep_mc[:, None], (n_ev, n_rec)).copy()
        ito = (np.pi - np.deg2rad(
            lookup_table(takeoff_v, depth_2d, dist_mc, dep_v, del_v)
        )).astype(np.float32)
        tt = lookup_table(tt_v, depth_2d, dist_mc, dep_v, del_v).astype(np.float32)
        r = np.sqrt(dist_mc ** 2 + dep_mc[:, None] ** 2).astype(np.float32)
        return {
            "traveltime": tt,
            "takeoff": ito,
            "raypath_length": r,
            "source_x": np.asarray(sx, dtype=np.float64),
            "source_y": np.asarray(sy, dtype=np.float64),
            "source_z": np.asarray(sz, dtype=np.float64),
        }

    # Multi-model, no perturbation: one trial per velocity model
    if lookup_tables and not perturb and n_vel_models > 1:
        for v_idx, (dep_v, del_v, takeoff_v, tt_v) in enumerate(lookup_tables):
            trials.append(_single_trial(
                dep_v, del_v, takeoff_v, tt_v,
                source_x, source_y, source_z,
                ev_depth_km, sr_dist_km,
            ))
            if logger:
                logger.log(f"    trial {v_idx + 1}/{n_vel_models} [v{v_idx}]")
        return trials

    # Perturbation (with optional multi-model draw)
    if perturb:
        rng = np.random.default_rng(42)
        for imc in range(nmc):
            v_idx = 0 if imc == 0 else rng.integers(0, n_vel_models)
            if lookup_tables:
                dep_v, del_v, takeoff_v, tt_v = lookup_tables[v_idx]
            else:
                dep_v, del_v, takeoff_v, tt_v = deptab, delttab, takeoff_table, tt_table

            if imc == 0:
                sx, sy, sz = source_x.copy(), source_y.copy(), source_z.copy()
                dep_mc = ev_depth_km.copy()
                dist_mc = sr_dist_km.copy()
            else:
                dep_mc = ev_depth_km + rng.normal(size=n_ev) * vert_unc
                dep_mc = np.clip(dep_mc, dep_v[0], dep_v[-1])
                rand_angle = rng.uniform(0, 2 * np.pi, size=n_ev)
                rand_dist = rng.normal(size=n_ev) * horz_unc
                sx = source_x + rand_dist * np.cos(rand_angle)
                sy = source_y + rand_dist * np.sin(rand_angle)
                sz = source_z + rng.normal(size=n_ev) * vert_unc
                sz = np.clip(sz, 0, None)
                dx_p = (rec_lon[None, :] - ev_lon[:, None]) * 111.2 * cos_lat_mid
                dy_p = (rec_lat[None, :] - ev_lat[:, None]) * 111.2
                dx_p += rand_dist[:, None] * np.cos(rand_angle[:, None])
                dy_p += rand_dist[:, None] * np.sin(rand_angle[:, None])
                dist_mc = np.sqrt(dx_p ** 2 + dy_p ** 2)

            trials.append(_single_trial(
                dep_v, del_v, takeoff_v, tt_v, sx, sy, sz, dep_mc, dist_mc,
            ))
            if logger and ((imc + 1) % 10 == 0 or imc == nmc - 1):
                v_label = f" [v{v_idx}]" if n_vel_models > 1 else ""
                logger.log(f"    trial {imc + 1}/{nmc}{v_label}")
        return trials

    # Nominal single trial
    trials.append(_single_trial(
        deptab, delttab, takeoff_table, tt_table,
        source_x, source_y, source_z,
        ev_depth_km, sr_dist_km,
    ))
    return trials
