"""step1_sta_1d — STA ray parameters via 1D SKHASH lookup table."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dasfm.forward.geometry import build_model_grid
from dasfm.io.velocity_io import load_velocity_1d, validate_velocity_1d
from dasfm.forward.takeoff_1d import create_lookup_table, compute_1d_ray_params
from dasfm.utils.step_utils import Logger, resolve_path, plot_ray_param_matrices
from dasfm.io.data import RayParamTable
from dasfm.io.event_catalog_io import validate_event_catalog
from dasfm.io.sta_io import validate_sta_geo


def _build_table_from_trial(
    trial: dict,
    receiver_x_all: np.ndarray,
    receiver_y_all: np.ndarray,
    network: np.ndarray,
    station: np.ndarray,
    location: np.ndarray,
    dasname: str | None,
    perturb_vert_uncert_km: float | None,
    perturb_horz_uncert_km: float | None,
) -> RayParamTable:
    """Wrap one compute_1d_ray_params trial in a nominal RayParamTable.

    Azimuth is computed geometrically from the trial's (possibly perturbed)
    source coordinates and the full receiver set.
    """
    sx = trial["source_x"]
    sy = trial["source_y"]
    sz = trial["source_z"]
    az = np.arctan2(
        receiver_y_all[None, :] - sy[:, None],
        receiver_x_all[None, :] - sx[:, None],
    ).astype(np.float32)
    return RayParamTable(
        traveltime=trial["traveltime"],
        takeoff=trial["takeoff"],
        azimuth=az,
        raypath_length=trial["raypath_length"],
        source_x=sx,
        source_y=sy,
        source_z=sz,
        receiver_x=np.asarray(receiver_x_all, dtype=np.float64),
        receiver_y=np.asarray(receiver_y_all, dtype=np.float64),
        forward_method="sta_1d",
        network=network,
        station=station,
        location=location,
        dasname=dasname,
        perturb_vert_uncert_km=perturb_vert_uncert_km,
        perturb_horz_uncert_km=perturb_horz_uncert_km,
    )


def run(
    project_dir="",
    event_catalog="",
    sta_geo="",
    vel_1d="",
    lookup_depth_km=(0, 39, 3),
    lookup_distance_km=(0, 200, 2),
    ray_parameter_count=9000,
    grid_spacing_km=0.2,
    compute_uncertainty=False,
    monte_carlo_trials=30,
    vertical_uncertainty_km=1.0,
    horizontal_uncertainty_km=1.0,
    dasname=None,
    show_plots=False):
    """Compute STA ray parameters via 1D SKHASH lookup table.

    Outputs:
        cache/ray_params/table_sta_1d.h5
        cache/figs/stage1_sta_1d/*.png
    """
    _required = {"project_dir": project_dir, "event_catalog": event_catalog,
                 "sta_geo": sta_geo, "vel_1d": vel_1d}
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")
    root       = Path(project_dir).resolve()
    CACHE_DIR  = root / "cache"
    logger = Logger("step1_sta_1d", log_dir=str(root / "logs"))

    LOOK_DEP       = tuple(lookup_depth_km)
    LOOK_DEL       = tuple(lookup_distance_km)
    NUMP           = ray_parameter_count
    NMC            = monte_carlo_trials
    VERT_UNCERT_KM = vertical_uncertainty_km
    HORZ_UNCERT_KM = horizontal_uncertainty_km
    PERTURB_LOCATION = compute_uncertainty
    dr             = grid_spacing_km

    if isinstance(vel_1d, (str, Path)):
        vel_1d_list = [vel_1d]
    else:
        vel_1d_list = list(vel_1d)
    n_vel_models = len(vel_1d_list)

    VEL_FILES = [resolve_path(v, root) for v in vel_1d_list]
    CAT_FILE = resolve_path(event_catalog, root)
    STA_FILE = resolve_path(sta_geo, root)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info()
    logger.info("  step1_sta_1d — Compute STA ray parameters (1D)")
    logger.info()
    logger.info("=" * 60)
    logger.info(f"  Project dir  : {root}")
    logger.info("[1/6] Loading input data...  (method: STA 1D)")

    validate_event_catalog(CAT_FILE)
    validate_sta_geo(STA_FILE)
    for f in VEL_FILES:
        validate_velocity_1d(f)

    catalog  = pd.read_csv(CAT_FILE)
    vel_models = [load_velocity_1d(f) for f in VEL_FILES]
    vel_depth, vel_vp = vel_models[0]
    if n_vel_models > 1:
        logger.info(f"  Velocity models: {n_vel_models} (multi-model uncertainty)")

    sta_df     = pd.read_csv(STA_FILE)
    sta_unique = sta_df.drop_duplicates(subset=["network", "station"])
    rec_lat    = sta_unique["latitude"].values
    rec_lon    = sta_unique["longitude"].values
    n_sta      = len(sta_unique)

    from dasfm.utils.step_utils import clean_location_array as _clean_loc
    network_arr  = sta_unique["network"].astype(str).values
    station_arr  = sta_unique["station"].astype(str).values
    location_arr = (_clean_loc(sta_unique["location"])
                    if "location" in sta_unique.columns
                    else np.array(["00"] * n_sta, dtype=object))

    logger.info(f"  Events   : {len(catalog)}")
    logger.info(f"  Stations : {n_sta} (unique)")

    # ── 2. Build Cartesian grid ───────────────────────────────────────────────
    depth_col = "depth" if "depth" in catalog.columns else "depth_km"
    depth_max = float(catalog[depth_col].max()) * 2
    logger.info(f"\n[2/6] Building Cartesian grid (dr=dz={dr} km, depth_max={depth_max:.1f} km)...")

    geo = build_model_grid(
        catalog,
        receiver_lat=rec_lat,
        receiver_lon=rec_lon,
        depth_max=depth_max,
        dx=dr, dy=dr, dz=dr,
        logger=logger,
    )

    fig_dir = CACHE_DIR / "figs/stage1_sta_1d"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    elev_m = (geo['h_max'] - geo['z_grid'][geo['surface_iz']]) * 1000
    plt.imshow(elev_m, aspect='auto', origin='lower', cmap='gist_earth')
    plt.colorbar(label="Elevation (m)")
    plt.plot(geo['receiver_iy'], geo['receiver_ix'], 'r.', ms=1)
    plt.plot(geo['source_iy'], geo['source_ix'], 'k*', ms=4)
    plt.title("Topography + receivers(red) + sources(black)")
    plt.savefig(fig_dir / "grid_map.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    if show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(fig_dir / "grid_map.png")))

    # ── 3. Source / receiver coordinates and lookup tables ──────────────────
    source_x = np.asarray(geo["source_x"], dtype=np.float64).ravel()
    source_y = np.asarray(geo["source_y"], dtype=np.float64).ravel()
    source_z = np.asarray(geo["source_z"], dtype=np.float64).ravel()
    n_ev = len(source_x)

    receiver_x_all = np.asarray(geo["receiver_x"], dtype=np.float64).ravel()
    receiver_y_all = np.asarray(geo["receiver_y"], dtype=np.float64).ravel()
    n_rec = len(receiver_x_all)

    ev_depth_km = catalog[depth_col].values.astype(np.float64)
    ev_lat = catalog["latitude"].values.astype(np.float64)
    ev_lon = catalog["longitude"].values.astype(np.float64)

    cos_lat_mid = np.cos(np.radians(0.5 * (ev_lat.mean() + rec_lat.mean())))
    dx_ll = (rec_lon[None, :] - ev_lon[:, None]) * 111.2 * cos_lat_mid
    dy_ll = (rec_lat[None, :] - ev_lat[:, None]) * 111.2
    sr_dist_km = np.sqrt(dx_ll ** 2 + dy_ll ** 2)

    logger.info(f"\n[3/6] Building 1-D takeoff & traveltime lookup tables (SKHASH-style)...")
    lookup_tables = []
    for v_idx, (vd, vv) in enumerate(vel_models):
        deptab, delttab, takeoff_table, tt_table = create_lookup_table(
            vd, vv, look_dep=LOOK_DEP, look_del=LOOK_DEL, nump=NUMP,
        )
        lookup_tables.append((deptab, delttab, takeoff_table, tt_table))
        if n_vel_models > 1:
            logger.info(f"  Table {v_idx}: Vp={vv.min():.2f}–{vv.max():.2f} km/s")
    deptab, delttab, takeoff_table, tt_table = lookup_tables[0]

    # ── 4. Compute per-trial ray parameters ─────────────────────────────────
    if PERTURB_LOCATION:
        logger.info(f"\n[4/6] Computing with location perturbation "
                    f"(nmc={NMC}, vert={VERT_UNCERT_KM} km, horz={HORZ_UNCERT_KM} km"
                    f", vel_models={n_vel_models})...")
    elif n_vel_models > 1:
        logger.info(f"\n[4/6] Computing {n_vel_models} velocity-model trials...")
    else:
        logger.info(f"\n[4/6] Computing takeoff angles for {n_ev} events × {n_rec} stations...")

    trials = compute_1d_ray_params(
        takeoff_table, tt_table, deptab, delttab,
        ev_depth_km, sr_dist_km,
        source_x, source_y, source_z,
        ev_lon, ev_lat, rec_lon, rec_lat, cos_lat_mid,
        n_ev, n_rec,
        perturb=PERTURB_LOCATION, nmc=NMC,
        vert_unc=VERT_UNCERT_KM, horz_unc=HORZ_UNCERT_KM,
        lookup_tables=lookup_tables if n_vel_models > 1 else None,
        logger=logger,
    )

    # ── 5. Assemble RayParamTable (per trial) ───────────────────────────────
    logger.info(f"\n[5/6] Packaging ray parameters ({len(trials)} trial(s))...")
    v_unc = VERT_UNCERT_KM if PERTURB_LOCATION else None
    h_unc = HORZ_UNCERT_KM if PERTURB_LOCATION else None
    trial_tables = [
        _build_table_from_trial(
            tr, receiver_x_all, receiver_y_all,
            network_arr, station_arr, location_arr,
            dasname, v_unc, h_unc,
        )
        for tr in trials
    ]
    if len(trial_tables) == 1:
        table = trial_tables[0]
    else:
        table = RayParamTable.stack_mc_trials(trial_tables)

    # ── 6. Save + summary plots ─────────────────────────────────────────────
    out_dir = CACHE_DIR / "ray_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_sta_1d.h5"
    logger.info(f"\n[6/6] Saving ray parameter table...")
    table.to_hdf5(out_path)
    logger.info(f"  → {out_path}")

    # Plot nominal (trial-0) slice
    nominal = table if not table.is_perturbed else table.trial(0)
    plot_ray_param_matrices(
        fig_dir, nominal.traveltime, nominal.takeoff, nominal.azimuth,
        xlabel="Station index", show_plots=show_plots,
    )
    logger.info(f"  → {fig_dir}  (3 plots)")

    logger.info("=" * 60)
    logger.info(f"  Done  ({time.time() - t0:.1f} s)")
    logger.info("=" * 60)
    logger.close()
