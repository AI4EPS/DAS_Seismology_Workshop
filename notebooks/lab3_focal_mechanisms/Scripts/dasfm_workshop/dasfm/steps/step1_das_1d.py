"""step1_das_1d — DAS ray parameters via 1D SKHASH lookup table."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dasfm.forward.geometry import build_model_grid, compute_fiber_orientation
from dasfm.io.velocity_io import load_velocity_1d, validate_velocity_1d
from dasfm.forward.takeoff_1d import create_lookup_table, compute_1d_ray_params
from dasfm.utils.step_utils import Logger, resolve_path, plot_ray_param_matrices
from dasfm.io.data import RayParamTable
from dasfm.io.event_catalog_io import validate_event_catalog
from dasfm.io.das_io import validate_das_geo


def _build_table_from_trial(
    trial: dict,
    receiver_x_all: np.ndarray,
    receiver_y_all: np.ndarray,
    rec_azi: np.ndarray,
    dasname: str | None,
    perturb_vert_uncert_km: float | None,
    perturb_horz_uncert_km: float | None,
) -> RayParamTable:
    """Wrap one compute_1d_ray_params trial in a nominal DAS RayParamTable."""
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
        forward_method="das_1d",
        rec_azi=rec_azi,
        dasname=dasname,
        perturb_vert_uncert_km=perturb_vert_uncert_km,
        perturb_horz_uncert_km=perturb_horz_uncert_km,
    )


def run(
    project_dir="",
    event_catalog="",
    das_geo="",
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
    """Compute DAS ray parameters via 1D SKHASH lookup table.

    Outputs:
        cache/ray_params/table_das_1d.h5
        cache/figs/stage1_das_1d/*.png
    """
    _required = {"project_dir": project_dir, "event_catalog": event_catalog,
                 "das_geo": das_geo, "vel_1d": vel_1d}
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")
    root      = Path(project_dir).resolve()
    CACHE_DIR = root / "cache"
    logger = Logger("step1_das_1d", log_dir=str(root / "logs"))

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
    DAS_FILE = resolve_path(das_geo, root)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info()
    logger.info("  step1_das_1d — Compute DAS ray parameters (1D)")
    logger.info()
    logger.info("=" * 60)
    logger.info(f"  Project dir  : {root}")
    logger.info("[1/6] Loading input data...  (method: 1D)")

    validate_event_catalog(CAT_FILE)
    validate_das_geo(DAS_FILE)
    for f in VEL_FILES:
        validate_velocity_1d(f)

    catalog  = pd.read_csv(CAT_FILE)
    das_info = pd.read_csv(DAS_FILE)
    vel_models = [load_velocity_1d(f) for f in VEL_FILES]
    vel_depth, vel_vp = vel_models[0]
    if n_vel_models > 1:
        logger.info(f"  Velocity models: {n_vel_models} (multi-model uncertainty)")

    rec_lat    = das_info["latitude"].values
    rec_lon    = das_info["longitude"].values

    logger.info(f"  Events        : {len(catalog)}")
    logger.info(f"  DAS receivers : {len(das_info)}")

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

    fig_dir = CACHE_DIR / "figs/stage1_das_1d"
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
    n_ch_total = len(receiver_x_all)

    ev_depth_km = catalog[depth_col].values.astype(np.float64)
    ev_lat = catalog["latitude"].values.astype(np.float64)
    ev_lon = catalog["longitude"].values.astype(np.float64)

    cos_lat_mid = np.cos(np.radians(0.5 * (ev_lat.mean() + rec_lat.mean())))
    dx_ll = (rec_lon[None, :] - ev_lon[:, None]) * 111.2 * cos_lat_mid
    dy_ll = (rec_lat[None, :] - ev_lat[:, None]) * 111.2
    sr_dist_km = np.sqrt(dx_ll ** 2 + dy_ll ** 2)

    # DAS fiber layout QC
    lon_min, lon_max = rec_lon.min(), rec_lon.max()
    lat_min, lat_max = rec_lat.min(), rec_lat.max()
    pad_lon = (lon_max - lon_min) * 0.05
    pad_lat = (lat_max - lat_min) * 0.05
    cos_lat = np.cos(np.radians(0.5 * (lat_min + lat_max)))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(rec_lon, rec_lat, 'b.', ms=1, label=f'DAS fiber ({n_ch_total} ch)')
    ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
    ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)
    ax.set_aspect(1.0 / cos_lat, adjustable='box')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('DAS fiber layout'); ax.legend()
    fig.savefig(fig_dir / "fiber_layout.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    if show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(fig_dir / "fiber_layout.png")))

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
        logger.info(f"\n[4/6] Computing takeoff angles for {n_ev} events × {n_ch_total} channels...")

    trials = compute_1d_ray_params(
        takeoff_table, tt_table, deptab, delttab,
        ev_depth_km, sr_dist_km,
        source_x, source_y, source_z,
        ev_lon, ev_lat, rec_lon, rec_lat, cos_lat_mid,
        n_ev, n_ch_total,
        perturb=PERTURB_LOCATION, nmc=NMC,
        vert_unc=VERT_UNCERT_KM, horz_unc=HORZ_UNCERT_KM,
        lookup_tables=lookup_tables if n_vel_models > 1 else None,
        logger=logger,
    )

    # DAS receiver orientation — fiber bearing in radians
    azi_fiber_deg = compute_fiber_orientation(receiver_x_all, receiver_y_all)
    rec_azi = np.deg2rad(azi_fiber_deg).astype(np.float32)

    # ── 5. Assemble RayParamTable ───────────────────────────────────────────
    logger.info(f"\n[5/6] Packaging ray parameters ({len(trials)} trial(s))...")
    v_unc = VERT_UNCERT_KM if PERTURB_LOCATION else None
    h_unc = HORZ_UNCERT_KM if PERTURB_LOCATION else None
    trial_tables = [
        _build_table_from_trial(
            tr, receiver_x_all, receiver_y_all, rec_azi,
            dasname, v_unc, h_unc,
        )
        for tr in trials
    ]
    table = trial_tables[0] if len(trial_tables) == 1 else RayParamTable.stack_mc_trials(trial_tables)

    # ── 6. Save + summary plots ─────────────────────────────────────────────
    out_dir = CACHE_DIR / "ray_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_das_1d.h5"
    logger.info(f"\n[6/6] Saving ray parameter table...")
    table.to_hdf5(out_path)
    logger.info(f"  → {out_path}")

    nominal = table if not table.is_perturbed else table.trial(0)
    plot_ray_param_matrices(
        fig_dir, nominal.traveltime, nominal.takeoff, nominal.azimuth,
        xlabel="Channel index", show_plots=show_plots,
    )
    logger.info(f"  → {fig_dir}  (3 plots)")

    logger.info("=" * 60)
    logger.info(f"  Done  ({time.time() - t0:.1f} s)")
    logger.info("=" * 60)
    logger.close()
