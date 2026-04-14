"""step1_sta_3d — STA ray parameters via 3D Eikonal (pykonal).

Uses a 3D tomographic velocity model (.npz) with per-receiver grid
alignment for exact receiver positioning.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from dasfm.forward.geometry import build_model_grid
from dasfm.forward.eikonal_3d import (
    load_velocity_3d, validate_velocity_3d,
    build_receiver_grid, interpolate_tomo_to_grid,
    compute_eikonal_lookup, check_eikonal_memory,
)
from dasfm.forward.eval_ray_lookup import interp_lookup_channels
from dasfm.utils.step_utils import Logger, resolve_path, plot_ray_param_matrices
from dasfm.io.data import RayParamDB, RayParamTable
from dasfm.io.event_catalog_io import validate_event_catalog
from dasfm.io.sta_io import validate_sta_geo
from dasfm.io.topo_io import validate_topo


from dasfm.utils.step_utils import station_filename as _station_filename
from dasfm.utils.step_utils import clean_location_array as _clean_location_array


def _compute_eikonal_worker(args):
    """Worker for parallel 3D Eikonal computation (single station)."""
    import time as _t
    from dasfm.forward.eikonal_3d import (
        build_receiver_grid, interpolate_tomo_to_grid, compute_eikonal_lookup,
    )
    (tomo, geo, rx, ry, rz, dr, lut_dir_str, filename, i) = args
    t0 = _t.perf_counter()

    x_g, y_g, z_g, rix, riy, riz, nx, ny, nz = build_receiver_grid(
        geo, rx, ry, rz, dr)
    vp_local = interpolate_tomo_to_grid(tomo, geo, x_g, y_g, z_g)
    db, _timing_str = compute_eikonal_lookup(
        vp_local, nx, ny, nz, dr,
        rix, riy, riz,
        x_g, y_g, z_g, geo)
    db.to_hdf5(Path(lut_dir_str) / filename)
    return {"i": i, "dt": _t.perf_counter() - t0}


def run(
    project_dir="",
    event_catalog="",
    sta_geo="",
    vel_3d="",
    topo="",
    grid_spacing_km=0.2,
    compute_uncertainty=False,
    monte_carlo_trials=30,
    vertical_uncertainty_km=1.0,
    horizontal_uncertainty_km=1.0,
    precomputed_lookup=None,
    num_cpu_workers=1,
    dasname=None,
    show_plots=False):
    """Compute STA ray parameters via 3D Eikonal (pykonal).

    Outputs:
        cache/ray_params/sta_3d/{network}_{station}_{location}.h5   (Layer 1)
        cache/ray_params/table_sta_3d.h5                              (Layer 2+3)
        cache/figs/stage1_sta_3d/*.png
    """
    _required = {"project_dir": project_dir, "event_catalog": event_catalog,
                 "sta_geo": sta_geo, "vel_3d": vel_3d, "topo": topo}
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")

    root      = Path(project_dir).resolve()
    CACHE_DIR = root / "cache"
    logger = Logger("step1_sta_3d", log_dir=str(root / "logs"))

    dr             = grid_spacing_km
    NMC            = monte_carlo_trials
    VERT_UNCERT_KM = vertical_uncertainty_km
    HORZ_UNCERT_KM = horizontal_uncertainty_km
    PERTURB_LOCATION = compute_uncertainty

    CAT_FILE    = resolve_path(event_catalog, root)
    STA_FILE    = resolve_path(sta_geo, root)
    TOPO_FILE   = resolve_path(topo, root)
    VEL_3D_FILE = resolve_path(vel_3d, root)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info()
    logger.info("  step1_sta_3d — Compute STA ray parameters (3D Eikonal)")
    logger.info()
    logger.info("=" * 60)
    logger.info("[1/7] Loading input data...  (method: STA 3D Eikonal)")

    validate_event_catalog(CAT_FILE)
    validate_sta_geo(STA_FILE)
    validate_topo(TOPO_FILE)
    validate_velocity_3d(VEL_3D_FILE)

    catalog = pd.read_csv(CAT_FILE)

    sta_df = pd.read_csv(STA_FILE)
    sta_unique = sta_df.drop_duplicates(subset=["network", "station"])
    rec_lat    = sta_unique["latitude"].values
    rec_lon    = sta_unique["longitude"].values
    n_sta      = len(sta_unique)

    network_arr  = sta_unique["network"].astype(str).values
    station_arr  = sta_unique["station"].astype(str).values
    location_arr = (_clean_location_array(sta_unique["location"])
                    if "location" in sta_unique.columns
                    else np.array(["00"] * n_sta, dtype=object))

    station_filenames = [
        _station_filename(network_arr[i], station_arr[i], location_arr[i])
        for i in range(n_sta)
    ]

    logger.info(f"  Events        : {len(catalog)}")
    logger.info(f"  Stations      : {n_sta} (unique)")

    # ── Load 3D tomo velocity model ──────────────────────────────────────
    logger.info("\n[2/7] Loading 3D tomographic velocity model...")
    tomo = load_velocity_3d(VEL_3D_FILE)
    logger.info(f"  Vp range: {tomo['vp_min']:.3f}–{tomo['vp_max']:.3f} km/s")
    logger.info(f"  Lat: {tomo['lat_axis'][0]:.4f}–{tomo['lat_axis'][-1]:.4f}")
    logger.info(f"  Lon: {tomo['lon_axis'][0]:.4f}–{tomo['lon_axis'][-1]:.4f}")
    logger.info(f"  Z:   {tomo['z_axis'][0]:.1f}–{tomo['z_axis'][-1]:.1f} km")

    # ── Build Cartesian grid ─────────────────────────────────────────────
    from dasfm.io.topo_io import load_topo
    elev_lat, elev_lon, elev = load_topo(TOPO_FILE)

    depth_col = "depth" if "depth" in catalog.columns else "depth_km"
    depth_max = float(catalog[depth_col].max()) * 2
    logger.info(f"\n[3/7] Building Cartesian grid (dr=dz={dr} km, depth_max={depth_max:.1f} km)...")

    geo = build_model_grid(
        catalog,
        receiver_lat=rec_lat, receiver_lon=rec_lon,
        topo_lat=elev_lat, topo_lon=elev_lon, topo_elev_m=elev,
        depth_max=depth_max, dx=dr, dy=dr, dz=dr,
        logger=logger,
    )

    fig_dir = CACHE_DIR / "figs/stage1_sta_3d"
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

    source_x = np.asarray(geo["source_x"], dtype=np.float64).ravel()
    source_y = np.asarray(geo["source_y"], dtype=np.float64).ravel()
    source_z = np.asarray(geo["source_z"], dtype=np.float64).ravel()
    n_ev = len(source_x)

    receiver_x_all = np.asarray(geo["receiver_x"], dtype=np.float64).ravel()
    receiver_y_all = np.asarray(geo["receiver_y"], dtype=np.float64).ravel()
    receiver_z_all = np.asarray(geo["receiver_z"], dtype=np.float64).ravel()

    # ── 3D Eikonal solve ─────────────────────────────────────────────────
    base_dir = CACHE_DIR / "ray_params"
    if precomputed_lookup is not None:
        LUT_DIR = resolve_path(precomputed_lookup, root)
        if not LUT_DIR.is_dir():
            raise FileNotFoundError(f"precomputed_lookup directory not found: {LUT_DIR}")
        missing = [n for n in station_filenames if not (LUT_DIR / n).exists()]
        if missing:
            raise FileNotFoundError(
                f"precomputed_lookup {LUT_DIR} is missing {len(missing)} "
                f"station files, e.g. {missing[:3]}"
            )
        logger.info(f"\n[4/7] Using precomputed lookup: {LUT_DIR} "
                    f"({len(station_filenames)} files)")
        lut_dir = LUT_DIR
    else:
        lut_dir = base_dir / "sta_3d"
        if lut_dir.exists():
            shutil.rmtree(lut_dir)
        lut_dir.mkdir(parents=True, exist_ok=True)

        check_eikonal_memory(geo, dr, num_cpu_workers, logger)

        logger.info(f"\n[4/7] Computing 3-D Eikonal for {n_sta} stations "
                    f"(per-receiver grid, tomo interpolation)...")

        worker_args = [
            (tomo, geo,
             float(receiver_x_all[i]), float(receiver_y_all[i]),
             float(receiver_z_all[i]),
             dr, str(lut_dir), station_filenames[i], i)
            for i in range(n_sta)
        ]

        if num_cpu_workers > 1:
            import multiprocessing
            logger.info(f"  Parallel: {num_cpu_workers} CPU workers")
            with multiprocessing.Pool(num_cpu_workers) as pool:
                results = list(tqdm(
                    pool.imap(_compute_eikonal_worker, worker_args),
                    total=n_sta, desc="stations", unit="sta", leave=True))
        else:
            results = []
            for args in tqdm(worker_args, desc="stations", unit="sta", leave=True):
                results.append(_compute_eikonal_worker(args))

        for r in results:
            logger.log(f"    station {r['i']:5d}  {r['dt']:.1f}s")

    # ── Extract + assemble ray params ─────────────────────────────────────
    logger.info("\n[5/7] Extracting source parameters...")
    channel_ids = np.arange(n_sta, dtype=np.int64)

    def _interp_for_trial(pts_arr: np.ndarray,
                          verbose: bool = True) -> RayParamTable:
        files = [lut_dir / name for name in station_filenames]
        return interp_lookup_channels(
            files, channel_ids, pts_arr,
            forward_method="sta_3d",
            receiver_x=receiver_x_all,
            receiver_y=receiver_y_all,
            network=network_arr,
            station=station_arr,
            location=location_arr,
            dasname=dasname,
            verbose=verbose,
        )

    pts = np.column_stack([source_z, source_x, source_y])
    nominal_table = _interp_for_trial(pts)

    # ── MC perturbation ──────────────────────────────────────────────────
    tables: list[RayParamTable] = [nominal_table]
    v_unc: float | None = None
    h_unc: float | None = None
    if PERTURB_LOCATION:
        from dasfm.forward.mc_perturbation import perturb_sources
        logger.info(f"\n[6/7] Generating {NMC} perturbation trials "
                    f"(vert={VERT_UNCERT_KM} km, horz={HORZ_UNCERT_KM} km)...")
        v_unc = VERT_UNCERT_KM
        h_unc = HORZ_UNCERT_KM
        rng = np.random.default_rng(42)
        tables = []
        for imc in range(NMC):
            if imc == 0:
                sx, sy, sz = source_x.copy(), source_y.copy(), source_z.copy()
            else:
                sx, sy, sz = perturb_sources(
                    source_x, source_y, source_z,
                    VERT_UNCERT_KM, HORZ_UNCERT_KM, rng,
                )
            pts_mc = np.column_stack([sz, sx, sy])
            tables.append(_interp_for_trial(pts_mc, verbose=False))
            if (imc + 1) % 10 == 0 or imc == NMC - 1:
                logger.log(f"    trial {imc + 1}/{NMC}")

    if PERTURB_LOCATION and v_unc is not None:
        tables = [
            replace(t, perturb_vert_uncert_km=v_unc, perturb_horz_uncert_km=h_unc)
            for t in tables
        ]
    table = tables[0] if len(tables) == 1 else RayParamTable.stack_mc_trials(tables)

    # ── Save + plots ─────────────────────────────────────────────────────
    out_dir = base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_sta_3d.h5"
    logger.info(f"\n[7/7] Saving ray parameter table...")
    table.to_hdf5(out_path)
    logger.info(f"  → {out_path}")

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
