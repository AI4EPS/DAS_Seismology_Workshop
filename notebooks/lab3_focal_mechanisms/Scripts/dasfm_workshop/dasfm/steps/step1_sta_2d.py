"""step1_sta_2d — STA ray parameters via 2D FSM radial cross-sections."""

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
from dasfm.io.velocity_io import load_velocity_1d, validate_velocity_1d
from dasfm.forward.ray_lookup_2d import compute_ray_lookup
from dasfm.forward.eval_ray_lookup import interp_lookup_channels
from dasfm.utils.step_utils import Logger, resolve_path, plot_ray_param_matrices
from dasfm.io.data import RayParamDB, RayParamTable
from dasfm.io.event_catalog_io import validate_event_catalog
from dasfm.io.sta_io import validate_sta_geo
from dasfm.io.topo_io import validate_topo


from dasfm.utils.step_utils import station_filename as _station_filename
from dasfm.utils.step_utils import clean_location_array as _clean_location_array


def compute_lookup_single_station(args):
    """Worker: compute 2-D ray lookup for a single station and save as RayParamDB."""
    (geo, idc, vd, vv, dr, daz, rx_exact, ry_exact, lut_dir, filename) = args
    from dasfm.forward.ray_lookup_2d import compute_ray_lookup as _cr
    import time as _time
    t_start = _time.perf_counter()
    db = _cr(
        geo=geo,
        receiver_ix=geo['receiver_ix'][idc],
        receiver_iy=geo['receiver_iy'][idc],
        receiver_iz=geo['receiver_iz'][idc],
        vel_1d_depth=vd, vel_1d_vp=vv, dr=dr, daz=daz,
        receiver_x_exact=rx_exact, receiver_y_exact=ry_exact,
    )
    db.to_hdf5(Path(lut_dir) / filename)
    return {"idc": idc, "dt": _time.perf_counter() - t_start}


def _collect_lut_files(lut_dir: Path, filenames: list[str]) -> list[Path]:
    """Return full paths to station Layer-1 files in the declared order."""
    return [lut_dir / name for name in filenames]


def run(
    project_dir="",
    event_catalog="",
    sta_geo="",
    vel_1d="",
    topo="",
    grid_spacing_km=0.2,
    azimuth_interp_deg=5.0,
    compute_uncertainty=False,
    monte_carlo_trials=30,
    vertical_uncertainty_km=1.0,
    horizontal_uncertainty_km=1.0,
    precomputed_lookup=None,
    num_cpu_workers=1,
    dasname=None,
    show_plots=False):
    """Compute STA ray parameters via 2D FSM radial cross-sections.

    Parameters
    ----------
    precomputed_lookup : str or None
        Path to a directory containing pre-computed station Layer-1 files
        (``{network}_{station}_{location}.h5``). If provided, forward
        computation is skipped and only interpolation is performed.

    Outputs:
        cache/ray_params/sta_2d/{network}_{station}_{location}.h5   (Layer 1)
        cache/ray_params/table_sta_2d.h5                              (Layer 2+3)
        cache/figs/stage1_sta_2d/*.png
    """
    _required = {"project_dir": project_dir, "event_catalog": event_catalog,
                 "sta_geo": sta_geo, "vel_1d": vel_1d, "topo": topo}
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")

    root       = Path(project_dir).resolve()
    CACHE_DIR  = root / "cache"
    logger = Logger("step1_sta_2d", log_dir=str(root / "logs"))

    dr             = grid_spacing_km
    daz            = azimuth_interp_deg
    NMC            = monte_carlo_trials
    VERT_UNCERT_KM = vertical_uncertainty_km
    HORZ_UNCERT_KM = horizontal_uncertainty_km
    PERTURB_LOCATION = compute_uncertainty

    if isinstance(vel_1d, (str, Path)):
        vel_1d_list = [vel_1d]
    else:
        vel_1d_list = list(vel_1d)
    n_vel_models = len(vel_1d_list)

    VEL_FILES = [resolve_path(v, root) for v in vel_1d_list]
    CAT_FILE  = resolve_path(event_catalog, root)
    STA_FILE  = resolve_path(sta_geo,       root)
    TOPO_FILE = resolve_path(topo,          root)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info()
    logger.info("  step1_sta_2d — Compute STA ray parameters (2D FSM)")
    logger.info()
    logger.info("=" * 60)
    logger.info("[1/6] Loading input data...  (method: STA 2D)")

    validate_event_catalog(CAT_FILE)
    validate_sta_geo(STA_FILE)
    validate_topo(TOPO_FILE)
    for f in VEL_FILES:
        validate_velocity_1d(f)

    catalog  = pd.read_csv(CAT_FILE)
    vel_models = [load_velocity_1d(f) for f in VEL_FILES]
    vel_depth, vel_vp = vel_models[0]
    if n_vel_models > 1:
        logger.info(f"  Velocity models: {n_vel_models} (multi-model uncertainty)")
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

    # Pre-compute deterministic Layer-1 filenames for all stations
    station_filenames = [
        _station_filename(network_arr[i], station_arr[i], location_arr[i])
        for i in range(n_sta)
    ]

    from dasfm.io.topo_io import load_topo
    elev_lat, elev_lon, elev = load_topo(TOPO_FILE)

    # ── 2. Build Cartesian grid ──────────────────────────────────────────
    depth_col = "depth" if "depth" in catalog.columns else "depth_km"
    depth_max = float(catalog[depth_col].max()) * 2
    logger.info(f"\n[2/6] Building Cartesian grid (dr=dz={dr} km, depth_max={depth_max:.1f} km)...")

    geo = build_model_grid(
        catalog,
        receiver_lat=rec_lat, receiver_lon=rec_lon,
        topo_lat=elev_lat, topo_lon=elev_lon, topo_elev_m=elev,
        depth_max=depth_max, dx=dr, dy=dr, dz=dr,
        logger=logger,
    )

    fig_dir = CACHE_DIR / "figs/stage1_sta_2d"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    x_grid = geo["x_grid"]
    y_grid = geo["y_grid"]
    elev_m = (geo['h_max'] - geo['z_grid'][geo['surface_iz']]) * 1000
    plt.imshow(elev_m, aspect='auto', origin='lower', cmap='gist_earth',
               extent=[y_grid[0], y_grid[-1], x_grid[0], x_grid[-1]])
    plt.colorbar(label="Elevation (m)")
    plt.plot(geo['receiver_y'], geo['receiver_x'], 'r.', ms=4)
    plt.plot(geo['source_y'], geo['source_x'], 'k*', ms=4)
    plt.xlabel("Easting (km)"); plt.ylabel("Northing (km)")
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

    # 2D radial sampling QC plot
    az_angles = np.arange(0.0, 360.0, float(daz))
    _rx_ix, _rx_iy = geo['receiver_ix'][0], geo['receiver_iy'][0]
    _line_len = max(
        np.hypot(_rx_ix, _rx_iy),
        np.hypot(_rx_ix, geo['ny'] - 1 - _rx_iy),
        np.hypot(geo['nx'] - 1 - _rx_ix, _rx_iy),
        np.hypot(geo['nx'] - 1 - _rx_ix, geo['ny'] - 1 - _rx_iy),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(elev_m, aspect='auto', origin='lower', cmap='gist_earth')
    for az_deg in az_angles:
        az_rad = np.radians(az_deg)
        end_iy = _rx_iy + _line_len * np.sin(az_rad)
        end_ix = _rx_ix + _line_len * np.cos(az_rad)
        ax.plot([_rx_iy, end_iy], [_rx_ix, end_ix],
                color='orange', lw=1.5, alpha=0.6)
    ax.plot(geo['receiver_iy'], geo['receiver_ix'], 'r.', ms=4)
    ax.plot(geo['source_iy'], geo['source_ix'], 'k*', ms=4)
    ax.plot(geo['receiver_iy'][0], geo['receiver_ix'][0], 'g*', ms=12, zorder=10)
    ax.set_xlim(0, geo['ny'] - 1)
    ax.set_ylim(0, geo['nx'] - 1)
    ax.set_title(f'2D FSM radial sampling (daz={daz}°, {len(az_angles)} azimuths)')
    fig.savefig(fig_dir / "radial_sampling.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    if show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(fig_dir / "radial_sampling.png")))

    # ── 3. Compute 2D ray lookups (Layer 1) ────────────────────────────
    base_dir = CACHE_DIR / "ray_params"
    if precomputed_lookup is not None:
        LUT_DIR = resolve_path(precomputed_lookup, root)
        if not LUT_DIR.is_dir():
            raise FileNotFoundError(f"precomputed_lookup directory not found: {LUT_DIR}")
        missing = [n for n in station_filenames if not (LUT_DIR / n).exists()]
        if missing:
            raise FileNotFoundError(
                f"precomputed_lookup {LUT_DIR} is missing {len(missing)} station files, "
                f"e.g. {missing[:3]}"
            )
        _cat0 = RayParamDB.from_hdf5(LUT_DIR / station_filenames[0])
        _nz_c = len(_cat0.grid_z)
        _dr_c = float(_cat0.dz if _cat0.dz else _cat0.dr)
        if abs(_dr_c - dr) > 1e-6:
            raise ValueError(
                f"Grid spacing mismatch: precomputed lookup has dr={_dr_c}, current dr={dr}")
        logger.info(f"\n[3/6] Using precomputed lookup: {LUT_DIR} "
                     f"({len(station_filenames)} files, nz={_nz_c}, dr={_dr_c})")
        _r_max = float(_cat0.grid_r[-1])
        for name in station_filenames:
            _c = RayParamDB.from_hdf5(LUT_DIR / name)
            _dist = np.hypot(source_x - _c.receiver_x, source_y - _c.receiver_y)
            if _dist.max() > _r_max:
                raise ValueError(
                    f"Source at distance {_dist.max():.1f} km exceeds lookup r_max={_r_max:.1f} km "
                    f"(receiver at ({_c.receiver_x:.1f}, {_c.receiver_y:.1f})). "
                    f"Re-run step1 without precomputed_lookup to generate new lookups.")
        LUT_DIRS = [LUT_DIR]
    else:
        LUT_DIRS = []
        for v_idx, (vd, vv) in enumerate(vel_models):
            lut_dir = base_dir / ("sta_2d" if n_vel_models == 1 else f"sta_2d_v{v_idx}")
            if lut_dir.exists():
                shutil.rmtree(lut_dir)
            lut_dir.mkdir(parents=True, exist_ok=True)
            LUT_DIRS.append(lut_dir)

            model_label = f" (model {v_idx}/{n_vel_models-1})" if n_vel_models > 1 else ""
            logger.info(f"\n[3/6] Computing 2-D lookup tables for "
                        f"{n_sta} stations{model_label}...")

            worker_args = [
                (geo, idc, vd, vv, dr, daz,
                 float(receiver_x_all[idc]), float(receiver_y_all[idc]),
                 lut_dir, station_filenames[idc])
                for idc in range(n_sta)
            ]

            if num_cpu_workers > 1:
                import multiprocessing
                logger.info(f"  Parallel: {num_cpu_workers} CPU workers")
                with multiprocessing.Pool(num_cpu_workers) as pool:
                    results = list(tqdm(
                        pool.imap(compute_lookup_single_station, worker_args),
                        total=len(worker_args),
                        desc=f"stations{model_label}", unit="sta", leave=True))
            else:
                results = []
                for args in tqdm(worker_args,
                                 desc=f"stations{model_label}", unit="sta", leave=True):
                    results.append(compute_lookup_single_station(args))

            for r in results:
                logger.log(f"    station {r['idc']:5d}  {r['dt']:.1f}s")

    # ── 4-5. Extract source parameters and assemble RayParamTable ───────
    logger.info("\n[4/6] Extracting source parameters...")
    channel_ids = np.arange(n_sta, dtype=np.int64)
    pts = np.column_stack([source_z, source_x, source_y])

    def _interp_for_trial(lut_dir: Path, pts_arr: np.ndarray,
                          verbose: bool = True) -> RayParamTable:
        files = _collect_lut_files(lut_dir, station_filenames)
        return interp_lookup_channels(
            files, channel_ids, pts_arr,
            forward_method="sta_2d",
            receiver_x=receiver_x_all,
            receiver_y=receiver_y_all,
            network=network_arr,
            station=station_arr,
            location=location_arr,
            dasname=dasname,
            verbose=verbose,
        )

    nominal_table = _interp_for_trial(LUT_DIRS[0], pts)

    # ── 6. Optional MC perturbation / multi-model trials ──────────────
    tables: list[RayParamTable] = [nominal_table]
    v_unc: float | None = None
    h_unc: float | None = None

    if PERTURB_LOCATION:
        from dasfm.forward.mc_perturbation import perturb_sources
        logger.info(f"\n  Generating {NMC} perturbation trials "
                    f"(vert={VERT_UNCERT_KM} km, horz={HORZ_UNCERT_KM} km"
                    f", vel_models={n_vel_models})...")
        v_unc = VERT_UNCERT_KM
        h_unc = HORZ_UNCERT_KM
        rng = np.random.default_rng(42)
        tables = []
        for imc in range(NMC):
            v_idx = 0 if imc == 0 else int(rng.integers(0, n_vel_models))
            lut_dir_v = LUT_DIRS[min(v_idx, len(LUT_DIRS) - 1)]
            if imc == 0:
                sx, sy, sz = source_x.copy(), source_y.copy(), source_z.copy()
            else:
                sx, sy, sz = perturb_sources(
                    source_x, source_y, source_z,
                    VERT_UNCERT_KM, HORZ_UNCERT_KM, rng,
                )
            pts_mc = np.column_stack([sz, sx, sy])
            tables.append(_interp_for_trial(lut_dir_v, pts_mc, verbose=False))
            if (imc + 1) % 10 == 0 or imc == NMC - 1:
                v_label = f" [v{v_idx}]" if n_vel_models > 1 else ""
                logger.log(f"    trial {imc + 1}/{NMC}{v_label}")
    elif n_vel_models > 1:
        logger.info(f"\n  Generating {n_vel_models} velocity-model trials "
                    f"(no location perturbation)...")
        tables = []
        for v_idx in range(n_vel_models):
            tables.append(_interp_for_trial(LUT_DIRS[v_idx], pts, verbose=False))
            logger.log(f"    trial {v_idx + 1}/{n_vel_models} [v{v_idx}]")

    if PERTURB_LOCATION and v_unc is not None:
        tables = [
            replace(t, perturb_vert_uncert_km=v_unc, perturb_horz_uncert_km=h_unc)
            for t in tables
        ]
    table = tables[0] if len(tables) == 1 else RayParamTable.stack_mc_trials(tables)

    # ── Save + summary plots ────────────────────────────────────────────
    out_dir = base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_sta_2d.h5"
    logger.info(f"\n[6/6] Saving ray parameter table...")
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
