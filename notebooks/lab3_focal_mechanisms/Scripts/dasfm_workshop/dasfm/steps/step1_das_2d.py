"""step1_das_2d — DAS ray parameters via 2D FSM radial cross-sections."""

from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from dasfm.forward.geometry import build_model_grid, compute_fiber_orientation
from dasfm.io.velocity_io import load_velocity_1d, validate_velocity_1d
from dasfm.forward.ray_lookup_2d import compute_ray_lookup
from dasfm.forward.eval_ray_lookup import interp_lookup_channels
from dasfm.forward.channel_selection import subsample_das_channels
from dasfm.utils.step_utils import Logger, resolve_path, plot_ray_param_matrices
from dasfm.io.data import RayParamDB, RayParamTable
from dasfm.io.event_catalog_io import validate_event_catalog
from dasfm.io.das_io import validate_das_geo
from dasfm.io.topo_io import validate_topo


def _channel_filename(ch_idx: int) -> str:
    return f"das_channel_{ch_idx:05d}.h5"


def compute_lookup_single_channel(args):
    """Worker: compute 2-D ray lookup for a single DAS channel and save as RayParamDB."""
    (geo, idc, vd, vv, dr, daz, rx_exact, ry_exact, lut_dir) = args
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
    db.to_hdf5(Path(lut_dir) / _channel_filename(idc))
    return {"idc": idc, "dt": _time.perf_counter() - t_start}


def _collect_lut_files(lut_dir: Path) -> tuple[list[Path], np.ndarray]:
    files = sorted(f for f in lut_dir.glob("das_channel_*.h5") if f.is_file())
    channel_ids = np.array(
        [int(f.stem.split("_")[-1]) for f in files], dtype=np.int64
    )
    return files, channel_ids


def run(
    project_dir="",
    event_catalog="",
    das_geo="",
    vel_1d="",
    topo="",
    grid_spacing_km=0.2,
    das_subsample_interval=20,
    fiber_bend_threshold_deg=5.0,
    smooth_window=20,
    azimuth_interp_deg=5.0,
    compute_uncertainty=False,
    monte_carlo_trials=30,
    vertical_uncertainty_km=1.0,
    horizontal_uncertainty_km=1.0,
    precomputed_lookup=None,
    num_cpu_workers=1,
    dasname=None,
    show_plots=False):
    """Compute DAS ray parameters via 2D FSM radial cross-sections.

    Outputs:
        cache/ray_params/das_2d/das_channel_XXXXX.h5   (Layer 1, one per
                                                        subsampled channel)
        cache/ray_params/table_das_2d.h5                (Layer 2+3)
        cache/figs/stage1_das_2d/*.png
    """
    _required = {"project_dir": project_dir, "event_catalog": event_catalog,
                 "das_geo": das_geo, "vel_1d": vel_1d, "topo": topo}
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")
    from scipy.ndimage import uniform_filter1d
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    root       = Path(project_dir).resolve()
    CACHE_DIR  = root / "cache"
    logger = Logger("step1_das_2d", log_dir=str(root / "logs"))

    dr                 = grid_spacing_km
    dc                 = das_subsample_interval
    bend_threshold_deg = fiber_bend_threshold_deg
    daz                = azimuth_interp_deg
    NMC                = monte_carlo_trials
    VERT_UNCERT_KM     = vertical_uncertainty_km
    HORZ_UNCERT_KM     = horizontal_uncertainty_km
    PERTURB_LOCATION   = compute_uncertainty

    if isinstance(vel_1d, (str, Path)):
        vel_1d_list = [vel_1d]
    else:
        vel_1d_list = list(vel_1d)
    n_vel_models = len(vel_1d_list)

    VEL_FILES = [resolve_path(v, root) for v in vel_1d_list]
    CAT_FILE  = resolve_path(event_catalog, root)
    DAS_FILE  = resolve_path(das_geo,       root)
    TOPO_FILE = resolve_path(topo,          root)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info()
    logger.info("  step1_das_2d — Compute DAS ray parameters (2D FSM)")
    logger.info()
    logger.info("=" * 60)
    logger.info("[1/7] Loading input data...  (method: 2D)")

    validate_event_catalog(CAT_FILE)
    validate_das_geo(DAS_FILE)
    validate_topo(TOPO_FILE)
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

    from dasfm.io.topo_io import load_topo
    elev_lat, elev_lon, elev = load_topo(TOPO_FILE)

    # ── 2. Build Cartesian grid ──────────────────────────────────────────
    depth_col = "depth" if "depth" in catalog.columns else "depth_km"
    depth_max = float(catalog[depth_col].max()) * 2
    logger.info(f"\n[2/7] Building Cartesian grid (dr=dz={dr} km, depth_max={depth_max:.1f} km)...")

    geo = build_model_grid(
        catalog,
        receiver_lat=rec_lat, receiver_lon=rec_lon,
        topo_lat=elev_lat, topo_lon=elev_lon, topo_elev_m=elev,
        depth_max=depth_max, dx=dr, dy=dr, dz=dr,
        logger=logger,
    )

    fig_dir = CACHE_DIR / "figs/stage1_das_2d"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    x_grid = geo["x_grid"]
    y_grid = geo["y_grid"]
    elev_m = (geo['h_max'] - geo['z_grid'][geo['surface_iz']]) * 1000
    plt.imshow(elev_m, aspect='auto', origin='lower', cmap='gist_earth',
               extent=[y_grid[0], y_grid[-1], x_grid[0], x_grid[-1]])
    plt.colorbar(label="Elevation (m)")
    plt.plot(geo['receiver_y'], geo['receiver_x'], 'r.', ms=1)
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
    n_ch_total = len(receiver_x_all)

    # ── 3. Subsample channels ────────────────────────────────────────────
    logger.info("\n[3/7] Subsampling DAS channels...")
    sample_channel = subsample_das_channels(
        receiver_x=geo['receiver_x'], receiver_y=geo['receiver_y'],
        dc=dc, bend_threshold_deg=bend_threshold_deg, smooth_window=smooth_window,
        logger=logger,
    )

    # Fiber layout QC plot
    lon_min, lon_max = rec_lon.min(), rec_lon.max()
    lat_min, lat_max = rec_lat.min(), rec_lat.max()
    pad_lon = (lon_max - lon_min) * 0.05
    pad_lat = (lat_max - lat_min) * 0.05
    cos_lat = np.cos(np.radians(0.5 * (lat_min + lat_max)))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(rec_lon, rec_lat, 'b.', ms=1, label='DAS fiber')
    ax.plot(rec_lon[sample_channel], rec_lat[sample_channel],
            'r.', ms=2, label=f'subsampled ({len(sample_channel)})')
    ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
    ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)
    ax.set_aspect(1.0 / cos_lat, adjustable='box')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('DAS fiber & subsampled channels'); ax.legend()
    y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo
    ax.set_ylim(y_lo - 0.6 * y_range, y_hi)
    dx_f = np.diff(rec_lon); dy_f = np.diff(rec_lat)
    angles = np.arctan2(dy_f, dx_f)
    dangle = np.abs(np.diff(angles))
    dangle = np.minimum(dangle, 2 * np.pi - dangle)
    dangle_smooth = uniform_filter1d(dangle, size=50)
    i_peak = np.argmax(dangle_smooth)
    center_lon, center_lat = rec_lon[i_peak], rec_lat[i_peak]
    zoom_r = 0.0075
    axins = fig.add_axes([0.12, 0.25, 0.35, 0.35])
    axins.plot(rec_lon, rec_lat, 'b-', lw=1)
    axins.plot(rec_lon[sample_channel], rec_lat[sample_channel], 'r.', ms=3)
    axins.set_xlim(center_lon - zoom_r, center_lon + zoom_r)
    axins.set_ylim(center_lat - zoom_r * cos_lat, center_lat + zoom_r * cos_lat)
    axins.set_aspect(1.0 / cos_lat, adjustable='box')
    axins.set_title('sharpest bend', fontsize=8); axins.tick_params(labelsize=7)
    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.5', lw=0.8)
    fig.savefig(fig_dir / "fiber_layout.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    if show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(fig_dir / "fiber_layout.png")))

    # 2-D radial sampling QC plot
    az_angles = np.arange(0.0, 360.0, float(daz))
    ch0 = int(sample_channel[0])
    _rx = float(receiver_x_all[ch0])
    _ry = float(receiver_y_all[ch0])
    _line_len = max(
        np.hypot(_rx - x_grid[0], _ry - y_grid[0]),
        np.hypot(_rx - x_grid[0], _ry - y_grid[-1]),
        np.hypot(_rx - x_grid[-1], _ry - y_grid[0]),
        np.hypot(_rx - x_grid[-1], _ry - y_grid[-1]),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(elev_m, aspect='auto', origin='lower', cmap='gist_earth',
              extent=[y_grid[0], y_grid[-1], x_grid[0], x_grid[-1]])
    for az_deg in az_angles:
        az_rad = np.radians(az_deg)
        end_y = _ry + _line_len * np.sin(az_rad)
        end_x = _rx + _line_len * np.cos(az_rad)
        ax.plot([_ry, end_y], [_rx, end_x],
                color='orange', lw=1.5, alpha=0.6)
    ax.plot(geo['receiver_y'], geo['receiver_x'], 'r.', ms=1)
    ax.plot(geo['source_y'], geo['source_x'], 'k*', ms=4)
    ax.plot(_ry, _rx, 'g*', ms=12, zorder=10)
    ax.set_xlim(y_grid[0], y_grid[-1])
    ax.set_ylim(x_grid[0], x_grid[-1])
    ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
    ax.set_title(f'2D FSM radial sampling (daz={daz}°, {len(az_angles)} azimuths)')
    fig.savefig(fig_dir / "radial_sampling.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    if show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(fig_dir / "radial_sampling.png")))

    # ── 4. Compute 2D ray lookups (Layer 1) ─────────────────────────────
    base_dir = CACHE_DIR / "ray_params"
    if precomputed_lookup is not None:
        LUT_DIR = resolve_path(precomputed_lookup, root)
        if not LUT_DIR.is_dir():
            raise FileNotFoundError(f"precomputed_lookup directory not found: {LUT_DIR}")
        _precomp_files, _precomp_cids = _collect_lut_files(LUT_DIR)
        if not _precomp_files:
            raise FileNotFoundError(
                f"precomputed_lookup {LUT_DIR} contains no das_channel_*.h5 files"
            )
        _cat0 = RayParamDB.from_hdf5(_precomp_files[0])
        _nz_c = len(_cat0.grid_z)
        _dr_c = float(_cat0.dz if _cat0.dz else _cat0.dr)
        if abs(_dr_c - dr) > 1e-6:
            raise ValueError(
                f"Grid spacing mismatch: precomputed lookup has dr={_dr_c}, current dr={dr}")
        logger.info(f"\n[4/7] Using precomputed lookup: {LUT_DIR} "
                     f"({len(_precomp_files)} files, nz={_nz_c}, dr={_dr_c})")
        # Check source coverage against Layer-1 radial extent
        _r_max = float(_cat0.grid_r[-1])
        for _f in _precomp_files:
            _c = RayParamDB.from_hdf5(_f)
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
            lut_dir = base_dir / ("das_2d" if n_vel_models == 1 else f"das_2d_v{v_idx}")
            if lut_dir.exists():
                shutil.rmtree(lut_dir)
            lut_dir.mkdir(parents=True, exist_ok=True)
            LUT_DIRS.append(lut_dir)

            model_label = f" (model {v_idx}/{n_vel_models-1})" if n_vel_models > 1 else ""
            logger.info(f"\n[4/7] Computing 2-D lookup tables for "
                        f"{len(sample_channel)} channels{model_label}...")

            worker_args = [
                (geo, int(idc), vd, vv, dr, daz,
                 float(receiver_x_all[idc]), float(receiver_y_all[idc]), lut_dir)
                for idc in sample_channel
            ]

            if num_cpu_workers > 1:
                import multiprocessing
                logger.info(f"  Parallel: {num_cpu_workers} CPU workers")
                with multiprocessing.Pool(num_cpu_workers) as pool:
                    results = list(tqdm(
                        pool.imap(compute_lookup_single_channel, worker_args),
                        total=len(worker_args),
                        desc=f"channels{model_label}", unit="ch", leave=True))
            else:
                results = []
                for args in tqdm(worker_args,
                                 desc=f"channels{model_label}", unit="ch", leave=True):
                    results.append(compute_lookup_single_channel(args))

            for r in results:
                logger.log(f"    channel {r['idc']:5d}  {r['dt']:.1f}s")

    # ── 5. Interpolate to all channels (per trial) ──────────────────────
    logger.info("\n[5/7] Interpolating source parameters...")

    azi_fiber_deg = compute_fiber_orientation(receiver_x_all, receiver_y_all)
    rec_azi = np.deg2rad(azi_fiber_deg).astype(np.float32)

    def _interp_for_trial(lut_dir: Path, pts_arr: np.ndarray,
                          verbose: bool = True) -> RayParamTable:
        files, channel_ids = _collect_lut_files(lut_dir)
        return interp_lookup_channels(
            files, channel_ids, pts_arr,
            forward_method="das_2d",
            receiver_x=receiver_x_all,
            receiver_y=receiver_y_all,
            n_ch_total=n_ch_total,
            rec_azi=rec_azi,
            dasname=dasname,
            verbose=verbose,
        )

    pts = np.column_stack([source_z, source_x, source_y])
    nominal_table = _interp_for_trial(LUT_DIRS[0], pts)

    # ── 6. Optional MC perturbation / multi-model trials ─────────────
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

    # ── 7. Save + summary plots ─────────────────────────────────────────
    out_dir = base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_das_2d.h5"
    logger.info(f"\n[7/7] Saving ray parameter table...")
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
