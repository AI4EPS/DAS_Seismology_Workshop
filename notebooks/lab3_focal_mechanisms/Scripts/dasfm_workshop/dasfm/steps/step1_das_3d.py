"""step1_das_3d — DAS ray parameters via 3D Eikonal (pykonal).

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

from dasfm.forward.geometry import build_model_grid, compute_fiber_orientation
from dasfm.forward.eikonal_3d import (
    load_velocity_3d, validate_velocity_3d,
    build_receiver_grid, interpolate_tomo_to_grid,
    compute_eikonal_lookup, check_eikonal_memory,
)
from dasfm.forward.eval_ray_lookup import interp_lookup_channels
from dasfm.forward.channel_selection import subsample_das_channels
from dasfm.utils.step_utils import Logger, resolve_path, plot_ray_param_matrices
from dasfm.io.data import RayParamDB, RayParamTable
from dasfm.io.event_catalog_io import validate_event_catalog
from dasfm.io.das_io import validate_das_geo
from dasfm.io.topo_io import validate_topo


def _channel_filename(ch_idx: int) -> str:
    return f"das_channel_{ch_idx:05d}.h5"


def _compute_eikonal_worker(args):
    """Worker for parallel 3D Eikonal computation (single channel)."""
    import time as _t
    from dasfm.forward.eikonal_3d import (
        build_receiver_grid, interpolate_tomo_to_grid, compute_eikonal_lookup,
    )
    (tomo, geo, rx, ry, rz, dr, lut_dir_str, idc) = args
    t0 = _t.perf_counter()

    x_g, y_g, z_g, rix, riy, riz, nx, ny, nz = build_receiver_grid(
        geo, rx, ry, rz, dr)
    vp_local = interpolate_tomo_to_grid(tomo, geo, x_g, y_g, z_g)
    db, _timing_str = compute_eikonal_lookup(
        vp_local, nx, ny, nz, dr,
        rix, riy, riz,
        x_g, y_g, z_g, geo)
    db.to_hdf5(Path(lut_dir_str) / _channel_filename(idc))
    return {"idc": idc, "dt": _t.perf_counter() - t0}


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
    vel_3d="",
    topo="",
    grid_spacing_km=0.2,
    das_subsample_interval=20,
    fiber_bend_threshold_deg=5.0,
    smooth_window=20,
    compute_uncertainty=False,
    monte_carlo_trials=30,
    vertical_uncertainty_km=1.0,
    horizontal_uncertainty_km=1.0,
    precomputed_lookup=None,
    num_cpu_workers=1,
    dasname=None,
    show_plots=False):
    """Compute DAS ray parameters via 3D Eikonal (pykonal).

    Outputs:
        cache/ray_params/das_3d/das_channel_XXXXX.h5   (Layer 1)
        cache/ray_params/table_das_3d.h5                (Layer 2+3)
        cache/figs/stage1_das_3d/*.png
    """
    _required = {"project_dir": project_dir, "event_catalog": event_catalog,
                 "das_geo": das_geo, "vel_3d": vel_3d, "topo": topo}
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Required parameters missing: {', '.join(_missing)}")

    root       = Path(project_dir).resolve()
    CACHE_DIR  = root / "cache"
    logger = Logger("step1_das_3d", log_dir=str(root / "logs"))

    dr                 = grid_spacing_km
    dc                 = das_subsample_interval
    bend_threshold_deg = fiber_bend_threshold_deg
    NMC                = monte_carlo_trials
    VERT_UNCERT_KM     = vertical_uncertainty_km
    HORZ_UNCERT_KM     = horizontal_uncertainty_km
    PERTURB_LOCATION   = compute_uncertainty

    CAT_FILE    = resolve_path(event_catalog, root)
    DAS_FILE    = resolve_path(das_geo, root)
    TOPO_FILE   = resolve_path(topo, root)
    VEL_3D_FILE = resolve_path(vel_3d, root)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info()
    logger.info("  step1_das_3d — Compute DAS ray parameters (3D Eikonal)")
    logger.info()
    logger.info("=" * 60)
    logger.info("[1/8] Loading input data...  (method: DAS 3D Eikonal)")

    validate_event_catalog(CAT_FILE)
    validate_das_geo(DAS_FILE)
    validate_topo(TOPO_FILE)
    validate_velocity_3d(VEL_3D_FILE)

    catalog  = pd.read_csv(CAT_FILE)
    das_info = pd.read_csv(DAS_FILE)
    rec_lat = das_info["latitude"].values
    rec_lon = das_info["longitude"].values

    logger.info(f"  Events        : {len(catalog)}")
    logger.info(f"  DAS receivers : {len(das_info)}")

    # ── Load 3D tomo velocity model ──────────────────────────────────────
    logger.info("\n[2/8] Loading 3D tomographic velocity model...")
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
    logger.info(f"\n[3/8] Building Cartesian grid (dr=dz={dr} km, depth_max={depth_max:.1f} km)...")

    geo = build_model_grid(
        catalog,
        receiver_lat=rec_lat, receiver_lon=rec_lon,
        topo_lat=elev_lat, topo_lon=elev_lon, topo_elev_m=elev,
        depth_max=depth_max, dx=dr, dy=dr, dz=dr,
        logger=logger,
    )

    fig_dir = CACHE_DIR / "figs/stage1_das_3d"
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
    n_ch_total = len(receiver_x_all)

    # ── Subsample DAS channels ───────────────────────────────────────────
    logger.info("\n[4/8] Subsampling DAS channels...")
    sample_channel = subsample_das_channels(
        receiver_x=geo['receiver_x'],
        receiver_y=geo['receiver_y'],
        dc=dc, bend_threshold_deg=bend_threshold_deg, smooth_window=smooth_window,
        logger=logger,
    )

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

    dx_f = np.diff(rec_lon)
    dy_f = np.diff(rec_lat)
    angles = np.arctan2(dy_f, dx_f)
    dangle = np.abs(np.diff(angles))
    dangle = np.minimum(dangle, 2 * np.pi - dangle)
    from scipy.ndimage import uniform_filter1d as uf1d
    dangle_smooth = uf1d(dangle, size=50)
    i_peak = np.argmax(dangle_smooth)
    center_lon, center_lat = rec_lon[i_peak], rec_lat[i_peak]
    zoom_r = 0.0075

    axins = fig.add_axes([0.12, 0.18, 0.35, 0.35])
    axins.plot(rec_lon, rec_lat, 'b-', lw=1)
    axins.plot(rec_lon[sample_channel], rec_lat[sample_channel], 'r.', ms=3)
    axins.set_xlim(center_lon - zoom_r, center_lon + zoom_r)
    axins.set_ylim(center_lat - zoom_r * cos_lat, center_lat + zoom_r * cos_lat)
    axins.set_aspect(1.0 / cos_lat, adjustable='box')
    axins.set_title('sharpest bend', fontsize=8)
    axins.tick_params(labelsize=7)
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.5', lw=0.8)
    fig.savefig(fig_dir / "fiber_layout.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    if show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(fig_dir / "fiber_layout.png")))

    # ── 3D Eikonal solve ─────────────────────────────────────────────────
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
        logger.info(f"\n[5/8] Using precomputed lookup: {LUT_DIR} "
                    f"({len(_precomp_files)} files)")
        lut_dir = LUT_DIR
    else:
        lut_dir = base_dir / "das_3d"
        if lut_dir.exists():
            shutil.rmtree(lut_dir)
        lut_dir.mkdir(parents=True, exist_ok=True)

        check_eikonal_memory(geo, dr, num_cpu_workers, logger)

        logger.info(f"\n[5/8] Computing 3-D Eikonal for {len(sample_channel)} channels "
                    f"(per-receiver grid, tomo interpolation)...")

        worker_args = [
            (tomo, geo,
             float(receiver_x_all[idc]), float(receiver_y_all[idc]),
             float(receiver_z_all[idc]),
             dr, str(lut_dir), int(idc))
            for idc in sample_channel
        ]

        if num_cpu_workers > 1:
            import multiprocessing
            logger.info(f"  Parallel: {num_cpu_workers} CPU workers")
            with multiprocessing.Pool(num_cpu_workers) as pool:
                results = list(tqdm(
                    pool.imap(_compute_eikonal_worker, worker_args),
                    total=len(worker_args), desc="channels", unit="ch", leave=True))
        else:
            results = []
            for args in tqdm(worker_args, desc="channels", unit="ch", leave=True):
                results.append(_compute_eikonal_worker(args))

        for r in results:
            logger.log(f"    channel {r['idc']:5d}  {r['dt']:.1f}s")

    # ── Interpolate + assemble ray params ──────────────────────────────
    logger.info("\n[6/8] Interpolating source parameters...")

    azi_fiber_deg = compute_fiber_orientation(receiver_x_all, receiver_y_all)
    rec_azi = np.deg2rad(azi_fiber_deg).astype(np.float32)

    def _interp_for_trial(pts_arr: np.ndarray,
                          verbose: bool = True) -> RayParamTable:
        files, channel_ids = _collect_lut_files(lut_dir)
        return interp_lookup_channels(
            files, channel_ids, pts_arr,
            forward_method="das_3d",
            receiver_x=receiver_x_all,
            receiver_y=receiver_y_all,
            n_ch_total=n_ch_total,
            rec_azi=rec_azi,
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
        logger.info(f"\n[7/8] Generating {NMC} perturbation trials "
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
    out_path = out_dir / "table_das_3d.h5"
    logger.info(f"\n[8/8] Saving ray parameter table...")
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
