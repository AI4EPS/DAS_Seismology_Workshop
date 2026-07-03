"""Pre-filter events and stations by distance to the DAS fiber."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def prefilter_by_distance(
    project_dir,
    event_catalog,
    das_geo,
    sta_geo=None,
    sta_sp_ratio=None,
    sta_polarity=None,
    max_distance_km=None,
):
    """Filter events and stations that are too far from the DAS fiber.

    Distance is defined as the minimum horizontal distance to any DAS channel.

    Parameters
    ----------
    project_dir : str or Path
        Project root directory.
    event_catalog : str or Path
        Path to event catalog CSV (relative to project_dir).
    das_geo : str or Path
        Path to DAS geometry CSV (relative to project_dir).
    sta_geo : str or Path or None
        Path to station geometry CSV. If None, station filtering is skipped.
    sta_sp_ratio : str or Path or None
        Path to station S/P ratio CSV. If provided, filtered copy is saved.
    sta_polarity : str or Path or None
        Path to station polarity CSV. If provided, filtered copy is saved.
    max_distance_km : float or None
        Maximum distance in km. If None, no filtering is performed.

    Returns
    -------
    dict with keys:
        "event_catalog" : str — path to filtered catalog CSV
        "sta_geo"       : str or None — path to filtered station CSV
        "sta_polarity"  : str or None — path to filtered polarity CSV
        "sta_sp_ratio"  : str or None — path to filtered S/P ratio CSV
        "n_events_kept" / "n_events_dropped"
        "n_stations_kept" / "n_stations_dropped"
    """
    root = Path(project_dir).resolve()

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    out_dir = root / "cache" / "pre_filt"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DAS channels
    das_df = pd.read_csv(_resolve(das_geo))
    das_lat = das_df["latitude"].values
    das_lon = das_df["longitude"].values

    # Load catalog
    cat = pd.read_csv(_resolve(event_catalog))
    n_ev_total = len(cat)

    # Load stations
    sta_df = None
    n_sta_total = 0
    if sta_geo is not None:
        sta_df = pd.read_csv(_resolve(sta_geo))
        n_sta_total = len(sta_df.drop_duplicates(subset=["network", "station"]))

    # No filtering
    if max_distance_km is None:
        result = {
            "event_catalog": str(event_catalog),
            "sta_geo": str(sta_geo) if sta_geo else None,
            "sta_polarity": str(sta_polarity) if sta_polarity else None,
            "sta_sp_ratio": str(sta_sp_ratio) if sta_sp_ratio else None,
            "n_events_kept": n_ev_total,
            "n_events_dropped": 0,
            "n_stations_kept": n_sta_total,
            "n_stations_dropped": 0,
        }
        print(f"  prefilter: max_distance_km=None, no filtering applied")
        return result

    # Cosine correction for longitude
    all_lat = np.concatenate([das_lat, cat["latitude"].values])
    cos_lat = np.cos(np.radians(all_lat.mean()))

    # ── Filter events ────────────────────────────────────────────────────
    ev_lat = cat["latitude"].values
    ev_lon = cat["longitude"].values

    ev_min_dist = _min_dist_to_fiber(ev_lat, ev_lon, das_lat, das_lon, cos_lat)
    ev_keep = ev_min_dist <= max_distance_km

    cat_filtered = cat[ev_keep].reset_index(drop=True)
    kept_event_ids = set(cat_filtered["event_id"].astype(str))
    n_ev_dropped = n_ev_total - len(cat_filtered)

    cat_out = out_dir / "catalog_filtered.csv"
    cat_filtered.to_csv(cat_out, index=False)

    print(f"  prefilter: events  {len(cat_filtered)}/{n_ev_total} kept "
          f"({n_ev_dropped} dropped beyond {max_distance_km} km)")

    # ── Filter stations ──────────────────────────────────────────────────
    sta_out = None
    n_sta_kept = n_sta_total
    n_sta_dropped = 0
    kept_stations = None

    if sta_df is not None:
        sta_unique = sta_df.drop_duplicates(subset=["network", "station"])
        sta_lat = sta_unique["latitude"].values
        sta_lon = sta_unique["longitude"].values

        sta_min_dist = _min_dist_to_fiber(sta_lat, sta_lon, das_lat, das_lon, cos_lat)
        sta_keep = sta_min_dist <= max_distance_km

        kept_stations = set(sta_unique.loc[sta_keep, "station"].astype(str))
        sta_filtered = sta_df[sta_df["station"].astype(str).isin(kept_stations)].reset_index(drop=True)
        n_sta_kept = len(sta_filtered.drop_duplicates(subset=["network", "station"]))
        n_sta_dropped = n_sta_total - n_sta_kept

        sta_out_path = out_dir / "stations_filtered.csv"
        sta_filtered.to_csv(sta_out_path, index=False)
        sta_out = str(sta_out_path.relative_to(root))

        print(f"  prefilter: stations {n_sta_kept}/{n_sta_total} kept "
              f"({n_sta_dropped} dropped beyond {max_distance_km} km)")

    # ── Filter station polarity CSV ──────────────────────────────────────
    pol_out = None
    if sta_polarity is not None:
        pol_path = _resolve(sta_polarity)
        if pol_path.exists():
            pol_df = pd.read_csv(pol_path)
            mask = pol_df["event_id"].astype(str).isin(kept_event_ids)
            if kept_stations is not None:
                mask &= pol_df["station"].astype(str).isin(kept_stations)
            pol_filtered = pol_df[mask].reset_index(drop=True)
            pol_out_path = out_dir / "polarity_filtered.csv"
            pol_filtered.to_csv(pol_out_path, index=False)
            pol_out = str(pol_out_path.relative_to(root))
            print(f"  prefilter: polarity {len(pol_filtered)}/{len(pol_df)} rows kept")

    # ── Filter station S/P ratio CSV ─────────────────────────────────────
    sp_out = None
    if sta_sp_ratio is not None:
        sp_path = _resolve(sta_sp_ratio)
        if sp_path.exists():
            sp_df = pd.read_csv(sp_path)
            mask = sp_df["event_id"].astype(str).isin(kept_event_ids)
            if kept_stations is not None:
                mask &= sp_df["station"].astype(str).isin(kept_stations)
            sp_filtered = sp_df[mask].reset_index(drop=True)
            sp_out_path = out_dir / "sp_ratio_filtered.csv"
            sp_filtered.to_csv(sp_out_path, index=False)
            sp_out = str(sp_out_path.relative_to(root))
            print(f"  prefilter: S/P ratio {len(sp_filtered)}/{len(sp_df)} rows kept")

    return {
        "event_catalog": str(cat_out.relative_to(root)),
        "sta_geo": sta_out,
        "sta_polarity": pol_out if pol_out else (str(sta_polarity) if sta_polarity else None),
        "sta_sp_ratio": sp_out if sp_out else (str(sta_sp_ratio) if sta_sp_ratio else None),
        "n_events_kept": len(cat_filtered),
        "n_events_dropped": n_ev_dropped,
        "n_stations_kept": n_sta_kept,
        "n_stations_dropped": n_sta_dropped,
    }


def _min_dist_to_fiber(lat, lon, das_lat, das_lon, cos_lat):
    """Compute minimum horizontal distance from each point to any DAS channel.

    Parameters
    ----------
    lat, lon : array (N,)
    das_lat, das_lon : array (M,)
    cos_lat : float

    Returns
    -------
    min_dist : array (N,) in km
    """
    min_dist = np.empty(len(lat))
    for i in range(len(lat)):
        dx = (lon[i] - das_lon) * 111.2 * cos_lat
        dy = (lat[i] - das_lat) * 111.2
        min_dist[i] = np.min(np.sqrt(dx**2 + dy**2))
    return min_dist
