"""dasfm/utils/das_cal.py — DAS calibration polarity computation.

Computes the nearest DAS channel for each conventional station and joins
with station polarity picks, producing a das_cal_polarity DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from dasfm.utils.step_utils import log_or_print


def compute_das_cal_polarity(
    sta_geo_csv: str | Path,
    pol_csv: str | Path,
    das_df: pd.DataFrame,
    min_picks: int = 5,
    logger=None,
) -> pd.DataFrame:
    """Compute DAS calibration polarity from station + DAS geometry.

    Selects the single nearest station (by distance to the DAS fiber) that
    has at least *min_picks* polarity picks, and returns all of that station's
    picks joined with its closest DAS channel index.

    Parameters
    ----------
    sta_geo_csv:
        Path to station geometry CSV (SKHASH stfile).
        Required columns: network, station, latitude, longitude.
    pol_csv:
        Path to station polarity CSV (SKHASH fpfile).
        Required columns: event_id, network, station, p_polarity.
    das_df:
        DataFrame with DAS channel geometry.
        Required columns: index (int), latitude (float), longitude (float).
    min_picks:
        Minimum number of polarity picks a station must have to be considered.
        Default 5.

    Returns
    -------
    pd.DataFrame with columns:
        event_id, network, station, location, channel,
        p_polarity (+1/-1), closest_das_ch (int),
        latitude (float), longitude (float)
    """
    _sta_geo = pd.read_csv(sta_geo_csv)
    _pol     = pd.read_csv(pol_csv)

    _das_lat = das_df["latitude"].values
    _das_lon = das_df["longitude"].values
    _das_idx = das_df["index"].values

    # Polarity count per (network, station)
    _pol_count_key: dict[tuple[str, str], int] = {
        (str(net), str(sta)): int(cnt)
        for (net, sta), cnt in _pol.groupby(["network", "station"]).size().items()
    }

    # Build (network, station) → closest DAS channel index + distance
    _sta_to_das: dict[tuple[str, str], int] = {}
    _sta_to_das_dist: dict[tuple[str, str], float] = {}
    for _, _sr in _sta_geo.iterrows():
        _dlat = _das_lat - float(_sr["latitude"])
        _dlon = (_das_lon - float(_sr["longitude"])) * np.cos(
            np.radians(float(_sr["latitude"]))
        )
        _dists_deg = np.hypot(_dlat, _dlon)
        _imin = int(np.argmin(_dists_deg))
        _closest = int(_das_idx[_imin])
        _dist_km = float(_dists_deg[_imin]) * 111.2
        _key = (str(_sr["network"]), str(_sr["station"]))
        _sta_to_das[_key] = _closest
        _sta_to_das_dist[_key] = _dist_km

    # Sort all stations by distance; mark which qualify (>= min_picks)
    _sorted_keys = sorted(_sta_to_das, key=lambda k: _sta_to_das_dist[k])
    _candidates = [
        k for k in _sorted_keys
        if _pol_count_key.get(k, 0) >= min_picks
    ]

    # Print summary table sorted by distance
    log_or_print(logger, f"\n[das_cal] Station → nearest DAS channel ({len(_sta_to_das)} stations,"
          f" min_picks={min_picks}):")
    log_or_print(logger, f"  {'Network':<8} {'Station':<8} {'DAS ch':>7} {'Dist(km)':>10}"
          f" {'Polarities':>12} {'Selected':>10}")
    log_or_print(logger, f"  {'-'*8} {'-'*8} {'-'*7} {'-'*10} {'-'*12} {'-'*10}")
    _selected = _candidates[0] if _candidates else None
    for _key in _sorted_keys:
        _net, _sta = _key
        _npol = _pol_count_key.get(_key, 0)
        _mark = "<-- selected" if _key == _selected else ""
        log_or_print(logger, f"  {_net:<8} {_sta:<8} {_sta_to_das[_key]:>7}"
              f" {_sta_to_das_dist[_key]:>10.3f} {_npol:>12}  {_mark}")

    _empty = pd.DataFrame(columns=[
        "event_id", "network", "station", "location",
        "channel", "p_polarity", "closest_das_ch",
        "latitude", "longitude",
    ])
    if _selected is None:
        log_or_print(logger, f"  WARN: no station with >= {min_picks} picks — returning empty DataFrame")
        return _empty

    log_or_print(logger, f"\n  Selected: {_selected[0]}.{_selected[1]}"
          f"  dist={_sta_to_das_dist[_selected]:.3f} km"
          f"  picks={_pol_count_key[_selected]}")

    # Look up station coordinates
    _sel_row = _sta_geo[
        (_sta_geo["network"].astype(str) == _selected[0]) &
        (_sta_geo["station"].astype(str) == _selected[1])
    ].iloc[0]
    _sel_lat = float(_sel_row["latitude"])
    _sel_lon = float(_sel_row["longitude"])

    # Return only picks from the selected station
    _rows = []
    for _, _pr in _pol.iterrows():
        _key = (str(_pr["network"]), str(_pr["station"]))
        if _key != _selected:
            continue
        _rows.append({
            "event_id":       str(_pr["event_id"]),
            "network":        _key[0],
            "station":        _key[1],
            "location":       _pr.get("location", ""),
            "channel":        _pr.get("channel", ""),
            "p_polarity":     int(_pr["p_polarity"]),
            "closest_das_ch": _sta_to_das[_key],
            "latitude":       _sel_lat,
            "longitude":      _sel_lon,
        })

    return pd.DataFrame(_rows) if _rows else _empty
