"""Multi-format readers for station polarity and S/P ratio files, plus
high-level loaders that match station observations with ray parameters.

Supported formats (auto-detected from file header):
- SKHASH CSV: CSV with ``event_id`` + ``p_polarity`` / ``sp_ratio`` columns
- QuakeML: XML with ``<quakeml`` or ``<?xml`` header
- NCSN/Hypoinverse: Fixed-width text (USGS NCSN catalog + phase)

Reading logic adapted from SKHASH (Skoumal et al., 2024):
    Skoumal, R.J., Hardebeck, J.L., & Shearer, P.M. (2024). SKHASH: A Python
    package for computing earthquake focal mechanisms. Seismological Research
    Letters, 95(4), 2519-2526. https://doi.org/10.1785/0220230329

Note: SKHASH also supports HASH driver 1/2/3/4/5 fixed-width formats, which
are legacy formats from the original HASH code (Hardebeck & Shearer, 2002).
Support for these HASH-specific formats has been intentionally dropped in
dasfm as they are rarely used with modern data workflows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
#  Pre-flight validators
# ═══════════════════════════════════════════════════════════════════════

def validate_sta_polarity(filepath) -> None:
    """Pre-flight check: station polarity file exists and is in a recognized format.

    Supports SKHASH CSV, QuakeML XML, and NCSN/Hypoinverse fixed-width formats —
    same as :func:`read_sta_polarity`.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format cannot be detected, or required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Station polarity file not found: {filepath}")
    fmt = _detect_format(filepath)
    if fmt == "skhash":
        header = set(pd.read_csv(filepath, nrows=0,
                                  skipinitialspace=True, comment="#").columns)
        required = {"event_id", "p_polarity"}
        missing = required - header
        if missing:
            raise ValueError(
                f"SKHASH polarity file {filepath} missing required columns: {sorted(missing)}"
            )
    # quakeml / ncsn formats are validated by their parsers when loaded — too
    # expensive to fully verify upfront, so existence + format detection is enough.


def validate_sta_sp_ratio(filepath) -> None:
    """Pre-flight check: station S/P ratio CSV exists and has required columns.

    Required columns: ``event_id``, ``station``, and either ``sp_ratio`` or
    raw amplitudes (``noise_p``, ``noise_s``, ``amp_p``, ``amp_s``).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Station S/P ratio file not found: {filepath}")
    header = set(pd.read_csv(filepath, nrows=0,
                              skipinitialspace=True, comment="#").columns)
    if "event_id" not in header or "station" not in header:
        raise ValueError(
            f"Station S/P ratio CSV {filepath} missing 'event_id' or 'station' column"
        )
    has_ratio = "sp_ratio" in header
    has_raw = {"noise_p", "noise_s", "amp_p", "amp_s"}.issubset(header)
    if not (has_ratio or has_raw):
        raise ValueError(
            f"Station S/P ratio CSV {filepath} must contain either 'sp_ratio' "
            f"or raw amplitude columns (noise_p, noise_s, amp_p, amp_s)"
        )


def validate_sta_geo(filepath) -> None:
    """Pre-flight check: station geometry CSV exists and has required columns.

    Required columns: ``station``, ``latitude``, ``longitude``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Station geometry CSV not found: {filepath}")
    header = set(pd.read_csv(filepath, nrows=0).columns)
    required = {"station", "latitude", "longitude"}
    missing = required - header
    if missing:
        raise KeyError(
            f"Station geometry CSV {filepath} missing required columns: {sorted(missing)}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def read_sta_polarity(filepath):
    """Read station polarity file, auto-detecting format.

    Returns
    -------
    pd.DataFrame with at least: event_id, station, p_polarity
        Optional: network, location, channel, takeoff, azimuth, sr_dist_km
    """
    filepath = Path(filepath)
    fmt = _detect_format(filepath)

    if fmt == "skhash":
        return _read_skhash_polarity(filepath)
    elif fmt == "quakeml":
        _, pol_df = _read_quakeml_polarity(filepath)
        return pol_df
    elif fmt == "ncsn":
        _, pol_df = _read_ncsn_polarity(filepath)
        return pol_df
    else:
        raise ValueError(f"Cannot auto-detect format of {filepath}")


def read_sta_sp_ratio(filepath):
    """Read station S/P ratio file, auto-detecting format.

    Returns
    -------
    pd.DataFrame with at least: event_id, station, sp_ratio
        Optional: network, location, channel
    """
    filepath = Path(filepath)
    # S/P ratio is only available in CSV format
    return _read_skhash_sp(filepath)


# ═══════════════════════════════════════════════════════════════════════
#  Format detection
# ═══════════════════════════════════════════════════════════════════════

def _detect_format(filepath):
    """Detect file format from content header.

    Returns: "skhash", "quakeml", or "ncsn"
    """
    with open(filepath, "r", errors="replace") as f:
        first_line = f.readline(1024).strip()

    # XML / QuakeML
    if first_line.startswith("<?xml") or first_line.lower().startswith("<quakeml"):
        return "quakeml"

    # CSV with header → skhash
    if "event_id" in first_line and "," in first_line:
        return "skhash"

    # Fixed-width text → NCSN/Hypoinverse
    return "ncsn"


# ═══════════════════════════════════════════════════════════════════════
#  SKHASH CSV reader
# ═══════════════════════════════════════════════════════════════════════

def _read_skhash_polarity(filepath):
    """Read SKHASH-format polarity CSV.

    Adapted from SKHASH ``in_pol.read_skhash_polarity_file``.
    Requires columns: ``event_id``, ``p_polarity``.
    """
    consider_cols = [
        "event_id", "network", "station", "location", "channel",
        "p_polarity", "takeoff", "takeoff_uncertainty",
        "azimuth", "azimuth_uncertainty", "sr_dist_km",
    ]
    df = pd.read_csv(filepath, skipinitialspace=True, comment="#",
                     usecols=lambda x: x in consider_cols)
    if not {"event_id", "p_polarity"}.issubset(df.columns):
        raise ValueError(
            f"SKHASH polarity file must contain 'event_id' and 'p_polarity': {filepath}")
    df["event_id"] = df["event_id"].astype(str)
    df["p_polarity"] = df["p_polarity"].astype(float)
    if "station" in df.columns:
        df["station"] = df["station"].astype(str).str.strip()
    return df


def _read_skhash_sp(filepath):
    """Read SKHASH-format S/P ratio CSV.

    Adapted from SKHASH ``in_sp.read_skhash_amp_file`` and ``read_amp_file``.
    Accepts either a direct ``sp_ratio`` column or raw amplitudes
    (``noise_p``, ``noise_s``, ``amp_p``, ``amp_s``) from which S/P ratio
    is computed.
    """
    consider_cols = [
        "event_id", "network", "station", "location", "channel",
        "noise_p", "noise_s", "amp_p", "amp_s", "sp_ratio",
    ]
    df = pd.read_csv(filepath, skipinitialspace=True, comment="#",
                     usecols=lambda x: x in consider_cols)
    if "event_id" not in df.columns:
        raise ValueError(f"S/P ratio file must contain 'event_id': {filepath}")

    df["event_id"] = df["event_id"].astype(str)

    if "sp_ratio" not in df.columns:
        if not {"noise_p", "noise_s", "amp_p", "amp_s"}.issubset(df.columns):
            raise ValueError(
                f"S/P file must contain 'sp_ratio' or 'noise_p/noise_s/amp_p/amp_s': {filepath}")
        df[["noise_p", "noise_s", "amp_p", "amp_s"]] = (
            df[["noise_p", "noise_s", "amp_p", "amp_s"]].abs())
        df = df[(df["amp_p"] > 0) & (df["amp_s"] > 0)].reset_index(drop=True)
        df["sp_ratio"] = (df["amp_s"] / df["amp_p"]).round(2)
        df = df.drop(columns=["noise_p", "noise_s", "amp_p", "amp_s"])

    df["sp_ratio"] = df["sp_ratio"].abs()
    df = df[df["sp_ratio"] > 0].reset_index(drop=True)
    if "station" in df.columns:
        df["station"] = df["station"].astype(str).str.strip()
    return df


# ═══════════════════════════════════════════════════════════════════════
#  NCSN / Hypoinverse reader (adapted from SKHASH)
# ═══════════════════════════════════════════════════════════════════════

def _read_ncsn_polarity(filepath, p_weight_I=1.0, p_weight_E=0.5):
    """Read USGS NCSN Hypoinverse-format polarity file.

    Adapted from SKHASH ``in_pol.read_ncsn_polarity_file``.
    Polarities can be obtained from the NCEDC
    (https://ncedc.org/ncedc/catalog-search.html) by selecting
    "NCSN catalog + Phase in Hypoinverse format".

    Polarity weights are derived from ``p_weight_code``:
        0 → 1.0, 1 → 0.5, 2 → 0.2, 3 → 0.1
    (see https://ncedc.org/pub/doc/ncsn/shadow2000.pdf)

    Returns (cat_df, pol_df).
    """
    col_spec = np.array([
        [0, 4, "year"], [4, 6, "month"], [6, 8, "day"],
        [8, 10, "hour"], [10, 12, "minute"], [12, 16, "second"],
        [16, 18, "lat_deg"], [18, 19, "latNS"], [19, 23, "lat_min"],
        [23, 26, "lon_deg"], [26, 27, "lonEW"], [27, 31, "lon_min"],
        [31, 36, "depth"],
        [85, 89, "x_error_km"], [89, 93, "z_error_km"],
        [136, 146, "event_id"], [147, 150, "mag_pref"],
    ])
    col_ind = col_spec[:, :2].astype(int)
    col_name = col_spec[:, 2]

    pick_spec = np.array([
        [0, 5, "station"], [5, 7, "network"], [9, 12, "channel"],
        [13, 14, "p_onset"], [15, 16, "p_first_motion"],
        [16, 17, "p_weight_code"], [111, 113, "location"],
    ])
    pick_ind = pick_spec[:, :2].astype(int)
    pick_name = pick_spec[:, 2]

    header_all = []
    pick_all = []
    with open(filepath, "r") as f:
        for line in f:
            if line[:3] == "   ":  # footer
                tmp_df = pd.DataFrame(tmp_pol, columns=pick_name)
                tmp_df = tmp_df.drop(
                    tmp_df.loc[tmp_df["p_first_motion"] == " "].index)
                tmp_df["event_id"] = str(event_id)
                pick_all.append(tmp_df)
            elif " " in line[:10]:  # phase line
                tmp_pol.append(
                    [line[i:j] for i, j in zip(pick_ind[:, 0], pick_ind[:, 1])])
            else:  # header
                header_all.append(
                    [line[i:j] for i, j in zip(col_ind[:, 0], col_ind[:, 1])])
                header_all[-1][-2] = header_all[-1][-2].strip()
                event_id = header_all[-1][-2]
                tmp_pol = []

    cat_df = pd.DataFrame(header_all, columns=col_name)
    cat_df = cat_df.astype({
        "lat_deg": "int32", "lat_min": "int32",
        "lon_deg": "int32", "lon_min": "int32",
        "depth": "int32", "x_error_km": "int32", "z_error_km": "int32",
        "mag_pref": "int32", "year": "int32", "month": "int32",
        "day": "int32", "hour": "int32", "minute": "int32",
        "second": "float",
    })
    cat_df["second"] = cat_df["second"] / 100
    cat_df["latitude"] = cat_df["lat_deg"] + cat_df["lat_min"] / 6000
    cat_df.loc[cat_df["latNS"] != " ", "latitude"] *= -1
    cat_df["longitude"] = cat_df["lon_deg"] + cat_df["lon_min"] / 6000
    cat_df.loc[cat_df["lonEW"] != "E", "longitude"] *= -1
    cat_df["depth"] = cat_df["depth"] / 100
    cat_df["event_id"] = cat_df["event_id"].astype(str).str.strip()

    if not pick_all:
        return cat_df, pd.DataFrame(
            columns=["event_id", "station", "network", "channel",
                     "location", "p_polarity"])

    pick_df = pd.concat(pick_all).reset_index(drop=True)
    pick_df["p_polarity"] = 0.0

    pick_df["p_weight_code"] = pick_df["p_weight_code"].astype(int)
    pick_df.loc[pick_df["p_weight_code"] == 0, "p_polarity"] = 1.0
    pick_df.loc[pick_df["p_weight_code"] == 1, "p_polarity"] = 0.5
    pick_df.loc[pick_df["p_weight_code"] == 2, "p_polarity"] = 0.2
    pick_df.loc[pick_df["p_weight_code"] == 3, "p_polarity"] = 0.1

    pick_df.loc[pick_df["p_first_motion"] == "D", "p_polarity"] *= -1
    pick_df = pick_df.drop(
        columns=["p_first_motion", "p_weight_code", "p_onset"])

    for col in ["station", "network", "channel", "location"]:
        pick_df[col] = pick_df[col].str.strip()

    return cat_df, pick_df


# ═══════════════════════════════════════════════════════════════════════
#  QuakeML reader (adapted from SKHASH)
# ═══════════════════════════════════════════════════════════════════════

def _read_quakeml_polarity(filepath, p_weight_I=1.0, p_weight_E=0.5):
    """Read QuakeML (XML) polarity file.

    Adapted from SKHASH ``in_pol.read_quakeml_polarity_file``.
    Supports standard QuakeML schema with namespace auto-detection.
    Impulsive picks receive weight ``p_weight_I`` (default 1.0),
    emergent picks receive ``p_weight_E`` (default 0.5).

    Returns (cat_df, pol_df).
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(filepath)
    root = tree.getroot()

    # Detect namespace
    ns = ""
    tag = root.tag
    if "}" in tag:
        ns = tag[: tag.index("}") + 1]

    events_data = []
    picks_data = []

    for event_el in root.iter(f"{ns}event"):
        # Event ID
        event_id = event_el.get("publicID", "")
        if "/" in event_id:
            event_id = event_id.rsplit("/", 1)[-1]
        event_id = event_id.replace("quakeml:us/", "").replace("smi:", "")

        # Origin
        origin_el = event_el.find(f"{ns}origin")
        lat = lon = depth = None
        if origin_el is not None:
            lat_el = origin_el.find(f"{ns}latitude/{ns}value")
            lon_el = origin_el.find(f"{ns}longitude/{ns}value")
            dep_el = origin_el.find(f"{ns}depth/{ns}value")
            if lat_el is not None:
                lat = float(lat_el.text)
            if lon_el is not None:
                lon = float(lon_el.text)
            if dep_el is not None:
                depth = float(dep_el.text) / 1000  # m → km

        events_data.append({
            "event_id": str(event_id),
            "latitude": lat,
            "longitude": lon,
            "depth": depth,
        })

        # Build waveformID → station mapping from picks
        pick_map = {}  # pickID → {station, network, channel, location}
        for pick_el in event_el.iter(f"{ns}pick"):
            pick_id = pick_el.get("publicID", "")
            wf = pick_el.find(f"{ns}waveformID")
            if wf is not None:
                pick_map[pick_id] = {
                    "station": wf.get("stationCode", ""),
                    "network": wf.get("networkCode", ""),
                    "channel": wf.get("channelCode", ""),
                    "location": wf.get("locationCode", ""),
                }

        # Arrivals with polarity
        for arrival_el in event_el.iter(f"{ns}arrival"):
            pick_ref = arrival_el.find(f"{ns}pickID")
            if pick_ref is None:
                continue
            pick_id = pick_ref.text
            if pick_id not in pick_map:
                continue

            phase_el = arrival_el.find(f"{ns}phase")
            if phase_el is None or phase_el.text is None:
                continue
            if not phase_el.text.upper().startswith("P"):
                continue

            # Get polarity from the pick element
            pick_el = None
            for p in event_el.iter(f"{ns}pick"):
                if p.get("publicID", "") == pick_id:
                    pick_el = p
                    break
            if pick_el is None:
                continue

            pol_el = pick_el.find(f"{ns}polarity")
            if pol_el is None or pol_el.text is None:
                continue

            pol_text = pol_el.text.lower()
            if pol_text in ("positive", "up", "compression"):
                p_pol = 1.0
            elif pol_text in ("negative", "down", "dilatation"):
                p_pol = -1.0
            else:
                continue

            # Onset weight
            onset_el = pick_el.find(f"{ns}onset")
            if onset_el is not None and onset_el.text:
                onset = onset_el.text.lower()
                if onset == "impulsive":
                    p_pol *= p_weight_I
                elif onset == "emergent":
                    p_pol *= p_weight_E

            info = pick_map[pick_id]
            picks_data.append({
                "event_id": str(event_id),
                "station": info["station"],
                "network": info["network"],
                "channel": info["channel"],
                "location": info["location"],
                "p_polarity": p_pol,
            })

    cat_df = pd.DataFrame(events_data)
    pol_df = pd.DataFrame(picks_data)
    if pol_df.empty:
        pol_df = pd.DataFrame(
            columns=["event_id", "station", "network", "channel",
                     "location", "p_polarity"])
    return cat_df, pol_df


# ═══════════════════════════════════════════════════════════════════════
#  High-level loaders (station obs + ray parameter matching)
# ═══════════════════════════════════════════════════════════════════════

def load_sta_polarity(sta_csv_path, sta_geo_path, das_event_ids,
                       all_event_ids, sta_ray_params, logger):
    """Load station polarity data with ray parameters.

    Parameters
    ----------
    sta_ray_params : RayParamTable or None
        Nominal-shape station ray-parameter table (from
        ``RayParamTable.from_hdf5('cache/ray_params/table_sta_*.h5')``).
        For MC tables, caller should pass ``table.trial(0)`` to get the
        unperturbed nominal slice.
    """
    if not sta_csv_path or not sta_csv_path.exists():
        logger.info(f"  WARN — Station polarity not found at {sta_csv_path}")
        return None

    sta_df = read_sta_polarity(sta_csv_path)
    sta_df["event_id"] = sta_df["event_id"].astype(str)
    sta_grouped = sta_df.groupby("event_id", sort=False)

    sta_geo_stations = None
    if sta_geo_path:
        sta_geo_df = pd.read_csv(sta_geo_path, dtype=str)
        sta_geo_stations = set(sta_geo_df["station"].unique())

    sta_name_to_col = {}
    npz_eid_to_row = {}
    if sta_ray_params is not None and sta_ray_params.station is not None:
        sta_name_to_col = {
            str(n): i for i, n in enumerate(sta_ray_params.station)
        }
        npz_eid_to_row = {str(eid): i for i, eid in enumerate(all_event_ids)}

    matched = [eid for eid in das_event_ids if eid in sta_grouped.groups]
    polarities, stations = [], []
    takeoffs, azimuths, distances = [], [], []
    n_dup_total = 0
    for eid in matched:
        g = sta_grouped.get_group(eid)
        if sta_geo_stations is not None:
            g = g[g["station"].isin(sta_geo_stations)]
        if len(g) == 0:
            continue
        if g["station"].duplicated().any():
            n_before = len(g)
            g = g.assign(_abs_pol=g["p_polarity"].abs())
            g = g.sort_values("_abs_pol", ascending=False).drop_duplicates(
                subset=["station"], keep="first").drop(columns=["_abs_pol"])
            n_dup_total += n_before - len(g)
        polarities.append(g["p_polarity"].values.astype(np.float64))
        names_i = g["station"].values.astype(str)
        stations.append(names_i)
        if sta_ray_params is not None and sta_name_to_col:
            ev_row = npz_eid_to_row.get(eid)
            cols = np.array([sta_name_to_col[n] for n in names_i
                             if n in sta_name_to_col])
            if ev_row is not None and len(cols) == len(names_i):
                takeoffs.append(np.degrees(sta_ray_params.takeoff[ev_row, cols]))
                azimuths.append(np.degrees(sta_ray_params.azimuth[ev_row, cols]))
                distances.append(sta_ray_params.raypath_length[ev_row, cols])
            else:
                takeoffs.append(np.full(len(g), np.nan))
                azimuths.append(np.full(len(g), np.nan))
                distances.append(np.full(len(g), np.nan))
        else:
            takeoffs.append(np.full(len(g), np.nan))
            azimuths.append(np.full(len(g), np.nan))
            distances.append(np.full(len(g), np.nan))

    src_label = "table_sta_*.h5" if sta_ray_params is not None else "none"
    logger.info(f"  Station polarity: {len(matched)} events (src: {src_label})")
    if n_dup_total > 0:
        logger.info(f"  Removed {n_dup_total} duplicate station entries")
    return {
        "event_ids": np.array(matched),
        "polarities": np.array(polarities, dtype=object),
        "stations": np.array(stations, dtype=object),
        "takeoffs": np.array(takeoffs, dtype=object),
        "azimuths": np.array(azimuths, dtype=object),
        "distances": np.array(distances, dtype=object),
        "eid_to_idx": {eid: i for i, eid in enumerate(matched)},
    }


def load_sta_sp_ratio(sta_sp_csv_path, das_event_ids, all_event_ids,
                       sta_ray_params, logger):
    """Load station S/P ratio data with ray parameters.

    Parameters
    ----------
    sta_ray_params : RayParamTable or None
        Nominal-shape station ray-parameter table. For MC tables the
        caller should pass ``table.trial(0)`` to extract the unperturbed
        nominal slice.
    """
    if not sta_sp_csv_path or not sta_sp_csv_path.exists():
        logger.info(f"  WARN — Station S/P not found at {sta_sp_csv_path}")
        return None

    sp_df = read_sta_sp_ratio(sta_sp_csv_path)
    sp_df["event_id"] = sp_df["event_id"].astype(str)
    sp_grouped = sp_df.groupby("event_id", sort=False)

    sta_name_to_col = {}
    npz_eid_to_row = {}
    if sta_ray_params is not None and sta_ray_params.station is not None:
        sta_name_to_col = {
            str(n): i for i, n in enumerate(sta_ray_params.station)
        }
        npz_eid_to_row = {str(eid): i for i, eid in enumerate(all_event_ids)}

    matched = [eid for eid in das_event_ids if eid in sp_grouped.groups]
    ratios, stations = [], []
    takeoffs, azimuths = [], []
    n_dup_total = 0
    for eid in matched:
        g = sp_grouped.get_group(eid)
        if g["station"].duplicated().any():
            n_before = len(g)
            g = g.drop_duplicates(subset=["station"], keep="first")
            n_dup_total += n_before - len(g)
        ratios.append(g["sp_ratio"].values.astype(np.float64))
        names_i = g["station"].values.astype(str)
        stations.append(names_i)
        if sta_ray_params is not None and sta_name_to_col:
            ev_row = npz_eid_to_row.get(eid)
            cols = np.array([sta_name_to_col[n] for n in names_i
                             if n in sta_name_to_col])
            if ev_row is not None and len(cols) == len(names_i):
                takeoffs.append(np.degrees(sta_ray_params.takeoff[ev_row, cols]))
                azimuths.append(np.degrees(sta_ray_params.azimuth[ev_row, cols]))
            else:
                takeoffs.append(np.full(len(g), np.nan))
                azimuths.append(np.full(len(g), np.nan))
        else:
            takeoffs.append(np.full(len(g), np.nan))
            azimuths.append(np.full(len(g), np.nan))

    src_label = "table_sta_*.h5" if sta_ray_params is not None else "none"
    logger.info(f"  Station S/P: {len(matched)} events (src: {src_label})")
    if n_dup_total > 0:
        logger.info(f"  Removed {n_dup_total} duplicate station entries")
    return {
        "event_ids": np.array(matched),
        "ratios": np.array(ratios, dtype=object),
        "stations": np.array(stations, dtype=object),
        "takeoffs": np.array(takeoffs, dtype=object),
        "azimuths": np.array(azimuths, dtype=object),
        "eid_to_idx": {eid: i for i, eid in enumerate(matched)},
    }
