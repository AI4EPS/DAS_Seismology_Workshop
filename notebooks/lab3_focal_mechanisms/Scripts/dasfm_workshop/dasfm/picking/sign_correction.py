"""sign_correction — DAS polarity sign calibration using a reference station.

Two entry points:

* :func:`resolve_or_compute_cal` — get or auto-compute the calibration CSV
* :func:`apply_calibration`      — vote on global ±1 flip using the CSV
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dasfm.picking.das_cal import compute_das_cal_polarity


def resolve_or_compute_cal(ctx, logger) -> Path:
    """Either return user-provided sta4mccc_ref, or auto-compute from sta CSVs.

    The dispatcher has already validated that one of the two paths is
    available, so we mirror that decision tree without any fallback.
    """
    if ctx.sta4mccc_ref is not None:
        return ctx.sta4mccc_ref

    cal_df_auto = compute_das_cal_polarity(
        sta_geo_csv=ctx.sta_geo,
        pol_csv=ctx.sta_polarity,
        das_df=ctx.das_geo_df,
        min_picks=ctx.cal_min_picks,
        logger=logger,
    )
    ctx.cal_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cal_df_auto.to_csv(ctx.cal_csv_path, index=False)
    logger.info(f"  DAS cal polarity auto-computed: {len(cal_df_auto)} picks "
                f"-> {ctx.cal_csv_path}")
    return ctx.cal_csv_path


def apply_calibration(Pkic, cal_path, ctx, logger):
    """Vote on global ±1 flip using the calibration CSV.

    Returns ``(Pkic, agree, disagree)``.  Raises ``RuntimeError`` if no
    calibration picks match (a hard failure — user must fix data).
    """
    if not cal_path.exists():
        return Pkic, 0, 0

    idx_torow = {int(idx): row
                 for row, idx in enumerate(ctx.das_geo_df["index"].values)}
    ev_to_col = {eid: i for i, eid in enumerate(ctx.event_ids)}

    cal_df = pd.read_csv(cal_path)
    cal_df["event_id"] = cal_df["event_id"].astype(str)

    agree = disagree = skipped = 0
    if not cal_df.empty:
        logger.info(
            f"  Cal station : {cal_df['network'].iloc[0]}.{cal_df['station'].iloc[0]}  "
            f"(DAS ch={cal_df['closest_das_ch'].iloc[0]},  {len(cal_df)} picks)"
        )
        for _, row in cal_df.iterrows():
            ev_col = ev_to_col.get(str(row["event_id"]))
            chrow  = idx_torow.get(int(row["closest_das_ch"]))
            if ev_col is None or chrow is None:
                skipped += 1
                continue
            if np.sign(Pkic[chrow, ev_col]) == int(row["p_polarity"]):
                agree += 1
            else:
                disagree += 1
        if skipped:
            logger.info(f"  ({skipped} cal picks skipped)")

    if agree + disagree == 0:
        raise RuntimeError("No matching calibration picks — check that the "
                         "calibration CSV event IDs and channel indices "
                         "overlap with the current dataset.")
    if disagree > agree:
        Pkic = Pkic * -1
        logger.info(f"  global flip applied  (agree={agree}, disagree={disagree})")
    else:
        logger.info(f"  sign correct, no flip  (agree={agree}, disagree={disagree})")
    return Pkic, agree, disagree
