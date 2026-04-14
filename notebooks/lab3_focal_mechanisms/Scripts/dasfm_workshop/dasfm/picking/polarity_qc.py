"""polarity_qc — QC plots for step2b polarity results.

Four plots:
    - DAS fiber + calibration station map
    - Polarity matrix (raw + sign)
    - Polarity along fiber (first event)
    - Calibration vote bar chart
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_polarity_qc(ctx, Pkic, cal_path, agree, disagree):
    """Generate all 4 QC plots.  Called from postprocess after SVD + calibration.

    Parameters
    ----------
    ctx : Step2bContext
        Must have: fig_dir, das_geo_df, sta_geo, event_ids, show_plots.
    Pkic : np.ndarray (n_ch, n_ev)
        Calibrated polarity matrix.
    cal_path : Path
        Calibration CSV path (auto-computed or user-provided).
    agree, disagree : int
        Calibration vote counts.
    """
    ctx.fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_das_station_map(ctx, cal_path)
    _plot_polarity_matrix(ctx, Pkic)
    _plot_polarity_along_fiber(ctx, Pkic, ctx.event_ids)
    _plot_calibration_vote(ctx, agree, disagree)


def _plot_polarity_matrix(ctx, Pkic):
    import matplotlib
    import matplotlib.pyplot as plt

    n_ch, n_ev = Pkic.shape
    fig_size = (14, 5)
    vmax = np.nanpercentile(np.abs(Pkic), 95) or 1.0

    # Plot 1: raw polarity values
    fig1, ax1 = plt.subplots(figsize=fig_size, dpi=150)
    im1 = ax1.imshow(Pkic, aspect="auto", cmap="seismic",
                     vmin=-vmax, vmax=vmax, origin="lower")
    ax1.set_xlabel("Event index"); ax1.set_ylabel("Channel index")
    ax1.set_title(f"Polarity  ({n_ch} ch × {n_ev} ev)")
    fig1.colorbar(im1, ax=ax1, label="Polarity amplitude")
    fig1.tight_layout()
    out1 = ctx.fig_dir / "polarity_value.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: polarity sign
    fig2, ax2 = plt.subplots(figsize=fig_size, dpi=150)
    ax2.imshow(np.sign(Pkic), aspect="auto", cmap="seismic",
               vmin=-1, vmax=1, origin="lower")
    ax2.set_xlabel("Event index"); ax2.set_ylabel("Channel index")
    ax2.set_title(f"Polarity sign  ({n_ch} ch × {n_ev} ev)")
    fig2.tight_layout()
    out2 = ctx.fig_dir / "polarity_sign.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    if ctx.show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(out1)))
        display(Image(filename=str(out2)))


def _plot_polarity_along_fiber(ctx, Pkic, event_ids):
    import matplotlib
    import matplotlib.pyplot as plt

    n_ch = Pkic.shape[0]
    pol_first = Pkic[:, 0]
    ch_axis = np.arange(n_ch)
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    ax.scatter(ch_axis[pol_first > 0], np.ones((pol_first > 0).sum()),
               c="tomato", s=8, label="+1")
    ax.scatter(ch_axis[pol_first < 0], -np.ones((pol_first < 0).sum()),
               c="steelblue", s=8, label="-1")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title(f"Polarity along fiber -- event {event_ids[0]}")
    ax.legend(fontsize=9)
    ax.set_ylim(-1.5, 1.5)
    fig.tight_layout()
    out = ctx.fig_dir / f"polarity_along_fiber_{event_ids[0]}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    if ctx.show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(out)))


def _plot_das_station_map(ctx, cal_path):
    """Map of DAS fiber + calibration station positions."""
    import matplotlib
    import matplotlib.pyplot as plt

    das_df = ctx.das_geo_df
    das_lon = das_df["longitude"].values
    das_lat = das_df["latitude"].values

    lon_min, lon_max = das_lon.min(), das_lon.max()
    lat_min, lat_max = das_lat.min(), das_lat.max()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(das_lon, das_lat, 'b.', ms=1, label='DAS fiber')

    if ctx.sta_geo is not None and ctx.sta_geo.exists():
        sta_df = pd.read_csv(ctx.sta_geo)
        sta_uniq = sta_df.drop_duplicates(subset=["network", "station"])
        sta_lon = sta_uniq["longitude"].values
        sta_lat = sta_uniq["latitude"].values
        ax.plot(sta_lon, sta_lat, 'k^', ms=6, label='Stations')
        lon_min = min(lon_min, sta_lon.min())
        lon_max = max(lon_max, sta_lon.max())
        lat_min = min(lat_min, sta_lat.min())
        lat_max = max(lat_max, sta_lat.max())

    if cal_path is not None and cal_path.exists():
        cal_df = pd.read_csv(cal_path)
        if not cal_df.empty:
            cal_net = str(cal_df["network"].iloc[0])
            cal_sta = str(cal_df["station"].iloc[0])
            c_lon = float(cal_df["longitude"].iloc[0])
            c_lat = float(cal_df["latitude"].iloc[0])
            ax.plot(c_lon, c_lat, 'r^', ms=10, zorder=5,
                    label=f'Cal: {cal_net}.{cal_sta}')
            lon_min = min(lon_min, c_lon)
            lon_max = max(lon_max, c_lon)
            lat_min = min(lat_min, c_lat)
            lat_max = max(lat_max, c_lat)

    pad_lon = (lon_max - lon_min) * 0.05
    pad_lat = (lat_max - lat_min) * 0.05
    cos_lat = np.cos(np.radians(0.5 * (lat_min + lat_max)))
    ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
    ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)
    ax.set_aspect(1.0 / cos_lat, adjustable='box')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('DAS fiber & calibration station')
    ax.legend()

    out = ctx.fig_dir / "das_station_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    if ctx.show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(out)))


def _plot_calibration_vote(ctx, agree, disagree):
    import matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ax.barh(["Agree", "Disagree"], [agree, disagree],
            color=["steelblue", "tomato"])
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title(f"Calibration vote  (flip={'YES' if disagree > agree else 'NO'})")
    fig.tight_layout()
    out = ctx.fig_dir / "cal_vote_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    if ctx.show_plots:
        from IPython.display import display, Image
        display(Image(filename=str(out)))
