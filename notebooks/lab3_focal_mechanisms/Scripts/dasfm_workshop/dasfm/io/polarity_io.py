"""polarity_io — Save/load step2b polarity H5 output.

Handles the HDF5 format produced by step2b (Pkic, pol_valid, sigma_perc_0,
event_ids, channel_ids).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dasfm.io.das_fft import load_das_fft_meta


def save_polarity_h5(pol_out_path, Pkic, pol_valid, svd_info,
                      event_ids, channel_ids, logger=None):
    """Write the main step2b output H5 (consumed by step3_invert).

    Parameters
    ----------
    pol_out_path : Path
        Output .h5 file path.
    Pkic : np.ndarray (n_ch, n_ev)
        Polarity matrix.
    pol_valid : np.ndarray (n_ch, n_ev) bool
        Polarity validity mask.
    svd_info : dict
        Must contain ``sigma_perc_0`` (torch tensor or numpy array).
    event_ids : list[str]
        Event identifiers.
    channel_ids : np.ndarray (n_ch,) int32
        DAS channel indices.
    """
    pol_out_path = Path(pol_out_path)
    pol_out_path.parent.mkdir(parents=True, exist_ok=True)
    sigma = svd_info["sigma_perc_0"]
    if hasattr(sigma, "cpu"):
        sigma = sigma.cpu().numpy()
    with h5py.File(pol_out_path, "w") as f:
        f.create_dataset("Pkic", data=Pkic)
        f.create_dataset("pol_valid", data=pol_valid)
        f.create_dataset("sigma_perc_0", data=sigma)
        f.create_dataset(
            "event_ids",
            data=np.array(event_ids, dtype=h5py.string_dtype()),
        )
        f.create_dataset("channel_ids", data=channel_ids)
    if logger is not None:
        logger.info(f"  -> {pol_out_path}")


def build_pol_valid(fft_dir, event_ids, n_ch):
    """Build ``(n_ch, n_ev) bool`` from per-event ``p_valid`` in das_fft cache.

    Parameters
    ----------
    fft_dir : Path
        Directory containing per-event FFT H5 files.
    event_ids : list[str]
        Event identifiers.
    n_ch : int
        Number of DAS channels.
    """
    n_ev = len(event_ids)
    pol_valid = np.zeros((n_ch, n_ev), dtype=bool)
    for i, eid in enumerate(event_ids):
        meta = load_das_fft_meta(fft_dir, eid)
        if meta is not None and meta["p_valid"] is not None:
            pol_valid[:, i] = meta["p_valid"]
        else:
            pol_valid[:, i] = True
    return pol_valid
