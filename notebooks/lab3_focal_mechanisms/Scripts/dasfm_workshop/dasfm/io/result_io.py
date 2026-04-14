"""result_io.py — Save/load step3_invert results in HDF5 format.

Replaces the old torch.save/torch.load pattern for inversion .pt files.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


# Candidate fields to skip when saving (internal sorting keys, not needed in output)
SKIP_CANDIDATE_KEYS = set()


def to_numpy(val):
    """Convert torch.Tensor or scalar to numpy/Python type.

    torch is lazy-imported inside the function so this module stays
    fork-safe (importing dasfm.io.result_io must NOT pull torch into
    sys.modules — required by the multi-CPU fork pool runner).
    """
    try:
        import torch  # noqa: PLC0415  (lazy import — fork-safety)
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
    except ImportError:
        pass
    return val


def save_inversion_result(filepath, sol, cand, *, corr_factor=None, obs=None, **meta):
    """Save inversion result to HDF5.

    Parameters
    ----------
    filepath : str or Path
        Output .h5 file path.
    sol : dict or list
        Solution dict (from build_solution_and_cluster) or [] for skipped.
    cand : list[dict]
        Candidate list (from cluster_solution_nooverlap) or [].
    corr_factor : np.ndarray or None
        S/P correction factor (joint modes only).
    obs : dict or None
        Observation data for plotting (das_az, das_takeoff, das_pol, das_sp,
        sta_az, sta_takeoff, sta_pol, sta_sp_az, sta_sp_takeoff, sta_sp).
    **meta : keyword arguments
        Scalar metadata: num_das_pol, num_sta_pol, num_das_sp, num_sta_sp, nmc, skipped.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        # Store inversion mode from filename for self-describing files
        f.attrs["mode"] = filepath.stem

        # File-level attrs
        for key, val in meta.items():
            if isinstance(val, bool):
                f.attrs[key] = val
            elif isinstance(val, (int, np.integer)):
                f.attrs[key] = int(val)
            elif isinstance(val, (float, np.floating)):
                f.attrs[key] = float(val)

        # Solution group
        if isinstance(sol, dict) and sol:
            g = f.create_group("solution")
            for key, val in sol.items():
                val_np = to_numpy(val)
                if isinstance(val_np, np.ndarray):
                    g.create_dataset(key, data=val_np)
                elif isinstance(val_np, (int, np.integer)):
                    g.attrs[key] = int(val_np)
                elif isinstance(val_np, (float, np.floating)):
                    g.attrs[key] = float(val_np)

        # Candidates group
        if cand:
            cg = f.create_group("candidates")
            for i, c in enumerate(cand):
                sg = cg.create_group(str(i))
                for key, val in c.items():
                    if key in SKIP_CANDIDATE_KEYS:
                        continue
                    val_np = to_numpy(val)
                    if isinstance(val_np, np.ndarray):
                        sg.create_dataset(key, data=val_np)
                    elif isinstance(val_np, (int, float, np.integer, np.floating)):
                        sg.attrs[key] = float(val_np)
                    elif isinstance(val_np, list):
                        sg.create_dataset(key, data=np.array(val_np))

        # Correction factor
        if corr_factor is not None:
            corr_np = to_numpy(corr_factor)
            f.create_dataset("corr_factor", data=corr_np)

        # Observation data for plotting
        if obs:
            og = f.create_group("obs")
            for key, val in obs.items():
                val_np = to_numpy(val)
                og.create_dataset(key, data=np.asarray(val_np))


def validate_result_dir(directory) -> None:
    """Pre-flight check: result directory exists and contains an inv_sol/ subdir.

    Required by step3_plot and step4_summarize.

    Raises
    ------
    FileNotFoundError
        If the directory or its inv_sol/ subdirectory does not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Result directory not found: {directory}")
    inv_sol = directory / "inv_sol"
    if not inv_sol.is_dir():
        raise FileNotFoundError(
            f"Result directory missing inv_sol/ subdirectory: {directory}\n"
            f"  Run step3_invert first."
        )


def load_inversion_result(filepath):
    """Load inversion result from HDF5.

    Returns a dict matching the old torch.load structure so downstream code
    (step3_plot, step4_summarize) needs minimal changes.

    Returns
    -------
    dict with keys like:
        solution_pol/solution_joint : dict or []
        candidates_pol/candidates_joint : list[dict] or []
        corr_factor : np.ndarray or None
        num_das_pol, num_sta_pol, num_das_sp, num_sta_sp, nmc : int
        skipped : bool (if present)
    """
    filepath = Path(filepath)
    result = {}

    with h5py.File(filepath, "r") as f:
        # File-level attrs (metadata)
        for key, val in f.attrs.items():
            result[key] = val

        # Solution
        sol = {}
        if "solution" in f:
            sg = f["solution"]
            for key in sg.keys():
                ds = sg[key]
                sol[key] = ds[()] if ds.shape == () else ds[:]
            for key, val in sg.attrs.items():
                sol[key] = val

        # Candidates
        cand = []
        if "candidates" in f:
            cg = f["candidates"]
            for idx in sorted(cg.keys(), key=int):
                c = {}
                sub = cg[idx]
                for key in sub.keys():
                    ds = sub[key]
                    val = ds[()] if ds.shape == () else ds[:]
                    # fm_mean: keep as list for compatibility
                    if key == "fm_mean":
                        c[key] = val.tolist()
                    else:
                        c[key] = float(val) if ds.shape == () else val
                for key, val in sub.attrs.items():
                    c[key] = float(val)
                cand.append(c)

        # Correction factor
        corr_factor = None
        if "corr_factor" in f:
            corr_factor = f["corr_factor"][:]

        # Observation data
        obs = {}
        if "obs" in f:
            for key in f["obs"].keys():
                obs[key] = f["obs"][key][:]
        result["obs"] = obs

        # Determine pol vs joint from filename
        stem = filepath.stem  # e.g. "STA_pol", "DAS_pol_sp"
        is_joint = stem in {
            "STA_pol_sp", "DAS_pol_sp", "STA_pol_DAS_sp", "STA_pol_DAS_pol_sp",
            # Legacy lowercase names
            "sta_pol_sp", "das_pol_sp", "sta_pol_das_sp", "sta_pol_das_pol_sp",
        }

        if is_joint:
            result["solution_joint"] = sol if sol else []
            result["candidates_joint"] = cand
            result["corr_factor"] = corr_factor
        else:
            result["solution_pol"] = sol if sol else []
            result["candidates_pol"] = cand

    return result
