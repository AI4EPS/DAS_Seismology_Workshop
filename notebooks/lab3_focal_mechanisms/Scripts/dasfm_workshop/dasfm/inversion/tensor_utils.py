"""
Tensor/array conversion utilities.

Extracted from util_ytf.py (old_code).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def numpy_to_torch(x, device="cpu"):
    """Convert numpy array / scalar to float32 torch tensor on *device*."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def torch_to_numpy(x):
    """Detach and move tensor to CPU numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def dict_to_torch(d: dict, device="cpu") -> dict:
    """Convert a mixed dict (numpy / pandas / scalar) to torch tensors."""
    if isinstance(device, str):
        device = torch.device(device)

    out = {}
    for k, v in d.items():
        if isinstance(v, pd.Series):
            arr = v.to_numpy()
            if np.issubdtype(arr.dtype, np.number) or arr.dtype == bool:
                out[k] = torch.from_numpy(arr).float().to(device)
            else:
                out[k] = v
        elif isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.number) or v.dtype == bool:
                out[k] = torch.from_numpy(v).float().to(device)
            else:
                out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = torch.tensor(v, dtype=torch.float32, device=device)
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            out[k] = torch.tensor(v, dtype=torch.float32, device=device)
        else:
            out[k] = v
    return out


def dict_to_numpy(d: dict) -> dict:
    """Convert dict of tensors to numpy arrays."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = v
    return out
