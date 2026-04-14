"""Multi-channel cross-correlation (MCCC) for P-wave relative polarity picking.

Adapted from old_code/util/mccc.py and old_code/util/transforms.py
(original author: Jiaxuan Li, Caltech).

Public API
----------
* :func:`xcorr_freq`            — per-channel normalised cross-correlation (FFT)
* :func:`Pkic_from_Ckij_Skij`   — SVD-based absolute polarity from pairwise matrix
* :class:`MCCCPicker`            — MCCC least-squares picker

All functions accept torch.Tensor (CPU or GPU).  Pass ``device='cuda'`` where
supported to use GPU acceleration.

Reference
---------
VanDecar & Crosson 1990 — Determination of teleseismic relative phase arrival
times using multi-channel cross-correlation and least squares.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.signal.windows import tukey
from scipy.sparse.linalg import lsmr
from tqdm import tqdm

from dasfm.utils.step_utils import log_or_print


# ===========================================================================
# Utility
# ===========================================================================

def nextpow2(i: int) -> int:
    n = 1
    while n < i:
        n *= 2
    return n


def moving_average(data: torch.Tensor, ma: int) -> torch.Tensor:
    """Moving average along the time axis (dim=-1)."""
    m = torch.nn.AvgPool1d(ma, stride=1, padding=ma // 2)
    return m(data.transpose(1, 0))[:, : data.shape[0]].transpose(1, 0)


def taper_time(data: torch.Tensor, alpha: float = 0.8) -> torch.Tensor:
    taper = tukey(data.shape[-1], alpha)
    return data * torch.tensor(taper, device=data.device, dtype=data.dtype)


def normalize_waveform(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x -= torch.mean(x, dim=-1, keepdim=True)
    norm = x.square().sum(dim=-1, keepdim=True).sqrt()
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    x /= norm
    return x


def gather_roll(data: torch.Tensor, shift_index: torch.Tensor) -> torch.Tensor:
    """Roll each row of *data* along axis 1 by the corresponding *shift_index*."""
    nrow, ncol = data.shape
    index = torch.arange(ncol, device=data.device).view(1, ncol).repeat(nrow, 1)
    index = (index - shift_index.view(nrow, 1)) % ncol
    return torch.gather(data, 1, index)


# ===========================================================================
# Cross-correlation
# ===========================================================================

def xcorr_freq(
    data1: torch.Tensor,
    data2: torch.Tensor,
    dt: float,
    maxlag: float = 0.5,
    channel_shift: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Per-channel normalised cross-correlation in the frequency domain.

    Parameters
    ----------
    data1, data2 : torch.Tensor, shape (n_ch, n_t)
        Input waveforms.  Must reside on the same device.
    dt : float
        Sampling interval [s].
    maxlag : float
        Maximum lag kept in the output [s].  Default 0.5.
    channel_shift : int
        If > 0, data2 is rolled by this many channels before correlation
        (used to compute the spatially-shifted polarity matrix).

    Returns
    -------
    xcor : torch.Tensor, shape (n_ch, 2*nlag-1)
        Normalised cross-correlation traces.
    xcor_info : dict
        ``nx``, ``nt``, ``dt``, ``time_axis`` metadata.
    """
    nlag = int(maxlag / dt)
    nfast = nextpow2(2 * data1.shape[-1] - 1)

    data_freq1 = torch.fft.rfft(normalize_waveform(data1), n=nfast, dim=-1)
    data_freq2 = torch.fft.rfft(normalize_waveform(data2), n=nfast, dim=-1)

    if channel_shift > 0:
        xcor_freq = data_freq1 * torch.roll(
            torch.conj(data_freq2), channel_shift, dims=0
        )
    else:
        xcor_freq = data_freq1 * torch.conj(data_freq2)

    xcor_time = torch.fft.irfft(xcor_freq, n=nfast, dim=-1)
    xcor = torch.roll(xcor_time, nfast // 2, dims=-1)[
        ..., nfast // 2 - nlag + 1 : nfast // 2 + nlag
    ]

    nlag_out = 2 * nlag - 1
    time_axis = (np.arange(nlag_out) - (nlag_out // 2)) * dt
    xcor_info = {
        "nx": data1.shape[0],
        "nt": nlag_out,
        "dt": dt,
        "time_axis": time_axis,
    }
    return xcor, xcor_info


def prepare_xcorr_freq(data: torch.Tensor) -> torch.Tensor:
    """Normalize and FFT one event's waveforms for use with :func:`xcorr_from_freq`.

    Call once per event *before* the pair loop so that each event's FFT is
    computed O(n_ev) times instead of O(n_ev²).

    Parameters
    ----------
    data : torch.Tensor, shape (n_ch, n_win)
        Waveform tensor already on the target device.

    Returns
    -------
    torch.Tensor, shape (n_ch, nfast//2 + 1), complex
        Frequency-domain representation ready for :func:`xcorr_from_freq`.
    """
    nfast = nextpow2(2 * data.shape[-1] - 1)
    return torch.fft.rfft(normalize_waveform(data), n=nfast, dim=-1)


def xcorr_from_freq(
    data_freq1: torch.Tensor,
    data_freq2: torch.Tensor,
    dt: float,
    maxlag: float = 0.5,
    channel_shift: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Per-channel normalised cross-correlation from pre-computed freq tensors.

    Drop-in replacement for :func:`xcorr_freq` when the frequency-domain
    representations have already been prepared with :func:`prepare_xcorr_freq`.
    Skips the normalize + rfft step, saving two O(n_ch * n_win * log n_win)
    FFTs per pair call.

    Parameters
    ----------
    data_freq1, data_freq2 : torch.Tensor, shape (n_ch, nfast//2 + 1)
        Outputs of :func:`prepare_xcorr_freq`.  Must reside on the same device.
    dt : float
        Sampling interval [s].
    maxlag : float
        Maximum lag kept in the output [s].  Default 0.5.
    channel_shift : int
        If > 0, data_freq2 is rolled by this many channels before correlation.

    Returns
    -------
    xcor : torch.Tensor, shape (n_ch, 2*nlag-1)
    xcor_info : dict
        ``nx``, ``nt``, ``dt``, ``time_axis`` metadata.
    """
    nlag  = int(maxlag / dt)
    nfast = (data_freq1.shape[-1] - 1) * 2   # recover from rfft output size

    if channel_shift > 0:
        xcor_f = data_freq1 * torch.roll(
            torch.conj(data_freq2), channel_shift, dims=0
        )
    else:
        xcor_f = data_freq1 * torch.conj(data_freq2)

    xcor_time = torch.fft.irfft(xcor_f, n=nfast, dim=-1)
    xcor = torch.roll(xcor_time, nfast // 2, dims=-1)[
        ..., nfast // 2 - nlag + 1 : nfast // 2 + nlag
    ]

    nlag_out  = 2 * nlag - 1
    time_axis = (np.arange(nlag_out) - (nlag_out // 2)) * dt
    xcor_info = {
        "nx": data_freq1.shape[0],
        "nt": nlag_out,
        "dt": dt,
        "time_axis": time_axis,
    }
    return xcor, xcor_info


# ===========================================================================
# SVD-based polarity resolution
# ===========================================================================

def Pkic_from_Ckij_Skij(Ckij, Skij):
    """Absolute polarity vectors from per-channel pairwise polarity matrices.

    Resolves the +/-1 sign ambiguity across channels using the spatially-shifted
    correlation matrix Skij.

    Parameters
    ----------
    Ckij : np.ndarray (n_ch, n_ev, n_ev) or list[sparse matrix]
        Relative polarity matrix at zero spatial shift.
    Skij : np.ndarray (n_ch, n_ev, n_ev) or list[sparse matrix]
        Relative polarity matrix with one-channel spatial shift.

    Returns
    -------
    Pkic : torch.Tensor, shape (n_ch, n_ev)
    info : dict
    """
    import numpy as np
    from scipy.sparse import issparse

    is_sparse = isinstance(Ckij, list)
    nchan = len(Ckij) if is_sparse else Ckij.shape[0]
    nevent = Ckij[0].shape[0] if is_sparse else Ckij.shape[1]

    Pki  = torch.zeros(nchan, nevent)
    Pkic = torch.zeros(nchan, nevent)
    sigma_perc_0  = torch.zeros(nchan)
    sigma_perc_1  = torch.zeros(nchan)
    sigma_ratio_0 = torch.zeros(nchan)
    sigma_ratio_1 = torch.zeros(nchan)

    for k in tqdm(range(nchan), desc="SVD polarity", leave=False):
        Ck = Ckij[k]

        if is_sparse or issparse(Ck):
            # Sparse SVD: only compute leading singular vector
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import svds
            C_csr = csr_matrix(Ck) if not issparse(Ck) else Ck.tocsr()
            if C_csr.nnz == 0:
                continue
            u, s, _ = svds(C_csr, k=1)
            Pki[k]  = torch.from_numpy(u[:, 0])
            Pkic[k] = torch.from_numpy(u[:, 0])
            sigma_perc_0[k] = float(s[0])
        else:
            # Dense SVD (torch)
            Ck_t = torch.as_tensor(Ck) if not isinstance(Ck, torch.Tensor) else Ck
            u, s, _ = torch.linalg.svd(Ck_t)
            Pki[k]  = u[:, 0]
            Pkic[k] = u[:, 0]
            sigma_perc_0[k]  = s[0] / s.sum()
            sigma_ratio_0[k] = s[0] / s[1] if s[1] > 0 else torch.inf

        if k >= 1:
            Sk = Skij[k]
            if is_sparse or issparse(Sk):
                from scipy.sparse import csr_matrix
                from scipy.sparse.linalg import svds
                S_csr = csr_matrix(Sk) if not issparse(Sk) else Sk.tocsr()
                if S_csr.nnz == 0:
                    continue
                u_s, s_s, vt_s = svds(S_csr, k=1)
                u_s0 = torch.from_numpy(u_s[:, 0])
                v_s0 = torch.from_numpy(vt_s[0])
                csign = torch.sign(torch.dot(u_s0, v_s0))
            else:
                Sk_t = torch.as_tensor(Sk) if not isinstance(Sk, torch.Tensor) else Sk
                u_s, s_s, v_s = torch.linalg.svd(Sk_t)
                csign = torch.sign(torch.dot(u_s[:, 0], v_s[0]))
                sigma_perc_1[k]  = s_s[0] / s_s.sum()
                sigma_ratio_1[k] = s_s[0] / s_s[1] if s_s[1] > 0 else torch.inf

            sign_val = torch.sign(
                torch.dot(Pkic[k], Pkic[k - 1]) * csign
            )
            if sign_val == 0:
                import warnings
                warnings.warn(
                    f"Pkic_from_Ckij_Skij: channel {k} sign-chain dot product "
                    f"is zero — polarity will be zeroed for this and all "
                    f"subsequent channels until a non-zero dot is encountered.",
                    stacklevel=2,
                )
            Pkic[k] *= sign_val

    info = {
        "Pki":          Pki,
        "sigma_perc_0": sigma_perc_0,
        "sigma_perc_1": sigma_perc_1,
        "sigma_ratio_0": sigma_ratio_0,
        "sigma_ratio_1": sigma_ratio_1,
    }
    return Pkic, info


# ===========================================================================
# Precomputed indices & resample parameters (shared across event pairs)
# ===========================================================================

class MCCCPrecomputed:
    """Precomputed data shared across MCCCPicker instances.

    Created once before the event-pair loop via :meth:`MCCCPrecomputed.build`
    and passed to every ``MCCCPicker(precomputed=...)`` call.  This avoids
    rebuilding the ``_form_coo`` index arrays (expensive Python list
    comprehension) and resample constants for every pair.
    """

    __slots__ = (
        "index_i", "index_j",
        "npts_xcor", "npts_xcor_pad", "resample_factor", "resample_dt",
        "zero_complex",
    )

    def __init__(
        self,
        index_i: torch.Tensor,
        index_j: torch.Tensor,
        npts_xcor: int,
        npts_xcor_pad: int,
        resample_factor: float,
        resample_dt: float,
        zero_complex: torch.Tensor,
    ) -> None:
        self.index_i = index_i
        self.index_j = index_j
        self.npts_xcor = npts_xcor
        self.npts_xcor_pad = npts_xcor_pad
        self.resample_factor = resample_factor
        self.resample_dt = resample_dt
        self.zero_complex = zero_complex

    @staticmethod
    def build(
        nchan: int,
        ntime: int,
        dt: float,
        mccc_maxwin: int = 10,
        scale_factor: int = 10,
        device: str | torch.device = "cpu",
    ) -> "MCCCPrecomputed":
        """Build precomputed data for a given (nchan, ntime, dt) geometry.

        Parameters
        ----------
        nchan : int
            Number of channels.
        ntime : int
            Number of time samples in the xcorr gather.
        dt : float
            Sampling interval [s].
        mccc_maxwin : int
            Maximum channel span for the MCCC sparse system.
        scale_factor : int
            Sub-sample upsampling factor.
        device : str or torch.device
            Target device for tensors.
        """
        dev = torch.device(device)

        # --- _form_coo indices ---
        ii = []
        jj = []
        for i in range(nchan - 1):
            for j in range(i + 1, min(i + mccc_maxwin, nchan)):
                ii.append(i)
                jj.append(j)
        index_i = torch.tensor(ii, device=dev)
        index_j = torch.tensor(jj, device=dev)

        # --- resample constants ---
        npts_xcor     = nextpow2(2 * ntime - 1)
        npts_xcor_pad = nextpow2(2 * scale_factor * ntime - 1)
        resample_factor = npts_xcor_pad / npts_xcor
        resample_dt     = dt / resample_factor
        zero_complex = torch.complex(
            torch.tensor(0.0, device=dev),
            torch.tensor(0.0, device=dev),
        )

        return MCCCPrecomputed(
            index_i=index_i,
            index_j=index_j,
            npts_xcor=npts_xcor,
            npts_xcor_pad=npts_xcor_pad,
            resample_factor=resample_factor,
            resample_dt=resample_dt,
            zero_complex=zero_complex,
        )


# ===========================================================================
# MCCCPicker
# ===========================================================================

class MCCCPicker:
    """Multi-channel cross-correlation picker / aligner.

    Reference: VanDecar & Crosson 1990.

    Parameters
    ----------
    data : torch.Tensor, shape (n_ch, n_t)
        Input data.  If the data **is** a cross-correlation gather (output of
        :func:`xcorr_freq`), set ``mode='pick'`` (default).
        If the data is a raw waveform gather, set ``mode='align'``.
    dt : float
        Sampling interval [s].
    taper : float or None
        Tukey taper applied to the input data.  Default 0.8.
    ma : int or None
        Moving-average smoothing window (samples).  Default 40.
    scale_factor : int
        Sub-sample upsampling factor via zero-padding in frequency domain.
    damp : float
        Spatial-smoothness damping weight.
    mccc_damp : float
        Weight factor for the MCCC difference equations.
    mccc_mincc : float
        Minimum correlation coefficient threshold for including a pair.
    mccc_maxlag : float
        Maximum allowable time lag [s] between neighbouring channels.
    mccc_maxwin : int
        Maximum channel span for building the MCCC sparse system.
    whitening : bool
        Apply spectral whitening before upsampling.
    mode : ``'pick'`` or ``'align'``
        Operating mode.
    """

    def __init__(
        self,
        data: torch.Tensor,
        dt: float,
        taper: float = 0.8,
        ma: int = 40,
        scale_factor: int = 10,
        damp: float = 1.0,
        mccc_damp: float = 1.0,
        mccc_mincc: float = 0.7,
        mccc_maxlag: float = 0.04,
        mccc_maxwin: int = 10,
        chunk_size: int = 50_000,
        whitening: bool = True,
        whitening_waterlevel: float = 0.1,
        win_main: float = 0.3,
        win_side: float = 0.1,
        w0: float = 10.0,
        max_niter: int = 5,
        refine_ma: int = 40,
        mode: str = "pick",
        return_data_align: bool = False,
        precomputed: MCCCPrecomputed | None = None,
        logger=None,
    ) -> None:
        self.logger = logger
        self.device = data.device
        self.taper  = taper
        self.ma     = ma
        self.nchan, self.ntime = data.shape
        self.dt  = dt
        self.damp = damp
        self.chunk_size   = chunk_size
        self.mccc_damp    = mccc_damp
        self.mccc_mincc   = mccc_mincc
        self.mccc_maxlag  = mccc_maxlag
        self.mccc_nlag    = int(mccc_maxlag / dt)
        self.mccc_maxwin  = mccc_maxwin

        assert int(scale_factor) >= 1
        self.scale_factor = int(scale_factor)

        # Use precomputed resample constants if available, else compute them
        if precomputed is not None:
            self._precomputed        = precomputed
            self.npts_xcor           = precomputed.npts_xcor
            self.npts_xcor_pad       = precomputed.npts_xcor_pad
            self.resample_factor     = precomputed.resample_factor
            self.resample_dt         = precomputed.resample_dt
            self.zero_complex        = precomputed.zero_complex
        else:
            self._precomputed        = None
            self.npts_xcor           = nextpow2(2 * self.ntime - 1)
            self.npts_xcor_pad       = nextpow2(2 * self.scale_factor * self.ntime - 1)
            self.resample_factor     = self.npts_xcor_pad / self.npts_xcor
            self.resample_dt         = self.dt / self.resample_factor
            self.zero_complex = torch.complex(
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
            )
        self.resample_mccc_nlag = int(mccc_maxlag / self.resample_dt)

        self.refine_ma   = refine_ma
        self.win_main    = win_main
        self.win_side    = win_side
        self.w0          = w0
        self.max_niter   = max_niter
        # preprocessing
        if self.taper is not None:
            data = taper_time(data, self.taper)
        self.data_raw = data
        if self.ma is not None:
            data = moving_average(data, self.ma)
        self.data      = data              # NOT normalized (matches original mccc.py)
        self.data_freq = self._fft(self.data)
        # spectral whitening parameters
        self.whitening   = whitening
        self.whitening_waterlevel = whitening_waterlevel
        self.mode              = mode
        self.return_data_align = return_data_align

    # ------------------------------------------------------------------
    def set_data(self, data: torch.Tensor) -> None:
        """Replace internal data without rebuilding the picker."""
        assert self.data.shape == data.shape
        if self.taper is not None:
            data = taper_time(data, self.taper)
        self.data_raw = data
        if self.ma is not None:
            data = moving_average(data, self.ma)
        self.data      = data
        self.data_freq = self._fft(self.data)

    def _fft(self, data: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft(data, n=self.npts_xcor, dim=-1)

    @property
    def fft_real_xcor_freq_axis(self) -> torch.Tensor:
        """Positive frequency axis for the xcorr FFT."""
        return torch.linspace(0, 0.5 / self.dt, self.npts_xcor // 2 + 1)

    def _spectral_whitening(self, df: torch.Tensor) -> torch.Tensor:
        e1 = torch.mean(torch.abs(df) ** 2, dim=-1, keepdim=True).sqrt()
        df = df / (torch.abs(df) + self.whitening_waterlevel)
        e2 = torch.mean(torch.abs(df) ** 2, dim=-1, keepdim=True).sqrt()
        return df / e2 * e1

    def _resample(self, data: torch.Tensor) -> tuple[torch.Tensor, float]:
        ntime_q    = data.shape[-1] * self.scale_factor
        dt_rs      = self.dt / self.scale_factor
        df         = torch.fft.rfft(data, dim=-1)
        if self.whitening:
            df = self._spectral_whitening(df)
        df_pad = F.pad(
            df,
            (0, ntime_q // 2 - data.shape[-1] // 2),
            "constant",
            self.zero_complex,
        )
        data_rs = (
            torch.fft.irfft(df_pad, n=ntime_q, dim=-1) * self.scale_factor
        )
        npts_rs = (data.shape[-1] - 1) * self.scale_factor + 1
        return data_rs[:, :npts_rs], dt_rs

    # ------------------------------------------------------------------
    def _pick_win_maxabs(
        self,
        data: torch.Tensor,
        dt: float,
        maxlag: float,
        update_vmin: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nlag = int(maxlag / dt)
        nt   = data.shape[-1]
        ic   = nt // 2
        ib   = max(0, ic - nlag + 1)
        ie   = min(nt, ic + nlag)
        vmax, imax = torch.max(data[:, ib:ie], dim=-1)
        vmin, imin = torch.min(data[:, ib:ie], dim=-1)
        ineg = torch.abs(vmin) > vmax
        if update_vmin:
            vmax[ineg], vmin[ineg] = vmin[ineg], vmax[ineg]
        else:
            vmax[ineg] = vmin[ineg]
        imax[ineg]  = imin[ineg]
        tmax = (imax - nlag + 1) * dt
        return vmax, vmin, tmax

    def _pick_side_lobe(
        self, data: torch.Tensor, dt: float, win_side: float
    ) -> torch.Tensor:
        nt  = data.shape[-1]
        ic  = nt // 2
        nw2 = int(win_side / dt)
        lb  = max(0, ic - nw2 + 1)
        le  = max(0, ic - nw2 // 2 + 1)
        rb  = min(nt, ic + nw2 // 2)
        re  = min(nt, ic + nw2)
        return torch.maximum(
            torch.max(torch.abs(data[:, lb:le]), dim=-1).values,
            torch.max(torch.abs(data[:, rb:re]), dim=-1).values,
        )

    def _pick_shift_win(
        self,
        data: torch.Tensor,
        dt: float,
        maxlag: float,
        shift_index: torch.Tensor,
        ma: int | None = None,
        update_vmin: bool = False,
    ) -> tuple:
        rolled = gather_roll(data, shift_index)
        if ma is not None:
            rolled = moving_average(rolled, ma)
        return self._pick_win_maxabs(rolled, dt, maxlag, update_vmin=update_vmin)

    # ------------------------------------------------------------------
    def _form_coo(self):
        """Build the sparse MCCC system from neighbouring-channel xcorr."""
        import time as _t_coo
        _dbg_coo = getattr(self, '_dbg_timing', False)

        if self._precomputed is not None:
            index_i = self._precomputed.index_i
            index_j = self._precomputed.index_j
        else:
            _ts = _t_coo.perf_counter()
            index_i = torch.tensor(
                [i for i in range(self.nchan - 1)
                   for _ in range(i + 1, min(i + self.mccc_maxwin, self.nchan))],
                device=self.device,
            )
            index_j = torch.tensor(
                [j for i in range(self.nchan - 1)
                   for j in range(i + 1, min(i + self.mccc_maxwin, self.nchan))],
                device=self.device,
            )
            if _dbg_coo:
                log_or_print(self.logger, f"      [_form_coo] build index: {(_t_coo.perf_counter()-_ts)*1000:.1f} ms", file_only=True)
        npair   = len(index_i)
        mccc_cc = torch.zeros(npair, device=self.device)
        mccc_dt = torch.zeros(npair, device=self.device)
        nchunk  = int(np.ceil(npair / self.chunk_size))
        if _dbg_coo:
            log_or_print(self.logger, f"      [_form_coo] npair={npair}  nchunk={nchunk}  chunk_size={self.chunk_size}", file_only=True)
        ib = 0
        _t_chunk_total = 0.0
        for ichunk in range(nchunk):
            _tc = _t_coo.perf_counter()
            ie = min(ib + self.chunk_size, npair)
            ii = index_i[ib:ie]
            jj = index_j[ib:ie]
            xcf = F.pad(
                self.data_freq[ii] * torch.conj(self.data_freq[jj]),
                (0, self.npts_xcor_pad // 2 - self.npts_xcor // 2),
                "constant",
                self.zero_complex,
            )
            xct = torch.roll(
                torch.fft.irfft(xcf, n=self.npts_xcor_pad, dim=-1),
                self.npts_xcor_pad // 2,
                dims=-1,
            )
            mccc_cc[ib:ie], _, mccc_dt[ib:ie] = self._pick_win_maxabs(
                xct, self.resample_dt, self.mccc_maxlag, update_vmin=False
            )
            ib = ie
            _t_chunk_total += _t_coo.perf_counter() - _tc
            if _dbg_coo:
                log_or_print(self.logger, f"      [_form_coo] chunk {ichunk+1}/{nchunk}: {(_t_coo.perf_counter()-_tc)*1000:.1f} ms", file_only=True)
        if _dbg_coo:
            log_or_print(self.logger, f"      [_form_coo] chunk loop total: {_t_chunk_total*1000:.1f} ms", file_only=True)
        mccc_cc *= self.resample_factor
        return index_i, index_j, mccc_cc, mccc_dt

    def _form_Ab(self, index_i, index_j, mccc_cc, mccc_dt):
        index_i = index_i.cpu()
        index_j = index_j.cpu()
        mccc_cc = mccc_cc.cpu()
        mccc_dt = mccc_dt.cpu()
        igood   = torch.where(torch.abs(mccc_cc) > self.mccc_mincc)[0]
        ngood   = len(igood)
        mccc_cc = mccc_cc[igood]
        mccc_dt = mccc_dt[igood]
        weight  = np.ones(ngood) * self.mccc_damp
        idx_ii  = np.tile(np.arange(ngood), 2)
        idx_jj  = torch.cat([index_i[igood], index_j[igood]]).cpu().numpy()
        val_ij  = np.concatenate([weight, -weight])
        D = sparse.coo_matrix(
            (val_ij, (idx_ii, idx_jj)), shape=(ngood, self.nchan)
        )
        d = mccc_dt.cpu().numpy() * weight
        S = (
            np.diag(np.ones(self.nchan))
            - np.diag(np.ones(self.nchan - 1), k=-1)
        )[1:, :]
        S = sparse.csr_matrix(S) * self.damp
        A = sparse.vstack((D, S))
        b = np.concatenate([d, np.zeros(S.shape[0])])
        return A, b

    def _concat_Ab_pick(self, A, b, pick_dt, pick_cc, w):
        isel = torch.where(
            torch.abs(pick_cc)
            > torch.quantile(torch.abs(pick_cc), 0.15)
        )[0]
        nsel = len(isel)
        P = sparse.coo_matrix(
            (np.ones(nsel), (np.arange(nsel), isel.cpu().numpy())),
            shape=(nsel, len(pick_dt)),
        )
        p = pick_dt[isel].cpu().numpy()
        return sparse.vstack([w * A, P]), np.concatenate([w * b, p])

    # ------------------------------------------------------------------
    def pick(self, pick_dt: torch.Tensor, win: float | None = None) -> dict:
        """Pick directly in a narrow window with known *pick_dt*.

        Parameters
        ----------
        pick_dt : torch.Tensor, shape (n_ch,)
            Known time shifts per channel.
        win : float or None
            Half-window for picking [s].  Default: 5 × resampled dt.

        Returns
        -------
        dict with keys ``cc_dt``, ``cc_main``, ``cc_mean``, ``cc_side``.
        """
        data_rs, dt_rs = self._resample(self.data)
        if win is None:
            win = dt_rs * 5
        shift_idx = -torch.round(pick_dt / dt_rs).int()
        pick_cc, _, pick_dt_refine = self._pick_shift_win(
            data_rs, dt_rs, win, shift_idx, ma=self.refine_ma,
        )
        pick_side = self._pick_side_lobe(data_rs, dt_rs, self.win_side)
        sol = {
            "cc_dt":   pick_dt + pick_dt_refine,
            "cc_main": pick_cc,
            "cc_mean": torch.mean(torch.abs(pick_cc)),
            "cc_side": pick_side,
        }
        if self.return_data_align:
            sol["data_align"] = moving_average(
                gather_roll(data_rs, shift_idx), 40
            )
            nt_rs = data_rs.shape[-1]
            sol["data_align_info"] = {
                "time_axis": (np.arange(nt_rs) - nt_rs // 2) * dt_rs,
                "nx": self.nchan,
                "dt": dt_rs,
            }
        return sol

    # ------------------------------------------------------------------
    def solve(self) -> dict:
        """Solve the MCCC least-squares system and pick polarities.

        Returns
        -------
        dict with keys:
            ``cc_dt``    — per-channel time shift [s]
            ``cc_main``  — signed peak xcorr (= relative polarity indicator)
            ``cc_mean``  — mean absolute xcorr
            ``cc_side``  — side-lobe peak (only in ``'pick'`` mode)
        """
        import time as _t_s
        _dbg = False
        self._dbg_timing = False
        _ts = _t_s.perf_counter()

        def _lap(label):
            nonlocal _ts
            if _dbg:
                log_or_print(self.logger, f"    [solve] {label}: {(_t_s.perf_counter()-_ts)*1000:.1f} ms", file_only=True)
            _ts = _t_s.perf_counter()

        index_i, index_j, value_cc, value_dt = self._form_coo()
        _lap("_form_coo")
        A0, b0 = self._form_Ab(index_i, index_j, value_cc, value_dt)
        _lap("_form_Ab")
        data_rs, dt_rs = self._resample(self.data)
        _lap("_resample")

        niter      = 0
        shift_idx  = torch.arange(self.nchan, dtype=torch.int, device=self.device)
        w          = self.w0
        win_main   = self.win_main
        pick_dt    = torch.zeros(self.nchan, device=self.device)
        pick_cc    = torch.zeros(self.nchan, device=self.device)

        while niter < self.max_niter:
            if niter == self.max_niter - 1:
                win_main = min(0.05, win_main)
            if niter == 0:
                pick_cc, _, pick_dt_refine = self._pick_win_maxabs(
                    moving_average(data_rs, self.refine_ma)
                    if self.mode == "pick"
                    else data_rs,
                    dt_rs,
                    win_main,
                    update_vmin=False,
                )
            else:
                pick_cc, _, pick_dt_refine = self._pick_win_maxabs(
                    moving_average(gather_roll(data_rs, shift_idx), self.refine_ma),
                    dt_rs,
                    win_main,
                    update_vmin=False,
                )
            if self.mode == "pick":
                A, b = self._concat_Ab_pick(
                    A0, b0, pick_dt + pick_dt_refine, pick_cc, w)
            else:
                A, b = A0, b0
            solution  = lsmr(A, b)
            pick_dt[:]  = torch.tensor(solution[0], device=self.device)
            shift_idx[:] = -torch.round(pick_dt / dt_rs).int()
            _lap(f"iter {niter} (pick+lsmr)")
            w        /= 1.1
            win_main /= 2.0
            niter    += 1

        sol = {
            "cc_dt":   pick_dt,
            "cc_main": pick_cc,
            "cc_mean": torch.mean(torch.abs(pick_cc)),
        }
        if self.mode == "pick":
            sol["cc_side"] = self._pick_side_lobe(data_rs, dt_rs, self.win_side)
        if self.return_data_align:
            sol["data_align"] = moving_average(
                gather_roll(data_rs, shift_idx), 40
            )
            nt_rs = data_rs.shape[-1]
            sol["data_align_info"] = {
                "time_axis": (np.arange(nt_rs) - nt_rs // 2) * dt_rs,
                "nx": self.nchan,
                "dt": dt_rs,
            }
        return sol
