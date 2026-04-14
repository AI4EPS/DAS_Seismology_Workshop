"""GPU-accelerated MCCCPicker using normal equations + torch.linalg.solve.

Replaces scipy sparse LSMR with the normal equations AᵀAx = Aᵀb,
solved by torch.linalg.solve (GPU Cholesky/LU) on an nchan×nchan dense
float64 matrix.  All other logic (xcorr, resample, pick) is inherited
from :class:`~dasfm.picking.mccc.MCCCPicker`.

AᵀA size: nchan × nchan = O(nchan²) ≈ 87 MB for nchan=4670 (float64).

Public API
----------
* :class:`MCCCPickerGPU`
"""

from __future__ import annotations

import numpy as np
import torch

from dasfm.picking.mccc import (
    MCCCPicker,
    gather_roll,
    moving_average,
)


class MCCCPickerGPU(MCCCPicker):
    """MCCCPicker with GPU normal-equations solver.

    Identical interface to :class:`MCCCPicker`.  Pass GPU tensors via
    the ``data`` argument and set the appropriate device (``data.device``
    is used automatically).

    The scipy LSMR call is replaced by:

    1. Build ``ATA_base`` (nchan × nchan, float64, GPU) and
       ``ATb_base`` (nchan, float64, GPU) once per :meth:`solve` call
       from the D (pair-correlation) and S (smoothness) constraints.
    2. Per iteration, add the P (pick) constraint to ATA/ATb and solve
       with ``torch.linalg.solve``.

    All other methods (``_form_coo``, ``_resample``, ``_pick_win_maxabs``,
    etc.) are inherited unchanged.
    """

    # ------------------------------------------------------------------
    def form_Ab_gpu(
        self,
        index_i: torch.Tensor,
        index_j: torch.Tensor,
        mccc_cc: torch.Tensor,
        mccc_dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the fixed part of the normal equations on GPU.

        Parameters
        ----------
        index_i, index_j : Tensor (npair,)  int
            Channel indices of each neighbouring pair.
        mccc_cc, mccc_dt : Tensor (npair,)  float
            Correlation coefficient and time lag for each pair.

        Returns
        -------
        ATA_base : Tensor (nchan, nchan) float64
        ATb_base : Tensor (nchan,)       float64
        """
        dev = self.device
        nc  = self.nchan

        # --- filter by minimum CC ---
        igood = torch.where(torch.abs(mccc_cc) > self.mccc_mincc)[0]
        ii  = index_i[igood].long()   # GPU, int64
        jj  = index_j[igood].long()   # GPU, int64
        dtt = mccc_dt[igood].double() # GPU, float64
        w2  = float(self.mccc_damp) ** 2

        ATA  = torch.zeros(nc, nc, dtype=torch.float64, device=dev)
        ATb  = torch.zeros(nc,     dtype=torch.float64, device=dev)
        flat = ATA.view(nc * nc)      # shared storage

        # --- D.T D diagonal: ATA[ii,ii] += w2, ATA[jj,jj] += w2 ---
        w2_vec = torch.full((len(ii),), w2, dtype=torch.float64, device=dev)
        flat.scatter_add_(0, ii * nc + ii, w2_vec)
        flat.scatter_add_(0, jj * nc + jj, w2_vec)

        # --- D.T D off-diagonal (symmetric): ATA[ii,jj] -= w2, ATA[jj,ii] -= w2 ---
        neg_w2 = torch.full((len(ii) * 2,), -w2, dtype=torch.float64, device=dev)
        flat.scatter_add_(0, torch.cat([ii * nc + jj, jj * nc + ii]), neg_w2)

        # --- D.T b ---
        ATb.scatter_add_(0, ii,  dtt * w2)
        ATb.scatter_add_(0, jj, -dtt * w2)

        # --- S.T S (tridiagonal smoothness constraint, b_S = 0) ---
        # S[k, k] = damp, S[k, k-1] = -damp for k in 1..nchan-1
        # S.T S: diag[0]=d2, diag[1..nchan-2]=2*d2, diag[-1]=d2
        #        off-diag[k, k+1] = off-diag[k+1, k] = -d2
        d2  = float(self.damp) ** 2
        idx = torch.arange(nc, device=dev, dtype=torch.long)
        ATA[idx[:-1], idx[:-1]] += d2
        ATA[idx[1:],  idx[1:]]  += d2
        ATA[idx[:-1], idx[1:]]  -= d2
        ATA[idx[1:],  idx[:-1]] -= d2

        return ATA, ATb

    # ------------------------------------------------------------------
    def solve(self) -> dict:
        """Solve MCCC using GPU normal equations.

        Returns
        -------
        dict with the same keys as :meth:`MCCCPicker.solve`:
            ``cc_dt``, ``cc_main``, ``cc_mean``, optionally ``cc_side``,
            ``data_align``, ``data_align_info``.
        """
        index_i, index_j, value_cc, value_dt = self._form_coo()
        ATA_base, ATb_base = self.form_Ab_gpu(
            index_i, index_j, value_cc, value_dt
        )
        data_rs, dt_rs = self._resample(self.data)

        nc        = self.nchan
        dev       = self.device
        niter     = 0
        shift_idx = torch.arange(nc, dtype=torch.int, device=dev)
        w         = self.w0
        win_main  = self.win_main
        pick_dt   = torch.zeros(nc, device=dev)
        pick_cc   = torch.zeros(nc, device=dev)
        ones_buf  = None  # reused in loop

        while niter < self.max_niter:
            if niter == self.max_niter - 1:
                win_main = min(0.05, win_main)

            if niter == 0:
                pick_cc, _, pick_dt_refine = self._pick_win_maxabs(
                    moving_average(data_rs, self.refine_ma)
                    if self.mode == "pick" else data_rs,
                    dt_rs, win_main, update_vmin=False,
                )
            else:
                pick_cc, _, pick_dt_refine = self._pick_win_maxabs(
                    moving_average(
                        gather_roll(data_rs, shift_idx), self.refine_ma
                    ),
                    dt_rs, win_main, update_vmin=False,
                )
            if self.mode == "pick":
                isel = torch.where(
                    torch.abs(pick_cc)
                    > torch.quantile(torch.abs(pick_cc), 0.15)
                )[0]
                p_vec = (pick_dt + pick_dt_refine)[isel].double()
                w2 = w * w

                # ATA_it = w² * ATA_base + diag(isel indicator)
                ATA_it = w2 * ATA_base.clone()
                if ones_buf is None or len(ones_buf) != len(isel):
                    ones_buf = torch.ones(
                        len(isel), dtype=torch.float64, device=dev
                    )
                ATA_it.view(nc * nc).scatter_add_(
                    0, isel.long() * nc + isel.long(), ones_buf
                )

                # ATb_it = w² * ATb_base + P.T p
                ATb_it = w2 * ATb_base.clone()
                ATb_it.scatter_add_(0, isel.long(), p_vec)
            else:
                ATA_it = ATA_base
                ATb_it = ATb_base

            # Tikhonov regularization: prevents singularity when damp=0
            # or some channels have no good-CC pairs (all-zero ATA rows).
            ATA_it.diagonal().add_(1e-8)

            # GPU solve (LU; ATA is symmetric PD after regularization)
            try:
                x = torch.linalg.solve(ATA_it, ATb_it)
            except torch.cuda.OutOfMemoryError:
                # Fallback to CPU if GPU OOM
                x = torch.linalg.solve(ATA_it.cpu(), ATb_it.cpu()).to(ATA_it.device)

            pick_dt[:]   = x.float()
            shift_idx[:] = -torch.round(pick_dt / dt_rs).int()
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
