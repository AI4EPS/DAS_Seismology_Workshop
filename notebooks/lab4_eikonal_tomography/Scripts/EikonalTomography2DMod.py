# Operators to perform 3D Eikonal tomography
import pykonal
import numpy as np
from numba import jit
from joblib import Parallel, delayed
import psutil 
from scipy.interpolate import griddata
# Maximum number of cores that can be employed
if psutil.cpu_count(logical = True) != psutil.cpu_count(logical = False):
    Ncores = int(psutil.cpu_count(logical = True)*0.5) 
else:
    Ncores = psutil.cpu_count(logical = False)


# tqdm bar with parallel processing
import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

# Solver library
import pyOperator as pyOp
import pyVector
from numba import jit, prange


@jit(nopython=True, cache=True)
def extract_tt_2D_FWD(tt_2D, ch_x, ch_z, ox, oz, dx, dz):
    """Function to extract the traveltime at specific channel locations"""
    tt_ch = np.zeros(ch_x.shape)
    for ich in range(tt_ch.shape[0]):
        wx = (ch_x[ich] - ox) / dx
        wz = (ch_z[ich] - oz) / dz
        ix = int(wx)
        iz = int(wz)
        # Interpolation weights
        wx -= ix
        wz -= iz
        tt_ch[ich] += tt_2D[ix,iz] * (1.0 - wx)*(1.0 - wz) + tt_2D[ix+1,iz] * (wx)*(1.0 - wz) + tt_2D[ix,iz + 1] * (1.0 - wx)*(wz) + tt_2D[ix + 1,iz + 1] * (wx)*(wz) 
    return tt_ch

def compute_travel_time(vel, ishot, ox, oz, dx, dz, SouPos, RecPos, Acc_Inj, TTsrc=None, returnTT=True):
    """Function to compute traveltime in parallel"""
    velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
    velocity.min_coords = 0.0, ox, oz
    velocity.node_intervals = 1.0, dx, dz
    velocity.npts = 1, vel.shape[0], vel.shape[1]
    if SouPos.ndim == 2:
        # Single point source
        if Acc_Inj:
            # Set Eikonal solver
            solver_ek = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
            solver_ek.vv.min_coords = velocity.min_coords
            solver_ek.vv.node_intervals = velocity.node_intervals
            solver_ek.vv.npts = velocity.npts
            solver_ek.vv.values[:] = vel
            # Setting source position (ys,xs,zs)
            solver_ek.src_loc = [0.0, SouPos[ishot,0], SouPos[ishot,1]] 
        else:
            # Set Eikonal solver
            solver_ek = pykonal.EikonalSolver(coord_sys="cartesian")
            solver_ek.vv.min_coords = velocity.min_coords
            solver_ek.vv.node_intervals = velocity.node_intervals
            solver_ek.vv.npts = velocity.npts
            solver_ek.vv.values[:] = vel
            # Initial conditions
            solver_ek.tt.values[:] = np.inf
            solver_ek.known[:] = False
            solver_ek.unknown[:] = True
            eq_iz = int((SouPos[ishot,1]-oz)/dz + 0.5)
            eq_iy = 0
            eq_ix = int((SouPos[ishot,0]-ox)/dx + 0.5)
            src_idx = (eq_iy, eq_ix, eq_iz)
            solver_ek.tt.values[src_idx] = 0.0
            solver_ek.unknown[src_idx] = False
            solver_ek.trial.push(*src_idx)
    else:
        # Set Eikonal solver
        solver_ek = pykonal.EikonalSolver(coord_sys="cartesian")
        solver_ek.vv.min_coords = velocity.min_coords
        solver_ek.vv.node_intervals = velocity.node_intervals
        solver_ek.vv.npts = velocity.npts
        solver_ek.vv.values[:] = vel
        # Multiple source points
        npnt_src = SouPos.shape[2]
        for iPnt in range(npnt_src):
            eq_iz = int((SouPos[ishot,1,iPnt]-oz)/dz + 0.5)
            eq_iy = 0
            eq_ix = int((SouPos[ishot,0,iPnt]-ox)/dx + 0.5)
            src_idx = (eq_iy, eq_ix, eq_iz)
            solver_ek.tt.values[src_idx] = 0.0 if TTsrc is None else TTsrc[iPnt]
            solver_ek.unknown[src_idx] = False
            solver_ek.trial.push(*src_idx)
    # Solving Eikonal equation
    solver_ek.solve()
    # Find indices where values are np.inf
    TT_Field = np.copy(solver_ek.tt.values)
    traveltimes = extract_tt_2D_FWD(TT_Field[0,:,:], RecPos[:,0], RecPos[:,1], ox, oz, dx, dz)
    if returnTT:
        return traveltimes, TT_Field
    else:
        return traveltimes

class EikonalTT_2D(pyOp.Operator):

    def __init__(self, vel, tt_data, ox, oz,dx, dz, SouPos, RecPos, TTsrc=None, verbose=False, **kwargs):
        """2D Eikonal-equation traveltime prediction operator"""
        # Setting Domain and Range of the operator
        self.setDomainRange(vel, tt_data)
        # Setting acquisition geometry
        self.nSou = SouPos.shape[0]
        # Get velocity array
        velNd = vel.getNdArray()
        # Accurate injection of initial conditions
        self.Acc_Inj = kwargs.get("Acc_Inj", True)
        # Getting number of threads to run the modeling code
        self.nthrs = min(self.nSou, Ncores, kwargs.get("nthreads", Ncores))
        self.nRec = RecPos.shape[0]
        self.SouPos = SouPos.copy()
        self.RecPos = RecPos.copy()
        if TTsrc is not None:
            if len(TTsrc) != self.nSou:
                raise ValueError("Number of initial traveltime (len(TTsrc)=%s) inconsistent with number of sources (%s)"%(len(TTsrc),self.nSou))
        else:
            TTsrc = [None]*self.nSou
        self.TTsrc = TTsrc # Traveltime vector for distributed sources
        dataShape = tt_data.shape
        self.ox = ox
        self.oz = oz
        self.dx = dx
        self.dz = dz
        self.ncomp = vel.shape[0] # Number of velocities to use
        self.nx = vel.shape[1]
        self.nz = vel.shape[2]
        self.xAxis = np.linspace(ox, ox+(self.nx-1)*dx, self.nx)
        self.zAxis = np.linspace(oz, oz+(self.nz-1)*dz, self.nz)
        # Use smallest possible domain (Ginsu knives)
        self.ginsu = kwargs.get("ginsu", False)
        buffer = kwargs.get("buffer", 2.0) # By default 2.0 km
        self.bufferX = int(buffer/dx)
        self.bufferZ = int(buffer/dz)
        if dataShape[0] != self.nSou*self.ncomp:
            raise ValueError("Number of sources inconsistent with traveltime vector (data_shape[0])")
        if dataShape[1] != self.nRec:
            raise ValueError("Number of receivers inconsistent with traveltime vector (data_shape[1])")
        # List of traveltime maps to avoid double computation
        self.tt_maps = []
        self.allocateTT = kwargs.get("allocateTT", False)
        if self.allocateTT:
            for _ in range(self.nSou*self.ncomp):
                self.tt_maps.append(np.zeros_like(velNd[0,:,:]))
        # verbosity of the program
        self.verbose = verbose
    
    def forward(self, add, model, data):
        """Forward non-linear traveltime prediction"""
        self.checkDomainRange(model, data)
        if not add: 
            data.zero()
        dataNd = data.getNdArray()
        velNd = model.getNdArray()
        # Parallel modeling 
        # result = Parallel(n_jobs=self.nthrs, verbose=self.verbose, backend="multiprocessing")(delayed(self.compute_travel_time)(velNd, ishot) for ishot in range(self.nSou))
        for icomp in range(self.ncomp):
            if self.ginsu:
                minX = np.zeros(self.nSou, dtype=int)
                maxX = np.zeros(self.nSou, dtype=int)
                minZ = np.zeros(self.nSou, dtype=int)
                maxZ = np.zeros(self.nSou, dtype=int)
                for ishot in range(self.nSou):
                    minX[ishot] =  max(0, min(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].min())))-self.bufferX)
                    maxX[ishot] =  min(self.nx, max(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].max())))+self.bufferX)
                    minZ[ishot] =  max(0, min(np.argmin(abs(self.zAxis-self.SouPos[ishot,1])), np.argmin(abs(self.zAxis-self.RecPos[:,1].min())))-self.bufferZ)
                    maxZ[ishot] =  min(self.nz, max(np.argmin(abs(self.zAxis-self.SouPos[ishot,1])), np.argmin(abs(self.zAxis-self.RecPos[:,1].max())))+self.bufferZ)
                result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_time)(velNd[icomp,minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]], ishot, self.xAxis[minX[ishot]], self.zAxis[minZ[ishot]], self.dx, self.dz, self.SouPos, self.RecPos, self.Acc_Inj, self.TTsrc[ishot], self.allocateTT) for ishot in range(self.nSou))
            else:
                result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_time)(velNd[icomp,:,:], ishot, self.ox, self.oz, self.dx, self.dz, self.SouPos, self.RecPos, self.Acc_Inj, self.TTsrc[ishot], self.allocateTT) for ishot in range(self.nSou))
            for ishot in range(self.nSou):
                if self.allocateTT:
                    dataNd[ishot+icomp*self.nSou, :] += result[ishot][0]
                    if self.ginsu:
                        self.tt_maps[ishot+icomp*self.nSou][minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]] = result[ishot][1][0,:,:]
                    else:
                        self.tt_maps[ishot+icomp*self.nSou][:] = result[ishot][1][0,:,:]
                else:
                    dataNd[ishot+icomp*self.nSou, :] += result[ishot]
        return


#######################################################
# Linearized operators
#######################################################


# Eikonal tomography-related operator
def sorting2D(tt: np.ndarray, idx_l: np.ndarray, ordering="a") -> np.ndarray:
    idx1 = idx_l[:, 0]
    idx2 = idx_l[:, 1]
    idx = np.ravel_multi_index((idx1, idx2), tt.shape)
    if ordering == "a":
        sorted_indices = np.argsort(tt.ravel()[idx])
    elif ordering == "d":
        sorted_indices = np.argsort(-tt.ravel()[idx])
    else:
        raise ValueError("Unknonw ordering: %s! Provide a or d for ascending or descending" % ordering)
        # Sorted indices for entire array
    sorted_indices = idx[sorted_indices]
    # Sorting indices
    idx1, idx2 = np.unravel_index(sorted_indices, tt.shape)
    idx_sort = np.array([idx1, idx2], dtype=np.int64).T
    return idx_sort

@jit(nopython=True, cache=True)
def FMM_tt_lin_fwd2D(delta_v: np.ndarray, delta_tt: np.ndarray, vv: np.ndarray, tt: np.ndarray, tt_idx: np.ndarray, dx: float, dz: float):
    """Fast-marching method linearized forward"""
    nx = delta_v.shape[0]
    nz = delta_v.shape[1]
    ns = np.array([nx, nz])
    drxns = [-1, 1]
    dx_inv = 1.0 / dx
    dz_inv = 1.0 / dz
    ds_inv = np.array([dx_inv, dz_inv])
    
    # Shift variables
    order = np.zeros(2, dtype=np.int64)
    shift = np.zeros(2, dtype=np.int64)
    idrx = np.zeros(2, dtype=np.int64)
    fdt0 = np.zeros(2)
    
    # Scaling the velocity perturbation
    delta_v_scaled = - 2.0 * delta_v / (vv * vv * vv)
    
    # Looping over all indices to solve linear equations from increasing traveltime values
    for idx_t0 in tt_idx:
        # If T = 0 or v = 0, then assuming zero to avoid singularity
        if tt[idx_t0[0], idx_t0[1]] == 0.0 or vv[idx_t0[0], idx_t0[1]] == 0.0:
            continue
        
        # Looping over
        fdt0.fill(0.0)
        idrx.fill(0)
        for iax in range(2):
            # Loop over neighbourning points to find up-wind direction
            fdt = np.zeros(2)
            order.fill(0)
            shift.fill(0)
            for idx in range(2):
                shift[iax] = drxns[idx]
                nb = idx_t0[:] + shift[:]
                # If point is outside the domain skip it
                if np.any(nb < 0) or np.any(nb >= ns):
                    continue
                if vv[nb[0], nb[1]] > 0.0:
                    order[idx] = 1
                    fdt[idx] = drxns[idx] * (tt[nb[0], nb[1]] - tt[idx_t0[0], idx_t0[1]]) * ds_inv[iax]
                else:
                    order[idx] = 0
            # Selecting upwind derivative
            shift.fill(0)
            if fdt[0] > -fdt[1] and order[0] > 0:
                idrx[iax], shift[iax] = -1, -1
            elif fdt[0] <= -fdt[1] and order[1] > 0:
                idrx[iax], shift[iax] = 1, 1
            else:
                idrx[iax] = 0
            nb = idx_t0[:] + shift[:]
            # Computing t0 space derivative
            fdt0[iax] = idrx[iax] * (tt[nb[0], nb[1]] - tt[idx_t0[0], idx_t0[1]]) * ds_inv[iax] * ds_inv[iax]
        # Using single stencil along z direction to update value
        if tt[idx_t0[0] + idrx[0], idx_t0[1]] > tt[idx_t0[0], idx_t0[1]]:
            denom = - 2.0 * idrx[1] * fdt0[1]
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1]] += (- idrx[1] * 2.0 * fdt0[1] * delta_tt[
                    idx_t0[0], idx_t0[1] + idrx[1]] +
                                                   delta_v_scaled[idx_t0[0], idx_t0[1]]) / denom
        # Using single stencil along x direction to update value
        elif tt[idx_t0[0], idx_t0[1] + idrx[1]] > tt[idx_t0[0], idx_t0[1]]:
            denom = - 2.0 * idrx[0] * fdt0[0]
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[
                    idx_t0[0] + idrx[0], idx_t0[1]] +
                                                   delta_v_scaled[idx_t0[0], idx_t0[1]]) / denom
        else:
            denom = - 2.0 * (idrx[0] * fdt0[0] + idrx[1] * fdt0[1])
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[
                    idx_t0[0] + idrx[0], idx_t0[1]] +
                                                   - idrx[1] * 2.0 * fdt0[1] * delta_tt[
                                                       idx_t0[0], idx_t0[1] + idrx[1]] +
                                                   delta_v_scaled[idx_t0[0], idx_t0[1]]) / denom
    return


@jit(nopython=True, cache=True)
def select_upwind_der2D(tt: np.ndarray, idx_t0: np.ndarray, vv: np.ndarray, ds_inv, iax: int):
    """Find upwind derivative along iax"""
    nx = vv.shape[0]
    nz = vv.shape[1]
    ns = np.array([nx, nz])
    nb = np.zeros(2, dtype=np.int64)
    shift = np.zeros(2, dtype=np.int64)
    drxns = [-1, 1]
    fdt = np.zeros(2)
    order = np.zeros(2, dtype=np.int64)
    
    # Computing derivative for the neighboring points along iax
    for idx in range(2):
        shift[iax] = drxns[idx]
        nb[:] = idx_t0[:] + shift[:]
        # If point is outside the domain skip it
        if np.any(nb < 0) or np.any(nb >= ns):
            continue
        if vv[nb[0], nb[1]] > 0.0:
            order[idx] = 1
            fdt[idx] = drxns[idx] * (tt[nb[0], nb[1]] - tt[idx_t0[0], idx_t0[1]]) * ds_inv[iax]
        else:
            order[idx] = 0
    # Selecting upwind derivative
    if fdt[0] > -fdt[1] and order[0] > 0:
        fd, idrx = fdt[0], -1
    elif fdt[0] <= -fdt[1] and order[1] > 0:
        fd, idrx = fdt[1], 1
    else:
        fd, idrx = 0.0, 0
    return fd, idrx

# Adjoint operator
@jit(nopython=True, cache=True)
def FMM_tt_lin_adj2D(delta_v: np.ndarray, delta_tt: np.ndarray, vv: np.ndarray, tt: np.ndarray, tt_idx: np.ndarray, dx: float, dz: float):
    """Fast-marching method linearized forward"""
    nx = delta_v.shape[0]
    nz = delta_v.shape[1]
    ns = np.array([nx, nz])
    drxns = [-1, 1]
    dx_inv = 1.0 / dx
    dz_inv = 1.0 / dz
    ds_inv = np.array([dx_inv, dz_inv])
    
    # Internal variables
    order = np.zeros(2, dtype=np.int64)
    shift = np.zeros(2, dtype=np.int64)
    nbrs = np.zeros((4, 2), dtype=np.int64)
    fdt_nb = np.zeros(4)
    order_nb = np.zeros(4, dtype=np.int64)
    idrx_nb = np.zeros(4, dtype=np.int64)
    
    # Looping over all indices to solve linear equations from increasing traveltime values
    for idx_t0 in tt_idx:
        # If T = 0 or v = 0, then assuming zero to avoid singularity
        if tt[idx_t0[0], idx_t0[1]] == 0.0 or vv[idx_t0[0], idx_t0[1]] == 0.0:
            continue
        
        # Creating indices of neighbouring points
        # Order left/right bottom/top
        inbr = 0
        for iax in range(2):
            shift.fill(0)
            for idx in range(2):
                shift[iax] = drxns[idx]
                nbrs[inbr][:] = idx_t0[:] + shift[:]
                inbr += 1
        
        # Looping over neighbouring points
        fdt_nb.fill(0)
        idrx_nb.fill(0)
        for ib, nb in enumerate(nbrs):
            # Point outside of modeling domain
            if np.any(nb < 0) or np.any(nb >= ns):
                order_nb[ib] = 0
                continue
            # Point with lower traveltime compared to current point
            if tt[idx_t0[0], idx_t0[1]] > tt[nb[0], nb[1]]:
                order_nb[ib] = 0
                continue
            order_nb[ib] = 1
            # Getting derivative along given axis
            iax = 0 if ib in [0, 1] else 1
            fdt_nb[ib], idrx_nb[ib] = select_upwind_der2D(tt, nb, vv, ds_inv, iax)
            # Removing point if derivative at nb did not use idx_t0
            if ib in [0, 1]:
                # Checking x direction
                if idx_t0[0] != nb[0] + idrx_nb[ib]:
                    fdt_nb[ib], idrx_nb[ib] = 0.0, 0
            else:
                # Checking z direction
                if idx_t0[1] != nb[1] + idrx_nb[ib]:
                    fdt_nb[ib], idrx_nb[ib] = 0.0, 0
        
        # Updating delta_v according to stencil
        fdt_nb *= -idrx_nb
        fdt0 = 0.0
        fdt_nb[0] *= dx_inv
        fdt_nb[1] *= dx_inv
        fdt_nb[2] *= dz_inv
        fdt_nb[3] *= dz_inv
        
        if np.all(order_nb[:2]):
            fdt0, idrx0 = select_upwind_der2D(tt, idx_t0, vv, ds_inv, 1)
            fdt0 *= np.sign(idrx0) * dz_inv
        elif np.all(order_nb[2:]):
            fdt0, idrx0 = select_upwind_der2D(tt, idx_t0, vv, ds_inv, 0)
            fdt0 *= np.sign(idrx0) * dx_inv
        else:
            fdt0x, idrx0x = select_upwind_der2D(tt, idx_t0, vv, ds_inv, 0)
            fdt0z, idrx0z = select_upwind_der2D(tt, idx_t0, vv, ds_inv, 1)
            # Necessary to consider correct stencil central value
            if tt[idx_t0[0], idx_t0[1]] < tt[idx_t0[0] + idrx0x, idx_t0[1]]:
                fdt0x, idrx0x = 0.0, 0
            if tt[idx_t0[0], idx_t0[1]] < tt[idx_t0[0], idx_t0[1] + idrx0z]:
                fdt0z, idrx0z = 0.0, 0
            fdt0 = idrx0x * fdt0x * dx_inv + idrx0z * fdt0z * dz_inv
        
        # Update delta_v value
        delta_v[idx_t0[0], idx_t0[1]] -= (fdt_nb[0] * delta_v[idx_t0[0] - order_nb[0], idx_t0[1]]
                                          + fdt_nb[1] * delta_v[idx_t0[0] + order_nb[1], idx_t0[1]]
                                          + fdt_nb[2] * delta_v[idx_t0[0], idx_t0[1] - order_nb[2]]
                                          + fdt_nb[3] * delta_v[idx_t0[0], idx_t0[1] + order_nb[3]]
                                          - 0.5 * delta_tt[idx_t0[0], idx_t0[1]]) / fdt0
    
    # Scaling the velocity perturbation
    delta_v[:] = 2.0 * delta_v / (vv * vv * vv)
    
    return

def compute_travel_timeLin(vel, ishot, ox, oz, dx, dz, SouPos, Acc_Inj, TTsrc=None):
    """Function to compute traveltime in parallel"""
    velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
    velocity.min_coords = 0.0, ox, oz
    velocity.node_intervals = 1.0, dx, dz
    velocity.npts = 1, vel.shape[0], vel.shape[1]
    if SouPos.ndim == 2:
        # Single point source
        if Acc_Inj:
            # Set Eikonal solver
            solver_ek = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
            solver_ek.vv.min_coords = velocity.min_coords
            solver_ek.vv.node_intervals = velocity.node_intervals
            solver_ek.vv.npts = velocity.npts
            solver_ek.vv.values[:] = vel
            # Setting source position (ys,xs,zs)
            solver_ek.src_loc = [0.0, SouPos[ishot,0], SouPos[ishot,1]] 
        else:
            # Set Eikonal solver
            solver_ek = pykonal.EikonalSolver(coord_sys="cartesian")
            solver_ek.vv.min_coords = velocity.min_coords
            solver_ek.vv.node_intervals = velocity.node_intervals
            solver_ek.vv.npts = velocity.npts
            solver_ek.vv.values[:] = vel
            # Initial conditions
            solver_ek.tt.values[:] = np.inf
            solver_ek.known[:] = False
            solver_ek.unknown[:] = True
            eq_iz = int((SouPos[ishot,1]-oz)/dz + 0.5)
            eq_iy = 0
            eq_ix = int((SouPos[ishot,0]-ox)/dx + 0.5)
            src_idx = (eq_iy, eq_ix, eq_iz)
            solver_ek.tt.values[src_idx] = 0.0
            solver_ek.unknown[src_idx] = False
            solver_ek.trial.push(*src_idx)
    else:
        # Set Eikonal solver
        solver_ek = pykonal.EikonalSolver(coord_sys="cartesian")
        solver_ek.vv.min_coords = velocity.min_coords
        solver_ek.vv.node_intervals = velocity.node_intervals
        solver_ek.vv.npts = velocity.npts
        solver_ek.vv.values[:] = vel
        # Multiple source points
        npnt_src = SouPos.shape[2]
        for iPnt in range(npnt_src):
            eq_iz = int((SouPos[ishot,1,iPnt]-oz)/dz + 0.5)
            eq_iy = 0
            eq_ix = int((SouPos[ishot,0,iPnt]-ox)/dx + 0.5)
            src_idx = (eq_iy, eq_ix, eq_iz)
            solver_ek.tt.values[src_idx] = 0.0 if TTsrc is None else TTsrc[iPnt]
            solver_ek.unknown[src_idx] = False
            solver_ek.trial.push(*src_idx)
    # Solving Eikonal equation
    solver_ek.solve()
    return solver_ek.tt.values

def solve_linearized_fwd(vel0, delta_v, ishot, dx, dz, ox, oz, tt0, tt_idx, RecPos):
    """Function to solve linearized problem"""
    # Sorting traveltime in ascending order
    tt_idx = sorting2D(tt0, tt_idx)
    delta_tt = np.zeros_like(vel0)
    FMM_tt_lin_fwd2D(delta_v, delta_tt, vel0, tt0, tt_idx, dx, dz)
    data_tt_lin = extract_tt_2D_FWD(delta_tt, RecPos[:,0], RecPos[:,1], ox, oz, dx, dz)
    return data_tt_lin

def solve_linearized_adj(vel0, data, ishot, ox, oz, dx, dz, tt0, tt_idx, RecPos):
    delta_tt = np.zeros_like(vel0)
    delta_v = np.zeros_like(vel0)
    # Sorting traveltime in descending order
    tt_idx = sorting2D(tt0, tt_idx, ordering="d")
    # Injecting traveltime to correct grid positions
    for iRec in range(RecPos.shape[0]):
        wx = (RecPos[iRec, 0] - ox) / dx
        wz = (RecPos[iRec, 1] - oz) / dz
        ix = int(wx)
        iz = int(wz)
        # Interpolation weights
        wx -= ix
        wz -= iz
        delta_tt[ix,iz]       += data[ishot, iRec] * (1.0 - wx)*(1.0 - wz) 
        delta_tt[ix,iz+1]     += data[ishot, iRec] * (1.0 - wx)*(wz) 
        delta_tt[ix+1,iz+1]   += data[ishot, iRec] * (wx)*(wz) 
        delta_tt[ix+1,iz]     += data[ishot, iRec] * (wx)*(1.0 - wz)  
    FMM_tt_lin_adj2D(delta_v, delta_tt, vel0, tt0, tt_idx, dx, dz)
    return delta_v

class EikonalTT_lin_2D(pyOp.Operator):

    def __init__(self, vel, tt_data, ox, oz, dx, dz, SouPos, RecPos, TTsrc=None, tt_maps=None, verbose=False, **kwargs):
        """2D Eikonal-equation traveltime prediction operator"""
        # Setting Domain and Range of the operator
        self.setDomainRange(vel, tt_data)
        # Setting acquisition geometry
        self.nSou = SouPos.shape[0]
        self.nRec = RecPos.shape[0]
        self.SouPos = SouPos.copy()
        self.RecPos = RecPos.copy()
        # Get velocity array
        velNd = vel.getNdArray()
        # Getting number of threads to run the modeling code
        self.nthrs = min(self.nSou, Ncores, kwargs.get("nthreads", Ncores))
        if TTsrc is not None:
            if len(TTsrc) != self.nSou:
                raise ValueError("Number of initial traveltime (len(TTsrc)=%s) inconsistent with number of sources (%s)"%(len(TTsrc),self.nSou))
        else:
            TTsrc = [None]*self.nSou
        # Accurate injection of initial conditions
        self.Acc_Inj = kwargs.get("Acc_Inj", False)
        self.TTsrc = TTsrc # Traveltime vector for distributed sources
        dataShape = tt_data.shape
        self.ox = ox
        self.oz = oz
        self.dx = dx
        self.dz = dz
        self.ncomp = vel.shape[0]
        self.nx = vel.shape[1]
        self.nz = vel.shape[2]
        # Use smallest possible domain (Ginsu knives)
        self.ginsu = kwargs.get("ginsu", False)
        buffer = kwargs.get("buffer", 2.0) # By default 2.0 km
        self.bufferX = int(buffer/dx)
        self.bufferZ = int(buffer/dz)
        self.xAxis = np.linspace(ox, ox+(self.nx-1)*dx, self.nx)
        self.zAxis = np.linspace(oz, oz+(self.nz-1)*dz, self.nz)
        if dataShape[0] != self.nSou*self.ncomp:
            raise ValueError("Number of sources inconsistent with traveltime vector (data_shape[0])")
        if dataShape[1] != self.nRec:
            raise ValueError("Number of receivers inconsistent with traveltime vector (data_shape[1])")
        # Internal velocity model
        self.vel0 = vel.clone()
        # Verbosity level
        self.verbose = verbose
        # General unsorted traveltime indices
        idx_1d = np.arange(velNd[0,:,:].size)
        idx,idz = np.unravel_index(idx_1d, velNd[0,:,:].shape)
        self.tt_idx = np.array([idx,idz]).T
        # Traveltime maps
        if tt_maps is None:
            self.tt_maps = []
            for _ in range(self.nSou*self.ncomp):
                self.tt_maps.append(np.zeros_like(velNd[0,:,:]))
        else:
            self.tt_maps = tt_maps
        # Variable for updating the traveltimes
        self.TT_update = True

    def reset_tt_maps(self):
        """Function to zero-out tt_maps variable to recompute traveltime maps"""
        for ishot in range(self.nSou*self.ncomp):
            self.tt_maps[ishot].fill(0.0)
      
    def forward(self, add, model, data):
        """Forward linearized traveltime prediction"""
        self.checkDomainRange(model, data)
        if not add: 
            data.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()
        vel0Nd = self.vel0.getNdArray()
        ###################################
        # Computing background traveltime #
        ###################################
        # if np.any([not np.any(self.tt_maps[ishot]) for ishot in range(self.nSou)]):
        if self.TT_update:
            for icomp in range(self.ncomp):
                if self.ginsu:
                    minX = np.zeros(self.nSou, dtype=int)
                    maxX = np.zeros(self.nSou, dtype=int)
                    minZ = np.zeros(self.nSou, dtype=int)
                    maxZ = np.zeros(self.nSou, dtype=int)
                    for ishot in range(self.nSou):
                        minX[ishot] =  max(0, min(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].min())))-self.bufferX)
                        maxX[ishot] =  min(self.nx, max(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].max())))+self.bufferX)
                        minZ[ishot] =  max(0, min(np.argmin(abs(self.zAxis-self.SouPos[ishot,1])), np.argmin(abs(self.zAxis-self.RecPos[:,1].min())))-self.bufferZ)
                        maxZ[ishot] =  min(self.nz, max(np.argmin(abs(self.zAxis-self.SouPos[ishot,1])), np.argmin(abs(self.zAxis-self.RecPos[:,1].max())))+self.bufferZ)
                    result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_timeLin)(vel0Nd[icomp,minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]], ishot, self.xAxis[minX[ishot]], self.zAxis[minZ[ishot]], self.dx, self.dz, self.SouPos, self.Acc_Inj, self.TTsrc[ishot]) for ishot in range(self.nSou))
                else:
                    result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_timeLin)(vel0Nd[icomp,:,:], ishot, self.ox, self.oz, self.dx, self.dz, self.SouPos, self.Acc_Inj, self.TTsrc[ishot]) for ishot in range(self.nSou))
                for ishot in range(self.nSou):
                    if self.ginsu:
                        self.tt_maps[ishot+icomp*self.nSou][minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]] = result[ishot][0,:,:]
                    else:
                        self.tt_maps[ishot+icomp*self.nSou][:] = result[ishot][0,:,:]
            self.TT_update = False
        ###################################
        # Computing linearized traveltime #
        ###################################
        for icomp in range(self.ncomp):
            result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(solve_linearized_fwd)(vel0Nd[icomp,:,:], modelNd[icomp,:,:], ishot, self.dx, self.dz, self.ox, self.oz, self.tt_maps[ishot+icomp*self.nSou], self.tt_idx, self.RecPos) for ishot in range(self.nSou))
            for ishot in range(self.nSou):
                dataNd[ishot+icomp*self.nSou,:] += result[ishot]
        return
    
    def adjoint(self, add, model, data):
        """Adjoint linearized traveltime prediction"""
        self.checkDomainRange(model, data)
        if not add: 
            model.zero()
        dataNd = data.getNdArray()
        modelNd = model.getNdArray()
        vel0Nd = self.vel0.getNdArray()
        ###################################
        # Computing background traveltime #
        ###################################
        # if np.any([not np.any(self.tt_maps[ishot]) for ishot in range(self.nSou)]):
        if self.TT_update:
            for icomp in range(self.ncomp):
                if self.ginsu:
                    minX = np.zeros(self.nSou, dtype=int)
                    maxX = np.zeros(self.nSou, dtype=int)
                    minZ = np.zeros(self.nSou, dtype=int)
                    maxZ = np.zeros(self.nSou, dtype=int)
                    for ishot in range(self.nSou):
                        minX[ishot] =  max(0, min(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].min())))-self.bufferX)
                        maxX[ishot] =  min(self.nx, max(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].max())))+self.bufferX)
                        minZ[ishot] =  max(0, min(np.argmin(abs(self.zAxis-self.SouPos[ishot,1])), np.argmin(abs(self.zAxis-self.RecPos[:,1].min())))-self.bufferZ)
                        maxZ[ishot] =  min(self.nz, max(np.argmin(abs(self.zAxis-self.SouPos[ishot,1])), np.argmin(abs(self.zAxis-self.RecPos[:,1].max())))+self.bufferZ)
                    result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_timeLin)(vel0Nd[icomp,minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]], ishot, self.xAxis[minX[ishot]], self.zAxis[minZ[ishot]], self.dy, self.dx, self.dz, self.SouPos, self.Acc_Inj, self.TTsrc[ishot]) for ishot in range(self.nSou))
                else:
                    result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_timeLin)(vel0Nd[icomp,:,:], ishot, self.ox, self.oz, self.dx, self.dz, self.SouPos, self.Acc_Inj, self.TTsrc[ishot]) for ishot in range(self.nSou))
                for ishot in range(self.nSou):
                    if self.ginsu:
                        self.tt_maps[ishot+icomp*self.nSou][minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]] = result[ishot][0,:,:]
                    else:
                        self.tt_maps[ishot+icomp*self.nSou][:] = result[ishot][0,:,:]
            self.TT_update = False
        ###################################
        # Computing velocity perturbation #
        ###################################
        for icomp in range(self.ncomp):
            result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(solve_linearized_adj)(vel0Nd[icomp,:,:], dataNd[icomp*self.nSou:icomp*self.nSou+self.nSou,:], ishot, self.ox, self.oz, self.dx, self.dz, self.tt_maps[ishot+icomp*self.nSou], self.tt_idx, self.RecPos) for ishot in range(self.nSou))
            for ishot in range(self.nSou):
                modelNd[icomp,:,:] += result[ishot]
        return
    
    def set_vel(self, vel):
        """Function to set background velocity model"""
        if self.vel0.isDifferent(vel):
            self.TT_update = True
        self.vel0.copy(vel)