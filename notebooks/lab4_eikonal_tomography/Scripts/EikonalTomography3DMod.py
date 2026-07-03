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
import sys
sys.path.insert(0,"python-solver/GenericSolver/python/")

import pyOperator as pyOp
import pyVector
from numba import jit, prange

@jit(nopython=True, cache=True)
def extract_tt_3D_FWD(tt_3D, ch_y, ch_x, ch_z, oy, ox, oz, dy, dx, dz):
    """Function to extract the traveltime at specific channel locations"""
    tt_ch = np.zeros(ch_x.shape)
    # ny = tt_3D.shape[0]
    # nx = tt_3D.shape[1]
    # nz = tt_3D.shape[2]
    for ich in range(tt_ch.shape[0]):
        wy = (ch_y[ich] - oy) / dy
        wx = (ch_x[ich] - ox) / dx
        wz = (ch_z[ich] - oz) / dz
        iy = int(wy)
        ix = int(wx)
        iz = int(wz)
        # Interpolation weights
        wy -= iy
        wx -= ix
        wz -= iz
        tt_ch[ich] += tt_3D[iy,ix,iz] * (1.0 - wy)*(1.0 - wx)*(1.0 - wz) + tt_3D[iy,ix,iz+1] * (1.0 - wy)*(1.0 - wx)*(wz) + tt_3D[iy,ix+1,iz] * (1.0 - wy)*(wx)*(1.0 - wz)  + tt_3D[iy+1,ix,iz] * (wy)*(1.0 - wx)*(1.0 - wz)  + tt_3D[iy,ix+1,iz+1] * (1.0 - wy)*(wx)*(wz)  + tt_3D[iy+1,ix,iz+1] * (wy)*(1.0 - wx)*(wz)  + tt_3D[iy+1,ix+1,iz] * (wy)*(wx)*(1.0 - wz)  + tt_3D[iy+1,ix+1,iz+1] * (wy)*(wx)*(wz) 
    return tt_ch

def compute_travel_time(vel, ishot, oy, ox, oz, dy, dx, dz, SouPos, RecPos, Acc_Inj, TTsrc=None, returnTT=True, computeRays=False):
    """Function to compute traveltime in parallel"""
    velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
    velocity.min_coords = oy, ox, oz
    velocity.node_intervals = dy, dx, dz
    velocity.npts = vel.shape[0], vel.shape[1], vel.shape[2]
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
            solver_ek.src_loc = [SouPos[ishot,1],SouPos[ishot,0],SouPos[ishot,2]] 
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
            eq_iz = int((SouPos[ishot,2]-oz)/dz + 0.5)
            eq_iy = int((SouPos[ishot,1]-oy)/dy + 0.5)
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
            eq_iz = int((SouPos[ishot,2,iPnt]-oz)/dz + 0.5)
            eq_iy = int((SouPos[ishot,1,iPnt]-oy)/dy + 0.5)
            eq_ix = int((SouPos[ishot,0,iPnt]-ox)/dx + 0.5)
            src_idx = (eq_iy, eq_ix, eq_iz)
            solver_ek.tt.values[src_idx] = 0.0 if TTsrc is None else TTsrc[iPnt]
            solver_ek.unknown[src_idx] = False
            solver_ek.trial.push(*src_idx)
    # Solving Eikonal equation
    solver_ek.solve()
    # Find indices where values are np.inf
    TT_Field = np.copy(solver_ek.tt.values)
    inf_indices = np.where(np.isinf(TT_Field[:,:,:]))
    # Loop through each inf value and replace it with linear interpolation
    for i in range(len(inf_indices[0])):
        idx0, idx1, idx2 = inf_indices[0][i], inf_indices[1][i], inf_indices[2][i]
        # Get neighboring values for interpolation
        values = TT_Field[max(0, idx0-1):idx0+2, max(0, idx1-1):idx1+2, max(0, idx2-1):idx2+2]
        mask_inf = np.isinf(values)
        # Get coordinates of non-inf values
        coords = np.array(np.where(~mask_inf)).T
        # Values of non-inf points
        values = values[~mask_inf]
        # Coordinates of points to interpolate
        coords_inf = np.array(np.where(mask_inf)).T
        # Linear interpolation
        interpolated_values = griddata(coords, values, coords_inf, method='linear')
        # Fill the original array with the interpolated values
        TT_Field[max(0, idx0-1):idx0+2, max(0, idx1-1):idx1+2, max(0, idx2-1):idx2+2][mask_inf] = interpolated_values
    traveltimes = extract_tt_3D_FWD(TT_Field, RecPos[:,1], RecPos[:,0], RecPos[:,2], oy, ox, oz, dy, dx, dz)

    if computeRays:
        rays = []
        for iRec in range(RecPos.shape[0]):
            ray = solver_ek.trace_ray(np.array((RecPos[iRec,1], RecPos[iRec,0], RecPos[iRec,2])))
            rays.append(ray)

    traveltimes = extract_tt_3D_FWD(solver_ek.tt.values, RecPos[:,1], RecPos[:,0], RecPos[:,2], oy, ox, oz, dy, dx, dz)
    # Returning variables
    tt_values = TT_Field if returnTT else None
    ray_paths = rays if computeRays else None
    result = traveltimes, tt_values, ray_paths
    return result

class EikonalTT_3D(pyOp.Operator):

    def __init__(self, vel, tt_data, oy, ox, oz, dy ,dx, dz, SouPos, RecPos, TTsrc=None, verbose=False, **kwargs):
        """3D Eikonal-equation traveltime prediction operator"""
        # Setting Domain and Range of the operator
        self.setDomainRange(vel, tt_data)
        # Setting acquisition geometry
        self.nSou = SouPos.shape[0]
        # Get velocity array
        velNd = vel.getNdArray()
        # Accurate injection of initial conditions
        self.Acc_Inj = kwargs.get("Acc_Inj", True)
         # Compute rays
        self.computeRays = kwargs.get("computeRays", False)
        self.ray_paths = None
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
        self.oy = oy
        self.ox = ox
        self.oz = oz
        self.dy = dy
        self.dx = dx
        self.dz = dz
        self.ncomp = vel.shape[0] # Number of velocities to use
        self.ny = vel.shape[1]
        self.nx = vel.shape[2]
        self.nz = vel.shape[3]
        self.xAxis = np.linspace(ox, ox+(self.nx-1)*dx, self.nx)
        self.yAxis = np.linspace(oy, oy+(self.ny-1)*dy, self.ny)
        self.zAxis = np.linspace(oz, oz+(self.nz-1)*dz, self.nz)
        # Use smallest possible domain (Ginsu knives)
        self.ginsu = kwargs.get("ginsu", False)
        buffer = kwargs.get("buffer", 2.0) # By default 2.0 km
        self.bufferX = int(buffer/dx)
        self.bufferY = int(buffer/dy)
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
                self.tt_maps.append(np.zeros_like(velNd[0,:,:,:]))
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
        if self.computeRays:
            self.ray_paths = []
        for icomp in range(self.ncomp):
            if self.ginsu:
                minX = np.zeros(self.nSou, dtype=int)
                maxX = np.zeros(self.nSou, dtype=int)
                minY = np.zeros(self.nSou, dtype=int)
                maxY = np.zeros(self.nSou, dtype=int)
                minZ = np.zeros(self.nSou, dtype=int)
                maxZ = np.zeros(self.nSou, dtype=int)
                for ishot in range(self.nSou):
                    minX[ishot] =  max(0, min(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].min())))-self.bufferX)
                    maxX[ishot] =  min(self.nx, max(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].max())))+self.bufferX)
                    minY[ishot] =  max(0, min(np.argmin(abs(self.yAxis-self.SouPos[ishot,1])), np.argmin(abs(self.yAxis-self.RecPos[:,1].min())))-self.bufferY)
                    maxY[ishot] =  min(self.ny, max(np.argmin(abs(self.yAxis-self.SouPos[ishot,1])), np.argmin(abs(self.yAxis-self.RecPos[:,1].max())))+self.bufferY)
                    minZ[ishot] =  max(0, min(np.argmin(abs(self.zAxis-self.SouPos[ishot,2])), np.argmin(abs(self.zAxis-self.RecPos[:,2].min())))-self.bufferZ)
                    maxZ[ishot] =  min(self.nz, max(np.argmin(abs(self.zAxis-self.SouPos[ishot,2])), np.argmin(abs(self.zAxis-self.RecPos[:,2].max())))+self.bufferZ)
                result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_time)(velNd[icomp,minY[ishot]:maxY[ishot],minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]], ishot, self.yAxis[minY[ishot]], self.xAxis[minX[ishot]], self.zAxis[minZ[ishot]], self.dy, self.dx, self.dz, self.SouPos, self.RecPos, self.Acc_Inj, self.TTsrc[ishot], self.allocateTT, self.computeRays) for ishot in range(self.nSou))
            else:
                result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_time)(velNd[icomp,:,:,:], ishot, self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, self.SouPos, self.RecPos, self.Acc_Inj, self.TTsrc[ishot], self.allocateTT, self.computeRays) for ishot in range(self.nSou))
            for ishot in range(self.nSou):
                if self.allocateTT:
                    dataNd[ishot+icomp*self.nSou, :] += result[ishot][0]
                    if self.ginsu:
                        self.tt_maps[ishot+icomp*self.nSou][minY[ishot]:maxY[ishot],minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]] = result[ishot][1]
                    else:
                        self.tt_maps[ishot+icomp*self.nSou][:] = result[ishot][1]
                else:
                    dataNd[ishot+icomp*self.nSou, :] += result[ishot][0]
                if self.computeRays:
                   self.ray_paths.append(result[ishot][2]) 
        return


#######################################################
# Linearized operators
#######################################################


# Eikonal tomography-related operator
def sorting3D(tt, idx_l, ordering="a"):
    idx1 = idx_l[:,0]
    idx2 = idx_l[:,1]
    idx3 = idx_l[:,2]
    idx = np.ravel_multi_index((idx1, idx2, idx3), tt.shape)
    if ordering == "a":
        sorted_indices = np.argsort(tt.ravel()[idx])
    elif ordering == "d":
        sorted_indices = np.argsort(-tt.ravel()[idx])
    else:
        raise ValueError("Unknonw ordering: %s! Provide a or d for ascending or descending" % ordering)
        
    # Sorted indices for entire array
    sorted_indices = idx[sorted_indices]
    
    # Sorting indices
    idx1, idx2, idx3 = np.unravel_index(sorted_indices, tt.shape)
    idx_sort = np.array([idx1,idx2,idx3], dtype=np.int64).T
    # idx_sort = [[iy,ix,iz] for iy, ix, iz in zip(idx1, idx2, idx3)]
    return idx_sort

@jit(nopython=True, cache=True)
def FMM_tt_lin_fwd3D(delta_v, delta_tt, vv, tt, tt_idx, dy, dx, dz):
    """Fast-marching method linearized forward"""
    ny = delta_v.shape[0]
    nx = delta_v.shape[1]
    nz = delta_v.shape[2]
    drxns = [-1, 1]
    dy_inv = 1.0 / dy
    dx_inv = 1.0 / dx
    dz_inv = 1.0 / dz
    ds_inv = np.array([dy_inv, dx_inv, dz_inv])
    
    # Shift variables
    order = np.zeros(2, dtype=np.int64)
    shift = np.zeros(3, dtype=np.int64)
    idrx = np.zeros(3, dtype=np.int64)
    fdt0 = np.zeros(3)
    
    # Scaling the velocity perturbation
    delta_v_scaled = - 2.0 * delta_v / (vv * vv * vv)
    
    # Looping over all indices to solve linear equations from increasing traveltime values
    for idx_t0 in tt_idx:
        tt0 = tt[idx_t0[0], idx_t0[1], idx_t0[2]]
        # If T = 0 or v = 0, then assuming zero to avoid singularity
        if tt0 == 0.0 or vv[idx_t0[0], idx_t0[1], idx_t0[2]] == 0.0:
            continue

        fdt0.fill(0.0)
        idrx.fill(0)
        for iax in range(3):
            # Loop over neighbourning points to find up-wind direction
            fdt = np.zeros(2)
            order.fill(0)
            shift.fill(0)
            for idx in range(2):
                shift[iax] = drxns[idx]
                nb = idx_t0[:] + shift[:]
                # If point is outside the domain skip it
                # if np.any(nb < 0) or np.any(nb >= ns):
                if nb[0] < 0 or nb[1] < 0 or nb[2] < 0 or nb[0] >= ny or nb[1] >= nx or nb[2] >= nz:
                    continue
                if vv[nb[0], nb[1], nb[2]] > 0.0:
                    order[idx] = 1
                    fdt[idx] = drxns[idx] * (tt[nb[0], nb[1], nb[2]] - tt0) * ds_inv[iax]
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
            fdt0[iax] = idrx[iax] * (tt[nb[0], nb[1], nb[2]] - tt0) * ds_inv[iax] * ds_inv[iax]

        # Checking traveltime values of neighbourning points
        tty = tt[idx_t0[0] + idrx[0], idx_t0[1], idx_t0[2]]
        ttx = tt[idx_t0[0], idx_t0[1] + idrx[1], idx_t0[2]]
        ttz = tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx[2]]
                                                            
        # Using single stencil along z direction to update value
        if ttx > tt0 and tty > tt0:
            denom = - 2.0 * idrx[2] * fdt0[2]
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[2] * 2.0 * fdt0[2] * delta_tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx[2]] 
                                                              + delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
        # Using single stencil along x direction to update value
        elif tty > tt0 and ttz > tt0:
            denom = - 2.0 * idrx[1] * fdt0[1]
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[1] * 2.0 * fdt0[1] * delta_tt[idx_t0[0], idx_t0[1] + idrx[1], idx_t0[2]]
                                                              + delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
        # Using single stencil along y direction to update value
        elif ttx > tt0 and ttz > tt0:
            denom = - 2.0 * idrx[0] * fdt0[0]
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[idx_t0[0] + idrx[0], idx_t0[1], idx_t0[2]]
                                                              + delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
        # Using single stencil along x-y direction to update value
        elif ttz > tt0:
            denom = - 2.0 * (idrx[0] * fdt0[0] + idrx[1] * fdt0[1])
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[idx_t0[0] + idrx[0], idx_t0[1], idx_t0[2]] +
                                                              - idrx[1] * 2.0 * fdt0[1] * delta_tt[idx_t0[0], idx_t0[1] + idrx[1], idx_t0[2]] +
                                                              delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
        # Using single stencil along x-z direction to update value
        elif tty > tt0:
            denom = - 2.0 * (idrx[1] * fdt0[1] + idrx[2] * fdt0[2])
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[1] * 2.0 * fdt0[1] * delta_tt[idx_t0[0], idx_t0[1] + idrx[1], idx_t0[2]] +
                                                              - idrx[2] * 2.0 * fdt0[2] * delta_tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx[2]] +
                                                              delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
        # Using single stencil along y-z direction to update value
        elif ttx > tt0:
            denom = - 2.0 * (idrx[0] * fdt0[0] + idrx[2] * fdt0[2])
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[idx_t0[0] + idrx[0], idx_t0[1], idx_t0[2]] +
                                                              - idrx[2] * 2.0 * fdt0[2] * delta_tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx[2]] +
                                                              delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
        else:
            denom = - 2.0 * (idrx[0] * fdt0[0] + idrx[1] * fdt0[1] + idrx[2] * fdt0[2])
            if abs(denom) > 0.0:
                delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]] += (- idrx[0] * 2.0 * fdt0[0] * delta_tt[idx_t0[0] + idrx[0], idx_t0[1], idx_t0[2]] +
                                                              - idrx[1] * 2.0 * fdt0[1] * delta_tt[idx_t0[0], idx_t0[1] + idrx[1], idx_t0[2]] +
                                                              - idrx[2] * 2.0 * fdt0[2] * delta_tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx[2]] +
                                                              delta_v_scaled[idx_t0[0], idx_t0[1], idx_t0[2]]) / denom
    return


@jit(nopython=True, cache=True)
def select_upwind_der3D(tt, idx_t0, vv, ds_inv, iax):
    """Find upwind derivative along iax"""
    ny = vv.shape[0]
    nx = vv.shape[1]
    nz = vv.shape[2]
    nb = np.zeros(3, dtype=np.int64)
    shift = np.zeros(3, dtype=np.int64)
    drxns = [-1, 1]
    fdt = np.zeros(2)
    order = np.zeros(2, dtype=np.int64)
    
    # Computing derivative for the neighboring points along iax
    for idx in range(2):
        shift[iax] = drxns[idx]
        nb[:] = idx_t0[:] + shift[:]
        # If point is outside the domain skip it
        # if np.any(nb < 0) or np.any(nb >= ns):
        if nb[0] < 0 or nb[1] < 0 or nb[2] < 0 or nb[0] >= ny or nb[1] >= nx or nb[2] >= nz:
            continue
        if vv[nb[0], nb[1], nb[2]] > 0.0:
            order[idx] = 1
            fdt[idx] = drxns[idx] * (tt[nb[0], nb[1], nb[2]] - tt[idx_t0[0], idx_t0[1], idx_t0[2]]) * ds_inv[iax]
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
def FMM_tt_lin_adj3D(delta_v, delta_tt, vv, tt, tt_idx, dy, dx, dz):
    """Fast-marching method linearized forward"""
    ny = delta_v.shape[0]
    nx = delta_v.shape[1]
    nz = delta_v.shape[2]
    drxns = [-1, 1]
    dy_inv = 1.0 / dy
    dx_inv = 1.0 / dx
    dz_inv = 1.0 / dz
    ds_inv = np.array([dy_inv, dx_inv, dz_inv])
    
    # Internal variables
    shift = np.zeros(3, dtype=np.int64)
    nbrs = np.zeros((6,3), dtype=np.int64)
    fdt_nb = np.zeros(6)
    order_nb = np.zeros(6, dtype=np.int64)
    idrx_nb = np.zeros(6, dtype=np.int64)
    
    # Looping over all indices to solve linear equations from increasing traveltime values
    for kk in range(tt_idx.shape[0]):
        idx_t0 = tt_idx[kk]
        tt0 =  tt[idx_t0[0], idx_t0[1], idx_t0[2]] 
        # If T = 0 or v = 0, then assuming zero to avoid singularity
        if tt0 == 0.0 or vv[idx_t0[0], idx_t0[1], idx_t0[2]] == 0.0:
            continue
        
        # Creating indices of neighbouring points
        # Order left/right bottom/top
        inbr = 0
        for iax in range(3):
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
            if nb[0] < 0 or nb[1] < 0 or nb[2] < 0 or nb[0] >= ny or nb[1] >= nx or nb[2] >= nz:
                order_nb[ib] = 0
                continue
            # Point with lower traveltime compared to current point
            if tt0 > tt[nb[0], nb[1], nb[2]]:
                order_nb[ib] = 0
                continue
            order_nb[ib] = 1
            # Getting derivative along given axis
            if ib in [0,1]:
                iax = 0
            elif ib in [2,3]:
                iax = 1
            elif ib in [4,5]:
                iax = 2
            fdt_nb[ib], idrx_nb[ib] = select_upwind_der3D(tt, nb, vv, ds_inv, iax)
            # Removing point if derivative at nb did not use idx_t0
            if ib in [0,1]:
                # Checking y direction
                if idx_t0[0] != nb[0] + idrx_nb[ib]:
                    fdt_nb[ib], idrx_nb[ib] = 0.0, 0
            elif ib in [2,3]:
                # Checking x direction
                if idx_t0[1] != nb[1] + idrx_nb[ib]:
                    fdt_nb[ib], idrx_nb[ib] = 0.0, 0
            else:
                # Checking z direction
                if idx_t0[2] != nb[2] + idrx_nb[ib]:
                    fdt_nb[ib], idrx_nb[ib] = 0.0, 0
        
        # Updating delta_v according to stencil
        fdt_nb *= -idrx_nb
        fdt0 = 0.0
        fdt_nb[0] *= dy_inv
        fdt_nb[1] *= dy_inv
        fdt_nb[2] *= dx_inv
        fdt_nb[3] *= dx_inv
        fdt_nb[4] *= dz_inv
        fdt_nb[5] *= dz_inv

        # Only z
        if order_nb[0] > 0 and order_nb[1] > 0 and order_nb[2] > 0 and order_nb[3] > 0:
            fdt0, idrx0 = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 2)
            fdt0 *= np.sign(idrx0) * dz_inv
        # Only x
        elif order_nb[0] > 0 and order_nb[1] > 0 and order_nb[4] > 0 and order_nb[5] > 0:
            fdt0, idrx0 = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 1)
            fdt0 *= np.sign(idrx0) * dx_inv
        # Only y
        elif order_nb[2] > 0 and order_nb[3] > 0 and order_nb[4] > 0 and order_nb[5] > 0:
            fdt0, idrx0 = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 0)
            fdt0 *= np.sign(idrx0) * dy_inv
        # Only x-y
        elif order_nb[4] > 0 and order_nb[5] > 0:
            fdt0y, idrx0y = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 0)
            fdt0x, idrx0x = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 1)
            # Necessary to consider correct stencil central value
            if tt0 < tt[idx_t0[0] + idrx0y, idx_t0[1], idx_t0[2]]: 
                fdt0y, idrx0y = 0.0, 0
            if tt0 < tt[idx_t0[0], idx_t0[1] + idrx0x, idx_t0[2]]: 
                fdt0x, idrx0x = 0.0, 0
            fdt0 = idrx0y * fdt0y * dy_inv + idrx0x * fdt0x * dx_inv
        # Only x-z
        elif order_nb[0] > 0 and order_nb[1] > 0:
            fdt0x, idrx0x = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 1)
            fdt0z, idrx0z = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 2)
            # Necessary to consider correct stencil central value
            if tt0 < tt[idx_t0[0], idx_t0[1] + idrx0x, idx_t0[2]]: 
                fdt0x, idrx0x = 0.0, 0
            if tt0 < tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx0z]: 
                fdt0z, idrx0z = 0.0, 0
            fdt0 = idrx0x * fdt0x * dx_inv + idrx0z * fdt0z * dz_inv
        # Only y-z
        elif order_nb[2] > 0 and order_nb[3] > 0:
            fdt0y, idrx0y = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 0)
            fdt0z, idrx0z = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 2)
            # Necessary to consider correct stencil central value
            if tt0 < tt[idx_t0[0] + idrx0y, idx_t0[1], idx_t0[2]]: 
                fdt0y, idrx0y = 0.0, 0
            if tt0 < tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx0z]: 
                fdt0z, idrx0z = 0.0, 0
            fdt0 = idrx0y * fdt0y * dy_inv + idrx0z * fdt0z * dz_inv
        else:
            fdt0y, idrx0y = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 0)
            fdt0x, idrx0x = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 1)
            fdt0z, idrx0z = select_upwind_der3D(tt, idx_t0, vv, ds_inv, 2)
            # Necessary to consider correct stencil central value
            if tt0 < tt[idx_t0[0] + idrx0y, idx_t0[1], idx_t0[2]]: 
                fdt0y, idrx0y = 0.0, 0
            if tt0 < tt[idx_t0[0], idx_t0[1] + idrx0x, idx_t0[2]]: 
                fdt0x, idrx0x = 0.0, 0
            if tt0 < tt[idx_t0[0], idx_t0[1], idx_t0[2] + idrx0z]: 
                fdt0z, idrx0z = 0.0, 0
            fdt0 = idrx0y * fdt0y * dy_inv + idrx0x * fdt0x * dx_inv + idrx0z * fdt0z * dz_inv
        
        # Update delta_v value
        if abs(fdt0) > 0.0:
            delta_v[idx_t0[0], idx_t0[1], idx_t0[2]] -= (  fdt_nb[0] * delta_v[idx_t0[0]-order_nb[0], idx_t0[1], idx_t0[2]] 
                                                        + fdt_nb[1] * delta_v[idx_t0[0]+order_nb[1], idx_t0[1], idx_t0[2]] 
                                                        + fdt_nb[2] * delta_v[idx_t0[0], idx_t0[1]-order_nb[2], idx_t0[2]] 
                                                        + fdt_nb[3] * delta_v[idx_t0[0], idx_t0[1]+order_nb[3], idx_t0[2]] 
                                                        + fdt_nb[4] * delta_v[idx_t0[0], idx_t0[1], idx_t0[2]-order_nb[4]] 
                                                        + fdt_nb[5] * delta_v[idx_t0[0], idx_t0[1], idx_t0[2]+order_nb[5]] 
                                                        - 0.5 * delta_tt[idx_t0[0], idx_t0[1], idx_t0[2]]) / fdt0
    
    # Scaling the velocity perturbation
    delta_v[:] = 2.0 * delta_v / (vv * vv * vv)
            
    return

def compute_travel_timeLin(vel, ishot, oy, ox, oz, dy, dx, dz, SouPos, Acc_Inj, TTsrc=None):
    """Function to compute traveltime in parallel"""
    velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
    velocity.min_coords = oy, ox, oz
    velocity.node_intervals = dy, dx, dz
    velocity.npts = vel.shape[0], vel.shape[1], vel.shape[2]
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
            solver_ek.src_loc = [SouPos[ishot,1],SouPos[ishot,0],SouPos[ishot,2]] 
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
            eq_iz = int((SouPos[ishot,2]-oz)/dz + 0.5)
            eq_iy = int((SouPos[ishot,1]-oy)/dy + 0.5)
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
            eq_iz = int((SouPos[ishot,2,iPnt]-oz)/dz + 0.5)
            eq_iy = int((SouPos[ishot,1,iPnt]-oy)/dy + 0.5)
            eq_ix = int((SouPos[ishot,0,iPnt]-ox)/dx + 0.5)
            src_idx = (eq_iy, eq_ix, eq_iz)
            solver_ek.tt.values[src_idx] = 0.0 if TTsrc is None else TTsrc[iPnt]
            solver_ek.unknown[src_idx] = False
            solver_ek.trial.push(*src_idx)
    # Solving Eikonal equation
    solver_ek.solve()
    return solver_ek.tt.values

def solve_linearized_fwd(vel0, delta_v, ishot, dy, dx, dz, oy, ox, oz, tt0, tt_idx, RecPos):
    """Function to solve linearized problem"""
    # Sorting traveltime in ascending order
    tt_idx = sorting3D(tt0, tt_idx)
    delta_tt = np.zeros_like(vel0)
    FMM_tt_lin_fwd3D(delta_v, delta_tt, vel0, tt0, tt_idx, dy, dx, dz)
    data_tt_lin = extract_tt_3D_FWD(delta_tt, RecPos[:,1], RecPos[:,0], RecPos[:,2], oy, ox, oz, dy, dx, dz)
    return data_tt_lin

def solve_linearized_adj(vel0, data, ishot, oy, ox, oz, dy, dx, dz, tt0, tt_idx, RecPos):
    delta_tt = np.zeros_like(vel0)
    delta_v = np.zeros_like(vel0)
    # Sorting traveltime in descending order
    tt_idx = sorting3D(tt0, tt_idx, ordering="d")
    # Injecting traveltime to correct grid positions
    for iRec in range(RecPos.shape[0]):
        wy = (RecPos[iRec, 1] - oy) / dy
        wx = (RecPos[iRec, 0] - ox) / dx
        wz = (RecPos[iRec, 2] - oz) / dz
        iy = int(wy)
        ix = int(wx)
        iz = int(wz)
        # Interpolation weights
        wy -= iy
        wx -= ix
        wz -= iz
        delta_tt[iy,ix,iz]       += data[ishot, iRec] * (1.0 - wy)*(1.0 - wx)*(1.0 - wz) 
        delta_tt[iy,ix,iz+1]     += data[ishot, iRec] * (1.0 - wy)*(1.0 - wx)*(wz) 
        delta_tt[iy,ix+1,iz+1]   += data[ishot, iRec] * (1.0 - wy)*(wx)*(wz) 
        delta_tt[iy,ix+1,iz]     += data[ishot, iRec] * (1.0 - wy)*(wx)*(1.0 - wz)  
        delta_tt[iy+1,ix,iz]     += data[ishot, iRec] * (wy)*(1.0 - wx)*(1.0 - wz) 
        delta_tt[iy+1,ix,iz+1]   += data[ishot, iRec] * (wy)*(1.0 - wx)*(wz) 
        delta_tt[iy+1,ix+1,iz+1] += data[ishot, iRec] * (wy)*(wx)*(wz) 
        delta_tt[iy+1,ix+1,iz]   += data[ishot, iRec] * (wy)*(wx)*(1.0 - wz)  
    FMM_tt_lin_adj3D(delta_v, delta_tt, vel0, tt0, tt_idx, dy, dx, dz)
    return delta_v

class EikonalTT_lin_3D(pyOp.Operator):

    def __init__(self, vel, tt_data, oy, ox, oz, dy, dx, dz, SouPos, RecPos, TTsrc=None, tt_maps=None, verbose=False, **kwargs):
        """3D Eikonal-equation traveltime prediction operator"""
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
        self.oy = oy
        self.ox = ox
        self.oz = oz
        self.dy = dy
        self.dx = dx
        self.dz = dz
        self.ncomp = vel.shape[0]
        self.ny = vel.shape[1]
        self.nx = vel.shape[2]
        self.nz = vel.shape[3]
        # Use smallest possible domain (Ginsu knives)
        self.ginsu = kwargs.get("ginsu", False)
        buffer = kwargs.get("buffer", 2.0) # By default 2.0 km
        self.bufferX = int(buffer/dx)
        self.bufferY = int(buffer/dy)
        self.bufferZ = int(buffer/dz)
        self.xAxis = np.linspace(ox, ox+(self.nx-1)*dx, self.nx)
        self.yAxis = np.linspace(oy, oy+(self.ny-1)*dy, self.ny)
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
        idx_1d = np.arange(velNd[0,:,:,:].size)
        idy,idx,idz = np.unravel_index(idx_1d, velNd[0,:,:,:].shape)
        self.tt_idx = np.array([idy,idx,idz]).T
        # Traveltime maps
        if tt_maps is None:
            self.tt_maps = []
            for _ in range(self.nSou*self.ncomp):
                self.tt_maps.append(np.zeros_like(velNd[0,:,:,:]))
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
                    minY = np.zeros(self.nSou, dtype=int)
                    maxY = np.zeros(self.nSou, dtype=int)
                    minZ = np.zeros(self.nSou, dtype=int)
                    maxZ = np.zeros(self.nSou, dtype=int)
                    for ishot in range(self.nSou):
                        minX[ishot] =  max(0, min(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].min())))-self.bufferX)
                        maxX[ishot] =  min(self.nx, max(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].max())))+self.bufferX)
                        minY[ishot] =  max(0, min(np.argmin(abs(self.yAxis-self.SouPos[ishot,1])), np.argmin(abs(self.yAxis-self.RecPos[:,1].min())))-self.bufferY)
                        maxY[ishot] =  min(self.ny, max(np.argmin(abs(self.yAxis-self.SouPos[ishot,1])), np.argmin(abs(self.yAxis-self.RecPos[:,1].max())))+self.bufferY)
                        minZ[ishot] =  max(0, min(np.argmin(abs(self.zAxis-self.SouPos[ishot,2])), np.argmin(abs(self.zAxis-self.RecPos[:,2].min())))-self.bufferZ)
                        maxZ[ishot] =  min(self.nz, max(np.argmin(abs(self.zAxis-self.SouPos[ishot,2])), np.argmin(abs(self.zAxis-self.RecPos[:,2].max())))+self.bufferZ)
                    result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_timeLin)(vel0Nd[icomp,minY[ishot]:maxY[ishot],minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]], ishot, self.yAxis[minY[ishot]], self.xAxis[minX[ishot]], self.zAxis[minZ[ishot]], self.dy, self.dx, self.dz, self.SouPos, self.Acc_Inj, self.TTsrc[ishot]) for ishot in range(self.nSou))
                else:
                    result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_timeLin)(vel0Nd[icomp,:,:,:], ishot, self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, self.SouPos, self.Acc_Inj, self.TTsrc[ishot]) for ishot in range(self.nSou))
                for ishot in range(self.nSou):
                    if self.ginsu:
                        self.tt_maps[ishot+icomp*self.nSou][minY[ishot]:maxY[ishot],minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]] = result[ishot]
                    else:
                        self.tt_maps[ishot+icomp*self.nSou][:] = result[ishot]
            self.TT_update = False
        ###################################
        # Computing linearized traveltime #
        ###################################
        for icomp in range(self.ncomp):
            result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(solve_linearized_fwd)(vel0Nd[icomp,:,:,:], modelNd[icomp,:,:,:], ishot, self.dy, self.dx, self.dz, self.oy, self.ox, self.oz, self.tt_maps[ishot+icomp*self.nSou], self.tt_idx, self.RecPos) for ishot in range(self.nSou))
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
                    minY = np.zeros(self.nSou, dtype=int)
                    maxY = np.zeros(self.nSou, dtype=int)
                    minZ = np.zeros(self.nSou, dtype=int)
                    maxZ = np.zeros(self.nSou, dtype=int)
                    for ishot in range(self.nSou):
                        minX[ishot] =  max(0, min(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].min())))-self.bufferX)
                        maxX[ishot] =  min(self.nx, max(np.argmin(abs(self.xAxis-self.SouPos[ishot,0])), np.argmin(abs(self.xAxis-self.RecPos[:,0].max())))+self.bufferX)
                        minY[ishot] =  max(0, min(np.argmin(abs(self.yAxis-self.SouPos[ishot,1])), np.argmin(abs(self.yAxis-self.RecPos[:,1].min())))-self.bufferY)
                        maxY[ishot] =  min(self.ny, max(np.argmin(abs(self.yAxis-self.SouPos[ishot,1])), np.argmin(abs(self.yAxis-self.RecPos[:,1].max())))+self.bufferY)
                        minZ[ishot] =  max(0, min(np.argmin(abs(self.zAxis-self.SouPos[ishot,2])), np.argmin(abs(self.zAxis-self.RecPos[:,2].min())))-self.bufferZ)
                        maxZ[ishot] =  min(self.nz, max(np.argmin(abs(self.zAxis-self.SouPos[ishot,2])), np.argmin(abs(self.zAxis-self.RecPos[:,2].max())))+self.bufferZ)
                    result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_timeLin)(vel0Nd[icomp,minY[ishot]:maxY[ishot],minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]], ishot, self.yAxis[minY[ishot]], self.xAxis[minX[ishot]], self.zAxis[minZ[ishot]], self.dy, self.dx, self.dz, self.SouPos, self.Acc_Inj, self.TTsrc[ishot]) for ishot in range(self.nSou))
                else:
                    result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(compute_travel_timeLin)(vel0Nd[icomp,:,:,:], ishot, self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, self.SouPos, self.Acc_Inj, self.TTsrc[ishot]) for ishot in range(self.nSou))
                for ishot in range(self.nSou):
                    if self.ginsu:
                        self.tt_maps[ishot+icomp*self.nSou][minY[ishot]:maxY[ishot],minX[ishot]:maxX[ishot],minZ[ishot]:maxZ[ishot]] = result[ishot]
                    else:
                        self.tt_maps[ishot+icomp*self.nSou][:] = result[ishot]
            self.TT_update = False
        ###################################
        # Computing velocity perturbation #
        ###################################
        for icomp in range(self.ncomp):
            result = Parallel(n_jobs=self.nthrs, verbose=self.verbose)(delayed(solve_linearized_adj)(vel0Nd[icomp,:,:,:], dataNd[icomp*self.nSou:icomp*self.nSou+self.nSou,:], ishot, self.oy, self.ox, self.oz, self.dy, self.dx, self.dz, self.tt_maps[ishot+icomp*self.nSou], self.tt_idx, self.RecPos) for ishot in range(self.nSou))
            for ishot in range(self.nSou):
                modelNd[icomp,:,:,:] += result[ishot]
        return
    
    def set_vel(self, vel):
        """Function to set background velocity model"""
        if self.vel0.isDifferent(vel):
            self.TT_update = True
        self.vel0.copy(vel)