# Module contaning useful functions for the 3D tomography example on the Long Valley caldera

import numpy as np
import utm
from scipy.interpolate import interp1d, RegularGridInterpolator

def load_velocity_model_npz(velModFile):
    """
    Load velocity model NPZ file and construct coordinate axes and metadata.

    Parameters
    ----------
    velModFile : str
        Path to NPZ velocity model.

    Returns
    -------
    Vp : ndarray (ny, nx, nz)
    Vs : ndarray (ny, nx, nz)
    VpInitEntire : ndarray
    VsInitEntire : ndarray
    dy, dx, dz : float
    ny, nx, nz : int
    minLon, maxLon, minLat, maxLat : float
    oy, ox, oz : float
    zAxis, xAxis, yAxis : ndarray
    latAxis, lonAxis : ndarray
    zone_n, zone_lt : int, str
    NLL_origin_utm : tuple
    """

    with np.load(velModFile) as velFile:

        Vp = velFile["vp"]
        Vs = velFile["vs"]

        dy, dx, dz = velFile["ds"]
        ny, nx, nz = Vp.shape

        minLon, maxLon, minLat, maxLat = velFile["boundbox"]

        oy, ox = 0.0, 0.0
        oz = velFile["oz"]

    latAxis = np.linspace(minLat, maxLat, ny)
    lonAxis = np.linspace(minLon, maxLon, nx)

    # UTM zone from model center
    lat0 = 0.5 * (minLat + maxLat)
    lon0 = 0.5 * (minLon + maxLon)
    _, _, zone_n, zone_lt = utm.from_latlon(lat0, lon0)

    # NLL origin at SW corner
    NLL_origin_utm = utm.from_latlon(minLat, minLon, zone_n, zone_lt)

    return (
        Vp, Vs,
        dy, dx, dz,
        ny, nx, nz,
        minLon, maxLon, minLat, maxLat,
        oy, ox, oz,
        zone_n, zone_lt,
        NLL_origin_utm
    )



import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

def _as_increasing_axis_and_data(axis, data, axis_dim):
    """
    Ensure axis is strictly increasing. If descending, flip axis and
    corresponding data along axis_dim.
    """
    axis = np.asarray(axis)
    if axis[0] <= axis[-1]:
        return axis, data
    # descending -> flip
    axis_flip = axis[::-1]
    data_flip = np.flip(data, axis=axis_dim)
    return axis_flip, data_flip

def _make_rgi_2d(xAx, yAx, Z2d, fill_value=np.nan):
    """
    Create a RegularGridInterpolator for a 2D field defined on (yAx, xAx),
    where Z2d has shape (len(yAx), len(xAx)).
    """
    xAx = np.asarray(xAx)
    yAx = np.asarray(yAx)
    Z2d = np.asarray(Z2d)

    if Z2d.shape != (yAx.size, xAx.size):
        raise ValueError(
            f"Z2d shape {Z2d.shape} does not match (len(yAx), len(xAx)) = {(yAx.size, xAx.size)}"
        )

    # Ensure increasing axes for RGI
    yAx_inc, Z2d_inc = _as_increasing_axis_and_data(yAx, Z2d, axis_dim=0)
    xAx_inc, Z2d_inc = _as_increasing_axis_and_data(xAx, Z2d_inc, axis_dim=1)

    return RegularGridInterpolator(
        (yAx_inc, xAx_inc),
        Z2d_inc,
        method="linear",
        bounds_error=False,
        fill_value=fill_value
    )

def extractSlice(model, xPos, yPos, xAx, yAx, fill_value=np.nan):
    """
    Extract values along an arbitrary horizontal polyline (xPos, yPos)
    for every depth slice in a 3D model.

    model shape: (ny, nx, nz) corresponding to yAx (ny), xAx (nx).
    xPos, yPos shape: (npts,)
    Returns: m_slice shape (npts, nz)
    """
    model = np.asarray(model)
    xPos = np.asarray(xPos)
    yPos = np.asarray(yPos)

    ny, nx, nz = model.shape
    if (len(yAx) != ny) or (len(xAx) != nx):
        raise ValueError(
            f"Axes do not match model: model (ny,nx)=({ny},{nx}), "
            f"len(yAx)={len(yAx)}, len(xAx)={len(xAx)}"
        )

    npts = xPos.size
    m_slice = np.empty((npts, nz), dtype=np.float32)

    # Stack query points once: RGI expects (N, 2) with columns (y, x)
    pts = np.column_stack([yPos, xPos])

    for idz in range(nz):
        rgi = _make_rgi_2d(xAx, yAx, model[:, :, idz], fill_value=fill_value)
        m_slice[:, idz] = rgi(pts).astype(np.float32)

    return m_slice

def resample_dep_slice(sliceIn, zAx, dzFine):
    """
    Resample a (npts, nz) slice onto a finer vertical grid.
    """
    sliceIn = np.asarray(sliceIn)
    zAx = np.asarray(zAx)

    npts, nz = sliceIn.shape
    if nz != zAx.size:
        raise ValueError(f"sliceIn nz={nz} does not match len(zAx)={zAx.size}")

    nzFine = int((zAx[-1] - zAx[0]) / dzFine + 1.5)
    zAxFine = np.linspace(zAx[0], zAx[-1], nzFine)

    sliceOut = np.empty((npts, nzFine), dtype=np.float32)

    for ipt in range(npts):
        fz = interp1d(zAx, sliceIn[ipt, :], kind="linear",
                     bounds_error=False, fill_value=np.nan)
        sliceOut[ipt, :] = fz(zAxFine).astype(np.float32)

    return sliceOut, zAxFine

def remove_above_topoAlongSlice(velSlice, DEM2, oztopo, dztopo,
                                xPos, yPos, xAx, yAx, fill_value=np.nan):
    """
    For each point along the slice, interpolate topography/elevation from DEM2
    and set velSlice above the surface to NaN.

    velSlice shape: (npts, nzFine)
    DEM2 shape: (len(yAx), len(xAx))
    oztopo: z origin for velSlice (same units as DEM2 and zAxFine)
    dztopo: vertical spacing of velSlice (dzFine)
    """
    velSlice = np.asarray(velSlice)
    DEM2 = np.asarray(DEM2)
    xPos = np.asarray(xPos)
    yPos = np.asarray(yPos)

    npts = xPos.size
    eleSlice = np.empty(npts, dtype=np.float32)

    rgi_dem = _make_rgi_2d(xAx, yAx, DEM2, fill_value=fill_value)
    pts = np.column_stack([yPos, xPos])
    ele = rgi_dem(pts)

    for ipt in range(npts):
        eleSlice[ipt] = ele[ipt]
        if not np.isfinite(eleSlice[ipt]):
            continue

        idz = int(np.floor(((eleSlice[ipt] - oztopo) / dztopo) + 0.5))
        idz = np.clip(idz, 0, velSlice.shape[1])  # safe bounds
        velSlice[ipt, :idz] = np.nan

    return eleSlice

def extract_profile(Vel, xAx, yAx, zAx, Xpts, Ypts, dzSlice, DEM1, fill_value=np.nan):
    """
    Your original pipeline:
      1) Extract 3D model along (Xpts,Ypts) for all depths
      2) Resample vertically to dzSlice
      3) Remove values above topography (DEM1) along the slice

    Returns:
      eleSlice, zAxFine, VelSlice
    """
    VelSlice = extractSlice(Vel, Xpts, Ypts, xAx, yAx, fill_value=fill_value)
    VelSlice, zAxFine = resample_dep_slice(VelSlice, zAx, dzSlice)

    dzFine = zAxFine[1] - zAxFine[0]
    eleSlice = remove_above_topoAlongSlice(
        VelSlice, DEM1, zAxFine[0], dzFine,
        Xpts, Ypts, xAx, yAx, fill_value=fill_value
    )
    return eleSlice, zAxFine, VelSlice


def slide_def(latPt, lonPt, orig_dom, zone_name, zone_letter):
    Xpt, Ypt, _, _ = utm.from_latlon(latPt, lonPt, zone_name, zone_letter)
    Xpt = (Xpt - orig_dom[0])*1e-3
    Ypt = (Ypt - orig_dom[1])*1e-3
    totalLength = np.sum(np.sqrt((Xpt[1:]-Xpt[:-1])**2+(Ypt[1:]-Ypt[:-1])**2))
    Ax = np.linspace(0.0, totalLength, Ypt.shape[0])
    return Ax


# Function to scale depth axis
def forward(z):
    z1 = z.copy()
    z1[z<=0.0] *= 2.0
    return z1

def inverse(z):
    z1 = z.copy()
    z1[z<=0.0] /= 2.0
    return z1