"""Core data types for DAS observations and ray-parameter I/O.

Four main classes:

* :class:`DASRawData`   — continuous waveforms from a raw DAS recording.
* :class:`DASData`      — pre-extracted P/S time-window data (post-picking).
* :class:`RayParamDB`   — per-receiver pre-computed traveltime / takeoff field
  (Layer 1: the "database" you query with a source point to obtain ray params).
* :class:`RayParamTable` — flat event × receiver ray-parameter snapshot
  (Layer 2+3: nominal or Monte-Carlo-stacked, one file per receiver group).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np


@dataclass
class DASRawData:
    """Continuous DAS waveforms for a single event.

    Matches the HDF5 layout of raw recordings::

        data  (n_channels, n_samples)  float32
          @begin_time        str   ISO-8601
          @end_time          str   ISO-8601
          @dt_s              float sampling interval [s]
          @dx_m              float channel spacing [m]
          @event_id          str
          @event_time        str   ISO-8601
          @event_time_index  int   sample index of origin time
          @latitude          float
          @longitude         float
          @depth_km          float
          @magnitude         float
          @magnitude_type    str
          @unit              str   e.g. 'microstrain/s'
          @time_before       float seconds before event in trace
          @time_after        float seconds after event in trace

    Parameters
    ----------
    waveforms : np.ndarray
        Strain-rate waveforms, shape ``(n_channels, n_samples)``.
    dt : float
        Sampling interval in seconds.
    dx_m : float
        Along-fiber channel spacing in metres.
    event_time_index : int
        Sample index of the event origin time within the trace.
    begin_time : str
        ISO-8601 start time of the recording.
    end_time : str
        ISO-8601 end time of the recording.
    event_id : str, optional
    event_time : str, optional
    latitude, longitude : float, optional
    depth_km : float, optional
    magnitude : float, optional
    magnitude_type : str, optional
    unit : str
        Physical unit of the waveforms (default ``'microstrain/s'``).
    time_before : float
        Seconds before the event included in the recording.
    time_after : float
        Seconds after the event included in the recording.
    channel_coords : np.ndarray, optional
        Channel positions, shape ``(n_channels, 3)`` as ``[x, y, z]`` in metres.
    valid_mask : np.ndarray, optional
        Boolean channel mask from QC; ``True`` = good channel.
    metadata : dict
        Any additional provenance information.
    """

    waveforms: np.ndarray
    dt: float
    dx_m: float
    event_time_index: int

    begin_time: str = ""
    end_time: str = ""
    event_id: Optional[str] = None
    event_time: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth_km: Optional[float] = None
    magnitude: Optional[float] = None
    magnitude_type: Optional[str] = None
    unit: str = "microstrain/s"
    time_before: float = 0.0
    time_after: float = 0.0

    channel_coords: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        return self.waveforms.shape[0]

    @property
    def n_samples(self) -> int:
        return self.waveforms.shape[1]

    @property
    def sampling_rate(self) -> float:
        return 1.0 / self.dt

    @property
    def duration(self) -> float:
        """Total recording duration in seconds."""
        return self.n_samples * self.dt

    @property
    def n_valid(self) -> int:
        if self.valid_mask is None:
            return self.n_channels
        return int(self.valid_mask.sum())

    def time_array(self) -> np.ndarray:
        """Time axis relative to ``begin_time``, shape ``(n_samples,)`` in seconds."""
        return np.arange(self.n_samples) * self.dt

    def time_array_relative(self) -> np.ndarray:
        """Time axis relative to event origin time, shape ``(n_samples,)`` in seconds."""
        return (np.arange(self.n_samples) - self.event_time_index) * self.dt

    def has_channel_coords(self) -> bool:
        return self.channel_coords is not None


@dataclass
class DASData:
    """Pre-extracted P and S time-window data from a single DAS array / event.

    Matches the HDF5 layout produced by the preprocessing pipeline::

        data/
          @event_id, @event_time, @latitude, @longitude, @depth_km, @magnitude
          P/
            data         (n_channels, n_samples)  float32
            shift_index  (n_channels,)             int64   – arrival sample in window
            snr          (n_channels,)             float64
            traveltime   (n_channels,)             float64 – seconds from tref
          S/
            data         (n_channels, n_samples)  float32
            shift_index  (n_channels,)             int64
            snr          (n_channels,)             float64
            traveltime   (n_channels,)             float64
          N/  (optional – pre-P background noise window, same length as P)
            data         (n_channels, n_samples)  float32

    Parameters
    ----------
    p_data : np.ndarray
        P-wave window waveforms, shape ``(n_channels, n_samples_p)``.
    p_shift_index : np.ndarray
        Sample index of the P arrival within each window, shape ``(n_channels,)``.
    p_snr : np.ndarray
        P-wave signal-to-noise ratio per channel, shape ``(n_channels,)``.
    p_traveltime : np.ndarray
        P-wave traveltime in seconds (from ``tref``), shape ``(n_channels,)``.
    s_data : np.ndarray
        S-wave window waveforms, shape ``(n_channels, n_samples_s)``.
    s_shift_index : np.ndarray
        Sample index of the S arrival within each window, shape ``(n_channels,)``.
    s_snr : np.ndarray
        S-wave signal-to-noise ratio per channel, shape ``(n_channels,)``.
    s_traveltime : np.ndarray
        S-wave traveltime in seconds (from ``tref``), shape ``(n_channels,)``.
    dt : float
        Sampling interval in seconds (e.g. 0.01 for 100 Hz).
    channel_coords : np.ndarray, optional
        Channel positions, shape ``(n_channels, 3)`` as ``[x, y, z]`` in metres.
        Must be supplied before running the forward model.
    event_id : str, optional
    event_time : str, optional
        ISO-8601 event origin time string.
    latitude, longitude : float, optional
        Event epicentre coordinates in decimal degrees.
    depth_km : float, optional
        Event depth in kilometres.
    magnitude : float, optional
    noise_data : np.ndarray, optional
        Pre-P background noise window, shape ``(n_channels, n_samples_noise)``.
        Same window length as ``p_data``.  ``None`` when not available.
    p_valid : np.ndarray, optional
        Per-channel P-pick validity mask, shape ``(n_channels,)``, bool.
    s_valid : np.ndarray, optional
        Per-channel S-pick validity mask, shape ``(n_channels,)``, bool.
    metadata : dict
        Any additional provenance information.
    """

    # P-wave window
    p_data: np.ndarray
    p_shift_index: np.ndarray
    p_snr: np.ndarray
    p_traveltime: np.ndarray

    # S-wave window
    s_data: np.ndarray
    s_shift_index: np.ndarray
    s_snr: np.ndarray
    s_traveltime: np.ndarray

    # Sampling
    dt: float

    # Channel geometry (loaded separately from the HDF5 file)
    channel_coords: Optional[np.ndarray] = None

    # Event metadata (from HDF5 group attributes)
    event_id: Optional[str] = None
    event_time: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth_km: Optional[float] = None
    magnitude: Optional[float] = None

    # Pre-P background noise window (optional)
    noise_data: Optional[np.ndarray] = None

    # Channel quality mask: True = channel passed all QC checks
    # Shape (n_channels,).  None means no QC was applied (all assumed valid).
    valid_mask: Optional[np.ndarray] = None

    # Per-phase pick validity: True = pick exists for this channel
    p_valid: Optional[np.ndarray] = None
    s_valid: Optional[np.ndarray] = None

    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        return self.p_data.shape[0]

    @property
    def n_valid(self) -> int:
        """Number of channels that passed QC (or all channels if no QC applied)."""
        if self.valid_mask is None:
            return self.n_channels
        return int(self.valid_mask.sum())

    @property
    def sampling_rate(self) -> float:
        return 1.0 / self.dt

    def has_channel_coords(self) -> bool:
        return self.channel_coords is not None


# ======================================================================
# Ray-parameter I/O types (Layer 1 / Layer 2+3)
# ======================================================================

_LAYER1_GEOMETRIES = ("cyl_2d", "cart_3d")
_LZF_CHUNKED = {"compression": "lzf", "chunks": True}


@dataclass
class RayParamDB:
    """Per-receiver pre-computed traveltime / takeoff / raypath-length volume.

    One instance holds **one receiver's** lookup field. At inversion time you
    call :meth:`query` (or pass a list of these to the step1 drivers) to
    trilinearly interpolate for a set of source points and obtain a
    :class:`RayParamTable`.

    The volume can have either cylindrical (2-D FSM) or Cartesian (3-D eikonal)
    geometry, selected by the :attr:`geometry` tag. Only the axis fields that
    match the geometry are populated; the others remain ``None``.

    On-disk layout (HDF5)::

        @geometry        str   "cyl_2d" | "cart_3d"
        @receiver_x/y/z  float km    Cartesian receiver position
        @dz, @dr         float km    grid spacings
        @dasname         str         optional DAS deployment label

        /traveltime      (nz, n1, n2) f32 s       lzf compressed
        /takeoff         (nz, n1, n2) f32 rad     lzf compressed
        /raypath_length  (nz, n1, n2) f32 km      lzf compressed
        /grid_z          (nz,) f32 km             always present

        # cyl_2d only:
        /grid_r          (n1,) f32 km
        /grid_az         (n2,) f32 rad

        # cart_3d only:
        /grid_x          (n1,) f32 km
        /grid_y          (n2,) f32 km
        /azimuth         (nz, n1, n2) f32 rad     lzf compressed

    Parameters
    ----------
    traveltime, takeoff, raypath_length : np.ndarray
        3-D volumes shaped ``(nz, n1, n2)`` in float32. Units: seconds,
        radians (0 = straight down, π = straight up), kilometres.
    grid_z : np.ndarray
        Depth axis, shape ``(nz,)``, float32, km.
    geometry : str
        ``"cyl_2d"`` or ``"cart_3d"``.
    receiver_x, receiver_y, receiver_z : float
        Receiver Cartesian coordinates in km (grid frame).
    dz, dr : float
        Grid spacings in km.
    grid_r, grid_az : np.ndarray, optional
        Radial and azimuthal axes (cyl_2d only).
    grid_x, grid_y : np.ndarray, optional
        Horizontal axes (cart_3d only).
    azimuth : np.ndarray, optional
        Full azimuth field ``(nz, n1, n2)`` in radians (cart_3d only).
    dasname : str, optional
        DAS deployment identifier, e.g. ``"longvalley"``.
    """

    # Always-present arrays
    traveltime: np.ndarray
    takeoff: np.ndarray
    raypath_length: np.ndarray
    grid_z: np.ndarray

    # Always-present scalar metadata
    geometry: str
    receiver_x: float
    receiver_y: float
    receiver_z: float
    dz: float
    dr: float

    # cyl_2d only
    grid_r: Optional[np.ndarray] = None
    grid_az: Optional[np.ndarray] = None

    # cart_3d only
    grid_x: Optional[np.ndarray] = None
    grid_y: Optional[np.ndarray] = None
    azimuth: Optional[np.ndarray] = None

    # optional metadata
    dasname: Optional[str] = None

    # Optional grid metadata. Useful for debugging, plotting, and
    # coordinate-system conversions, but not required by the inversion.
    receiver_ix: Optional[int] = None
    receiver_iy: Optional[int] = None
    receiver_iz: Optional[int] = None
    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None
    x_offset: Optional[float] = None
    y_offset: Optional[float] = None
    h_max: Optional[float] = None

    # ------------------------------------------------------------------
    # Level-1 validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        tt_shape = self.traveltime.shape
        for name in ("takeoff", "raypath_length"):
            if getattr(self, name).shape != tt_shape:
                raise ValueError(
                    f"RayParamDB.{name}.shape {getattr(self, name).shape} "
                    f"!= traveltime.shape {tt_shape}"
                )
        if self.geometry not in _LAYER1_GEOMETRIES:
            raise ValueError(
                f"RayParamDB.geometry must be one of {_LAYER1_GEOMETRIES}, "
                f"got {self.geometry!r}"
            )
        if self.geometry == "cyl_2d":
            if self.grid_r is None or self.grid_az is None:
                raise ValueError(
                    "RayParamDB(geometry='cyl_2d') requires grid_r and grid_az"
                )
        else:  # cart_3d
            if self.grid_x is None or self.grid_y is None or self.azimuth is None:
                raise ValueError(
                    "RayParamDB(geometry='cart_3d') requires grid_x, grid_y, and azimuth"
                )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def to_hdf5(self, path: Union[str, Path]) -> None:
        """Serialize this object to an HDF5 file.

        Large arrays (``traveltime``, ``takeoff``, ``raypath_length``,
        ``azimuth``) are written with ``lzf`` compression and chunked.
        Grid axes are uncompressed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            # scalar attrs
            f.attrs["geometry"] = self.geometry
            for name in ("receiver_x", "receiver_y", "receiver_z", "dz", "dr"):
                f.attrs[name] = float(getattr(self, name))
            if self.dasname is not None:
                f.attrs["dasname"] = self.dasname
            # optional grid metadata
            for name in ("receiver_ix", "receiver_iy", "receiver_iz"):
                val = getattr(self, name)
                if val is not None:
                    f.attrs[name] = int(val)
            for name in ("origin_lat", "origin_lon",
                         "x_offset", "y_offset", "h_max"):
                val = getattr(self, name)
                if val is not None:
                    f.attrs[name] = float(val)
            # big arrays (compressed)
            for name in ("traveltime", "takeoff", "raypath_length"):
                f.create_dataset(name, data=getattr(self, name), **_LZF_CHUNKED)
            # depth axis (always present)
            f.create_dataset("grid_z", data=self.grid_z)
            # geometry-specific axes
            if self.geometry == "cyl_2d":
                f.create_dataset("grid_r", data=self.grid_r)
                f.create_dataset("grid_az", data=self.grid_az)
            else:  # cart_3d
                f.create_dataset("grid_x", data=self.grid_x)
                f.create_dataset("grid_y", data=self.grid_y)
                f.create_dataset("azimuth", data=self.azimuth, **_LZF_CHUNKED)

    @classmethod
    def from_hdf5(cls, path: Union[str, Path]) -> "RayParamDB":
        """Load a :class:`RayParamDB` from an HDF5 file written by :meth:`to_hdf5`."""
        path = Path(path)
        with h5py.File(path, "r") as f:
            geometry = str(f.attrs["geometry"])
            kwargs = dict(
                traveltime=f["traveltime"][()],
                takeoff=f["takeoff"][()],
                raypath_length=f["raypath_length"][()],
                grid_z=f["grid_z"][()],
                geometry=geometry,
                receiver_x=float(f.attrs["receiver_x"]),
                receiver_y=float(f.attrs["receiver_y"]),
                receiver_z=float(f.attrs["receiver_z"]),
                dz=float(f.attrs["dz"]),
                dr=float(f.attrs["dr"]),
            )
            if geometry == "cyl_2d":
                kwargs["grid_r"] = f["grid_r"][()]
                kwargs["grid_az"] = f["grid_az"][()]
            else:
                kwargs["grid_x"] = f["grid_x"][()]
                kwargs["grid_y"] = f["grid_y"][()]
                kwargs["azimuth"] = f["azimuth"][()]
            if "dasname" in f.attrs:
                kwargs["dasname"] = str(f.attrs["dasname"])
            for name in ("receiver_ix", "receiver_iy", "receiver_iz"):
                if name in f.attrs:
                    kwargs[name] = int(f.attrs[name])
            for name in ("origin_lat", "origin_lon",
                         "x_offset", "y_offset", "h_max"):
                if name in f.attrs:
                    kwargs[name] = float(f.attrs[name])
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Query — trilinear interpolation at source points
    # ------------------------------------------------------------------
    def query(self, source_pts: np.ndarray):
        """Trilinearly interpolate this volume at a batch of source points.

        Parameters
        ----------
        source_pts : np.ndarray
            Query points in the grid frame. Shape ``(n_ev, 3)``, columns
            ``(source_z, source_x, source_y)`` in km.

        Returns
        -------
        traveltime : np.ndarray
            ``(n_ev,)`` float32, seconds.
        takeoff : np.ndarray
            ``(n_ev,)`` float32, radians.
        raypath_length : np.ndarray
            ``(n_ev,)`` float32, km.
        azimuth : np.ndarray or None
            ``(n_ev,)`` float32 radians for ``cart_3d`` (read from the
            stored ``azimuth`` field); ``None`` for ``cyl_2d`` (caller is
            expected to compute azimuth geometrically from receiver
            position, since cylindrical lookups don't store an azimuth field).

        Notes
        -----
        Out-of-grid query points produce ``NaN``. This mirrors the behavior
        of the legacy ``interp_lookup_channels``.
        """
        # Local import to avoid paying the scipy cost unless query() is called
        from scipy.interpolate import RegularGridInterpolator

        pts = np.asarray(source_pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"source_pts must have shape (n_ev, 3), got {pts.shape}"
            )

        def _rgi(axes, values):
            return RegularGridInterpolator(
                axes,
                values.astype(np.float64),
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )

        if self.geometry == "cyl_2d":
            # Convert source (x, y) → (r, az) relative to the receiver.
            dx = pts[:, 1] - self.receiver_x
            dy = pts[:, 2] - self.receiver_y
            r = np.sqrt(dx * dx + dy * dy)
            az = np.arctan2(dy, dx) % (2.0 * np.pi)
            query_pts = np.column_stack([pts[:, 0], r, az])
            axes = (
                np.asarray(self.grid_z, dtype=np.float64),
                np.asarray(self.grid_r, dtype=np.float64),
                np.asarray(self.grid_az, dtype=np.float64),
            )
            tt = _rgi(axes, self.traveltime)(query_pts).astype(np.float32)
            takeoff = _rgi(axes, self.takeoff)(query_pts).astype(np.float32)
            rpl = _rgi(axes, self.raypath_length)(query_pts).astype(np.float32)
            return tt, takeoff, rpl, None

        # cart_3d
        axes = (
            np.asarray(self.grid_z, dtype=np.float64),
            np.asarray(self.grid_x, dtype=np.float64),
            np.asarray(self.grid_y, dtype=np.float64),
        )
        tt = _rgi(axes, self.traveltime)(pts).astype(np.float32)
        takeoff = _rgi(axes, self.takeoff)(pts).astype(np.float32)
        rpl = _rgi(axes, self.raypath_length)(pts).astype(np.float32)
        az = _rgi(axes, self.azimuth)(pts).astype(np.float32)
        return tt, takeoff, rpl, az


@dataclass
class RayParamTable:
    """Flat event × receiver snapshot of ray geometry. Unified Layer 2 / Layer 3.

    The ray-geometry arrays have shape ``(n_ev, n_rx)`` in the nominal case or
    ``(n_trials, n_ev, n_rx)`` when Monte-Carlo perturbations are stored.
    By convention trial index 0 is always the unperturbed snapshot.

    Source locations share the leading axis: ``(n_ev,)`` nominal or
    ``(n_trials, n_ev)`` when perturbed.

    On-disk layout (HDF5)::

        @n_ev, @n_rx, @n_trials       int    shape metadata
        @forward_method               str    e.g. "sta_2d", "das_1d"
        @dasname                      str    optional deployment label
        @perturb_vert_uncert_km       float  optional
        @perturb_horz_uncert_km       float  optional

        /traveltime      (n_ev, n_rx) OR (n_trials, n_ev, n_rx)  f32 s
        /takeoff                    same shape                    f32 rad
        /azimuth                    same shape                    f32 rad
        /raypath_length             same shape                    f32 km
        /source_x        (n_ev,) OR (n_trials, n_ev)               f64 km
        /source_y                   same                           f64
        /source_z                   same                           f64
        /receiver_x      (n_rx,)                                   f64
        /receiver_y      (n_rx,)                                   f64

        # STA only (absent for DAS):
        /network         (n_rx,) str
        /station         (n_rx,) str
        /location        (n_rx,) str

        # DAS only (absent for STA):
        /rec_azi         (n_rx,) f32 rad

    Parameters
    ----------
    traveltime, takeoff, azimuth, raypath_length : np.ndarray
        Ray geometry arrays. All four must share the same shape. Nominal:
        ``(n_ev, n_rx)``. MC: ``(n_trials, n_ev, n_rx)`` with trial 0 = unperturbed.
    source_x, source_y, source_z : np.ndarray
        Source (event) coordinates in km. Leading shape matches ``traveltime``
        (``(n_ev,)`` nominal, ``(n_trials, n_ev)`` MC).
    receiver_x, receiver_y : np.ndarray
        Receiver coordinates in km, shape ``(n_rx,)``.
    forward_method : str
        Label such as ``"sta_1d"``, ``"das_2d"``, ``"sta_3d"``.
    network, station, location : np.ndarray, optional
        Station SEED identifiers, shape ``(n_rx,)`` string arrays (STA only).
    rec_azi : np.ndarray, optional
        Per-receiver orientation in radians, shape ``(n_rx,)`` (DAS fiber
        bearing; typically ``None`` for stations).
    dasname : str, optional
        DAS deployment identifier.
    perturb_vert_uncert_km, perturb_horz_uncert_km : float, optional
        Gaussian perturbation sigmas used when MC trials were generated.
    """

    # Ray geometry
    traveltime: np.ndarray
    takeoff: np.ndarray
    azimuth: np.ndarray
    raypath_length: np.ndarray

    # Source / receiver coordinates
    source_x: np.ndarray
    source_y: np.ndarray
    source_z: np.ndarray
    receiver_x: np.ndarray
    receiver_y: np.ndarray

    # Method label
    forward_method: str

    # Optional STA metadata
    network: Optional[np.ndarray] = None
    station: Optional[np.ndarray] = None
    location: Optional[np.ndarray] = None

    # Optional DAS metadata
    rec_azi: Optional[np.ndarray] = None

    # Optional scalar attrs
    dasname: Optional[str] = None
    perturb_vert_uncert_km: Optional[float] = None
    perturb_horz_uncert_km: Optional[float] = None

    # ------------------------------------------------------------------
    # Level-1 validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        tt = self.traveltime
        if tt.ndim not in (2, 3):
            raise ValueError(
                f"RayParamTable.traveltime.ndim must be 2 (nominal) or 3 (MC), "
                f"got ndim={tt.ndim}"
            )
        for name in ("takeoff", "azimuth", "raypath_length"):
            if getattr(self, name).shape != tt.shape:
                raise ValueError(
                    f"RayParamTable.{name}.shape {getattr(self, name).shape} "
                    f"!= traveltime.shape {tt.shape}"
                )

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------
    @property
    def n_trials(self) -> int:
        """Number of Monte-Carlo trials; 1 for nominal tables."""
        return 1 if self.traveltime.ndim == 2 else self.traveltime.shape[0]

    @property
    def n_ev(self) -> int:
        return self.traveltime.shape[-2]

    @property
    def n_rx(self) -> int:
        return self.traveltime.shape[-1]

    @property
    def is_perturbed(self) -> bool:
        return self.n_trials > 1

    def trial(self, i: int) -> "RayParamTable":
        """Return a nominal-shape copy containing only trial ``i``.

        For a nominal table, ``trial(0)`` returns ``self`` unchanged.
        """
        if not self.is_perturbed:
            if i != 0:
                raise IndexError(f"nominal RayParamTable has no trial {i}")
            return self
        return RayParamTable(
            traveltime=self.traveltime[i],
            takeoff=self.takeoff[i],
            azimuth=self.azimuth[i],
            raypath_length=self.raypath_length[i],
            source_x=self.source_x[i],
            source_y=self.source_y[i],
            source_z=self.source_z[i],
            receiver_x=self.receiver_x,
            receiver_y=self.receiver_y,
            forward_method=self.forward_method,
            network=self.network,
            station=self.station,
            location=self.location,
            rec_azi=self.rec_azi,
            dasname=self.dasname,
            perturb_vert_uncert_km=self.perturb_vert_uncert_km,
            perturb_horz_uncert_km=self.perturb_horz_uncert_km,
        )

    @classmethod
    def stack_mc_trials(cls, tables: list["RayParamTable"]) -> "RayParamTable":
        """Combine N nominal tables into a single Monte-Carlo-shaped table.

        Input tables must all be nominal (``ndim=2``) and agree on
        :attr:`forward_method`, :attr:`n_ev`, :attr:`n_rx`, receiver
        coordinates, and optional metadata. Ray-geometry arrays and
        source coordinates may differ across trials.

        By convention the first table is trial 0 (unperturbed).

        Parameters
        ----------
        tables : list of RayParamTable
            Per-trial nominal tables to stack. ``len(tables) == n_trials``.

        Returns
        -------
        RayParamTable
            Single table with ``(n_trials, n_ev, n_rx)`` shape.
            Metadata is inherited from ``tables[0]``.
        """
        if not tables:
            raise ValueError("stack_mc_trials requires at least one table")
        t0 = tables[0]
        if any(t.is_perturbed for t in tables):
            raise ValueError(
                "stack_mc_trials: all input tables must be nominal "
                "(traveltime.ndim == 2)"
            )
        for i, t in enumerate(tables[1:], start=1):
            if t.traveltime.shape != t0.traveltime.shape:
                raise ValueError(
                    f"stack_mc_trials: tables[{i}].traveltime.shape "
                    f"{t.traveltime.shape} != tables[0] {t0.traveltime.shape}"
                )
            if t.forward_method != t0.forward_method:
                raise ValueError(
                    f"stack_mc_trials: tables[{i}].forward_method "
                    f"{t.forward_method!r} != tables[0] {t0.forward_method!r}"
                )
        return cls(
            traveltime=np.stack([t.traveltime for t in tables], axis=0),
            takeoff=np.stack([t.takeoff for t in tables], axis=0),
            azimuth=np.stack([t.azimuth for t in tables], axis=0),
            raypath_length=np.stack([t.raypath_length for t in tables], axis=0),
            source_x=np.stack([t.source_x for t in tables], axis=0),
            source_y=np.stack([t.source_y for t in tables], axis=0),
            source_z=np.stack([t.source_z for t in tables], axis=0),
            receiver_x=t0.receiver_x,
            receiver_y=t0.receiver_y,
            forward_method=t0.forward_method,
            network=t0.network,
            station=t0.station,
            location=t0.location,
            rec_azi=t0.rec_azi,
            dasname=t0.dasname,
            perturb_vert_uncert_km=t0.perturb_vert_uncert_km,
            perturb_horz_uncert_km=t0.perturb_horz_uncert_km,
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def to_hdf5(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype()
        with h5py.File(path, "w") as f:
            # shape + method
            f.attrs["n_ev"] = int(self.n_ev)
            f.attrs["n_rx"] = int(self.n_rx)
            f.attrs["n_trials"] = int(self.n_trials)
            f.attrs["forward_method"] = self.forward_method
            if self.dasname is not None:
                f.attrs["dasname"] = self.dasname
            if self.perturb_vert_uncert_km is not None:
                f.attrs["perturb_vert_uncert_km"] = float(self.perturb_vert_uncert_km)
            if self.perturb_horz_uncert_km is not None:
                f.attrs["perturb_horz_uncert_km"] = float(self.perturb_horz_uncert_km)
            # ray geometry
            for name in ("traveltime", "takeoff", "azimuth", "raypath_length"):
                f.create_dataset(name, data=getattr(self, name))
            # source / receiver coords
            for name in ("source_x", "source_y", "source_z",
                         "receiver_x", "receiver_y"):
                f.create_dataset(name, data=getattr(self, name))
            # optional station metadata
            for name in ("network", "station", "location"):
                arr = getattr(self, name)
                if arr is not None:
                    f.create_dataset(
                        name,
                        data=np.array([str(v) for v in arr], dtype=string_dtype),
                    )
            # optional DAS metadata
            if self.rec_azi is not None:
                f.create_dataset("rec_azi", data=self.rec_azi)

    @classmethod
    def from_hdf5(cls, path: Union[str, Path]) -> "RayParamTable":
        path = Path(path)
        with h5py.File(path, "r") as f:
            kwargs = dict(
                traveltime=f["traveltime"][()],
                takeoff=f["takeoff"][()],
                azimuth=f["azimuth"][()],
                raypath_length=f["raypath_length"][()],
                source_x=f["source_x"][()],
                source_y=f["source_y"][()],
                source_z=f["source_z"][()],
                receiver_x=f["receiver_x"][()],
                receiver_y=f["receiver_y"][()],
                forward_method=str(f.attrs["forward_method"]),
            )
            # optional STA metadata
            for name in ("network", "station", "location"):
                if name in f:
                    ds = f[name]
                    kwargs[name] = np.array(ds.asstr()[:], dtype=object)
            # optional DAS metadata
            if "rec_azi" in f:
                kwargs["rec_azi"] = f["rec_azi"][()]
            # scalar attrs
            if "dasname" in f.attrs:
                kwargs["dasname"] = str(f.attrs["dasname"])
            if "perturb_vert_uncert_km" in f.attrs:
                kwargs["perturb_vert_uncert_km"] = float(f.attrs["perturb_vert_uncert_km"])
            if "perturb_horz_uncert_km" in f.attrs:
                kwargs["perturb_horz_uncert_km"] = float(f.attrs["perturb_horz_uncert_km"])
        return cls(**kwargs)
