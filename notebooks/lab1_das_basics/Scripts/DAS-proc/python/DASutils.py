# Module containing definition of useful function to read/write/process DAS data
from re import L
import time
import numpy as np
import h5py
import tqdm # Progress bar
from scipy.signal import sosfilt, iirfilter
from scipy.signal import zpk2sos
import struct
import warnings
import dateutil.parser
import datetime
import utm
from scipy.signal.windows import tukey
from copy import deepcopy
from scipy.interpolate import interp1d
import math
import os
import gzip
# Inversion modules
from scipy import sparse
# Time zone variables
import pytz
UTC = pytz.timezone("UTC")
PST = pytz.timezone("US/Pacific")
import dateutil.parser as dateparse

# Parallelization
from numba import jit, prange, njit
import psutil
Ncores = psutil.cpu_count(logical = False)


#############################################################
################ PROCESSING FUNCTIONS #######################
#############################################################

# C++/pybind11 functions
path_dasutil = os.path.split(os.path.abspath(__file__))[0]
pyDAS_path = os.path.join(path_dasutil, "../build")
if "LD_LIBRARY_PATH" not in os.environ.keys():
    os.environ["LD_LIBRARY_PATH"] = pyDAS_path
elif pyDAS_path not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"] + pyDAS_path + ":"
else:
    # print("pyDAS_path already in LD_LIBRARY_PATH")
    # os.environ["LD_LIBRARY_PATH"] = pyDAS_path
    pass

import sys
sys.path.insert(0, pyDAS_path)
sys.path.insert(0, path_dasutil)
import pyDAS

# Adding ASN simpledas reader
ASN_path = os.path.join(path_dasutil, "../external/ASN_module/")
sys.path.insert(0, ASN_path)
import h5pydict, warnings
from asn_mod import _fix_meta_back_compability, _set_sensitivity, unwrap

def readASNfile(filename, firstCh=0, lastCh=None, firstSample=0, lastSample=None, unwr=True, spikeThr=None):
    """
    Function to read ASN formatted file

    unwr: bool
        Unwrap strain rate along spatial axis before time-integration. This may
        be needed for strain rate amplitudes > 8π/(dt*gaugeLength*sensitivity).
        If only smaller ampliudes are expected, uwr can be set to False.
        Default is True.

    spikeThr: float or None
        Threshold (in rad/m/s) for spike detection and removal.
        Sweep rates exceeding this threshold in absolute value are set to zero
        before integration.
        If there is steps or spikes in the data one may try setting
        with spikeThr = 3/(gaugeLength*dt). Higher values may also be usefull.
        Default is None, which deactivates spike removal.
        Be aware that spike removal disables unwrapping, and uwr should
        therefore be set to False when spike removal is activated.
    """
    with h5pydict.DictFile(filename, "r") as f:
        m = f.load_dict(skipFields=["data"])
        nSamplesRead, nChRaw = f["data"].shape
        _fix_meta_back_compability(m, nSamplesRead, nChRaw)
        # Setting sensitivity toread directly strain rate
        sensitivity, unit_out, sensitivities_out, sensitivityUnits_out = _set_sensitivity(m["header"], 0, None)
        if not isinstance(sensitivity, np.ndarray) or sensitivity.size == 1:  # singelton
            if sensitivity != 0.0:
                scale = np.float32(m["header"]["dataScale"] / sensitivity)
            else:
                raise ValueError(
                    "Sensitivity value set to zero, use sensitivitySelect=-1 or -2 to avoid error"
                )
        else:  # array with different sensitivity each channel
            if nSamplesRead != len(sensitivity):
                raise ValueError(
                    "Length of sensitivity vector does not match number of channels"
                )
            if np.any(sensitivity != 0.0):
                sensitivity = np.atleast_2d(sensitivity[firstCh:lastCh])
                scale = np.array(
                    m["header"]["dataScale"] / sensitivity, dtype=np.float32
                )  # make 2D
            else:
                raise ValueError(
                    "Sensitivity value set to zero, use sensitivitySelect=-1 or -2 to avoid error"
                )
        # Minus sign to change convention to positive strain -> fiber stretching/elongation
        data = f["data"][int(firstSample):int(lastSample), int(firstCh):int(lastCh)] * -scale
        if (unwr or spikeThr) and m["header"]["dataType"] == 2:
            unwr, spikeThr = (False,) * 2
            warnings.warn(
                "Options unwr or spikeThr can only be\
                                used with time differentiated phase data",
                UserWarning,
            )

        if unwr and m["header"]["spatialUnwrRange"]:
            data = unwrap(data, m["header"]["spatialUnwrRange"] / sensitivity, axis=1)

        if spikeThr is not None:
            data[np.abs(data) > spikeThr / sensitivity] = 0

    return data

@jit(nopython=True, parallel=True)
def detrend(y):
    """Function to remove linear trend from DAS data"""
    x = np.arange(len(y))
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    m = np.sum((x-x_mean)*(y-y_mean))/np.sum((x-x_mean)**2)
    b = y_mean - m * x_mean
    return y - (m*x + b)

@njit(parallel=True)
def preprocess_int(data, dt):
    """Faster integration"""
    nchan = data.shape[0]
    for idx in prange(nchan):
        data[idx, :] = np.cumsum(data[idx, :])*dt
    data[:, 0] = 0
    return data

@njit(parallel=True)
def detrend_2D(data):
    """Function to apply detrend to 2D data"""
    nCh = data.shape[0]
    x = np.arange(data.shape[1])
    x_mean = np.mean(x)
    for idx in prange(nCh):
        y = data[idx, :]
        y_mean = np.mean(y)
        m = np.sum((x-x_mean)*(y-y_mean))/np.sum((x-x_mean)**2)
        b = y_mean - m * x_mean
        data[idx, :] -= (m*x + b)
    return data

@njit(parallel=True)
def preprocess_diff(data, dt):
    """Faster differentiate"""
    nchan = data.shape[0]
    for idx in prange(nchan):
        data[idx, 1:] = np.diff(data[idx, :])/dt
    data[:, 0] = 0
    return data

@njit(parallel=True)
def preprocess_medfilt(data):
    """Jiaxuan implementation of parallel median filter"""
    nt = data.shape[1]
    for idx in prange(nt):
        data[:, idx] -= np.median(data[:, idx])
    return data


@njit(parallel=True)
def preprocess_unwrap(data, factor=1, threshold=0.99):
    """
    Correct the clipping caused by int32 overflow or underflow
    Args:
        data: raw count in int32 or converted micrsostrain
        factor: if raw count: factor=1
                if microstrain: factor that converts phase to microstrain
        threshold: threshold for detecting clipping
    Returns:
        data: unwrapped data; Note that data is modified in place
    """
    clip = 2**32 * factor
    clip_threshold = clip * threshold

    nx, nt = data.shape
    for ix in prange(nx):
        data_correction = np.zeros(nt-1, dtype=data.dtype)
        data_diff = np.diff(data[ix, :])
        idx_clip = data_diff < -clip_threshold
        data_correction[idx_clip] = 1.
        idx_clip = data_diff > clip_threshold
        data_correction[idx_clip] = -1.
        data[ix, 1:] += np.cumsum(data_correction) * clip
    return data


# Bandpass from obspy
def bandpass(data, freqmin, freqmax, df, corners=4, zerophase=False):
    """
    Butterworth-Bandpass Filter.

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        raise NotImplementedError("High-pass not supported")
        # msg = ("Selected high corner frequency ({}) of bandpass is at or "
        #        "above Nyquist ({}). Applying a high-pass instead.").format(
        #     freqmax, fe)
        # warnings.warn(msg)
        # return highpass(data, freq=freqmin, df=df, corners=corners,
        #                 zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)

def bandpass2D(data, freqmin, freqmax, df, corners=4, zerophase=False):
    """Function to bandpass 2D datasets [channels, time]"""
    dataBp = np.zeros_like(data)
    nch = dataBp.shape[0]
    for ich in range(nch):
        dataBp[ich, :] = bandpass(data[ich, :], freqmin, freqmax, df, corners, zerophase)
    return dataBp

def bandpass2D_vec(data, freqmin, freqmax, df, corners=4, zerophase=False):
    """
    Butterworth-Bandpass Filter.

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        raise NotImplementedError("High-pass not supported")
        # msg = ("Selected high corner frequency ({}) of bandpass is at or "
        #        "above Nyquist ({}). Applying a high-pass instead.").format(
        #     freqmax, fe)
        # warnings.warn(msg)
        # return highpass(data, freq=freqmin, df=df, corners=corners,
        #                 zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    if low == 0.0:
        z, p, k = iirfilter(corners, high, btype='lowpass',
                        ftype='butter', output='zpk')
    elif high == 0.0:
        z, p, k = iirfilter(corners, low, btype='highpass',
                        ftype='butter', output='zpk')
    else:
        z, p, k = iirfilter(corners, [low, high], btype='band',
                            ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis=1)
        return sosfilt(sos, firstpass[:,::-1], axis=1)[:,::-1]
    else:
        return sosfilt(sos, data)

def bandpass2D_c(data, freqmin, freqmax, dt, order=6, zerophase=False, nThreads=Ncores):
    """Function to bandpass 2D data along the fast axis using C++ code"""
    phase = 0 if zerophase else 1
    if data.dtype == np.float32:
        data_bp = pyDAS.lfilter(data, freqmin*dt, order, freqmax*dt, order, phase, nThreads)
    elif data.dtype == np.float64:
        data_bp = pyDAS.lfilter_double(data, freqmin*dt, order, freqmax*dt, order, phase, nThreads)
    else:
        raise ValueError("Array type not supported: input format %s"%data.dtype)
    return data_bp

#############################################################
################## READING FUNCTIONS ########################
#############################################################
def readFile_Segy(infile, nTraces, dataLen=4, endian=">", nTxtFileHeader=3200, nBinFileHeader = 400, nTraceHeader = 240):
    """Function to read entire SEGY file"""
    # Reading header information
    with open(infile, 'rb') as fid:
        # Reading number of samples per trace
        startData = nTxtFileHeader+84
        fid.seek(startData)
        BinBuffer = fid.read(4)
        nSamples = int.from_bytes(BinBuffer, byteorder='big', signed=False)
        # Reading sampling rate (micro-seconds)
        startData = nTxtFileHeader+17
        fid.seek(startData)
        BinBuffer = fid.read(1)
        fs = int.from_bytes(BinBuffer, byteorder='little', signed=False)*1e-6
        fs = 1.0 / fs
    data = np.zeros((nTraces, nSamples), dtype=np.float32)
    with open(infile, 'rb') as fid:
        for itrace in range(nTraces):
            startData = nTxtFileHeader+nBinFileHeader+nTraceHeader+(itrace)*(nTraceHeader+dataLen*nSamples)
            fid.seek(startData)
            BinDataBuffer = fid.read(nSamples*dataLen) # read binary bytes from file
            data[itrace, :] = struct.unpack_from(endian+('f')*nSamples, BinDataBuffer) # get data as a tuple of floats
            data[itrace, :] = detrend(data[itrace, :])
    MeanTrc = np.tile(np.mean(data,axis=0),(nTraces,1))
    data[:,:] -= MeanTrc
    return data, fs

def read_PASSCAL_SEGY_headers(infile, ending='big'):
    """Function to read information within SEGY PASSCAL headers"""
    nTxtFileHeader=3200
    nBinFileHeader=400
    nTraceHeader=240
    if infile.split(".")[-1] == "sgy":
        fid = open(infile, 'rb')
    elif infile.split(".")[-1] == "gz":
        fid = gzip.open(infile, 'rb')
    fid.seek(0,2)
    filesize = fid.tell()
    startData = nTxtFileHeader+20
    fid.seek(startData)
    BinBuffer = fid.read(2)
    nt = int.from_bytes(BinBuffer, byteorder=ending, signed=False)
    fid.seek(nTxtFileHeader+16)
    dt = int.from_bytes(fid.read(2), byteorder=ending, signed=False)*1e-6
    if dt == 0.0:
        fs = 0.0
    else:
        fs = 1.0/dt
    # Getting UTC time first sample
    fid.seek(nTxtFileHeader+nBinFileHeader+156)
    year = int.from_bytes(fid.read(2), byteorder=ending, signed=False)
    day = int.from_bytes(fid.read(2), byteorder=ending, signed=False)
    hour = int.from_bytes(fid.read(2), byteorder=ending, signed=False)
    minute = int.from_bytes(fid.read(2), byteorder=ending, signed=False)
    second = int.from_bytes(fid.read(2), byteorder=ending, signed=False)
    TIME_BASIS_CODE = int.from_bytes(fid.read(2), byteorder=ending, signed=False)
    micsec = int.from_bytes(fid.read(4), byteorder=ending, signed=False)
    second = second+micsec*1e-6
    startTime = datetime.datetime.strptime("%s-%s"%(year,day),"%Y-%j") + datetime.timedelta(hours=hour, minutes=minute, seconds=second)
    if TIME_BASIS_CODE == 4:
        startTime = startTime.replace(tzinfo=UTC)
    else:
        raise ValueError("Unknown time zone!")
    endTime = startTime + datetime.timedelta(seconds=nt*dt)
    nTraces = int((filesize - nTxtFileHeader - nBinFileHeader)/(nTraceHeader+nt*4))
    fid.close()
    return nt, fs, startTime, endTime, nTraces

def read_PASSCAL_segy(infile, nTraces, nSample, TraceOff=0):
    """Function to read PASSCAL segy raw data"""
    data = np.zeros((nTraces, nSample), dtype=np.float32)
    gzFile = False
    if infile.split(".")[-1] == "sgy":
        fid = open(infile, 'rb')
    elif infile.split(".")[-1] == "gz":
        gzFile = True
        fid = gzip.open(infile, 'rb')
    fid.seek(3600)
    # Skipping traces if necessary
    fid.seek(TraceOff*(240+nSample*4),1)
    # Looping over traces
    for ii in range(nTraces):
        fid.seek(240, 1)
        if gzFile:
            # np.fromfile does not work on gzip file
            BinDataBuffer = fid.read(nSample*4) # read binary bytes from file
            data[ii, :] = struct.unpack_from(">"+('f')*nSample, BinDataBuffer)
        else:
            data[ii, :] = np.fromfile(fid, dtype=">f", count=nSample)
    fid.close()
    return data


def readFile_HDF(filelist, fmin, fmax, desampling=True, taper=0.4, nChbuffer=1000, verbose=False, system=None, **kwargs):
    """
        Function to read and pre-process list of HDF5 files using a trace buffer
        INPUTS:
            - filelist [list]: list of files to be read whose data are assumed contiguous
            - fmin [float]: minimum frequency to be retained in the processed data
            - fmax [float]: maximum frequency to be retained in the processed data
            - desampling [boolean]: desampling the processed data to 2.5*fmax
            - nChbuffer [int]: size of the channel buffer to use to read and process the raw data
            - verbose [boolean]: level of verbosity
            - system [string]: name of the system recording the data to convert raw amplitudes to micro strain (suppported: OptaSense, ASN)
        OUTPUT:
            - DAS_data [2D numpy array]: array of the data (channels,time samples)
            - info: header dict
            - fs [float]: sampling rate
            - dt [float]: sampling interval
            - nt [int]:   sampling number of points
            - dx [int]:   channel distance
            - nx [int]:   number of channels
            - begTime [iso format datetime str]:  record span begin time
            - endTime [iso format datetime str]:  record span end time
    """
    nFiles = len(filelist)
    fsRaw = np.zeros(nFiles, dtype=float)
    ntRaw = np.zeros(nFiles, dtype=int)
    nChRaw = np.zeros(nFiles, dtype=int)
    nChSamp = np.zeros(nFiles, dtype=float)
    zfpComp = np.zeros(nFiles, dtype=bool)
    polarity = 1.0 # Polarity factor to applied depending on instrument

    # kwargs parameters
    order = kwargs.get("order", 4)
    zerophase = kwargs.get("zerophase", True)
    removeMedian = kwargs.get("median", True)
    unwrap = kwargs.get("unwrap", False)
    diff = kwargs.get("diff", False)
    detrend = kwargs.get("detrend", True)
    tapering = kwargs.get("tapering", True)
    filter_app = kwargs.get("filter", True)

    # First reading sampling parameters
    for idx, ifile in enumerate(filelist):
        with h5py.File(ifile,'r') as fid:
            # Checking type of format
            if 'Data' in fid:
                dataKind = "Proc"
            elif "acqSpec" in fid.keys():
                dataKind = "ASN"
            elif "Acquisition" in fid.keys():
                dataKind = "OptaSense"
            else:
                raise ValueError("Unknown formatted HDF5 for %s"%ifile)
            if dataKind == "Proc":
                # Converting ping period from nanoseconds to seconds and compute sampling rate
                fsRaw[idx] = fid['Data'].attrs["fs"]
                ntRaw[idx] = fid['Data'].attrs["nt"]
                nChRaw[idx] = fid['Data'].attrs["nCh"]
                attrs_names = fid['Data'].attrs.keys()
                # Possible key name for channel sampling interval
                if 'dCh' in attrs_names:
                    nChSamp[idx] = fid['Data'].attrs["dCh"]
                elif 'ChSamp' in attrs_names:
                    nChSamp[idx] = fid['Data'].attrs["ChSamp"]
                # Check if data is ZFP compressed
                if "ZFPtolerance" in attrs_names:
                    zfpComp[idx] = True
            elif dataKind == "OptaSense":
                # Converting ping period from nanoseconds to seconds and compute sampling rate
                desampleFactor = fid['Acquisition']['Custom'].attrs.get("Decimation Factor")
                fsRaw[idx] = fid['Acquisition'].attrs.get("PulseRate")/desampleFactor
                ntRaw[idx] = len(fid['Acquisition']['Raw[0]']['Custom']['SampleCount'][:])
                nChRaw[idx] = fid['Acquisition']["Custom"].attrs.get("Num Output Channels")
                nChSamp[idx] = fid['Acquisition'].attrs.get("SpatialSamplingInterval")
            elif "acqSpec" in fid.keys():
                fsRaw[idx] = 1.0/fid["header"].get("dt")[()]
                ntRaw[idx] = fid["data"].shape[0]
                nChRaw[idx] = fid["data"].shape[1]
                nChSamp[idx] = np.diff(fid["header"].get("channels")[:])[0]*fid["header"].get("dx")[()]

    # Checking if all files have same number of channels and sampling rate
    if not np.all(fsRaw == fsRaw[0]):
        raise ValueError("Data do not have same sampling rate!")
    if not np.any(nChRaw == nChRaw[0]):
        raise ValueError("Data do not have same number of channels!")
    if not np.any(nChSamp == nChSamp[0]):
        raise ValueError("Data do not have same channel sampling!")
    fsRaw = fsRaw[0]
    nCh = nChRaw[0]
    nChSamp = nChSamp[0]
    if any(zfpComp):
        # Changing nChbuffer to read all channels to speedup reading compressed data
        nChbuffer = nCh
    # For reading fewer channels from original data
    min_ch = int(kwargs.get("min_ch",0))
    max_ch = int(kwargs.get("max_ch",nCh))
    if max_ch > nCh:
        raise ValueError("Maximum channel number (max_ch=%s) greater than total number of channels (nCh=%s)!"%(max_ch,nCh))
    nCh = max_ch - min_ch
    # Setting size of channel buffer based on number of channels to read
    nChbuffer = min(nChbuffer,nCh)
    # Total number of time samples
    ntRawTot = np.sum(ntRaw)
    # Getting additional parameters
    minTime = kwargs.get("minTime", None) # Datetime value of the minimum request time
    maxTime = kwargs.get("maxTime", None) # Datetime value of the maximum request time
    if minTime is not None and maxTime is not None:
        if minTime >= maxTime:
            raise ValueError("minTime smaller or equal than maxTime! Change parameter values!")
        intervalSec = (maxTime - minTime) / datetime.timedelta(seconds=1)
        ntRawTot = np.ceil(intervalSec*fsRaw).astype(int)
    # Determine whether to desample or not the data
    if desampling:
        newNy = 2.5 * fmax
        if newNy < fsRaw:
            fsRatio = int(fsRaw/newNy)
            dfactor = kwargs.get("dfactor", None)
            if dfactor is not None and dfactor < fsRatio:
                fsRatio = dfactor
            fs = fsRaw/fsRatio
            nt = int(ntRawTot/fsRatio+0.5)
        else:
            if verbose:
                print("WARNING! Cannot desample")
            nt = ntRawTot
            fs = fsRaw
            fsRatio = 1
    else:
        nt = ntRawTot
        fs = fsRaw
        fsRatio = 1
    # Allocating memory
    DAS_data = np.zeros((nCh, nt), dtype=np.float32)
    trace_buffer = np.zeros((nChbuffer, ntRawTot), dtype=np.float32)
    w_taper = tukey(ntRawTot, alpha=taper)
    nChunks = np.ceil(nCh/nChbuffer).astype(int)
    itraces = 0
    if verbose:
        rng = tqdm.tqdm(range(nChunks), desc="Processing data...")
    else:
        rng = range(nChunks)
    for _ in rng:
        # Pointer to initial time sample
        itime = 0
        # Getting number of traces to process
        ntraces = min(nChbuffer, nCh-itraces)
        ntRawLeft = np.copy(ntRawTot)
        # Reading raw data from all files
        for idx, ifile in enumerate(filelist):
            with h5py.File(ifile,'r') as fid:
                UnWrapDiff = True
                # Checking type of format
                if 'Data' in fid:
                    dataKind = "Proc"
                    # Checking if data was derived from ASN instrument
                    if "acqSpec.YvsXDelay" in fid["Acquisition_origin"].attrs:
                        UnWrapDiff = False
                elif "acqSpec" in fid.keys():
                    dataKind = "ASN"
                elif "Acquisition" in fid.keys():
                    dataKind = "OptaSense"
                else:
                    raise ValueError("Unknown formatted HDF5 for %s"%ifile)

                if dataKind == "Proc":
                    startTime = dateutil.parser.parse(fid["Data"].attrs["startTime"])
                    RawDataTime = np.array([(startTime + datetime.timedelta(seconds=ii/fsRaw)).timestamp() for ii in range(ntRaw[idx])])
                elif dataKind == "OptaSense":
                    RawDataTime = fid["Acquisition"]["Raw[0]"]["RawDataTime"][:].astype(np.float64)*1e-6
                    startTime = datetime.datetime.fromtimestamp(RawDataTime[0], tz=UTC)
                elif dataKind == "ASN":
                    RawDataTime = fid["header"].get("time")[()] + fid["timing"].get("sampleSkew")[()]+ np.arange(ntRaw[idx])/fsRaw
                    startTime = datetime.datetime.fromtimestamp(RawDataTime[0], tz=UTC)

                if idx == 0:
                    begTime = deepcopy(startTime)
                # Reading time interval
                it_min = 0
                it_max = ntRaw[idx]
                if it_max == 0:
                    # case for an empty file
                    print(f'Warning: {ifile} has zero samples, skip this file')
                    continue
                # Reading only data interval
                if minTime is not None and maxTime is not None:
                    it_min = max(0, np.argmin(abs(RawDataTime-minTime.timestamp())))
                    it_max = min(ntRaw[idx], it_min+ntRawLeft)
                    ntRawLeft = ntRawLeft - it_max + it_min
                    if idx == 0:
                        if it_min > 0:
                            begTime = datetime.datetime.fromtimestamp(RawDataTime[it_min], tz=UTC)
                        else:
                            begTime = deepcopy(startTime)
                ntSamp = it_max - it_min

                # Reading data
                if dataKind == "Proc":
                    # Decompress data if necessary
                    if zfpComp[idx]:
                        data = DAScompress.zfp_decompress(fid["Data"][:], (nCh, ntRaw[idx]), np.dtype("float32"), fid["Data"].attrs["ZFPtolerance"])
                        # Overwriting system and diff variables (data must be already scaled to microstrain rate when compressed)
                        system = None
                        diff = False
                    else:
                        data = fid['Data']
                    # Identify time axis
                    time_pos = data.shape == ntRaw[idx]
                    if time_pos[1]:
                        trace_buffer[:ntraces,itime:itime+ntSamp] = data[min_ch+itraces:min_ch+itraces+ntraces, it_min:it_max]
                    else:
                        trace_buffer[:ntraces,itime:itime+ntSamp] = data[it_min:it_max,min_ch+itraces:min_ch+itraces+ntraces].T
                elif dataKind == "OptaSense":
                    # Identify time axis
                    time_pos = fid['Acquisition']['Raw[0]']['RawData'].shape == ntRaw[idx]
                    if time_pos[1]:
                        trace_buffer[:ntraces,itime:itime+ntSamp] = fid['Acquisition']['Raw[0]']['RawData'][min_ch+itraces:min_ch+itraces+ntraces, it_min:it_max]
                    else:
                        trace_buffer[:ntraces,itime:itime+ntSamp] = fid['Acquisition']['Raw[0]']['RawData'][it_min:it_max,min_ch+itraces:min_ch+itraces+ntraces].T
                elif dataKind == "ASN":
                    trace_buffer[:ntraces,itime:itime+ntSamp] = readASNfile(ifile, firstCh=min_ch+itraces, lastCh=min_ch+itraces+ntraces, firstSample=it_min, lastSample=it_max).T
                itime += ntSamp
        # unwrap to correct int32 overflow / underflow
        if unwrap:
            if dataKind != "ASN" and UnWrapDiff:
                trace_buffer = preprocess_unwrap(trace_buffer, factor=1)
        # differentiate data
        if diff:
            if dataKind != "ASN" and UnWrapDiff:
                trace_buffer = preprocess_diff(trace_buffer, 1/fsRaw)
        # Detrending data
        if detrend:
            trace_buffer =  detrend_2D(trace_buffer)
        # Tapering
        if tapering:
            trace_buffer = trace_buffer*w_taper
        # Bandpassing the data
        if filter_app:
            trace_buffer = bandpass2D_c(trace_buffer, fmin, fmax, 1.0/fsRaw, order=order, zerophase=zerophase)
        # desampling the data if desampling is True and valid
        DAS_data[itraces:itraces+ntraces, :] = (trace_buffer[:ntraces, ::fsRatio])[:,:nt]
        itraces += ntraces
        # time_done_process = time.time()
    if removeMedian:
        DAS_data = preprocess_medfilt(DAS_data)
    # Converting raw amplitude to micro-strain
    # System here is used to indicate instrument type
    if system is not None:
        if system == "OptaSense":
            with h5py.File(filelist[0],'r') as fid:
                preproc = True if 'Data' in fid else False
                if "Acquisition_origin" in fid or "Acquisition" in fid:
                    if preproc:
                        G = fid["Acquisition_origin"].attrs.get("GaugeLength")
                        n = fid["Acquisition_origin"].attrs.get("Fibre Refractive Index")
                        lamd = fid["Acquisition_origin"].attrs.get("Laser Wavelength (nm)")*1e-9
                        # Polarity check
                        fpgaDRN = fid['Acquisition_origin'].attrs.get("FPGA Drawing Number")
                        fpgaVersion = float(fid['Acquisition_origin'].attrs.get("FPGA Version").decode('utf-8'))
                        if (fpgaDRN == 7804701) and (fpgaVersion <= 2.0):
                            polarity = -1.0
                    else:
                        G = fid['Acquisition'].attrs.get("GaugeLength")
                        n = fid['Acquisition']['Custom'].attrs.get("Fibre Refractive Index")
                        lamd = fid['Acquisition']['Custom'].attrs.get("Laser Wavelength (nm)")*1e-9
                        fpgaDRN = fid['Acquisition']['Custom'].attrs.get("FPGA Drawing Number")
                        fpgaVersion = float(fid['Acquisition']['Custom'].attrs.get("FPGA Version").decode('utf-8'))
                        if (fpgaDRN == 7804701) and (fpgaVersion <= 2.0):
                            polarity = -1.0
                    eta = 0.78 # photo-elastic scaling factor for longitudinal strain in isotropic material
                    factor = 4.0*np.pi*eta*n*G/lamd
                    # Conversion factor from raw to delta phase
                    radconv = 2**15 / np.pi # 10430.378350470453
                    DAS_data = DAS_data/factor/radconv*1e6
                else:
                    print("WARNING! Missing GaugeLength, Fibre Refractive Index, and Laser Wavelength info from metadata. Skipping data strain conversion!")
        elif system == "ASN":
            # ASN processed data already converted to strain/s, just multiply by 1e6 to obtain microstrain/s
            DAS_data *= 1e6
        else:
            print("WARNING! %s is not a known system"%system)
    # header info
    info = {}
    info['dx'] = nChSamp
    info['dt'] = 1/fs
    info['fs'] = fs
    info['nt'] = nt
    info['nx'] = nCh
    info['begTime'] = begTime
    info['endTime'] = begTime + datetime.timedelta(seconds=nt/fs)
    return DAS_data*polarity, info

