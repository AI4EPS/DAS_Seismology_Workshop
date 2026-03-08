# Module containing the definition of an abstract, in-core, and out-of-core vectors
import math
import os
import re
import time
from copy import deepcopy
from shutil import copyfile
from sys import version_info

import numpy as np
import pickle, h5py
import sep_util
# other modules
import sys_util

# regex to read output of Solver_ops
re_dpr = re.compile("DOT RESULT(.*)")


# Vector class and derived classes
class vector:
    """Abstract python vector class"""

    def __init__(self):
        """Default constructor"""

    def __del__(self):
        """Default destructor"""

    def __add__(self, other):  # self + other
        if type(other) in [int, float]:
            self.addbias(other)
            return self
        elif isinstance(other, vector):
            self.scaleAdd(other)
            return self
        else:
            raise TypeError('Argument has to be either scalar or vector, got %r instead' % other)

    def __sub__(self, other):  # self - other
        self.__add__(-other)
        return self

    def __neg__(self):  # -self
        self.scale(-1)
        return self

    def __mul__(self, other):  # self * other
        if type(other) in [int, float]:
            self.scale(other)
            return self
        elif isinstance(other, vector):
            self.multiply(other)
            return self
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        if type(other) in [int, float]:
            self.scale(other)
            return self
        elif isinstance(other, vector):
            self.multiply(other)
            return self
        else:
            raise NotImplementedError

    def __pow__(self, power, modulo=None):
        if type(power) in [int, float]:
            self.pow(power)
        else:
            raise TypeError('power has to be a scalar')

    def __abs__(self):
        self.abs()

    def __truediv__(self, other):  # self / other
        if type(other) in [int, float]:
            self.scale(1 / other)
        elif isinstance(other, vector):
            self.multiply(other.clone().reciprocal())
        else:
            raise TypeError('other has to be either a scalar or a vector')

    # Class vector operations
    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        raise NotImplementedError("getNdArray must be overwritten")

    def shape(self):
        """Property to get the vector shape (number of samples for each axis)"""
        raise NotImplementedError("shape must be overwritten")

    def size(self):
        """Property to compute the vector size (number of samples)"""
        raise NotImplementedError("size must be overwritten")

    def norm(self, N=2):
        """Function to compute vector N-norm"""
        raise NotImplementedError("norm must be overwritten")

    def zero(self):
        """Function to zero out a vector"""
        raise NotImplementedError("zero must be overwritten")

    def max(self):
        """Function to obtain maximum value within a vector"""
        raise NotImplementedError("max must be overwritten")

    def min(self):
        """Function to obtain minimum value within a vector"""
        raise NotImplementedError("min must be overwritten")

    def set(self, val):
        """Function to set all values in the vector"""
        raise NotImplementedError("set must be overwritten")

    def scale(self, sc):
        """Function to scale a vector"""
        raise NotImplementedError("scale must be overwritten")

    def addbias(self, bias):
        """Function to add bias to a vector"""
        raise NotImplementedError("addbias must be overwritten")

    def rand(self):
        """Function to randomize a vector"""
        raise NotImplementedError("rand must be overwritten")

    def clone(self):
        """Function to clone (deep copy) a vector from a vector or a Space"""
        raise NotImplementedError("clone must be overwritten")

    def cloneSpace(self):
        """Function to clone vector space"""
        raise NotImplementedError("cloneSpace must be overwritten")

    def checkSame(self):
        """Function to check to make sure the vectors exist in the same space"""
        raise NotImplementedError("checkSame must be overwritten")

    # def writeVec(self, filename, mode='w'):
    #     """Function to write vector to file"""
    #     raise NotImplementedError("writeVec must be overwritten")
    def writeVec(self, filename, mode='w'):
        """Function to write vector to file"""
        # Check writing mode
        if not mode in 'wa':
            raise ValueError("Mode must be appending 'a' or writing 'w' ")
        # Construct ax_info if the object has getHyper
        if hasattr(self, "getHyper"):
            hyper = self.getHyper()
            self.ax_info = []
            for iaxis in range(hyper.getNdim()):
                self.ax_info.append([hyper.getAxis(iaxis + 1).n,
                                     hyper.getAxis(iaxis + 1).o,
                                     hyper.getAxis(iaxis + 1).d,
                                     hyper.getAxis(iaxis + 1).label])
        
        # check output file type
        _, ext = os.path.splitext(filename)  # file extension
        
        # SEP vector with header in filename.H and binary in DATAPATH/filename.H@
        if ext == ".H":
            # writing header/pointer file if not present and not append mode
            if not (os.path.isfile(filename) and mode in 'a'):
                binfile = sep_util.datapath + filename.split('/')[-1] + '@'
                with open(filename, mode) as f:
                    # Writing axis info
                    if self.ax_info:
                        for ii, ax_info in enumerate(self.ax_info):
                            ax_id = ii + 1
                            f.write("n%s=%s o%s=%s d%s=%s label%s='%s'\n" % (
                                ax_id, ax_info[0], ax_id, ax_info[1], ax_id, ax_info[2], ax_id, ax_info[3]))
                    else:
                        for ii, n_axis in enumerate(tuple(reversed(self.shape))):
                            ax_id = ii + 1
                            f.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (ax_id, n_axis, ax_id, ax_id))
                    # Writing last axis for allowing appending (unless we are dealing with a scalar)
                    if self.shape != (1,):
                        ax_id = self.ndims + 1
                        f.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (ax_id, 1, ax_id, ax_id))
                    f.write("in='%s'\n" % (binfile))
                    esize = "esize=4\n"
                    if self.getNdArray().dtype == np.complex64:
                        esize = "esize=8\n"
                    f.write(esize)
                    f.write("data_format=\"native_float\"\n")
                f.close()
            else:
                binfile = sep_util.get_binary(filename)
                if (mode in 'a'):
                    axes = sep_util.get_axes(filename)
                    # Number of vectors already present in the file
                    if self.shape == (1,):
                        n_vec = axes[0][0]
                        append_dim = self.ndims
                    else:
                        n_vec = axes[self.ndims][0]
                        append_dim = self.ndims + 1
                    with open(filename, mode) as f:
                        f.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (append_dim, n_vec + 1, append_dim, append_dim))
                    f.close()
            # Writing binary file
            fmt = '>f'
            if self.getNdArray().dtype == np.complex64 or self.getNdArray().dtype == np.complex128:
                format = '>c8'
            with open(binfile, mode + 'b') as f:
                # Writing big-ending floating point number
                if np.isfortran(self.getNdArray()):  # Forcing column-wise binary writing
                    # self.getNdArray().flatten('F').astype(fmt,copy=False).tofile(fid)
                    self.getNdArray().flatten('F').tofile(f, format=fmt)
                else:
                    # self.getNdArray().astype(fmt,order='C',subok=False,copy=False).tofile(fid)
                    self.getNdArray().tofile(f, format=fmt)
            f.close()
        
        # numpy dictionary
        elif ext == '.npy':
            if mode not in 'a':
                if self.ax_info:  # TODO fix saving of ax_info
                    axes = dict()
                    for ii, ax_info in enumerate(self.ax_info):
                        axes['%s' %ii+1] = dict(n=ax_info[0], o=ax_info[1], d=ax_info[2], label=ax_info[3])
                        
                    np.save(file=filename,
                            arr=dict(arr=self.getNdArray(),
                                     ax_info=axes),
                            allow_pickle=True)
                else:
                    np.save(file=filename, arr=self.getNdArray(), allow_pickle=False)
        
        elif ext == '.h5':  # TODO implement saving to hdf5
            # if mode not in 'a':
            #     with h5py.File(filename, 'wb') as f:
            #         dset = f.create_dataset("vec", data=self.getNdArray())
            # else:
            raise NotImplementedError
        
        elif ext == '.pkl':
            raise NotImplementedError
            # # Save the whole vector in a pickle file
            # if mode not in 'a':
            #     with open(filename, 'wb') as f:
            #         pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            #     f.close()
            # else:
            #     raise NotImplementedError
            
        else:
            raise ValueError("ERROR! Output format has to be H, npy, h5 or pkl")
            
        return

    # TODO implement on seplib
    def abs(self):
        """Return a vector containing the absolute values"""
        raise NotImplementedError('abs method must be implemented')

    # TODO implement on seplib
    def sign(self):
        """Return a vector containing the signs"""
        raise NotImplementedError('sign method have to be implemented')

    # TODO implement on seplib
    def reciprocal(self):
        """Return a vector containing the reciprocals of self"""
        raise NotImplementedError('reciprocal method must be implemented')

    # TODO implement on seplib
    def maximum(self, vec2):
        """Return a new vector of element-wise maximum of self and vec2"""
        raise NotImplementedError('maximum method must be implemented')

    # TODO implement on seplib
    def conj(self):
        """Compute conjugate transpose of the vector"""
        raise NotImplementedError('conj method must be implemented')

    # TODO implement on seplib
    def pow(self, power):
        """Compute element-wise power of the vector"""
        raise NotImplementedError('pow method must be implemented')

    # TODO implement on seplib
    def real(self):
        """Return the real part of the vector"""
        raise NotImplementedError('real method must be implemented')

    # TODO implement on seplib
    def imag(self):
        """Return the imaginary part of the vector"""
        raise NotImplementedError('imag method must be implemented')

    # Combination of different vectors

    def copy(self, vec2):
        """Function to copy vector"""
        raise NotImplementedError("copy must be overwritten")

    def scaleAdd(self, vec2, sc1=1.0, sc2=1.0):
        """Function to scale two vectors and add them to the first one"""
        raise NotImplementedError("scaleAdd must be overwritten")

    def dot(self, vec2):
        """Function to compute dot product between two vectors"""
        raise NotImplementedError("dot must be overwritten")

    def multiply(self, vec2):
        """Function to multiply element-wise two vectors"""
        raise NotImplementedError("multiply must be overwritten")

    def isDifferent(self, vec2):
        """Function to check if two vectors are identical"""

        raise NotImplementedError("isDifferent must be overwritten")

    def clipVector(self, low, high):
        """
           Function to bound vector values based on input vectors min and max
        """
        raise NotImplementedError("clipVector must be overwritten")


# Set of vectors (useful to store results and same-Space vectors together)
class vectorSet:
    """Class to store different vectors that live in the same Space"""

    def __init__(self):
        """Default constructor"""
        self.vecSet = []  # List of vectors of the set

    def __del__(self):
        """Default destructor"""

    def append(self, vec_in, copy=True):
        """Method to add vector to the set"""
        # Checking dimensionality if a vector is present
        if self.vecSet:
            if not self.vecSet[0].checkSame(vec_in):
                raise ValueError("ERROR! Provided vector not in the same Space of the vector set")

        self.vecSet.append(vec_in.clone()) if copy else self.vecSet.append(vec_in)

    def writeSet(self, filename, mode="a"):
        """Method to write to SEPlib file (by default it appends vectors to file)"""
        if mode not in "aw":
            raise ValueError("ERROR! mode must be either a (append) or w (write)")
        for iVec, vec_i in enumerate(self.vecSet):
            if mode == "w" and iVec == 0:
                wr_mode = "w"
            else:
                wr_mode = "a"
            vec_i.writeVec(filename, wr_mode)
        self.vecSet = []  # List of vectors of the set


class superVector(vector):

    def __init__(self, *args):
        """
        superVector constructor
        :param args: vectors or superVectors or vectors list objects
        """
        super(superVector, self).__init__()

        self.vecs = []
        for v in args:
            if v is None:
                continue
            elif isinstance(v, vector):
                self.vecs.append(v)
            elif isinstance(v, list):
                for vv in v:
                    if vv is None:
                        continue
                    elif isinstance(vv, vector):
                        self.vecs.append(vv)
            else:
                raise TypeError('Argument must be either a vector or a superVector')

        self.n = len(self.vecs)

    def __del__(self):
        """superVector destructor"""
        del self.vecs, self.n

    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        return [self.vecs[idx].getNdArray() for idx in range(self.n)]

    @property
    def shape(self):
        return [self.vecs[idx].shape for idx in range(self.n)]

    @property
    def size(self):
        return sum([self.vecs[idx].size for idx in range(self.n)])

    def norm(self, N=2):
        """Function to compute vector N-norm"""
        norm = np.power([self.vecs[idx].norm(N) for idx in range(self.n)], N)
        return np.power(sum(norm), 1. / N)

    def set(self, val):
        """Function to set all values in the vector"""
        for idx in range(self.n):
            self.vecs[idx].set(val)
        return self

    def zero(self):
        """Function to zero out a vector"""
        for idx in range(self.n):
            self.vecs[idx].zero()
        return self

    def max(self):
        """Function to obtain maximum value within a vector"""
        return np.max([self.vecs[idx].max() for idx in range(self.n)])

    def min(self):
        """Function to obtain minimum value within a vector"""
        return np.min([self.vecs[idx].min() for idx in range(self.n)])

    def scale(self, sc):
        """Function to scale a vector"""
        if type(sc) is not list:
            sc = [sc] * self.n
        for idx in range(self.n):
            self.vecs[idx].scale(sc[idx])
        return self

    def addbias(self, bias):
        """Add a constant to the vector"""
        if type(bias) is not list:
            bias = [bias] * self.n
        for idx in range(self.n):
            self.vecs[idx].addbias(bias[idx])
        return self

    def rand(self, snr=1.0):
        """Function to randomize a vector"""
        for idx in range(self.n):
            self.vecs[idx].rand()
        return self

    def clone(self):
        """Function to clone (deep copy) a vector from a vector or a Space"""
        vecs = [self.vecs[idx].clone() for idx in range(self.n)]
        return superVector(vecs)

    def cloneSpace(self):
        """Function to clone vector space"""
        return superVector([self.vecs[idx].cloneSpace() for idx in range(self.n)])

    def checkSame(self, other):
        """Function to check to make sure the vectors exist in the same space"""
        # Checking type
        if not isinstance(other, superVector):
            raise TypeError('Input variable is not a superVector')
        checkspace = np.asarray([self.vecs[idx].checkSame(other.vecs[idx]) for idx in range(self.n)])
        notsame = np.where(checkspace is False)[0]
        for v in notsame:
            raise Warning('Component %d not in the same space!' % v)
        return np.all(checkspace == True)

    # Combination of different vectors
    def copy(self, vecs_in):
        """Function to copy vector from input vector"""
        # Checking type
        if type(vecs_in) is not superVector:
            raise TypeError("Input variable is not a superVector")
        # Checking dimensionality
        if not self.checkSame(vecs_in):
            raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
        for idx in range(self.n):
            self.vecs[idx].copy(vecs_in.vecs[idx])
        return self

    def scaleAdd(self, vecs_in, sc1=1.0, sc2=1.0):
        """Function to scale input vectors and add them to the original ones"""
        # Checking type
        if type(vecs_in) is not superVector:
            raise TypeError("Input variable is not a superVector")
        # Checking dimensionality
        if not self.checkSame(vecs_in):
            raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
        for idx in range(self.n):
            self.vecs[idx].scaleAdd(vecs_in.vecs[idx], sc1, sc2)
        return self

    def dot(self, vecs_in):
        """Function to compute dot product between two vectors"""
        # Checking type
        if type(vecs_in) is not superVector:
            raise TypeError("Input variable is not a superVector")
        # Checking dimensionality
        if not self.checkSame(vecs_in):
            raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
        return np.sum([self.vecs[idx].dot(vecs_in.vecs[idx]) for idx in range(self.n)])

    def multiply(self, vecs_in):
        """Function to multiply element-wise two vectors"""
        # Checking type
        if type(vecs_in) is not superVector:
            raise TypeError("Input variable is not a superVector")
        # Checking dimensionality
        if not self.checkSame(vecs_in):
            raise ValueError("ERROR! Dimensionality mismatching between given superVectors")
        for idx in range(self.n):
            self.vecs[idx].multiply(vecs_in.vecs[idx])
        return self

    def isDifferent(self, vecs_in):
        """Function to check if two vectors are identical"""
        # Checking type
        if type(vecs_in) is not superVector:
            raise TypeError("Input variable is not a superVector")
        return any([self.vecs[idx].isDifferent(vecs_in.vecs[idx]) for idx in range(self.n)])

    def clipVector(self, lows, highs):
        for idx in range(self.n):
            self.vecs[idx].clipVector(lows[idx], highs[idx])
        return self

    def abs(self):
        for idx in range(self.n):
            self.vecs[idx].abs()
        return self

    def sign(self):
        for idx in range(self.n):
            self.vecs[idx].sign()
        return self

    def reciprocal(self):
        for idx in range(self.n):
            self.vecs[idx].reciprocal()
        return self

    def maximum(self, other):
        if np.isscalar(other):
            for idx in range(self.n):
                self.vecs[idx].maximum(other)
            return self
        elif type(other) is not superVector:
            raise TypeError("Input variable is not a superVector")
        if other.n != self.n:
            raise ValueError('Input must have the same length of self')
        for idx in range(self.n):
            self.vecs[idx].maximum(other.vecs[idx])
        return self

    def conj(self):
        for idx in range(self.n):
            self.vecs[idx].conj()
        return self

    def real(self):
        for idx in range(self.n):
            self.vecs[idx].real()
        return self

    def imag(self,):
        for idx in range(self.n):
            self.vecs[idx].imag()
        return self

    def pow(self, power):
        for idx in range(self.n):
            self.vecs[idx].pow(power)
        return self

    def writeVec(self, filename, mode="w"):
        """Method to write to vector to file within a Vector set"""
        _, ext = os.path.splitext(filename)
        for ii, vec_cmp in enumerate(self.vecs):
            # Writing components to different files
            filename_cmp = ".".join(filename.split('.')[:-1]) + "_comp%s" % (ii + 1)
            # Writing files (recursively)
            vec_cmp.writeVec(filename_cmp+ext, mode)
        return


class vectorIC(vector):
    """In-core python vector class"""

    def __init__(self, in_vec):
        """
        VectorIC constructor: arr=np.array
        This class stores array with C memory order (i.e., row-wise sorting)
        """

        # Verify that input is a numpy array or header file or vectorOC
        if isinstance(in_vec, vectorOC):  # VectorOC passed to constructor
            self.arr, self.ax_info = sep_util.read_file(in_vec.vecfile)
        elif isinstance(in_vec, str):  # Header file passed to constructor
            self.arr, self.ax_info = sep_util.read_file(in_vec)
        elif isinstance(in_vec, np.ndarray):  # Numpy array passed to constructor
            if np.isfortran(in_vec):
                raise TypeError('Input array not a C contiguous array!')
            self.arr = np.array(in_vec, copy=False)
            self.ax_info = None
        elif isinstance(in_vec, tuple):  # Tuple size passed to constructor
            self.arr = np.zeros(tuple(reversed(in_vec)))
            self.ax_info = None
        else:  # Not supported type
            raise ValueError("ERROR! Input variable not currently supported!")

        # Number of elements per axis (tuple). Checking also the memory order
        self.shape = self.arr.shape  # If fortran the first axis is the "fastest"
        self.ndims = len(self.shape)  # Number of axes integer
        self.size = self.arr.size # Total number of elements
        super(vectorIC, self).__init__()

    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        return self.arr

    def size(self):
        return self.getNdArray().size

    def norm(self, N=2):
        """Function to compute vector N-norm using Numpy"""
        return np.linalg.norm(self.getNdArray().flatten(), ord=N)

    def zero(self):
        """Function to zero out a vector"""
        self.getNdArray().fill(0)
        return self

    def max(self):
        """Function to obtain maximum value in the vector"""
        return self.getNdArray().max()

    def min(self):
        """Function to obtain minimum value in the vector"""
        return self.getNdArray().min()

    def set(self, val):
        """Function to set all values in the vector"""
        self.getNdArray().fill(val)
        return self

    def scale(self, sc):
        """Function to scale a vector"""
        self.getNdArray()[:] *= sc
        return self

    def addbias(self, bias):
        self.getNdArray()[:] += bias
        return self

    def rand(self, snr=1.):
        """Fill vector with random number (~U[1,-1]) with a given SNR"""
        rms = np.sqrt(np.mean(np.square(self.getNdArray())))
        amp_noise = 1.0
        if rms != 0.:
            amp_noise = math.sqrt(3. / snr) * rms  # sqrt(3*Power_signal/SNR)
        self.getNdArray()[:] = amp_noise * (2. * np.random.random(self.getNdArray().shape) - 1.)
        return self

    def clone(self):
        """Function to clone (deep copy) a vector from a vector or a Space"""
        vec_clone = deepcopy(self)  # Deep clone of vector
        # Checking if a vector space was provided
        if vec_clone.getNdArray().size == 0:
            vec_clone.arr = np.zeros(vec_clone.shape, dtype=self.getNdArray().dtype)
        return vec_clone

    def cloneSpace(self):
        """Function to clone vector space only (vector without actual vector array by using empty array of size 0)"""
        vec_space = vectorIC(np.empty(0, dtype=self.getNdArray().dtype))
        # Cloning space of input vector
        vec_space.ndims = self.ndims
        vec_space.shape = self.shape
        vec_space.size = self.size
        return vec_space

    def checkSame(self, other):
        """Function to check dimensionality of vectors"""
        return self.shape == other.shape

    # def writeVec(self, filename, mode='w'):
    #     """Function to write vector to file"""
    #     # Check writing mode
    #     if not mode in 'wa':
    #         raise ValueError("Mode must be appending 'a' or writing 'w' ")
    #     # Construct ax_info if the object has getHyper
    #     if hasattr(self, "getHyper"):
    #         hyper = self.getHyper()
    #         self.ax_info = []
    #         for iaxis in range(hyper.getNdim()):
    #             self.ax_info.append([hyper.getAxis(iaxis + 1).n, hyper.getAxis(iaxis + 1).o, hyper.getAxis(iaxis + 1).d,
    #                                  hyper.getAxis(iaxis + 1).label])
    #     # writing header/pointer file if not present and not append mode
    #     if not (os.path.isfile(filename) and mode in 'a'):
    #         binfile = sep_util.datapath + filename.split('/')[-1] + '@'
    #         with open(filename, mode) as fid:
    #             # Writing axis info
    #             if self.ax_info:
    #                 for ii, ax_info in enumerate(self.ax_info):
    #                     ax_id = ii + 1
    #                     fid.write("n%s=%s o%s=%s d%s=%s label%s='%s'\n" % (
    #                         ax_id, ax_info[0], ax_id, ax_info[1], ax_id, ax_info[2], ax_id, ax_info[3]))
    #             else:
    #                 for ii, n_axis in enumerate(tuple(reversed(self.shape))):
    #                     ax_id = ii + 1
    #                     fid.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (ax_id, n_axis, ax_id, ax_id))
    #             # Writing last axis for allowing appending (unless we are dealing with a scalar)
    #             if self.shape != (1,):
    #                 ax_id = self.ndims + 1
    #                 fid.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (ax_id, 1, ax_id, ax_id))
    #             fid.write("in='%s'\n" % (binfile))
    #             esize = "esize=4\n"
    #             if self.getNdArray().dtype == np.complex64:
    #                 esize = "esize=8\n"
    #             fid.write(esize)
    #             fid.write("data_format=\"native_float\"\n")
    #         fid.close()
    #     else:
    #         binfile = sep_util.get_binary(filename)
    #         if (mode in 'a'):
    #             axes = sep_util.get_axes(filename)
    #             # Number of vectors already present in the file
    #             if self.shape == (1,):
    #                 n_vec = axes[0][0]
    #                 append_dim = self.ndims
    #             else:
    #                 n_vec = axes[self.ndims][0]
    #                 append_dim = self.ndims + 1
    #             with open(filename, mode) as fid:
    #                 fid.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (append_dim, n_vec + 1, append_dim, append_dim))
    #             fid.close()
    #     # Writing binary file
    #     fmt = '>f'
    #     if self.getNdArray().dtype == np.complex64 or self.getNdArray().dtype == np.complex128:
    #         format = '>c8'
    #     with open(binfile, mode + 'b') as fid:
    #         # Writing big-ending floating point number
    #         if np.isfortran(self.getNdArray()):  # Forcing column-wise binary writing
    #             # self.getNdArray().flatten('F').astype(fmt,copy=False).tofile(fid)
    #             self.getNdArray().flatten('F').tofile(fid, format=fmt)
    #         else:
    #             # self.getNdArray().astype(fmt,order='C',subok=False,copy=False).tofile(fid)
    #             self.getNdArray().tofile(fid, format=fmt)
    #     fid.close()
    #     return

    def abs(self):
        self.getNdArray()[:] = np.abs(self.getNdArray())
        return self

    def sign(self):
        self.getNdArray()[:] = np.sign(self.getNdArray())
        return self

    def reciprocal(self):
        self.getNdArray()[:] = 1. / self.getNdArray()
        return self

    def maximum(self, vec2):
        if np.isscalar(vec2):
            self.getNdArray()[:] = np.maximum(self.getNdArray(), vec2)
            return self
        elif isinstance(vec2, vectorIC):
            if not self.checkSame(vec2):
                raise ValueError('Dimensionality not equal: self = %s; vec2 = %s' % (self.shape, vec2.shape))
            self.getNdArray()[:] = np.maximum(self.getNdArray(), vec2.getNdArray())
            return self
        else:
            raise TypeError('Provided input has to be either a scalar or a vectorIC')

    def conj(self):
        self.getNdArray()[:] = np.conjugate(self.getNdArray())
        return self

    def pow(self, power):
        """Compute element-wise power of the vector"""
        self.getNdArray()[:] = self.getNdArray() ** power
        return self

    def real(self):
        """Return the real part of the vector"""
        self.getNdArray()[:] = self.getNdArray().real
        return self

    def imag(self,):
        """Return the imaginary part of the vector"""
        self.getNdArray()[:] = self.getNdArray().imag
        return self

    def copy(self, vec2):
        """Function to copy vector from input vector"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, vectorIC):
            raise TypeError("Provided input vector not a vectorIC!")
        # Checking dimensionality
        if not self.checkSame(vec2):
            raise ValueError('Dimensionality not equal: self = %s; vec2 = %s' % (self.shape, vec2.shape))
        # Element-wise copy of the input array
        self.getNdArray()[:] = vec2.getNdArray()
        return self

    def scaleAdd(self, vec2, sc1=1.0, sc2=1.0):
        """Function to scale a vector"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, vectorIC):
            raise TypeError("Provided input vector not a vectorIC!")
        # Checking dimensionality
        if not self.checkSame(vec2):
            raise ValueError('Dimensionality not equal: self = %s; vec2 = %s' % (self.shape, vec2.shape))
        # Performing scaling and addition
        self.getNdArray()[:] = sc1 * self.getNdArray() + sc2 * vec2.getNdArray()
        return self

    def dot(self, vec2):
        """Function to compute dot product between two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, vectorIC):
            raise TypeError("Provided input vector not a vectorIC!")
        # Checking size (must have same number of elements)
        if self.size != vec2.size:
            raise ValueError("Vector size mismatching: vec1 = %d; vec2 = %d" % (self.size, vec2.size))
        # Checking dimensionality
        if not self.checkSame(vec2):
            raise ValueError('Dimensionality not equal: self = %s; vec2 = %s' % (self.shape, vec2.shape))
        return np.vdot(self.getNdArray().ravel(), vec2.getNdArray().ravel())

    def multiply(self, vec2):
        """Function to multiply element-wise two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, vectorIC):
            raise TypeError("Provided input vector not a vectorIC!")
        # Checking size (must have same number of elements)
        if self.size != vec2.size:
            raise ValueError("Vector size mismatching: vec1 = %s; vec2 = %s" % (self.size, vec2.size))
        # Checking dimensionality
        if not self.checkSame(vec2):
            raise ValueError('Dimensionality not equal: self = %s; vec2 = %s' % (self.shape, vec2.shape))
        # Performing element-wise multiplication
        self.getNdArray()[:] = np.multiply(self.getNdArray(), vec2.getNdArray())
        return self

    def isDifferent(self, vec2):
        """Function to check if two vectors are identical using built-in hash function"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, vectorIC):
            raise TypeError("Provided input vector not a vectorIC!")
        # Using Hash table for python2 and numpy built-in function array_equal otherwise
        if version_info[0] == 2:
            # First make both array buffers read-only
            self.arr.flags.writeable = False
            vec2.arr.flags.writeable = False
            chcksum1 = hash(self.getNdArray().data)
            chcksum2 = hash(vec2.getNdArray().data)
            # Remake array buffers writable
            self.arr.flags.writeable = True
            vec2.arr.flags.writeable = True
            isDiff = (chcksum1 != chcksum2)
        else:
            isDiff = (not np.array_equal(self.getNdArray(), vec2.getNdArray()))
        return isDiff

    def clipVector(self, low, high):
        """Function to bound vector values based on input vectors low and high"""
        if not isinstance(low, vectorIC):
            raise TypeError("Provided input low vector not a vectorIC!")
        if not isinstance(high, vectorIC):
            raise TypeError("Provided input high vector not a vectorIC!")
        self.getNdArray()[:] = np.minimum(np.maximum(low.getNdArray(), self.getNdArray()), high.getNdArray())
        return self


# TODO add methods
class vectorOC(vector):
    """Out-of-core python vector class"""

    def __init__(self, input):
        """VectorOC constructor: input= numpy array, header file, vectorIC"""
        # Verify that input is a numpy array or header file or vectorOC
        if isinstance(input, vectorIC):
            # VectorIC passed to constructor
            # Placing temporary file into datapath folder
            tmp_vec = sep_util.datapath + "tmp_vectorOC" + str(int(time.time() * 1000000)) + ".H"
            sep_util.write_file(tmp_vec, input.arr, input.ax_info)
            self.vecfile = tmp_vec  # Assigning internal vector array
            # Removing header file? (Default behavior is to remove temporary file)
            self.remove_file = True
        elif isinstance(input, np.ndarray):
            # Numpy array passed to constructor
            tmp_vec = sep_util.datapath + "tmp_vectorOC" + str(int(time.time() * 1000000)) + ".H"
            sep_util.write_file(tmp_vec, input)
            self.vecfile = tmp_vec  # Assigning internal vector array
            # Removing header file? (Default behavior is to remove temporary file)
            self.remove_file = True
        elif isinstance(input, str):
            # Header file passed to constructor
            self.vecfile = input  # Assigning internal vector array
            # Removing header file? (Default behavior is to preserve user file)
            self.remove_file = False
        else:
            # Not supported type
            raise ValueError("ERROR! Input variable not currently supported!")
        # Assigning binary file pointer
        self.binfile = sep_util.get_binary(self.vecfile)
        # Number of axes integer
        self.ndims = sep_util.get_num_axes(self.vecfile)
        # Number of elements per axis (tuple)
        axes_info = sep_util.get_axes(self.vecfile)
        axis_elements = tuple([ii[0] for ii in axes_info[:self.ndims]])
        self.shape = tuple(reversed(axis_elements))
        self.size = np.product(self.shape)
        self.ndims = len(self.shape)
        return

    def __del__(self):
        """VectorOC destructor"""
        if self.remove_file:
            # Removing both header and binary files (using os.system to make module compatible with python3.5)
            os.system("rm -f %s %s" % (self.vecfile, self.binfile))
        return

    def getNdArray(self):
        """Function to return Ndarray of the vector"""
        ndarray, _ = sep_util.read_file(self.vecfile)
        return ndarray

    def norm(self, N=2):
        """Function to compute vector N-norm"""
        if N != 2:
            raise NotImplementedError("Norm different than L2 not currently supported")
        # Running Solver_ops to compute norm value
        find = re_dpr.search(sys_util.RunShellCmd("Solver_ops file1=%s op=dot" % self.vecfile, get_stat=False)[0])
        if find:
            return np.sqrt(float(find.group(1)))
        else:
            raise ValueError("ERROR! Trouble parsing dot product!")
        return

    def zero(self):
        """Function to zero out a vector"""
        sys_util.RunShellCmd("head -c %s </dev/zero > %s" % (self.size * 4, self.binfile), get_stat=False,
                             get_output=False)
        # sys_util.RunShellCmd("Solver_ops file1=%s op=zero"%(self.vecfile),get_stat=False,get_output=False)
        return

    def scale(self, sc):
        """Function to scale a vector"""
        import sys_util
        sys_util.RunShellCmd("Solver_ops file1=%s scale1_r=%s op=scale" % (self.vecfile, sc), get_stat=False,
                             get_output=False)
        return

    def rand(self, snr=1.0):
        """Fill vector with random number (~U[1,-1]) with a given SNR"""
        # Computing RMS amplitude of the vector
        rms = sys_util.RunShellCmd("Attr < %s want=rms param=1 maxsize=5000" % (self.vecfile), get_stat=False)[0]
        rms = float(rms.split("=")[1])  # Standard deviation of the signal
        amp_noise = 1.0
        if rms != 0.:
            amp_noise = math.sqrt(3.0 / snr) * rms  # sqrt(3*Power_signal/SNR)
        # Filling file with random number with the proper scale
        sys_util.RunShellCmd("Noise file=%s rep=1 type=0 var=0.3333333333; Solver_ops file1=%s scale1_r=%s op=scale" % (
            self.vecfile, self.vecfile, amp_noise), get_stat=False, get_output=False)
        return

    def clone(self):
        """Function to clone (deep copy) a vector or from a space and creating a copy of the associated header file"""
        # First performing a deep copy of the vector
        vec_clone = deepcopy(self)
        if vec_clone.vecfile is None:
            # Creating header and binary files from vector space
            # Placing temporary file into datapath folder
            tmp_vec = sep_util.datapath + "clone_tmp_vector" + str(int(time.time() * 1000000)) + ".H"
            axis_file = ""
            for iaxis, naxis in enumerate(tuple(reversed(vec_clone.shape))):
                axis_file += "n%s=%s " % (iaxis + 1, naxis)
            # Creating temporary vector file
            cmd = "Spike %s | Add scale=0.0 > %s" % (axis_file, tmp_vec)
            sys_util.RunShellCmd(cmd, get_stat=False, get_output=False)
            vec_clone.vecfile = tmp_vec
            vec_clone.binfile = sep_util.get_binary(vec_clone.vecfile)
        else:
            # Creating a temporary file with similar name but computer time at the end
            tmp_vec = self.vecfile.split(".H")[0].split("/")[-1]  # Getting filename only
            # Placing temporary file into datapath folder
            tmp_vec = sep_util.datapath + tmp_vec + "_clone_" + str(int(time.time() * 1000000)) + ".H"
            tmp_bin = tmp_vec + "@"
            # Copying header and binary files and setting pointers to new file
            copyfile(self.vecfile, tmp_vec)  # Copying header
            copyfile(self.binfile, tmp_bin)  # Copying binary
            vec_clone.vecfile = tmp_vec
            vec_clone.binfile = tmp_bin
            # "Fixing" header file
            with open(vec_clone.vecfile, "a") as fid:
                fid.write("in='%s\n'" % tmp_bin)
        # By default the clone file is going to be removed once the vector is deleted
        vec_clone.remove_file = True
        return vec_clone

    def cloneSpace(self):
        """Function to clone vector space only (vector without actual vector binary file by using None values)"""
        vec_space = vectorOC(self.vecfile)
        # Removing header vector file
        vec_space.vecfile = None
        vec_space.binfile = None
        vec_space.remove_file = False
        return vec_space

    def checkSame(self, vec2):
        """Function to check dimensionality of vectors"""
        return self.shape == vec2.shape

    def writeVec(self, filename, mode='w'):
        """Function to write vector to file"""
        # Check writing mode
        if not mode in 'wa':
            raise ValueError("Mode must be appending 'a' or writing 'w' ")
        # writing header/pointer file if not present and not append mode
        if not (os.path.isfile(filename) and mode in 'a'):
            binfile = sep_util.datapath + filename.split('/')[-1] + '@'
            # Copying SEPlib header file
            copyfile(self.vecfile, filename)
            # Substituting binary file
            with open(filename, 'a') as fid:
                fid.write("\nin='%s'\n" % binfile)
            fid.close()
        else:
            binfile = sep_util.get_binary(filename)
            if mode in 'a':
                axes = sep_util.get_axes(filename)
                # Number of vectors already present in the file
                if self.shape == (1,):
                    n_vec = axes[0][0]
                    append_dim = self.ndims
                else:
                    n_vec = axes[self.ndims][0]
                    append_dim = self.ndims + 1
                with open(filename, mode) as fid:
                    fid.write("n%s=%s o%s=0.0 d%s=1.0 \n" % (append_dim, n_vec + 1, append_dim, append_dim))
                fid.close()
        # Writing or Copying binary file
        if not (os.path.isfile(binfile) and mode in 'a'):
            copyfile(self.binfile, binfile)
        else:
            # Writing file if
            with open(binfile, mode + 'b') as fid, open(self.binfile, 'rb') as fid_toread:
                while True:
                    data = fid_toread.read(sys_util.BUF_SIZE)
                    if not data:
                        break
                    fid.write(data)
            fid.close()
            fid_toread.close()
        return

    def copy(self, vec2):
        """Function to copy vector from input vector"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, vectorOC):
            raise TypeError("ERROR! Provided input vector not a vectorOC!")
        # Checking dimensionality
        if not self.checkSame(vec2):
            raise ValueError("ERROR! Vector dimensionality mismatching: vec1 = %s; vec2 = %s" % (self.shape, vec2.shape))
        # Copy binary file of input vector
        copyfile(vec2.binfile, self.binfile)  # Copying binary
        return

    def scaleAdd(self, vec2, sc1=1.0, sc2=1.0):
        """Function to scale a vector"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, vectorOC):
            raise TypeError("ERROR! Provided input vector not a vectorOC!")
        # Checking dimensionality
        if not self.checkSame(vec2):
            raise ValueError("ERROR! Vector dimensionality mismatching: vec1 = %s; vec2 = %s" % (self.shape, vec2.shape))
        # Performing scaling and addition
        cmd = "Solver_ops file1=%s scale1_r=%s file2=%s scale2_r=%s op=scale_addscale" % (
            self.vecfile, sc1, vec2.vecfile, sc2)
        sys_util.RunShellCmd(cmd, get_stat=False, get_output=False)
        return

    def dot(self, vec2):
        """Function to compute dot product between two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, vectorOC):
            raise TypeError("ERROR! Provided input vector not a vectorOC!")
        # Checking size (must have same number of elements)
        if self.size != vec2.size:
            raise ValueError("ERROR! Vector size mismatching: vec1 = %s; vec2 = %s" % (self.size, vec2.size))
        # Checking dimensionality
        if not self.checkSame(vec2):
            raise ValueError("ERROR! Vector dimensionality mismatching: vec1 = %s; vec2 = %s" % (self.shape, vec2.shape))
        # Running Solver_ops to compute norm value
        cmd = "Solver_ops file1=%s file2=%s op=dot" % (self.vecfile, vec2.vecfile)
        find = re_dpr.search(sys_util.RunShellCmd(cmd, get_stat=False)[0])
        if find:
            return float(find.group(1))
        else:
            raise ValueError("ERROR! Trouble parsing dot product!")
        return float(out_dot)

    def multiply(self, vec2):
        """Function to multiply element-wise two vectors"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, vectorOC):
            raise TypeError("ERROR! Provided input vector not a vectorOC!")
        # Checking size (must have same number of elements)
        if self.size != vec2.size:
            raise ValueError("ERROR! Vector size mismatching: vec1 = %s; vec2 = %s" % (self.size, vec2.size))
        # Checking dimensionality
        if not self.checkSame(vec2):
            raise ValueError("ERROR! Vector dimensionality mismatching: vec1 = %s; vec2 = %s" % (self.shape, vec2.shape))
        # Performing scaling and addition
        cmd = "Solver_ops file1=%s file2=%s op=multiply" % (self.vecfile, vec2.vecfile)
        sys_util.RunShellCmd(cmd, get_stat=False, get_output=False)
        return

    def isDifferent(self, vec2):
        """Function to check if two vectors are identical using M5 hash scheme"""
        # Checking whether the input is a vector or not
        if not isinstance(vec2, vectorOC):
            raise TypeError("ERROR! Provided input vector not a vectorOC!")
        hashmd5_vec1 = sys_util.hashfile(self.binfile)
        hashmd5_vec2 = sys_util.hashfile(vec2.binfile)
        return hashmd5_vec1 != hashmd5_vec2
