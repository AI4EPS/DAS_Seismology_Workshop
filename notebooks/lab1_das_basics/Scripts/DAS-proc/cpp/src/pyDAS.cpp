#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cassert>
#include <iostream>
#include <omp.h>
#include "filters.hpp"

using namespace std;
namespace py = pybind11;


// Exposing function
template <typename T>
py::array_t<T> lfilter(py::array_t<T> in_arr, T flo, int nplo, T fhi, int nphi, int phase, int nthreads) {
    // Checking if the input array is 2D
    auto buf_in = in_arr.request();
    if (buf_in.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two!");

    // Getting size of the input array
    long long nt = in_arr.shape(1);
    int nch = in_arr.shape(0);
    long long ntpad = nt+2;

    // Allocating output array
    py::array_t<T> out_arr({static_cast<py::ssize_t>(nch), static_cast<py::ssize_t>(nt)});
    // The below code assumes that the array is C-contiguous
    assert(out_arr.flags() & py::array::c_style);
    auto buf_out = out_arr.request();


    // Checking if input cutoff frequencies are correct
    if (flo < 0.0001 && fhi > 0.4999) {
        throw std::runtime_error("Incorrect cutoff frequencies!");
    }

    // Filtering order
    if (nplo < 1) nplo = 1;
    if (nphi < 1) nphi = 1;

    // Get pointers to data arrays
    T * data_p = (T *) buf_in.ptr;
    T * data_o = (T *) buf_out.ptr;

    // Checking number of threads
    nthreads = std::min(nch, nthreads);

    // Allocating temporary arrays
    auto ** data = new T*[nthreads];
    auto ** newdata = new T*[nthreads];
    auto ** tempdata = new T*[nthreads];
    for (int ithread = 0; ithread < nthreads; ithread++){
        data[ithread] = new T[ntpad];
        newdata[ithread] = new T[ntpad];
        tempdata[ithread] = new T[ntpad];
    }

    #pragma omp parallel for schedule(dynamic,1) num_threads(nthreads)
    for (long long ich = 0; ich < nch; ich++){
        int ithread = omp_get_thread_num();

        // Zeroing out temporary arrays
        std::memset(data[ithread], 0, ntpad*sizeof(T));
        std::memset(newdata[ithread], 0, ntpad*sizeof(T));
        std::memset(tempdata[ithread], 0, ntpad*sizeof(T));

        // Copying input data into temporary arrray
        std::memcpy(data[ithread]+2, data_p+ich*nt, nt*sizeof(T));

        // Applying highcut filter
        if (flo > 0.0001){
            lowcut(flo, nplo, phase, ntpad, data[ithread], newdata[ithread], tempdata[ithread]);
        }

        // Applying highcut filter
        if (fhi < 0.4999){
            highcut(fhi, nphi, phase, ntpad, data[ithread], newdata[ithread], tempdata[ithread]);
        }

        // Copying result to output array
        std::memcpy(data_o+ich*nt, data[ithread]+2, nt*sizeof(T));

    }

    // Deallocating temporary memory
    for (int ithread = 0; ithread < nthreads; ithread++){
        delete data[ithread];
        delete newdata[ithread];
        delete tempdata[ithread];
    }
    delete [] data;
    delete [] newdata;
    delete [] tempdata;

    return out_arr;

}


/**
 * Expose the highcut filter operation into Python.
 */
template <typename T>
py::array_t<T> highcut_py(py::array_t<T> in_arr, T fhi, int nphi, int phase) {
    auto buf_in = in_arr.request();
    if (buf_in.ndim != 1)
        throw std::runtime_error("in_arr:  number of dimensions must be 1");

    // Number of samples in each channel
    int64_t nt = in_arr.shape(0);

    // Add 2 since the filter algorithms require space for two extra elements
    // at the start of the data signal.
    int64_t ntpad = nt + 2;

    // Allocating output array
    py::array_t<T> out_arr = py::array_t<T>(buf_in.size);
    auto buf_out = out_arr.request();

    // Filtering order
    if (nphi < 1) nphi = 1;

    // Get pointers to data arrays
    const T *data_p = (T *) buf_in.ptr;
    T *data_o = (T *) buf_out.ptr;

    // Allocating and zeroing out temporary arrays

    std::unique_ptr<T[]> tmp_fwd = make_unique<T[]>(ntpad);
    std::memset(tmp_fwd.get(), 0, ntpad * sizeof(T));

    std::unique_ptr<T[]> tmp_rev;
    if (phase == 0) {
        tmp_rev = make_unique<T[]>(ntpad);
        std::memset(tmp_rev.get(), 0, ntpad * sizeof(T));
    }

    // Allocate temporary array and copy in input data

    std::unique_ptr<T[]> data = make_unique<T[]>(ntpad);
    data[0] = 0;
    data[1] = 0;
    std::memcpy(data.get() + 2, data_p, nt * sizeof(T));

    // Filter
    highcut(fhi, nphi, phase, ntpad, data.get(), tmp_fwd.get(), tmp_rev.get());

    // Copying result to output array
    std::memcpy(data_o, data.get() + 2, nt * sizeof(T));

    return out_arr;
}


/**
 * Expose the lowcut filter operation into Python.
 */
template <typename T>
py::array_t<T> lowcut_py(py::array_t<T> in_arr, T flo, int nplo, int phase) {
    auto buf_in = in_arr.request();
    if (buf_in.ndim != 1)
        throw std::runtime_error("in_arr:  number of dimensions must be 1");

    // Number of samples
    int64_t nt = in_arr.shape(0);

    // Add 2 since the filter algorithms require space for two extra elements
    // at the start of the data signal.
    int64_t ntpad = nt + 2;

    // Allocating output array
    py::array_t<T> out_arr = py::array_t<T>(buf_in.size);
    auto buf_out = out_arr.request();

    // Filtering order
    if (nplo < 1) nplo = 1;

    // Get pointers to data arrays
    const T *data_p = (T *) buf_in.ptr;
    T *data_o = (T *) buf_out.ptr;

    // Allocating and zeroing out temporary arrays

    std::unique_ptr<T[]> tmp_fwd = make_unique<T[]>(ntpad);
    std::memset(tmp_fwd.get(), 0, ntpad * sizeof(T));

    std::unique_ptr<T[]> tmp_rev;
    if (phase == 0) {
        tmp_rev = make_unique<T[]>(ntpad);
        std::memset(tmp_rev.get(), 0, ntpad * sizeof(T));
    }

    // Allocate temporary array and copy in input data

    std::unique_ptr<T[]> data = make_unique<T[]>(ntpad);
    data[0] = 0;
    data[1] = 0;
    std::memcpy(data.get() + 2, data_p, nt * sizeof(T));

    // Filter
    lowcut(flo, nplo, phase, ntpad, data.get(), tmp_fwd.get(), tmp_rev.get());

    // Copying result to output array
    std::memcpy(data_o, data.get() + 2, nt * sizeof(T));

    return out_arr;
}


// Deciding what to expose in the library python can import
PYBIND11_MODULE(pyDAS, m) {
    m.doc() = "pybind11 module for DAS data processing";

    // The lfilter function is the workhorse of DAS data processing.

    m.def("lfilter", &lfilter<float>, "Apply linear filter (bandpass, high-pass, low-pass)");
    m.def("lfilter_double", &lfilter<double>, "Apply linear filter on double precision arrays (bandpass, high-pass, low-pass)");

    // Expose the highcut and lowcut functions for easier testing of these
    // operations in isolation.

    m.def("highcut", &highcut_py<float>, "Apply highcut (low-pass) filter on 1D array");
    m.def("highcut_double", &highcut_py<double>, "Apply highcut (low-pass) filter on 1D double precision array");

    m.def("lowcut", &lowcut_py<float>, "Apply lowcut (high-pass) filter on 1D array");
    m.def("lowcut_double", &lowcut_py<double>, "Apply lowcut (high-pass) filter on 1D double precision array");
}