# DAS-Utilities

[![Build and Test Software](https://github.com/pinkston3/DAS-utilities-DP/actions/workflows/build-test.yml/badge.svg?branch=main)](https://github.com/pinkston3/DAS-utilities-DP/actions/workflows/build-test.yml)

This project contains some useful functions for analyzing DAS data, and also
some programs and scripts for managing large DAS data sets.

The project is primarily Python, but also includes some optimized C++
filtering code to efficiently perform some of the key processing steps.

## Library

The primary function library is in the file **`DASutils.py`**.

## Programs

Here are the key programs in the repository, and their purposes:

**`DAS_db.py`** generates or updates a CSV database file with important details
of a set of DAS data files, such as the start and end time of each file, the
sample rate of the file, etc.  ([See here for docs.](./docs/DAS_db.md))

**`DAScompress.py`** applies various lossy compression algorithms to DAS data
files to reduce their overall size.

**`Desample_DAS.py`** performs several key operations to combine DAS data
files into contiguous data of a specific length of time, and to filter and
downsample the files as well.

**`DAS_cut.py`** cuts down input DAS data files into a smaller version
suitable for testing.  ([See here for docs.](./docs/DAS_cut.md))

# Building and Testing DAS-utilities

## Building the C++ Code

The native C++ code lives in the `cpp/src` directory.  Currently this code is
an efficient, multithreaded implementation of a Butterworth filter used by
some of the above programs.

To build the code run these steps.  Note that we build the library into a
separate `build` subdirectory that is not checked in.

```
# From top level of Git repository
git submodule update --init

# Build the shared library with fast implementations of key operations.
# NOTE:  OpenMP must be supported by your compiler.  See below for details.
mkdir build
cd build
cmake -DPython_ROOT_DIR=$CONDA_PREFIX -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python ..
make
cd ..
```

Once the build process is done, we must ensure that the Python code can
find the shared library.  Right now we use `LD_LIBRARY_PATH` (or
`DYLD_LIBRARY_PATH` on macOS).  In the future this will be handled in a
different way.

```sh
# bash:
export LD_LIBRARY_PATH="${PWD}/build/":${LD_LIBRARY_PATH}
export PYTHONPATH="${PWD}/python/":${PYTHONPATH}

# csh:
setenv LD_LIBRARY_PATH "${PWD}/build/":${LD_LIBRARY_PATH}
setenv PYTHONPATH "${PWD}/python/":${PYTHONPATH}
```

To install on an Ubuntu VM:
```
sudo apt update

sudo apt upgrade

sudo apt install build-essential dkms linux-headers-$(uname -r)

sudo apt install cmake

sudo apt-get install python3-dev

sudo apt-get isntall python3-pip

git clone https://github.com/biondiettore/DAS-utilities.git

cd DAS-utilities/

git submodule update --init --recursive external/pybind11

mkdir build

cd build

cmake -DCMAKE_INSTALL_PREFIX=../local .. -DCMAKE_CXX_COMPILER=/usr/bin/g++

make

pip install jupyterlab

```

## Testing DAS-utilities

Tests using the Python `unittest` framework are in the `tests` subdirectory.
See [the README.md file](./tests/README.md) in this directory for details.

## Debugging Issues with the C++ Code

Here are some common issues and ways you can work towards their resolution.

### Does my compiler support OpenMP?

The `clang` compiler included with Xcode on macOS doesn't support OpenMP by
default.  You will need to use something like MacPorts or Homebrew to address
this issue, either by installing OpenMP support and pointing the bundled
compiler at it, or by installing a separate compiler that includes OpenMP
support by default.  You can search on the Internet to find information about
how to do these things.

Here is a simple program to test whether OpenMP support is included in your
compiler:

```c
/******************************************************************************
 * FILE: omp_hello.c
 * DESCRIPTION:
 *   OpenMP Example - Hello World - C/C++ Version
 *   In this simple example, the master thread forks a parallel region.
 *   All threads in the team obtain their unique thread number and print it.
 *   The master thread only prints the total number of threads.  Two OpenMP
 *   library routines are used to obtain the number of threads and each
 *   thread's number.
 * AUTHOR: Blaise Barney  5/99
 * LAST REVISED: 04/06/05
 ******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
  int nthreads, tid;

  /* Fork a team of threads giving them their own copies of variables */
  #pragma omp parallel private(nthreads, tid)
  {
    /* Obtain thread number */
    tid = omp_get_thread_num();
    printf("Hello World from thread = %d\n", tid);

    /* Only master thread does this */
    if (tid == 0)
    {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }

  }  /* All threads join master thread and disband */
}
```

Paste this code into a file `omp_hello.c` and then try to build/run it:

```sh
# For clang/llvm:
clang++ -fopenmp omp_hello-c -o omp_hello

# For GCC:
g++ -fopenmp omp_hello.c -o omp_hello
```
