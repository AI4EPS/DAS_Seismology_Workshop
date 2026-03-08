# PySolver
## DESCRIPTION
Package containing generic in-core/out-of-core python solver for optimization based on operators  

## PREREQUISITES 
The prerequisites packages are the following:

1. Numpy (https://numpy.org/) 
2. Matplotlib (https://matplotlib.org/)
3. Scipy (https://www.scipy.org/)
4. Cupy (https://cupy.chainer.org/)
5. Dask (https://dask.org/)
6. Dask Distributed (https://distributed.dask.org/en/latest/)
7. Dask Jobqueue (https://jobqueue.dask.org/en/latest/)

The first four packages can be easily installed by running:
```
pip install numpy matplotlib scipy cupy
``` 

The last three packages are necessary to use the Dask interface. To install them run:
```
pip install dask
pip install dask distributed --upgrade
pip install dask-jobqueue --upgrade
```
Add `--user` if the user does not have root privileges to all previous commands.

## INSTALLATION
The code runs on python3. All the library modules are contained in the folder named python.
Meaning if that folder is added to the user's PYTHONPATH, then the library can be already employed.

A different way to install all the library modules is to use cmake with the following commands:
```
cd build

cmake -DCMAKE_INSTALL_PREFIX=folder_for_buiding ../GenericSolver

make install
```
This method is useful is PySolver is used with other packages/repos using cmake.

## USAGE
Some usage examples are provided in a few Jupyter Notebooks within the folder notebooks. 
Additionally, other tests and examples are given in the folder notebooks/unit_tests.