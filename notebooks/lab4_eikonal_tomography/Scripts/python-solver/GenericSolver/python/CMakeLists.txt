project(python-solver-python_modules)

find_package(PythonLibs)

set(PythonFiles
pyParOperator.py pyStepper.py sep_util.py pyProblem.py sys_util.py pyOperator.py pySolver.py pyLinearSolver.py pyNonLinearSolver.py pyNpOperator.py pySparseSolver.py pyStopper.py pyVector.py pyDaskOperator.py pyDaskVector.py pyCuVector.py pyCuOperator.py dask_util.py)

install(FILES ${PythonFiles}  DESTINATION lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
