cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(python-solver)
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(PYTHON_SOLVER_MAJOR_VERSION 0)
set(PYTHON_SOLVER_MINOR_VERSION 3)
set(PYTHON_SOLVER_VERSION_STRING
    ${PYTHON_SOLVER_MAJOR_VERSION}.${genericIO_MINOR_VERSION})

add_subdirectory(python)

include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${PYTHON_SOLVER_MAJOR_VERSION}")
set(CPACK_PACKAGE_VERSION_MINOR "${PYTHON_SOLVER_MINOR_VERSION}")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_NAME "GenericSolver")
set(CPACK_PACKAGE_CONTACT "ettore88@stanford.edu/gbarnier@stanford.edu/bob@sep.stanford.edu/francesco.picetti@polimi.it/sfarris@stanford.edu")
include(CPack)
