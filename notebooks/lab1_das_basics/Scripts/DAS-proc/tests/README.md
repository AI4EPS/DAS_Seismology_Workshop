# Tests for DAS-utilities

Run tests with this command from inside the `tests` directory:

    python -m unittest

Native filtering tests compare results against the files in the
`tests/dasutil_tests/native/data` directory.  To rebuild these files,
simply delete the contents of the `data` directory (or just delete the
directory itself) and rerun the tests.

When these files are generated for the first time, the testing harness
will also create PDF files of inputs and outputs for each test, so they
may be manually inspected.  These files do not need to be checked in,
but they are small so it's probably fine to do so.

# Common Issues

The native code tests require that the pyDAS shared library in `cpp/src`
must be built.  The `dasutil_tests/native/ncontext.py` file currently
expects the file to be found in the `build` subdirectory of the top-level
directory of the project.

Additionally, tests may fail if the following environment variables haven't
been set:

*   `PYTHONPATH`
*   `LD_LIBRARY_PATH` (for Linux) or `DYLD_LIBRARY_PATH` (for macOS)

