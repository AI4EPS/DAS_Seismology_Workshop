import os
import sys

# C++/pybind11 functions

local_path = os.path.dirname(__file__)
dasutil_path = os.path.join(local_path, "../../../python")
pyDAS_path = os.path.join(local_path, "../../../build")

# print(f'local_path = {local_path}')
# print(f'dasutil_path = {dasutil_path}')
# print(f'pyDAS_path = {pyDAS_path}')
# sys.stdout.flush()

# print(os.environ)
if "LD_LIBRARY_PATH" not in os.environ.keys():
    # LD_LIBRARY_PATH not defined
    os.environ["LD_LIBRARY_PATH"] = pyDAS_path

elif pyDAS_path not in os.environ["LD_LIBRARY_PATH"]:
    # LD_LIBRARY_PATH defined, but doesn't include pyDAS path
    os.environ["LD_LIBRARY_PATH"] = pyDAS_path + ":" + os.environ["LD_LIBRARY_PATH"]

else:
    print("pyDAS_path already in LD_LIBRARY_PATH")
    # os.environ["LD_LIBRARY_PATH"] = pyDAS_path

sys.path.insert(0, pyDAS_path)
sys.path.insert(0, dasutil_path)
