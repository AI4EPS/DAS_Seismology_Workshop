# =============================================================
# Dockerfile — DAS Seismology Workshop
# Repo: https://github.com/AI4EPS/DAS_Seismology_Workshop.git
# Env: das-proc (Python 3.9, cmake, gcc, DAS-proc C++ build)
# =============================================================

FROM nvcr.io/nvidia/pytorch:24.04-py3

USER root
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
  software-properties-common \
  build-essential \
  curl \
  wget \
  ca-certificates \
  git \
  vim \
  && rm -rf /var/lib/apt/lists/*

# ── Install micromamba ───────────────────────────────────────── #this updates to micromamba instead of miniconda
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba && \
  micromamba shell init -s bash -p "$MAMBA_ROOT_PREFIX" && \
  mkdir -p "$MAMBA_ROOT_PREFIX" && \
  printf 'channels:\n  - conda-forge\nchannel_priority: strict\n' \
  > "$MAMBA_ROOT_PREFIX/.condarc"

# Ensure micromamba is activated in every RUN shell
SHELL ["/bin/bash", "-l", "-c"]

# ── JupyterHub + JupyterLab in base conda env ────────────────
# nb_conda_kernels must be installed via conda (not pip) to work correctly.
RUN micromamba install -n base -y \
  jupyterhub \
  jupyterlab \
  jupyter_server \
  ipykernel \
  nb_conda_kernels && \
  micromamba clean -afy

# ── Setup user ────────────────────────────────────────────────
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER}
ENV HOME=/home/${NB_USER}

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
  chown -R $NB_USER:users $MAMBA_ROOT_PREFIX

# ── Clone workshop repo ───────────────────────────────────────
# Cloned as root so cmake build can run, then handed to jovyan
RUN git clone https://github.com/AI4EPS/DAS_Seismology_Workshop.git /opt/workshop-repo && \
  chown -R $NB_USER:users /opt/workshop-repo

# ── Create das-proc conda environment ────────────────────────
# Combine env + C++ build + kernel creation into single layer
ENV CONDA_ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/das-proc"

RUN micromamba create -n das-proc -y \
  python=3.9 \
  cmake=4.0.3 \
  gcc=14.2.0 \
  gxx=14.2.0 && \
  micromamba run -n das-proc pip install --no-cache-dir \
  h5py==3.14.0 \
  joblib==1.5.1 \
  matplotlib \
  numba==0.55.2 \
  numpy \
  pandas==2.3.1 \
  geopandas \
  pillow==11.1.0 \
  psutil==7.0.0 \
  python-dateutil==2.9.0.post0 \
  scipy \
  cartopy \
  sympy==1.14.0 \
  tqdm \
  utm \
  pykonal==0.3.2b3 \
  gdown \
  obspy \
  basemap \
  ipykernel && \
  ln -s $CONDA_ENV_PREFIX/bin/gdown /usr/local/bin/gdown && \
  mkdir -p /opt/workshop-repo/notebooks/lab1_das_basics/Scripts/DAS-proc/build && \
  cd /opt/workshop-repo/notebooks/lab1_das_basics/Scripts/DAS-proc/build && \
  $CONDA_ENV_PREFIX/bin/cmake \
  -DCMAKE_CXX_COMPILER=$CONDA_ENV_PREFIX/bin/g++ \
  -DPython_ROOT_DIR=$CONDA_ENV_PREFIX \
  -DPython_EXECUTABLE=$CONDA_ENV_PREFIX/bin/python \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -DPYBIND11_LTO_CXX_FLAGS="" \
  -DPYBIND11_LTO_LINKER_FLAGS="" \
  -DCMAKE_CXX_FLAGS="-fno-lto" \
  -DCMAKE_EXE_LINKER_FLAGS="-fno-lto" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fno-lto" \
  .. && \
  make -j$(nproc) && \
  $CONDA_ENV_PREFIX/bin/python -m ipykernel install \
  --name das-proc \
  --display-name "DAS Processing (Python 3.9)" && \
  micromamba clean -afy && \
  rm -rf /opt/workshop-repo/notebooks/lab1_das_basics/Scripts/DAS-proc/build/CMakeFiles

# ── Create Eikonal conda environment ────────────────────────
RUN micromamba create -n Eikonal -y \
  python=3.9 \
  pip \
  ipykernel && \
  micromamba run -n Eikonal pip install --no-cache-dir \
  numpy==1.23.0 \
  scipy \
  matplotlib \
  pandas \
  h5py \
  psutil \
  joblib==1.5.1 \
  tqdm \
  utm \
  numba==0.58.0 \
  cython==3.0.3 \
  setuptools \
  wheel && \
  micromamba run -n Eikonal pip install --no-cache-dir \
  git+https://github.com/malcolmw/pykonal@0.2.3b3 && \
  micromamba run -n Eikonal python -m ipykernel install \
  --prefix=$MAMBA_ROOT_PREFIX \
  --name Eikonal \
  --display-name "Python Eikonal" && \
  micromamba clean -afy

# ── Create PhaseNet-DAS environment ───────────────────────────
RUN micromamba create -n phasenet-das -y \
  python=3.10 && \
  micromamba run -n phasenet-das pip install --no-cache-dir \
  torch \
  einops \
  numpy \
  scipy \
  h5py \
  matplotlib \
  pandas \
  tqdm \
  fsspec \
  obspy \
  gcsfs \
  datasets \
  pyarrow \
  wandb \
  ipykernel && \
  $MAMBA_ROOT_PREFIX/envs/phasenet-das/bin/python -m ipykernel install \
  --name phasenet-das \
  --display-name "PhaseNet-DAS" && \
  micromamba clean -afy

# ── Fix ownership ─────────────────────────────────────────────
RUN chown -R $NB_USER:users /opt/workshop-repo \
  $MAMBA_ROOT_PREFIX/envs/das-proc \
  $MAMBA_ROOT_PREFIX/envs/Eikonal \
  $MAMBA_ROOT_PREFIX/envs/phasenet-das

# ── CUDA environment variables ────────────────────────────────
ENV PATH=/usr/local/cuda/bin:/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

USER ${NB_USER}
WORKDIR ${HOME}

# ── Symlink repo into home so it appears in JupyterLab ────────
RUN ln -s /opt/workshop-repo $HOME/workshop-repo

EXPOSE 8888