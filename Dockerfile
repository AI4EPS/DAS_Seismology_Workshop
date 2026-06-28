# =============================================================
# Dockerfile — DAS Seismology Workshop
# Repo: https://github.com/AI4EPS/DAS_Seismology_Workshop.git
# Env: das-proc (Python 3.9, cmake, gcc, DAS-proc C++ build)
# =============================================================

FROM nvcr.io/nvidia/pytorch:24.04-py3

USER root
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y \
  software-properties-common \
  build-essential \
  curl \
  wget \
  ca-certificates \
  git \
  vim \
  && rm -rf /var/lib/apt/lists/*

# ── Install Miniconda ─────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR" && \
  rm /tmp/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# ── Configure conda: accept ToS + use conda-forge only ──────
# Write .condarc directly — most reliable method across base images.
# Also accept ToS explicitly so non-interactive builds never block.
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
  printf 'channels:\n  - conda-forge\nchannel_priority: strict\ndefault_channels: []\n' > /opt/conda/.condarc

# ── JupyterHub + JupyterLab in base conda env ────────────────
# nb_conda_kernels must be installed via conda (not pip) to work correctly.
RUN conda install -n base -y \
  jupyterhub \
  jupyterlab \
  jupyter_server \
  ipykernel \
  nb_conda_kernels && \
  conda clean -afy

# ── Setup user ────────────────────────────────────────────────
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER=${NB_USER}
ENV HOME=/home/${NB_USER}

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
  chown -R $NB_USER:users $CONDA_DIR

# ── Clone workshop repo ───────────────────────────────────────
# Cloned as root so cmake build can run, then handed to jovyan
RUN git clone https://github.com/AI4EPS/DAS_Seismology_Workshop.git /opt/workshop-repo && \
  chown -R $NB_USER:users /opt/workshop-repo

# ── Create das-proc conda environment ────────────────────────
# Bypass conda env create (no --override-channels support) and build manually.
# Conda create + install split avoids default_channels being re-injected.
RUN conda create -n das-proc -c conda-forge --override-channels -y \
  python=3.9 \
  pip \
  ipykernel \
  cmake=4.0.3 \
  gcc_linux-64=14.2.0 \
  gxx_linux-64=14.2.0 \
  h5py=3.14.0 \
  joblib=1.5.1 \
  matplotlib \
  numba \
  numpy=1.26.4 \
  pandas=2.3.1 \
  geopandas \
  pillow=11.1.0 \
  psutil=7.0.0 \
  python-dateutil=2.9.0.post0 \
  scipy \
  cartopy \
  sympy=1.14.0 \
  tqdm \
  utm \
  gdown \
  obspy=1.3.0 \
  basemap && \
  conda run -n das-proc pip install --no-cache-dir \
  pykonal==0.3.2b3 && \
  conda clean -afy

RUN ln -s $CONDA_DIR/envs/das-proc/bin/gdown /usr/local/bin/gdown

# ── Build DAS-proc C++ extension ─────────────────────────────
# Mirrors the cmake/make steps from the original setup script
ENV CONDA_ENV_PREFIX="$CONDA_DIR/envs/das-proc"

RUN mkdir -p /opt/workshop-repo/notebooks/lab1_das_basics/Scripts/DAS-proc/build && \
  cd /opt/workshop-repo/notebooks/lab1_das_basics/Scripts/DAS-proc/build && \
  $CONDA_ENV_PREFIX/bin/cmake \
  -DCMAKE_CXX_COMPILER=$CONDA_ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
  -DPython_ROOT_DIR=$CONDA_ENV_PREFIX \
  -DPython_EXECUTABLE=$CONDA_ENV_PREFIX/bin/python \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -DPYBIND11_LTO_CXX_FLAGS="" \
  -DPYBIND11_LTO_LINKER_FLAGS="" \
  -DCMAKE_CXX_FLAGS="-fno-lto" \
  -DCMAKE_EXE_LINKER_FLAGS="-fno-lto" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fno-lto" \
  .. && \
  make -j$(nproc)

# ── Register das-proc as a Jupyter kernel ────────────────────
RUN $CONDA_ENV_PREFIX/bin/python -m ipykernel install \
  --name das-proc \
  --display-name "DAS Processing (Python 3.9)"


# ---------------------------------------------------------------------------
# Layer A: dasio environment + PyTorch + dependencies + Jupyter kernel
# ---------------------------------------------------------------------------
RUN conda create -n dasio -c conda-forge --override-channels -y \
  python=3.11 \
  pip \
  ipykernel \
  ipython \
  ipywidgets \
  git \
  cxx-compiler \
  cmake \
  make && \
  conda run -n dasio python -m pip install --no-cache-dir \
  torch --index-url https://download.pytorch.org/whl/cu126 && \
  conda run -n dasio python -m pip install --no-cache-dir \
  numpy \
  scipy \
  pandas \
  pyarrow \
  matplotlib \
  h5py \
  numba \
  joblib \
  tqdm \
  phasenet && \
  conda run -n dasio python -m ipykernel install \
  --prefix=/opt/conda \
  --name dasio \
  --display-name "dasio" && \
  conda clean -afy

# ---------------------------------------------------------------------------
# Layer B: clone pinned dasio tag and editable-install it
# ---------------------------------------------------------------------------
RUN conda run -n dasio git clone --branch v0.1.0 --depth 1 \
  https://github.com/jxli2a/dasio.git /opt/dasio && \
  conda run -n dasio python -m pip install --no-cache-dir -e \
  '/opt/dasio[noise,pick]'

# ── Create Eikonal conda environment ────────────────────────
RUN apt-get update && apt-get install -y git

RUN conda create -n Eikonal -c conda-forge --override-channels -y python=3.9 pip ipykernel && \
  conda run -n Eikonal pip install --no-cache-dir \
  numpy==1.23.0 \
  scipy \
  matplotlib \
  pandas \
  h5py \
  psutil \
  joblib==1.5.1 \
  tqdm \
  utm \ 
  numba==0.58.0 && \
  conda run -n Eikonal pip install --no-cache-dir \
  cython==3.0.3 setuptools wheel && \
  conda run -n Eikonal pip install --no-cache-dir \
  git+https://github.com/malcolmw/pykonal@0.2.3b3 && \
  conda run -n Eikonal python -m ipykernel install \
  --prefix=/opt/conda \
  --name Eikonal \
  --display-name "Python Eikonal" && \
  conda clean -afy

# ── Create PhaseNet-DAS environment ───────────────────────────
RUN conda create -n phasenet-das -c conda-forge --override-channels -y \
  python=3.10 \
  pip \
  ipykernel && \
  conda run -n phasenet-das pip install --no-cache-dir \
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
  huggingface_hub \
  torchvision \
  scikit-learn && \
  conda run -n phasenet-das python -m ipykernel install \
  --prefix=/opt/conda \
  --name phasenet-das \
  --display-name "PhaseNet-DAS" && \
  conda clean -afy


# ── Create dasfm environment ──────────────────────────────────
RUN conda create -n dasfm -c conda-forge --override-channels -y \
  python=3.11 \
  pip \
  ipykernel \
  ipython \
  ipywidgets && \
  conda run -n dasfm python -m pip install --no-cache-dir \
  torch --index-url https://download.pytorch.org/whl/cu126 && \
  conda run -n dasfm python -m pip install --no-cache-dir \
  numpy \
  scipy \
  pandas \
  matplotlib \
  h5py \
  numba \
  rasterio \
  tqdm \
  psutil \
  obspy \
  pykonal \
  huggingface_hub && \
  conda run -n dasfm python -m pip install --no-cache-dir -e \
  /opt/workshop-repo/notebooks/lab3_focal_mechanisms/Scripts/dasfm_workshop && \
  conda run -n dasfm python -m ipykernel install \
  --prefix=/opt/conda \
  --name dasfm \
  --display-name "DASFM" && \
  conda clean -afy

# ── Fix ownership ─────────────────────────────────────────────
RUN chown -R $NB_USER:users /opt/workshop-repo && \
  chown -R $NB_USER:users $CONDA_DIR/envs/das-proc && \
  chown -R $NB_USER:users $CONDA_DIR/envs/Eikonal && \
  chown -R $NB_USER:users $CONDA_DIR/envs/phasenet-das && \
  chown -R $NB_USER:users $CONDA_DIR/envs/dasfm

# ── CUDA environment variables ────────────────────────────────
ENV PATH=/usr/local/cuda/bin:/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

USER ${NB_USER}
WORKDIR ${HOME}

# ── Symlink repo into home so it appears in JupyterLab ────────
RUN ln -s /opt/workshop-repo $HOME/workshop-repo

EXPOSE 8888