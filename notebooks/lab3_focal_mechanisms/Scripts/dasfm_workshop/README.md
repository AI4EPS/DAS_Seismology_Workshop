# dasfm

**Focal mechanism inversion from DAS strain-rate data and seismic stations.**

## Installation

**Requirements:** Python >= 3.11. GPU is optional but recommended. PyTorch must be installed separately.

```bash
conda create -n dasfm python=3.11
conda activate dasfm
```

Install PyTorch (choose ONE):

```bash
# GPU (CUDA 12.6 — most compatible)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install dasfm:

```bash
pip install -e .          # core
pip install -e ".[3d]"    # + 3-D Eikonal support (pykonal)
```

Verify GPU after installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
