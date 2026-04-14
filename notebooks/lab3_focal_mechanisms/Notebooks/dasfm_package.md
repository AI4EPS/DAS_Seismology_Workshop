# dasfm

**Focal mechanism inversion from DAS strain-rate data and seismic stations.**

`dasfm` jointly inverts P-wave polarity and S/P amplitude ratios from Distributed Acoustic Sensing (DAS) arrays and conventional seismic stations to determine earthquake focal mechanisms.

## Features

- **Joint DAS + station inversion** -- 7 modes combining polarity and S/P amplitude ratios
- **Three forward methods** -- 1-D ray tracing, 2-D Fast Sweeping with topography, 3-D Eikonal with tomographic velocity
- **MCCC polarity picking** -- Multi-channel cross-correlation with SVD sign recovery; optional Hilbert wave separation
- **GPU acceleration** -- PyTorch-based, with multi-CPU/multi-GPU parallelism
- **Quality assessment** -- Hierarchical clustering, Kagan angle uncertainty, STDR, A/B/C/D grading, SKHASH-format export

## Installation

**Requirements:** Python >= 3.11. GPU optional but recommended.

```bash
conda create -n dasfm python=3.11
conda activate dasfm
```

Install PyTorch first (choose one):

```bash
# GPU (CUDA 12.6)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install dasfm:

```bash
pip install -e .          # core package
pip install -e ".[3d]"    # + 3-D Eikonal support (pykonal)
```

Verify GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Pipeline Overview

```
Step 1   Ray parameters (traveltime, takeoff, azimuth, distance)
Step 2   DAS waveform processing (windowing, polarity picking, S/P ratios)
Step 3   Focal mechanism grid search + clustering
Step 4   Quality grading and summary
```

## References

- Li, J., Zhu, W., Biondi, E., & Zhan, Z. (2023). Earthquake focal mechanisms with distributed acoustic sensing. *Nature Communications*, 14, 4181.
