# [DAS Seismology Workshop](https://ai4eps.github.io/DAS_Seismology_Workshop/)

[![Website](https://github.com/AI4EPS/DAS_Seismology_Workshop/actions/workflows/docs.yml/badge.svg)](https://ai4eps.github.io/DAS_Seismology_Workshop/)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FAI4EPS%2FDAS_Seismology_Workshop&label=views&labelColor=%23d9e3f0&countColor=%23263759&style=flat)

## Introduction to DAS for Seismology: From Data Acquisition to Analysis

**Date:** Tuesday, 14 April 2026, 9:00 AM – 4:00 PM

**Location:** SSA 2026 Annual Meeting

**Contributors:** 
[Ettore Biondi](https://geophysics.stanford.edu/people/ettore-biondi-0),
[Jiaxuan Li](https://jxli2a.github.io/),
[Chun Zhang](),
[Weiqiang Zhu](https://ai4eps.github.io/homepage/) (alphabetical order)

---

## Schedule

### Morning — Concepts & Methods

| Time | Topic |
|------|-------|
| 09:00 – 09:10 | Welcome, logistics, GCP login check |
| 09:10 – 09:35 | [DAS Basics: Instruments & Measurements](https://ai4eps.github.io/DAS_Seismology_Workshop/das_basics/) |
| 09:35 – 10:20 | [Deep Learning for DAS](https://ai4eps.github.io/DAS_Seismology_Workshop/phasenet_das/) |
| 10:20 – 10:30 | *Coffee Break* |
| 10:30 – 11:15 | [Focal Mechanisms & Source Parameters from DAS](https://ai4eps.github.io/DAS_Seismology_Workshop/focal_mechanisms/) |
| 11:15 – 12:00 | [Eikonal Traveltime Tomography with DAS](https://ai4eps.github.io/DAS_Seismology_Workshop/eikonal_tomography/) |
| 12:00 – 13:00 | *Lunch* |

### Afternoon — Hands-on Jupyter Labs (Google Cloud Platform)

| Time | Lab |
|------|-----|
| 13:00 – 13:30 | [Lab 1: DAS Basics](https://ai4eps.github.io/DAS_Seismology_Workshop/notebooks/lab1_das_basics/Notebooks/Basic_DAS_data/) (reading, filtering, plotting) |
| 13:30 – 14:15 | Lab 2: Deep Learning for DAS <br> [2a: PhaseNet-DAS](https://ai4eps.github.io/DAS_Seismology_Workshop/notebooks/lab2_phasenet_das/Notebooks/lab2a_phasenet_das_inference/), [2b: Association](https://ai4eps.github.io/DAS_Seismology_Workshop/notebooks/lab2_phasenet_das/Notebooks/lab2b_phase_association/), [2c: Training](https://ai4eps.github.io/DAS_Seismology_Workshop/notebooks/lab2_phasenet_das/Notebooks/lab2c_semisupervised_training/), [2d: DASNet](https://ai4eps.github.io/DAS_Seismology_Workshop/notebooks/lab2_phasenet_das/Notebooks/lab2d_dasnet_inference/) |
| 14:15 – 15:00 | Lab 3: Focal Mechanism Inversion <br> [3a: Ray Parameters](https://ai4eps.github.io/DAS_Seismology_Workshop/notebooks/lab3_focal_mechanisms/Notebooks/lab3a_ray_parameters_2d/), [3b: DAS Preprocessing](https://ai4eps.github.io/DAS_Seismology_Workshop/notebooks/lab3_focal_mechanisms/Notebooks/lab3b_das_preprocessing/), [3c: Inversion](https://ai4eps.github.io/DAS_Seismology_Workshop/notebooks/lab3_focal_mechanisms/Notebooks/lab3c_focal_mechanism_inversion/), [3d: Results](https://ai4eps.github.io/DAS_Seismology_Workshop/notebooks/lab3_focal_mechanisms/Notebooks/lab3d_result_summary/) |
| 15:00 – 15:30 | *Coffee Break* |
| 15:30 – 16:00 | [Lab 4: Eikonal Traveltime Tomography](https://ai4eps.github.io/DAS_Seismology_Workshop/notebooks/lab4_eikonal_tomography/Notebooks/lab4_eikonal_tomography_2d/) ([2D](https://ai4eps.github.io/DAS_Seismology_Workshop/notebooks/lab4_eikonal_tomography/Notebooks/lab4_eikonal_tomography_2d/), [3D](https://ai4eps.github.io/DAS_Seismology_Workshop/notebooks/lab4_eikonal_tomography/Notebooks/lab4_eikonal_tomography_3d/)) |

---

## About

This one-day workshop is designed for researchers at all levels who are interested in learning how to collect, process, and analyze Distributed Acoustic Sensing (DAS) data recorded on telecommunication fiber-optic cables. The session will begin with an overview of DAS technology and include an on-site demonstration showing how to configure a DAS experiment and acquire data. We will cover the unique capabilities of DAS for both temperature and deformation sensing, emphasizing its advantages in seismological and environmental applications.

Participants will then be introduced to data processing workflows for extracting meaningful seismic information from DAS recordings. This includes techniques for high-precision earthquake travel-time picking using machine learning, and methods for estimating focal mechanisms across a wide range of event magnitudes. In the final segment, we will demonstrate how to compute noise cross-correlations from DAS data and how to combine these with earthquake arrivals to perform high-resolution seismic tomography.

By the end of the workshop, participants will gain a comprehensive understanding of the end-to-end DAS workflow — from field setup to advanced data analysis — empowering them to fully utilize the high spatial and temporal resolution provided by DAS arrays in their research.

## Acknowledgments

The workshop is generously supported by Google through the provision of cloud computing resources on Google Cloud Platform, which were instrumental in enabling the execution of the DAS lab sessions. We gratefully acknowledge the Doerr School of Sustainability Computational Support Team at Stanford University for their support, and we specifically thank Brian Tempero, Brian Chivers, and Ellianna Abrahams for their assistance with computational infrastructure and deployment.

---

If you have any questions about the workshop materials or encounter any issues, please [open an issue](https://github.com/AI4EPS/DAS_Seismology_Workshop/issues) on our GitHub repository.

```
@misc{das_seismology_workshop_2026,
  author = {Biondi, Ettore and Li, Jiaxuan and Zhang, Chun and Zhu, Weiqiang},
  title = {Introduction to DAS for Seismology: From Data Acquisition to Analysis},
  year = {2026},
  url = {https://ai4eps.github.io/DAS_Seismology_Workshop/},
  note = {Seismological Society of America (SSA) Annual Meeting, 2026}
}
```
