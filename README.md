# CWT-BNN for Leaf Nitrogen & Dry Matter Estimation (Proximal Hyperspectral)

This repository provides an implementation of a **Continuous Wavelet Transform + Bayesian Neural Network (CWT-BNN)** pipeline for estimating **leaf nitrogen concentration (LNC)** and **leaf dry matter (LDM)** from leaf-scale proximal hyperspectral reflectance data.

## Overview
The pipeline includes:
- preprocessing of hyperspectral reflectance (optional)
- **CWT-based feature extraction** from spectra
- **BNN regression** (Bayesian regularization) for LNC/LDM estimation
- reproducible training/validation with fixed random seeds

> Note: This code is designed for **leaf-scale proximal sensing**. It is not intended as a satellite-operational implementation.

## Repository Structure

## Data Format
### Spectra
- `X`: matrix of size **N × B**, where:
  - `N` = number of samples
  - `B` = number of spectral bands
- `wl`: vector of length **B** containing wavelengths (nm)

### Targets
- `y_LNC`: vector of length **N** (unit: % )
- `y_LDM`: vector of length **N** (t/ha)

## Requirements
### MATLAB
- MATLAB R2020+ recommended (tested with R2024b)
- Wavelet Toolbox (for `cwt`)
- Deep Learning Toolbox (or Neural Network Toolbox) for `fitnet`

## Quick Start (MATLAB)
1. Open MATLAB and add the repo to your path:
```matlab
addpath(genpath(pwd));
