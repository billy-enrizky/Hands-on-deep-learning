# Changelog - STAD68 Assignment 4: Training Transformer

## 2026-03-11 20:07:52

### Added
- New notebook variant (A4_Transformers_STAD68_2026-New.ipynb) aligned with updated assignment requirements
- FID evaluation using `torchmetrics.image.fid.FrechetInceptionDistance` with InceptionV3 features (2048-dim), replacing manual scipy pixel-level implementation
- PDR (Pokemon Detection Rate) metric replacing FDR, per professor's updated instructions
- Stronger classifier negatives: random pixel permutation, block shuffle (5x5), and random noise (1/3 each)
- MPS device fallback for Apple Silicon in new notebook

### Changed
- FDR (False Discovery Rate) renamed to PDR (Pokemon Detection Rate) -- PDR = proportion classified AS Pokemon (higher is better)
- FID now uses InceptionV3 features via torchmetrics instead of raw pixel covariance via scipy
- Classifier negatives expanded from random-noise-only to three strategies per professor's hints
- Results: FID=81.42 (InceptionV3), PDR=0.975 (78/80 classified as Pokemon)

## 2026-03-10 14:36:00

### Added
- Completed transformer decoder-only (GPT-2) notebook for Pokemon image generation via next-token prediction
- Implemented FID (Frechet Inception Distance) evaluation using pixel-level features with regularized covariance matrices
- Implemented FDR (False Discovery Rate) evaluation using a trained CNN classifier (Pokemon vs random noise)
- Added results summary with model configuration, evaluation metrics (FID=107.77, FDR=0.00), and analysis
- Added MPS device fallback for Apple Silicon compatibility

### Changed
- Updated device selection to support CUDA, MPS, and CPU fallback chain
- Fixed scipy `sqrtm` deprecation warning by removing deprecated `disp` parameter
