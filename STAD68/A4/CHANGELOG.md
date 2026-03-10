# Changelog - STAD68 Assignment 4: Training Transformer

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
