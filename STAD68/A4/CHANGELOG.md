# Changelog - STAD68 Assignment 4: Training Transformer

## 2026-03-13 22:13:00

### Changed
- Added all 7 model configuration details to notebook results summary (was previously only Qwen2 Medium)
  - GPT-2 Baseline: layers, heads, embedding dim, dropout settings
  - GPT-2 Scaled: deeper/wider GPT-2 config
  - LLaMA Small/Medium/Large: hidden size, intermediate size, heads
  - Qwen2 Small/Medium: including key-value head counts for GQA

## 2026-03-13 12:33:57

### Fixed
- Increased Modal timeout from 3600s (1 hour) to 7200s (2 hours) to prevent llama_large training timeout
- Added checkpoint resume fallback: if `latest.pth` is corrupted (killed mid-save), falls back to `best_model.pth`

### Changed
- llama_large training completed: 200 epochs, best_loss=0.0156, 19.0M params
- llama_large evaluation: FID=66.03, PDR=0.963 (77/80)
- Updated notebook results table with complete llama_large results (was previously marked as in-progress)
- Updated notebook training results: llama_large params corrected to 19,009,152
- Strengthened LLaMA vs Qwen2 analysis with llama_large evidence: scale alone does not fix PDR gap
- Updated key findings: added "diminishing returns from scale" insight

## 2026-03-13 11:40:35

### Added
- Multi-architecture exploration: 7 configurations across GPT-2, LLaMA, and Qwen2 architectures
- Modal.com cloud training script (train_modal.py) for parallel GPU training on A100s
- Standalone evaluation script (evaluate.py) with FID and PDR computation
- Cosine learning rate schedule with linear warmup across all configs
- Gradient clipping (max norm = 1.0) for training stability
- Checkpoint resume support for fault tolerance
- Comprehensive results comparison table in notebook for all 7 configs
- Analysis of FID vs PDR discrepancy between LLaMA and Qwen2 architectures

### Changed
- Best model upgraded from GPT-2 baseline (136K params) to Qwen2 Medium (6.4M params)
- FID improved from 81.42 (baseline) to 65.06 (Qwen2 Medium)
- PDR improved from 0.975 (78/80, baseline) to 1.0000 (80/80, Qwen2 Medium) -- perfect detection rate
- Notebook restructured to include all training code, evaluation code, and multi-config results
- Imports updated to include LlamaConfig, Qwen2Config alongside GPT2Config

### Training Results (All 7 Configurations)
- gpt2_baseline: FID=94.39, PDR=0.975, 136K params, 50 epochs
- gpt2_scaled: FID=68.16, PDR=0.988, 4.9M params, 100 epochs
- llama_small: FID=64.95, PDR=0.913, 1.1M params, 100 epochs
- qwen2_small: FID=65.88, PDR=0.938, 1.1M params, 100 epochs
- llama_medium: FID=65.81, PDR=0.963, 6.4M params, 150 epochs
- llama_large: FID=66.03, PDR=0.963, 19.0M params, 200 epochs
- **qwen2_medium: FID=65.06, PDR=1.000, 6.4M params, 150 epochs (BEST)**

### Infrastructure
- Used Modal.com with NVIDIA A100 GPUs for parallel training of all 7 configs
- Checkpoints stored on Modal Volume (pokemon-training-vol) with automatic download
- Total Modal cost: estimated ~$15-20 for all 7 configs

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
