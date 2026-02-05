# Changelog

All notable changes to STAD68 Assignment 2 (CNN CIFAR-10) will be documented in this file.

## [1.4.0] - 2026-02-04 17:48:41 EST

### Fixed
- Corrected terminology: distinguished between "Test Accuracy" and "Generalization Gap"
- **Test Accuracy**: Absolute performance on unseen data (higher is better)
- **Generalization Gap**: Train Accuracy - Test Accuracy (lower is better, indicates less overfitting)

### Updated Analysis
- SimpleCNN: Best generalization (1.38% gap), lowest test accuracy (72.12%) - underfitting
- VGGSmallCIFAR: Best test accuracy (87.92%), good generalization (1.82% gap)
- ResNet-20: Worst generalization (5.99% gap), medium test accuracy (85.85%) - most overfitting

### Key Correction
- VGG achieves best TEST ACCURACY (not best generalization) due to high capacity + dropout regularization
- SimpleCNN has best GENERALIZATION but underfits due to limited capacity
- ResNet-20 overfits most despite skip connections (lacks dropout)

### Cells Updated
- Cell 22: SimpleCNN Training Analysis - added generalization gap metric and key distinction
- Cell 29: VGGSmallCIFAR Training Analysis - clarified it has best test acc, not best generalization
- Cell 37: ResNet-20 Training Analysis - highlighted worst generalization gap
- Cell 39: Summary of Findings - complete rewrite with correct terminology and trade-off analysis

## [1.3.0] - 2026-02-04 17:33:30 EST

### Added
- Added training analysis sections for all three models with consistent format
- SimpleCNN Training Analysis (Cell 22): Learning progression table, key observations, limitations
- VGGSmallCIFAR Training Analysis (Cell 29): Learning progression table, comparison with SimpleCNN, insights
- Updated ResNet-20 Training Analysis (Cell 37): Added learning progression table, comparison table with all models

### Format Consistency
- All training analysis sections now include:
  - Training dynamics summary (epochs, best val acc, test acc, test loss)
  - Learning progression table (Early/Middle/Late phases with accuracy metrics)
  - Key observations and insights
  - Comparison tables where applicable

## [1.2.0] - 2026-02-04 16:34:45 EST

### Added
- Added model architecture verification sections after each `torchinfo.summary()` output
- SimpleCNN verification: Architecture flow, parameter breakdown (141,354 total), model characteristics
- VGGSmallCIFAR verification: Architecture flow, parameter breakdown by block (1,149,770 total), comparison with SimpleCNN
- ResNet-20 verification: Architecture flow, layer count verification (6n+2=20), parameter breakdown by stage (272,474 total), comparison table

### Documentation
- Each verification section includes detailed parameter calculations
- Added comparison tables showing parameters, model size, mult-adds, and memory usage
- Verified all models are built correctly according to specifications

## [1.1.0] - 2026-02-04

### Updated
- Updated final summary with actual experimental results from training runs
- SimpleCNN: 72.12% test accuracy (141,354 parameters, 16 epochs)
- VGGSmallCIFAR: 87.92% test accuracy (1,149,770 parameters, 28 epochs) - Best model
- ResNet-20: 85.85% test accuracy (272,474 parameters, 48 epochs)
- Added detailed ResNet-20 analysis section as requested ("Please repeat your analysis and then summarize your findings")

### Analysis
- VGGSmallCIFAR achieved the best performance due to higher model capacity
- ResNet-20 showed competitive results with fewer parameters (4x fewer than VGG)
- ResNet-20 trained longest (48 epochs) due to stable learning from skip connections
- SimpleCNN serves as a lightweight baseline but lacks capacity for complex patterns

## [1.0.0] - 2026-02-04

### Added
- Implemented data augmentation in `get_cifar10_loaders()` function using RandomCrop (32x32 with padding=4) and RandomHorizontalFlip
- Implemented training function with AdamW optimizer and CosineAnnealingLR scheduler
- Implemented SimpleCNN model architecture with Conv-BN-ReLU blocks, MaxPool, Dropout, and GlobalAvgPool
- Implemented VGGSmallCIFAR model with three VGG-style blocks (64, 128, 256 channels)
- Completed ResNetCIFAR (ResNet-20) implementation with stem, three residual stages, and classification head
- Added test evaluation code for all three models
- Added analysis and summary of findings comparing all three architectures

### Technical Details
- Model 1 (SimpleCNN): 5 conv layers with batch normalization, 141,354 parameters
- Model 2 (VGGSmallCIFAR): VGG-style architecture with 3 blocks, 1,149,770 parameters
- Model 3 (ResNet-20): CIFAR-style ResNet with BasicBlocks, 272,474 parameters
- All models trained with AdamW (lr=3e-4, weight_decay=1e-2) and early stopping (patience=3)
