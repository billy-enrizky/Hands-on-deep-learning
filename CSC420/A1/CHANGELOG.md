# Changelog

All notable changes to this project will be documented in this file.

## [0.9.1] - 2026-01-21 18:00

### Updated
- `report.ipynb`: Updated Task V analysis cell with actual experimental results
  - **Results Summary** (sorted by Test Accuracy):
    - Swin-T: **91.84%** test accuracy (DBI 84.54%, SDD 95.68%) - best overall
    - ResNeXt-50: 89.01% test accuracy (DBI 84.54%, SDD 91.35%)
    - ResNet-18: 87.94% test accuracy (DBI 88.66%, SDD 87.57%) - most balanced
  - **Key findings**:
    - SDD images easier to detect (87-96%) than DBI images (84-89%)
    - Swin Transformer outperforms CNNs for dataset detection
    - All models achieve ~99.5% training accuracy but 88-92% test accuracy
  - **Confusion matrices** for all three models
  - **Conclusions**:
    - Dataset detection achieves 91.84% accuracy, confirming domain shift
    - Self-attention more effective than convolutions for this task
    - Results validate domain shift concerns from earlier tasks

### Experimental Results Summary (Task V)
- **Best Test Accuracy**: Swin-T (91.84%)
- **Best DBI Detection**: ResNet-18 (88.66%)
- **Best SDD Detection**: Swin-T (95.68%)
- **Dataset Split**: 1315 train / 282 val / 282 test (stratified)

## [0.9.0] - 2026-01-21 16:49

### Added
- `report.ipynb`: Task V - Dataset Detection (80 marks)
  - **Binary classification model** to distinguish DBI vs SDD images
  - Custom `DatasetDetectionDataset` class combining both datasets with binary labels
  - Stratified train/val/test split (70/15/15) preserving class distribution
  - **Fine-tuning 3 pretrained models** for binary classification:
    1. ResNeXt-50 (primary model) - 23.0M parameters
    2. Swin-T (comparison) - 27.5M parameters
    3. ResNet-18 (baseline) - 11.2M parameters
  - Training configuration: 224x224 input, Adam optimizer, LR=0.0001, 15 epochs
  - Comprehensive evaluation with per-class accuracy and confusion matrices
  - Visualization: test accuracy comparison, per-class detection, training curves, confusion matrices
  - Results saved to `task5_results.pkl` for persistence

### Model Selection Justification
- **ResNeXt-50** chosen as primary model based on Task IV results:
  - Best domain generalization (smallest gap +1.22% between DBI and SDD)
  - Grouped convolutions learn diverse, domain-invariant features
  - Appropriate capacity (23M parameters) for detecting dataset characteristics

### Technical Details
- Combined dataset: 646 DBI + 1233 SDD = 1879 total images
- Binary labels: DBI=0, SDD=1
- Stratified splitting ensures class ratio preserved across splits
- Same data augmentation strategy as Task IV (random crop, flip, color jitter, rotation)

### Data Organization
- Custom `DatasetDetectionDataset` class (alternative to reorganizing with ImageFolder)
- `SplitDataset` wrapper for applying different transforms to train/val/test
- Efficient data loading without physically reorganizing the dataset structure

## [0.8.1] - 2026-01-15 15:43

### Updated
- `report.ipynb`: Updated Task IV analysis cell with actual experimental results
  - **Results Summary** (sorted by DBI Test Accuracy):
    - Swin-T: DBI 97.96%, SDD 95.86%, Gap +2.10% (best overall)
    - ResNet-34: DBI 96.94%, SDD 88.97%, Gap +7.97%
    - ResNeXt-50: DBI 94.90%, SDD 93.67%, Gap +1.22% (best generalization)
    - ResNet-18: DBI 93.88%, SDD 89.13%, Gap +4.75%
    - EfficientNet-B0: DBI 84.69%, SDD 76.48%, Gap +8.21%
  - **Cross-dataset performance analysis**:
    - ResNeXt-50 has smallest gap (+1.22%) - best domain generalization
    - Swin-T achieves highest accuracy on both datasets
    - ResNet-34 vs ResNeXt-50: similar DBI accuracy but 4.7% difference on SDD
  - **Comparison with Task III** (training from scratch):
    - ResNet-18 fine-tuned: 93.88% vs 41.84% (+52.04% improvement)
    - SDD accuracy: 89.13% vs 30.82% (+58.31% improvement)
    - Performance gap reduced from +11.02% to +4.75%
  - **Key conclusions**:
    - Transfer learning is essential for small datasets
    - Swin Transformer achieves best overall performance
    - ResNeXt-50 has best cross-domain generalization
    - Architecture choice significantly impacts domain transfer

### Experimental Results Summary (Task IV)
- **Best DBI Test Accuracy**: Swin-T (97.96%)
- **Best SDD Accuracy**: Swin-T (95.86%)
- **Best Generalization (smallest gap)**: ResNeXt-50 (+1.22%)
- **Fine-tuning vs Scratch**: +52% improvement on DBI, +58% on SDD

## [0.8.0] - 2026-01-15 12:43

### Added
- `report.ipynb`: Task IV - Fine-tuning on the DBI (100 marks)
  - **Fine-tuning 5 pretrained models** on DBI dataset:
    1. ResNet-18 (torchvision) - 11.2M parameters
    2. ResNet-34 (torchvision) - 21.3M parameters
    3. ResNeXt-50 (torchvision) - 23.0M parameters
    4. Swin Transformer tiny (torchvision) - 28.3M parameters
    5. EfficientNet-B0 (timm) - 4.0M parameters
  - Transfer learning with ImageNet pretrained weights
  - Final layer replacement for 7-class dog breed classification
  - Training configuration: 224x224 input, Adam optimizer, LR=0.0001, 10 epochs
  - Evaluation on both DBI test set and entire SDD dataset
  - Cross-dataset performance comparison and analysis
  - Visualization: DBI vs SDD accuracy, performance gaps, training history
  - Results saved to `task4_results.pkl` for persistence

### Dependencies
- Added `timm>=1.0.0` to `pyproject.toml` for EfficientNet-B0 model

### Technical Details
- Fine-tuning uses lower learning rate (0.0001) than training from scratch
- Best model checkpoint saved during training based on validation accuracy
- Same data split (70/15/15) and augmentation strategy as previous tasks
- Cross-dataset evaluation measures generalization to different data distribution

## [0.7.5] - 2026-01-15 12:13

### Updated
- `report.ipynb`: Updated Task III analysis cells with actual experimental results
  - **Task III.a Analysis** (Cell 34):
    - Full comparison table with all metrics (architecture, parameters, epochs, accuracies, gaps)
    - ResNet-18: Train 52.65%, Val 33.33%, Test 41.84%, Gap +19.32%
    - Key finding: SimpleCNN (no dropout) achieves best test accuracy (47.96%)
    - Analysis of overfitting in ResNet-18 due to high capacity vs small dataset
    - Conclusion: model complexity must match dataset size
  - **Task III.b Analysis** (Cell 39):
    - DBI Test: 41.84%, SDD: 30.82%, Difference: +11.02%
    - Confirmed DBI accuracy higher due to domain shift
    - Concise 1-2 sentence explanation as required by assignment

### Experimental Results Summary (Task III)
- **ResNet-18 (from scratch)**: Train 52.65%, Val 33.33%, Test 41.84%, Gap +19.32%
- **DBI vs SDD**: DBI test 41.84% > SDD 30.82% (domain shift)

## [0.7.4] - 2026-01-15 11:41

### Fixed
- `report.ipynb`: Task III comparison cell failing with `NameError: name 'history_with_dropout' is not defined`
  - Added pickle save/load mechanism to persist results between tasks
  - Task II summary cell (Cell 24) now saves results to `task2_results.pkl`
  - Task III ResNet evaluation cell (Cell 32) now saves results to `task3_results.pkl`
  - Task III comparison cell (Cell 33) now loads both pickle files instead of relying on in-memory variables
  - Task III.b cells (Cells 37, 38) now load `resnet_test_acc` from pickle
  - This allows cells to be run independently without re-running all previous cells

## [0.7.3] - 2026-01-15 11:31

### Fixed
- `report.ipynb`: Task II (SimpleCNN) data loading - removed redundant transform
  - Changed `datasets.ImageFolder(DATA_DIR, transform=train_transform)` to `transform=None`
  - Now consistent with Task III fix - transforms only applied via `TransformSubset`
  - Updated comments to match Task III style

## [0.7.2] - 2026-01-15 11:11

### Fixed
- `report.ipynb`: Task III.a logging not showing output in Jupyter notebook
  - Added `logging.basicConfig()` before `logging.getLogger(__name__)` 
  - Without basicConfig, the logger has no handler and messages are not displayed
  - Now consistent with logging setup in Task I and Task II cells

## [0.7.1] - 2026-01-15 10:58

### Fixed
- `report.ipynb`: Task III data loading - removed redundant transform
  - Changed `datasets.ImageFolder(DATA_DIR, transform=resnet_train_transform)` to `transform=None`
  - Transforms are now only applied via `TransformSubset` class (cleaner, no double transformation)
  - Added clearer comments explaining why different transforms are needed for train vs val/test

## [0.7.0] - 2026-01-15 10:37

### Added
- `report.ipynb`: Task III - ResNet Training on the DBI (80 marks)
  - **Task III.a (40 marks)**: ResNet-18 training from scratch
    - Load ResNet-18 architecture from PyTorch (without pretrained weights)
    - Modified input/output layers to match DBI dataset (7 classes)
    - Training configuration: 224x224 input, Adam optimizer, LR=0.001, 15 epochs
    - Training and validation accuracy plots
    - Comparison table: ResNet-18 vs SimpleCNN (with/without dropout)
    - Analysis of model capacity vs dataset size trade-offs
  - **Task III.b (40 marks)**: ResNet-18 evaluation on SDD dataset
    - Evaluate trained model on entire SDDsubset (1,233 images)
    - Compare DBI test accuracy vs SDD accuracy
    - Analysis of domain shift between DBI and SDD datasets
    - Explanation of why accuracy differs between datasets

### Technical Details
- ResNet-18 parameters: ~11.2M (vs ~68K for SimpleCNN)
- Input size: 224x224 (vs 64x64 for SimpleCNN)
- Same data augmentation strategy as Task II
- Same train/val/test split (70/15/15) with seed=42 for fair comparison

## [0.6.2] - 2026-01-14 17:58

### Updated
- `report.ipynb`: Updated Task II analysis with actual experimental results
  - Documented hyperparameter tuning findings:
    - Best config: Adam, LR=0.001, WD=1e-4 (37.50% val accuracy)
    - Adam outperformed SGD at LR=0.001; SGD better at LR=0.01
    - LR=0.0001 too slow for convergence in 5 epochs
  - Added results table comparing dropout vs no-dropout models
  - Explained negative generalization gap (-11.42%) with dropout
  - Analyzed why no-dropout model has higher test accuracy despite overfitting
  - Discussed small dataset challenges (646 images, 7 classes)

### Experimental Results Summary
- **With Dropout**: Train 29.20%, Val 40.62%, Test 34.69%, Gap -11.42%
- **Without Dropout**: Train 53.10%, Val 39.58%, Test 47.96%, Gap +13.51%

## [0.6.1] - 2026-01-14 16:39

### Changed
- `report.ipynb`: Replaced `x.view(x.size(0), -1)` with `nn.Flatten()` in SimpleCNN model
  - Added `self.flatten = nn.Flatten()` as a proper module in `__init__`
  - Using `nn.Flatten()` is cleaner, more explicit, and follows PyTorch best practices
  - The flatten layer now appears in the model architecture printout

## [0.6.0] - 2026-01-14 16:35

### Added
- `report.ipynb`: Hyperparameter tuning for Task II
  - Grid search over learning rates [0.01, 0.001, 0.0001]
  - Grid search over optimizers [Adam, SGD with momentum=0.9]
  - Grid search over weight decay [0, 1e-4, 1e-3]
  - 5 epochs per configuration for tuning
  - Heatmap visualization of tuning results
  - Automatic selection of best hyperparameters for final training

### Changed
- Training cells now use best hyperparameters from tuning instead of hardcoded values
- Updated analysis markdown to document hyperparameter tuning process

## [0.5.1] - 2026-01-14 01:11

### Fixed
- `report.ipynb`: Updated Task II analysis to accurately reflect experimental results
  - Corrected explanation of negative generalization gap with dropout (validation > training accuracy)
  - Explained why dropout model has lower test accuracy (needs more epochs to converge)
  - Added detailed analysis of overfitting evidence in model without dropout
  - Clarified the trade-off between regularization and convergence speed

## [0.5.0] - 2026-01-13 22:13

### Added
- `report.ipynb`: Task II - Simple CNN Training on DBI (60 marks)
  - Implemented SimpleCNN architecture with configurable dropout
  - CNN Architecture: Conv(16)->BN->Conv(16)->MaxPool->Conv(8)->BN->Conv(8)->MaxPool->Dropout->FC(32)->Dropout->Softmax
  - Data augmentation: random cropping, horizontal flipping, color jitter, rotation
  - Training with Adam optimizer, CrossEntropyLoss, and learning rate scheduler
  - Dataset split: 70% train, 15% validation, 15% test

### Task II Implementation
- **Model Architecture**:
  - 4 convolutional layers (16-16-8-8 filters, 3x3 kernels)
  - 2 batch normalization layers
  - 2 max pooling layers (2x2)
  - Dropout layers (rate=0.5, configurable)
  - Fully connected layer (32 units)
  - ReLU activation throughout
  
- **Training Configuration**:
  - Image size: 64x64
  - Batch size: 32
  - Learning rate: 0.001
  - Weight decay: 1e-4
  - Epochs: 10
  
- **Experiments**:
  - Training with dropout (0.5) - plots training/validation accuracy over 10 epochs
  - Training without dropout - plots training/validation accuracy over 10 epochs
  - Comparison analysis of dropout impact on generalization
  - Test accuracy reported for both models

## [0.4.0] - 2026-01-13 21:07

### Added
- `pyproject.toml`: Project configuration for isolated Python environment
  - Python >=3.10 required
  - Dependencies: numpy, matplotlib, pillow, torch, torchvision, scikit-learn, ipykernel, jupyter
  - Dev dependencies: ipython

### Environment Setup
- Created `.venv` virtual environment using `uv venv`
- Installed all dependencies with `uv sync` (117 packages)
- Registered Jupyter kernel: `csc420-a1-cnn` (display name: "CSC420 A1 CNN")
- Kernel location: `/Users/billy/Library/Jupyter/kernels/csc420-a1-cnn`

### Usage
```bash
# Activate environment
source .venv/bin/activate

# Or run directly with uv
uv run python script.py
uv run jupyter notebook
```

## [0.3.0] - 2026-01-13 16:45

### Added
- `report.ipynb`: Jupyter notebook for CSC420 Assignment 1 report
  - Task I - Inspection: Visual comparison and statistical analysis of DBIsubset vs SDDsubset
  - Code to display side-by-side image comparisons across all 7 breeds
  - Image statistics analysis (width, height, aspect ratio, file size distributions)
  - Visualization of image dimension distributions (histograms and scatter plots)
  - Detailed written analysis of systematic differences between datasets

### Task I Findings
- **DBIsubset**: Professional/stock photos, clean backgrounds, well-centered dogs, variable resolutions
- **SDDsubset**: User-generated/web-scraped images, cluttered backgrounds, candid poses, more consistent resolution
- Identified implications for CNN training including domain shift concerns

## [0.2.0] - 2026-01-13 16:26

### Added
- Installed `cursor-notebook-mcp` MCP server for Jupyter notebook editing in Cursor
  - Installed via pipx with Python 3.13
  - Injected compatible pydantic version (<2.12.0) to fix compatibility issue
  - Configured in `~/.cursor/mcp.json` with stdio transport
  - Allowed roots: `/Users/billy/Documents/CSC420-A1-CNN` and `/Users/billy/Documents`

### Configuration
- MCP server location: `/Users/billy/.local/bin/cursor-notebook-mcp`
- Transport: stdio (Cursor manages the server automatically)
- Available tools: notebook_create, notebook_read, notebook_edit_cell, notebook_add_cell, and 20+ more

## [0.1.0] - 2026-01-13

### Added
- `prepare_data.py`: Data preparation script for CSC420 Assignment 1 - CNN Dog Breed Classification
  - Filters datasets to include only 7 common dog breeds present in both SDD and DBI:
    - Bernese mountain dog
    - Border collie
    - Chihuahua
    - Golden retriever
    - Labrador retriever
    - Pug
    - Siberian husky
  - Deletes non-common breeds from DBI (corgi, dachshund, jack_russell)
  - Deletes SDD annotations folder (bounding boxes not needed for classification)
  - Creates `DBIsubset/` folder with standardized breed folder names
  - Creates `SDDsubset/` folder with standardized breed folder names
  - Generates `DBIsubset.zip` and `SDDsubset.zip`.

### Dataset Summary
- **DBIsubset**: 646 total images
  - bernese_mountain_dog: 96 images
  - border_collie: 93 images
  - chihuahua: 92 images
  - golden_retriever: 83 images
  - labrador_retriever: 93 images
  - pug: 94 images
  - siberian_husky: 95 images

- **SDDsubset**: 1233 total images
  - bernese_mountain_dog: 218 images
  - border_collie: 150 images
  - chihuahua: 152 images
  - golden_retriever: 150 images
  - labrador_retriever: 171 images
  - pug: 200 images
  - siberian_husky: 192 images

### Zip Files
- `DBIsubset.zip`: 66.88 MB
- `SDDsubset.zip`: 43.17 MB
