# Changelog

All notable changes to this project will be documented in this file.

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
