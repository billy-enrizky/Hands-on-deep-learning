#!/usr/bin/env python3
"""
Data preparation script for CSC420 Assignment 1 - CNN Dog Breed Classification.

This script prepares the Stanford Dogs Dataset (SDD) and Dog Breed Images (DBI)
datasets by filtering to only include the 7 common breeds, renaming folders
to match, and creating zip files.
"""

import logging
import os
import shutil
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).parent

# Dataset paths
DBI_DIR = BASE_DIR / "DBI"
SDD_DIR = BASE_DIR / "SDD"
SDD_IMAGES_DIR = SDD_DIR / "images" / "Images"
SDD_ANNOTATIONS_DIR = SDD_DIR / "annotations"

# Output directories
DBI_SUBSET_DIR = BASE_DIR / "DBIsubset"
SDD_SUBSET_DIR = BASE_DIR / "SDDsubset"

# The 7 common breeds and their folder names in each dataset
# Format: standardized_name -> (DBI_folder_name, SDD_folder_name)
COMMON_BREEDS = {
    "bernese_mountain_dog": ("bernese_mountain_dog", "n02107683-Bernese_mountain_dog"),
    "border_collie": ("border_collie", "n02106166-Border_collie"),
    "chihuahua": ("chihuahua", "n02085620-Chihuahua"),
    "golden_retriever": ("golden_retriever", "n02099601-golden_retriever"),
    "labrador_retriever": ("labrador", "n02099712-Labrador_retriever"),
    "pug": ("pug", "n02110958-pug"),
    "siberian_husky": ("siberian_husky", "n02110185-Siberian_husky"),
}

# Breeds to delete from DBI (not in common breeds)
DBI_BREEDS_TO_DELETE = ["corgi", "dachshund", "jack_russell"]


def delete_non_common_breeds_from_dbi():
    """Delete folders for breeds that are not in the common 7 breeds from DBI."""
    logger.info("Deleting non-common breeds from DBI...")
    
    for breed in DBI_BREEDS_TO_DELETE:
        breed_path = DBI_DIR / breed
        if breed_path.exists():
            logger.info("  Deleting: %s", breed_path)
            shutil.rmtree(breed_path)
        else:
            logger.warning("  Folder not found (already deleted?): %s", breed_path)
    
    logger.info("Completed deleting non-common breeds from DBI")


def delete_sdd_annotations():
    """Delete the annotations folder from SDD (bounding boxes not needed)."""
    logger.info("Deleting SDD annotations folder...")
    
    if SDD_ANNOTATIONS_DIR.exists():
        # Use ignore_errors=True to handle any permission issues
        shutil.rmtree(SDD_ANNOTATIONS_DIR, ignore_errors=True)
        if not SDD_ANNOTATIONS_DIR.exists():
            logger.info("  Deleted: %s", SDD_ANNOTATIONS_DIR)
        else:
            logger.warning("  Could not fully delete: %s", SDD_ANNOTATIONS_DIR)
    else:
        logger.warning("  Annotations folder not found (already deleted?): %s", SDD_ANNOTATIONS_DIR)
    
    logger.info("Completed deleting SDD annotations")


def create_dbi_subset():
    """Create DBIsubset folder with the 7 common breeds, renamed to standard names."""
    logger.info("Creating DBIsubset folder...")
    
    # Create the subset directory
    if DBI_SUBSET_DIR.exists():
        logger.info("  Removing existing DBIsubset folder...")
        shutil.rmtree(DBI_SUBSET_DIR)
    
    DBI_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    
    for standard_name, (dbi_name, _) in COMMON_BREEDS.items():
        src_path = DBI_DIR / dbi_name
        dst_path = DBI_SUBSET_DIR / standard_name
        
        if src_path.exists():
            logger.info("  Copying: %s -> %s", src_path.name, dst_path.name)
            shutil.copytree(src_path, dst_path)
        else:
            logger.error("  Source folder not found: %s", src_path)
    
    logger.info("Completed creating DBIsubset")


def create_sdd_subset():
    """Create SDDsubset folder with the 7 common breeds, renamed to standard names."""
    logger.info("Creating SDDsubset folder...")
    
    # Create the subset directory
    if SDD_SUBSET_DIR.exists():
        logger.info("  Removing existing SDDsubset folder...")
        shutil.rmtree(SDD_SUBSET_DIR)
    
    SDD_SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    
    for standard_name, (_, sdd_name) in COMMON_BREEDS.items():
        src_path = SDD_IMAGES_DIR / sdd_name
        dst_path = SDD_SUBSET_DIR / standard_name
        
        if src_path.exists():
            logger.info("  Copying: %s -> %s", src_path.name, dst_path.name)
            shutil.copytree(src_path, dst_path)
        else:
            logger.error("  Source folder not found: %s", src_path)
    
    logger.info("Completed creating SDDsubset")


def create_zip_files():
    """Create zip files for DBIsubset and SDDsubset."""
    logger.info("Creating zip files...")
    
    # Create DBIsubset.zip
    dbi_zip_path = BASE_DIR / "DBIsubset.zip"
    if dbi_zip_path.exists():
        dbi_zip_path.unlink()
    
    logger.info("  Creating DBIsubset.zip...")
    with zipfile.ZipFile(dbi_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(DBI_SUBSET_DIR):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(BASE_DIR)
                zipf.write(file_path, arcname)
    logger.info("  Created: %s (%.2f MB)", dbi_zip_path, dbi_zip_path.stat().st_size / (1024 * 1024))
    
    # Create SDDsubset.zip
    sdd_zip_path = BASE_DIR / "SDDsubset.zip"
    if sdd_zip_path.exists():
        sdd_zip_path.unlink()
    
    logger.info("  Creating SDDsubset.zip...")
    with zipfile.ZipFile(sdd_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(SDD_SUBSET_DIR):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(BASE_DIR)
                zipf.write(file_path, arcname)
    logger.info("  Created: %s (%.2f MB)", sdd_zip_path, sdd_zip_path.stat().st_size / (1024 * 1024))
    
    logger.info("Completed creating zip files")


def print_summary():
    """Print a summary of the created datasets."""
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    # DBIsubset summary
    logger.info("\nDBIsubset contents:")
    if DBI_SUBSET_DIR.exists():
        for breed_dir in sorted(DBI_SUBSET_DIR.iterdir()):
            if breed_dir.is_dir():
                num_images = len(list(breed_dir.glob("*.jpg")))
                logger.info("  %s: %d images", breed_dir.name, num_images)
    
    # SDDsubset summary
    logger.info("\nSDDsubset contents:")
    if SDD_SUBSET_DIR.exists():
        for breed_dir in sorted(SDD_SUBSET_DIR.iterdir()):
            if breed_dir.is_dir():
                num_images = len(list(breed_dir.glob("*.jpg")))
                logger.info("  %s: %d images", breed_dir.name, num_images)
    
    # Zip files
    logger.info("\nZip files created:")
    dbi_zip = BASE_DIR / "DBIsubset.zip"
    sdd_zip = BASE_DIR / "SDDsubset.zip"
    if dbi_zip.exists():
        logger.info("  DBIsubset.zip: %.2f MB", dbi_zip.stat().st_size / (1024 * 1024))
    if sdd_zip.exists():
        logger.info("  SDDsubset.zip: %.2f MB", sdd_zip.stat().st_size / (1024 * 1024))
    
    logger.info("=" * 60)


def main():
    """Main function to run all data preparation steps."""
    logger.info("Starting data preparation for CSC420 Assignment 1")
    logger.info("=" * 60)
    
    # Step 1: Delete non-common breeds from DBI
    delete_non_common_breeds_from_dbi()
    
    # Step 2: Delete SDD annotations folder
    delete_sdd_annotations()
    
    # Step 3: Create DBIsubset with standardized folder names
    create_dbi_subset()
    
    # Step 4: Create SDDsubset with standardized folder names
    create_sdd_subset()
    
    # Step 5: Create zip files
    create_zip_files()
    
    # Print summary
    print_summary()
    
    logger.info("Data preparation completed successfully!")


if __name__ == "__main__":
    main()
