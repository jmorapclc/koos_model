#!/usr/bin/env python3
# data/test_local_processor.py
"""
Test script for Local Image Processor

This script tests the local image processor functionality.
"""

import os
import sys
from pathlib import Path

# Add the data directory to the path so we can import the processor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from local_image_processor import LocalImageProcessor
from config import *


def test_filename_validation():
    """Test filename validation functionality."""
    print("Testing filename validation...")
    
    processor = LocalImageProcessor(LOCAL_DCM_FILE_PATH, CSV_FILE)
    
    # Test valid filenames
    valid_filenames = [
        "100101.jpeg",
        "100102a.jpg", 
        "111501.dcm",
        "111502a.dicom"
    ]
    
    # Test invalid filenames
    invalid_filenames = [
        "1001.jpeg",  # Too short
        "1001011.jpeg",  # Too long
        "abc123.jpeg",  # Non-numeric
        "100101.txt",  # Wrong extension
        "100101.jpeg.bak"  # Extra extension
    ]
    
    print("Valid filenames:")
    for filename in valid_filenames:
        is_valid = processor.validate_filename_format(filename)
        print(f"  {filename}: {'✓' if is_valid else '✗'}")
    
    print("\nInvalid filenames:")
    for filename in invalid_filenames:
        is_valid = processor.validate_filename_format(filename)
        print(f"  {filename}: {'✓' if is_valid else '✗'}")


def test_hals_mrn_extraction():
    """Test HALS_MRN extraction functionality."""
    print("\nTesting HALS_MRN extraction...")
    
    processor = LocalImageProcessor(LOCAL_DCM_FILE_PATH, CSV_FILE)
    
    test_cases = [
        ("100101.jpeg", "100101"),
        ("100102a.jpg", "100102a"),
        ("111501.dcm", "111501"),
        ("111502a.dicom", "111502a")
    ]
    
    for filename, expected in test_cases:
        result = processor.extract_hals_mrn(filename)
        print(f"  {filename} -> {result} {'✓' if result == expected else '✗'}")


def test_image_discovery():
    """Test image file discovery functionality."""
    print("\nTesting image file discovery...")
    
    processor = LocalImageProcessor(LOCAL_DCM_FILE_PATH, CSV_FILE)
    
    # Test with the actual img_repo directory
    img_repo_path = Path("data/img_repo")
    if img_repo_path.exists():
        image_files = processor.find_image_files(img_repo_path)
        print(f"Found {len(image_files)} valid image files")
        
        # Show first 10 files
        for i, file_path in enumerate(image_files[:10]):
            print(f"  {i+1}. {file_path.name}")
        
        if len(image_files) > 10:
            print(f"  ... and {len(image_files) - 10} more files")
    else:
        print(f"Directory {img_repo_path} does not exist")


def main():
    """Run all tests."""
    print("Local Image Processor Test Suite")
    print("=" * 40)
    
    test_filename_validation()
    test_hals_mrn_extraction()
    test_image_discovery()
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()
