#!/usr/bin/env python3
"""
Test script for DCM File Processor

This script creates a test directory structure to validate the processing logic
before running on the actual data.
"""

import os
import tempfile
import shutil
from pathlib import Path
from dcm_file_processor import DCMFileProcessor

def create_test_structure():
    """Create a test directory structure with sample DCM files."""
    
    # Create temporary directory
    test_dir = Path(tempfile.mkdtemp(prefix="dcm_test_"))
    print(f"Created test directory: {test_dir}")
    
    # Create test structure
    # Single subdirectory case
    single_dir = test_dir / "0047140" / "E15047136" / "DX"
    single_dir.mkdir(parents=True)
    
    # Create dummy DCM files
    (single_dir / "1___AP_KNEES_BI_T_STANDING").mkdir()
    (single_dir / "2__PA_FLEXION_B_T_STANDING").mkdir()
    (single_dir / "7__TABLE_LAT_KNEE").mkdir()
    (single_dir / "9__SKYLINE").mkdir()
    
    # Create dummy .dcm files
    for subdir in single_dir.iterdir():
        if subdir.is_dir():
            dcm_file = subdir / "test.dcm"
            dcm_file.write_text("dummy dcm content")
    
    # Multiple subdirectories case
    multi_dir = test_dir / "0408991"
    multi_dir.mkdir()
    
    # Lower value subdirectory
    lower_subdir = multi_dir / "E10809926" / "DX" / "1547657505_Lower_limbs"
    lower_subdir.mkdir(parents=True)
    (lower_subdir / "test1.dcm").write_text("dummy dcm content")
    
    lower_subdir_sr = multi_dir / "E10809926" / "SR" / "1__1"
    lower_subdir_sr.mkdir(parents=True)
    (lower_subdir_sr / "test2.dcm").write_text("dummy dcm content")
    
    # Higher value subdirectory
    higher_subdir = multi_dir / "E11193722" / "DX"
    higher_subdir.mkdir(parents=True)
    
    (higher_subdir / "10_SKYLINE").mkdir()
    (higher_subdir / "2___AP_KNEES_BI_T_STANDING").mkdir()
    (higher_subdir / "8__TABLE_LAT_KNEE").mkdir()
    
    for subdir in higher_subdir.iterdir():
        if subdir.is_dir():
            dcm_file = subdir / "test.dcm"
            dcm_file.write_text("dummy dcm content")
    
    return test_dir

def create_test_csv():
    """Create a test CSV file."""
    csv_content = """EMPI,Date_of_Surgery,Accession_Number,HALS_MRN,Sex,Age,BMI,Side,Type_of_TKA,Patellar_Replacement,Preoperative_KOOS-PS,1-Year_Postop_KOOS-PS
100055527,4/14/17,E15047136,1001,1,75,27,1,1,1,42.1,81.4
100055528,4/15/17,E10809926,1002,0,80,25,0,1,1,50.0,75.0
100055529,4/16/17,E11193722,1003,1,70,30,1,1,1,45.0,80.0"""
    
    csv_file = Path("test_data.csv")
    csv_file.write_text(csv_content)
    return csv_file

def run_test():
    """Run the test."""
    print("Creating test directory structure...")
    test_dir = create_test_structure()
    
    print("Creating test CSV file...")
    csv_file = create_test_csv()
    
    try:
        print("Running DCM processor test...")
        processor = DCMFileProcessor(str(test_dir), str(csv_file), "test_img_repo")
        processor.process_all()
        
        print("\nTest completed successfully!")
        print(f"Check the 'test_img_repo' directory for renamed files")
        print(f"Check 'updated_hals_dataset.csv' for updated CSV data")
        
        # Show results
        if processor.output_dir.exists():
            files = list(processor.output_dir.glob("*.dcm"))
            print(f"\nRenamed files ({len(files)}):")
            for file in sorted(files):
                print(f"  {file.name}")
        
        if Path("updated_hals_dataset.csv").exists():
            print(f"\nUpdated CSV file created with {len(processor.accession_mapping)} accessions processed")
    
    finally:
        # Cleanup
        print("\nCleaning up test files...")
        shutil.rmtree(test_dir, ignore_errors=True)
        csv_file.unlink(missing_ok=True)
        if Path("updated_hals_dataset.csv").exists():
            Path("updated_hals_dataset.csv").unlink()
        if Path("test_img_repo").exists():
            shutil.rmtree("test_img_repo", ignore_errors=True)
        if Path("dcm_processor.log").exists():
            Path("dcm_processor.log").unlink()

if __name__ == "__main__":
    run_test()
