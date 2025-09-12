#!/usr/bin/env python3
"""
DCM File Processor for HALS Dataset

This script processes DICOM files from a tree structure and renames them according to
specific rules, then updates the corresponding CSV file with new HALS_MRN values.

Author: AI Assistant
Date: 2024
"""

import os
import shutil
import csv
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DCMFileProcessor:
    """Main class for processing DCM files and updating CSV data."""
    
    def __init__(self, source_path: str, csv_file_path: str, output_dir: str = "img_repo"):
        """
        Initialize the processor.
        
        Args:
            source_path: Path to the source directory containing DCM files
            csv_file_path: Path to the CSV file to update
            output_dir: Directory to store renamed DCM files
        """
        self.source_path = Path(source_path)
        self.csv_file_path = Path(csv_file_path)
        self.output_dir = Path(output_dir)
        self.file_counter = STARTING_COUNTER  # Starting counter for main directories
        self.file_mapping = {}  # Maps old paths to new names
        self.accession_mapping = {}  # Maps accession numbers to file lists
        
    def find_dcm_files(self, directory: Path) -> List[Path]:
        """
        Recursively find all .dcm files in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of Path objects for .dcm files
        """
        dcm_files = []
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.dcm'):
                        dcm_files.append(Path(root) / file)
        except Exception as e:
            logger.error(f"Error searching for DCM files in {directory}: {e}")
        return sorted(dcm_files)
    
    def get_first_level_subdirs(self, main_dir: Path) -> List[Path]:
        """
        Get first-level subdirectories of a main directory.
        
        Args:
            main_dir: Main directory path
            
        Returns:
            List of first-level subdirectory paths
        """
        try:
            subdirs = [item for item in main_dir.iterdir() 
                      if item.is_dir() and not item.name.startswith('.')]
            return sorted(subdirs, key=lambda x: x.name)
        except Exception as e:
            logger.error(f"Error getting subdirectories for {main_dir}: {e}")
            return []
    
    def process_single_subdir(self, subdir: Path, accession_number: str, 
                            use_suffix: bool = False) -> List[str]:
        """
        Process a single subdirectory and return list of new file names.
        
        Args:
            subdir: Subdirectory to process
            accession_number: Accession number for this directory
            use_suffix: Whether to add 'a' suffix to filenames
            
        Returns:
            List of new file names created
        """
        dcm_files = self.find_dcm_files(subdir)
        new_names = []
        
        for i, dcm_file in enumerate(dcm_files, 1):
            # Create new filename: 6 digits (4 for counter + 2 for file number)
            file_number = f"{i:02d}"
            if use_suffix:
                new_name = f"{self.file_counter:04d}{file_number}a.dcm"
            else:
                new_name = f"{self.file_counter:04d}{file_number}.dcm"
            
            # Store mapping
            self.file_mapping[str(dcm_file)] = new_name
            
            # Add to accession mapping
            if accession_number not in self.accession_mapping:
                self.accession_mapping[accession_number] = []
            self.accession_mapping[accession_number].append(new_name)
            
            new_names.append(new_name)
            logger.info(f"Mapping: {dcm_file} -> {new_name}")
        
        return new_names
    
    def process_main_directory(self, main_dir: Path) -> None:
        """
        Process a main directory and all its subdirectories.
        
        Args:
            main_dir: Main directory to process
        """
        accession_number = main_dir.name
        logger.info(f"Processing main directory: {accession_number}")
        
        subdirs = self.get_first_level_subdirs(main_dir)
        
        if len(subdirs) == 0:
            logger.warning(f"No subdirectories found in {main_dir}")
            return
        elif len(subdirs) == 1:
            # Single subdirectory - process normally
            logger.info(f"Single subdirectory found: {subdirs[0].name}")
            self.process_single_subdir(subdirs[0], accession_number)
        else:
            # Multiple subdirectories - sort and process with different logic
            logger.info(f"Multiple subdirectories found: {[s.name for s in subdirs]}")
            
            # Sort subdirectories by name
            sorted_subdirs = sorted(subdirs, key=lambda x: x.name)
            
            # Process lowest value with 'a' suffix
            logger.info(f"Processing lowest subdirectory with 'a' suffix: {sorted_subdirs[0].name}")
            self.process_single_subdir(sorted_subdirs[0], accession_number, use_suffix=True)
            
            # Process highest value without suffix
            logger.info(f"Processing highest subdirectory without suffix: {sorted_subdirs[-1].name}")
            self.process_single_subdir(sorted_subdirs[-1], accession_number, use_suffix=False)
        
        # Increment counter for next main directory
        self.file_counter += 1
    
    def copy_files_to_output(self) -> None:
        """
        Copy all DCM files to the output directory with new names.
        """
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Created output directory: {self.output_dir}")
        
        # Copy files
        for old_path, new_name in self.file_mapping.items():
            try:
                source_file = Path(old_path)
                dest_file = self.output_dir / new_name
                
                if source_file.exists():
                    shutil.copy2(source_file, dest_file)
                    logger.info(f"Copied: {source_file} -> {dest_file}")
                else:
                    logger.warning(f"Source file not found: {source_file}")
            except Exception as e:
                logger.error(f"Error copying {old_path} to {new_name}: {e}")
    
    def read_csv_data(self) -> List[Dict]:
        """
        Read the CSV file and return data as list of dictionaries.
        
        Returns:
            List of dictionaries representing CSV rows
        """
        data = []
        try:
            with open(self.csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
            logger.info(f"Read {len(data)} rows from CSV file")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
        return data
    
    def update_csv_data(self, data: List[Dict]) -> List[Dict]:
        """
        Update CSV data with new HALS_MRN values.
        
        Args:
            data: Original CSV data
            
        Returns:
            Updated CSV data with expanded rows
        """
        updated_data = []
        
        for row in data:
            accession_number = row.get('Accession_Number', '')
            
            if accession_number in self.accession_mapping:
                # This row has corresponding DCM files
                file_names = self.accession_mapping[accession_number]
                
                for file_name in file_names:
                    # Create new row with updated HALS_MRN
                    new_row = row.copy()
                    new_row['HALS_MRN'] = file_name.replace('.dcm', '')  # Remove .dcm extension
                    updated_data.append(new_row)
                    logger.info(f"Updated row for {accession_number} with HALS_MRN: {new_row['HALS_MRN']}")
            else:
                # No corresponding DCM files, keep original row
                updated_data.append(row)
        
        logger.info(f"Updated CSV data: {len(data)} -> {len(updated_data)} rows")
        return updated_data
    
    def write_updated_csv(self, data: List[Dict], output_file: str = OUTPUT_CSV_FILE) -> None:
        """
        Write updated CSV data to file.
        
        Args:
            data: Updated CSV data
            output_file: Output filename
        """
        if not data:
            logger.warning("No data to write")
            return
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            logger.info(f"Updated CSV written to: {output_file}")
        except Exception as e:
            logger.error(f"Error writing updated CSV: {e}")
    
    def process_all(self) -> None:
        """
        Main processing function that orchestrates the entire workflow.
        """
        logger.info("Starting DCM file processing")
        
        # Check if source path exists
        if not self.source_path.exists():
            logger.error(f"Source path does not exist: {self.source_path}")
            return
        
        # Get all main directories and sort them
        main_dirs = [item for item in self.source_path.iterdir() 
                    if item.is_dir() and not item.name.startswith('.')]
        main_dirs.sort(key=lambda x: x.name)
        
        logger.info(f"Found {len(main_dirs)} main directories to process")
        
        # Process each main directory
        for main_dir in main_dirs:
            try:
                self.process_main_directory(main_dir)
            except Exception as e:
                logger.error(f"Error processing main directory {main_dir}: {e}")
                continue
        
        # Copy files to output directory
        logger.info("Copying files to output directory")
        self.copy_files_to_output()
        
        # Read and update CSV data
        logger.info("Reading CSV data")
        csv_data = self.read_csv_data()
        
        if csv_data:
            logger.info("Updating CSV data with new HALS_MRN values")
            updated_data = self.update_csv_data(csv_data)
            
            # Write updated CSV
            self.write_updated_csv(updated_data)
        
        logger.info("Processing completed successfully")
        logger.info(f"Total files processed: {len(self.file_mapping)}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")


def main():
    """Main function to run the processor."""
    # Check if we're running from the correct directory
    if not os.path.exists(CSV_FILE):
        logger.error(f"CSV file not found: {CSV_FILE}")
        logger.error("Please run this script from the project root directory")
        return
    
    # Create processor instance
    processor = DCMFileProcessor(SOURCE_PATH, CSV_FILE, OUTPUT_DIR)
    
    # Run processing
    processor.process_all()


if __name__ == "__main__":
    main()
