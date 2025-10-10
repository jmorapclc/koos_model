#!/usr/bin/env python3
# data/local_image_processor.py
"""
Local Image Processor for HALS Dataset

This script processes local image files (JPEG or DICOM) and updates the CSV file
with new HALS_MRN identifiers. It can convert DICOM files to JPEG format if needed.

Author: AI Assistant
Date: 2024
"""

import os
import csv
import re
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import pydicom
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('local_image_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LocalImageProcessor:
    """Main class for processing local image files and updating CSV data."""
    
    def __init__(self, local_image_path: str, csv_file_path: str, output_dir: str = "img_repo"):
        """
        Initialize the processor.
        
        Args:
            local_image_path: Path to the directory containing local image files
            csv_file_path: Path to the CSV file to update
            output_dir: Directory to store processed images
        """
        self.local_image_path = Path(local_image_path)
        self.csv_file_path = Path(csv_file_path)
        self.output_dir = Path(output_dir)
        self.processed_files = {}  # Maps original files to processed files
        self.hals_mrn_list = []  # List of HALS_MRN values from processed files
        
    def validate_filename_format(self, filename: str) -> bool:
        """
        Validate if filename matches expected format (6 digits + optional 'a' + extension).
        
        Args:
            filename: Name of the file to validate
            
        Returns:
            True if format is valid, False otherwise
        """
        # Pattern: 6 digits, optional 'a', then extension
        pattern = r'^\d{6}a?\.(jpg|jpeg|dcm|dicom)$'
        return bool(re.match(pattern, filename.lower()))
    
    def extract_hals_mrn(self, filename: str) -> str:
        """
        Extract HALS_MRN from filename (remove extension).
        
        Args:
            filename: Name of the file
            
        Returns:
            HALS_MRN identifier
        """
        return Path(filename).stem
    
    def find_image_files(self, directory: Path) -> List[Path]:
        """
        Find all image files in the directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of Path objects for image files
        """
        image_files = []
        image_extensions = {'.jpg', '.jpeg', '.dcm', '.dicom'}
        
        try:
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    if self.validate_filename_format(file_path.name):
                        image_files.append(file_path)
                    else:
                        logger.warning(f"Invalid filename format: {file_path.name}")
        except Exception as e:
            logger.error(f"Error searching for image files in {directory}: {e}")
        
        return sorted(image_files)
    
    def convert_dicom_to_jpeg(self, dcm_path: Path, output_path: Path) -> bool:
        """
        Convert DICOM file to JPEG format.
        
        Args:
            dcm_path: Path to DICOM file
            output_path: Path for output JPEG file
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Read DICOM file
            ds = pydicom.dcmread(str(dcm_path))
            
            # Get pixel array
            pixel_array = ds.pixel_array
            
            # Convert to PIL Image
            if pixel_array.dtype != 'uint8':
                # Normalize to 0-255 range
                pixel_array = ((pixel_array - pixel_array.min()) / 
                             (pixel_array.max() - pixel_array.min()) * 255).astype('uint8')
            
            image = Image.fromarray(pixel_array)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as JPEG
            image.save(output_path, 'JPEG', quality=95)
            logger.info(f"Converted DICOM to JPEG: {dcm_path.name} -> {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting DICOM file {dcm_path}: {e}")
            return False
    
    def process_image_file(self, image_path: Path) -> Optional[Path]:
        """
        Process a single image file (convert DICOM to JPEG if needed).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to processed file, or None if processing failed
        """
        file_extension = image_path.suffix.lower()
        hals_mrn = self.extract_hals_mrn(image_path.name)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        if file_extension in ['.jpg', '.jpeg']:
            # Already JPEG, just copy
            output_path = self.output_dir / f"{hals_mrn}.jpeg"
            try:
                shutil.copy2(image_path, output_path)
                logger.info(f"Copied JPEG: {image_path.name} -> {output_path.name}")
                return output_path
            except Exception as e:
                logger.error(f"Error copying JPEG file {image_path}: {e}")
                return None
                
        elif file_extension in ['.dcm', '.dicom']:
            # Convert DICOM to JPEG
            output_path = self.output_dir / f"{hals_mrn}.jpeg"
            if self.convert_dicom_to_jpeg(image_path, output_path):
                return output_path
            else:
                return None
        else:
            logger.warning(f"Unsupported file format: {image_path}")
            return None
    
    def process_all_images(self) -> None:
        """
        Process all image files in the local directory.
        """
        logger.info(f"Processing images from: {self.local_image_path}")
        
        # Check if source path exists
        if not self.local_image_path.exists():
            logger.error(f"Source path does not exist: {self.local_image_path}")
            return
        
        # Find all image files
        image_files = self.find_image_files(self.local_image_path)
        
        if not image_files:
            logger.warning("No valid image files found")
            return
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        # Process each image file
        for image_path in image_files:
            try:
                processed_path = self.process_image_file(image_path)
                if processed_path:
                    hals_mrn = self.extract_hals_mrn(processed_path.name)
                    self.processed_files[str(image_path)] = str(processed_path)
                    self.hals_mrn_list.append(hals_mrn)
                    logger.info(f"Processed: {image_path.name} -> {processed_path.name}")
                else:
                    logger.error(f"Failed to process: {image_path.name}")
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
        
        logger.info(f"Successfully processed {len(self.processed_files)} files")
    
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
        Update CSV data with new HALS_MRN values from processed images.
        
        Args:
            data: Original CSV data
            
        Returns:
            Updated CSV data with new rows for processed images
        """
        updated_data = []
        
        # Create a set of existing HALS_MRN values for quick lookup
        existing_hals_mrn = {row.get('HALS_MRN', '') for row in data}
        
        # Add new rows for processed images
        for hals_mrn in self.hals_mrn_list:
            if hals_mrn not in existing_hals_mrn:
                # Create new row with default values
                new_row = {
                    'HALS_MRN': hals_mrn,
                    'Sex': '',  # Empty values to be filled manually
                    'Age': '',
                    'BMI': '',
                    'Side': '',
                    'Type_of_TKA': '',
                    'Patellar_Replacement': '',
                    'Preoperative_KOOS-PS': '',
                    '1-Year_Postop_KOOS-PS': ''
                }
                updated_data.append(new_row)
                logger.info(f"Added new row for HALS_MRN: {hals_mrn}")
            else:
                logger.info(f"HALS_MRN already exists in CSV: {hals_mrn}")
        
        # Add all original data
        updated_data.extend(data)
        
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
        logger.info("Starting local image processing")
        
        # Process all images
        self.process_all_images()
        
        if not self.processed_files:
            logger.warning("No files were processed successfully")
            return
        
        # Read and update CSV data
        logger.info("Reading CSV data")
        csv_data = self.read_csv_data()
        
        if csv_data:
            logger.info("Updating CSV data with new HALS_MRN values")
            updated_data = self.update_csv_data(csv_data)
            
            # Write updated CSV
            self.write_updated_csv(updated_data)
        
        logger.info("Processing completed successfully")
        logger.info(f"Total files processed: {len(self.processed_files)}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")


def get_user_input() -> str:
    """
    Get the local image directory path from user input.
    
    Returns:
        Path to the local image directory
    """
    while True:
        user_input = input("Enter the path to the directory containing image files: ").strip()
        
        if not user_input:
            print("Please enter a valid path.")
            continue
        
        # Expand user path (handle ~ and relative paths)
        expanded_path = os.path.expanduser(user_input)
        path = Path(expanded_path)
        
        if not path.exists():
            print(f"Path does not exist: {path}")
            continue
        
        if not path.is_dir():
            print(f"Path is not a directory: {path}")
            continue
        
        return str(path)


def main():
    """Main function to run the processor."""
    print("Local Image Processor for HALS Dataset")
    print("=" * 40)
    
    # Get user input for local image directory
    local_image_path = get_user_input()
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):
        logger.error(f"CSV file not found: {CSV_FILE}")
        logger.error("Please run this script from the project root directory")
        return
    
    # Create processor instance
    processor = LocalImageProcessor(local_image_path, CSV_FILE, OUTPUT_DIR)
    
    # Run processing
    processor.process_all()


if __name__ == "__main__":
    main()
