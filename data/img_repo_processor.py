#!/usr/bin/env python3
# data/img_repo_processor.py
"""
Image Repository Processor for HALS Dataset

This script processes image files in the img_repo directory:
- Identifies file types (.dicom, .jpg, .jpeg, or other image formats)
- Converts .dicom files to .jpeg format
- Renames files to match HALS_MRN values from CSV file
- Preserves names that already follow the naming convention

Naming Convention (following dcm_to_hals_id.txt rules):
- 6 digits: First 4 digits start from 1116 and increment for each main directory
- Last 2 digits: Sequential file numbers (01, 02, 03, etc.)
- Optional 'a' suffix: For files from lower alphanumeric subdirectories
- Examples: 111601.jpeg, 111602.jpeg, 111901a.jpeg, 111901.jpeg

The script validates that filenames follow this pattern and ensures new names
match available HALS_MRN values from the CSV metadata file.

Author: AI Assistant
Date: 2024
"""

import os
import csv
import re
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from PIL import Image
import pydicom
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('img_repo_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImgRepoProcessor:
    """Main class for processing image files in img_repo directory."""
    
    def __init__(self, img_repo_path: str = "img_repo", csv_file_path: Optional[str] = None):
        """
        Initialize the processor.
        
        Args:
            img_repo_path: Path to the img_repo directory
            csv_file_path: Path to the CSV file (if None, will search for it)
        """
        self.img_repo_path = Path(img_repo_path)
        self.csv_file_path = self._find_csv_file() if csv_file_path is None else Path(csv_file_path)
        self.hals_mrn_set: Set[str] = set()
        self.processed_files = {}  # Maps original files to processed files
        self.assigned_mrn = set()  # Track MRNs assigned during this processing run
        self.conversion_stats = {
            'dicom_to_jpeg': 0,
            'renamed': 0,
            'preserved': 0,
            'already_jpeg': 0,
            'errors': 0
        }
        
    def _find_csv_file(self) -> Optional[Path]:
        """
        Find CSV file in the data directory that contains HALS_MRN column.
        
        Returns:
            Path to CSV file, or None if not found
        """
        data_dir = Path("data")
        if not data_dir.exists():
            logger.warning("Data directory not found, trying current directory")
            data_dir = Path(".")
        
        # Look for CSV files in data directory
        csv_files = list(data_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if 'HALS_MRN' in reader.fieldnames:
                        logger.info(f"Found CSV file with HALS_MRN column: {csv_file}")
                        return csv_file
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")
                continue
        
        logger.error("No CSV file with HALS_MRN column found")
        return None
    
    def load_hals_mrn_from_csv(self) -> Set[str]:
        """
        Load all HALS_MRN values from the CSV file.
        
        Returns:
            Set of HALS_MRN values
        """
        if self.csv_file_path is None or not self.csv_file_path.exists():
            logger.warning("CSV file not available, cannot load HALS_MRN values")
            return set()
        
        hals_mrn_set = set()
        try:
            with open(self.csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    hals_mrn = row.get('HALS_MRN', '').strip()
                    if hals_mrn:
                        hals_mrn_set.add(hals_mrn)
            logger.info(f"Loaded {len(hals_mrn_set)} HALS_MRN values from CSV")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
        
        return hals_mrn_set
    
    def validate_filename_format(self, filename: str) -> bool:
        """
        Validate if filename matches expected format (6 digits + optional 'a' + extension).
        The first 4 digits should be >= 1116 (starting counter).
        
        Args:
            filename: Name of the file to validate
            
        Returns:
            True if format is valid, False otherwise
        """
        # Pattern: 6 digits, optional 'a', then extension
        pattern = r'^(\d{6})a?\.(jpg|jpeg|dcm|dicom|png|bmp|tiff|tif)$'
        match = re.match(pattern, filename.lower())
        if not match:
            return False
        
        # Extract the 6-digit number
        digits = match.group(1)
        first_four = int(digits[:4])
        
        # First 4 digits should be >= 1116 (starting counter from config)
        return first_four >= STARTING_COUNTER
    
    def extract_hals_mrn_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract HALS_MRN from filename if it follows the naming convention.
        
        Args:
            filename: Name of the file
            
        Returns:
            HALS_MRN identifier if valid, None otherwise
        """
        if self.validate_filename_format(filename):
            # Remove extension and return the stem
            stem = Path(filename).stem
            return stem
        return None
    
    def find_image_files(self, directory: Path) -> List[Path]:
        """
        Find all image files in the directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of Path objects for image files
        """
        image_files = []
        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.dcm', '.dicom', '.png', '.bmp', '.tiff', '.tif'}
        
        try:
            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
        except Exception as e:
            logger.error(f"Error searching for image files in {directory}: {e}")
        
        return sorted(image_files)
    
    def get_file_type_summary(self, files: List[Path]) -> Dict[str, int]:
        """
        Get summary of file types in the directory.
        
        Args:
            files: List of file paths
            
        Returns:
            Dictionary with file type counts
        """
        summary = {}
        for file_path in files:
            ext = file_path.suffix.lower()
            summary[ext] = summary.get(ext, 0) + 1
        return summary
    
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
                if pixel_array.max() > pixel_array.min():
                    pixel_array = ((pixel_array - pixel_array.min()) / 
                                 (pixel_array.max() - pixel_array.min()) * 255).astype('uint8')
                else:
                    pixel_array = pixel_array.astype('uint8')
            
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
    
    def get_next_available_hals_mrn(self, used_mrn: Set[str], start_counter: int = None) -> str:
        """
        Get next available HALS_MRN that's not in the used set.
        Follows the naming pattern: first 4 digits start from 1116, last 2 digits are sequential (01-99).
        
        Args:
            used_mrn: Set of already used HALS_MRN values
            start_counter: Starting counter value (defaults to STARTING_COUNTER from config)
            
        Returns:
            Next available HALS_MRN following the pattern XXXXYY or XXXXYYa
        """
        if start_counter is None:
            start_counter = STARTING_COUNTER
        
        counter = start_counter
        
        # Try all file numbers (01-99) for each counter value
        # First try all without 'a' suffix, then all with 'a' suffix
        while counter <= 9999:
            # Try all file numbers without 'a' suffix first
            for file_num in range(1, 100):  # 01 to 99
                candidate = f"{counter:04d}{file_num:02d}"
                if candidate not in used_mrn:
                    return candidate
            
            # If all without 'a' are used, try with 'a' suffix
            for file_num in range(1, 100):  # 01 to 99
                candidate = f"{counter:04d}{file_num:02d}a"
                if candidate not in used_mrn:
                    return candidate
            
            # All file numbers (with and without 'a') used for this counter, move to next counter
            counter += 1
        
        logger.error("Counter exceeded maximum value (9999)")
        return f"{start_counter:04d}01"
    
    def process_image_file(self, image_path: Path) -> Optional[Path]:
        """
        Process a single image file (convert DICOM to JPEG and rename if needed).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to processed file, or None if processing failed
        """
        file_extension = image_path.suffix.lower()
        filename = image_path.name
        
        # Check if filename already follows the naming convention
        existing_hals_mrn = self.extract_hals_mrn_from_filename(filename)
        
        if existing_hals_mrn:
            # Filename already follows the rule - preserve the name (just change extension)
            new_name = f"{existing_hals_mrn}.jpeg"
            output_path = self.img_repo_path / new_name
            # Track this MRN to avoid conflicts with files that need renaming
            self.assigned_mrn.add(existing_hals_mrn)
            
            if file_extension in ['.dcm', '.dicom']:
                # Convert DICOM to JPEG
                if self.convert_dicom_to_jpeg(image_path, output_path):
                    if image_path != output_path:
                        # Remove original DICOM file
                        try:
                            image_path.unlink()
                            logger.info(f"Removed original DICOM file: {image_path.name}")
                        except Exception as e:
                            logger.warning(f"Could not remove original file {image_path}: {e}")
                    self.conversion_stats['dicom_to_jpeg'] += 1
                    self.conversion_stats['preserved'] += 1
                    return output_path
                else:
                    self.conversion_stats['errors'] += 1
                    return None
            elif file_extension in ['.jpg', '.jpeg']:
                # Already JPEG, just rename if needed
                if filename.lower() != new_name.lower():
                    try:
                        shutil.move(str(image_path), str(output_path))
                        logger.info(f"Renamed JPEG: {filename} -> {new_name}")
                        self.conversion_stats['renamed'] += 1
                    except Exception as e:
                        logger.error(f"Error renaming file {image_path}: {e}")
                        self.conversion_stats['errors'] += 1
                        return None
                else:
                    self.conversion_stats['already_jpeg'] += 1
                return output_path
            else:
                # Other image format, convert to JPEG
                try:
                    img = Image.open(image_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(output_path, 'JPEG', quality=95)
                    if image_path != output_path:
                        image_path.unlink()
                    logger.info(f"Converted {file_extension} to JPEG: {filename} -> {new_name}")
                    self.conversion_stats['preserved'] += 1
                    return output_path
                except Exception as e:
                    logger.error(f"Error converting {image_path}: {e}")
                    self.conversion_stats['errors'] += 1
                    return None
        
        else:
            # Filename doesn't follow the rule - need to rename to match HALS_MRN from CSV
            # Get list of available HALS_MRN values that don't have corresponding files yet
            existing_files = {f.stem for f in self.img_repo_path.glob("*.jpeg")}
            # Combine existing files and already assigned MRNs in this run
            used_mrn = existing_files.union(self.assigned_mrn)
            available_mrn = None
            
            if self.hals_mrn_set:
                # Use first available HALS_MRN from CSV that doesn't have a file yet
                for mrn in sorted(self.hals_mrn_set):
                    if mrn not in used_mrn:
                        available_mrn = mrn
                        break
                
                if available_mrn is None:
                    # All MRNs from CSV are used, generate a new one following the pattern
                    available_mrn = self.get_next_available_hals_mrn(used_mrn)
                    logger.warning(f"No available HALS_MRN in CSV, generated: {available_mrn}")
            else:
                # No CSV available, generate a new MRN following the pattern
                available_mrn = self.get_next_available_hals_mrn(used_mrn)
                logger.warning(f"No CSV available, generated HALS_MRN: {available_mrn}")
            
            # Track this assignment
            self.assigned_mrn.add(available_mrn)
            
            new_name = f"{available_mrn}.jpeg"
            output_path = self.img_repo_path / new_name
            
            if file_extension in ['.dcm', '.dicom']:
                # Convert DICOM to JPEG
                if self.convert_dicom_to_jpeg(image_path, output_path):
                    if image_path != output_path:
                        try:
                            image_path.unlink()
                        except Exception as e:
                            logger.warning(f"Could not remove original file {image_path}: {e}")
                    self.conversion_stats['dicom_to_jpeg'] += 1
                    self.conversion_stats['renamed'] += 1
                    return output_path
                else:
                    self.conversion_stats['errors'] += 1
                    return None
            elif file_extension in ['.jpg', '.jpeg']:
                # Already JPEG, just rename
                try:
                    shutil.move(str(image_path), str(output_path))
                    logger.info(f"Renamed JPEG: {filename} -> {new_name}")
                    self.conversion_stats['renamed'] += 1
                    return output_path
                except Exception as e:
                    logger.error(f"Error renaming file {image_path}: {e}")
                    self.conversion_stats['errors'] += 1
                    return None
            else:
                # Other image format, convert to JPEG
                try:
                    img = Image.open(image_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(output_path, 'JPEG', quality=95)
                    if image_path != output_path:
                        image_path.unlink()
                    logger.info(f"Converted {file_extension} to JPEG: {filename} -> {new_name}")
                    self.conversion_stats['renamed'] += 1
                    return output_path
                except Exception as e:
                    logger.error(f"Error converting {image_path}: {e}")
                    self.conversion_stats['errors'] += 1
                    return None
    
    def process_all(self) -> None:
        """
        Main processing function that orchestrates the entire workflow.
        """
        logger.info("Starting img_repo processing")
        logger.info(f"Image repository path: {self.img_repo_path.absolute()}")
        
        # Check if img_repo directory exists
        if not self.img_repo_path.exists():
            logger.error(f"img_repo directory does not exist: {self.img_repo_path}")
            logger.info(f"Creating directory: {self.img_repo_path}")
            self.img_repo_path.mkdir(parents=True, exist_ok=True)
            return
        
        if not self.img_repo_path.is_dir():
            logger.error(f"Path is not a directory: {self.img_repo_path}")
            return
        
        # Load HALS_MRN values from CSV
        logger.info("Loading HALS_MRN values from CSV")
        self.hals_mrn_set = self.load_hals_mrn_from_csv()
        
        # Initialize assigned MRN set (reset for each run)
        self.assigned_mrn = set()
        
        # Find all image files
        logger.info("Finding image files in img_repo")
        image_files = self.find_image_files(self.img_repo_path)
        
        if not image_files:
            logger.info("No image files found in img_repo directory")
            return
        
        # Get file type summary
        file_summary = self.get_file_type_summary(image_files)
        logger.info(f"Found {len(image_files)} image files")
        logger.info(f"File type summary: {file_summary}")
        
        # Process each image file
        logger.info("Processing image files...")
        for image_path in image_files:
            try:
                processed_path = self.process_image_file(image_path)
                if processed_path:
                    self.processed_files[str(image_path)] = str(processed_path)
                    logger.info(f"Processed: {image_path.name} -> {processed_path.name}")
                else:
                    logger.error(f"Failed to process: {image_path.name}")
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                self.conversion_stats['errors'] += 1
        
        # Print summary statistics
        logger.info("=" * 60)
        logger.info("Processing completed successfully")
        logger.info(f"Total files processed: {len(self.processed_files)}")
        logger.info(f"Conversion statistics:")
        logger.info(f"  - DICOM to JPEG conversions: {self.conversion_stats['dicom_to_jpeg']}")
        logger.info(f"  - Files renamed: {self.conversion_stats['renamed']}")
        logger.info(f"  - Files preserved (name kept): {self.conversion_stats['preserved']}")
        logger.info(f"  - Already JPEG (no change): {self.conversion_stats['already_jpeg']}")
        logger.info(f"  - Errors: {self.conversion_stats['errors']}")
        logger.info(f"Output directory: {self.img_repo_path.absolute()}")


def main():
    """Main function to run the processor."""
    # Check if we're running from the correct directory
    if not os.path.exists("data"):
        logger.warning("Data directory not found, continuing anyway...")
    
    # Create processor instance
    processor = ImgRepoProcessor()
    
    # Run processing
    processor.process_all()


if __name__ == "__main__":
    main()

