#!/usr/bin/env python3
# data/demo_local_processor.py
"""
Demo script for Local Image Processor

This script demonstrates how to use the local image processor.
"""

import os
import sys
from pathlib import Path

# Add the data directory to the path so we can import the processor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from local_image_processor import LocalImageProcessor
from config import *


def demo_with_existing_images():
    """Demo using the existing images in data/img_repo."""
    print("Demo: Processing existing images in data/img_repo")
    print("=" * 50)
    
    # Use the existing img_repo directory
    local_image_path = "data/img_repo"
    csv_file_path = "data/HALS_Dataset_v1.csv"
    output_dir = "data/processed_images"
    
    # Create processor instance
    processor = LocalImageProcessor(local_image_path, csv_file_path, output_dir)
    
    # Process all images
    processor.process_all()
    
    print(f"\nDemo completed!")
    print(f"Processed files: {len(processor.processed_files)}")
    print(f"Output directory: {processor.output_dir.absolute()}")


def demo_with_user_input():
    """Demo using user input for directory selection."""
    print("Demo: Interactive directory selection")
    print("=" * 50)
    
    # This would normally get user input, but for demo we'll use a known path
    local_image_path = "data/img_repo"  # In real usage, this would come from user input
    csv_file_path = "data/HALS_Dataset_v1.csv"
    output_dir = "data/processed_images_interactive"
    
    print(f"Using directory: {local_image_path}")
    
    # Create processor instance
    processor = LocalImageProcessor(local_image_path, csv_file_path, output_dir)
    
    # Process all images
    processor.process_all()
    
    print(f"\nDemo completed!")
    print(f"Processed files: {len(processor.processed_files)}")
    print(f"Output directory: {processor.output_dir.absolute()}")


def main():
    """Run the demo."""
    print("Local Image Processor Demo")
    print("=" * 30)
    
    # Demo 1: Using existing images
    demo_with_existing_images()
    
    print("\n" + "=" * 50)
    
    # Demo 2: Interactive mode (simulated)
    demo_with_user_input()


if __name__ == "__main__":
    main()
