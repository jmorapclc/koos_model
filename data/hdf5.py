#!/usr/bin/env python3
# data/hdf5.py
"""
Binary PyTorch optimized format for medical imaging pipeline.
"""

import h5py
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import json
import os

class HDF5Builder:
    """
    Constructs binary PyTorch optimized format from input files.
    Implements chunked storage, gzip compression, and metadata preservation.
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 bit_depth: int = 8,
                 compression_level: int = 9):
        """
        Initialize builder with compression parameters.
        
        Args:
            target_size: Spatial dimensions for tensor storage (H, W)
            bit_depth: Bit depth for pixel quantization (8 or 16)
            compression_level: Gzip compression level (0-9)
        """
        self.target_size = target_size
        self.bit_depth = bit_depth
        self.compression_level = compression_level
        self.dtype = np.uint8 if bit_depth <= 8 else np.uint16
        self.max_value = (2 ** bit_depth) - 1
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load and preprocess input file to standardized tensor format.
        Applies grayscale conversion, CLAHE, and spatial normalization.
        
        Args:
            image_path: Path to input file
            
        Returns:
            Preprocessed image array (H, W, C) in uint8/uint16
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img is None:
            raise ValueError(f"Failed to load: {image_path}")
        
        # CLAHE for medical imaging contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Spatial normalization
        img = cv2.resize(img, self.target_size[::-1], interpolation=cv2.INTER_CUBIC)
        
        # Convert to RGB for compatibility (grayscale -> 3-channel)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Quantization to target bit depth
        if img.dtype != self.dtype:
            if self.bit_depth == 8:
                img = img.astype(np.uint8)
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    img = (normalized * self.max_value).astype(self.dtype)
                else:
                    img = img.astype(self.dtype)
        
        return img
    
    def _extract_metadata_features(self, row: pd.Series) -> np.ndarray:
        """
        Extract numerical features from metadata row.
        Normalizes continuous variables to [0, 1] range.
        
        Args:
            row: Pandas Series containing metadata
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        features.append(float(row.get('Sex', 0)))
        features.append(min(float(row.get('Age', 50)) / 100.0, 1.0))
        features.append(min(float(row.get('BMI', 25)) / 50.0, 1.0))
        features.append(float(row.get('Side', 0)))
        features.append(float(row.get('Type_of_TKA', 0)))
        features.append(float(row.get('Patellar_Replacement', 0)))
        features.append(min(float(row.get('Preoperative_KOOS-PS', 50)) / 100.0, 1.0))
        return np.array(features, dtype=np.float32)
    
    def build(self,
              image_dir: str,
              csv_file: str,
              output_file: str) -> str:
        """
        Construct binary PyTorch optimized format from input files.
        
        Args:
            image_dir: Directory containing input files
            csv_file: Path to metadata CSV
            output_file: Output path for binary format
            
        Returns:
            Path to created binary format file
        """
        image_dir = Path(image_dir)
        csv_file = Path(csv_file)
        output_file = Path(output_file)
        
        # Load metadata
        df = pd.read_csv(csv_file, dtype={'HALS_MRN': str})
        
        # Filter valid samples
        valid_samples = []
        for idx, row in df.iterrows():
            image_path = image_dir / f"{row['HALS_MRN']}.jpeg"
            if not image_path.exists():
                continue
            target = row.get('1-Year_Postop_KOOS-PS')
            if pd.isna(target) or target < 0 or target > 100:
                continue
            valid_samples.append((idx, row, image_path))
        
        num_samples = len(valid_samples)
        if num_samples == 0:
            raise ValueError("No valid samples found")
        
        # Determine chunk shape for optimal I/O
        chunk_size = min(10, num_samples)
        chunk_shape = (chunk_size, self.target_size[0], self.target_size[1], 3)
        
        # Create HDF5 structure
        with h5py.File(output_file, 'w') as hf:
            # Image tensor dataset (N, H, W, C)
            img_dataset = hf.create_dataset(
                'images',
                shape=(num_samples, self.target_size[0], self.target_size[1], 3),
                dtype=self.dtype,
                compression='gzip',
                compression_opts=self.compression_level,
                chunks=chunk_shape,
                shuffle=True,
                fletcher32=True
            )
            
            # Metadata features dataset (N, F)
            metadata_dataset = hf.create_dataset(
                'metadata',
                shape=(num_samples, 7),
                dtype=np.float32,
                compression='gzip',
                compression_opts=self.compression_level
            )
            
            # Target values dataset (N,)
            targets_dataset = hf.create_dataset(
                'targets',
                shape=(num_samples,),
                dtype=np.float32,
                compression='gzip',
                compression_opts=self.compression_level
            )
            
            # HALS_MRN identifiers (string array)
            hals_mrn_data = [str(row['HALS_MRN']) for _, row, _ in valid_samples]
            max_mrn_len = max(len(mrn) for mrn in hals_mrn_data)
            mrn_dataset = hf.create_dataset(
                'hals_mrn',
                shape=(num_samples,),
                dtype=f'S{max_mrn_len}',
                compression='gzip',
                compression_opts=self.compression_level
            )
            
            # Store complete CSV metadata as separate dataset
            # Serialize each row as JSON string for flexibility
            csv_rows = []
            for _, row, _ in valid_samples:
                row_dict = row.to_dict()
                csv_rows.append(json.dumps(row_dict))
            
            max_csv_len = max(len(s) for s in csv_rows)
            csv_dataset = hf.create_dataset(
                'csv_metadata',
                shape=(num_samples,),
                dtype=f'S{max_csv_len}',
                compression='gzip',
                compression_opts=self.compression_level
            )
            
            # Process and store samples
            for idx, (orig_idx, row, image_path) in enumerate(valid_samples):
                # Load and preprocess image
                img = self._load_image(image_path)
                img_dataset[idx] = img
                
                # Extract metadata features
                metadata = self._extract_metadata_features(row)
                metadata_dataset[idx] = metadata
                
                # Store target
                targets_dataset[idx] = float(row['1-Year_Postop_KOOS-PS'])
                
                # Store identifiers
                mrn_dataset[idx] = hals_mrn_data[idx].encode('utf-8')
                csv_dataset[idx] = csv_rows[idx].encode('utf-8')
            
            # Store global attributes
            hf.attrs['num_samples'] = num_samples
            hf.attrs['image_shape'] = (self.target_size[0], self.target_size[1], 3)
            hf.attrs['bit_depth'] = self.bit_depth
            hf.attrs['metadata_features'] = 7
            hf.attrs['compression_level'] = self.compression_level
            hf.attrs['csv_columns'] = json.dumps(list(df.columns))
        
        return str(output_file)


def main():
    """Build binary PyTorch optimized format from input files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build binary PyTorch optimized format')
    parser.add_argument('--image_dir', type=str, default='data/img_repo',
                       help='Directory containing input files')
    parser.add_argument('--csv_file', type=str, default='data/HALS_Dataset_v1.csv',
                       help='Path to metadata CSV')
    parser.add_argument('--output', type=str, default='data/koos_dataset.h5',
                       help='Output path for binary format')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Target image size')
    parser.add_argument('--bit_depth', type=int, default=8,
                       help='Bit depth for quantization (8 or 16)')
    parser.add_argument('--compression', type=int, default=9,
                       help='Gzip compression level (0-9)')
    
    args = parser.parse_args()
    
    builder = HDF5Builder(
        target_size=(args.image_size, args.image_size),
        bit_depth=args.bit_depth,
        compression_level=args.compression
    )
    
    output_path = builder.build(
        image_dir=args.image_dir,
        csv_file=args.csv_file,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

